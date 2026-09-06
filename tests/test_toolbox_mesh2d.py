"""`mesh2d.py`: any closed planar outline to a runnable 2D case, drawn and measured
first.

The parsing, the edge classification, the roles and the case text are ordinary Python
and are tested without gmsh; the builds that need the kernel are gated on it and run
the script end to end into a tmp dir, the way the agent would.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mesh2d():
    return load("mesh2d")


@pytest.fixture(scope="module")
def case_gen():
    return load("case_gen")


TESLA = {"ops": [
    {"op": "rect", "name": "main", "origin": [0, -1.5], "size": [60, 3]},
    {"op": "channel", "name": "bypass", "width": 3, "start": [9, 1.5], "heading": 60,
     "path": [{"line": 2}, {"arc": {"radius": 3.5, "angle": 240}}, {"line": 2.5}]},
    {"op": "repeat", "name": "bypasses", "target": "bypass", "count": 4, "step": [12, 0]},
    {"op": "fuse", "name": "body", "of": ["main", "bypasses"]}],
    "patches": [{"name": "inlet", "at": "x:min"}, {"name": "outlet", "at": "x:max"}]}

CHANNEL = [{"op": "rect", "name": "body", "origin": [0, 0], "size": [1.0, 0.3]}]


# -- the spec --------------------------------------------------------------------


def test_a_good_spec_normalises_and_names_its_body(mesh2d):
    ops, rules = mesh2d.parse_spec(TESLA)
    assert [op["op"] for op in ops] == ["rect", "channel", "repeat", "fuse"]
    assert ops[0]["origin"] == (0.0, -1.5) and ops[0]["size"] == (60.0, 3.0)
    assert ops[1]["path"][1]["arc"] == {"radius": 3.5, "angle": 240.0}
    assert ops[2]["count"] == 4 and ops[2]["step"] == (12.0, 0.0)
    assert mesh2d.body_name(ops) == "body"
    assert [(r["name"], r["kind"], r["at"]) for r in rules] == [
        ("inlet", "inlet", (0, "min", None)), ("outlet", "outlet", (0, "max", None))]


def test_a_bare_list_is_a_spec_with_no_rules(mesh2d):
    ops, rules = mesh2d.parse_spec(CHANNEL)
    assert len(ops) == 1 and rules == []
    assert mesh2d.body_name(ops) == "body"


def test_a_leg_can_run_to_a_line_instead_of_a_length(mesh2d):
    ops, _ = mesh2d.parse_spec([{"op": "channel", "width": 3, "start": [0, 0], "heading": 90,
                                 "path": [{"line": {"to": "y:10"}}]}])
    assert ops[0]["path"] == [{"to": (1, "at", 10.0)}]
    with pytest.raises(SystemExit) as err:
        mesh2d.parse_spec([{"op": "channel", "width": 3, "start": [0, 0], "heading": 90,
                            "path": [{"line": {"until": "y:10"}}]}])
    assert "to" in str(err.value)


def test_a_return_leg_written_to_the_wall_closes_every_bypass(mesh2d, tmp_path, capsys):
    """The premise gate's failure: a return leg typed as a length ended in mid-air and
    the loops only enclosed islands where they overlapped their neighbours (3 of 4).
    Written `to` the wall, all four close, whatever the arc left the heading at."""
    pytest.importorskip("gmsh")
    spec = dict(TESLA)
    spec["ops"] = [dict(op) for op in TESLA["ops"]]
    # a shallow 20-degree departure, which leaves the return heading at 260 degrees --
    # exactly the case where a typed length missed the wall; the radius keeps adjacent
    # loops apart (outer reach 5 mm at a 12 mm pitch)
    spec["ops"][1] = dict(spec["ops"][1], heading=20,
                          path=[{"line": 2}, {"arc": {"radius": 3.5, "angle": 240}}, {"line": {"to": "y:1.5"}}])
    path = tmp_path / "to.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert mesh2d.main(["--spec", str(path), "--scale", "0.001", "--dry-run"]) == 0
    assert "islands 4" in capsys.readouterr().out


def test_a_centred_rect_is_the_same_as_its_origin_form(mesh2d):
    ops, _ = mesh2d.parse_spec([{"op": "rect", "center": [1, 1], "size": [2, 4]}])
    assert ops[0]["origin"] == (0.0, -1.0)


@pytest.mark.parametrize("bad, words", [
    ([], "at least one op"),
    ([{"name": "x"}], 'string "op"'),
    ([{"op": "sphere"}], "no op called"),
    ([{"op": "rect", "size": [1, 1]}], "one of origin or center"),
    ([{"op": "rect", "origin": [0, 0], "size": [1, 0]}], "size must be positive"),
    ([{"op": "disk", "center": [0, 0], "radius": 0}], "radius must be positive"),
    ([{"op": "annulus", "center": [0, 0], "r_inner": 2, "r_outer": 1}], "r_inner must be smaller"),
    ([{"op": "band", "center": [0, 0], "r_inner": 1, "r_outer": 2, "start": 0, "end": 0}], "between 0 and 360"),
    ([{"op": "band", "center": [0, 0], "r_inner": 1, "r_outer": 2, "start": 0, "end": 400}], "between 0 and 360"),
    ([{"op": "polygon", "points": [[0, 0], [1, 0]]}], "at least three"),
    ([{"op": "disk", "name": "a", "center": [0, 0], "radius": 1}, {"op": "fuse", "of": ["a"]}], "two or more"),
    ([{"op": "disk", "name": "a", "center": [0, 0], "radius": 1}, {"op": "cut", "from": "a"}], "take needs"),
    ([{"op": "fuse", "of": ["a", "b"]}], "must name an earlier op"),
    ([{"op": "disk", "name": "a", "center": [0, 0], "radius": 1},
      {"op": "disk", "name": "a", "center": [0, 0], "radius": 1}], "already used"),
    ([{"op": "disk", "name": "a", "center": [0, 0], "radius": 1},
      {"op": "repeat", "target": "a", "count": 1, "step": [1, 0]}], "at least 2"),
    ([{"op": "disk", "name": "a", "center": [0, 0], "radius": 1},
      {"op": "mirror", "target": "a", "axis": "q"}], "axis must be x or y"),
    ([{"op": "disk", "name": "a", "center": [0, 0], "radius": 1},
      {"op": "fillet", "target": "a", "radius": 0}], "radius must be positive"),
    ([{"op": "channel", "width": 3, "start": [0, 0], "path": [{"arc": {"radius": 1, "angle": 90}}]}],
     "exceed half the width"),
    ({"ops": CHANNEL, "patches": [{"name": "lid", "at": "w:min"}]}, "must be x or y"),
    ({"ops": CHANNEL, "patches": [{"at": "y:max"}]}, "needs a name"),
])
def test_a_bad_spec_stops_before_the_kernel_and_says_which_entry(mesh2d, bad, words):
    with pytest.raises(SystemExit) as err:
        mesh2d.parse_spec(bad)
    assert words in str(err.value)


def test_parse_where_reads_the_three_forms(mesh2d):
    assert mesh2d.parse_where("x:min") == (0, "min", None)
    assert mesh2d.parse_where("y:max") == (1, "max", None)
    assert mesh2d.parse_where("x:0.012") == (0, "at", 0.012)
    with pytest.raises(SystemExit):
        mesh2d.parse_where("min")


# -- geometry helpers ------------------------------------------------------------


def test_a_wedge_polygon_spans_its_angles_counter_clockwise(mesh2d, case_gen):
    pts = mesh2d.wedge_polygon((0.0, 0.0), 1.0, 0.0, 240.0)
    assert pts[0] == (0.0, 0.0)
    assert pts[-1] == pytest.approx((-0.5, -0.8660254), abs=1e-6)
    assert case_gen.signed_area(pts) > 0


def test_an_outline_file_is_read_closed_and_counter_clockwise(mesh2d, case_gen, tmp_path):
    path = tmp_path / "octagon.csv"
    # clockwise, closed, with the aerofoil-file habit of a name on line one
    ring = [(1, 0), (0.7, -0.7), (0, -1), (-0.7, -0.7), (-1, 0), (-0.7, 0.7), (0, 1), (0.7, 0.7)]
    path.write_text("octagon\n" + "\n".join(f"{x},{y}" for x, y in ring) + "\n1,0\n", encoding="utf-8")
    pts = mesh2d.outline_points(path, None, 0.0)
    assert len(pts) == 8
    assert case_gen.signed_area(pts) > 0
    scaled = mesh2d.outline_points(path, 2.0, 0.0)
    assert max(p[0] for p in scaled) - min(p[0] for p in scaled) == pytest.approx(2.0)


# -- which edge is which -----------------------------------------------------------


def test_a_passage_finds_its_ends_along_its_longest_axis(mesh2d):
    domain, tol = (0.0, -1.5, 60.0, 1.5), 1e-3
    assert mesh2d.classify_edge((0.0, -1.5, 0.0, 1.5), domain, 0, tol) == "inlet"
    assert mesh2d.classify_edge((60.0, -1.5, 60.0, 1.5), domain, 0, tol) == "outlet"
    assert mesh2d.classify_edge((0.0, 1.5, 20.0, 10.0), domain, 0, tol) == "walls", "touches x=0 at a point only"
    tall = (0.0, 0.0, 3.0, 60.0)
    assert mesh2d.longest_axis2d((3.0, 60.0)) == 1
    assert mesh2d.classify_edge((0.0, 0.0, 3.0, 0.0), tall, 1, tol) == "inlet"
    assert mesh2d.classify_edge((0.0, 60.0, 3.0, 60.0), tall, 1, tol) == "outlet"


def test_a_box_round_a_body_reads_its_four_sides_and_the_rest_is_the_body(mesh2d):
    box, tol = (-2.0, -2.0, 7.0, 3.0), 1e-3
    assert mesh2d.classify_external((-2.0, -2.0, -2.0, 3.0), box, tol) == "inlet"
    assert mesh2d.classify_external((7.0, -2.0, 7.0, 3.0), box, tol) == "outlet"
    assert mesh2d.classify_external((-2.0, 3.0, 7.0, 3.0), box, tol) == "farfield"
    assert mesh2d.classify_external((0.2, 0.2, 0.8, 0.8), box, tol) == "body"


def test_a_rule_names_an_edge_before_the_automatic_reading(mesh2d):
    domain, tol = (0.0, 0.0, 1.0, 0.3), 1e-4
    lid = [{"name": "lid", "kind": "slip", "at": (1, "max", None), "box": None}]
    assert mesh2d.patch_for_edge((0.0, 0.3, 1.0, 0.3), domain, 0, tol, lid, False) == "lid"
    assert mesh2d.patch_for_edge((0.0, 0.0, 1.0, 0.0), domain, 0, tol, lid, False) == "walls"
    boxed = [{"name": "probe", "kind": "wall", "at": None, "box": (0.4, -0.1, 0.6, 0.05)}]
    assert mesh2d.patch_for_edge((0.45, 0.0, 0.55, 0.0), domain, 0, tol, boxed, False) == "probe"
    at_value = [{"name": "inlet", "kind": "inlet", "at": (0, "at", 1.0), "box": None}]
    assert mesh2d.patch_for_edge((1.0, 0.0, 1.0, 0.3), domain, 0, tol, at_value, False) == "inlet"


def test_rules_are_in_the_specs_units_and_scale_with_it(mesh2d):
    """The serpentine run: a spec in mm carried `"box": [-0.6, -1.2, 0.6, 1.2]` round
    its inlet, the shape was built at --scale 0.001, and the unscaled box -- 1.2 m
    across a 38 mm shape -- made every one of the 20 edges the inlet."""
    _, rules = mesh2d.parse_spec({"ops": CHANNEL, "patches": [
        {"name": "inlet", "box": [-0.6, -1.2, 0.6, 1.2]},
        {"name": "outlet", "at": "x:30"},
        {"name": "lid", "at": "y:max", "kind": "slip"}]})
    scaled = mesh2d.scale_rules(rules, 0.001)
    assert scaled[0]["box"] == pytest.approx((-0.0006, -0.0012, 0.0006, 0.0012))
    assert scaled[1]["at"] == (0, "at", pytest.approx(0.03))
    assert scaled[2]["at"] == (1, "max", None), "min/max name an end, not a length"
    assert mesh2d.scale_rules(rules, 1.0) is rules
    # a 38 mm-wide passage's far wall no longer sits inside the inlet's box
    domain, tol = (-0.004, -0.001, 0.034, 0.019), 4e-6
    assert mesh2d.patch_for_edge((0.03, 0.017, 0.03, 0.019), domain, 0, tol, scaled, False) != "inlet"
    assert mesh2d.patch_for_edge((0.0, -0.001, 0.0, 0.001), domain, 0, tol, scaled, False) == "inlet"


def test_the_spec_rules_are_scaled_where_the_shape_is(mesh2d, tmp_path, capsys):
    pytest.importorskip("gmsh")
    spec = tmp_path / "mm.json"
    spec.write_text(json.dumps({"ops": [{"op": "rect", "name": "body", "origin": [0, 0], "size": [30, 2]}],
                                "patches": [{"name": "inlet", "box": [-0.5, -0.5, 0.5, 2.5]},
                                            {"name": "outlet", "at": "x:30"}]}), encoding="utf-8")
    assert mesh2d.main(["--spec", str(spec), "--scale", "0.001", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "inlet (1 edge, 0.002 m), outlet (1 edge, 0.002 m), walls (2 edges, 0.06 m)" in out


# -- roles, Allmesh and the case text ----------------------------------------------


def test_roles_carry_the_empty_patch_and_the_walls_are_the_walls(mesh2d):
    roles = mesh2d.build_roles(["inlet", "outlet", "walls", "frontAndBack"], [], 0, "slip", False)
    assert roles["frontAndBack"] == {"kind": "empty"}
    assert roles["inlet"] == {"kind": "inlet", "direction": (1.0, 0.0, 0.0)}
    assert roles["outlet"]["kind"] == "outlet" and roles["walls"]["kind"] == "wall"
    assert mesh2d.wall_patches(roles) == ["walls"]
    assert mesh2d.empty_patches(roles) == ["frontAndBack"]
    up = mesh2d.build_roles(["inlet"], [], 1, "slip", False)
    assert up["inlet"]["direction"] == (0.0, 1.0, 0.0)
    ext = mesh2d.build_roles(["inlet", "outlet", "farfield", "body", "frontAndBack"], [], 0, "symmetry", True)
    assert ext["farfield"]["kind"] == "symmetry" and ext["body"]["kind"] == "wall"
    lid = mesh2d.build_roles(["lid"], [{"name": "lid", "kind": "slip", "at": None, "box": None}], 0, "slip", False)
    assert lid["lid"]["kind"] == "slip"


def test_allmesh_retypes_the_walls_then_the_empty_patch_before_checkmesh(mesh2d):
    roles = mesh2d.build_roles(["inlet", "outlet", "walls", "frontAndBack"], [], 0, "slip", False)
    script = mesh2d.allmesh(roles, 4)
    convert = script.index("gmshToFoam body.msh")
    wall = script.index("entry0/walls/type -set wall")
    empty = script.index("entry0/frontAndBack/type -set empty")
    check = script.index("checkMesh")
    assert convert < wall < empty < check
    assert "entry0/inlet/type" not in script and "entry0/outlet/type" not in script
    assert "hexahedra:" in script, "the grep shows the all-hex confirmation"


def _plan(mesh2d, case_gen, faces, external=False, far="slip", length=0.006):
    patches = mesh2d.Patches2D(faces, 3e-4, 3e-4)
    roles = mesh2d.build_roles(patches.patch_faces, [], 0, far, external)
    return case_gen.Plan(patches, roles, length, {"source": "test"})


OPTS = {"cores": 4, "iterations": 100, "writes": 10, "end_time": 1.0, "delta_t": None,
        "courant": 0.9, "purge": None, "speed": 1.0, "nu": 1.5e-5, "reynolds": None,
        "body_patch": "body", "turbulent_intensity": None, "mixing_length": None,
        "viscosity_ratio": None}


def test_the_case_names_every_patch_and_the_empty_patch_is_empty_in_every_field(mesh2d, case_gen):
    plan = _plan(mesh2d, case_gen, {"inlet": 1, "outlet": 1, "walls": 50, "frontAndBack": 2})
    flow = case_gen.derive_flow(OPTS, 0.006)
    files = mesh2d.case_files(plan, flow, OPTS, "laminar", "steady", False)
    for required in ("system/controlDict", "system/fvSchemes", "system/fvSolution",
                     "system/decomposeParDict", "constant/transportProperties",
                     "constant/turbulenceProperties", "0/U", "0/p", "Allmesh", "Allrun", "case.foam"):
        assert required in files
    for name in ("0/U", "0/p"):
        text = files[name]
        for patch in ("inlet", "outlet", "walls", "frontAndBack"):
            assert re.search(rf"\n\s+{patch}\n\s+\{{", text), f"{patch} missing from {name}"
        assert re.search(r"frontAndBack\s*\{\s*type\s+empty;", text), f"frontAndBack is not empty in {name}"
    assert "application     simpleFoam;" in files["system/controlDict"]
    assert "forceCoeffs" not in files["system/controlDict"], "a passage has no body to take forces"
    assert "fileHandler     collated;" in files["system/controlDict"]
    assert "-set empty" in files["Allmesh"]
    assert "0/k" not in files


def test_an_external_case_measures_its_reference_area_from_length_and_thickness(mesh2d, case_gen):
    plan = _plan(mesh2d, case_gen, {"inlet": 1, "outlet": 1, "farfield": 2, "body": 8, "frontAndBack": 2},
                 external=True, length=0.1)
    flow = case_gen.derive_flow({**OPTS, "nu": None, "reynolds": 1e5}, 0.1)
    files = mesh2d.case_files(plan, flow, {**OPTS, "nu": None, "reynolds": 1e5}, "kOmegaSST", "steady", True)
    control = files["system/controlDict"]
    assert "forceCoeffs" in control and "patches         (body);" in control
    assert re.search(r"Aref\s+3e-05;", control), "0.1 m chord x 3e-4 m thickness"
    assert "liftDir         (0 1 0);" in control
    assert "0/k" in files and "0/omega" in files and "0/nut" in files
    assert re.search(r"farfield\s*\{\s*type\s+slip;", files["0/U"])
    assert re.search(r"frontAndBack\s*\{\s*type\s+empty;", files["0/k"])


def test_the_schemes_are_the_hex_set_not_the_tet_set(mesh2d, case_gen):
    plan = _plan(mesh2d, case_gen, {"inlet": 1, "outlet": 1, "walls": 4, "frontAndBack": 2})
    flow = case_gen.derive_flow(OPTS, 0.006)
    files = mesh2d.case_files(plan, flow, OPTS, "laminar", "steady", False)
    assert "Gauss linear corrected" in files["system/fvSchemes"]
    assert "limited corrected 0.5" not in files["system/fvSchemes"], "cad_gen's tet detuning"
    assert "Phi" in files["system/fvSolution"], "the potentialFoam warm start Allrun uses"


def test_mesh_sizes_default_from_the_short_side_and_thickness_follows_the_cell(mesh2d):
    assert mesh2d.mesh_sizes2d((0.06, 0.012), {}) == pytest.approx((0.0004, 0.0004, 0.0004))
    assert mesh2d.mesh_sizes2d((0.06, 0.012), {"cell": 0.001, "wall_cell": 0.0005, "thickness": 0.002}) == (0.001, 0.0005, 0.002)
    with pytest.raises(SystemExit):
        mesh2d.mesh_sizes2d((0.06, 0.012), {"cell": -1})


def test_the_report_counts_islands_and_edges_per_patch(mesh2d):
    built = mesh2d.Built2D(face=1, bounds=(0, 0, 0.06, 0.0115), extent=(0.06, 0.0115), area=4e-4,
                           islands=4, curves=[1, 2, 3, 4], lengths={1: 0.003, 2: 0.003, 3: 0.1, 4: 0.155},
                           short=[(4, 0.00001)], domain=(0, 0, 0.06, 0.0115))
    lines = mesh2d.report_lines(built, {1: "inlet", 2: "outlet", 3: "walls", 4: "walls"}, "tesla.json")
    text = "\n".join(lines)
    assert "islands 4" in text
    assert "inlet (1 edge, 0.003 m)" in text and "walls (2 edges, 0.255 m)" in text
    assert "!!" in text and "1 edge of 1e-05 m" in text and "short against the narrowest" in text


# -- the kernel: real builds into a tmp dir ---------------------------------------


def _msh_element_types(path: Path) -> dict[int, int]:
    """Count elements per gmsh type in a v2.2 ascii .msh (5 = hex, 4 = tet, 6 = prism)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index("$Elements")
    n = int(lines[start + 1])
    counts: dict[int, int] = {}
    for line in lines[start + 2:start + 2 + n]:
        kind = int(line.split()[1])
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def test_a_channel_meshes_into_hexahedra_only(mesh2d, tmp_path, capsys):
    pytest.importorskip("gmsh")
    spec = tmp_path / "c.json"
    spec.write_text(json.dumps(CHANNEL), encoding="utf-8")
    case = tmp_path / "case"
    assert mesh2d.main([str(case), "--spec", str(spec), "--cell", "0.1"]) == 0
    out = capsys.readouterr().out
    assert "hexahedra" in out and "prisms" not in out and "tetrahedra" not in out
    msh = (case / "body.msh").read_text(encoding="utf-8")
    for name in ("inlet", "outlet", "walls", "frontAndBack", "fluid"):
        assert f'"{name}"' in msh
    kinds = _msh_element_types(case / "body.msh")
    assert kinds.get(5, 0) > 0 and kinds.get(4, 0) == 0 and kinds.get(6, 0) == 0
    assert re.search(r"patches\s+inlet \(1\), outlet \(1\), walls \(2\), frontAndBack \(2\)", out)


def test_the_tesla_valve_builds_with_four_islands_and_a_preview(mesh2d, tmp_path, capsys):
    pytest.importorskip("gmsh")
    pytest.importorskip("matplotlib")
    spec = tmp_path / "tesla.json"
    spec.write_text(json.dumps(TESLA), encoding="utf-8")
    case = tmp_path / "tesla"
    png = tmp_path / "outline.png"
    assert mesh2d.main([str(case), "--spec", str(spec), "--scale", "0.001", "--cell", "0.0005",
                        "--preview", str(png)]) == 0
    out = capsys.readouterr().out
    assert "islands 4" in out
    assert re.search(r"extent\s+0\.06 x 0\.011", out)
    assert "hexahedra" in out and "prisms" not in out
    assert png.stat().st_size > 10_000
    u = (case / "0" / "U").read_text(encoding="utf-8")
    for patch in ("inlet", "outlet", "walls", "frontAndBack"):
        assert patch in u
    assert "-set empty" in (case / "Allmesh").read_text(encoding="utf-8")
    assert (case / "constant" / "geometry" / "body.json").exists()
    assert (case / "constant" / "geometry" / "body.step").stat().st_size > 1000


def test_a_box_round_a_disk_is_an_external_case(mesh2d, tmp_path, capsys):
    pytest.importorskip("gmsh")
    spec = tmp_path / "disk.json"
    spec.write_text(json.dumps([{"op": "disk", "name": "body", "center": [0, 0], "radius": 0.05}]), encoding="utf-8")
    case = tmp_path / "ext"
    assert mesh2d.main([str(case), "--spec", str(spec), "--external", "--cell", "0.02"]) == 0
    out = capsys.readouterr().out
    u = (case / "0" / "U").read_text(encoding="utf-8")
    for patch in ("inlet", "outlet", "farfield", "body", "frontAndBack"):
        assert patch in u
    assert "forceCoeffs" in (case / "system" / "controlDict").read_text(encoding="utf-8")
    assert "a flow box" in out


def test_a_disjoint_fuse_is_refused_by_name(mesh2d, tmp_path):
    pytest.importorskip("gmsh")
    spec = tmp_path / "apart.json"
    spec.write_text(json.dumps([
        {"op": "disk", "name": "a", "center": [0, 0], "radius": 1},
        {"op": "disk", "name": "b", "center": [10, 0], "radius": 1},
        {"op": "fuse", "name": "body", "of": ["a", "b"]}]), encoding="utf-8")
    with pytest.raises(SystemExit) as err:
        mesh2d.main(["--spec", str(spec), "--dry-run"])
    assert "separate faces" in str(err.value)


def test_a_rule_names_the_lid(mesh2d, tmp_path):
    pytest.importorskip("gmsh")
    spec = tmp_path / "lid.json"
    spec.write_text(json.dumps({"ops": CHANNEL,
                                "patches": [{"name": "lid", "at": "y:max", "kind": "slip"}]}), encoding="utf-8")
    case = tmp_path / "lid"
    assert mesh2d.main([str(case), "--spec", str(spec), "--cell", "0.1"]) == 0
    u = (case / "0" / "U").read_text(encoding="utf-8")
    assert re.search(r"lid\s*\{\s*type\s+slip;", u)


def test_dry_run_writes_only_the_preview(mesh2d, tmp_path, capsys):
    pytest.importorskip("gmsh")
    pytest.importorskip("matplotlib")
    spec = tmp_path / "c.json"
    spec.write_text(json.dumps(CHANNEL), encoding="utf-8")
    png = tmp_path / "out" / "c.png"
    assert mesh2d.main(["--spec", str(spec), "--dry-run", "--preview", str(png)]) == 0
    assert png.exists()
    assert not (tmp_path / "case").exists()
    assert "drawn to" in capsys.readouterr().out


def test_repeat_and_mirror_make_one_face(mesh2d, tmp_path, capsys):
    """A row of pins on a channel floor, mirrored onto the ceiling: the copies of a
    repeat stand apart (that is what a repeat means) and the whole is one face."""
    pytest.importorskip("gmsh")
    spec = tmp_path / "rm.json"
    spec.write_text(json.dumps([
        {"op": "rect", "name": "duct", "origin": [0, 0], "size": [10, 4]},
        {"op": "disk", "name": "pin", "center": [2, 0.5], "radius": 0.6},
        {"op": "repeat", "name": "row", "target": "pin", "count": 3, "step": [2, 0]},
        {"op": "mirror", "name": "pins", "target": "row", "axis": "y", "at": 2, "keep": True},
        {"op": "fuse", "name": "body", "of": ["duct", "pins"]}]), encoding="utf-8")
    assert mesh2d.main(["--spec", str(spec), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "islands 0" in out
    assert re.search(r"extent\s+10 x 4", out)
    assert "gap between copies 0.8" in out


# -- the checks: what a count cannot see ------------------------------------------


def test_the_docstring_carries_no_worked_shape(mesh2d):
    """A shape described by hand is a shape nobody measured; the one that was here was
    copied into a wrong valve by every author that read it."""
    assert "Tesla" not in mesh2d.__doc__.split("There is deliberately")[0]
    assert '"op": "channel"' not in mesh2d.__doc__


def test_overlapping_copies_of_a_repeat_are_refused_with_the_overlap_measured(mesh2d, tmp_path, capsys):
    """Four bypasses at a 12 mm pitch whose footprint is 20 mm: each pair overlaps by a
    quarter of a loop, the union absorbs it, `islands 4` still holds -- and the case
    that passed every count is refused here, with the picture still drawn."""
    pytest.importorskip("gmsh")
    pytest.importorskip("matplotlib")
    spec = tmp_path / "overlap.json"
    spec.write_text(json.dumps({"ops": [
        {"op": "rect", "name": "main", "origin": [0, -1.5], "size": [60, 3]},
        {"op": "channel", "name": "bypass", "width": 3, "start": [14, 1.0], "heading": 25, "from": "y:1.5",
         "path": [{"line": 5}, {"arc": {"radius": 4.5, "angle": 200}}, {"line": {"to": "y:1.5"}}]},
        {"op": "repeat", "name": "bypasses", "target": "bypass", "count": 4, "step": [12, 0]},
        {"op": "fuse", "name": "body", "of": ["main", "bypasses"]}],
        "patches": [{"name": "inlet", "at": "x:min"}, {"name": "outlet", "at": "x:max"}]}), encoding="utf-8")
    png = tmp_path / "overlap.png"
    case = tmp_path / "case"
    assert mesh2d.main([str(case), "--spec", str(spec), "--scale", "0.001", "--preview", str(png)]) == 2
    out = capsys.readouterr().out
    assert "islands 4" in out
    assert re.search(r"!! ERROR\s+repeat 'bypasses': copies 1 and 2 overlap by 2\d\.\d", out)
    assert "not written" in out
    assert png.exists() and not (case / "Allmesh").exists()


def test_a_leg_that_lands_off_the_body_is_refused(mesh2d, tmp_path, capsys):
    """A bypass leaving at 25 degrees from x=6 comes back down 10 mm upstream of where
    it left: past the inlet face, onto nothing. Before, this was a note and a case."""
    pytest.importorskip("gmsh")
    spec = tmp_path / "off.json"
    spec.write_text(json.dumps({"ops": [
        {"op": "rect", "name": "main", "origin": [0, -1.5], "size": [60, 3]},
        {"op": "channel", "name": "bypass", "width": 3, "start": [6, 1.5], "heading": 25,
         "path": [{"line": 5}, {"arc": {"radius": 4.5, "angle": 200}}, {"line": {"to": "y:1.5"}}]},
        {"op": "fuse", "name": "body", "of": ["main", "bypass"]}]}), encoding="utf-8")
    assert mesh2d.main(["--spec", str(spec), "--scale", "0.001", "--dry-run"]) == 2
    out = capsys.readouterr().out
    assert re.search(r"!! ERROR\s+channel 'bypass': the leg to y=1.5 lands at \(-3\.9\d+, 1\.5\)", out)
    assert re.search(r"!! ERROR\s+0 inlet edges", out)


def test_a_u_bend_needs_its_ends_named_and_near_names_them(mesh2d, tmp_path, capsys):
    """Both ends of a U sit on x=0: the automatic reading calls both the inlet and
    nothing the outlet, which is refused; `near:x,y` picks each end by where it is."""
    pytest.importorskip("gmsh")
    u = {"scale": 0.001, "ops": [{"op": "channel", "name": "body", "width": 2, "start": [0, 0], "heading": 0,
         "path": [{"line": 30}, {"arc": {"radius": 3, "angle": 180}}, {"line": 30}]}]}
    spec = tmp_path / "u.json"
    spec.write_text(json.dumps(u), encoding="utf-8")
    assert mesh2d.main(["--spec", str(spec), "--scale", "0.001", "--dry-run"]) == 2
    out = capsys.readouterr().out
    assert re.search(r"!! ERROR\s+2 inlet edges", out) and re.search(r"!! ERROR\s+0 outlet edges", out)
    u["patches"] = [{"name": "inlet", "at": "near:0,0"}, {"name": "outlet", "at": "near:0,6"}]
    spec.write_text(json.dumps(u), encoding="utf-8")
    case = tmp_path / "u"
    assert mesh2d.main([str(case), "--spec", str(spec), "--scale", "0.001", "--cell", "0.0005"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"patches\s+inlet \(1 edge, 0\.002 m\), outlet \(1 edge, 0\.002 m\)", out)
    assert re.search(r"patches\s+inlet \(1\), outlet \(1\), walls", out)   # the extruded surfaces too
    assert "ERROR" not in out
    record = json.loads((case / "constant" / "geometry" / "body.json").read_text(encoding="utf-8"))
    assert record["patches"][1]["at"] == "near:0,6" and record["scale"] == 0.001


def test_the_leg_table_states_headings_radii_and_where_a_leg_lands(mesh2d, tmp_path, capsys):
    """'Leaves at 60 degrees, outer radius 4, returns heading 300 (+x: with the flow)':
    the numbers a request states, read off what was built rather than off the picture."""
    pytest.importorskip("gmsh")
    spec = tmp_path / "legs.json"
    spec.write_text(json.dumps({"ops": [
        {"op": "rect", "name": "main", "origin": [0, -1.5], "size": [60, 3]},
        {"op": "channel", "name": "bypass", "width": 3, "start": [9, 1.5], "heading": 60, "from": "y:1.5",
         "path": [{"line": 1.5}, {"arc": {"radius": 2.5, "angle": 240}}, {"line": {"to": "y:1.5"}}]},
        {"op": "fuse", "name": "body", "of": ["main", "bypass"]}],
        "patches": [{"name": "inlet", "at": "x:min"}, {"name": "outlet", "at": "x:max"}]}), encoding="utf-8")
    assert mesh2d.main(["--spec", str(spec), "--scale", "0.001", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"leg 1\s+line 1\.5\s+from \(9, 1\.5\) heading 60 deg \(\+x\)", out)
    assert re.search(r"leg 2\s+arc\s+r 2\.5 \(outer 4, inner 1\).*240 deg left: heading 60 -> 300", out)
    assert re.search(r"leg 3\s+to y=1\.5\s+from .* heading 300 deg \(\+x\) to \(6\.1\d+, 1\.5\)", out)


def test_a_channel_from_a_wall_starts_flush_with_it(mesh2d, tmp_path, capsys):
    """A leg leaving a wall at 60 degrees has a square start cap, one corner of which
    sits outside the wall; `from` cuts the start flush, so no wall edge of the union
    lies above the line the bypass leaves from except the bypass's own walls."""
    pytest.importorskip("gmsh")
    import gmsh
    ops, _ = mesh2d.parse_spec([
        {"op": "rect", "name": "main", "origin": [0, -1.5], "size": [60, 3]},
        {"op": "channel", "name": "bypass", "width": 3, "start": [9, 1.5], "heading": 60, "from": "y:1.5",
         "path": [{"line": 6}]},
        {"op": "fuse", "name": "body", "of": ["main", "bypass"]}])
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        face = mesh2d.build_face(gmsh, ops)
        built = mesh2d.measure2d(gmsh, face)
        # a flush start: the main rect's four sides (its top split either side of the
        # bypass) and the bypass's two walls and end cap -- eight edges. A square start
        # cap leaves a ninth, the piece of cap standing above the wall.
        assert len(built.curves) == 8, sorted(round(v, 3) for v in built.lengths.values())
        assert not built.short
        assert all(mesh2d.curve_bounds(gmsh, c)[1] >= 1.5 - 1e-6 for c in built.curves
                   if mesh2d.curve_bounds(gmsh, c)[3] > 1.5 + 1e-6)
    finally:
        gmsh.finalize()


def test_near_names_a_closed_curve_and_its_extruded_surface(mesh2d, tmp_path, capsys):
    """A cylinder in a channel named `near` its centre: the curve is a full circle,
    whose gmsh bounding box is a little off centre, so a tolerance match between the
    curve and the extruded surface failed and the cylinder came out as `walls` in the
    OpenFOAM mesh (study 20260907-011928-d6c4). Nearest is nearest, in both passes."""
    pytest.importorskip("gmsh")
    spec = tmp_path / "cyl.json"
    spec.write_text(json.dumps({"scale": 0.001, "ops": [
        {"op": "rect", "name": "box", "origin": [0, 0], "size": [300, 60]},
        {"op": "disk", "name": "cyl", "center": [100, 30], "radius": 5},
        {"op": "cut", "name": "body", "from": "box", "take": ["cyl"]}],
        "patches": [{"name": "inlet", "at": "x:min"}, {"name": "outlet", "at": "x:max"},
                    {"name": "walls", "at": "y:min"}, {"name": "walls", "at": "y:max"},
                    {"name": "cylinder", "at": "near:100,30"}]}), encoding="utf-8")
    case = tmp_path / "cyl"
    assert mesh2d.main([str(case), "--spec", str(spec), "--scale", "0.001", "--cell", "0.003"]) == 0
    out = capsys.readouterr().out
    assert "cylinder (1 edge, 0.03142 m)" in out
    assert re.search(r"patches\s+inlet \(1\), outlet \(1\), cylinder \(1\), walls \(2\), frontAndBack \(2\)", out), out
    assert re.search(r"entry0/cylinder/type\s+-set wall", (case / "Allmesh").read_text(encoding="utf-8"))
    assert re.search(r"cylinder\s*\{\s*type\s+noSlip", (case / "0" / "U").read_text(encoding="utf-8"))
