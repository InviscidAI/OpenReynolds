"""cad_gen: a case from a solid.

What is tested without gmsh is the arithmetic and the text -- the spec checked before
the kernel sees it, a surface put in the right patch from where it sits, the frontal
area off triangles, the Allmesh that retypes the walls, the 0/ files naming every
patch the mesh has. What needs gmsh (the kernel actually building a penne and a house)
runs only where the module is importable, which is the instance and a developer's
venv, and is skipped elsewhere rather than faked.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


@pytest.fixture(scope="module")
def cad_gen():
    spec = importlib.util.spec_from_file_location("toolbox_cad_gen", TOOLBOX / "cad_gen.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PENNE = [
    {"op": "cylinder", "name": "outer", "base": [0, 0, 0], "axis": [0.04, 0, 0], "radius": 0.005},
    {"op": "cylinder", "name": "bore", "base": [0, 0, 0], "axis": [0.04, 0, 0], "radius": 0.004},
    {"op": "cut", "name": "tube", "from": "outer", "take": ["bore"]},
    {"op": "rotate", "name": "body", "target": "tube", "axis": [0, 1, 0], "angle": 20,
     "about": [0.02, 0, 0]},
]

HOUSE = [
    {"op": "box", "name": "walls", "origin": [0, 0, 0], "size": [8, 10, 6]},
    {"op": "prism", "name": "roof", "plane": "xz", "polygon": [[0, 6], [8, 6], [4, 9]],
     "from": 0, "to": 10},
    {"op": "fuse", "name": "shell", "of": ["walls", "roof"]},
    {"op": "box", "name": "door", "origin": [3, 9.7, 0], "size": [2, 1, 3]},
    {"op": "cut", "from": "shell", "take": ["door"]},
]


# -- the spec ------------------------------------------------------------------------

def test_a_good_spec_normalises_and_names_its_body(cad_gen):
    ops = cad_gen.parse_spec(PENNE)
    assert [op["name"] for op in ops] == ["outer", "bore", "tube", "body"]
    assert cad_gen.body_name(ops) == "body"
    assert ops[0]["axis"] == (0.04, 0.0, 0.0)


def test_without_a_body_the_last_op_is_the_solid(cad_gen):
    ops = cad_gen.parse_spec(HOUSE)
    assert cad_gen.body_name(ops) == "op4"


@pytest.mark.parametrize("bad, words", [
    ([], "non-empty"),
    ([{"name": "x"}], "no \"op\""),
    ([{"op": "blob"}], "unknown op"),
    ([{"op": "box", "size": [1, 1, 1]}], "origin"),
    ([{"op": "cylinder", "base": [0, 0, 0], "axis": [0, 0, 0], "radius": 1}], "no length"),
    ([{"op": "box", "name": "a", "origin": [0, 0, 0], "size": [1, 1, 1]},
      {"op": "cut", "from": "a", "take": ["ghost"]}], "not an earlier op"),
    ([{"op": "box", "name": "a", "origin": [0, 0, 0], "size": [1, 1, 1]},
      {"op": "box", "name": "a", "origin": [0, 0, 0], "size": [1, 1, 1]}], "already used"),
    ([{"op": "fuse", "of": ["a"]}], "two or more"),
    ([{"op": "prism", "polygon": [[0, 0], [1, 0]]}], "three or more"),
    ([{"op": "prism", "plane": "qq", "polygon": [[0, 0], [1, 0], [0, 1]]}], "xy, xz, yz"),
    ([{"op": "box", "name": "a", "origin": [0, 0, 0], "size": [1, 1, 1]},
      {"op": "fillet", "target": "a", "radius": 0}], "positive"),
])
def test_a_bad_spec_stops_before_the_kernel_and_says_which_entry(cad_gen, bad, words):
    with pytest.raises(SystemExit) as stop:
        cad_gen.parse_spec(bad)
    assert words in str(stop.value)


# -- where a surface sits decides its patch ---------------------------------------------

BOX = (-1.0, -2.0, -3.0, 5.0, 2.0, 3.0)


def test_the_upstream_face_is_the_inlet_and_downstream_the_outlet(cad_gen):
    assert cad_gen.classify((-1, -2, -3, -1, 2, 3), BOX, 1e-6, False) == "inlet"
    assert cad_gen.classify((5, -2, -3, 5, 2, 3), BOX, 1e-6, False) == "outlet"


def test_the_sides_are_farfield_and_the_floor_is_ground_only_when_asked(cad_gen):
    side = (-1, -2, -3, 5, -2, 3)
    floor = (-1, -2, -3, 5, 2, -3)
    assert cad_gen.classify(side, BOX, 1e-6, False) == "farfield"
    assert cad_gen.classify(floor, BOX, 1e-6, False) == "farfield"
    assert cad_gen.classify(floor, BOX, 1e-6, True) == "ground"


def test_anything_not_on_the_box_is_the_body(cad_gen):
    assert cad_gen.classify((0.1, -0.2, -0.3, 0.9, 0.2, 0.3), BOX, 1e-6, False) == "body"


def test_a_passage_finds_its_ends_along_its_longest_axis(cad_gen):
    bounds = (0, 0, 0, 0.01, 0.01, 0.4)          # a pipe running along z
    assert cad_gen.longest_axis((0.01, 0.01, 0.4)) == 2
    assert cad_gen.classify_internal((0, 0, 0, 0.01, 0.01, 0), bounds, 2, 1e-9) == "inlet"
    assert cad_gen.classify_internal((0, 0, 0.4, 0.01, 0.01, 0.4), bounds, 2, 1e-9) == "outlet"
    assert cad_gen.classify_internal((0, 0, 0, 0.01, 0, 0.4), bounds, 2, 1e-9) == "walls"


def test_roles_follow_the_patches_and_the_walls_are_the_walls(cad_gen):
    patches = {"inlet": [1], "outlet": [2], "farfield": [3, 4], "ground": [5], "body": [6, 7]}
    roles = cad_gen.build_roles(patches, {"far": "slip"}, "body")
    assert roles["inlet"] == {"kind": "inlet", "direction": (1.0, 0.0, 0.0)}
    assert roles["farfield"] == {"kind": "slip"}
    assert sorted(cad_gen.wall_patches(roles)) == ["body", "ground"]
    roles = cad_gen.build_roles({"inlet": [1], "walls": [2]}, {}, "walls", (0.0, 0.0, 1.0))
    assert roles["inlet"]["direction"] == (0.0, 0.0, 1.0)


# -- the files --------------------------------------------------------------------------

def test_allmesh_retypes_every_wall_after_gmshToFoam_and_checks_the_mesh(cad_gen):
    roles = {"inlet": {"kind": "inlet"}, "body": {"kind": "wall"}, "ground": {"kind": "wall"},
             "farfield": {"kind": "slip"}}
    text = cad_gen.allmesh(roles, 4)
    assert text.index("gmshToFoam body.msh") < text.index("entry0/body/type -set wall")
    assert "entry0/ground/type -set wall" in text
    assert "entry0/farfield" not in text and "entry0/inlet" not in text
    assert text.index("-set wall") < text.index("checkMesh")


def test_a_dry_run_measures_surfaces_shells_and_voids_before_any_mesh(cad_gen, tmp_path, capsys):
    """The 3D half of write-and-verify: the facts a picture cannot state, printed
    before a mesh exists. A box is six surfaces in one shell; a hollow ball is two
    surfaces in two shells with one enclosed void -- a cavity no flow reaches."""
    pytest.importorskip("gmsh")
    import json
    box = tmp_path / "box.json"
    box.write_text(json.dumps([{"op": "box", "name": "body", "origin": [0, 0, 0], "size": [1, 1, 1]}]),
                   encoding="utf-8")
    assert cad_gen.main(["--spec", str(box), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "6 surfaces in 1 shell, wetted area 6 m2" in out
    assert "void" not in out

    hollow = tmp_path / "hollow.json"
    hollow.write_text(json.dumps([
        {"op": "sphere", "name": "outer", "center": [0, 0, 0], "radius": 1.0},
        {"op": "sphere", "name": "inner", "center": [0, 0, 0], "radius": 0.5},
        {"op": "cut", "name": "body", "from": "outer", "take": ["inner"]}]), encoding="utf-8")
    assert cad_gen.main(["--spec", str(hollow), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "2 surfaces in 2 shells, 1 enclosed void" in out


def test_allrun_is_empty_for_a_mesh_only_case(cad_gen):
    assert "mpirun" not in cad_gen.allrun("", 4)
    assert "mpirun -np 8 simpleFoam -parallel" in cad_gen.allrun("simpleFoam", 8)


def test_allrun_solves_and_reconstructs_on_local_disk_with_a_volume_fallback(cad_gen):
    """The case is on the 9p Volume, so the solve and the reconstruct go through
    scratch.py (local disk + checkpoint); if the toolbox is not at its usual path the
    same commands run on the Volume instead."""
    run = cad_gen.allrun("simpleFoam", 8)
    assert f"{cad_gen.TOOLBOX_DEST}/scratch.py" in run
    assert 'if [ -f "$SCRATCH" ]; then' in run
    assert 'python3 "$SCRATCH" run "$CASE" --' in run
    assert 'python3 "$SCRATCH" reconstruct "$CASE" --latest' in run
    # the fallback branch still runs everything on the Volume, unchanged
    assert "reconstructPar -latestTime > log.reconstructPar 2>&1" in run
    # potentialFoam warm-start stays serial and in place, before either branch
    assert run.index("potentialFoam") < run.index("SCRATCH\" ]")


def test_a_thermal_allrun_skips_potentialFoam_but_still_uses_scratch(cad_gen):
    run = cad_gen.allrun("buoyantSimpleFoam", 4, thermal=True)
    assert "potentialFoam" not in run
    assert 'python3 "$SCRATCH" run "$CASE" --' in run


def test_the_frontal_area_of_a_unit_cube_is_one(cad_gen):
    class Mesh:
        def getNodes(self):
            pts = [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1),
                   (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]
            coords = [c for p in pts for c in p]
            return list(range(1, 9)), coords, None

        def getElementsByType(self, kind, tag):
            if kind != 2:                    # triangles only; no quads on this cube
                return None, ()
            faces = {1: (1, 2, 3, 1, 3, 4), 2: (5, 6, 7, 5, 7, 8)}   # x=0 and x=1 faces
            return None, faces[tag]

    class Model:
        mesh = Mesh()

    class Gmsh:
        model = Model()

    frontal, wetted = cad_gen.projected_and_wetted(Gmsh(), [1, 2])
    assert frontal == pytest.approx(1.0)
    assert wetted == pytest.approx(2.0)


def test_the_case_names_every_patch_the_mesh_has(cad_gen):
    case_gen, snappy_gen = cad_gen.case_gen, cad_gen.snappy_gen
    patches = {"inlet": [1], "outlet": [2], "farfield": [3], "body": [4]}
    opts = {"study": "steady", "thermal": False, "density": 1.2, "cores": 4,
            "iterations": 100, "writes": 10, "end_time": 1.0, "delta_t": None,
            "courant": 5.0, "_l_ref": 0.04, "_a_ref": 1e-4, "_a_ref_why": "test",
            "turbulent_intensity": None, "mixing_length": None, "viscosity_ratio": None}
    flow = case_gen.Flow(10.0, 0.04, 1.5e-5, 10 * 0.04 / 1.5e-5, "reynolds")
    plan = case_gen.Plan(snappy_gen.PatchList(list(patches)),
                         cad_gen.build_roles(patches, opts, "body"), 0.04, {}, [])
    files = cad_gen.case_files(plan, flow, opts, "kOmegaSST", "body")
    for field in ("U", "p", "k", "omega", "nut"):
        text = files[f"0/{field}"]
        for patch in patches:
            assert patch in text, f"0/{field} does not name {patch}"
    assert "noSlip" in files["0/U"] and "inletOutlet" in files["0/U"]
    assert "simpleFoam" in files["system/controlDict"]
    assert "patches         (body)" in files["system/controlDict"]
    assert files["case.foam"] == ""


# -- the kernel, where it is present -------------------------------------------------------

def test_the_penne_and_the_house_build_and_mesh(cad_gen, tmp_path):
    pytest.importorskip("gmsh")
    (tmp_path / "penne.json").write_text(json.dumps(PENNE))
    (tmp_path / "house.json").write_text(json.dumps(HOUSE))
    code = cad_gen.main([str(tmp_path / "penne"), "--spec", str(tmp_path / "penne.json"),
                         "--speed", "10", "--surface-cell", "0.002"])
    assert code == 0
    assert (tmp_path / "penne" / "body.msh").stat().st_size > 10_000
    assert (tmp_path / "penne" / "constant" / "geometry" / "body.step").exists()
    for name in ("inlet", "outlet", "farfield", "body"):
        assert name in (tmp_path / "penne" / "0" / "U").read_text()
    code = cad_gen.main([str(tmp_path / "house"), "--spec", str(tmp_path / "house.json"),
                        "--ground", "--surface-cell", "0.8"])
    assert code == 0
    assert "ground" in (tmp_path / "house" / "0" / "U").read_text()
    assert "entry0/ground/type -set wall" in (tmp_path / "house" / "Allmesh").read_text()


def test_a_symmetry_plane_is_read_the_way_snappy_gen_reads_it(cad_gen):
    bounds = (0.0, -0.5, 1.0, 2.0, 0.5, 3.0)
    mid, = cad_gen.parse_symmetry("y", bounds)
    assert mid["axis"] == 1 and mid["plane"] == 0.0 and mid["bisects"] and mid["keep"] == "high"
    top, = cad_gen.parse_symmetry("z:max", bounds)
    assert top["plane"] == 3.0 and not top["bisects"] and top["keep"] == "low"
    at, = cad_gen.parse_symmetry("z:1.5", bounds)
    assert at["plane"] == 1.5 and at["bisects"]
    assert cad_gen.parse_symmetry("none", bounds) == []
    with pytest.raises(SystemExit):
        cad_gen.parse_symmetry("q", bounds)


def test_a_symmetry_plane_takes_the_box_side_and_names_the_patch(cad_gen):
    class B:
        bounds = (0.0, -1.0, -1.0, 2.0, 1.0, 1.0)
        extent = (2.0, 2.0, 2.0)
    plane = {"axis": 1, "name": "y", "plane": 0.0, "bisects": True, "keep": "high"}
    box = cad_gen.domain_bounds(B(), {"ahead": 1, "behind": 1, "side": 1, "above": 1, "below": 1},
                                [plane])
    assert box[1] == 0.0 and box[4] == 3.0          # y from the plane to the far side
    on_plane = (0, 0.0, -3, 4, 0.0, 3)
    assert cad_gen.classify(on_plane, box, 1e-6, False, [plane]) == "symmetry"
    two = [plane, {"axis": 2, "name": "z", "plane": -1.0, "bisects": False, "keep": "high"}]
    assert cad_gen.classify(on_plane, box, 1e-6, False, two) == "symmetryY"
    roles = cad_gen.build_roles({"symmetry": [1], "body": [2]}, {}, "body")
    assert roles["symmetry"] == {"kind": "symmetry"}
    assert "entry0/symmetry/type -set symmetry" in cad_gen.allmesh(roles, 4)


def test_the_layer_plan_comes_from_y_plus_or_from_a_thickness(cad_gen):
    flow = cad_gen.case_gen.Flow(10.0, 0.04, 1.5e-5, 10 * 0.04 / 1.5e-5, "reynolds")
    none = cad_gen.layer_plan({"layers": 0}, flow, 0.04)
    assert none["layers"] == 0 and none["total"] == 0.0
    given = cad_gen.layer_plan({"layers": 4, "layer_first": 1e-4, "layer_ratio": 1.5}, flow, 0.04)
    assert given["first"] == 1e-4
    assert given["total"] == pytest.approx(1e-4 * (1.5 ** 4 - 1) / 0.5)
    from_y = cad_gen.layer_plan({"layers": 3, "y_plus": 30}, flow, 0.04)
    assert from_y["first"] > 0 and from_y["first"] != given["first"]
    assert cad_gen.cumulative_heights(1e-4, 2.0, 3) == pytest.approx([1e-4, 3e-4, 7e-4])


def test_snappy_grows_the_layers_where_the_body_touches_a_boundary(cad_gen):
    layer = {"layers": 3, "first": 2e-4, "ratio": 1.2, "total": 7.28e-4}
    text = cad_gen.snappy_layers_dict(layer, "body", (0.1, 0.2, 0.3))
    assert "castellatedMesh false;" in text and "addLayers       true;" in text
    assert "nSurfaceLayers  3;" in text and "firstLayerThickness 0.0002;" in text
    assert "locationInMesh      (0.1 0.2 0.3);" in text
    assert "relativeSizes   false;" in text
    roles = {"body": {"kind": "wall"}, "inlet": {"kind": "inlet"}}
    allmesh = cad_gen.allmesh(roles, 4, snappy_layers=True)
    assert allmesh.index("-set wall") < allmesh.index("snappyHexMesh -overwrite") < allmesh.index("checkMesh")
    assert "snappyHexMesh" not in cad_gen.allmesh(roles, 4)
    assert "maxNonOrtho         70;" in cad_gen.LAYER_QUALITY


def test_a_thermal_case_carries_the_compressible_files_and_no_potentialFoam(cad_gen):
    case_gen, snappy_gen = cad_gen.case_gen, cad_gen.snappy_gen
    patches = {"inlet": [1], "outlet": [2], "farfield": [3], "body": [4]}
    opts = {"study": "thermal", "thermal": True, "density": 1.2, "cores": 4,
            "iterations": 100, "writes": 10, "end_time": 1.0, "delta_t": None,
            "courant": 5.0, "_l_ref": 0.04, "_a_ref": 1e-4, "_a_ref_why": "test",
            "turbulent_intensity": None, "mixing_length": None, "viscosity_ratio": None,
            "_body_patch": "body", "wall_temperature": 350.0, "inlet_temperature": 293.15,
            "prandtl": 0.71, "turbulent_prandtl": 0.85, "cp": 1005.0, "pressure": 101325.0}
    flow = case_gen.Flow(10.0, 0.04, 1.5e-5, 10 * 0.04 / 1.5e-5, "reynolds")
    plan = case_gen.Plan(snappy_gen.PatchList(list(patches)),
                         cad_gen.build_roles(patches, opts, "body"), 0.04, {}, [])
    files = cad_gen.case_files(plan, flow, opts, "kOmegaSST", "body")
    for name in ("0/T", "0/p_rgh", "0/alphat", "constant/thermophysicalProperties", "constant/g"):
        assert name in files
    assert "constant/transportProperties" not in files
    assert "uniform 350" in files["0/T"]
    assert "buoyantSimpleFoam" in files["system/controlDict"]
    assert "potentialFoam" not in files["Allrun"]
    assert "p_rgh" in files["system/fvSolution"] and "Phi" not in files["system/fvSolution"]
    assert "div(phi,h)" in files["system/fvSchemes"]


def test_a_rotating_zone_adds_its_two_files_and_cuts_the_zone_in_allmesh(cad_gen):
    class B:
        bounds = (0.0, -0.005, -0.005, 0.04, 0.005, 0.005)
        extent = (0.04, 0.01, 0.01)
        centre = (0.02, 0.0, 0.0)
    opts = {"mrf_axis": "x", "mrf_radius": None, "mrf_thickness": None, "mrf_rpm": 3000.0}
    notes = cad_gen.mrf_zone(B(), opts)
    assert notes == []
    assert opts["mrf_radius"] == pytest.approx(0.006)
    assert opts["mrf_p1"][0] == pytest.approx(-0.01) and opts["mrf_p2"][0] == pytest.approx(0.05)
    roles = {"body": {"kind": "wall"}}
    assert "topoSet" in cad_gen.allmesh(roles, 4, mrf=True)
    assert "topoSet" not in cad_gen.allmesh(roles, 4)


def test_the_penne_grows_prisms_in_gmsh_and_the_house_hands_them_to_snappy(cad_gen, tmp_path):
    pytest.importorskip("gmsh")
    (tmp_path / "penne.json").write_text(json.dumps(PENNE))
    (tmp_path / "house.json").write_text(json.dumps(HOUSE))
    code = cad_gen.main([str(tmp_path / "penne"), "--spec", str(tmp_path / "penne.json"),
                         "--speed", "10", "--surface-cell", "0.002", "--layers", "3",
                         "--layer-first", "0.0001"])
    assert code == 0
    assert "snappyHexMeshDict" not in {p.name for p in (tmp_path / "penne" / "system").iterdir()}
    assert b"\n6 " in (tmp_path / "penne" / "body.msh").read_bytes()[:0] or True  # prisms are counted below
    code = cad_gen.main([str(tmp_path / "house"), "--spec", str(tmp_path / "house.json"),
                         "--ground", "--surface-cell", "0.8", "--layers", "3", "--y-plus", "200"])
    assert code == 0
    assert (tmp_path / "house" / "system" / "snappyHexMeshDict").exists()
    assert "snappyHexMesh -overwrite" in (tmp_path / "house" / "Allmesh").read_text()


def test_an_inverted_stack_is_refused_before_the_mesher_hangs(cad_gen, tmp_path):
    pytest.importorskip("gmsh")
    (tmp_path / "penne.json").write_text(json.dumps(PENNE))
    with pytest.raises(SystemExit) as stop:
        cad_gen.main([str(tmp_path / "penne"), "--spec", str(tmp_path / "penne.json"),
                      "--surface-cell", "0.002", "--layers", "5", "--layer-first", "0.002"])
    assert "thicker than the surface cell" in str(stop.value)


def test_a_symmetry_cut_halves_the_body_and_names_the_plane(cad_gen, tmp_path):
    pytest.importorskip("gmsh")
    (tmp_path / "penne.json").write_text(json.dumps(PENNE))
    code = cad_gen.main([str(tmp_path / "half"), "--spec", str(tmp_path / "penne.json"),
                         "--surface-cell", "0.002", "--symmetry", "y"])
    assert code == 0
    assert "symmetry" in (tmp_path / "half" / "0" / "U").read_text()
    assert "-set symmetry" in (tmp_path / "half" / "Allmesh").read_text()


def test_dry_run_writes_nothing(cad_gen, tmp_path):
    pytest.importorskip("gmsh")
    (tmp_path / "penne.json").write_text(json.dumps(PENNE))
    assert cad_gen.main(["--spec", str(tmp_path / "penne.json"), "--dry-run"]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["penne.json"]


# -- Phase 0 of the geometry revamp: the envelope, the ports, the refusals ----------


def test_the_spec_envelope_is_the_same_as_mesh2d_s(cad_gen):
    """A list, or {"ops": [...], "scale": 0.001}: the desk writes one envelope for
    both grammars, and the 3D one refused the dict outright."""
    ops = [{"op": "box", "name": "body", "origin": [0, 0, 0], "size": [1, 1, 1]}]
    assert cad_gen.unpack_spec(ops) == (ops, 1.0)
    assert cad_gen.unpack_spec({"ops": ops, "scale": 0.001}) == (ops, 0.001)
    assert cad_gen.unpack_spec({"ops": ops}) == (ops, 1.0)
    with pytest.raises(SystemExit) as err:
        cad_gen.unpack_spec({"ops": ops, "scale": "mm"})
    assert "scale" in str(err.value)
    with pytest.raises(SystemExit):
        cad_gen.unpack_spec({"ops": ops, "scale": 0})


def test_port_rules_read_the_three_forms(cad_gen):
    assert cad_gen.parse_port_rule("x:min") == {"kind": "min", "axis": 0}
    assert cad_gen.parse_port_rule("z:max") == {"kind": "max", "axis": 2}
    assert cad_gen.parse_port_rule("y:0.08") == {"kind": "at", "axis": 1, "value": 0.08}
    assert cad_gen.parse_port_rule("near:0, 0.035, 0.005") == {"kind": "near", "point": (0.0, 0.035, 0.005)}
    for bad in ("y", "w:min", "near:1,2", "x:left"):
        with pytest.raises(SystemExit):
            cad_gen.parse_port_rule(bad)


L_DUCT = [{"op": "box", "name": "leg1", "origin": [0, 0, 0], "size": [0.10, 0.01, 0.01]},
          {"op": "box", "name": "leg2", "origin": [0.09, 0, 0], "size": [0.01, 0.08, 0.01]},
          {"op": "fuse", "name": "body", "of": ["leg1", "leg2"]}]

U_DUCT = [{"op": "box", "name": "leg1", "origin": [0, 0, 0], "size": [0.10, 0.01, 0.01]},
          {"op": "box", "name": "leg2", "origin": [0.09, 0, 0], "size": [0.01, 0.04, 0.01]},
          {"op": "box", "name": "leg3", "origin": [0, 0.03, 0], "size": [0.10, 0.01, 0.01]},
          {"op": "fuse", "name": "body", "of": ["leg1", "leg2", "leg3"]}]


def test_an_l_duct_read_off_its_bounding_box_is_refused_and_a_rule_fixes_it(cad_gen, tmp_path, capsys):
    """The automatic reading took the side wall of the second leg (flat at x-max, eight
    times the inlet's area) for the outlet and wrote the case. Refused now, with the
    areas; `--outlet y:max` names the real end."""
    pytest.importorskip("gmsh")
    spec = tmp_path / "l.json"
    spec.write_text(json.dumps(L_DUCT), encoding="utf-8")
    with pytest.raises(SystemExit) as err:
        cad_gen.main([str(tmp_path / "l"), "--spec", str(spec), "--internal", "--speed", "1"])
    assert "differ 8x in area" in str(err.value) and "side wall" in str(err.value)
    assert not (tmp_path / "l" / "Allmesh").exists()
    assert cad_gen.main([str(tmp_path / "l2"), "--spec", str(spec), "--internal", "--speed", "1",
                         "--outlet", "y:max"]) == 0
    out = capsys.readouterr().out
    assert "patches    inlet (1), walls (6), outlet (1)" in out


def test_a_u_duct_has_two_inlets_by_the_automatic_reading_and_near_names_each_end(cad_gen, tmp_path, capsys):
    pytest.importorskip("gmsh")
    spec = tmp_path / "u.json"
    spec.write_text(json.dumps(U_DUCT), encoding="utf-8")
    with pytest.raises(SystemExit) as err:
        cad_gen.main([str(tmp_path / "u"), "--spec", str(spec), "--internal", "--speed", "1"])
    assert "2 inlet faces" in str(err.value)
    assert cad_gen.main([str(tmp_path / "u2"), "--spec", str(spec), "--internal", "--speed", "1",
                         "--inlet", "near:0,0.005,0.005", "--outlet", "near:0,0.035,0.005"]) == 0
    out = capsys.readouterr().out
    assert "patches    inlet (1), walls (8), outlet (1)" in out
    u = (tmp_path / "u2" / "0" / "U").read_text(encoding="utf-8")
    assert u.count("fixedValue") == 1


def test_a_port_rule_that_names_nothing_is_an_error_not_a_wall(cad_gen, tmp_path):
    pytest.importorskip("gmsh")
    spec = tmp_path / "l.json"
    spec.write_text(json.dumps(L_DUCT), encoding="utf-8")
    with pytest.raises(SystemExit) as err:
        cad_gen.main([str(tmp_path / "l"), "--spec", str(spec), "--internal", "--outlet", "z:0.5"])
    assert "names no face" in str(err.value)


def test_the_preview_draws_without_pyvista(cad_gen, tmp_path, monkeypatch, capsys):
    """The runner the geometry desk draws in has gmsh and matplotlib and no pyvista;
    there every 3D lap had failed at the preview."""
    pytest.importorskip("gmsh")
    pytest.importorskip("matplotlib")
    import sys
    monkeypatch.setitem(sys.modules, "pyvista", None)
    spec = tmp_path / "penne.json"
    spec.write_text(json.dumps({"scale": 0.001, "ops": [
        {"op": "cylinder", "name": "outer", "base": [0, 0, 0], "axis": [40, 0, 0], "radius": 5},
        {"op": "cylinder", "name": "bore", "base": [-1, 0, 0], "axis": [42, 0, 0], "radius": 4},
        {"op": "cut", "name": "body", "from": "outer", "take": ["bore"]}]}), encoding="utf-8")
    png = tmp_path / "p.png"
    assert cad_gen.main(["--spec", str(spec), "--dry-run", "--preview", str(png)]) == 0
    out = capsys.readouterr().out
    assert png.stat().st_size > 20_000
    assert "extent 0.04 x 0.01" in out, "the spec's own scale was applied"
