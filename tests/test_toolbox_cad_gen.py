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


def test_allrun_is_empty_for_a_mesh_only_case(cad_gen):
    assert "mpirun" not in cad_gen.allrun("", 4)
    assert "mpirun -np 8 simpleFoam -parallel" in cad_gen.allrun("simpleFoam", 8)


def test_the_frontal_area_of_a_unit_cube_is_one(cad_gen):
    class Mesh:
        def getNodes(self):
            pts = [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1),
                   (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]
            coords = [c for p in pts for c in p]
            return list(range(1, 9)), coords, None

        def getElementsByType(self, kind, tag):
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


def test_dry_run_writes_nothing(cad_gen, tmp_path):
    pytest.importorskip("gmsh")
    (tmp_path / "penne.json").write_text(json.dumps(PENNE))
    assert cad_gen.main(["--spec", str(tmp_path / "penne.json"), "--dry-run"]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["penne.json"]
