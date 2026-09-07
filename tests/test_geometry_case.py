"""The case on disk and the finish on the instance (DESIGN.md 3.13).

`write_case` hands the record to `mesh2d.main` in a child and returns the sizes the
writer meshed with; the finish reads the OpenFOAM boundary file back so a patch the
extrusion lost is found in the same tool call as the mesh (the T05 class).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from openreynolds.backend.base import ExecResult
from openreynolds.geometry import case
from openreynolds.geometry.claims import ComplianceRow, ComplianceTable
from openreynolds.geometry.strategy import Sizes

from geometry_fakes import BOUNDARY, CHECKMESH_LOG, PNG, measurements_dict

T05_CLAIMS = {
    "schema": "openreynolds.geometry/claims-1", "unit": "mm", "kind": "passage", "flow": "+x",
    "claims": [
        {"id": "c1", "kind": "measure", "says": "a 10 mm cylinder", "measure": "diameter", "of": "cyl", "value": 10},
        {"id": "c2", "kind": "measure", "says": "a channel 60 mm high", "measure": "height", "of": "duct", "value": 60},
        {"id": "c3", "kind": "measure", "says": "300 mm long", "measure": "length", "of": "duct", "value": 300},
        {"id": "c10", "kind": "count", "says": "one cylinder", "of": "holes", "value": 1},
    ],
}

T05_SCRIPT = '''s = Sketch(units="mm")
duct = s.rect(origin=(0, 0), size=(300, 60), name="duct")
cyl = s.disk(centre=(100, 30), diameter=10, name="cyl")
s.fluid = duct - cyl
s.inlet(duct.left)
s.outlet(duct.right)
s.wall(cyl.edge, name="cylinder")
s.wall(duct.top, duct.bottom, name="walls")
'''


def t05_record() -> dict:
    """The T05 record of DESIGN.md 7.4 as far as the case writer reads it: the ops,
    the resolved patch rules, the claims, the measurements."""
    return {
        "format": "openreynolds.geometry/1", "units": "mm", "scale": 0.001,
        "ops": [
            {"op": "rect", "name": "duct", "origin": [0, 0], "size": [300, 60]},
            {"op": "disk", "name": "cyl", "center": [100, 30], "radius": 5},
            {"op": "cut", "name": "fluid", "from": "duct", "take": ["cyl"]},
        ],
        "patches": [
            {"name": "inlet", "at": "x:min"}, {"name": "outlet", "at": "x:max"},
            {"name": "cylinder", "at": "near:95,30"},
        ],
        "ports": [{"name": "inlet", "kind": "inlet", "edges": ["duct.left"]},
                  {"name": "outlet", "kind": "outlet", "edges": ["duct.right"]}],
        "script": T05_SCRIPT, "features": {}, "instances": {}, "apart": [],
        "claims": T05_CLAIMS, "lint": [],
        "measurements": measurements_dict(extent=(300.0, 60.0), passage_min=25.0),
    }


def t05_table() -> ComplianceTable:
    return ComplianceTable(rows=[ComplianceRow(id=c["id"], says=c["says"], kind=c["kind"], expected=str(c["value"]),
                                               measured="measured", verdict="pass") for c in T05_CLAIMS["claims"]])


def test_write_case_writes_the_five_files_and_mesh2d_writes_the_rest(tmp_path):
    pytest.importorskip("gmsh")
    local = tmp_path / "cyl"
    paths, sizes = case.write_case(local, t05_record(), T05_SCRIPT, T05_CLAIMS, t05_table(), "mesh", 0.001)
    for name in ("geometry.json", "geometry.py", "claims.json", "compliance.json", "outline.png",
                 "body.msh", "Allmesh", "system/controlDict", "constant/geometry/body.step",
                 "constant/geometry/body.json"):
        assert (local / name).exists(), name
    assert isinstance(paths, case.CasePaths) and paths.record == "geometry.json" and paths.case_rel == ""
    assert (local / paths.script).read_text(encoding="utf-8") == T05_SCRIPT
    assert json.loads((local / paths.claims).read_text())["claims"][0]["id"] == "c1"
    assert json.loads((local / paths.compliance).read_text())["rows"][0]["verdict"] == "pass"
    record = json.loads((local / paths.record).read_text())
    assert record["claims"] == T05_CLAIMS and record["ops"][2]["op"] == "cut"
    # mesh2d copies the spec it read, so the instance's copy carries the claims too
    body = json.loads((local / "constant" / "geometry" / "body.json").read_text())
    assert body["claims"] == T05_CLAIMS and body["scale"] == 0.001
    # the sizes are the writer's: mesh_sizes2d on the extent in metres, short side / 30
    assert isinstance(sizes, Sizes)
    assert sizes.cell == pytest.approx(0.06 / 30) and sizes.wall_cell == sizes.cell and sizes.thickness == sizes.cell
    assert b"cylinder" in (local / "body.msh").read_bytes()


def test_write_case_reports_the_writers_refusal(tmp_path):
    pytest.importorskip("gmsh")
    record = t05_record()
    record["ops"][0]["size"] = [0, 60]
    with pytest.raises(RuntimeError, match="size must be positive"):
        case.write_case(tmp_path / "bad", record, T05_SCRIPT, None, None, "mesh", 0.001)


def test_the_sizes_come_from_the_record_or_the_writers_own_line():
    assert case.extent_m({"measurements": {"extent": [60, 14.25]}}, "", 0.001) == pytest.approx((0.06, 0.01425))
    assert case.extent_m({}, "shape      extent 0.3 x 0.06 m, area 0.0179 m2", 0.001) == pytest.approx((0.3, 0.06))
    with pytest.raises(RuntimeError, match="measurements"):
        case.extent_m({}, "nothing here", 0.001)


def test_case_args_are_todays_writer_call(tmp_path):
    args = case.case_args(tmp_path, "mesh", 0.001, case.CasePaths())
    assert args[1].endswith("mesh2d.py") and args[2] == str(tmp_path)
    assert "--spec" in args and args[args.index("--spec") + 1] == str(tmp_path / "geometry.json")
    assert args[-2:] == ["--scale", "0.001"] and "--force" in args and "--study" in args
    assert "--scale" not in case.case_args(tmp_path, "mesh", 1.0, case.CasePaths())


def test_patches_agree_on_a_boundary_file():
    assert case.boundary_patch_names(BOUNDARY) == ["inlet", "outlet", "walls", "frontAndBack"]
    assert case.patches_agree(BOUNDARY, ["inlet", "outlet"]) == ([], [])
    assert case.patches_agree(BOUNDARY, ["inlet", "outlet", "cylinder"]) == (["cylinder"], [])
    with_cyl = BOUNDARY.replace("    walls\n", "    cylinder\n    {\n        type wall;\n    }\n    walls\n")
    assert case.patches_agree(with_cyl, ["inlet", "outlet", "cylinder"]) == ([], [])
    assert case.patches_agree(with_cyl, ["inlet", "outlet"]) == ([], ["cylinder"])
    # walls and frontAndBack are expected of every 2D case whether or not the record names them
    assert case.patches_agree(BOUNDARY.replace("frontAndBack", "sides"), ["inlet", "outlet"]) == (["frontAndBack"], ["sides"])
    assert "FoamFile" not in case.boundary_patch_names(BOUNDARY)


def test_the_finish_reads_the_boundary_back(backend):
    backend.exec_result = ExecResult(0, "Mesh OK.", False, None)
    backend.files["/work/s/cyl/log.checkMesh"] = CHECKMESH_LOG.encode()
    backend.files["/work/s/cyl/renders/mesh_z.png"] = PNG + b"mesh"
    backend.files["/work/s/cyl/constant/polyMesh/boundary"] = BOUNDARY.encode()
    report = case.finish_on_instance(backend, "/work/s/cyl", ["inlet", "outlet", "cylinder"], 240)
    cmd, cwd, timeout = backend.last_exec
    assert cmd == case.FINISH_CMD and cwd == "/work/s/cyl" and timeout == 240
    assert "sh Allmesh" in cmd and "--scene mesh" in cmd and "render.py . " not in cmd
    assert report.rc == 0 and "Mesh OK" in report.digest and report.digest_data["counts"]["cells"] == 3750
    assert report.mesh_png == PNG + b"mesh"
    assert report.boundary_patches == ["inlet", "outlet", "walls", "frontAndBack"]
    assert report.missing == ["cylinder"] and report.extra == []
    assert case.FinishReport.from_dict(report.as_dict()) == report


def test_a_finish_without_a_log_or_a_boundary_says_what_it_has(backend):
    backend.exec_result = ExecResult(1, "gmshToFoam: cannot open body.msh", False, None)
    report = case.finish_on_instance(backend, "/work/s/cyl", ["inlet"], 240)
    assert report.rc == 1 and report.digest == "" and report.digest_data is None
    assert report.mesh_png is None and report.boundary_patches == [] and report.missing == []
    assert "cannot open body.msh" in report.output


def test_a_backend_that_cannot_exec_is_a_finish_that_did_not_run():
    report = case.finish_on_instance(object(), "/work/s/cyl", ["inlet"], 240)
    assert report.rc == 1 and "was not run" in report.output
