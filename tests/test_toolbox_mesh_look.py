"""`mesh_look.py`: the instrument the mesh desk, the main agent and a person all use.

The drawing needs pyvista and a mesh, so what is pinned here is everything around it --
the boundary file read as text with no OpenFOAM environment at all, the checkMesh log
read for its verdict and its three metrics, what counts as a rebuild script, and the
report a person actually reads.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load():
    spec = importlib.util.spec_from_file_location("mesh_look", TOOLBOX / "mesh_look.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mesh_look = load()


BOUNDARY = """\
FoamFile { version 2.0; format ascii; class polyBoundaryMesh; object boundary; }

4
(
    inlet
    {
        type            patch;
        nFaces          20;
        startFace       7000;
    }
    outlet
    {
        type            patch;
        nFaces          20;
        startFace       7020;
    }
    walls
    {
        type            wall;
        inGroups        1(wall);
        nFaces          400;
        startFace       7040;
    }
    frontAndBack
    {
        type            empty;
        nFaces          7500;
        startFace       7440;
    }
)
"""

CHECK_OK = """\
Create polyMesh for time = constant

Mesh stats
    points:           7600
    faces:            15000
    cells:            3750

Overall domain bounding box (0 0 0) (0.1 0.05 0.001)
Mesh non-orthogonality Max: 12.4 average: 3.1
Max skewness = 0.61 OK.
Max aspect ratio = 4.2 OK.

Mesh OK.

End
"""

CHECK_BAD = """\
Mesh stats
    points:           100
    cells:            40

 ***Number of edges not in a single cell = 12
 ***Zero or negative cell volume detected.
Failed 2 mesh checks.

End
"""


def case_with(tmp_path, boundary=BOUNDARY, files=()):
    case = tmp_path / "mesh"
    (case / "constant" / "polyMesh").mkdir(parents=True)
    (case / "constant" / "polyMesh" / "boundary").write_text(boundary)
    (case / "constant" / "polyMesh" / "points").write_text("0()")
    for name in files:
        path = case / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n")
    return case


# -- the boundary file ---------------------------------------------------------


def test_the_patches_are_read_with_no_openfoam_environment(tmp_path):
    case = case_with(tmp_path)
    entries = mesh_look.boundary_entries(case)
    assert [e["name"] for e in entries] == ["inlet", "outlet", "walls", "frontAndBack"]
    assert [e["type"] for e in entries] == ["patch", "patch", "wall", "empty"]
    assert [e["nFaces"] for e in entries] == [20, 20, 400, 7500]
    assert entries[2]["inGroups"] == "1(wall)"


def test_a_case_with_no_mesh_reports_no_patches(tmp_path):
    assert mesh_look.boundary_entries(tmp_path) == []


def test_an_empty_patch_is_what_makes_it_a_plane_case(tmp_path):
    payload = mesh_look.look(case_with(tmp_path), None, check=False)
    assert payload["two_d"] is True
    plain = BOUNDARY.replace("empty", "wall")
    assert mesh_look.look(case_with(tmp_path / "b", plain), None, check=False)["two_d"] is False


# -- the checkMesh log ---------------------------------------------------------


def test_a_passing_log_gives_the_verdict_the_counts_and_the_metrics():
    parsed = mesh_look.parse_check(CHECK_OK)
    assert parsed["ok"] and parsed["verdict"] == "Mesh OK."
    assert parsed["counts"] == {"cells": 3750, "faces": 15000, "points": 7600}
    assert parsed["bounds"] == [0, 0, 0, 0.1, 0.05, 0.001]
    assert parsed["metrics"] == {"max_non_orthogonality": 12.4, "max_skewness": 0.61,
                                 "max_aspect_ratio": 4.2}


def test_a_failing_log_names_the_failing_checks():
    parsed = mesh_look.parse_check(CHECK_BAD)
    assert not parsed["ok"]
    assert parsed["verdict"] == "failed 2 mesh checks"
    assert any("Zero or negative cell volume" in line for line in parsed["failing"])


def test_a_log_that_is_neither_says_so_rather_than_passing():
    assert mesh_look.parse_check("")["verdict"] == "no verdict in the checkMesh log"
    assert not mesh_look.parse_check("")["ok"]
    fatal = mesh_look.parse_check("--> FOAM FATAL ERROR:\ncannot find file points")
    assert not fatal["ok"] and "cannot find file points" in fatal["verdict"]


# -- what would rebuild it -----------------------------------------------------


def test_the_rebuild_scripts_are_the_ones_that_are_there(tmp_path):
    case = case_with(tmp_path, files=("Allmesh", "build.py", "system/blockMeshDict"))
    (case / "shape.geo").write_text("// gmsh")
    found = mesh_look.build_files(case)
    assert found == ["Allmesh", "build.py", "system/blockMeshDict", "shape.geo"]
    assert mesh_look.build_files(case_with(tmp_path / "b")) == []


# -- the report ----------------------------------------------------------------


def test_a_case_with_no_mesh_says_that_and_nothing_else(tmp_path):
    payload = mesh_look.look(tmp_path, None, check=False)
    assert payload["polymesh"] is False
    assert "nothing has been meshed here yet" in mesh_look.report(payload)


def test_the_report_puts_the_patch_table_under_the_counts(tmp_path):
    case = case_with(tmp_path, files=("Allmesh",))
    payload = mesh_look.look(case, None, check=False)
    payload.update(mesh_look.parse_check(CHECK_OK)["counts"])
    payload["checkmesh"] = "Mesh OK."
    payload["metrics"] = {"max_skewness": 0.61}
    text = mesh_look.report(payload)
    assert "3,750 cells" in text and "one cell thick" in text
    assert "checkMesh: Mesh OK." in text
    assert "max skewness: 0.61" in text
    assert "inlet" in text and "frontAndBack" in text and "empty" in text
    assert "rebuilds with: Allmesh" in text


def test_the_report_says_when_nothing_would_rebuild_it(tmp_path):
    payload = mesh_look.look(case_with(tmp_path), None, check=False)
    assert "no Allmesh or build script" in mesh_look.report(payload)


def test_the_facts_are_the_ones_the_finish_check_reads(tmp_path):
    """`mesher/check.py` reads this dictionary; the two must agree on the key names."""
    from openreynolds.mesher.check import read

    payload = mesh_look.look(case_with(tmp_path, files=("Allmesh",)), None, check=False)
    # The fixture's polyMesh is three files, not a mesh, so the reader refuses it and
    # says so; the finish check reads that as a failure, which is the right answer for
    # a real case and the wrong one for this fixture.
    payload.pop("error", None)
    payload.update({"cells": 3750, "checkmesh_ok": True, "checkmesh": "Mesh OK.",
                    "render": "look.png"})
    check = read(payload, "mesh", "/work/s/mesh")
    assert check.ok, check.missing
    assert check.cells == 3750 and check.two_d
    assert [p["name"] for p in check.patches] == ["inlet", "outlet", "walls", "frontAndBack"]
