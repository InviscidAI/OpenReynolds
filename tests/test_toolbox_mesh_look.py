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


# -- where the picture goes ----------------------------------------------------
#
# Study 20260920-161908-c7ef, 16:26:34 UTC: `cd /work/<study> && python3
# /work/.toolbox/mesh_look.py mesh --out look.png` wrote `mesh/look.png` and printed
# `picture: look.png`; four seconds later `read_file /work/<study>/look.png` answered 404,
# and three turns went on finding the file. A relative `--out` was joined to the case,
# which no other toolbox script does (`render.py`, `results.py`, `showcase.py`,
# `geometry_view.py`, `animate.py` all read one against the working directory). The
# drawing needs pyvista and a mesh, so `main()` is driven here with `look` standing in.


def _drive(monkeypatch, argv, cwd, recorded):
    import sys

    def standing_in(case, out_png, check=True, region="", views=None):
        recorded["case"], recorded["out"], recorded["check"] = case, out_png, check
        recorded["region"], recorded["views"] = region, views
        payload = {"case": str(case), "polymesh": True, "patches": [], "zones": [],
                   "build": [], "cells": 0, "faces": 0, "points": 0, "bounds": [],
                   "two_d": False, "checkmesh": "", "checkmesh_ok": False, "metrics": {},
                   "render": ""}
        if out_png is not None:
            payload["render"] = str(out_png)
            payload["render_abs"] = str(Path(out_png).resolve())
        if views is not None:
            payload["views"] = [str(Path(views) / name) for name in mesh_look.VIEWS_3D]
        return payload

    monkeypatch.chdir(cwd)
    monkeypatch.setattr(mesh_look, "look", standing_in)
    monkeypatch.setattr(sys, "argv", ["mesh_look.py", *argv])
    with pytest.raises(SystemExit) as stop:
        mesh_look.main()
    return stop.value.code


def test_a_relative_out_is_where_you_run_it_from_not_in_the_case(tmp_path, monkeypatch, capsys):
    """The c7ef command, from the study directory: the picture lands beside the caller,
    and the report names the absolute path so there is nothing to work out."""
    study = tmp_path / "study"
    case_with(study)
    recorded: dict = {}

    code = _drive(monkeypatch, ["mesh", "--out", "look.png"], study, recorded)

    assert code == 0
    assert recorded["case"] == study / "mesh"
    assert recorded["out"] == study / "look.png", "beside the caller, not inside the case"
    printed = capsys.readouterr().out
    assert f"picture: {(study / 'look.png').resolve()}" in printed
    assert "picture: look.png" not in printed


def test_without_out_the_picture_is_look_png_in_the_case(tmp_path, monkeypatch, capsys):
    study = tmp_path / "study"
    case_with(study)
    recorded: dict = {}

    _drive(monkeypatch, ["mesh"], study, recorded)

    assert recorded["out"] == study / "mesh" / "look.png"
    assert f"picture: {(study / 'mesh' / 'look.png').resolve()}" in capsys.readouterr().out


def test_an_absolute_out_is_taken_as_it_is(tmp_path, monkeypatch, capsys):
    study = tmp_path / "study"
    case_with(study)
    elsewhere = tmp_path / "renders" / "duct.png"
    recorded: dict = {}

    _drive(monkeypatch, ["mesh", "--out", str(elsewhere)], study, recorded)

    assert recorded["out"] == elsewhere
    assert f"picture: {elsewhere.resolve()}" in capsys.readouterr().out


def test_the_json_goes_where_you_run_it_from_and_keeps_both_paths(tmp_path, monkeypatch):
    """`--json` follows the same rule. Inside it `render` stays relative to the case
    when the picture is in the case -- what `mesher/check.py` has always read and joins
    to the case's paths -- and `render_abs` is the absolute path either way."""
    import json

    study = tmp_path / "study"
    case_with(study)
    recorded: dict = {}

    _drive(monkeypatch, ["mesh", "--out", "mesh/look.png", "--json", "measured.json"], study, recorded)
    written = json.loads((study / "measured.json").read_text(encoding="utf-8"))
    assert written["render"] == "look.png", "relative to the case, as check.py reads it"
    assert written["render_abs"] == str((study / "mesh" / "look.png").resolve())

    _drive(monkeypatch, ["mesh", "--out", "look.png", "--json", "out/measured.json"], study, recorded)
    written = json.loads((study / "out" / "measured.json").read_text(encoding="utf-8"))
    assert written["render"] == str(study / "look.png"), "outside the case: absolute"
    assert written["render_abs"] == str((study / "look.png").resolve())


def test_the_desk_and_the_templates_run_from_the_case_and_are_unchanged(tmp_path, monkeypatch):
    """`mesher/check.py` runs `mesh_look.py . --out renders/mesh_look.png --json
    renders/mesh_look.json` with the case as the working directory, and the templates
    run `. --out look.png --json look.json` from the case. With `.` as the case, a path
    against the working directory and one against the case are the same path."""
    case = case_with(tmp_path)
    recorded: dict = {}

    _drive(monkeypatch, [".", "--out", "renders/mesh_look.png", "--json", "renders/mesh_look.json"],
           case, recorded)

    assert recorded["case"] == case
    assert recorded["out"] == case / "renders" / "mesh_look.png"
    assert (case / "renders" / "mesh_look.json").is_file()


def test_views_follow_the_same_path_rule_and_work_with_no_check(tmp_path, monkeypatch):
    """`--views DIR` is read against the working directory like `--out`, and in the JSON
    the three pictures are listed relative to the case when they are inside it -- the
    reviewer runs `mesh_look.py . --no-check --views renders/review` from the case and
    fetches what the list names."""
    import json

    case = case_with(tmp_path)
    recorded: dict = {}

    code = _drive(monkeypatch, [".", "--no-check", "--views", "renders/review",
                                "--json", "renders/look.json"], case, recorded)

    assert code == 0
    assert recorded["check"] is False
    assert recorded["views"] == case / "renders" / "review"
    written = json.loads((case / "renders" / "look.json").read_text(encoding="utf-8"))
    assert written["views"] == [str(Path("renders/review") / name) for name in mesh_look.VIEWS_3D]

    recorded.clear()
    _drive(monkeypatch, ["mesh"], tmp_path, recorded)
    assert recorded["views"] is None, "without --views nothing is drawn and nothing is listed"


def test_the_report_prints_the_absolute_path_when_it_has_one():
    payload = {"case": "/work/s/mesh", "polymesh": True, "patches": [], "zones": [],
               "build": ["Allmesh"], "render": "look.png",
               "render_abs": "/work/s/look.png"}
    assert "picture: /work/s/look.png" in mesh_look.report(payload)
    del payload["render_abs"]
    assert "picture: look.png" in mesh_look.report(payload), "an older payload still reads"


def test_from_cwd_is_the_convention(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert mesh_look.from_cwd(Path("look.png")) == tmp_path / "look.png"
    assert mesh_look.from_cwd(tmp_path / "abs.png") == tmp_path / "abs.png"


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


def test_a_patchs_normal_is_reported_out_of_the_fluid(tmp_path):
    """An outlet came back with the same normal as the inlet -- two opposing flat
    faces, which is geometrically impossible -- because `auto_orient_normals` is
    defined for a closed surface and a boundary patch is not one. The winding is now
    signed by asking the mesh which side the fluid is on."""
    payload = mesh_look.look(case_with(tmp_path), None, check=False)
    for patch in payload["patches"]:
        if patch.get("inward"):
            assert patch["inward"] == -1, "a measured normal must point out of the fluid"


# -- several views, for a reviewer who did not build it ------------------------
#
# `draw()` is one picture for whoever made the mesh. `draw_views` is three composite
# pictures for a reviewer with no context: 2D gets the patches, the cells and a 2x2
# zoom into the quadrants; 3D gets four isometric corners, six orthographic sides and
# four cuts. These run the real renderer on the hand-written box from
# `test_toolbox_layer_report`, so they skip where pyvista or its OpenFOAM reader is not
# there to run.


def _views_fixture(tmp_path, two_d: bool):
    pytest.importorskip("pyvista")
    from test_toolbox_layer_report import write_box

    if two_d:
        case = write_box(tmp_path / "flat", [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
                         [0, 0.2, 0.4, 0.7, 1.0], [0, 0.1], wall=("zmin", "zmax"))
        boundary = case / "constant" / "polyMesh" / "boundary"
        # The one-cell-thick faces become `empty`, which is what makes it a plane case.
        boundary.write_text(boundary.read_text().replace("type            wall;",
                                                         "type            empty;"))
    else:
        case = write_box(tmp_path / "box", [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
                         [0, 0.4, 1.0], [0, 0.2, 0.5, 1.0])
    return case


def _views_of(case, views_dir):
    payload = mesh_look.look(case, None, check=False, views=views_dir)
    if str(payload.get("error", "")).startswith("the mesh could not be opened"):
        pytest.skip(payload["error"])
    return payload


def test_a_3d_case_gets_iso_ortho_and_cuts(tmp_path):
    case = _views_fixture(tmp_path, two_d=False)
    views_dir = case / "renders" / "review"

    payload = _views_of(case, views_dir)

    assert payload["two_d"] is False
    assert not payload.get("error"), payload.get("error")
    assert [Path(p).name for p in payload["views"]] == list(mesh_look.VIEWS_3D)
    assert list(mesh_look.VIEWS_3D) == ["iso.png", "ortho.png", "cuts.png"]
    for written in payload["views"]:
        assert Path(written).is_file() and Path(written).parent == views_dir
        assert Path(written).stat().st_size > 1000, "a blank picture is not a picture"
    assert sorted(p.name for p in views_dir.iterdir()) == sorted(mesh_look.VIEWS_3D), (
        "exactly three pictures, no more")


def test_a_2d_case_gets_overview_cells_and_detail(tmp_path):
    case = _views_fixture(tmp_path, two_d=True)
    views_dir = case / "renders" / "review"

    payload = _views_of(case, views_dir)

    assert payload["two_d"] is True
    assert not payload.get("error"), payload.get("error")
    assert [Path(p).name for p in payload["views"]] == list(mesh_look.VIEWS_2D)
    assert list(mesh_look.VIEWS_2D) == ["overview.png", "cells.png", "detail.png"]
    for written in payload["views"]:
        assert Path(written).is_file() and Path(written).parent == views_dir
        assert Path(written).stat().st_size > 1000
    assert "views" in mesh_look.report(payload)


def test_without_views_the_payload_is_what_it_always_was(tmp_path):
    payload = mesh_look.look(case_with(tmp_path), None, check=False)
    assert "views" not in payload


def test_looking_at_a_case_for_its_views_leaves_no_time_directory_behind(tmp_path):
    """`open_mesh` makes an empty `0/` for the reader; looking must not change the case."""
    case = _views_fixture(tmp_path, two_d=False)
    _views_of(case, case / "renders" / "review")
    assert not (case / "0").exists()


# -- the zones -----------------------------------------------------------------
#
# Nothing here reported a cellZone or a faceZone until 2026-09-12, and everything that
# moves a mesh or freezes a rotor is named against one: `cyclicAMI` couples a pair of
# face zones, `MRFProperties` names a cell zone, an overset region is a cell zone. A
# dictionary that named a zone the mesh did not have failed at the first solver step and
# there was nothing in the toolbox that could have said so beforehand.

CELL_ZONES = """\
FoamFile { version 2.0; format ascii; class regIOobject; object cellZones; }

2
(
rotor
{
    type            cellZone;
    cellLabels      List<label>
2
(
0
1
)
;
}
empty_zone
{
    type            cellZone;
    cellLabels      List<label> 0();
}
)
"""

FACE_ZONES = """\
FoamFile { version 2.0; format ascii; class regIOobject; object faceZones; }

1
(
ami1
{
    type            faceZone;
    faceLabels      List<label> 3(4 5 6);
    flipMap         List<bool> 3(0 0 0);
}
)
"""


def test_a_mesh_with_no_zone_files_reports_none_rather_than_saying_nothing(tmp_path):
    """"No zones" and "zones were not looked for" are different facts, and only the
    first one means an AMI or MRF dictionary has nothing yet to name."""
    payload = mesh_look.look(case_with(tmp_path), None, check=False)
    assert payload["zones"] == []
    assert "zones: none" in mesh_look.report(payload)


def test_the_zones_are_read_with_no_openfoam_environment(tmp_path):
    case = case_with(tmp_path)
    poly = case / "constant" / "polyMesh"
    (poly / "cellZones").write_text(CELL_ZONES)
    (poly / "faceZones").write_text(FACE_ZONES)

    zones = mesh_look.zone_entries(case)
    assert [(z["name"], z["kind"], z["count"]) for z in zones] == [
        ("rotor", "cell", 2), ("empty_zone", "cell", 0), ("ami1", "face", 3),
    ]
    assert zones[0]["file"] == "constant/polyMesh/cellZones"


def test_the_foamfile_header_is_not_reported_as_a_zone(tmp_path):
    """The header is a named block immediately before the first zone's label list, so a
    span that is allowed to run past a closing brace reads `FoamFile` as a cell zone."""
    case = case_with(tmp_path)
    (case / "constant" / "polyMesh" / "cellZones").write_text(CELL_ZONES)
    assert "FoamFile" not in {z["name"] for z in mesh_look.zone_entries(case)}


def test_a_binary_zone_file_still_gives_its_names_and_counts(tmp_path):
    """The label count is written in plain text immediately before the binary block, so
    the name and the size survive a format this cannot read the labels out of -- and the
    extents say they were not measured rather than coming back as zero."""
    from test_toolbox_layer_report import write_box

    case = tmp_path / "box"
    write_box(case, [0, 1, 2, 3], [0, 1], [0, 1])
    binary = (CELL_ZONES.replace("format ascii;", "format binary;")
              .replace("2\n(\n0\n1\n)\n;", "2(\x00\x01\x02\x03)\n;"))
    (case / "constant" / "polyMesh" / "cellZones").write_text(binary)

    zones = mesh_look.zone_entries(case)
    assert [(z["name"], z["count"]) for z in zones] == [("rotor", 2), ("empty_zone", 0)]
    assert mesh_look._zone_labels(case, "cell") == {}

    # The rest of this polyMesh is ascii and readable, so the only thing between the
    # zone and its extents is the zone file's own format, and the message says that
    # rather than handing back a box of zeros.
    mesh_look.measure_cell_zones(case, zones)
    assert "bounds" not in zones[0]
    assert "not ascii" in zones[0]["unmeasured"]


def test_a_cell_zones_extent_comes_from_the_cells_it_names(tmp_path):
    """The number that says whether the rotating region actually surrounds the blade.
    Measured on a three-cell box whose cell centres are known by hand: naming cells 0
    and 1 of a 3 x 1 x 1 box of unit cells puts the centroid at x = 1."""
    from test_toolbox_layer_report import write_box

    case = tmp_path / "box"
    write_box(case, [0, 1, 2, 3], [0, 1], [0, 1])
    (case / "constant" / "polyMesh" / "cellZones").write_text(
        "FoamFile { version 2.0; format ascii; class regIOobject; object cellZones; }\n"
        "1\n(\nrotor\n{\n    type cellZone;\n    cellLabels List<label> 2(0 1);\n}\n)\n"
    )
    zones = mesh_look.zone_entries(case)
    mesh_look.measure_cell_zones(case, zones)

    rotor = zones[0]
    assert rotor["centre"] == pytest.approx([1.0, 0.5, 0.5])
    # The box spans the cell CENTRES, not the outer vertices: cells 0 and 1 are centred
    # at x = 0.5 and x = 1.5, and the report says which of the two it is showing.
    assert rotor["bounds"] == pytest.approx([0.5, 0.5, 0.5, 1.5, 0.5, 0.5])
    assert "cell centres span" in mesh_look.report(
        {"polymesh": True, "zones": zones, "case": str(case)})


def test_a_zone_naming_cells_the_mesh_does_not_have_is_said_rather_than_drawn(tmp_path):
    from test_toolbox_layer_report import write_box

    case = tmp_path / "box"
    write_box(case, [0, 1, 2], [0, 1], [0, 1])
    (case / "constant" / "polyMesh" / "cellZones").write_text(
        "FoamFile { version 2.0; format ascii; class regIOobject; object cellZones; }\n"
        "1\n(\nrotor\n{\n    type cellZone;\n    cellLabels List<label> 1(99);\n}\n)\n"
    )
    zones = mesh_look.zone_entries(case)
    mesh_look.measure_cell_zones(case, zones)
    assert "does not have" in zones[0]["unmeasured"]
    assert "bounds" not in zones[0]


def test_the_json_shape_grew_and_did_not_change(tmp_path):
    """`mesher/check.py` and `case_gen.py` both read this payload; zones were added
    beside the patch table rather than into it, so a reader that has never heard of a
    zone keeps working and inherits the facts the day it asks for them."""
    from openreynolds.mesher.check import read

    case = case_with(tmp_path, files=("Allmesh",))
    (case / "constant" / "polyMesh" / "cellZones").write_text(CELL_ZONES)
    payload = mesh_look.look(case, None, check=False)
    payload.pop("error", None)
    payload.update({"cells": 3750, "checkmesh_ok": True, "checkmesh": "Mesh OK.",
                    "render": "look.png"})
    assert payload["zones"], "the zones are in the payload"

    check = read(payload, "mesh", "/work/s/mesh")
    assert check.ok, check.missing
    assert [p["name"] for p in check.patches] == ["inlet", "outlet", "walls", "frontAndBack"]
