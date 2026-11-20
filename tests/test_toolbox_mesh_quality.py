"""`mesh_quality.py`: the bad cells located, and the converters that are not here.

Three scripts already report mesh quality numbers and none of them reports a place, so
what is pinned here is the location half: that a failing metric in a log is named as
failing, that the cells over a threshold are the ones the geometry says they are, that
each comes back with coordinates and a patch, and that a mesh with nothing wrong with
it says so instead of inventing something.

Everything runs on the hand-written ASCII polyMesh `test_toolbox_layer_report.py`
builds, with one plane of its points sheared sideways so the skewness is known before
the script is asked: a uniform shear of 2.0 on a 0.25 m cell row is a skewness of 8.0
and a non-orthogonality of 84.3 degrees, and both are arithmetic rather than a
tolerance. The clean box answers 0 to both, which is the half of the test that matters
most -- a selector that cannot say "none" cannot be trusted when it says "sixteen".

The conversion half is tested by lying to `shutil.which`, because the whole point of
the probe is that it believes the PATH it is on and not a table in a docstring. The
drawing is gated on a child-process OpenGL probe, the way
`tests/test_toolbox_mesh_look.py` and `tests/test_felix_results.py` gate theirs: on a
machine with no context the draw is a segfault that no `except` in this process sees.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load(name: str):
    """Import a toolbox script by path; the directory is data, not a package."""
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mesh_quality = load("mesh_quality")
locate = load("locate")


# -- the meshes ----------------------------------------------------------------


XS = [0.0, 0.25, 0.5, 0.75, 1.0]
YS = [0.0, 0.4, 1.0]
ZS = [0.0, 0.2, 0.5, 1.0]
SHEAR = 2.0
"""The sideways pull on the z = 0.2 plane, in metres. On a 0.25 m row of cells this
comes out as a skewness of exactly 8.0 -- over preflight's warn of 4.0 and under its
fail of 10.0, so the selection and the grading are tested at different points."""

POINT = re.compile(r"\(([-\d.eE+ ]+)\)")


def _box(root: Path, name: str, shear: float = 0.0) -> Path:
    """A structured hex box, optionally sheared along one interior plane.

    The shear moves every point at z = 0.2 in +x and nothing else, which leaves a
    valid mesh (no negative volumes, no inside-out cells) whose faces across that
    plane are skewed and non-orthogonal by a known amount. A single dragged point
    would be a more lifelike sliver and gives a much milder number, so the plane is
    used: the test is about the selection, not about how ugly a mesh can get.
    """
    from test_toolbox_layer_report import write_box

    case = write_box(root / name, XS, YS, ZS)
    if not shear:
        return case
    path = case / "constant" / "polyMesh" / "points"
    text = path.read_text()
    at = text.index("(", text.index("}"))
    coords = np.array([[float(v) for v in found.split()] for found in POINT.findall(text[at:])])
    coords[np.abs(coords[:, 2] - 0.2) < 1e-12, 0] += shear
    body = "\n".join("(%.10g %.10g %.10g)" % tuple(row) for row in coords)
    path.write_text(text[:at] + "(\n" + body + "\n)\n")
    return case


@pytest.fixture
def bent(tmp_path) -> Path:
    return _box(tmp_path, "bent", SHEAR)


@pytest.fixture
def clean(tmp_path) -> Path:
    return _box(tmp_path, "clean")


def mesh_of(case: Path) -> dict:
    mesh = locate.read_mesh(case / "constant" / "polyMesh")
    assert isinstance(mesh, dict), mesh
    return mesh


# -- a checkMesh log -----------------------------------------------------------


FAILING_LOG = """\
Mesh stats
    points:           98765
    faces:           287654
    internal faces:  278000
    cells:            94321
    boundary patches:     4
Checking geometry...
    Overall domain bounding box (0 0 0) (1 1 1)
    Max cell openness = 2.1e-16 OK.
    Max aspect ratio = 3.24 OK.
    Minimum face area = 1.2e-08. Maximum face area = 2.4e-06.  Face area magnitudes OK.
    Min volume = 1.1e-12. Max volume = 3.4e-09.  Total volume = 0.0003.
    Mesh non-orthogonality Max: 78.4 average: 11.2
   *Number of severely non-orthogonal (> 70 degrees) faces: 74.
 ***Max skewness = 12.41, 9 highly skew faces detected which may impair the quality of the results
    Minimum face determinant = 0.41
Failed 1 mesh checks.
"""


def write_log(case: Path, text: str = FAILING_LOG) -> None:
    """The log beside the case, with its `cells:` line set to the mesh's own count.

    A log that disagrees with the mesh about how many cells there are is a foreign or
    stale log and is now dropped rather than reported as this mesh's (#187), so a
    fixture log has to agree with the fixture mesh.
    """
    cells = mesh_of(case)["n_cells"]
    (case / "log.checkMesh").write_text(
        re.sub(r"cells:\s+\d+", f"cells:            {cells}", text))


def test_the_failing_metric_in_the_log_is_the_one_that_is_named(bent):
    """Which metric failed, by preflight's thresholds, on the log's own numbers.

    Skewness 12.41 is past the fail threshold of 10, non-orthogonality 78.4 is past
    the warn threshold of 70 and short of the fail at 85, and aspect ratio 3.24 is
    fine. All three have to come back separately: a report that said only "failed"
    would send somebody looking at the aspect ratio.
    """
    write_log(bent)
    found = mesh_quality.measure(bent)

    status = {row["metric"]: row["status"] for row in found["metrics"]}
    assert status["skewness"] == "fail"
    assert status["non_ortho"] == "warn"
    assert status["aspect_ratio"] == "ok"

    values = {row["metric"]: row["log_value"] for row in found["metrics"]}
    assert values["skewness"] == pytest.approx(12.41)
    assert values["non_ortho"] == pytest.approx(78.4)
    assert found["verdict"]["status"] == "fail"
    assert found["log"].endswith("log.checkMesh")
    # The log is the source of the numbers and never of the cells: no set was written
    # beside it, so the cells can only have come from the mesh.
    skew = next(row for row in found["metrics"] if row["metric"] == "skewness")
    assert skew["route"] == "computed"


def test_the_aspect_ratio_is_never_recomputed_here():
    """The one metric deliberately left to the log and the sets.

    `results.py` can ask VTK for an aspect ratio and it is not OpenFOAM's definition,
    so a cell selected on it would be red for a reason `checkMesh` never gave.
    """
    (key, _, _, _, recomputable) = next(
        row for row in mesh_quality.METRICS if row[0] == "aspect_ratio"
    )
    assert key == "aspect_ratio" and recomputable is False


def test_a_clean_mesh_has_no_skewness_and_no_non_orthogonality(clean):
    """The zero that makes every other number here mean something."""
    measured = mesh_quality.face_metrics(mesh_of(clean))

    assert measured["skewness"].max() < 1e-12
    assert measured["non_ortho"].max() < 1e-12


def test_cells_over_the_skewness_threshold_are_selected_and_counted(bent):
    """The shear is 2.0 across a 0.25 m row, which is a skewness of exactly 8.0.

    8.0 is over preflight's warn of 4.0, so the default threshold selects; the count
    is the cells on both sides of the sheared plane, which on this 4 x 2 x 3 box is
    the 8 below it and the 8 above.
    """
    mesh = mesh_of(bent)
    measured = mesh_quality.face_metrics(mesh)

    assert measured["skewness"].max() == pytest.approx(8.0, rel=1e-6)
    assert measured["non_ortho"].max() == pytest.approx(84.289, abs=1e-2)

    cells = mesh_quality.cells_above(mesh, measured["skewness"], 4.0)
    assert cells.size == 16
    assert mesh_quality.cells_above(mesh, measured["skewness"], 8.5).size == 0, (
        "nothing is over a threshold above the measured maximum"
    )

    # The same count through the command path, which is what a study would run.
    found = mesh_quality.measure(bent)
    skew = next(row for row in found["metrics"] if row["metric"] == "skewness")
    assert skew["route"] == "computed"
    assert skew["cells"] == 16
    assert skew["threshold"] == pytest.approx(4.0)
    assert skew["fraction_of_mesh"] == pytest.approx(16 / 24, abs=1e-4)


def test_a_higher_threshold_can_be_asked_for(bent):
    """`--skewness` is the flag, and above the measured maximum it selects nothing
    rather than falling back to the default and reporting cells anyway."""
    found = mesh_quality.measure(bent, thresholds={"skewness": 9.0})
    skew = next(row for row in found["metrics"] if row["metric"] == "skewness")

    assert skew["threshold"] == pytest.approx(9.0)
    assert skew["cells"] == 0


def test_the_worst_cells_come_back_with_coordinates_and_a_patch(bent):
    """A count is not a location. Each worst cell carries its centre, the patch it
    belongs to (or the nearest one when it is interior), and its distance to the
    domain bounding box, which are `locate.py`'s three measurements."""
    found = mesh_quality.measure(bent, worst=4)
    skew = next(row for row in found["metrics"] if row["metric"] == "skewness")

    assert len(skew["worst"]) == 4
    patches = set(mesh_of(bent)["boundary"])
    for record in skew["worst"]:
        assert len(record["centre"]) == 3
        assert all(isinstance(value, float) for value in record["centre"])
        assert record["patch"] in patches, record["patch"]
        assert record["distance_to_patch"] is not None
        assert "distance_to_bbox" in record
        assert record["value"] == pytest.approx(8.0, rel=1e-6)

    # The patch tally over the whole selection, not only the worst few: it is what
    # says whether the bad cells sit on one surface or run through the domain.
    assert skew["patches"], "no patch was attributed to any of the 16 cells"
    assert set(skew["patches"]) <= patches
    span = skew["box"]["span_fraction_of_domain"]
    assert len(span) == 3 and all(0.0 <= value <= 1.0 for value in span)


def test_an_interior_cell_is_placed_against_its_nearest_patch(tmp_path):
    """A cell with no boundary face of its own still sits near something, and which
    something is the part a coordinate triple does not carry. `heat.nearest_patch`
    answers it, and the answer is reused rather than written again.

    On a 4 x 3 x 3 box rather than the fixture's 4 x 2 x 3, because a box two cells
    thick in any direction has no interior cell at all and the case this is about
    would skip forever.
    """
    from test_toolbox_layer_report import write_box

    mesh = mesh_of(write_box(tmp_path / "wide", XS, [0.0, 0.3, 0.6, 1.0], ZS))
    interior = [
        cell for cell in range(mesh["n_cells"]) if not locate.cell_patches(mesh, cell)
    ]
    assert interior, "a 4 x 3 x 3 box has cells that touch no patch"
    placed = mesh_quality.place(mesh, np.array(interior[:1]), None, worst=1)
    record = placed["worst"][0]

    assert record["on_patches"] == []
    assert record["patch"] in mesh["boundary"]
    assert record["distance_to_patch"] > 0.0


def test_a_mesh_with_no_bad_cells_says_so_and_still_exits_zero(clean, capsys):
    """The commonest case, and the one a tool is most tempted to fill with something.

    Nothing on disk names a bad cell and the recomputed metrics are zero, so the
    report says there is nothing and there is no picture because there is nothing to
    put in one. Exit 0: a clean mesh is not an error.
    """
    code = mesh_quality.main(["report", str(clean)])
    text = capsys.readouterr().out

    assert code == 0
    assert all(row["cells"] == 0 for row in
               mesh_quality.measure(clean)["metrics"])
    assert "no cells were flagged by either route" in text
    assert "picture:" not in text


def test_the_report_runs_without_a_checkmesh_log_at_all(clean, capsys):
    """No log is a fact about the case, not a reason to stop: the mesh is still there
    and the recomputed metrics are still available off it."""
    code = mesh_quality.main(["report", str(clean)])

    assert code == 0
    assert "none here" in capsys.readouterr().out


def test_a_mesh_that_is_not_there_is_refused_rather_than_given_a_clean_bill(tmp_path,
                                                                            capsys):
    """#175: with no `constant/polyMesh` the report claimed "the metrics recomputed here
    are under the thresholds" two lines above "the mesh could not be read", and exited 0.
    No metric was recomputed, so a CI gate and a skimming reader both read "nothing
    wrong" off a mesh nobody measured."""
    case = tmp_path / "empty"
    (case / "system").mkdir(parents=True)

    code = mesh_quality.main(["report", str(case)])
    text = capsys.readouterr().out

    assert code != 0, "a mesh nobody could read is not a clean mesh"
    assert "could not be read" in text
    assert "no cells were flagged by either route" not in text
    assert "recomputed here are under the thresholds" not in text


def test_a_mistyped_region_is_refused_the_same_way(clean, capsys):
    """#175: the likelier way in. A region name that is not there reads the same as a
    clean single-region mesh, and that is how a typo passes a gate."""
    assert mesh_quality.main(["report", str(clean), "--region", "rotor"]) != 0
    assert "could not be read" in capsys.readouterr().out


def test_the_json_sidecar_carries_the_same_facts(bent, tmp_path):
    write_log(bent)
    out = tmp_path / "side" / "bad.json"

    assert mesh_quality.main(["report", str(bent), "--json", str(out)]) == 0
    payload = json.loads(out.read_text())

    assert payload["case"] == bent.as_posix()
    skew = next(row for row in payload["metrics"] if row["metric"] == "skewness")
    assert skew["cells"] == 16 and skew["route"] == "computed"
    assert "drawable" not in skew, "a numpy array of cell ids is not JSON for a reader"
    assert payload["worst"] in ("ok", "warn", "fail", "skipped")
    assert any(item["check"] == "cells:skewness" for item in payload["findings"])


# -- the set route -------------------------------------------------------------


def _write_set(case: Path, name: str, cls: str, labels: list[int]) -> Path:
    """A `faceSet` or `cellSet` the way checkMesh leaves one, by hand."""
    folder = case / "constant" / "polyMesh" / "sets"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    body = f"{len(labels)}\n(\n" + "\n".join(str(value) for value in labels) + "\n)\n"
    path.write_text(
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
        f"    class       {cls};\n    location    \"constant/polyMesh/sets\";\n"
        f"    object      {name};\n}}\n\n" + body
    )
    return path


def test_a_set_on_disk_wins_over_the_recomputed_metric_and_is_labelled(bent):
    """The set route is preferred because those cells are the ones that failed, by
    definition. Which route produced the answer is on the row either way: the two are
    different measurements and a reader who cannot tell them apart can check neither.
    """
    mesh = mesh_of(bent)
    faces = [0, 1, 2]
    _write_set(bent, "skewFaces", "faceSet", faces)
    found = mesh_quality.measure(bent)

    skew = next(row for row in found["metrics"] if row["metric"] == "skewness")
    assert skew["route"] == "set"
    assert skew["from_set"] == "skewFaces"
    expected = {int(mesh["owner"][face]) for face in faces}
    expected |= {int(mesh["neighbour"][face]) for face in faces if face < mesh["neighbour"].size}
    assert skew["cells"] == len(expected)
    assert [entry["set"] for entry in found["sets"]] == ["skewFaces"]
    assert found["sets"][0]["kind"] == "faceSet"


def test_a_set_whose_name_this_file_has_never_heard_of_is_still_read(bent):
    """The names an ESI v2512 checkMesh writes were not established here, so nothing
    is looked up by name: the directory is listed and the class is read off the file.
    A set that maps to no metric is reported under its own name rather than dropped."""
    _write_set(bent, "wrongOrientedFaces", "faceSet", [4, 5])
    found = mesh_quality.measure(bent)

    assert found["sets"][0]["set"] == "wrongOrientedFaces"
    assert found["sets"][0]["metric"] == ""
    row = next(row for row in found["metrics"] if row["metric"] == "wrongOrientedFaces")
    assert row["route"] == "set" and row["cells"] > 0


def test_a_cell_set_is_taken_as_cells_and_a_face_set_as_both_sides(bent):
    mesh = mesh_of(bent)
    cells = mesh_quality.cells_of_set(
        mesh, {"kind": "cellSet", "labels": np.array([3, 4, 4])}
    )
    assert cells.tolist() == [3, 4]

    both = mesh_quality.cells_of_set(mesh, {"kind": "faceSet", "labels": np.array([0])})
    assert both.tolist() == sorted({int(mesh["owner"][0]), int(mesh["neighbour"][0])})


def test_a_set_checkmesh_never_writes_is_not_reported_as_a_cell_it_flagged(clean, capsys):
    """#173: `region0` and `region1` are what a `topoSet` rotor/stator split leaves on an
    AMI mesh, and the mesh's own `log.checkMesh` said `Mesh OK` and wrote no sets at all.
    Every one of its 12,000 cells was reported as a cell checkMesh flagged and 6,000 were
    drawn red. The names checkMesh writes are in `CHECKMESH_SETS`, read out of
    `checkGeometry.C` and `checkTopology.C`; a set under any other name is listed with
    its count and its provenance named as unknown.
    """
    _write_set(clean, "region0", "cellSet", [0, 1, 2, 3])
    found = mesh_quality.measure(clean)
    print(mesh_quality.report(found))
    text = capsys.readouterr().out

    entry = next(row for row in found["sets"] if row["set"] == "region0")
    assert entry["provenance"] == "unknown"
    assert entry["count"] == 4, "it is still read: dropping it would hide a real failure"
    assert not any(row["metric"] == "region0" for row in found["metrics"])
    assert all(row["cells"] == 0 for row in found["metrics"])
    assert mesh_quality.worst_metric(found) is None, "nothing of unknown provenance is drawn"
    assert "the cells checkMesh itself flagged" not in text
    assert "unknown provenance" in text


def test_a_set_checkmesh_does_write_is_still_the_cells_it_flagged(bent):
    """The other half of #173: the gate is the name, so every name in checkMesh's own
    source still arrives by the set route and still wins over the recomputed metric."""
    _write_set(bent, "highAspectRatioCells", "cellSet", [0, 1])
    found = mesh_quality.measure(bent)

    entry = next(row for row in found["sets"] if row["set"] == "highAspectRatioCells")
    assert entry["provenance"] == "checkMesh" and entry["check"] == "aspect ratio"
    row = next(row for row in found["metrics"] if row["metric"] == "aspect_ratio")
    assert row["route"] == "set" and row["cells"] == 2


def test_set_provenance_is_an_exact_name_and_not_a_substring():
    assert mesh_quality.set_provenance("skewFaces") == "checkMesh"
    assert mesh_quality.set_provenance("nonOrthoFaces") == "checkMesh"
    assert mesh_quality.set_provenance("wrongOrientedFaces") == "checkMesh"
    assert mesh_quality.set_provenance("highAspectRatioCells") == "checkMesh"
    assert mesh_quality.set_provenance("mySkewCells") == "unknown"
    assert mesh_quality.set_provenance("region0") == "unknown", (
        "checkTopology writes region<n> for a disconnected region and topoSet writes it "
        "for a mesh zone; the file does not say which")


def test_a_log_describing_another_mesh_is_dropped_rather_than_reported_as_this_one(bent,
                                                                                  capsys):
    """#187: the tool printed its own correct cell count on one line and a foreign log's
    `aspect_ratio_max 11.8432` on the next, where pitzDaily's own log says 8.1407. The
    log's `cells:` line was one comparison away and went unread, and `find_log` picks up
    a stale `log.checkMesh` with no --log typed at all. Aspect ratio is never recomputed
    here, so a foreign value there has no second opinion to argue with it.
    """
    (bent / "log.checkMesh").write_text(FAILING_LOG)  # cells: 94321, not this mesh's
    found = mesh_quality.measure(bent)
    print(mesh_quality.report(found))
    text = capsys.readouterr().out

    assert found["log_describes_another_mesh"]["log_cells"] == 94321
    assert found["log_describes_another_mesh"]["mesh_cells"] == found["n_cells"]
    assert all(row["log_value"] is None for row in found["metrics"])
    assert not (found["verdict"] or {}).get("numbers")
    assert "11.8432" not in text and "12.41" not in text, "a foreign skewness either"
    assert "foreign or stale log" in text and "94,321 cells" in text


def test_a_log_whose_cell_count_matches_is_read_as_before(bent):
    """The check is a comparison and not a refusal of every log: a log that agrees with
    the mesh is the normal case and nothing about it changes."""
    write_log(bent)
    found = mesh_quality.measure(bent)

    assert "log_describes_another_mesh" not in found
    skew = next(row for row in found["metrics"] if row["metric"] == "skewness")
    assert skew["log_value"] == 12.41


def test_set_names_map_onto_metrics_by_substring():
    assert mesh_quality.metric_from_set("skewFaces") == "skewness"
    assert mesh_quality.metric_from_set("nonOrthoFaces") == "non_ortho"
    assert mesh_quality.metric_from_set("highAspectRatioCells") == "aspect_ratio"
    assert mesh_quality.metric_from_set("wrongOrientedFaces") == ""


# -- conversion ----------------------------------------------------------------


def test_the_probe_names_the_converter_it_could_not_find(monkeypatch):
    """A missing converter is reported by the name that was looked for.

    "fluent is not available" sends somebody to the documentation; "fluentMeshToFoam
    and fluent3DMeshToFoam are not on PATH" is something they can check in one call.
    """
    monkeypatch.setattr(mesh_quality.shutil, "which", lambda name: None)
    found = mesh_quality.probe()

    fluent = found["formats"]["fluent"]
    assert fluent["to_foam"]["reachable"] is False
    assert fluent["to_foam"]["missing"] == ["fluentMeshToFoam", "fluent3DMeshToFoam"]
    assert "fluent" in found["unreachable"]
    text = mesh_quality.report_formats(found)
    assert "fluentMeshToFoam" in text, "the report names the binary, not only the format"


def test_a_converter_that_is_there_is_reported_as_reachable(monkeypatch):
    """The other direction, so the probe is known to be looking rather than guessing."""
    monkeypatch.setattr(
        mesh_quality.shutil, "which",
        lambda name: "/usr/bin/" + name if name in ("gmshToFoam", "foamToVTK") else None,
    )
    found = mesh_quality.probe()

    assert found["formats"]["gmsh"]["to_foam"]["reachable"] is True
    assert found["formats"]["vtk"]["from_foam"]["found"][0]["name"] == "foamToVTK"
    assert "gmsh" in found["reachable"] and "vtk" in found["reachable"]
    # gmsh has no converter OUT of OpenFOAM, which is not the same as a missing one.
    assert found["formats"]["gmsh"]["from_foam"]["looked_for"] == []


def test_cgns_has_no_converter_in_either_direction_whatever_is_on_path(monkeypatch):
    """CGNS is the honest part of the table. There is no converter in the OpenFOAM
    distribution in either direction, so no PATH makes it reachable, and the probe
    says that rather than printing an empty row and leaving it to be read as a
    missing install."""
    monkeypatch.setattr(mesh_quality.shutil, "which", lambda name: "/usr/bin/" + name)
    found = mesh_quality.probe()

    cgns = found["formats"]["cgns"]
    assert cgns["to_foam"]["looked_for"] == [] and cgns["from_foam"]["looked_for"] == []
    assert cgns["to_foam"]["reachable"] is False and cgns["from_foam"]["reachable"] is False
    assert "cgns" in found["unreachable"]
    assert "cgns" in cgns["note"].lower() or "CGNS" in cgns["note"]
    assert "no converter exists" in mesh_quality.report_formats(found)


def test_converting_from_a_format_with_no_converter_exits_zero_and_says_why(tmp_path, capsys):
    source = tmp_path / "wing.cgns"
    source.write_text("not really a cgns file")

    code = mesh_quality.main(
        ["convert", str(source), "--from", "cgns", "--to", "openfoam"]
    )
    text = capsys.readouterr().out

    assert code == 0
    assert "no OpenFOAM converter exists for cgns" in text


def test_converting_without_the_binary_names_what_it_looked_for(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mesh_quality.shutil, "which", lambda name: None)
    source = tmp_path / "duct.msh"
    source.write_text("not really a fluent mesh")

    code = mesh_quality.main(
        ["convert", str(source), "--from", "fluent", "--to", "openfoam",
         "--case", str(tmp_path / "case")]
    )
    text = capsys.readouterr().out

    assert code == 0
    assert "fluentMeshToFoam" in text and "not reachable" in text


def test_a_conversion_that_does_not_go_through_openfoam_is_refused(tmp_path, capsys):
    """OpenFOAM is the hub here. Fluent to VTK exists as two calls and not as one, and
    saying so is better than running a converter that would write the wrong thing."""
    source = tmp_path / "duct.msh"
    source.write_text("x")

    assert mesh_quality.main(
        ["convert", str(source), "--from", "fluent", "--to", "vtk"]
    ) == 0
    assert "OpenFOAM is the hub" in capsys.readouterr().out


def test_the_probe_answers_for_every_format_it_lists(monkeypatch):
    """No format is left without an answer in either direction, which is what makes
    the table readable as a whole rather than as the rows that happened to be filled."""
    monkeypatch.setattr(mesh_quality.shutil, "which", lambda name: None)
    found = mesh_quality.probe()

    assert set(found["formats"]) == set(mesh_quality.CONVERTERS)
    for entry in found["formats"].values():
        for direction in ("to_foam", "from_foam"):
            assert isinstance(entry[direction]["reachable"], bool)


def test_the_probe_against_the_real_openfoam_on_this_instance():
    """Inside the workspace image, where the answer is the real one.

    Skipped anywhere OpenFOAM is not on PATH, which is every laptop and every matrix
    runner. What it pins is the claim this script makes about the image: foamToVTK is
    there (`render.py` falls back to it and ENVIRONMENT.md names it), and CGNS is not
    reachable however the probe comes out.
    """
    if mesh_quality.shutil.which("checkMesh") is None:
        pytest.skip("no OpenFOAM on PATH here; the workspace image has it")
    found = mesh_quality.probe()

    assert found["formats"]["vtk"]["from_foam"]["reachable"], "foamToVTK is not on PATH"
    assert "cgns" in found["unreachable"]
    # Printed so the image job's log carries the real table rather than only a verdict.
    print(mesh_quality.report_formats(found))


# -- the picture ---------------------------------------------------------------


_PROBE = (
    "import pyvista as pv\n"
    "pv.OFF_SCREEN = True\n"
    "p = pv.Plotter(off_screen=True, window_size=(64, 64))\n"
    "p.add_mesh(pv.Cube())\n"
    "p.screenshot(return_img=True)\n"
    "p.close()\n"
)


@functools.lru_cache(maxsize=1)
def _can_draw() -> bool:
    """Whether VTK draws off-screen here, asked in a CHILD process.

    Without an OpenGL context the draw is a segmentation fault inside VTK, which no
    `try` in this process catches and which takes the whole pytest run with it. The
    workspace image has OSMesa and draws; the advisory `image` CI job is where this
    actually runs.
    """
    try:
        done = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, timeout=120)
        return done.returncode == 0
    except Exception:  # noqa: BLE001 - a probe that will not run is a no
        return False


def needs_a_renderer():
    pytest.importorskip("pyvista", reason="the drawing half runs in the workspace image")
    if not _can_draw():
        pytest.skip("no off-screen OpenGL here; the workspace image draws")


def test_the_bad_cell_picture_is_not_blank(bent, tmp_path):
    """`docker/workspace/check.sh`'s own check: a render with no spread in its pixels
    is a blank image, and a blank PNG written without complaint is the failure mode
    that looks like success."""
    needs_a_renderer()
    imageio = pytest.importorskip("imageio.v2")

    found = mesh_quality.measure(bent)
    out = tmp_path / "bad.png"
    drawn = mesh_quality.draw(found, out)

    assert out.is_file() and out.stat().st_size > 1000
    assert drawn["drawn_route"] == "computed"
    assert drawn["drawn_cells"] == 16
    image = np.asarray(imageio.imread(out), dtype=float)
    assert image.std() > 0, "the render is blank"


def test_the_picture_comes_out_of_the_command_line_too(bent, tmp_path, capsys):
    needs_a_renderer()
    pytest.importorskip("imageio.v2")

    code = mesh_quality.main(["report", str(bent), "--out", str(tmp_path / "shots")])
    text = capsys.readouterr().out

    assert code == 0
    assert (tmp_path / "shots" / "mesh_quality.png").is_file()
    assert (tmp_path / "shots" / "mesh_quality.json").is_file()
    assert "picture:" in text


def test_nothing_is_drawn_for_a_mesh_with_nothing_wrong_with_it(clean, tmp_path, capsys):
    """And the absence is reported rather than left as a missing file. This needs no
    renderer: there is nothing to draw, which is the point."""
    code = mesh_quality.main(["report", str(clean), "--png", str(tmp_path / "none.png")])
    text = capsys.readouterr().out

    assert code == 0
    assert not (tmp_path / "none.png").exists()
    assert "nothing was drawn" in text


def test_the_faces_drawn_are_the_mesh_own_faces(bent):
    """The picture is built from `constant/polyMesh` rather than read back through
    `pyvista.OpenFOAMReader`, whose default polyhedron decomposition renumbers cells.
    A red cell at the wrong index is a picture that lies, so the connectivity is
    checked here where it needs no renderer at all."""
    mesh = mesh_of(bent)
    faces = mesh_quality.faces_of_cells(mesh, np.array([0]))
    assert faces.size == 6, "a hexahedron has six faces"

    owner, neighbour = mesh["owner"], mesh["neighbour"]
    for face in faces:
        assert owner[face] == 0 or (face < neighbour.size and neighbour[face] == 0)

    ring = mesh_quality.one_ring(mesh, np.array([0]))
    assert 0 in ring.tolist() and ring.size > 1, "the close-up needs neighbours in it"
    assert mesh_quality.boundary_faces(mesh).size == int(
        sum(entry["nFaces"] for entry in mesh["boundary"].values())
    )
