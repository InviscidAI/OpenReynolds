"""`paraview_export.py`: the study's camera and colour limits, in a file ParaView opens.

ParaView is not in the image and is not installed here either, so what is tested is the
script that is written and not a picture it drew: that it parses, that it carries the
numbers the study's own sidecars recorded, and that an absent sidecar becomes an honest
"ParaView will rescale" rather than an invented limit.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from openreynolds.toolbox import paraview_export as pvx


@pytest.fixture
def case(tmp_path: Path) -> Path:
    case = tmp_path / "pitzDaily"
    (case / "system").mkdir(parents=True)
    (case / "constant").mkdir()
    (case / "system" / "controlDict").write_text("writeInterval 1;\n", encoding="utf-8")
    return case


def difference_sidecar(case: Path, **over) -> Path:
    """A `render.py --diff` sidecar, in `diff_record`'s shape (`render.py:478-498`)."""
    record = {"field": "pressure", "runs": ["a", "b"], "times": [0.5, 0.5],
              "clim": [-12.5, 44.0], "cmap": "viridis",
              "diff_clim": [-3.0, 3.0], "diff_cmap": "coolwarm",
              "normal": "z", "view_vector": [0.0, 0.0, 1.0], "viewup": [0.0, 1.0, 0.0]}
    record.update(over)
    path = case / "renders" / "p_difference.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def frames_sidecar(case: Path, **over) -> Path:
    """An `animate.py` sidecar, in `build_sidecar`'s shape (`animate.py:405-444`)."""
    record = {"version": 1, "container": "gif", "fps": 10.0, "field": "temperature",
              "scalar": "T", "cmap": "inferno", "normal": "y", "clim": [290.0, 350.0],
              "frames": ["frame_0000.png"]}
    record.update(over)
    path = case / "renders" / "T_frames" / "frames.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


# -- what it refuses ---------------------------------------------------------------


def test_a_directory_that_is_not_a_case_is_refused_by_name(tmp_path, capsys):
    (tmp_path / "renders").mkdir()

    assert pvx.main([str(tmp_path)]) == 2
    assert "not an OpenFOAM case" in capsys.readouterr().out


def test_a_case_whose_dictionaries_are_only_half_written_is_still_a_case(tmp_path):
    """`system/` beside `constant/` is the fallback: a case being built has no
    controlDict yet and exporting it is not an error."""
    case = tmp_path / "half"
    (case / "system").mkdir(parents=True)
    (case / "constant").mkdir()

    assert pvx.is_case(case)


# -- the script it writes ----------------------------------------------------------


def test_the_script_it_writes_is_valid_python(case):
    report = pvx.export(case, case / "paraview")

    ast.parse(Path(report["script"]).read_text(encoding="utf-8"))


def test_the_script_is_named_after_the_case_and_says_how_to_run_it(case):
    report = pvx.export(case, case / "paraview")

    assert Path(report["script"]).name == "pitzDaily_paraview.py"
    assert report["open_with"] == ["paraview --script=pitzDaily_paraview.py",
                                  "pvpython pitzDaily_paraview.py"]


def test_it_runs_no_paraview_and_converts_nothing(case):
    """ParaView and pvpython are not in the image (`docker/workspace`), and an image
    change is a 1.2 GB rebuild. So the export is a file, not a conversion: no `.vtk`,
    no `.vtu`, no `.pvd`, and no new file inside the case itself."""
    before = {p.name for p in case.rglob("*")}

    pvx.export(case, case / "paraview")

    written = {p for p in case.rglob("*") if p.name not in before}
    assert all(p.suffix not in (".vtk", ".vtu", ".pvd", ".vtm") for p in written)
    assert {p.name for p in written if p.is_file()} == {"pitzDaily_paraview.py"}


def test_the_marker_is_created_by_the_script_and_not_by_the_export(case):
    """The mirror drops `.foam` as "written for a viewer that is not here"
    (`mirror.py:195-197`), and that policy is right: the marker is an empty file whose
    contents ParaView ignores. So the generated script makes its own on the machine
    that has ParaView, and the mirror is left alone."""
    report = pvx.export(case, case / "paraview")

    assert not list(case.glob("*.foam"))
    text = Path(report["script"]).read_text(encoding="utf-8")
    assert '".foam"' in text and "open(marker, \"w\").close()" in text


# -- the colours -------------------------------------------------------------------


def test_the_colour_limits_are_the_ones_the_study_drew_its_figures_on(case):
    """`render.py:487-491` writes one `clim` over both runs because "two panels each
    scaled to their own range are not a comparison". Those two numbers went into PNGs
    and nowhere a user's ParaView could read them, which is what this closes."""
    difference_sidecar(case)

    report = pvx.export(case, case / "paraview")

    assert report["clim"] == [-12.5, 44.0]
    assert "CLIM = [-12.5, 44.0]" in Path(report["script"]).read_text(encoding="utf-8")


def test_a_matplotlib_ramp_becomes_the_paraview_preset_that_is_the_same_ramp(case):
    difference_sidecar(case, cmap="coolwarm")

    assert pvx.export(case, case / "paraview")["preset"] == "Cool to Warm"


def test_a_colour_map_nobody_listed_leaves_the_preset_unset(case):
    """Guessing is how an export starts lying about what the figure looked like."""
    difference_sidecar(case, cmap="some-custom-ramp")

    report = pvx.export(case, case / "paraview")

    assert report["preset"] is None
    assert "PRESET = ''" in Path(report["script"]).read_text(encoding="utf-8")


def test_without_a_sidecar_it_says_paraview_will_rescale(case):
    report = pvx.export(case, case / "paraview")

    assert report["clim"] is None
    assert "rescale to the data range" in report["note"]
    assert "CLIM = None" in Path(report["script"]).read_text(encoding="utf-8")


def test_an_explicit_limit_beats_the_sidecar_and_says_where_it_came_from(case):
    difference_sidecar(case)

    report = pvx.export(case, case / "paraview", clim=[0.0, 1.0])

    assert report["clim"] == [0.0, 1.0]
    assert report["sources"]["clim"] == "--clim"


def test_the_newest_sidecar_wins_over_an_older_one(case):
    """A study renders several times. The last thing it drew is the figure its answer
    talks about."""
    difference_sidecar(case)
    newest = frames_sidecar(case)
    import os
    import time
    os.utime(newest, (time.time() + 10, time.time() + 10))

    report = pvx.export(case, case / "paraview")

    assert report["clim"] == [290.0, 350.0]
    assert report["sources"]["clim"] == str(newest)


def test_the_field_is_the_array_name_vtk_uses_rather_than_the_label(case):
    """`animate.py` records both: `field` is what a person called it and `scalar` is
    what the array is named in the data. ParaView colours by the second."""
    frames_sidecar(case)

    assert pvx.export(case, case / "paraview")["field"] == "T"


def test_an_explicit_field_beats_both(case):
    frames_sidecar(case)

    report = pvx.export(case, case / "paraview", field="U")

    assert report["field"] == "U" and report["sources"]["field"] == "--field"


# -- the camera --------------------------------------------------------------------


def test_a_slice_sidecar_pins_the_camera_down_its_own_normal(case):
    """`render.py`'s `aim()` points the camera down the cut normal, and the sidecar
    writes that pair of vectors down (`render.py:495-498`)."""
    difference_sidecar(case)

    report = pvx.export(case, case / "paraview")

    assert report["camera"] == "slice normal"
    text = Path(report["script"]).read_text(encoding="utf-8")
    assert "VIEW_VECTOR = [0.0, 0.0, 1.0]" in text
    assert "VIEWUP = [0.0, 1.0, 0.0]" in text
    assert "CameraParallelProjection = 1" in text, "a flat thing is looked at flatly"


def test_with_no_slice_sidecar_the_camera_is_showcase_s_own_three_numbers(case):
    """`showcase.py:717-729`: 235 degrees round the subject, 22 up, at 1.15 times the
    bounding-box diagonal. Repeated as constants here because the generated script has
    to run under ParaView's interpreter with none of this package importable."""
    report = pvx.export(case, case / "paraview")

    assert report["camera"] == "orbit"
    text = Path(report["script"]).read_text(encoding="utf-8")
    assert pvx.ORBIT == (235.0, 22.0, 1.15)
    assert "ORBIT_ANGLE = 235.0" in text
    assert "ORBIT_ELEVATION = 22.0" in text
    assert "ORBIT_RADIUS_FACTOR = 1.15" in text


def test_the_script_finds_its_own_case_when_it_sits_in_the_case(case):
    """The study directory gets copied, moved and renamed on the way home. A script
    that hard-codes `/work/<study>/case` is a script that works on the instance and
    nowhere else."""
    report = pvx.export(case, case)

    text = Path(report["script"]).read_text(encoding="utf-8")
    assert "os.path.dirname(os.path.abspath(__file__)) if True" in text


def test_an_out_directory_outside_the_case_keeps_the_absolute_path(case, tmp_path):
    report = pvx.export(case, tmp_path / "elsewhere")

    text = Path(report["script"]).read_text(encoding="utf-8")
    assert "if False" in text and repr(str(case)) in text


# -- the manifest ------------------------------------------------------------------


def test_the_manifest_names_every_sidecar_it_read(case):
    one = difference_sidecar(case)
    two = frames_sidecar(case)

    report = pvx.export(case, case / "paraview")

    assert sorted(report["sidecars_read"]) == sorted([str(one), str(two)])


def test_a_sidecar_that_will_not_parse_is_skipped_rather_than_fatal(case):
    (case / "renders").mkdir()
    (case / "renders" / "frames.json").write_text("{not json", encoding="utf-8")

    report = pvx.export(case, case / "paraview")

    assert report["sidecars_read"] == []


def test_the_command_line_prints_the_manifest_as_json(case, capsys):
    difference_sidecar(case)

    assert pvx.main([str(case)]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["clim"] == [-12.5, 44.0]
    assert Path(report["script"]).is_file()
