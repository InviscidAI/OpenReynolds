"""`progress_report.py`: the arithmetic behind "what is happening?".

The script itself renders nothing and needs nothing from the container, so all of it is
testable here -- which is the point, because the ETA is the one number in the toolbox
that is an estimate rather than a reading and a wrong one is worse than none.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load(name: str):
    """Import a toolbox script by path; the directory is data, not a package."""
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def progress_report():
    return load("progress_report")


@pytest.fixture
def state():
    return load("study_state")


# -- logs to parse -----------------------------------------------------------------


def transient_log(steps: int = 40, dt: float = 0.05, wall_per_step: float = 2.0) -> str:
    """A pimpleFoam-shaped log: fractional times, falling residuals, a Courant line."""
    lines = []
    for step in range(1, steps + 1):
        residual = 1.0e-2 * (0.9 ** step)
        lines += [
            f"Time = {step * dt:g}",
            "",
            f"smoothSolver:  Solving for Ux, Initial residual = {residual:.6e}, "
            f"Final residual = {residual / 100:.6e}, No Iterations 3",
            f"GAMG:  Solving for p, Initial residual = {residual * 9:.6e}, "
            f"Final residual = {residual / 10:.6e}, No Iterations 7",
            "time step continuity errors : sum local = 1.5e-06, global = -2.0e-09, cumulative = -3.0e-09",
            f"Courant Number mean: 0.12 max: {0.8 + step * 0.001:g}",
            f"ExecutionTime = {step * wall_per_step:g} s  ClockTime = {step * wall_per_step:g} s",
            "",
        ]
    return "\n".join(lines) + "\n"


def steady_log(iterations: int = 40, wall_per_iteration: float = 0.5) -> str:
    """A simpleFoam-shaped log: Time is the iteration number, one apart."""
    lines = []
    for step in range(1, iterations + 1):
        residual = 1.0e-1 * (0.8 ** step)
        lines += [
            f"Time = {step}",
            "",
            f"smoothSolver:  Solving for Ux, Initial residual = {residual:.6e}, "
            f"Final residual = {residual / 50:.6e}, No Iterations 4",
            f"ExecutionTime = {step * wall_per_iteration:g} s  ClockTime = {step} s",
            "",
        ]
    return "\n".join(lines) + "\n"


CONTROL_DICT = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}

application     pimpleFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         3.1;

deltaT          2e-4;
adjustTimeStep  yes;
maxCo           1.5;

writeControl    adjustableRunTime;
writeInterval   0.25;

functions
{
    avg
    {
        type            fieldAverage;
        writeControl    writeTime;
        writeInterval   1;
        fields
        (
            U  { mean on; prime2Mean off; base time; }
        );
    }
}
"""


def make_case(tmp_path: Path, control: str = CONTROL_DICT, log: str = "") -> Path:
    """A study directory with one case in it, laid out the way `study_state` expects:
    the `.reynolds/` marker sits on the study and the case lives beneath it."""
    case = tmp_path / "study" / "case"
    (case / "system").mkdir(parents=True)
    (case.parent / ".reynolds").mkdir(exist_ok=True)
    (case / "system" / "controlDict").write_text(control, encoding="utf-8")
    if log:
        (case / "log.pimpleFoam").write_text(log, encoding="utf-8")
    return case


# -- controlDict -------------------------------------------------------------------


def test_the_target_time_comes_out_of_a_real_control_dict(progress_report):
    control = progress_report.parse_control_dict(CONTROL_DICT)

    assert control["end_time"] == pytest.approx(3.1)
    assert control["start_time"] == pytest.approx(0.0)
    assert control["delta_t"] == pytest.approx(2e-4)
    assert control["application"] == "pimpleFoam"
    assert control["adjust_time_step"] is True


def test_a_function_objects_write_interval_is_not_the_cases(progress_report):
    """The `avg` block sets writeInterval 1; the run's is 0.25 and must stay so."""
    control = progress_report.parse_control_dict(CONTROL_DICT)

    assert control["write_interval"] == pytest.approx(0.25)
    assert control["write_control"] == "adjustableRunTime"


def test_stop_at_something_other_than_end_time_withdraws_the_target(progress_report):
    control = progress_report.parse_control_dict(
        CONTROL_DICT.replace("stopAt          endTime;", "stopAt          writeNow;")
    )

    assert "end_time" not in control
    assert control["end_time_declared"] == pytest.approx(3.1)
    assert "writeNow" in control["end_time_note"]


def test_comments_do_not_become_entries(progress_report):
    control = progress_report.parse_control_dict(
        "// endTime 999;\n/* endTime 888; */\nendTime 4;\n"
    )
    assert control["end_time"] == pytest.approx(4.0)


def test_a_missing_control_dict_is_an_empty_dict_not_an_error(tmp_path, progress_report):
    assert progress_report.read_control_dict(tmp_path) == {}


# -- pace and ETA ------------------------------------------------------------------


def test_pace_pairs_each_time_with_its_execution_time(progress_report):
    points = progress_report.pace_points(transient_log(steps=5, dt=0.1, wall_per_step=3.0))

    assert len(points) == 5
    assert points[0] == pytest.approx((0.1, 3.0))
    assert points[-1] == pytest.approx((0.5, 15.0))


def test_a_step_still_being_written_is_not_counted(progress_report):
    """The solve is mid-step: `Time =` has printed, `ExecutionTime =` has not."""
    text = transient_log(steps=3, dt=0.1, wall_per_step=3.0) + "Time = 0.4\n"
    points = progress_report.pace_points(text)

    assert len(points) == 3
    assert points[-1][0] == pytest.approx(0.3)


def test_transient_eta_is_wall_clock_per_simulated_second(progress_report):
    # 2 s of wall per 0.05 s of simulation = 40 s per simulated second.
    points = progress_report.pace_points(transient_log(steps=40, dt=0.05, wall_per_step=2.0))

    eta = progress_report.estimate_eta(points, target=4.0, steady=False)

    assert eta["unit"] == "simulated second"
    assert eta["rate_seconds_per_unit"] == pytest.approx(40.0)
    assert eta["units_remaining"] == pytest.approx(2.0)  # reached 2.0 of 4.0
    assert eta["seconds_remaining"] == pytest.approx(80.0)
    assert eta["confidence"] == "high"


def test_steady_eta_is_wall_clock_per_iteration(progress_report):
    points = progress_report.pace_points(steady_log(iterations=40, wall_per_iteration=0.5))
    times = [time for time, _exec in points]

    assert progress_report.looks_steady(times) is True

    eta = progress_report.estimate_eta(points, target=1000.0, steady=True)

    assert eta["unit"] == "iteration"
    assert eta["rate_seconds_per_unit"] == pytest.approx(0.5)
    assert eta["units_remaining"] == pytest.approx(960.0)
    assert eta["seconds_remaining"] == pytest.approx(480.0)


def test_a_transient_is_not_mistaken_for_a_steady_run(progress_report):
    points = progress_report.pace_points(transient_log(steps=10, dt=0.05))
    assert progress_report.looks_steady([t for t, _ in points]) is False


def test_the_window_bounds_how_far_back_the_rate_looks(progress_report):
    """The solve slowed down; a rate over the whole run would report the old pace."""
    fast = [(step * 1.0, step * 1.0) for step in range(1, 21)]
    slow = [(20.0 + step, 20.0 + step * 10.0) for step in range(1, 11)]

    eta = progress_report.estimate_eta(fast + slow, target=40.0, steady=True, window=10)

    assert eta["window_points"] == 10
    assert eta["rate_seconds_per_unit"] == pytest.approx(10.0)
    assert eta["seconds_remaining"] == pytest.approx(100.0)


def test_an_erratic_pace_lowers_the_confidence(progress_report):
    erratic = [(0.0, 0.0), (1.0, 1.0), (2.0, 40.0), (3.0, 41.0), (4.0, 90.0), (5.0, 91.0)]

    eta = progress_report.estimate_eta(erratic, target=10.0, steady=True)

    assert eta["confidence"] == "low"
    assert "varied" in eta["why"]


def test_an_adjustable_time_step_is_never_high_confidence(progress_report):
    points = progress_report.pace_points(transient_log(steps=40, dt=0.05, wall_per_step=2.0))

    steady_pace = progress_report.estimate_eta(points, target=4.0, adjustable_dt=False)
    adjustable = progress_report.estimate_eta(points, target=4.0, adjustable_dt=True)

    assert steady_pace["confidence"] == "high"
    assert adjustable["confidence"] == "medium"
    assert "adjustTimeStep" in adjustable["why"]


def test_no_end_time_still_reports_a_rate(progress_report):
    points = progress_report.pace_points(transient_log(steps=20, dt=0.05, wall_per_step=2.0))

    eta = progress_report.estimate_eta(points, target=None)

    assert eta["rate_seconds_per_unit"] == pytest.approx(40.0)
    assert eta["seconds_remaining"] is None
    assert "no end is known" in eta["why"]


def test_a_target_already_reached_counts_down_to_zero(progress_report):
    points = progress_report.pace_points(transient_log(steps=20, dt=0.05, wall_per_step=2.0))

    eta = progress_report.estimate_eta(points, target=0.5)

    assert eta["seconds_remaining"] == pytest.approx(0.0)
    assert "already reached" in eta["why"]


def test_one_step_is_not_enough_for_an_eta(progress_report):
    eta = progress_report.estimate_eta([(0.1, 3.0)], target=1.0)

    assert eta["basis"] is None
    assert eta["seconds_remaining"] is None
    assert "fewer than two" in eta["why"]


def test_a_stalled_clock_does_not_produce_an_infinite_eta(progress_report):
    eta = progress_report.estimate_eta([(1.0, 5.0), (1.0, 9.0)], target=10.0)

    assert eta["basis"] is None
    assert "did not advance" in eta["why"]


# -- residual trend ----------------------------------------------------------------


def test_a_falling_series_reads_as_falling(progress_report):
    values = [1.0 * (0.5 ** step) for step in range(40)]

    row = progress_report.trend_of(values, window=10)

    assert row["trend"] == "falling"
    assert row["recent"] < row["earlier"]
    assert row["decades"] < 0


def test_a_rising_series_reads_as_rising(progress_report):
    values = [1e-6 * (1.5 ** step) for step in range(40)]
    assert progress_report.trend_of(values, window=10)["trend"] == "rising"


def test_noise_around_one_level_reads_as_flat(progress_report):
    values = [1e-4, 1.05e-4, 0.95e-4, 1.02e-4] * 10
    row = progress_report.trend_of(values, window=10)

    assert row["trend"] == "flat"
    assert abs(row["decades"]) < progress_report.FLAT_DECADES


def test_one_spike_does_not_flip_a_falling_trend(progress_report):
    """The mean is geometric for exactly this reason: residual traces spike."""
    values = [1.0 * (0.5 ** step) for step in range(40)]
    values[-3] = 1.0

    assert progress_report.trend_of(values, window=10)["trend"] == "falling"


def test_too_short_a_series_says_so_rather_than_guessing(progress_report):
    row = progress_report.trend_of([1e-3, 2e-3], window=10)

    assert row["trend"] == "unknown"
    assert "fewer than four" in row["why"]


def test_fields_moving_opposite_ways_read_as_mixed(progress_report):
    residuals = {
        "Ux": [(step, 1.0 * (0.5 ** step)) for step in range(40)],
        "p": [(step, 1e-6 * (1.5 ** step)) for step in range(40)],
    }

    summary = progress_report.residual_trend(residuals, window=10)

    assert summary["overall"] == "mixed"
    assert summary["fields"]["Ux"]["trend"] == "falling"
    assert summary["fields"]["p"]["trend"] == "rising"


def test_all_falling_reads_as_falling_overall(progress_report):
    residuals = {
        name: [(step, 1.0 * (0.5 ** step)) for step in range(40)] for name in ("Ux", "p")
    }
    assert progress_report.residual_trend(residuals, window=10)["overall"] == "falling"


def test_no_residuals_at_all_is_unknown_not_flat(progress_report):
    assert progress_report.residual_trend({})["overall"] == "unknown"


def test_the_window_reported_is_the_window_actually_compared(progress_report):
    """Twelve steps cannot be split into two windows of ten, and the rendered line
    names the number -- so the number has to be the one the comparison used."""
    residuals = {"Ux": [(step, 10.0 ** -(step / 10.0)) for step in range(12)]}

    summary = progress_report.residual_trend(residuals, window=10)

    assert summary["window_requested"] == 10
    assert summary["window"] == 6 == summary["fields"]["Ux"]["window"]

    text = progress_report.render_text(
        {
            "study_name": "s", "case_name": "c", "generated_at": "now",
            "phase": {"current": "solve", "status": "running", "note": "", "next_unsettled": ""},
            "solve": {}, "residuals": summary, "eta": progress_report.no_eta("x"),
            "frames": [], "previews": [], "notes": [],
        }
    )
    assert "last 6 steps against the 6 before them" in text


def test_a_window_too_small_to_compare_says_that_and_not_that_the_series_is_short(
    progress_report,
):
    """`--window 1` on a forty-step series is a bad window, not a short log."""
    values = [10.0 ** -(step / 10.0) for step in range(40)]

    row = progress_report.trend_of(values, window=1)

    assert row["trend"] == "unknown"
    assert "too short to compare" in row["why"]
    assert "fewer than four" not in row["why"]


def test_the_trend_offers_no_verdict_on_convergence(tmp_path, progress_report):
    """Falling, flat and rising are readings. Whether they are good news is not."""
    case = make_case(tmp_path, log=transient_log())
    text = progress_report.render_text(progress_report.collect(case)).lower()

    for verdict in ("converged", "not converged", "diverging", "looks good", "healthy", "you should"):
        assert verdict not in text


# -- frames ------------------------------------------------------------------------


def frames_dir(case: Path, name: str, pngs: int) -> Path:
    directory = case.parent / name
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(pngs):
        (directory / f"frame_{index:04d}.png").write_bytes(b"\x89PNG")
    return directory


def test_frames_are_counted_against_the_write_times(tmp_path, progress_report):
    case = make_case(tmp_path)
    for step in range(6):
        (case / f"{step * 0.5:g}").mkdir()
    frames_dir(case, "case_frames", 3)

    expected = len(progress_report.write_times(case))
    row = progress_report.frame_progress(case.parent / "case_frames", expected)

    assert expected == 6
    assert row["count"] == 3
    assert row["expected"] == 6
    assert row["expected_from"] == "write times"
    assert row["fraction"] == pytest.approx(0.5)


def test_a_sidecar_beats_the_write_time_count(tmp_path, progress_report):
    case = make_case(tmp_path)
    directory = frames_dir(case, "wake_frames", 4)
    (directory / "frames.json").write_text(json.dumps({"expected": 16}), encoding="utf-8")

    row = progress_report.frame_progress(directory, fallback_expected=99)

    assert row["expected"] == 16
    assert row["expected_from"] == "sidecar"
    assert row["fraction"] == pytest.approx(0.25)


def test_a_sidecar_listing_the_frames_is_read_by_length(tmp_path, progress_report):
    case = make_case(tmp_path)
    directory = frames_dir(case, "wake_frames", 1)
    (directory / "frames.json").write_text(json.dumps({"frames": ["a", "b", "c"]}), encoding="utf-8")

    assert progress_report.frame_progress(directory)["expected"] == 3


def test_an_unreadable_sidecar_falls_back_rather_than_raising(tmp_path, progress_report):
    case = make_case(tmp_path)
    directory = frames_dir(case, "wake_frames", 2)
    (directory / "frames.json").write_text("{not json", encoding="utf-8")

    row = progress_report.frame_progress(directory, fallback_expected=8)

    assert row["expected"] == 8
    assert row["expected_from"] == "write times"


def test_with_nothing_expected_the_count_is_still_reported(tmp_path, progress_report):
    case = make_case(tmp_path)
    directory = frames_dir(case, "wake_frames", 2)

    row = progress_report.frame_progress(directory)

    assert row["count"] == 2
    assert row["expected"] is None
    assert row["expected_from"] == "unknown"
    assert row["fraction"] is None


def test_the_walk_finds_frame_dirs_without_descending_into_time_dirs(tmp_path, progress_report):
    case = make_case(tmp_path)
    for step in range(3):
        (case / f"{step}").mkdir()
    (case / "0" / "deep_frames").mkdir()  # a numeric dir is not walked into
    (case / "processor0").mkdir()
    (case / "processor0" / "other_frames").mkdir()
    frames_dir(case, "case_frames", 1)
    (case / "inner_frames").mkdir()

    found = {path.name for path in progress_report.find_frame_dirs(case.parent)}

    assert "case_frames" in found
    assert "inner_frames" in found
    assert "deep_frames" not in found
    assert "other_frames" not in found


# -- the whole report --------------------------------------------------------------


def test_a_running_transient_reports_where_it_is(tmp_path, progress_report):
    case = make_case(tmp_path, log=transient_log(steps=40, dt=0.05, wall_per_step=2.0))

    report = progress_report.collect(case)
    solve = report["solve"]

    assert Path(solve["log"]).name == "log.pimpleFoam"
    assert solve["steps"] == 40
    assert solve["time"] == pytest.approx(2.0)
    assert solve["end_time"] == pytest.approx(3.1)
    assert solve["fraction"] == pytest.approx(2.0 / 3.1)
    assert solve["steady"] is False
    assert solve["courant"]["mean"] == pytest.approx(0.12)
    assert report["residuals"]["overall"] == "falling"
    assert report["eta"]["seconds_remaining"] == pytest.approx((3.1 - 2.0) * 40.0)
    assert report["eta"]["confidence"] == "medium"  # adjustTimeStep is on


def test_the_text_and_the_json_are_the_same_facts(tmp_path, progress_report):
    case = make_case(tmp_path, log=transient_log(steps=20))

    report = progress_report.collect(case)
    text = progress_report.render_text(report)

    assert "pimpleFoam" in text
    assert "endTime 3.1" in text
    assert "falling" in text
    # The JSON mode is this same dict, so it has to survive a round trip.
    assert json.loads(json.dumps(report, default=str))["solve"]["steps"] == 20


def test_prose_wraps_but_paths_are_never_broken(tmp_path, progress_report):
    """The confidence sentence is the long one; a wrapped path would be unusable."""
    case = make_case(tmp_path, log=transient_log(steps=40))
    report = progress_report.collect(case)
    report["notes"].append("a note long enough to need wrapping " * 4)

    for line in progress_report.render_text(report).splitlines():
        if str(case.parent) in line:
            continue
        assert len(line) <= 100, line


def test_a_case_that_has_not_started_says_so(tmp_path, progress_report):
    case = make_case(tmp_path)

    report = progress_report.collect(case)

    assert report["solve"]["log"] is None
    assert "no solver log" in report["solve"]["why"]
    assert report["eta"]["basis"] is None
    assert report["residuals"]["overall"] == "unknown"
    assert "no solver log" in progress_report.render_text(report)


def test_the_eta_has_one_shape_whether_or_not_it_was_computed(tmp_path, progress_report):
    """`--json` is read by something that indexes it; an eta missing half its keys
    when there is no log makes every reader write the same defensive `.get`."""
    started = progress_report.collect(make_case(tmp_path / "a", log=transient_log(steps=20)))
    never_started = progress_report.collect(make_case(tmp_path / "b"))

    assert never_started["solve"]["log"] is None
    assert set(never_started["eta"]) == set(started["eta"])
    assert never_started["eta"]["seconds_remaining"] is None
    assert never_started["eta"]["confidence"] == "none"
    assert "no solver log" in never_started["eta"]["why"]


def test_an_empty_log_is_read_without_an_eta(tmp_path, progress_report):
    case = make_case(tmp_path)
    (case / "log.pimpleFoam").write_text("", encoding="utf-8")

    report = progress_report.collect(case, log=case / "log.pimpleFoam")

    assert report["solve"]["steps"] == 0
    assert report["solve"]["time"] is None
    assert report["solve"]["fraction"] is None
    assert report["eta"]["basis"] is None
    assert progress_report.render_text(report)


def test_a_log_with_a_banner_but_no_time_loop_yet(tmp_path, progress_report):
    """`Create mesh` has printed and nothing else; the solve is starting."""
    case = make_case(tmp_path)
    (case / "log.pimpleFoam").write_text(
        "/*-- OpenFOAM --*/\nCreate mesh for time = 0\n\nStarting time loop\n", encoding="utf-8"
    )

    report = progress_report.collect(case, log=case / "log.pimpleFoam")

    assert report["solve"]["steps"] == 0
    assert "the time loop has not started" in progress_report.render_text(report)


def test_a_log_with_a_banner_is_found_without_being_pointed_at(tmp_path, progress_report):
    """The question is asked most often in the first minute, before any `Time =` has
    printed. `log.<application>` is named by the case, so it is the solver's log then
    too -- reporting "no solver log found" at a file sitting in the directory was the
    old behaviour and it was wrong."""
    case = make_case(tmp_path)
    (case / "log.pimpleFoam").write_text(
        "/*-- OpenFOAM --*/\nCreate mesh for time = 0\n\nStarting time loop\n", encoding="utf-8"
    )

    report = progress_report.collect(case)  # no --log

    assert Path(report["solve"]["log"]).name == "log.pimpleFoam"
    assert report["solve"]["steps"] == 0
    assert "the time loop has not started" in progress_report.render_text(report)


def test_a_mesher_log_is_not_mistaken_for_the_solver(tmp_path, progress_report):
    case = make_case(tmp_path, log=transient_log(steps=6))
    mesh_log = case / "log.snappyHexMesh"
    mesh_log.write_text("Layer addition iteration 3\nFinished meshing\n", encoding="utf-8")
    # Newer than the solver log, so only the content check can separate them.
    import os
    os.utime(mesh_log, (2 ** 31, 2 ** 31))

    found = progress_report.find_solver_log(case)

    assert found is not None and found.name == "log.pimpleFoam"


def test_the_phase_table_is_where_the_phase_comes_from(tmp_path, progress_report, state):
    case = make_case(tmp_path)
    state.set_phase("mesh", "done", root=case)
    state.set_phase("solve", "running", root=case, note="pimpleFoam under jobd")

    report = progress_report.collect(case)

    assert report["phase"]["current"] == "solve"
    assert report["phase"]["status"] == "running"
    assert report["phase"]["note"] == "pimpleFoam under jobd"
    assert report["phase"]["next_unsettled"] == "geometry"


def test_previews_are_absolute_paths_and_the_newest_of_each_kind(tmp_path, progress_report, state):
    case = make_case(tmp_path)
    study = case.parent
    (study / "renders").mkdir()
    for name in ("mesh_v1.png", "mesh_v2.png", "U.png"):
        (study / "renders" / name).write_bytes(b"\x89PNG")

    state.record("mesh-full", study / "renders" / "mesh_v1.png", root=case)
    state.record("mesh-full", study / "renders" / "mesh_v2.png", root=case, label="after refinement")
    state.record("velocity", study / "renders" / "U.png", root=case)

    rows = progress_report.previews(study)
    by_kind = {row["kind"]: row for row in rows}

    assert set(by_kind) == {"mesh-full", "velocity"}
    assert by_kind["mesh-full"]["path"].endswith("mesh_v2.png")
    assert by_kind["mesh-full"]["label"] == "after refinement"
    assert Path(by_kind["velocity"]["path"]).is_absolute()
    assert by_kind["velocity"]["path"] in progress_report.render_text(progress_report.collect(case))


def test_a_registered_preview_that_was_deleted_is_not_offered(tmp_path, progress_report, state):
    case = make_case(tmp_path)
    study = case.parent
    state.record("velocity", study / "gone.png", root=case)

    assert progress_report.previews(study) == []


def test_the_report_writes_nothing_into_the_case(tmp_path, progress_report):
    """It is called in a loop while a solve is running, so it must not touch disk."""
    case = make_case(tmp_path, log=transient_log(steps=10))
    before = sorted(str(p) for p in case.parent.rglob("*"))

    progress_report.render_text(progress_report.collect(case))

    assert sorted(str(p) for p in case.parent.rglob("*")) == before


# -- odds and ends -----------------------------------------------------------------


def test_the_tail_reader_drops_the_partial_first_line(tmp_path, progress_report):
    path = tmp_path / "log"
    path.write_text("\n".join(f"line {index:05d}" for index in range(1000)) + "\n", encoding="utf-8")

    text = progress_report.tail_text(path, limit=200)

    assert len(text) < 200
    assert text.splitlines()[0].startswith("line ")
    assert text.rstrip().endswith("line 00999")


def test_the_tail_of_a_short_file_is_the_whole_file(tmp_path, progress_report):
    path = tmp_path / "log"
    path.write_text("Time = 1\nExecutionTime = 2 s\n", encoding="utf-8")

    assert progress_report.pace_points(progress_report.tail_text(path)) == [(1.0, 2.0)]


def test_a_missing_file_tails_to_nothing(tmp_path, progress_report):
    assert progress_report.tail_text(tmp_path / "nope") == ""


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "0 s"),
        (45, "45 s"),
        (600, "10 min"),
        (7200, "2 h 0 min"),
        (60 * 60 * 50, "2 d 2 h"),
        (10796, "3 h 0 min"),  # 2.999 h: not "2 h 60 min"
        (60 * 60 * 47.999, "2 d 0 h"),  # and not "1 d 24 h"
    ],
)
def test_durations_read_as_a_person_would_say_them(progress_report, seconds, expected):
    assert progress_report.duration(seconds) == expected


def test_no_carried_minute_or_hour_is_ever_printed(progress_report):
    """Rounding each part on its own is how "2 h 60 min" gets written down."""
    for seconds in range(90, 60 * 60 * 60, 137):
        text = progress_report.duration(seconds)
        assert " 60 min" not in text
        assert " 24 h" not in text


def test_the_cli_prints_json_when_asked(tmp_path, progress_report, capsys):
    case = make_case(tmp_path, log=transient_log(steps=12))

    assert progress_report.main([str(case), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["solve"]["steps"] == 12
    assert payload["case_name"] == "case"


def test_the_cli_prints_text_by_default(tmp_path, progress_report, capsys):
    case = make_case(tmp_path, log=transient_log(steps=12))

    assert progress_report.main([str(case)]) == 0

    out = capsys.readouterr().out
    assert out.startswith("progress:")
    assert "eta" in out


def test_the_script_needs_nothing_the_image_does_not_have(progress_report):
    """Every import root is in the image, and nothing here renders."""
    import ast

    source = (TOOLBOX / "progress_report.py").read_text(encoding="utf-8")
    allowed = set(sys.stdlib_module_names) | {
        "numpy", "matplotlib", "pandas", "pyvista",
    } | {path.stem for path in TOOLBOX.glob("*.py")}

    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])

    assert roots <= allowed, sorted(roots - allowed)
    assert "import pyvista" not in source
    assert not any(name in roots for name in ("scipy", "imageio", "cv2", "PIL"))
    source.encode("ascii")  # raises if a non-ASCII character crept in
