"""The bar: log facts, phases, estimates, and the one picture they compose into."""

from __future__ import annotations

import time
from types import SimpleNamespace

from rich.console import Console

from openreynolds.backend.base import JobStatus
from openreynolds.progress import (
    Progress,
    Tracker,
    bar,
    case_dir_from_cmd,
    duration,
    log_target_from_cmd,
    parse_control_dict,
    parse_log_tail,
    phase_from_cmd,
)
from openreynolds.view import ConsoleView

HOME = "/work/study-test"

PIMPLE_TAIL = """\
Courant Number mean: 0.0812 max: 0.4987
deltaT = 0.000251
Time = 1.1908618

PIMPLE: iteration 1
smoothSolver:  Solving for Ux, Initial residual = 0.000312, Final residual = 1.2e-06, No Iterations 2
smoothSolver:  Solving for Uy, Initial residual = 0.000201, Final residual = 9e-07, No Iterations 2
GAMG:  Solving for p, Initial residual = 0.0087, Final residual = 4e-07, No Iterations 8
time step continuity errors : sum local = 2.1e-06, global = 1e-08, cumulative = 3e-06
ExecutionTime = 812.3 s  ClockTime = 840 s
"""

SNAPPY_TAIL = """\
Refinement phase
----------------

Refinement iteration 3
----------------------
Refined mesh : cells:123456  faces:400000  points:150000
"""

CONTROL_DICT = """\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
// endTime 99;  -- an old value, commented out
application     pimpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         6;
deltaT          0.0001;
writeControl    adjustableRunTime;
writeInterval   0.05;
"""


# -- reading logs --------------------------------------------------------------


def test_a_solver_tail_yields_time_residuals_and_clock():
    facts = parse_log_tail(PIMPLE_TAIL)
    assert facts.sim_time == 1.1908618
    assert facts.delta_t == 0.000251
    assert facts.residuals == {"Ux": 0.000312, "Uy": 0.000201, "p": 0.0087}
    assert facts.continuity == 2.1e-06
    assert facts.courant_max == 0.4987
    assert facts.clock_s == 840.0
    assert facts.is_solver
    assert facts.last_line.startswith("ExecutionTime")


def test_residuals_come_from_the_last_time_step_only():
    two = "Time = 1\nGAMG:  Solving for p, Initial residual = 0.5, Final residual = 0.1, No Iterations 3\n" + PIMPLE_TAIL
    assert parse_log_tail(two).residuals["p"] == 0.0087


def test_a_mesher_tail_yields_its_phase_and_iteration():
    facts = parse_log_tail(SNAPPY_TAIL)
    assert facts.mesh_phase == "refining"
    assert facts.mesh_iteration == 3
    assert facts.mesh_cells == 123456
    assert not facts.is_solver


def test_the_last_mesh_phase_marker_wins():
    text = SNAPPY_TAIL + "\nMorphing phase\n--------------\n"
    assert parse_log_tail(text).mesh_phase == "snapping"


def test_an_empty_tail_is_no_facts():
    facts = parse_log_tail("")
    assert facts.sim_time is None and not facts.residuals and facts.last_line == ""


def test_control_dict_gives_the_bounds_and_ignores_comments():
    found = parse_control_dict(CONTROL_DICT)
    assert found["startTime"] == 0.0
    assert found["endTime"] == 6.0
    assert found["writeInterval"] == 0.05


def test_a_stop_at_that_is_not_end_time_means_no_end():
    text = CONTROL_DICT.replace("stopAt          endTime;", "stopAt writeNow;")
    found = parse_control_dict(text)
    assert "endTime" not in found
    assert found["stopAt"] == "writeNow"


# -- reading commands ----------------------------------------------------------


def test_phases_come_from_the_executable():
    assert phase_from_cmd("cd case && blockMesh > log.blockMesh 2>&1") == ("meshing", "blockMesh")
    assert phase_from_cmd("mpirun -np 6 pimpleFoam -parallel > log 2>&1") == ("solving", "pimpleFoam")
    assert phase_from_cmd("gmshToFoam mesh.msh") == ("meshing", "gmshToFoam")
    assert phase_from_cmd("foamToVTK -latestTime") == ("post-processing", "foamToVTK")
    assert phase_from_cmd("decomposePar -force") == ("decomposing", "decomposePar")
    assert phase_from_cmd("python3 tools/analyze.py case") == ("running python", "python3")
    assert phase_from_cmd("foamRun -solver incompressibleFluid") == ("solving", "foamRun")
    assert phase_from_cmd("ls -la") == ("running", "")


def test_a_chain_reads_as_its_last_recognised_part():
    assert phase_from_cmd("blockMesh && snappyHexMesh -overwrite && simpleFoam")[0] == "solving"


def test_the_case_directory_is_the_last_cd_resolved_against_cwd():
    assert case_dir_from_cmd("cd /work/s/case && pimpleFoam", HOME) == "/work/s/case"
    assert case_dir_from_cmd("cd lshape; simpleFoam", HOME) == f"{HOME}/lshape"
    assert case_dir_from_cmd("cd 'a b' && ls", HOME) == f"{HOME}/a b"
    assert case_dir_from_cmd("pimpleFoam > log", HOME) == HOME
    assert case_dir_from_cmd("", HOME) == HOME


def test_the_log_target_is_the_last_real_redirect():
    case = f"{HOME}/case"
    assert log_target_from_cmd("pimpleFoam > log.pimpleFoam 2>&1", case) == f"{case}/log.pimpleFoam"
    assert log_target_from_cmd("snappyHexMesh | tee log.snappy", case) == f"{case}/log.snappy"
    assert log_target_from_cmd("blockMesh > /dev/null 2>&1", case) is None
    assert log_target_from_cmd("blockMesh", case) is None
    assert log_target_from_cmd("x > /abs/log", case) == "/abs/log"


# -- the picture ---------------------------------------------------------------


def start_solve(backend, store, tail=PIMPLE_TAIL, cwd=HOME, cmd=None):
    cmd = cmd or f"cd {HOME}/case && mpirun -np 6 pimpleFoam -parallel > log.pimpleFoam 2>&1"
    job_id = backend.job_start(cmd, cwd=cwd, name="solve")
    store.record_job(job_id, cmd=cmd, name="solve", cwd=cwd)
    backend.logs[job_id] = tail.encode()
    backend.jobs[job_id] = JobStatus(job_id=job_id, status="running", name="solve", log_size=len(tail))
    return job_id


def test_a_running_solve_gets_a_fraction_against_end_time(backend, store, view):
    backend.files[f"{HOME}/case/system/controlDict"] = CONTROL_DICT.encode()
    start_solve(backend, store)
    tracker = Tracker(view, backend=backend, store=store, home=HOME)

    [job] = tracker.refresh_jobs(force=True)

    assert job.phase == "solving"
    assert job.end_time == 6.0
    assert abs(job.fraction - 1.1908618 / 6) < 1e-6
    # ClockTime 840 s for 1.19 s of simulated time: about 3,400 s more to go.
    assert 3000 < job.eta_s < 4000
    assert job.elapsed_s is not None

    snap = tracker.snapshot()
    assert snap.phase == "solving"
    assert "solving solve" in snap.headline
    assert "Time 1.19086 / 6 s" in snap.headline
    assert "left" in snap.headline
    assert "Ux 3.1e-04" in snap.detail and "Co max 0.50" in snap.detail
    assert snap.percent().strip() == "20%"


def test_the_control_dict_is_read_from_the_mirror_first(backend, store, view):
    """A copy already on this machine costs no round trip; the instance is asked
    only when the mirror has not brought it home."""
    local = store.fetch_dir() / "study-test" / "case" / "system" / "controlDict"
    local.parent.mkdir(parents=True)
    local.write_text(CONTROL_DICT.replace("endTime         6;", "endTime 12;"), encoding="utf-8")
    backend.files[f"{HOME}/case/system/controlDict"] = CONTROL_DICT.encode()
    start_solve(backend, store)
    tracker = Tracker(view, backend=backend, store=store, home=HOME, local_dir=store.fetch_dir())

    [job] = tracker.refresh_jobs(force=True)

    assert job.end_time == 12.0


def test_a_solve_with_no_known_end_has_no_percentage(backend, store, view):
    start_solve(backend, store)  # no controlDict anywhere
    tracker = Tracker(view, backend=backend, store=store, home=HOME)

    [job] = tracker.refresh_jobs(force=True)
    snap = tracker.snapshot()

    assert job.fraction is None and job.eta_s is None
    assert "Time 1.19086 s" in snap.headline
    assert snap.percent().strip() == ""
    assert snap.busy, "a running job pulses the bar even without a fraction"


def test_a_job_that_was_recorded_without_a_cwd_still_shows(backend, store, view):
    """Records from before `cwd` existed load with it empty; the home directory
    stands in, and the line is still drawn."""
    cmd = "mpirun -np 6 pimpleFoam -parallel > log 2>&1"
    job_id = backend.job_start(cmd, name="old")
    store.record_job(job_id, cmd=cmd, name="old")
    backend.logs[job_id] = PIMPLE_TAIL.encode()
    backend.jobs[job_id] = JobStatus(job_id=job_id, status="running", name="old", log_size=99)
    backend.files[f"{HOME}/system/controlDict"] = CONTROL_DICT.encode()
    tracker = Tracker(view, backend=backend, store=store, home=HOME)

    [job] = tracker.refresh_jobs(force=True)

    assert job.end_time == 6.0


def test_a_mesher_job_shows_its_phase(backend, store, view):
    cmd = f"cd {HOME}/case && snappyHexMesh -overwrite > log.snappyHexMesh 2>&1"
    start_solve(backend, store, tail=SNAPPY_TAIL, cmd=cmd)
    tracker = Tracker(view, backend=backend, store=store, home=HOME)

    tracker.refresh_jobs(force=True)
    snap = tracker.snapshot()

    assert snap.phase == "meshing"
    assert "refining iteration 3" in snap.headline
    assert "123,456 cells" in snap.detail


def test_refreshing_is_cheap_when_recent(backend, store, view):
    start_solve(backend, store)
    tracker = Tracker(view, backend=backend, store=store, home=HOME)
    tracker.refresh_jobs(force=True)
    calls = len(backend.execs)
    before = dict(backend.jobs)

    tracker.refresh_jobs()  # not forced, and just looked

    assert backend.jobs == before and len(backend.execs) == calls


def test_a_job_that_ended_leaves_the_picture(backend, store, view):
    job_id = start_solve(backend, store)
    tracker = Tracker(view, backend=backend, store=store, home=HOME)
    tracker.refresh_jobs(force=True)
    store.update_job(job_id, status="exited")

    assert tracker.refresh_jobs(force=True) == []
    assert tracker.snapshot().phase == "waiting"


def test_an_unreadable_job_is_skipped_not_raised(backend, store, view):
    from openreynolds.backend.base import BackendError

    start_solve(backend, store)

    def broken(job_id):
        raise BackendError("cold start", code="unavailable", status=503)

    backend.job_status = broken
    tracker = Tracker(view, backend=backend, store=store, home=HOME)

    assert tracker.refresh_jobs(force=True) == []


def test_the_session_thread_says_what_it_is_doing(view):
    tracker = Tracker(view)

    tracker.begin("thinking")
    snap = tracker.snapshot()
    assert snap.phase == "thinking" and snap.headline.startswith("thinking") and snap.busy

    tracker.begin("writing")
    assert tracker.snapshot().headline.startswith("writing")

    tracker.begin("tool", "bash", cmd="cd case && blockMesh > log.blockMesh 2>&1", cwd=HOME)
    snap = tracker.snapshot()
    assert snap.phase == "meshing"
    assert snap.headline.startswith("bash: blockMesh")

    tracker.idle()
    assert tracker.snapshot().phase == "waiting"
    assert tracker.snapshot().headline == "waiting for you"
    assert not tracker.snapshot().busy

    tracker.begin("waiting")
    assert tracker.snapshot().phase == "waiting"


def test_a_running_job_outranks_the_thought_and_keeps_it_on_the_line(backend, store, view):
    """The complaint: the solve vanished from the screen the moment the model
    started thinking. Now the solve is the headline and the thought rides along."""
    backend.files[f"{HOME}/case/system/controlDict"] = CONTROL_DICT.encode()
    start_solve(backend, store)
    tracker = Tracker(view, backend=backend, store=store, home=HOME)
    tracker.refresh_jobs(force=True)

    tracker.begin("thinking")
    snap = tracker.snapshot()

    assert snap.phase == "solving"
    assert snap.fraction is not None
    assert snap.detail.startswith("thinking ·")
    assert "Ux" in snap.detail


def test_a_bash_command_that_names_its_log_is_watched_while_it_runs(backend, store, view):
    log = f"{HOME}/case/log.snappyHexMesh"
    backend.files[log] = SNAPPY_TAIL.encode()
    tracker = Tracker(view, backend=backend, store=store, home=HOME)
    tracker.begin(
        "tool", "bash",
        cmd="cd case && snappyHexMesh -overwrite > log.snappyHexMesh 2>&1", cwd=HOME,
    )

    tracker._poll_tool_log()
    snap = tracker.snapshot()

    assert snap.phase == "meshing"
    assert "refining iteration 3" in snap.headline
    assert "123,456 cells" in snap.detail


def test_a_solver_in_bash_gets_a_fraction_too(backend, store, view):
    backend.files[f"{HOME}/case/log.simpleFoam"] = PIMPLE_TAIL.encode()
    backend.files[f"{HOME}/case/system/controlDict"] = CONTROL_DICT.encode()
    tracker = Tracker(view, backend=backend, store=store, home=HOME)
    tracker.begin("tool", "bash", cmd="cd case && simpleFoam > log.simpleFoam 2>&1", cwd=HOME)

    tracker._poll_tool_log()
    snap = tracker.snapshot()

    assert snap.phase == "solving"
    assert abs(snap.fraction - 1.1908618 / 6) < 1e-6
    assert "Time 1.19086 / 6 s" in snap.headline


def test_a_bash_command_without_a_log_just_has_a_clock(backend, store, view):
    tracker = Tracker(view, backend=backend, store=store, home=HOME)
    tracker.begin("tool", "bash", cmd="checkMesh", cwd=HOME)
    tracker._poll_tool_log()  # nothing to read, nothing to raise
    snap = tracker.snapshot()
    assert snap.phase == "checking the mesh"
    assert snap.headline.startswith("bash: checkMesh")
    assert snap.fraction is None and snap.busy


def test_the_mirror_shows_up_while_it_copies(view):
    tracker = Tracker(view)
    tracker.sync_begin()
    snap = tracker.snapshot()
    assert snap.phase == "syncing" and snap.busy

    tracker.sync_end(SimpleNamespace(pulled=["a", "b"]))
    snap = tracker.snapshot()
    assert snap.phase == "waiting"
    assert snap.detail == "2 file(s) arrived"


def test_the_mirror_rides_along_when_something_else_is_happening(view):
    tracker = Tracker(view)
    tracker.begin("thinking")
    tracker.sync_begin()
    assert tracker.snapshot().detail.startswith("syncing files")


def test_every_change_is_pushed_to_the_view(view):
    tracker = Tracker(view)
    tracker.begin("thinking")
    tracker.idle()
    tracker.sync_end()
    assert len(view.progresses) == 3
    assert all(isinstance(p, Progress) for p in view.progresses)


def test_a_view_that_fails_does_not_end_the_session():
    class Broken:
        def progress(self, snapshot):
            raise RuntimeError("the app is gone")

    Tracker(Broken()).push()  # must not raise


def test_the_wake_gets_facts_and_not_the_estimate(backend, store, view):
    backend.files[f"{HOME}/case/system/controlDict"] = CONTROL_DICT.encode()
    start_solve(backend, store)
    tracker = Tracker(view, backend=backend, store=store, home=HOME)
    tracker.refresh_jobs(force=True)

    [line] = tracker.facts_for_wake()

    assert line == "solve: solver time 1.19086 of endTime 6, ClockTime 840 s"
    assert "left" not in line


def test_the_thread_redraws_on_its_own(view):
    import openreynolds.progress as progress

    original = progress.TICK_S
    progress.TICK_S = 0.01
    tracker = Tracker(view)
    try:
        tracker.start()
        deadline = time.time() + 5.0
        while len(view.progresses) < 3 and time.time() < deadline:
            time.sleep(0.01)
    finally:
        tracker.stop()
        progress.TICK_S = original
    assert len(view.progresses) >= 3
    assert view.progresses[-1].tick > view.progresses[0].tick


# -- drawing -------------------------------------------------------------------


def test_the_bar_fills_pulses_or_sits():
    half = bar(Progress(fraction=0.5), width=10)
    assert half == "█████░░░░░"
    pulse_a = bar(Progress(busy=True, tick=0), width=10)
    pulse_b = bar(Progress(busy=True, tick=1), width=10)
    assert "▓" in pulse_a and pulse_a != pulse_b
    assert bar(Progress(), width=10) == "░" * 10


def test_durations_read_like_a_person_wrote_them():
    assert duration(45) == "45 s"
    assert duration(14 * 60) == "14 min"
    assert duration(2 * 3600 + 5 * 60) == "2 h 5 min"


def test_the_plain_terminal_prints_a_line_when_the_picture_changes():
    console = Console(record=True, width=120, force_terminal=False)
    view = ConsoleView(console)

    view.progress(Progress("solving", "solving solve · Time 1 / 6 s", "p 1.0e-03", 0.2, True, 1))
    view.progress(Progress("solving", "solving solve · Time 1.1 / 6 s", "p 1.0e-03", 0.21, True, 2))
    view.progress(Progress("solving", "solving solve · Time 2 / 6 s", "p 1.0e-03", 0.34, True, 3))
    view.progress(Progress("thinking", "thinking · 4 s", "", None, True, 4))
    view.progress(Progress("waiting", "waiting for you", "", None, False, 5))

    out = console.export_text()
    assert out.count("solving solve") == 2, "same fifth of the bar: not repeated"
    assert "20%" in out and "34%" in out
    assert "p 1.0e-03" in out
    assert "thinking" not in out, "the stream already shows those"
    assert "waiting" not in out


# -- when the bar should show at all -------------------------------------------


def test_the_bar_shows_for_real_work_and_hides_otherwise():
    from openreynolds.tui import _bar_worth_showing

    assert _bar_worth_showing(Progress("solving", "solving x", "", 0.3, True, 1))
    assert _bar_worth_showing(Progress("meshing", "meshing", "", None, True, 1))
    assert _bar_worth_showing(Progress("syncing", "syncing files", "", None, True, 1))
    # A bare turn is not enough: thinking and writing are not the bar's to show.
    assert not _bar_worth_showing(Progress("thinking", "thinking · 12 s", "", None, True, 1))
    assert not _bar_worth_showing(Progress("writing", "writing · 3 s", "", None, True, 1))
    assert not _bar_worth_showing(Progress("waiting", "waiting for you", "", None, False, 1))
