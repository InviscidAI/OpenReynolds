"""Stopping work, and confirming it stopped.

The service marks a job killed whether or not the signal reached anything, and a solver
launched through mpirun puts its ranks outside the job's process group. Both were true
at once in a live run: the wrapper died, the record said killed, and eight cores kept
going. Everything here exists because of that.
"""

from __future__ import annotations

import pytest

from openreynolds.backend.base import BackendError, ExecResult, JobStatus
from openreynolds.stopping import StopReport, running_solvers, stop_everything


@pytest.fixture(autouse=True)
def no_settling(monkeypatch):
    monkeypatch.setattr("openreynolds.stopping.SETTLE_S", 0.0)


def with_jobs(backend, store, *names):
    for name in names:
        job_id = backend.job_start(f"run {name}", name=name)
        store.record_job(job_id, cmd=f"run {name}", name=name)
    return store


def processes(backend, *names):
    """Make `ps -eo comm=` report these."""
    backend.exec_result = ExecResult(0, "\n".join(("bash", *names, "ps")), False, None)


# -- seeing what is actually running -------------------------------------------


def test_solvers_are_read_from_ps_not_a_pattern_search(backend):
    """`pgrep -f simpleFoam` also matches the shell doing the searching, which is how
    a killed solve looked alive."""
    processes(backend, "simpleFoam", "mpirun")
    assert running_solvers(backend) == ["simpleFoam", "mpirun"]


def test_unrelated_processes_are_not_reported(backend):
    processes(backend, "sshd", "python3", "grep")
    assert running_solvers(backend) == []


def test_an_unreachable_instance_reports_nothing_rather_than_raising(backend):
    def refuse(cmd, cwd=None, timeout_s=120):
        raise BackendError("gone", code="not_found")

    backend.exec = refuse
    assert running_solvers(backend) == []


# -- stopping ------------------------------------------------------------------


def test_it_stops_every_job_and_records_them_stopped(backend, store):
    with_jobs(backend, store, "solve", "mesh")
    processes(backend)

    report = stop_everything(backend, store)

    assert sorted(report.killed) == ["mesh", "solve"]
    assert report.clean
    assert store.live_jobs() == []
    assert all(j.end_reason == "killed_by_client" for j in store.session.jobs.values())


def test_nothing_running_is_not_a_failure(backend, store):
    processes(backend)
    report = stop_everything(backend, store)
    assert report.clean
    assert "nothing was running" in report.lines()


def test_a_survivor_is_reported_rather_than_assumed_dead(backend, store):
    """The whole point: the job says killed and the solver is still burning cores."""
    with_jobs(backend, store, "sweep")
    processes(backend, "simpleFoam", "mpirun")

    report = stop_everything(backend, store)

    assert report.killed == ["sweep"]
    assert not report.clean
    assert sorted(report.survivors) == ["mpirun", "simpleFoam"]
    joined = " ".join(report.lines())
    assert "still running after every job was signalled" in joined
    assert "how compute leaks" in joined


def test_force_kills_what_outlived_its_job(backend, store):
    with_jobs(backend, store, "sweep")
    commands: list[str] = []
    state = {"alive": True}

    def exec_(cmd, cwd=None, timeout_s=120):
        commands.append(cmd)
        if cmd.startswith("pkill"):
            state["alive"] = False
            return ExecResult(0, "", False, None)
        listing = ["bash"] + (["simpleFoam"] if state["alive"] else []) + ["ps"]
        return ExecResult(0, "\n".join(listing), False, None)

    backend.exec = exec_

    report = stop_everything(backend, store, force=True)

    assert any(c.startswith("pkill -9") and "-x simpleFoam" in c for c in commands)
    assert report.survivors == []
    assert report.clean


def test_without_force_nothing_is_pkilled(backend, store):
    with_jobs(backend, store, "sweep")
    commands: list[str] = []

    def exec_(cmd, cwd=None, timeout_s=120):
        commands.append(cmd)
        return ExecResult(0, "bash\nsimpleFoam\nps", False, None)

    backend.exec = exec_
    stop_everything(backend, store, force=False)

    assert not any(c.startswith("pkill") for c in commands)


def test_a_kill_that_fails_is_named_not_swallowed(backend, store):
    with_jobs(backend, store, "solve")
    processes(backend)

    def refuse(job_id, signal="TERM"):
        raise BackendError("job not found", code="not_found", status=404)

    backend.job_kill = refuse
    report = stop_everything(backend, store)

    assert report.failed and report.failed[0][0] == "solve"
    assert not report.clean
    assert any("could not stop solve" in line for line in report.lines())


def test_survivors_trigger_an_escalated_signal(backend, store):
    with_jobs(backend, store, "sweep")
    signals: list[str] = []
    original = backend.job_kill

    def record(job_id, signal="TERM"):
        signals.append(signal)
        return original(job_id)

    backend.job_kill = record
    processes(backend, "simpleFoam")

    stop_everything(backend, store)

    assert signals[0] == "TERM"
    assert "KILL" in signals, "a solver that ignored TERM gets one it cannot ignore"


# -- the report ----------------------------------------------------------------


def test_a_clean_report_says_the_instance_is_idle():
    report = StopReport(killed=["solve"])
    assert report.clean
    assert "the instance is idle" in report.lines()


def test_an_escalation_is_visible_in_the_report():
    report = StopReport(escalated=["sweep"])
    assert "ignored the first signal" in " ".join(report.lines())


# -- the force path itself ------------------------------------------------------


def busy(backend, *names):
    """The instance reports these solvers as running."""
    backend.exec_result = ExecResult(0, "\n".join(names) + "\n", False, None)


def test_force_kills_each_leftover_with_its_own_pkill(backend, store):
    """pkill takes exactly one pattern. A second `-x name` makes it exit 2 having
    killed nothing, which is how a stop that stops nothing reports success."""
    busy(backend, "mpirun", "simpleFoam")
    store.record_job("job-1", cmd="mpirun simpleFoam", name="solve")
    backend.jobs["job-1"] = JobStatus(job_id="job-1", status="running", name="solve")

    stop_everything(backend, store, force=True)

    pkills = [cmd for cmd in backend.execs if cmd.startswith("pkill")]
    assert set(pkills) == {"pkill -9 -x mpirun", "pkill -9 -x simpleFoam"}
    for command in pkills:
        assert command.count("-x") == 1, f"more than one pattern in {command!r}"


def test_a_name_is_only_killed_once_however_many_copies_are_running(backend, store):
    busy(backend, "simpleFoam", "simpleFoam", "simpleFoam")
    store.record_job("job-1", cmd="simpleFoam", name="solve")
    backend.jobs["job-1"] = JobStatus(job_id="job-1", status="running", name="solve")

    stop_everything(backend, store, force=True)

    assert set(c for c in backend.execs if c.startswith("pkill")) == {"pkill -9 -x simpleFoam"}


def test_pkill_finding_nothing_is_not_a_failure(backend, store):
    """Exit 1 means it was already gone, which is the outcome that was wanted."""
    busy(backend, "simpleFoam")
    backend.exec_results["pkill -9 -x simpleFoam"] = ExecResult(1, "", False, None)
    store.record_job("job-1", cmd="simpleFoam", name="solve")
    backend.jobs["job-1"] = JobStatus(job_id="job-1", status="running", name="solve")

    report = stop_everything(backend, store, force=True)

    assert not report.failed


def test_a_pkill_that_errors_is_reported_rather_than_swallowed(backend, store):
    busy(backend, "simpleFoam")
    backend.exec_results["pkill -9 -x simpleFoam"] = ExecResult(
        2, "pkill: only one pattern can be provided\n", False, None
    )
    store.record_job("job-1", cmd="simpleFoam", name="solve")
    backend.jobs["job-1"] = JobStatus(job_id="job-1", status="running", name="solve")

    report = stop_everything(backend, store, force=True)

    assert not report.clean
    assert any("only one pattern" in why for _name, why in report.failed)
    assert any("could not stop simpleFoam" in line for line in report.lines())


def test_without_force_nothing_is_pkilled(backend, store):
    busy(backend, "simpleFoam")
    store.record_job("job-1", cmd="simpleFoam", name="solve")
    backend.jobs["job-1"] = JobStatus(job_id="job-1", status="running", name="solve")

    stop_everything(backend, store, force=False)

    assert not [c for c in backend.execs if c.startswith("pkill")]


def quietening(backend, *names, after=1):
    """Report these solvers as running, then report the instance quiet."""
    calls = {"n": 0}

    def looking(cmd, cwd=None, timeout_s=120):
        backend.execs.append(cmd)
        if not cmd.startswith("ps "):
            return ExecResult(0, "", False, None)
        calls["n"] += 1
        running = "\n".join(names) if calls["n"] <= after else ""
        return ExecResult(0, running, False, None)

    backend.exec = looking


def test_stopping_stops_looking_once_the_instance_is_quiet(backend, store):
    quietening(backend, "simpleFoam", after=1)
    store.record_job("job-1", cmd="simpleFoam", name="solve")
    backend.jobs["job-1"] = JobStatus(job_id="job-1", status="running", name="solve")

    report = stop_everything(backend, store, force=True)

    assert report.clean
    assert report.passes == 0, "it did not keep hammering a quiet instance"
    assert len([c for c in backend.execs if c.startswith("pkill")]) == 1


def test_a_ladder_that_starts_the_next_solve_is_still_stopped(backend, store):
    """Killing the solver a driver script is running just frees it to start the next
    one, so a single look three seconds later finds a brand new simpleFoam and calls
    it a failure while everything did in fact die."""
    quietening(backend, "simpleFoam", after=3)
    store.record_job("job-1", cmd="./ladder.sh", name="ladder")
    backend.jobs["job-1"] = JobStatus(job_id="job-1", status="running", name="ladder")

    report = stop_everything(backend, store, force=True)

    assert report.clean, "it kept going until the instance was actually quiet"
    assert report.passes >= 1
    assert any("passes" in line for line in report.lines())


def test_something_that_never_dies_is_reported_rather_than_looped_on_forever(backend, store):
    quietening(backend, "simpleFoam", after=999)
    store.record_job("job-1", cmd="simpleFoam", name="solve")
    backend.jobs["job-1"] = JobStatus(job_id="job-1", status="running", name="solve")

    report = stop_everything(backend, store, force=True)

    assert not report.clean
    assert "simpleFoam" in " ".join(report.lines())


# -- staying inside this study's own work --------------------------------------
#
# The account is capped at one instance and `acquire()` joins the one already there,
# so "what is running on this instance" and "what this study started" are different
# questions. Everything below is the difference between them. A live run lost a
# 22-minute solve to a second terminal's ordinary `/exit`.


def own(backend, *pairs, home="/work/study-test"):
    """Make the /proc probe report these `(pid, name)` as working under `home`."""
    body = "\n".join(f"{pid} {name}" for pid, name in pairs)
    backend.exec_result = ExecResult(0, body, False, None)
    return home


def test_a_scoped_sweep_asks_where_a_process_is_working_not_just_what_it_is(backend, store):
    home = own(backend, ("101", "simpleFoam"))
    stop_everything(backend, store, home=home)
    probe = [c for c in backend.execs if "/proc" in c]
    assert probe, "it looked at ps instead of at working directories"
    assert "/work/study-test" in probe[0]
    assert not any(c.startswith("ps ") for c in backend.execs)


def test_a_scoped_sweep_kills_by_pid_never_by_name(backend, store):
    home = own(backend, ("101", "simpleFoam"), ("102", "mpirun"))
    store.record_job("job-1", cmd="simpleFoam", name="solve")
    backend.jobs["job-1"] = JobStatus(job_id="job-1", status="running", name="solve")

    stop_everything(backend, store, home=home)

    assert not any(c.startswith("pkill") for c in backend.execs), (
        "pkill -9 -x <name> reaches every copy on the instance, including other studies'"
    )
    kills = [c for c in backend.execs if c.startswith("kill -9")]
    assert kills and "101" in kills[0] and "102" in kills[0]


def test_a_scoped_sweep_needs_no_force_flag(backend, store):
    """Scoped, the kill cannot reach past this study, so it is safe by construction.
    `--force` stayed a flag only for the unscoped instance-wide sweep."""
    home = own(backend, ("101", "simpleFoam"))
    stop_everything(backend, store, home=home)
    assert any(c.startswith("kill -9") for c in backend.execs)


def test_another_session_solving_in_its_own_directory_is_not_touched(backend, store):
    """The whole finding, in one test: the probe is anchored on this study's home, so a
    solver working under a different one is never returned and never killed."""

    def only_ours(cmd, cwd=None, timeout_s=120):
        backend.execs.append(cmd)
        if "/proc" in cmd:
            # The real probe filters by cwd; nothing of ours is running.
            return ExecResult(0, "", False, None)
        return ExecResult(0, "ok", False, None)

    backend.exec = only_ours
    report = stop_everything(backend, store, home="/work/study-test")

    assert report.clean
    assert report.survivors == []
    assert not any(c.startswith(("pkill", "kill -9")) for c in backend.execs)


def test_a_session_with_no_jobs_of_its_own_still_swept_the_whole_instance(backend, store):
    """`survivors` was computed whether or not this study had killed anything, so a
    session that started zero jobs -- a question answered without solving, or a run that
    failed early -- still enumerated the instance and pkilled by name on the way out."""
    processes(backend, "pimpleFoam")
    assert store.live_jobs() == []

    unscoped = stop_everything(backend, store, force=True)
    assert any(c.startswith("pkill") for c in backend.execs), "the old, instance-wide path"
    assert unscoped.survivors  # somebody else's solver, reported as this study's leftover

    backend.execs.clear()
    own(backend)  # the /proc probe finds nothing of ours
    scoped = stop_everything(backend, store, force=True, home="/work/study-test")
    assert scoped.clean
    assert not any(c.startswith(("pkill", "kill -9")) for c in backend.execs)


def test_a_study_that_predates_homes_is_treated_as_unscoped(backend, store):
    """Studies resumed from before homes existed keep the volume root, where "under my
    home" is true of every study at once and so distinguishes nothing."""
    processes(backend, "simpleFoam")
    stop_everything(backend, store, force=True, home="/work")
    assert any(c.startswith("ps ") for c in backend.execs)
    assert not any("/proc" in c for c in backend.execs)
