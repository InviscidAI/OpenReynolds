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
