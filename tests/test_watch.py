"""Watch mode: wake on facts, and the situation blurb a fresh thread starts from."""

from __future__ import annotations

import pytest
from rich.console import Console

from openreynolds.backend.base import BackendError, JobStatus
from openreynolds.watch import (
    NullReader,
    _collect_finished,
    _job_report,
    _tail,
    situation,
    watch,
)


@pytest.fixture
def console():
    import os

    return Console(file=open(os.devnull, "w"), force_terminal=False)


class ScriptedReader:
    """Hands back queued lines, then nothing."""

    def __init__(self, lines):
        self._lines = list(lines)

    def poll(self):
        from openreynolds.watch import NOTHING

        return self._lines.pop(0) if self._lines else NOTHING

    def get(self, timeout=None):
        return self._lines.pop(0) if self._lines else None


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch):
    monkeypatch.setattr("openreynolds.watch.POLL_MIN_S", 0.001)
    monkeypatch.setattr("openreynolds.watch.POLL_MAX_S", 0.002)
    monkeypatch.setattr("openreynolds.watch.TICK_S", 0.001)


def start_job(backend, store, name="solve", cmd="simpleFoam"):
    job_id = backend.job_start(cmd, name=name)
    store.record_job(job_id, cmd=cmd, name=name)
    return job_id


# -- the tail ------------------------------------------------------------------


def test_tail_offsets_from_the_end(backend):
    """job_tail reads forward, so a tail has to be asked for by offset."""
    backend.logs["job-1"] = b"A" * 10_000 + b"THE-END"
    status = JobStatus(job_id="job-1", status="exited", log_size=10_007)

    tail = _tail(backend, status)

    assert tail.endswith("THE-END")
    assert len(tail.encode()) <= 2_000
    assert "A" * 3_000 not in tail


def test_tail_of_a_short_log_is_the_whole_thing(backend):
    backend.logs["job-1"] = b"short\n"
    assert _tail(backend, JobStatus("job-1", "exited", log_size=6)) == "short\n"


def test_tail_survives_an_unreadable_log(backend):
    class Broken:
        def job_tail(self, job_id, offset=0):
            raise BackendError("gone", code="not_found")

    assert _tail(Broken(), JobStatus("job-1", "exited", log_size=10)) == ""


# -- the wake message ----------------------------------------------------------


def test_wake_report_is_facts_only(backend):
    backend.logs["job-1"] = b"End\nFinalising\n"
    status = JobStatus(
        job_id="job-1",
        name="solve",
        status="exited",
        exit_code=0,
        end_reason="completed",
        log_size=14,
    )

    report = _job_report(backend, status)

    assert "name=solve" in report
    assert "exit_code=0" in report
    assert "end_reason=completed" in report
    assert "Finalising" in report
    for suggestion in ("should", "try", "next", "recommend", "consider"):
        assert suggestion not in report.lower()


def test_wake_report_carries_the_kill_on_line(backend):
    status = JobStatus(
        job_id="job-1",
        name="solve",
        status="killed",
        end_reason="kill_on_match",
        killed_by="--> FOAM FATAL ERROR: Maximum number of iterations exceeded",
        log_size=0,
    )
    report = _job_report(backend, status)
    assert "kill_on_match" in report
    assert "Maximum number of iterations exceeded" in report


def test_expired_sandbox_is_reported_as_such(backend, store):
    """The volume survives this, so it is a fact the model needs, not a plain exit."""
    job_id = start_job(backend, store)
    backend.jobs[job_id] = JobStatus(
        job_id=job_id, name="solve", status="killed", end_reason="sandbox_expired", log_size=0
    )
    report = _collect_finished(backend, store, Console(quiet=True))
    assert "sandbox_expired" in report


# -- the poll loop -------------------------------------------------------------


def test_idle_when_nothing_is_running(backend, store, console):
    assert watch(backend, store, console, NullReader()).kind == "idle"


def test_typed_input_interrupts_immediately(backend, store, console):
    start_job(backend, store)
    wake = watch(backend, store, console, ScriptedReader(["stop what you are doing"]))
    assert wake.kind == "user"
    assert wake.text == "stop what you are doing"


def test_blank_input_does_not_count_as_an_interruption(backend, store, console):
    job_id = start_job(backend, store)
    backend.jobs[job_id] = JobStatus(job_id=job_id, name="solve", status="exited", exit_code=0)
    wake = watch(backend, store, console, ScriptedReader(["   "]))
    assert wake.kind == "job"


def test_eof_ends_the_session(backend, store, console):
    start_job(backend, store)
    wake = watch(backend, store, console, ScriptedReader([None]))
    assert wake.kind == "eof"


def test_a_finished_job_wakes_with_its_outcome(backend, store, console):
    job_id = start_job(backend, store)
    backend.logs[job_id] = b"time step continuity errors\nEnd\n"
    backend.jobs[job_id] = JobStatus(
        job_id=job_id,
        name="solve",
        status="exited",
        exit_code=0,
        end_reason="completed",
        log_size=len(backend.logs[job_id]),
    )

    wake = watch(backend, store, console, NullReader())

    assert wake.kind == "job"
    assert "exit_code=0" in wake.text
    assert "End" in wake.text
    assert store.session.jobs[job_id].status == "exited"


def test_an_unreadable_job_is_reported_rather_than_polled_forever(backend, store, console):
    job_id = start_job(backend, store)

    def broken(_job_id):
        raise BackendError("instance is deleted", code="conflict", status=409)

    backend.job_status = broken
    wake = watch(backend, store, console, NullReader())

    assert wake.kind == "job"
    assert "could not be read" in wake.text
    assert store.session.jobs[job_id].status == "unknown"


# -- the situation blurb -------------------------------------------------------


def test_blurb_states_the_facts_a_fresh_thread_needs(store):
    store.session.instance_id = "inst-42"
    assert "study" in situation(store)
    assert "inst-42" in situation(store)
    assert "No jobs have been started" in situation(store)


def test_blurb_separates_running_from_finished(backend, store):
    running = start_job(backend, store, name="solve")
    done = start_job(backend, store, name="mesh", cmd="snappyHexMesh")
    backend.jobs[done] = JobStatus(
        job_id=done, name="mesh", status="exited", exit_code=0, end_reason="completed"
    )

    blurb = situation(store, backend)

    assert "Jobs still running:" in blurb
    assert f"{running} (solve): simpleFoam" in blurb
    assert "Jobs that have finished:" in blurb
    assert "completed, exit_code=0" in blurb
    assert store.session.jobs[done].status == "exited"


def test_blurb_offers_no_next_step(backend, store):
    start_job(backend, store)
    blurb = situation(store, backend).lower()
    for suggestion in ("should", "you could", "recommend", "next step", "suggest"):
        assert suggestion not in blurb
