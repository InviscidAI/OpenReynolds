"""Watch mode: wake on facts, and the situation blurb a fresh thread starts from."""

from __future__ import annotations

import time

import pytest
from rich.console import Console

from openreynolds.backend.base import BackendError, JobStatus
from openreynolds.watch import (
    NOTHING,
    LineReader,
    NullReader,
    _collect_finished,
    _job_report,
    _tail,
    situation,
    watch,
)


from openreynolds.view import ConsoleView

from conftest import ScriptedReader  # noqa: E402

pytestmark = pytest.mark.usefixtures("fast_polling")


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


def test_expired_sandbox_is_reported_as_such(backend, store, view):
    """The volume survives this, so it is a fact the model needs, not a plain exit."""
    job_id = start_job(backend, store)
    backend.jobs[job_id] = JobStatus(
        job_id=job_id, name="solve", status="killed", end_reason="sandbox_expired", log_size=0
    )
    report = _collect_finished(backend, store, view)
    assert "sandbox_expired" in report


# -- the poll loop -------------------------------------------------------------


def test_idle_when_nothing_is_running(backend, store, view):
    assert watch(backend, store, view, NullReader()).kind == "idle"


def test_typed_input_interrupts_immediately(backend, store, view):
    start_job(backend, store)
    wake = watch(backend, store, view, ScriptedReader(["stop what you are doing"]))
    assert wake.kind == "user"
    assert wake.text == "stop what you are doing"


def test_blank_input_does_not_count_as_an_interruption(backend, store, view):
    job_id = start_job(backend, store)
    backend.jobs[job_id] = JobStatus(job_id=job_id, name="solve", status="exited", exit_code=0)
    wake = watch(backend, store, view, ScriptedReader(["   "]))
    assert wake.kind == "job"


def test_eof_ends_the_session(backend, store, view):
    start_job(backend, store)
    wake = watch(backend, store, view, ScriptedReader([None]))
    assert wake.kind == "eof"


def test_a_finished_job_wakes_with_its_outcome(backend, store, view):
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

    wake = watch(backend, store, view, NullReader())

    assert wake.kind == "job"
    assert "exit_code=0" in wake.text
    assert "End" in wake.text
    assert store.session.jobs[job_id].status == "exited"


def test_an_unreadable_job_is_reported_rather_than_polled_forever(backend, store, view):
    job_id = start_job(backend, store)

    def broken(_job_id):
        raise BackendError("instance is deleted", code="conflict", status=409)

    backend.job_status = broken
    wake = watch(backend, store, view, NullReader())

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


def test_the_blurb_refreshes_a_job_whose_status_could_not_be_read(backend, store):
    """A resume after an outage starts from "unknown", and the blurb used to report
    that word to the model instead of the real end reason."""
    job_id = start_job(backend, store)
    store.update_job(job_id, status="unknown")
    backend.jobs[job_id] = JobStatus(
        job_id=job_id, name="solve", status="killed", end_reason="sandbox_expired"
    )

    blurb = situation(store, backend)

    assert "sandbox_expired" in blurb
    assert "unknown" not in blurb
    assert store.session.jobs[job_id].end_reason == "sandbox_expired"


def test_watch_gives_up_at_a_deadline_without_killing_anything(backend, store, view):
    """The bound ends the waiting, never the job -- it lives on the instance."""
    import time as _time

    start_job(backend, store)
    wake = watch(backend, store, view, NullReader(), deadline=_time.monotonic() - 1)

    assert wake.kind == "timeout"
    assert store.live_jobs(), "the job was left alone"


# -- a long solve is not a silent one ------------------------------------------


def test_watch_narrates_progress_facts_on_an_interval(backend, store, view):
    job_id = start_job(backend, store)
    backend.logs[job_id] = b"Time = 41\nTime = 42\n"
    backend.jobs[job_id] = JobStatus(
        job_id=job_id, name="solve", status="running",
        log_size=len(backend.logs[job_id]),
    )

    wake = watch(backend, store, view, NullReader(), narrate_every_s=0.001)

    assert wake.kind == "narrate"
    assert "solve" in wake.text
    assert "Time = 42" in wake.text
    assert "nothing has ended" in wake.text
    assert store.session.jobs[job_id].status == "running", "narrating ended nothing"


def test_a_narration_offers_no_next_step(backend, store, view):
    job_id = start_job(backend, store)
    backend.logs[job_id] = b"Time = 7\n"
    backend.jobs[job_id] = JobStatus(
        job_id=job_id, name="solve", status="running", log_size=9
    )

    wake = watch(backend, store, view, NullReader(), narrate_every_s=0.001)

    lowered = wake.text.lower()
    for suggestion in ("you should", "try ", "recommend", "next step", "consider"):
        assert suggestion not in lowered


def test_progress_facts_reach_the_stage_line_every_poll(backend, store, view):
    import time as _time

    job_id = start_job(backend, store)
    backend.logs[job_id] = b"Time = 7\n"
    backend.jobs[job_id] = JobStatus(
        job_id=job_id, name="solve", status="running", log_size=9
    )

    watch(backend, store, view, NullReader(), deadline=_time.monotonic() + 0.2)

    shown = " ".join(view.stages)
    assert "solve" in shown
    assert "Time = 7" in shown


def test_the_watch_deadline_is_not_replaced_by_the_poll_pause(backend, store, view):
    """--max-wait 60 used to become a thirty-second wait: the poll pause reused the
    deadline variable and clobbered the caller's bound."""
    import time as _time

    start_job(backend, store)
    began = _time.monotonic()
    wake = watch(backend, store, view, NullReader(), deadline=began + 0.35)

    assert wake.kind == "timeout"
    assert _time.monotonic() - began >= 0.3, "it waited to the caller's bound"


# -- one message, however many lines it has ------------------------------------


class FakeStdin:
    """Hands lines to the reader thread the way a terminal or a paste would."""

    def __init__(self, script):
        self._script = list(script)

    def readline(self):
        if not self._script:
            time.sleep(0.05)
            return ""
        delay, line = self._script.pop(0)
        if delay:
            time.sleep(delay)
        return line


def reader_over(monkeypatch, script):
    monkeypatch.setattr("openreynolds.watch.sys.stdin", FakeStdin(script))
    return LineReader()


def test_a_pasted_paragraph_is_one_message(monkeypatch):
    """A live run split four messages into six turns this way: the agent answered the
    first sentence while the rest of the paragraph landed on it as interruptions, and
    the user's actual question went unanswered."""
    reader = reader_over(
        monkeypatch,
        [
            (0, "Whoa - hang on, that is not what I asked for.\n"),
            (0, "\n"),
            (0, "Let's keep it simple: say 5 m/s, sharp mitre.\n"),
        ],
    )

    message = reader.get(timeout=5)

    assert message is not None
    assert "Whoa - hang on" in message
    assert "5 m/s" in message, "the second paragraph is part of the same message"
    assert message.count("\n") == 2, "and the blank line between them is kept"


def test_messages_typed_apart_stay_apart(monkeypatch):
    reader = reader_over(
        monkeypatch,
        [(0, "how many cells?\n"), (0.6, "actually, never mind\n")],
    )

    assert reader.get(timeout=5) == "how many cells?"
    assert reader.get(timeout=5) == "actually, never mind"


def test_end_of_input_is_still_end_of_input(monkeypatch):
    reader = reader_over(monkeypatch, [(0, "one last thing\n")])

    assert reader.get(timeout=5) == "one last thing"
    assert reader.get(timeout=5) is None


def test_an_end_of_input_arriving_mid_paste_is_kept_for_the_next_read(monkeypatch):
    """EOF ends the session, not the message that happened to be in flight."""
    reader = reader_over(monkeypatch, [(0, "first line\n"), (0, "second line\n")])

    assert reader.get(timeout=5) == "first line\nsecond line"
    assert reader.get(timeout=5) is None


def test_polling_coalesces_the_same_way(monkeypatch):
    """Between tool calls the reader is polled, not waited on -- a paste must not be
    half-delivered there either."""
    reader = reader_over(monkeypatch, [(0, "stop the fine one\n"), (0, "coarse is enough\n")])
    time.sleep(0.3)

    assert reader.poll() == "stop the fine one\ncoarse is enough"
    assert reader.poll() in (NOTHING, None)


# -- waiting on a job is still waiting on the user ------------------------------


def test_watch_mode_shows_a_prompt(backend, store, view, fast_polling):
    """Waiting on a job is still waiting on the user - they can speak at any moment
    and be heard. Without a prompt the screen looks locked, and a twenty-minute solve
    reads as a session that has stopped responding."""
    store.record_job("job-1", cmd="simpleFoam", name="solve")
    backend.jobs["job-1"] = JobStatus(job_id="job-1", status="running", name="solve")

    watch(backend, store, view, ScriptedReader(["how is it going?"]))

    assert view.prompts == 1, "the user was shown it is their turn"


def test_a_run_nobody_is_watching_is_not_invited_to_type(backend, store, view, fast_polling):
    """A prompt in a one-shot run is a lie: there is nobody there and no way in."""
    store.record_job("job-1", cmd="simpleFoam", name="solve")
    backend.jobs["job-1"] = JobStatus(
        job_id="job-1", status="exited", name="solve", end_reason="completed"
    )

    watch(backend, store, view, NullReader())

    assert view.prompts == 0


def test_the_watch_line_is_not_repeated_for_the_same_jobs(console):
    """Watch mode is re-entered after every local command, and repeating the line
    each time is how a screen fills up with nothing."""
    plain = ConsoleView(console)
    plain.watching(["solve"])
    plain.watching(["solve"])
    plain.watching(["solve", "mesh"])

    assert plain._watching == ["solve", "mesh"]
