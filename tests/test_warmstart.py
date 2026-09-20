"""The conversation starts while the workspace starts.

A hosted workspace takes seconds to minutes to come up, and every session used to
wait for it before saying a word: no header, no model, nothing for the person to do.
Now `hosted.reserve` settles WHICH workspace in a fraction of a second, the start
runs on a thread, and a `PendingBackend` stands in for the workspace until it is
there -- every tool call waits for it, nothing else does. The header goes out at
once (which is what the hosted page turns into "running"), the model is told the
workspace is coming and what the wait is good for, and a note follows when it is
here. These tests hold each of those claims to the code.
"""

from __future__ import annotations

import re
import threading
import time

import pytest

from conftest import (
    FakeBackend,
    RecordingView,
    ScriptedReader,
    install_model,
    message,
    reserved,
    text_block,
    tool_block,
)
from openreynolds import cli
from openreynolds.backend import hosted as hosted_mod
from openreynolds.backend import pending as pending_mod
from openreynolds.backend.base import BackendError, ExecResult
from openreynolds.backend.pending import PendingBackend
from openreynolds.config import Config
from openreynolds.loop import Loop
from openreynolds.progress import Tracker
from test_prompt import IMPERATIVE_PATTERNS


# -- reserve, and the starter --------------------------------------------------------


class _SlowService:
    """A service whose start takes as long as the test says: `release` lets it answer."""

    def __init__(self, rows=None, reply=None):
        self.rows = rows if rows is not None else [{"id": "iid-1", "status": "stopped"}]
        self.reply = reply or {"id": "iid-1", "status": "running", "started_new": True}
        self.release = threading.Event()
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.created = 0
        self.closed = False

    def list_instances(self):
        return list(self.rows)

    def create_instance(self):
        self.created += 1
        return "iid-new"

    def start_instance(self, instance_id):
        self.started.append(instance_id)
        self.release.wait(5.0)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    def stop_instance(self, instance_id):
        self.stopped.append(instance_id)
        return {"id": instance_id, "status": "stopped"}

    def close(self):
        self.closed = True


def _reserve(monkeypatch, service):
    monkeypatch.setattr(hosted_mod, "FoamdClient", lambda *a, **k: service)
    return hosted_mod.reserve("https://svc.example", "of_live_test")


def test_reserve_answers_before_the_start_does(monkeypatch):
    """The whole point: which workspace is known in a fraction of a second, and the
    start -- seconds to minutes -- has not even been asked for until the starter is
    kicked, and blocks nothing when it is."""
    service = _SlowService()
    t0 = time.monotonic()
    client, instance_id, starter = _reserve(monkeypatch, service)
    assert time.monotonic() - t0 < 1.0
    assert client is service and instance_id == "iid-1"
    assert service.started == [], "reserve started nothing"
    assert starter.instances_held == 1 and starter.listed_as_running is False

    starter.start()
    deadline = time.monotonic() + 2.0
    while not service.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert service.started == ["iid-1"], "the kick sent the start"
    assert not starter.done, "and it is still under way"

    service.release.set()
    backend = starter.result(timeout=5.0)
    assert backend.instance_id == "iid-1"
    assert backend.was_already_running is False, "the start reply says this call built it"
    assert backend.instances_held == 1


def test_the_start_reply_still_decides_who_owns_the_workspace(monkeypatch):
    service = _SlowService(reply={"id": "iid-1", "status": "running", "started_new": False})
    _client, _iid, starter = _reserve(monkeypatch, service)
    service.release.set()
    assert starter.result(timeout=5.0).was_already_running is True


def test_a_start_that_fails_fails_result_with_the_services_own_error(monkeypatch):
    service = _SlowService(reply=BackendError("no capacity", code="unavailable", status=503))
    _client, _iid, starter = _reserve(monkeypatch, service)
    service.release.set()
    with pytest.raises(BackendError) as raised:
        starter.result(timeout=5.0)
    assert raised.value.code == "unavailable"


def test_cancel_only_succeeds_before_the_start_was_sent(monkeypatch):
    service = _SlowService()
    _client, _iid, starter = _reserve(monkeypatch, service)
    assert starter.cancel() is True
    with pytest.raises(BackendError) as raised:
        starter.result(timeout=1.0)
    assert raised.value.code == "start_cancelled"
    assert service.started == [], "a cancelled start is never sent"

    service = _SlowService()
    _client, _iid, starter = _reserve(monkeypatch, service)
    starter.start()
    assert starter.cancel() is False, "the service has the request; it cannot be called back"
    service.release.set()
    starter.result(timeout=5.0)


def test_acquire_is_reserve_plus_the_wait(monkeypatch):
    """The blocking shape the read-only commands still use: the same triple, the same
    ownership answer, and the client closed on a failed start."""
    service = _SlowService()
    service.release.set()
    monkeypatch.setattr(hosted_mod, "FoamdClient", lambda *a, **k: service)
    backend, client, instance_id = hosted_mod.acquire("https://svc.example", "of_live_test")
    assert (backend.instance_id, client, instance_id) == ("iid-1", service, "iid-1")
    assert backend.was_already_running is False

    failing = _SlowService(reply=BackendError("down", code="unavailable"))
    failing.release.set()
    monkeypatch.setattr(hosted_mod, "FoamdClient", lambda *a, **k: failing)
    with pytest.raises(BackendError):
        hosted_mod.acquire("https://svc.example", "of_live_test")
    assert failing.closed, "the client is closed when the start fails, as it always was"


def test_reserve_creates_a_row_when_the_account_holds_none_and_closes_on_failure(monkeypatch):
    service = _SlowService(rows=[])
    _client, instance_id, starter = _reserve(monkeypatch, service)
    assert instance_id == "iid-new" and service.created == 1 and starter.instances_held == 1

    class Refusing(_SlowService):
        def list_instances(self):
            raise BackendError("no", code="unreachable")

    refusing = Refusing()
    monkeypatch.setattr(hosted_mod, "FoamdClient", lambda *a, **k: refusing)
    with pytest.raises(BackendError):
        hosted_mod.reserve("https://svc.example", "of_live_test")
    assert refusing.closed


# -- the pending backend --------------------------------------------------------------


class _Stoppable(FakeBackend):
    def __init__(self):
        super().__init__()
        self.stopped = 0
        self.closed = 0
        self.was_already_running = False
        self.instances_held = 2

    def shutdown(self):
        self.stopped += 1

    def close(self):
        self.closed += 1


def test_a_call_waits_for_the_workspace_and_then_runs_on_it():
    live = _Stoppable()
    pending = PendingBackend("iid-1")
    answered = []

    def call():
        answered.append(pending.exec("nproc"))

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    time.sleep(0.1)
    assert not answered and live.execs == [], "blocked: nothing ran yet"
    pending.resolve(live)
    thread.join(2.0)
    assert live.execs == ["nproc"]
    assert answered[0].output == "8\n"
    assert pending.ready() and pending.exec("true") is live.exec_result, "and every later call goes straight through"


def test_a_poll_made_while_the_workspace_is_starting_answers_idle_at_once():
    """`background=True` means "only if the workspace is already up", by the protocol's
    own words; a mirror cycle must never be what everyone waits behind."""
    pending = PendingBackend("iid-1")
    t0 = time.monotonic()
    result = pending.exec("find /work", background=True)
    assert time.monotonic() - t0 < 0.5
    assert result.idle is True and result.output == ""


def test_a_failed_start_fails_every_call_with_the_starts_own_error():
    pending = PendingBackend("iid-1")
    pending.fail(BackendError("no capacity", code="unavailable", status=503))
    with pytest.raises(BackendError) as raised:
        pending.exec("nproc")
    assert raised.value.code == "unavailable"
    assert pending.failed and not pending.ready()
    pending.shutdown()  # nothing to stop, nothing raised


def test_the_wait_is_bounded_and_says_so(monkeypatch):
    monkeypatch.setattr(pending_mod, "WAIT_TICK_S", 0.01)
    pending = PendingBackend("iid-1")
    with pytest.raises(BackendError) as raised:
        pending.wait(timeout=0.05)
    assert raised.value.code == "workspace_not_ready"


def test_a_blocked_call_says_it_is_waiting(monkeypatch):
    """The view hears "waiting for the workspace, N s" while a tool is held up, so a
    call taking a minute reads as the wait it is and not as a slow machine."""
    monkeypatch.setattr(pending_mod, "WAIT_TICK_S", 0.02)
    heard = []
    pending = PendingBackend("iid-1", on_wait=heard.append)
    live = FakeBackend()
    live.dirs["/work"] = []
    threading.Timer(0.15, pending.resolve, args=(live,)).start()
    assert pending.stat("/work").is_dir
    assert heard, "the wait was said"
    assert all(isinstance(s, float) and s >= 0 for s in heard)


def test_what_is_known_before_the_start_and_what_the_start_settles():
    pending = PendingBackend("iid-1", instances_held=3, was_already_running=True)
    assert pending.instance_id == "iid-1"
    assert (pending.instances_held, pending.was_already_running) == (3, True)
    live = _Stoppable()
    pending.resolve(live)
    assert (pending.instances_held, pending.was_already_running) == (2, False), (
        "the live backend's answer, from the start reply, replaces the listing's"
    )
    assert pending.elapsed_s >= 0


def test_shutdown_waits_for_the_workspace_and_then_stops_it():
    live = _Stoppable()
    pending = PendingBackend("iid-1")
    threading.Timer(0.1, pending.resolve, args=(live,)).start()
    pending.shutdown()
    assert live.stopped == 1, "stopped exactly once, once it was there to stop"


def test_close_closes_whichever_side_is_there():
    closed = []
    pending = PendingBackend("iid-1", closer=lambda: closed.append("client"))
    pending.close()
    assert closed == ["client"], "unresolved: the client under it is what there is to close"
    live = _Stoppable()
    resolved = PendingBackend("iid-1", closer=lambda: closed.append("client"))
    resolved.resolve(live)
    resolved.close()
    assert live.closed == 1 and closed == ["client"]


def test_the_pending_backend_is_a_backend():
    from openreynolds.backend.base import Backend

    assert isinstance(PendingBackend("iid-1"), Backend)


def test_the_pending_module_knows_nothing_about_transport():
    from pathlib import Path

    from test_negative_obligation import FORBIDDEN

    source = (Path(pending_mod.__file__)).read_text(encoding="utf-8")
    for token in FORBIDDEN:
        assert token.lower() not in source.lower(), f"pending.py references {token!r}"


# -- the loop: a fact from another thread --------------------------------------------


def test_a_posted_fact_rides_in_the_next_request(loop):
    """`post` is `tell` from a thread that is not the loop's; it is handed over at
    the next point the loop would have said it itself."""
    fake = install_model(loop, [message([text_block("ok")])])
    loop.post("The workspace is ready.")
    loop.say("hello")
    loop.run()
    sent = fake.calls[-1]["messages"]
    said = [m for m in sent if "workspace is ready" in str(m.get("content"))]
    assert said, "the posted fact reached the model"
    assert sent[-1] is not sent[0], "after the user's own message"


def test_a_fact_posted_mid_turn_rides_with_the_tool_results(loop, backend):
    fake = install_model(loop, [
        message([tool_block("bash", {"cmd": "nproc"})], stop_reason="tool_use"),
        message([text_block("done")]),
    ])
    original = backend.exec

    def exec_and_post(cmd, cwd=None, timeout_s=120, *, background=False):
        loop.post("The workspace is ready (12 s after the session began).")
        return original(cmd, cwd, timeout_s, background=background)

    backend.exec = exec_and_post
    loop.say("go")
    loop.run()
    results = fake.calls[-1]["messages"][-1]["content"]
    assert any(
        isinstance(b, dict) and b.get("type") == "text" and "workspace is ready" in b["text"]
        for b in results
    ), "the note rode in the tool-results message"


# -- the tracker's bar ----------------------------------------------------------------


def test_the_bar_says_the_workspace_is_starting_and_what_a_tool_is_waiting_for(view):
    tracker = Tracker(view)
    tracker.workspace_starting(30.0)
    snap = tracker.snapshot()
    assert snap.phase == "starting" and snap.headline.startswith("workspace starting")
    assert "usually about 30 s" in snap.detail and snap.busy
    tracker.begin("thinking")
    assert tracker.snapshot().phase == "thinking", "the model's own work stays the headline"
    tracker.begin("tool", "bash", cmd="nproc")
    assert tracker.snapshot().headline.startswith("bash waiting for the workspace")
    tracker.workspace_ready()
    tracker.idle()
    assert tracker.snapshot().phase == "waiting"
    assert not tracker.workspace_pending


# -- the briefing -----------------------------------------------------------------------


def _brief(store, backend, **kwargs):
    defaults = dict(resuming=False, interactive=True, browser=None, starting_eta_s=30.0)
    defaults.update(kwargs)
    return cli._situation_brief(store, backend, **defaults)


def test_the_briefing_says_the_workspace_is_coming_and_asks_nothing_of_it(store, backend):
    store.session.instance_id = "iid-1"
    brief = _brief(store, backend)
    assert "still starting" in brief and "usually about 30 seconds" in brief
    assert "waits for it rather than failing" in brief
    assert "geometry" in brief and "boundary conditions" in brief and "upload" in brief
    assert backend.execs == [], "nothing was asked of a workspace that is not there"
    assert "person is at the terminal" in brief


def test_a_resume_ahead_of_its_workspace_repeats_the_record_and_re_reads_it_when_ready(store, backend):
    """A resumed study's running jobs are re-read from the workspace at the briefing;
    ahead of the workspace the briefing can only say what the record says, and the
    ready note asks the workspace."""
    from openreynolds.backend.base import JobStatus
    from openreynolds.store import JobRecord

    store.session.jobs["job-1"] = JobRecord(job_id="job-1", name="solve", cmd="simpleFoam", status="running")
    backend.jobs["job-1"] = JobStatus(job_id="job-1", status="exited", name="solve", exit_code=0,
                                      end_reason="completed")
    brief = _brief(store, backend, resuming=True)
    assert "Jobs still running:" in brief and "still starting" in brief
    assert backend.execs == []
    note = cli._workspace_ready_note(store, backend, True, None, 40.0, True)
    assert "Jobs that have finished:" in note, "re-read from the workspace once it was there"


def test_a_one_shot_run_ahead_of_its_workspace_is_told_differently(store, backend):
    brief = _brief(store, backend, interactive=False)
    assert "still starting" in brief and "prompt is already here" in brief
    assert "questions the person can answer" not in brief
    assert "non-interactive" in brief


@pytest.mark.parametrize("pattern", IMPERATIVE_PATTERNS)
def test_the_starting_briefing_tells_the_model_nothing_to_do(pattern, store, backend):
    for interactive in (True, False):
        brief = _brief(store, backend, interactive=interactive)
        match = re.search(pattern, brief, re.IGNORECASE)
        assert match is None, f"imperative language in the starting briefing: {match!r}"
    note = cli._workspace_ready_note(store, backend, False, None, 12.0, True)
    assert re.search(pattern, note, re.IGNORECASE) is None


def test_the_ready_note_carries_what_the_briefing_left_out(store, backend):
    from openreynolds.browse import Browser

    store.session.home = "/work/study-test"
    backend.exec_result = ExecResult(0, "", False, None)
    note = cli._workspace_ready_note(store, backend, False, Browser(backend, store), 12.0, True)
    assert note.startswith("The workspace is ready (12 s")
    assert "Your directory is /work/study-test" in note
    assert "This machine has 8 cores" in note
    fallen = cli._workspace_ready_note(store, backend, False, None, 12.0, False)
    assert "could not be made" in fallen and "/work" in fallen


def test_the_default_eta_and_the_environment_knob(monkeypatch):
    monkeypatch.delenv("OPENREYNOLDS_WORKSPACE_ETA_S", raising=False)
    monkeypatch.setenv("OPENREYNOLDS_CONFIG", "nonexistent-config.json")
    assert Config.load().workspace_eta_s == 30.0
    monkeypatch.setenv("OPENREYNOLDS_WORKSPACE_ETA_S", "90")
    assert Config.load().workspace_eta_s == 90.0
    monkeypatch.setenv("OPENREYNOLDS_WORKSPACE_ETA_S", "soon")
    assert Config.load().workspace_eta_s == 30.0, "a hint that is not a number is not worth refusing to start"


# -- a whole session, ahead of its workspace ---------------------------------------------


class _WatchingView(RecordingView):
    """Records, beside each event, whether the workspace was up when it was said."""

    def __init__(self, hold):
        super().__init__()
        self.hold = hold
        self.header_while_starting = None
        self.ready = []
        self.on_tool = None
        self.before_step = None

    def header(self, study_id, instance_id, model, mirror):
        super().header(study_id, instance_id, model, mirror)
        self.header_while_starting = not self.hold.is_set()

    def tool(self, name, summary):
        super().tool(name, summary)
        if self.on_tool is not None:
            self.on_tool()

    def step(self, number, seconds, tool_calls):
        # Called by the loop after a batch of tool calls and before it drains what
        # other threads posted: a hook here is how a test makes "the note was posted
        # while the model was mid-turn" a certainty rather than a race.
        if self.before_step is not None:
            self.before_step()
        super().step(number, seconds, tool_calls)

    def workspace_ready(self, instance_id, seconds):
        self.ready.append((instance_id, seconds))


def _config(tmp_path):
    return Config(
        foamd_url="u", foamd_api_key="k", llm_api_key="sk-test", model="m",
        studies_dir=tmp_path / "studies", capture=False, desk=False, mesh_tool=False,
        mirror_interval_s=0.0, workspace_eta_s=20.0,
    )


def _scripted_loop(responses, made):
    class ScriptedLoop(Loop):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fake = install_model(self, list(responses))
            self.posted = threading.Event()
            made.append(self)

        def post(self, text):
            super().post(text)
            self.posted.set()

    return ScriptedLoop


def test_the_header_goes_out_and_the_model_is_briefed_while_the_workspace_starts(
    tmp_path, monkeypatch, quiet_console
):
    """End to end: the session says who it is at once, the first turn is briefed on a
    workspace that is coming, the first tool call waits for it and then runs, and the
    ready note follows into the thread."""
    monkeypatch.setattr(pending_mod, "WAIT_CEILING_S", 10.0)
    monkeypatch.setattr(pending_mod, "WAIT_TICK_S", 0.05)
    hold = threading.Event()
    backend = _Stoppable()
    ran_while_starting = []
    original = backend.exec

    def exec_recording(cmd, cwd=None, timeout_s=120, *, background=False):
        ran_while_starting.append(not hold.is_set())
        return original(cmd, cwd, timeout_s, background=background)

    backend.exec = exec_recording
    monkeypatch.setattr(cli.hosted, "reserve", reserved(backend, "iid-7", hold=hold))
    loops = []
    monkeypatch.setattr(cli, "Loop", _scripted_loop([
        message([tool_block("bash", {"cmd": "nproc"})], stop_reason="tool_use"),
        message([text_block("The machine has eight cores.")]),
    ], loops))
    view = _WatchingView(hold)
    # The model's first tool call is what lets the workspace come up: so the call
    # was made -- and blocked -- before the workspace was there.
    view.on_tool = lambda: threading.Timer(0.3, hold.set).start()
    view.before_step = lambda: loops[0].posted.wait(5.0)

    def interface(drive):
        drive(view, ScriptedReader(["what is this machine?", "/exit"]))
        return False

    cli.session(_config(tmp_path), study_id=None, instance_id=None, one_shot=None, interface=interface)

    assert view.headers and view.headers[0][1] == "iid-7"
    assert view.header_while_starting is True, "the header did not wait for the workspace"
    (loop,) = loops
    first = str(loop.messages[0]["content"])
    assert "still starting" in first and "usually about 20 seconds" in first
    assert "nproc" in backend.execs, "the tool call ran once the workspace was up"
    assert not any(ran_while_starting), "and never before"
    assert view.ready and view.ready[0][0] == "iid-7", "the view was told, once"
    assert len(view.ready) == 1
    results = loop.fake.calls[-1]["messages"][-1]["content"]
    assert any(
        isinstance(b, dict) and b.get("type") == "text" and "The workspace is ready" in b["text"]
        for b in results
    ), "the ready note rode into the thread with the tool results"
    assert "/work/.toolbox" in [remote for _local, remote in backend.trees], "the toolbox went up first"
    assert any("waiting for the workspace" in s for s in view.stages), "the wait was shown"
    assert backend.stopped == 1, "the ordinary close-down put the workspace down"


def test_ending_during_the_start_puts_the_workspace_down_once_it_is_up(
    tmp_path, monkeypatch, recording_console
):
    """End pressed while the machine is still coming: the start cannot be called back,
    so the close-down waits for it and stops it, exactly once, rather than leaving a
    workspace running for nobody -- and does not pick up or sync a workspace nothing
    ran on."""
    monkeypatch.setattr(pending_mod, "WAIT_CEILING_S", 10.0)
    hold = threading.Event()
    backend = _Stoppable()
    monkeypatch.setattr(cli.hosted, "reserve", reserved(backend, hold=hold))
    monkeypatch.setattr(cli, "Loop", _scripted_loop([message([text_block("hi")])], []))
    view = _WatchingView(hold)

    def interface(drive):
        drive(view, ScriptedReader(["/exit"]))
        threading.Timer(0.3, hold.set).start()
        return False

    cli.session(_config(tmp_path), study_id=None, instance_id=None, one_shot=None, interface=interface)

    assert backend.stopped == 1
    said = recording_console.getvalue()
    assert "still starting when the session ended" in said
    assert "instance stopped" in said
    assert backend.fetched == [], "nothing was picked up from a workspace nothing ran on"


def test_ending_during_the_start_leaves_a_workspace_somebody_else_had_up(
    tmp_path, monkeypatch, quiet_console
):
    monkeypatch.setattr(pending_mod, "WAIT_CEILING_S", 10.0)
    hold = threading.Event()
    backend = _Stoppable()
    backend.was_already_running = True
    monkeypatch.setattr(cli.hosted, "reserve", reserved(backend, hold=hold, listed_as_running=True))
    monkeypatch.setattr(cli, "Loop", _scripted_loop([message([text_block("hi")])], []))

    def interface(drive):
        drive(_WatchingView(hold), ScriptedReader(["/exit"]))
        threading.Timer(0.2, hold.set).start()
        return False

    cli.session(_config(tmp_path), study_id=None, instance_id=None, one_shot=None, interface=interface)

    assert backend.stopped == 0


def test_ending_during_the_start_with_keep_alive_leaves_it_to_the_reaper(
    tmp_path, monkeypatch, quiet_console
):
    """The hosted runner passes `keep_alive` above a session cap of one: the reaper owns
    the workspace's lifetime there, the start this session made included."""
    monkeypatch.setattr(pending_mod, "WAIT_CEILING_S", 10.0)
    hold = threading.Event()
    backend = _Stoppable()
    monkeypatch.setattr(cli.hosted, "reserve", reserved(backend, hold=hold))
    monkeypatch.setattr(cli, "Loop", _scripted_loop([message([text_block("hi")])], []))

    def interface(drive):
        drive(_WatchingView(hold), ScriptedReader(["/exit"]))
        hold.set()
        return False

    t0 = time.monotonic()
    cli.session(_config(tmp_path), study_id=None, instance_id=None, one_shot=None,
                interface=interface, keep_alive=True)

    assert backend.stopped == 0
    assert time.monotonic() - t0 < 5.0


def test_a_start_that_fails_tells_the_view_and_the_model_and_keeps_the_conversation(
    tmp_path, monkeypatch, quiet_console
):
    monkeypatch.setattr(pending_mod, "WAIT_CEILING_S", 10.0)
    backend = _Stoppable()
    monkeypatch.setattr(cli.hosted, "reserve", reserved(
        backend, error=BackendError("no capacity", code="unavailable", status=503)))
    loops = []
    monkeypatch.setattr(cli, "Loop", _scripted_loop([
        message([tool_block("bash", {"cmd": "nproc"})], stop_reason="tool_use"),
        message([text_block("The workspace is not available; here is what I can plan.")]),
    ], loops))
    view = _WatchingView(threading.Event())

    def failure_has_been_said():
        deadline = time.monotonic() + 5.0
        while not any("could not be started" in n for n in view.notices) and time.monotonic() < deadline:
            time.sleep(0.01)

    # The tool call is made after the start has failed -- which is the ordinary
    # order, the failure being immediate here, and held to so the test cannot race.
    view.on_tool = failure_has_been_said

    def interface(drive):
        drive(view, ScriptedReader(["go", "/exit"]))
        return False

    cli.session(_config(tmp_path), study_id=None, instance_id=None, one_shot=None, interface=interface)

    assert any("could not be started" in n for n in view.notices)
    assert view.ready == [], "a workspace that never came is never said to be ready"
    (loop,) = loops
    assert backend.execs == [], "nothing ran"
    assert any("unavailable" in e for e in view.tool_errors), "the tool call failed with the start's own error"
    assert loop.fake.calls, "and the conversation went on"
    assert backend.stopped == 0


def test_the_stream_json_reader_hears_workspace_ready(tmp_path, monkeypatch):
    import io
    import json
    import sys

    from test_jsonview import FakeLoop

    # `--output-format stream-json` moves the module console to stderr for the life
    # of the process; the patch puts it back for the tests after this one.
    monkeypatch.setattr(cli, "console", cli.console)
    backend = FakeBackend()
    monkeypatch.setattr(cli.hosted, "reserve", reserved(backend))
    monkeypatch.setattr(cli, "Loop", FakeLoop)
    sink = io.StringIO()
    monkeypatch.setattr(sys, "stdout", sink)

    cli.session(_config(tmp_path), study_id=None, instance_id=None, one_shot="go",
                output_format="stream-json")

    rows = [json.loads(line) for line in sink.getvalue().splitlines() if line.strip()]
    kinds = [row["type"] for row in rows]
    assert kinds[0] == "session_start"
    assert "workspace_ready" in kinds
    ready = next(row for row in rows if row["type"] == "workspace_ready")
    assert ready["instance_id"] == "iid-1" and "seconds" in ready


@pytest.fixture
def quiet_console(monkeypatch):
    import os

    from rich.console import Console

    monkeypatch.setattr(cli, "console", Console(file=open(os.devnull, "w"), force_terminal=False))


@pytest.fixture
def recording_console(monkeypatch):
    import io

    from rich.console import Console

    sink = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=sink, force_terminal=False, width=200))
    return sink


@pytest.fixture
def loop(backend, store, view):
    from openreynolds.tools import ToolContext

    cfg = Config(foamd_url="u", foamd_api_key="k", llm_api_key="sk-test", model="m",
                 studies_dir=store.dir.parent)
    ctx = ToolContext(backend=backend, store=store, max_output=10_000)
    return Loop(cfg, ctx, store, view)

