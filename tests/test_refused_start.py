"""A workspace that cannot start, and a study resumed afterwards (OpenReynolds#91).

On staging a Starter account started study B while study A was running. B's start was
refused (403 plan_limit), the model was asked anyway and wrote a full plan for a case
nothing could run; resumed later, the model had lost the request entirely ("This study
folder is empty, and this thread does not yet identify the geometry or flow").

Change 1: while the session's workspace start has failed the model is not called for
the person's message. The refusal is shown, the message stays in the transcript.

Change 2: a resumed thread is briefed with the study's recent conversation, the
person's messages that never got a reply marked as awaiting an answer, and the agent
answers them without being asked again.
"""

from __future__ import annotations

import io
import json
import threading
from types import SimpleNamespace

import pytest
from rich.console import Console

from openreynolds import cli, watch
from openreynolds.backend import pending as pending_mod
from openreynolds.backend.base import BackendError
from openreynolds.config import Config
from openreynolds.loop import Loop
from openreynolds.store import Store

from conftest import (FakeBackend, ReadyStarter, RecordingView, ScriptedReader,
                      install_model, message, text_block)

PLAN_LIMIT = BackendError(
    "max_concurrent 2 is over the limit: the Starter plan runs 1 study at a time",
    code="plan_limit", status=403,
)
CAVITY = "2D lid-driven cavity, Re 100, coarse mesh"


class _Platform:
    """The service through the client `reserve` hands back: a study row, the messages
    posted to it, and a transcript to read back."""

    def __init__(self, transcript=None, row=None):
        self.posted: list[dict] = []
        self.transcript = list(transcript or [])
        self.row = row
        self.reads: list[dict] = []

    def create_study(self, title, instance_id, **extra):
        return "remote-1"

    def post_messages(self, study_id, messages):
        self.posted.extend(messages)

    def get_study(self, study_id):
        return self.row

    def list_messages(self, study_id, after=-1, limit=999, newest=False):
        """foamd's route: with `newest`, the LAST `limit` rows after `after`, oldest
        first, `more` meaning earlier ones were left out. `ignores_newest` is a
        service from before that parameter, which answers the first rows."""
        self.reads.append({"study_id": study_id, "after": after, "limit": limit,
                           "newest": newest})
        rows = sorted((r for r in self.transcript if r["seq"] > after), key=lambda r: r["seq"])
        if newest and not self.ignores_newest:
            return {"messages": rows[-limit:], "more": len(rows) > limit}
        return {"messages": rows[:limit], "more": len(rows) > limit}

    ignores_newest = False

    def __getattr__(self, name):
        return lambda *a, **k: None


def _session(monkeypatch, tmp_path, responses, *, lines=(), one_shot=None, study_id=None,
             starter=None, platform=None, on_call=None, capture=False):
    """One session end to end against a scripted model and a stand-in service."""
    monkeypatch.setattr(pending_mod, "WAIT_CEILING_S", 10.0)
    monkeypatch.delenv("OPENREYNOLDS_LOCAL", raising=False)
    monkeypatch.delenv("OPENREYNOLDS_MODE", raising=False)
    said = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=said, width=200, force_terminal=False))
    backend = FakeBackend()
    platform = platform or _Platform()
    starter = starter or ReadyStarter(backend, "iid-9")
    loops: list[Loop] = []

    def reserve(url, key, iid=None, **kwargs):
        return platform, "iid-9", starter

    class ScriptedLoop(Loop):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fake = install_model(self, list(responses))
            if on_call is not None:
                original = self.fake.stream

                def stream(**kw):
                    on_call(len(self.fake.calls))
                    return original(**kw)

                self.fake.stream = stream
            loops.append(self)

    monkeypatch.setattr(cli.hosted, "reserve", reserve)
    monkeypatch.setattr(cli, "Loop", ScriptedLoop)
    cfg = Config(foamd_url="u", foamd_api_key="k", llm_api_key="sk-test", model="m",
                 studies_dir=tmp_path / "studies", capture=capture, desk=False,
                 mesh_tool=False, mirror_interval_s=0.0, workspace_eta_s=20.0)
    view = RecordingView()
    reader = ScriptedReader(list(lines))

    def interface(drive):
        drive(view, reader)
        return False

    outcome = cli.session(cfg, study_id=study_id, instance_id=None, one_shot=one_shot,
                          interface=interface)
    loop = loops[0]
    return SimpleNamespace(loop=loop, calls=loop.fake.calls, view=view, outcome=outcome,
                           store=loop.store, said=said.getvalue(), platform=platform,
                           backend=backend)


def _rows(store):
    path = store.dir / "messages.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _first_request_text(call) -> str:
    """Everything the model was sent as text in one request, joined."""
    parts = []
    for m in call["messages"]:
        content = m["content"]
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.extend(b.get("text", "") for b in content if isinstance(b, dict))
    return "\n".join(parts)


def _refused_starter(backend=None):
    return ReadyStarter(backend or FakeBackend(), "iid-9", error=PLAN_LIMIT)


# -- Change 1: no agent turn when the workspace cannot start -------------------------


def test_a_refused_start_calls_no_model_keeps_the_message_and_shows_the_refusal(
        monkeypatch, tmp_path):
    run = _session(monkeypatch, tmp_path, [message([text_block("a plan nobody can run")])],
                   lines=[CAVITY], starter=_refused_starter())

    assert run.calls == [], "the model must not be asked while the start has failed"
    rows = _rows(run.store)
    assert [r["content"] for r in rows if r["role"] == "user"] == [CAVITY]
    assert not [r for r in rows if r["role"] == "assistant"]
    # The refusal is the transcript's event row and the page's notice.
    assert any(r["role"] == "event" and "could not be started" in str(r["content"])
               for r in rows)
    assert any("max_concurrent 2" in str(r["content"]) for r in rows if r["role"] == "event")
    assert any("could not be started" in n for n in run.view.notices)
    assert any("not asked" in i for i in run.view.infos)
    assert "The agent was not asked" in run.said


def test_every_later_message_is_held_back_too_and_the_session_waits_for_input(
        monkeypatch, tmp_path):
    """An interactive session keeps waiting after the refusal; the person may End it."""
    run = _session(monkeypatch, tmp_path, [message([text_block("no")])],
                   lines=[CAVITY, "continue"], starter=_refused_starter())

    assert run.calls == []
    assert [r["content"] for r in _rows(run.store) if r["role"] == "user"] == [CAVITY, "continue"]
    assert run.view.prompts >= 3, "back at the prompt after each message, until the End"


def test_a_one_shot_run_exits_non_zero_with_the_refusal(monkeypatch, tmp_path):
    run = _session(monkeypatch, tmp_path, [message([text_block("plan")])],
                   one_shot=CAVITY, starter=_refused_starter())

    assert run.calls == []
    assert run.outcome == "refused"
    assert cli.ONE_SHOT_EXIT_CODES["refused"] != 0
    assert "max_concurrent 2" in run.said


def test_a_start_still_pending_lets_the_model_run_ahead(monkeypatch, tmp_path):
    """The model runs ahead of a workspace that is still coming, on purpose."""
    monkeypatch.setattr(cli, "ADMISSION_WAIT_S", 0.05)
    hold = threading.Event()
    starter = ReadyStarter(FakeBackend(), "iid-9", hold=hold)
    up_when_asked: list[bool] = []

    def on_call(n):
        up_when_asked.append(hold.is_set())
        hold.set()  # and the workspace comes up afterwards

    run = _session(monkeypatch, tmp_path, [message([text_block("assuming a square cavity")])],
                   lines=[CAVITY], starter=starter, on_call=on_call)

    assert up_when_asked == [False], "asked while the workspace was still coming"
    assert len(run.calls) == 1
    assert run.loop.start_refusal is None


def test_a_start_that_fails_during_a_turn_holds_back_the_next_message(monkeypatch, tmp_path):
    """The turn already running is left as it is; the next message is not sent."""
    hold = threading.Event()

    class FailsDuringTheTurn(ReadyStarter):
        @property
        def elapsed_s(self):
            # Long started when the first message comes (the model runs ahead at
            # once); just answered when the second does, so its gate waits for the
            # thread that fails the promise rather than racing it.
            return 0.0 if hold.is_set() else 99.0

    starter = FailsDuringTheTurn(FakeBackend(), "iid-9", hold=hold, error=PLAN_LIMIT)
    monkeypatch.setattr(cli, "ADMISSION_WAIT_S", 5.0)

    def on_call(n):
        if n == 0:  # asked before the call is counted: this is the first
            hold.set()  # the refusal comes back while the first turn is with the model

    run = _session(monkeypatch, tmp_path, [message([text_block("assuming a square cavity")])],
                   lines=[CAVITY, "continue"], starter=starter, on_call=on_call)

    assert len(run.calls) == 1, "the first turn ran ahead; the second was held back"
    assert [r["content"] for r in _rows(run.store) if r["role"] == "user"] == [CAVITY, "continue"]
    assert run.loop.start_refusal and "max_concurrent 2" in run.loop.start_refusal


def test_a_refused_start_is_waited_for_before_the_first_turn(monkeypatch, tmp_path):
    """The first turn waits for the start's answer, so a refusal is caught before the
    model is asked even when the message was already queued as the session began."""
    hold = threading.Event()
    starter = ReadyStarter(FakeBackend(), "iid-9", hold=hold, error=PLAN_LIMIT)
    threading.Timer(0.2, hold.set).start()
    run = _session(monkeypatch, tmp_path, [message([text_block("plan")])],
                   lines=[CAVITY], starter=starter)

    assert run.calls == []


def test_a_paused_study_asks_for_the_ceiling_and_not_the_model(monkeypatch, tmp_path):
    """A study paused at its ceiling is a refusal too: nobody can raise it in a one-shot
    run, so the run stops on it without a model call."""
    paused = BackendError("the study is paused at its spending ceiling", code="study_paused",
                          status=409)
    run = _session(monkeypatch, tmp_path, [message([text_block("plan")])],
                   one_shot=CAVITY, starter=ReadyStarter(FakeBackend(), "iid-9", error=paused))

    assert run.calls == []
    assert run.outcome == "refused"


# -- Change 2: the resume briefing carries the recent conversation -------------------


def _left_refused(tmp_path, *extra_rows):
    """A study whose first session was refused its workspace: briefing, the person's
    request, the refusal -- and no reply. Written as `Store` writes it."""
    store = Store(tmp_path / "studies", "20261110-130725-b0b0")
    store.session.instance_id = "iid-9"
    store.session.solver = "openfoam"
    store.session.home = "/work/20261110-130725-b0b0"
    store.save()
    store.append_message("event", "study 20261110-130725-b0b0 on instance 27fe.")
    store.append_message("user", CAVITY)
    store.append_message("event", "The workspace could not be started (plan_limit (403)).")
    for role, content in extra_rows:
        store.append_message(role, content)
    return store.session.study_id


def test_a_resume_after_a_refused_start_is_briefed_with_the_request_and_answers_it(
        monkeypatch, tmp_path):
    study = _left_refused(tmp_path)
    run = _session(monkeypatch, tmp_path,
                   [message([text_block("Setting up the 2D lid-driven cavity at Re 100.")])],
                   lines=[], study_id=study)

    assert len(run.calls) == 1, "the agent answers what it owes without being asked again"
    briefing = _first_request_text(run.calls[0])
    assert CAVITY in briefing
    assert f"[the person, " in briefing and watch.AWAITING in briefing
    assert "Answer it now" in briefing
    # The harness's events are not quoted as conversation.
    quoted = briefing.split(watch.CONVERSATION_OPEN)[1].split(watch.CONVERSATION_CLOSE)[0]
    assert "could not be started" not in quoted and "on instance 27fe" not in quoted
    assert any(r["role"] == "assistant" and "lid-driven" in r["content"]
               for r in _rows(run.store))


def test_the_resumed_sessions_own_rows_are_not_briefed_back(monkeypatch, tmp_path):
    study = _left_refused(tmp_path, ("assistant", "Assuming a square cavity of side 1 m."))
    run = _session(monkeypatch, tmp_path, [message([text_block("ok")])],
                   lines=["now refine the mesh"], study_id=study)

    briefing = run.calls[0]["messages"][0]["content"]
    assert CAVITY in briefing and "square cavity of side 1 m" in briefing
    assert "now refine the mesh" not in briefing
    assert watch.AWAITING not in briefing, "the request was answered"
    assert briefing.count("study 20261110-130725-b0b0 on instance") == 1


def test_a_resume_whose_start_is_refused_again_still_owes_the_answer(monkeypatch, tmp_path):
    """The staging sequence: refused, resumed, refused again. Nothing is spent, and the
    resume after that is still briefed with both messages awaiting an answer."""
    study = _left_refused(tmp_path)
    run = _session(monkeypatch, tmp_path, [message([text_block("plan")])],
                   lines=["continue"], study_id=study, starter=_refused_starter())
    assert run.calls == []

    again = _session(monkeypatch, tmp_path, [message([text_block("Setting up the cavity.")])],
                     lines=[], study_id=study)
    briefing = again.calls[0]["messages"][0]["content"]
    quoted = briefing.split(watch.CONVERSATION_OPEN)[1].split(watch.CONVERSATION_CLOSE)[0]
    assert quoted.count(watch.AWAITING) == 2
    assert CAVITY in quoted and "continue" in quoted
    assert "last 2 messages" in briefing


def test_a_resume_on_a_machine_without_the_transcript_reads_the_platforms(
        monkeypatch, tmp_path):
    """`openreynolds --study <id>` on a laptop that never held the study: no
    `messages.jsonl` here, so the briefing is made from what capture posted."""
    transcript = [
        {"seq": 0, "role": "event", "content": "study x on instance 27fe.",
         "created_at": "2026-11-10T13:07:25+00:00"},
        {"seq": 1, "role": "user", "content": CAVITY, "created_at": "2026-11-10T13:07:25+00:00"},
        {"seq": 2, "role": "tool", "content": {"tool": "exec", "input": {}, "output": "ls"},
         "created_at": "2026-11-10T13:07:26+00:00"},
        {"seq": 3, "role": "event", "content": "The workspace could not be started.",
         "created_at": "2026-11-10T13:07:25+00:00"},
    ]
    platform = _Platform(transcript, row={"id": "remote-7", "home": "/work/x",
                                          "instance_id": "iid-9", "title": "cavity"})
    run = _session(monkeypatch, tmp_path, [message([text_block("Setting up the cavity.")])],
                   lines=[], study_id="20261110-130725-cafe", platform=platform)

    assert platform.reads and platform.reads[0]["study_id"] == "remote-7"
    briefing = run.calls[0]["messages"][0]["content"]
    assert CAVITY in briefing and watch.AWAITING in briefing
    assert "2026-11-10 13:07 UTC" in briefing


def test_a_resume_with_its_transcript_does_not_ask_the_platform(monkeypatch, tmp_path):
    study = _left_refused(tmp_path)
    platform = _Platform([{"seq": 0, "role": "user", "content": "from the platform"}])
    run = _session(monkeypatch, tmp_path, [message([text_block("ok")])],
                   lines=[], study_id=study, platform=platform)

    assert platform.reads == []
    assert "from the platform" not in run.calls[0]["messages"][0]["content"]


def test_a_new_study_is_briefed_with_no_conversation(monkeypatch, tmp_path):
    run = _session(monkeypatch, tmp_path, [message([text_block("ok")])], lines=[CAVITY])

    assert watch.CONVERSATION_OPEN not in run.calls[0]["messages"][0]["content"]


# -- the briefing text ---------------------------------------------------------------


def _row(role, content, at="2026-11-10T13:07:25Z"):
    return {"role": role, "content": content, "at": at}


def test_tool_rows_events_and_empty_replies_are_left_out():
    rows = [
        _row("event", "briefing: study x on instance y"),
        _row("user", "mesh the elbow"),
        _row("assistant", ""),
        _row("tool", {"tool": "exec", "input": {"cmd": "blockMesh"}, "output": "done"}),
        _row("event", {"desk": "cad", "cmd": "step"}),
        _row("assistant", "The elbow is meshed."),
    ]
    told = watch.earlier_conversation(rows)

    quoted = told.text.split(watch.CONVERSATION_OPEN)[1].split(watch.CONVERSATION_CLOSE)[0]
    assert "mesh the elbow" in quoted and "The elbow is meshed." in quoted
    assert "blockMesh" not in quoted and "briefing" not in quoted and "desk" not in quoted
    assert told.unanswered == 0 and watch.AWAITING not in told.text
    assert quoted.index("mesh the elbow") < quoted.index("The elbow is meshed.")


def test_the_budget_keeps_the_most_recent_and_says_what_was_left_out():
    rows = [_row("user" if i % 2 == 0 else "assistant", f"message {i:02d} " + "x" * 90)
            for i in range(40)]
    told = watch.earlier_conversation(rows, budget=1000)

    quoted = told.text.split(watch.CONVERSATION_OPEN)[1].split(watch.CONVERSATION_CLOSE)[0]
    assert "message 39" in quoted and "message 00" not in quoted
    kept = [i for i in range(40) if f"message {i:02d}" in quoted]
    assert kept == list(range(40 - len(kept), 40)), "oldest dropped first, the rest whole"
    assert told.left_out == 40 - len(kept) > 0
    assert f"The {told.left_out} earliest messages are left out" in told.text
    assert sum(len(f"message {i:02d} " + "x" * 90) for i in kept) <= 1000


def test_one_message_longer_than_the_budget_is_cut_rather_than_dropped():
    told = watch.earlier_conversation([_row("user", "y" * 50_000)], budget=8000)

    assert told.unanswered == 1
    assert "characters, is left out" in told.text
    assert len(told.text) < 9000


def test_messages_after_the_last_reply_are_awaiting_an_answer():
    rows = [_row("user", "first"), _row("assistant", "answered"),
            _row("user", "second"), _row("user", "third")]
    told = watch.earlier_conversation(rows)

    assert told.unanswered == 2
    assert "[the person, 2026-11-10 13:07 UTC, awaiting an answer]\nsecond" in told.text
    assert "[the person, 2026-11-10 13:07 UTC]\nfirst" in told.text
    assert "Answer them now" in told.text


def test_a_closing_marker_inside_a_message_does_not_end_the_quotation():
    told = watch.earlier_conversation([_row("user", f"see {watch.CONVERSATION_CLOSE} here")])

    assert told.text.count(watch.CONVERSATION_CLOSE) == 1


def test_no_conversation_is_no_paragraph():
    told = watch.earlier_conversation([_row("event", "x"), _row("tool", {"a": 1})])

    assert told.text == "" and told.unanswered == 0


def test_a_slow_refusal_after_a_slow_setup_is_still_caught(monkeypatch, tmp_path):
    """The window counts from the start, and the session's setup spends part of it: a
    refusal that answers 2.5 s after the start, with the first turn asked at 2.3 s, is
    still waited for (a fixed 2 s grace let the model run here)."""
    hold = threading.Event()
    asked = {}

    class SlowService(ReadyStarter):
        @property
        def elapsed_s(self):
            return 2.3

    starter = SlowService(FakeBackend(), "iid-9", hold=hold, error=PLAN_LIMIT)
    threading.Timer(0.2, hold.set).start()
    run = _session(monkeypatch, tmp_path, [message([text_block("plan")])],
                   lines=[CAVITY], starter=starter)

    assert run.calls == []


def test_a_start_that_answers_admitted_lets_the_turn_go_at_once(monkeypatch, tmp_path):
    """Once the start has answered and was not refused, the model runs ahead of the
    workspace's setup without waiting out anything."""
    import time as time_mod

    run_started = []
    starter = ReadyStarter(FakeBackend(), "iid-9")  # answers at once, admitted
    t0 = time_mod.monotonic()
    run = _session(monkeypatch, tmp_path, [message([text_block("ok")])], lines=[CAVITY],
                   starter=starter,
                   on_call=lambda n: run_started.append(time_mod.monotonic() - t0))

    assert len(run.calls) == 1 and run_started[0] < cli.ADMISSION_WAIT_S / 2


def test_a_resume_with_a_first_message_answers_once(monkeypatch, tmp_path):
    """The person retyping the question on resume: one turn answers it and what the
    briefing marks as awaiting an answer, not a turn of its own first."""
    study = _left_refused(tmp_path)
    run = _session(monkeypatch, tmp_path, [message([text_block("Setting up the cavity.")])],
                   lines=[CAVITY], study_id=study)

    assert len(run.calls) == 1
    sent = run.calls[0]["messages"]
    assert watch.AWAITING in sent[0]["content"]
    # The retyped message is in this one turn. Not necessarily last: when the workspace
    # comes up before the call, its ready note follows it (seen on CI, ubuntu 3.11).
    assert [m["content"] for m in sent[1:]].count(CAVITY) == 1


def test_the_platform_transcript_is_read_newest_first_in_one_call(monkeypatch, tmp_path):
    transcript = [{"seq": i, "role": "user" if i % 2 else "assistant",
                   "content": f"message {i}", "created_at": "2026-11-10T13:07:25+00:00"}
                  for i in range(1500)]
    platform = _Platform(transcript, row={"id": "remote-7", "home": "/work/x"})
    run = _session(monkeypatch, tmp_path, [message([text_block("ok")])],
                   lines=[], study_id="20261110-130725-cafe", platform=platform)

    assert platform.reads == [{"study_id": "remote-7", "after": -1, "limit": 999,
                               "newest": True}]
    briefing = run.calls[0]["messages"][0]["content"]
    assert "message 1499" in briefing and "message 0\n" not in briefing
    assert "the ones before them" in briefing or "Earlier messages are left out" in briefing


def test_a_service_that_ignores_newest_is_not_briefed_from_the_studys_start(
        monkeypatch, tmp_path):
    transcript = [{"seq": i, "role": "user", "content": f"message {i}"} for i in range(1500)]
    platform = _Platform(transcript, row={"id": "remote-7", "home": "/work/x"})
    platform.ignores_newest = True
    run = _session(monkeypatch, tmp_path, [message([text_block("ok")])],
                   lines=["hello"], study_id="20261110-130725-cafe", platform=platform)

    assert watch.CONVERSATION_OPEN not in run.calls[0]["messages"][0]["content"]


def test_more_before_says_earlier_messages_were_left_out():
    told = watch.earlier_conversation([_row("user", "hi")], more_before=True)

    assert "Earlier messages are left out" in told.text


class _Flips:
    """The ui's `keep_alive`: a question re-asked on each truth test."""

    def __init__(self, answers):
        self.answers = list(answers)

    def __bool__(self):
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


class _Stoppable(FakeBackend):
    def __init__(self):
        super().__init__()
        self.was_already_running = False
        self.stopped = 0

    def shutdown(self):
        self.stopped += 1


def test_keep_alive_is_asked_again_right_before_the_shutdown(store, monkeypatch):
    monkeypatch.setattr(cli, "console", Console(file=io.StringIO(), force_terminal=False))
    backend = _Stoppable()
    cli._close_down(backend, store, keep_alive=_Flips([False, True]))

    assert backend.stopped == 0, "another session joined while this one was closing"

    backend = _Stoppable()
    cli._close_down(backend, store, keep_alive=_Flips([False]))
    assert backend.stopped == 1, "nobody joined: put down as before"


def test_keep_alive_is_asked_again_before_a_pending_workspace_is_put_down(store, monkeypatch):
    monkeypatch.setattr(cli, "console", Console(file=io.StringIO(), force_terminal=False))
    live = _Stoppable()
    pending = pending_mod.PendingBackend("iid-1")
    threading.Timer(0.05, pending.resolve, args=(live,)).start()
    cli._close_down_pending(pending, store, keep_alive=_Flips([False, True]))

    assert live.stopped == 0


def test_a_resume_from_the_platform_numbers_its_rows_after_the_platforms(monkeypatch, tmp_path):
    """A machine with no record of the study started its counter at 0, and wrote seq
    0.. again into the platform's transcript. Its rows now go on from the highest."""
    transcript = [
        {"seq": 0, "role": "event", "content": "briefing"},
        {"seq": 1, "role": "user", "content": CAVITY},
        {"seq": 7, "role": "event", "content": "The workspace could not be started."},
    ]
    platform = _Platform(transcript, row={"id": "remote-7", "home": "/work/x"})
    run = _session(monkeypatch, tmp_path, [message([text_block("Setting up the cavity.")])],
                   lines=[], study_id="20261110-130725-cafe", platform=platform)

    seqs = [r["seq"] for r in _rows(run.store)]
    assert seqs and min(seqs) == 8, seqs


def test_rows_sharing_a_seq_are_all_kept_in_the_services_order(monkeypatch, tmp_path):
    """Studies resumed before the fix repeat seqs; every row is the person's or the
    agent's words all the same, in the order the service inserted them."""
    transcript = [
        {"seq": 0, "role": "user", "content": "first request"},
        {"seq": 1, "role": "assistant", "content": "first answer"},
        {"seq": 0, "role": "user", "content": "second request"},
    ]
    platform = _Platform(transcript, row={"id": "remote-7", "home": "/work/x"})
    platform.list_messages = lambda study_id, after=-1, limit=999, newest=False: {
        "messages": list(transcript), "more": False}
    run = _session(monkeypatch, tmp_path, [message([text_block("ok")])],
                   lines=[], study_id="20261110-130725-cafe", platform=platform)

    briefing = run.calls[0]["messages"][0]["content"]
    assert briefing.index("first request") < briefing.index("first answer") \
        < briefing.index("second request")
    assert "awaiting an answer]\nsecond request" in briefing
    assert min(r["seq"] for r in _rows(run.store)) == 2


def test_a_queued_eof_does_not_stop_the_owed_answer(monkeypatch, tmp_path):
    """An End already queued is not a message: the owed turn runs, then the EOF ends
    the session."""
    study = _left_refused(tmp_path)
    run = _session(monkeypatch, tmp_path, [message([text_block("Setting up the cavity.")])],
                   lines=[None], study_id=study)

    assert len(run.calls) == 1


def test_peeking_the_reader_leaves_it_as_it_was():
    reader = ScriptedReader(["one", "two"])
    assert cli._waiting_line(reader) and reader._lines == ["one", "two"]
    assert not cli._waiting_line(ScriptedReader([None]))
    assert not cli._waiting_line(ScriptedReader([]))
