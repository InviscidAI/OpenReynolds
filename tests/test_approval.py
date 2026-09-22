"""Putting a question to the person, and reading their answer from the same reader
every interface already feeds."""

from __future__ import annotations

import io
import json

import pytest

from conftest import RecordingView, ScriptedReader, install_model, message, text_block, tool_block
from openreynolds import cli, commands
from openreynolds.approval import LEAVING, NOBODY, Approver, Decision
from openreynolds.browse import Browser
from openreynolds.config import Config
from openreynolds.jsonview import JsonView
from openreynolds.loop import Loop
from openreynolds.view import ConsoleView
from test_cli import finishing_jobs

pytestmark = pytest.mark.usefixtures("fast_polling")


class AskingView(RecordingView):
    def __init__(self):
        super().__init__()
        self.questions = []
        self.answers = []
        self.modes = []

    def approval(self, request_id, kind, title, detail, choices):
        self.questions.append((request_id, kind, title, detail, choices))

    def approval_done(self, request_id, outcome, note=""):
        self.answers.append((request_id, outcome, note))

    def mode(self, mode):
        self.modes.append(mode)


def ask(lines, local=None):
    view = AskingView()
    reader = ScriptedReader(lines)
    decision = Approver(view, reader, local=local).ask("job", "Start a job: solve", "cmd: x")
    return decision, view, reader


@pytest.mark.parametrize("line,approved,all_,note", [
    ("/yes", True, False, ""),
    ("/approve", True, False, ""),
    ("/y", True, False, ""),
    ("y", True, False, ""),
    ("YES", True, False, ""),
    ("ok", True, False, ""),
    ("go", True, False, ""),
    ("/no", False, False, ""),
    ("/no too expensive", False, False, "too expensive"),
    ("/deny", False, False, ""),
    ("n", False, False, ""),
    ("/all", True, True, ""),
    ("/yes all", True, True, ""),
    ("a", True, True, ""),
    ("all", True, True, ""),
    ("run it on 16 cores instead", False, False, "run it on 16 cores instead"),
])
def test_answers(line, approved, all_, note):
    decision, view, _ = ask([line])
    assert (decision.approved, decision.all, decision.note) == (approved, all_, note)
    (request_id, kind, title, detail, choices), = view.questions
    assert kind == "job" and title == "Start a job: solve" and choices
    assert view.answers == [(request_id, decision.outcome, note)]


def test_outcomes_are_named():
    assert Decision(True).outcome == "approved"
    assert Decision(False).outcome == "declined"
    assert Decision(True, all=True).outcome == "approved_all"


def test_local_commands_are_answered_and_the_question_stays_open():
    handled = []
    decision, view, _ = ask(["/status", "", "/mode", "/yes"], local=handled.append)
    assert decision.approved
    assert [c.kind for c in handled] == [commands.STATUS, commands.MODE]
    assert len(view.answers) == 1


def test_nobody_answering_declines_and_leaves_the_eof_for_the_prompt():
    decision, _, reader = ask([])
    assert not decision.approved and decision.note == NOBODY
    assert reader.get() is None


def test_leaving_declines_and_the_exit_still_happens():
    decision, _, reader = ask(["/exit"])
    assert not decision.approved and decision.note == LEAVING
    assert reader.get() == "/exit"


def test_the_json_stream_carries_the_question_and_its_answer():
    sink = io.StringIO()
    view = JsonView(sink)
    view.approval("ab12", "mesh", "Build a mesh", "a pipe", ["approve", "decline", "approve all"])
    view.approval_done("ab12", "declined", "smaller")
    view.mode("structured")
    rows = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert {k: rows[0][k] for k in ("type", "id", "kind", "title", "detail", "choices")} == {
        "type": "approval", "id": "ab12", "kind": "mesh", "title": "Build a mesh",
        "detail": "a pipe", "choices": ["approve", "decline", "approve all"],
    }
    assert {k: rows[1][k] for k in ("type", "id", "outcome", "note")} == {
        "type": "approval_done", "id": "ab12", "outcome": "declined", "note": "smaller",
    }
    assert (rows[2]["type"], rows[2]["mode"], rows[2]["label"]) == ("mode", "structured", "Structured")


def test_the_plain_terminal_shows_the_question_and_how_to_answer(console):
    out = io.StringIO()
    from rich.console import Console

    view = ConsoleView(Console(file=out, width=120))
    view.approval("ab12", "job", "Start a job: solve", "cmd: simpleFoam", ["approve"])
    view.approval_done("ab12", "approved_all")
    text = out.getvalue()
    assert "Start a job: solve" in text and "cmd: simpleFoam" in text
    assert "y approve" in text and "a approve all" in text
    assert "full auto" in text


class AnsweringView(AskingView):
    """Types the answer only once the question is on screen, as a person would."""

    def __init__(self, reader, answer):
        super().__init__()
        self.reader = reader
        self.answer = answer

    def approval(self, request_id, kind, title, detail, choices):
        super().approval(request_id, kind, title, detail, choices)
        self.reader._lines.append(self.answer)


def _wired(loop, ctx, store, backend, view, reader):
    browser = Browser(backend, store)
    loop.approver = Approver(view, reader, local=lambda c: cli._local(c, view, browser, store, loop))
    loop.interject = lambda: cli._typed_while_working(loop, view, browser, store, reader)
    return loop


def test_a_line_typed_before_the_question_is_an_interjection_not_its_answer(ctx, store, backend):
    """Typed while the model was still streaming the turn that ends in job_start. It
    used to be read as the answer at once: the job was declined with the message as
    the reason before the person ever saw the question."""
    reader = ScriptedReader(["does the inlet look right to you?"])
    view = AnsweringView(reader, "/yes")
    ctx.mode = "partial"
    loop = _wired(Loop(Config(llm_api_key="k", model="claude-opus-5", mode="partial"), ctx, store, view),
                  ctx, store, backend, view, reader)
    finishing_jobs(backend)
    install_model(loop, [
        message([tool_block("job_start", {"cmd": "simpleFoam", "name": "solve"})], stop_reason="tool_use"),
        message([text_block("started")]),
    ])
    loop.say("start the solve")
    loop.run()

    assert [a[1] for a in view.answers] == ["approved"]
    assert [s["cmd"] for s in backend.started] == ["simpleFoam"]
    carrier = next(m for m in loop.messages if m["role"] == "user" and isinstance(m["content"], list))
    assert carrier["content"][0]["type"] == "tool_result" and not carrier["content"][0].get("is_error")
    assert any(b.get("type") == "text" and "does the inlet look right" in b.get("text", "")
               for b in carrier["content"]), "the words reach the model as an interjection"


def test_a_mode_switch_typed_during_one_call_governs_the_next_in_the_same_turn(ctx, store, backend):
    """One turn asks for [bash, job_start]; `/mode partial` is typed while the bash
    runs. The switch applies from the next tool call, so the job is put to the person."""
    reader = ScriptedReader([])
    view = AnsweringView(reader, "/no not yet")
    loop = _wired(Loop(Config(llm_api_key="k", model="claude-opus-5"), ctx, store, view),
                  ctx, store, backend, view, reader)
    finishing_jobs(backend)
    real = loop._dispatch

    def dispatching(block, tool_input):
        if block.name == "bash":
            reader._lines.append("/mode partial")
        return real(block, tool_input)

    loop._dispatch = dispatching
    install_model(loop, [
        message([tool_block("bash", {"cmd": "ls"}, "tu_1"),
                 tool_block("job_start", {"cmd": "simpleFoam", "name": "solve"}, "tu_2")],
                stop_reason="tool_use"),
        message([text_block("waiting")]),
    ])
    loop.say("go")
    loop.run()

    assert ctx.mode == "partial"
    assert [q[1] for q in view.questions] == ["job"]
    assert view.answers[0][1:] == ("declined", "not yet")
    assert backend.started == []


def test_leaving_at_a_question_ends_the_turn_without_asking_the_model_again(ctx, store, backend):
    """`/exit` typed to an open question declines the call and is put back, as before;
    the loop's next drain meets the put-back line, so the turn ends there -- the model
    is not asked for the reply nobody would read -- and the session follows."""
    from openreynolds.loop import LEFT_MID_TURN

    reader = ScriptedReader([])
    view = AnsweringView(reader, "/exit")
    ctx.mode = "partial"
    loop = _wired(Loop(Config(llm_api_key="k", model="claude-opus-5", mode="partial"), ctx, store, view),
                  ctx, store, backend, view, reader)
    fake = install_model(loop, [
        message([tool_block("job_start", {"cmd": "simpleFoam", "name": "solve"})], stop_reason="tool_use"),
        message([text_block("understood, leaving it")]),
    ])
    loop.say("start the solve")
    loop.run()

    assert [a[1:] for a in view.answers] == [("declined", LEAVING)]
    assert backend.started == []
    assert len(fake.calls) == 1, "the model was not asked again"
    assert loop.leaving is True
    carrier = next(m for m in loop.messages if m["role"] == "user" and isinstance(m["content"], list))
    assert carrier["content"][0]["is_error"] and LEAVING in carrier["content"][0]["content"]
    assert carrier["content"][-1]["text"].endswith(LEFT_MID_TURN)
    assert reader.poll() == "/exit", "still there for the prompt, as before"


def test_a_partial_session_waits_for_yes_before_starting_the_job(ctx, store, backend):
    """End to end through the interactive loop: the question is answered by the same
    reader the prompt reads, and the job starts only after the "/yes"."""
    view = AskingView()
    loop = Loop(Config(llm_api_key="k", model="claude-opus-5", mode="partial"), ctx, store, view)
    ctx.mode = "partial"
    finishing_jobs(backend)
    reader = ScriptedReader(["start the solve", "/yes", "/exit"])
    browser = Browser(backend, store)
    loop.approver = Approver(view, reader, local=lambda c: cli._local(c, view, browser, store, loop))
    install_model(loop, [
        message([tool_block("job_start", {"cmd": "simpleFoam", "name": "solve"})],
                stop_reason="tool_use"),
        message([text_block("the solve is running")]),
    ])

    cli._run_interactive(loop, backend, store, view, browser, reader)

    assert [q[1] for q in view.questions] == ["job"]
    assert view.answers[0][1] == "approved"
    assert [s["cmd"] for s in backend.started] == ["simpleFoam"]
