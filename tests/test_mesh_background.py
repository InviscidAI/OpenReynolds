"""The mesh desk in the background: started and left, noted to, watched, woken on.

What these pin is the plumbing that moved the desk off the loop's thread. Measured in
production (study 20260920-161908-c7ef): one `mesh` call held the agent for 402 s,
during which it answered nobody and every line the person typed was drained into the
desk by its `interject`. The desk itself is `test_mesher.py`'s business; here a fake
one stands in, blocking on an event so each test chooses when it finishes.
"""

from __future__ import annotations

import json
import re
import threading
import time

import pytest

from conftest import ScriptedReader, install_model, message, text_block, tool_block
from openreynolds import cli
from openreynolds import loop as loop_mod
from openreynolds.backend.base import ExecResult
from openreynolds.browse import Browser
from openreynolds.config import Config
from openreynolds.loop import Loop
from openreynolds.mesher import MESH_DONE, DeskRun, MeshResult, Step
from openreynolds.mesher.check import Check
from openreynolds.store import Store
from openreynolds.tools import describe, dispatch, take_desk
from openreynolds.watch import NullReader, watch
from test_mesher import NOT_YET_JSON, OK_JSON, RAW_PNG, ScriptedProvider, answers, block, mesher
from test_prompt import IMPERATIVE_PATTERNS

pytestmark = pytest.mark.usefixtures("fast_polling")


class BlockingMesher:
    """Stands in for `Mesher`: `run` waits on `gate`, then answers with `result`.

    `DeskRun` runs on a shallow copy of the desk it is given, so what a test reads
    afterwards -- the calls, what the desk heard -- lives in lists, which the copy
    shares, rather than in counters, which it would not."""

    def __init__(self, result, gate=None):
        self.result = result
        self.gate = gate if gate is not None else threading.Event()
        self.calls: list[tuple[str, str | None]] = []
        self.heard: list[str] = []
        self.on_step = None
        self.interject = None

    def run(self, request, case=None):
        self.calls.append((request, case))
        self.gate.wait(10)
        if self.on_step is not None:
            self.on_step(Step(cmd="python3 build.py\necho ok", exit_code=0, seconds=2.0,
                              image="/work/study/mesh/look.png"))
        if self.interject is not None:
            said = self.interject()
            if said:
                self.heard.append(said)
        return self.result


def a_result(**over):
    values = dict(
        ok=True, case_rel="mesh", case_dir="/work/study/mesh", summary="a 2D duct",
        check=Check(ok=True, cells=3750, checkmesh="Mesh OK.",
                    render="mesh/renders/mesh_look.png",
                    render_abs="/work/study/mesh/renders/mesh_look.png"),
        png=RAW_PNG, seconds=61.0, tokens={"input": 10, "output": 5},
    )
    values.update(over)
    return MeshResult(**values)


def until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting for the desk"
        time.sleep(0.01)


@pytest.fixture
def fake_desk(ctx):
    fake = BlockingMesher(a_result())
    ctx.mesher = fake
    yield fake
    fake.gate.set()


@pytest.fixture
def loop(ctx, store, view):
    return Loop(Config(llm_api_key="k", model="claude-opus-5"), ctx, store, view)


# -- the tools -----------------------------------------------------------------


def test_mesh_starts_the_desk_and_returns_at_once(ctx, fake_desk):
    content, is_error = dispatch(ctx, "mesh", {"request": "a duct 20 mm tall", "case": "duct"})

    assert not is_error
    assert "has started on `duct`" in content
    assert "mesh_note" in content and "mesh_wait" in content
    assert ctx.desk is not None and not ctx.desk.done.is_set()
    until(lambda: fake_desk.calls)
    assert fake_desk.calls == [("a duct 20 mm tall", "duct")]
    # The template desk keeps its own ears: the run got a copy with the run's.
    assert fake_desk.interject is None and fake_desk.on_step is None
    # Recorded for a resume to find, on disk and not only in memory.
    assert ctx.store.session.desk["case_rel"] == "duct"
    assert ctx.store.session.desk["request"] == "a duct 20 mm tall"
    reopened = Store(ctx.store.dir.parent, ctx.store.session.study_id)
    assert reopened.session.desk["case_rel"] == "duct"


def test_mesh_wait_reports_progress_and_then_the_result(ctx, fake_desk):
    counted = []
    ctx.on_tokens = counted.append
    dispatch(ctx, "mesh", {"request": "a duct"})

    progress, _ = dispatch(ctx, "mesh_wait", {"wait_s": 0})
    assert "mesh desk on `mesh`" in progress and "running for" in progress
    assert "0 steps" in progress
    assert ctx.desk is not None, "not done, so nothing was handed over"

    fake_desk.gate.set()
    content, is_error = dispatch(ctx, "mesh_wait", {"wait_s": 5})

    assert not is_error
    assert isinstance(content, list) and content[0]["type"] == "image"
    text = describe(content)
    assert "[waited" in text
    assert "meshed: mesh/constant/polyMesh" in text
    assert ctx.desk is None and ctx.store.session.desk == {}
    assert counted == [{"input": 10, "output": 5}], "the desk's tokens, counted once"


def test_mesh_note_reaches_the_desk_and_nobody_after_it(ctx, fake_desk):
    dispatch(ctx, "mesh", {"request": "a duct"})

    said, _ = dispatch(ctx, "mesh_note", {"text": "make it 2 mm wider"})
    assert "noted for the mesh desk on `mesh`" in said
    dispatch(ctx, "mesh_note", {"text": "and 30 mm tall"})
    fake_desk.gate.set()
    until(lambda: ctx.desk.done.is_set())
    assert fake_desk.heard == ["make it 2 mm wider\nand 30 mm tall"], (
        "both notes, in one drain, the way the inbox used to arrive")

    late, _ = dispatch(ctx, "mesh_note", {"text": "too late"})
    assert "already finished" in late and "mesh_wait" in late
    dispatch(ctx, "mesh_wait", {})
    nobody, _ = dispatch(ctx, "mesh_note", {"text": "anyone?"})
    assert "no mesh desk is running" in nobody


def test_a_second_mesh_while_one_runs_starts_nothing(ctx, fake_desk):
    dispatch(ctx, "mesh", {"request": "a duct"})
    first = ctx.desk
    until(lambda: fake_desk.calls)

    answer, is_error = dispatch(ctx, "mesh", {"request": "a second duct", "case": "other"})

    assert not is_error
    assert "started nothing" in answer and "was not queued" in answer
    assert "mesh_note" in answer and "mesh_wait" in answer
    assert ctx.desk is first
    assert fake_desk.calls == [("a duct", None)]


def test_a_finished_desk_nobody_collected_is_delivered_before_a_new_one_starts(ctx, fake_desk):
    """The wake would have handed it over at the end of the turn; a `mesh` call in
    between gets it first, so the new run does not bury the old one's answer."""
    fake_desk.gate.set()
    dispatch(ctx, "mesh", {"request": "a duct"})
    first = ctx.desk
    until(lambda: first.done.is_set())

    content, is_error = dispatch(ctx, "mesh", {"request": "a second duct", "case": "other"})

    assert not is_error
    assert content[0]["type"] == "image", "the finished run's picture, first"
    text = describe(content)
    assert "had finished; its result first" in text
    assert "meshed: mesh/constant/polyMesh" in text
    assert "has started on `other`" in text
    assert ctx.desk is not first and ctx.desk.case_rel == "other"


def test_mesh_wait_ends_early_when_the_person_types(ctx, fake_desk):
    ctx.on_wait_input = lambda: True
    dispatch(ctx, "mesh", {"request": "a duct"})

    began = time.monotonic()
    answer, _ = dispatch(ctx, "mesh_wait", {"wait_s": 60})

    assert time.monotonic() - began < 5
    assert "the person wrote, so this answered early" in answer
    assert "their words follow this result" in answer
    assert "call mesh_wait again -- the desk is still building" in answer
    assert ctx.desk is not None


def test_mesh_wait_says_when_the_ceiling_cut_the_wait(ctx, fake_desk, monkeypatch):
    monkeypatch.setattr("openreynolds.tools.JOB_WAIT_MAX_S", 0)
    dispatch(ctx, "mesh", {"request": "a duct"})

    answer, _ = dispatch(ctx, "mesh_wait", {"wait_s": 900})

    assert "exceeds the 0s ceiling" in answer and "waiting again is free" in answer


def test_the_foreground_path_is_the_old_shape_whole(ctx):
    """`wait: true`: the call holds, the desk hears the session's inbox directly, and
    nothing is registered as a background run."""
    gate = threading.Event()
    gate.set()
    desk = BlockingMesher(a_result(), gate=gate)
    desk.interject = lambda: "from the inbox"
    ctx.mesher = desk
    counted = []
    ctx.on_tokens = counted.append

    content, is_error = dispatch(ctx, "mesh", {"request": "a duct", "wait": True})

    assert not is_error
    assert desk.heard == ["from the inbox"]
    assert isinstance(content, list) and "meshed:" in describe(content)
    assert ctx.desk is None and ctx.store.session.desk == {}
    assert counted == [{"input": 10, "output": 5}]


def test_a_background_run_never_reads_the_session_inbox(ctx, fake_desk):
    """The measured failure: the desk's `interject` was the loop's own drain, so for
    the 402 s the desk ran, every line the person typed went to it and not to the
    agent they were talking to. A background run hears `mesh_note` and nothing else."""
    reads = []
    fake_desk.interject = lambda: reads.append(1) or "from the inbox"
    dispatch(ctx, "mesh", {"request": "a duct"})

    fake_desk.gate.set()
    until(lambda: ctx.desk.done.is_set())

    assert reads == [] and fake_desk.heard == []


def test_the_real_desk_hears_a_note_at_its_next_step(backend, store, ctx):
    """Through the real `Mesher`: the note lands in the thread as a remark, the step
    display still runs, the template is untouched, the inbox is never read."""
    gate = threading.Event()

    class Gated(ScriptedProvider):
        def stream(self, **kwargs):
            gate.wait(5)
            return super().stream(**kwargs)

    inbox = []
    shown = []

    def inbox_reader():
        inbox.append(1)
        return None

    desk = mesher(backend, store, [block("python3 build.py"), block(f"echo {MESH_DONE}")],
                  interject=inbox_reader)
    desk.provider = Gated([block("python3 build.py"), block(f"echo {MESH_DONE}")])
    desk.on_step = shown.append
    answers(backend, {"mesh_look.py": ExecResult(0, OK_JSON, False, None)})
    ctx.mesher = desk

    dispatch(ctx, "mesh", {"request": "a duct"})
    dispatch(ctx, "mesh_note", {"text": "make it 2 mm wider"})
    gate.set()
    content, is_error = dispatch(ctx, "mesh_wait", {"wait_s": 10})

    assert not is_error
    text = describe(content)
    assert "make it 2 mm wider" in text and "said this to the mesh desk directly" in text
    assert inbox == [], "the session's inbox was never read on the desk's behalf"
    assert [s.cmd for s in shown] == ["python3 build.py"], "the step display still ran"
    assert desk.interject is inbox_reader and desk.on_step == shown.append, (
        "the session's desk is the template, and the run left its ears alone")


def test_an_abandoned_run_ends_at_its_next_lap_and_leaves_the_template_alone(backend, store):
    """The session ended with the desk still building. In a process that hosts several
    sessions the thread would otherwise spend its whole budget on nobody."""
    gate = threading.Event()

    class Gated(ScriptedProvider):
        def stream(self, **kwargs):
            gate.wait(5)
            return super().stream(**kwargs)

    desk = mesher(backend, store, [block("echo working")])
    desk.provider = Gated([block("echo working")])
    answers(backend, {"mesh_look.py": ExecResult(0, NOT_YET_JSON, False, None)})
    run = DeskRun(desk, "a duct").start()

    run.abandon()
    gate.set()

    assert run.done.wait(5)
    assert run.result is not None and run.result.stopped in ("steps", "time")
    assert len(run.result.steps) <= 1
    assert desk.max_steps > 0 and desk.max_seconds > 0, "the session's desk keeps its budgets"


def test_a_run_that_raises_reports_the_fault_rather_than_hanging(ctx):
    class Broken:
        on_step = None
        interject = None

        def run(self, request, case=None):
            raise RuntimeError("the provider client is gone")

    run = DeskRun(Broken(), "a duct").start()

    assert run.done.wait(5)
    assert run.result is None and "RuntimeError" in run.error
    assert "stopped without a result" in run.report()
    ctx.desk = run
    answer, _ = dispatch(ctx, "mesh_wait", {"wait_s": 1})
    assert "stopped without a result" in answer and "RuntimeError" in answer
    assert ctx.desk is None


def test_a_result_is_handed_over_once(ctx):
    counted = []
    ctx.on_tokens = counted.append
    fake = BlockingMesher(a_result())
    fake.gate.set()
    run = DeskRun(fake, "a duct").start()
    run.done.wait(5)
    ctx.desk = run
    ctx.store.session.desk = run.record()

    assert take_desk(ctx, run) is fake.result
    assert take_desk(ctx, run) is fake.result

    assert counted == [{"input": 10, "output": 5}]
    assert ctx.desk is None and ctx.store.session.desk == {}


# -- the wait and the inbox ----------------------------------------------------
#
# Measured in production (study 20260921-033019-e1b4, 2026-09-21): `mesh_wait` returned
# three times in nine seconds -- 03:33:40, 03:33:41, 03:33:45 -- the last two with
# `[waited 0s] [the user said something, so this answered early]`, and no words from the
# person ever followed; the model switched to `bash sleep 115` and paced the remaining
# thirty-seven minutes with it. The same prompt on the same model, side by side
# (20260921-033356-076b), held its `job_check(wait_s=120..300)` waits in full. The wait
# asked the reader's `pending()` -- is anything in the queue -- and the loop's drain had
# put a line back that was nobody's to deliver to the model. These drive the real loop,
# the real drain (`cli._typed_while_working`) and the real `Loop.heard`.


def _wired(loop, ctx, view, backend, store, reader):
    """The session's wiring for what is typed mid-turn, as `cli.drive` makes it."""
    loop.interject = lambda: cli._typed_while_working(
        loop, view, Browser(backend, store), store, reader)
    ctx.on_wait_input = loop.heard
    ctx.on_leaving = lambda: loop.leaving


def _results(loop, index):
    """The user message carrying the results of the model's `index`th tool batch."""
    carriers = [m for m in loop.messages if m["role"] == "user" and isinstance(m["content"], list)]
    return carriers[index]["content"]


@pytest.fixture
def quick_waits(monkeypatch):
    monkeypatch.setattr("openreynolds.tools.DESK_WAIT_POLL_S", 0.01)


def test_end_pressed_during_a_held_mesh_wait_ends_the_wait_and_stops_the_desk(
    loop, ctx, view, backend, store, fake_desk, quick_waits
):
    """The e1b4 shape, as it should have gone. End (`/exit`) pressed while `mesh_wait`
    holds: the wait returns within a poll saying the person ended the session, with
    where the desk had got to; the model is not asked again; the turn's last message
    carries the harness's one line; the desk's budgets are zeroed so it stops at its
    next command rather than building for nobody; and the `/exit` is still in the
    reader for whoever reads next. Before this the put-back `/exit` was honoured when
    the turn ended on its own -- thirty-seven minutes later, in production."""
    reader = ScriptedReader([])
    _wired(loop, ctx, view, backend, store, reader)
    fake = install_model(loop, [
        message([tool_block("mesh_wait", {"wait_s": 60})], stop_reason="tool_use"),
        message([text_block("the mesh is in")]),
    ])
    dispatch(ctx, "mesh", {"request": "a duct"})
    run = ctx.desk
    threading.Timer(0.3, lambda: reader._lines.append("/exit")).start()

    loop.say("mesh it")
    began = time.monotonic()
    loop.run()
    elapsed = time.monotonic() - began

    assert 0.25 <= elapsed < 5, "held until End, then answered at once, not at 60 s"
    assert len(fake.calls) == 1, "the model was not asked again"
    results = _results(loop, 0)
    assert [b["type"] for b in results] == ["tool_result", "text"]
    cut = describe(results[0]["content"])
    assert "running for" in cut, "where the desk had got to"
    assert "the person ended the session, so this answered early" in cut
    assert "the person wrote" not in cut and "call mesh_wait again" not in cut
    assert results[1]["text"].endswith(loop_mod.LEFT_MID_TURN)
    assert view.interjections == [], "nothing was said for the model"
    assert reader.poll() == "/exit", "still there for whoever reads next, as before"
    # The desk: not delivered (it has not finished), and told to stop -- its run's
    # budgets are zero, so its loop ends at the next lap (`DeskRun.abandon`).
    assert ctx.desk is run and not run.done.is_set()
    assert run.mesher.max_steps == 0 and run.mesher.max_seconds == 0.0


def test_leaving_stops_a_building_desk_at_its_next_lap(backend, store, ctx, view):
    """Through the real `Mesher`: told by `Loop.leave` rather than by the close-down,
    a run in the middle of a step ends after that step with the finish check run on
    what is on disk, exactly as `abandon` does -- and the session's own desk, the
    template, is stopped too, since a session that is ending never meshes again."""
    gate = threading.Event()

    class Gated(ScriptedProvider):
        def stream(self, **kwargs):
            gate.wait(5)
            return super().stream(**kwargs)

    desk = mesher(backend, store, [block("echo working")])
    desk.provider = Gated([block("echo working")])
    answers(backend, {"mesh_look.py": ExecResult(0, NOT_YET_JSON, False, None)})
    ctx.mesher = desk
    loop = Loop(Config(llm_api_key="k", model="claude-opus-5"), ctx, store, view)
    dispatch(ctx, "mesh", {"request": "a duct"})
    run = ctx.desk

    loop.leave()
    loop.leave()  # the drain meets the same put-back line every second a wait asks
    gate.set()

    assert loop.leaving is True
    assert run.done.wait(5)
    assert run.result is not None and run.result.stopped in ("steps", "time")
    assert len(run.result.steps) <= 1
    assert desk.max_steps == 0 and desk.max_seconds == 0.0, (
        "a foreground `mesh` holding the turn ends at its next lap the same way")


def test_leaving_is_what_the_session_wires_the_wait_to(loop, ctx, view, backend, store):
    """Unit: the drain tells the loop on `/exit`, `/quit` and EOF and on nothing else;
    `heard` stays what it was -- a `/exit` carries no words for the model."""
    reader = ScriptedReader([])
    _wired(loop, ctx, view, backend, store, reader)

    assert loop.leaving is False and ctx.on_leaving() is False
    reader._lines.append("/status")
    assert loop.heard() is False and loop.leaving is False
    reader._lines.append("coarser, please")
    assert loop.heard() is True and loop.leaving is False
    reader._lines.append("/quit")
    assert loop.heard() is True, "the words are still unseen by the model"
    assert loop.leaving is True and ctx.on_leaving() is True
    assert reader.poll() == "/quit", "put back for the prompt, as before"

    for line in ("/exit", None):
        fresh = Loop(Config(llm_api_key="k", model="claude-opus-5"), ctx, store, view)
        _wired(fresh, ctx, view, backend, store, ScriptedReader([line]))
        assert fresh.heard() is False and fresh.leaving is True


def test_a_message_waiting_before_the_wait_cuts_it_once_and_rides_with_the_result(
    loop, ctx, view, backend, store, fake_desk, quick_waits
):
    """What the owner took e1b4 to be: a message pending as the wait begins. It cuts
    the wait at 0 s -- once -- and the words are in the same message as the result
    that says so; the next wait is not cut by it again."""
    reader = ScriptedReader(["how many cells so far?"])
    _wired(loop, ctx, view, backend, store, reader)
    install_model(loop, [
        message([tool_block("mesh_wait", {"wait_s": 60})], stop_reason="tool_use"),
        message([tool_block("mesh_wait", {"wait_s": 60})], stop_reason="tool_use"),
        message([text_block("about 3,750; the mesh is in")]),
    ])
    dispatch(ctx, "mesh", {"request": "a duct"})
    threading.Timer(0.6, fake_desk.gate.set).start()

    loop.say("mesh it")
    loop.run()

    first = _results(loop, 0)
    assert [b["type"] for b in first] == ["tool_result", "text"]
    cut = describe(first[0]["content"])
    assert "[waited 0s]" in cut
    assert "the person wrote, so this answered early" in cut
    assert "their words follow this result" in cut
    assert "call mesh_wait again -- the desk is still building" in cut
    assert first[1]["text"] == "how many cells so far?", "and they do, in this message"
    assert view.interjections == ["how many cells so far?"]

    second = _results(loop, 1)
    assert [b["type"] for b in second] == ["tool_result"], "delivered once, not again"
    held = describe(second[0]["content"])
    assert "answered early" not in held and "waited 0s" not in held
    assert "meshed: mesh/constant/polyMesh" in held, "the second wait held for the desk"
    assert ctx.desk is None


def test_a_message_typed_mid_wait_cuts_it_once_and_reaches_the_model(
    loop, ctx, view, backend, store, fake_desk, quick_waits
):
    """The good behaviour, pinned: a line typed while the wait holds ends it within a
    poll, the words ride with that result, and the wait after it holds again."""
    reader = ScriptedReader([])
    _wired(loop, ctx, view, backend, store, reader)
    install_model(loop, [
        message([tool_block("mesh_wait", {"wait_s": 60})], stop_reason="tool_use"),
        message([tool_block("mesh_wait", {"wait_s": 60})], stop_reason="tool_use"),
        message([text_block("done")]),
    ])
    dispatch(ctx, "mesh", {"request": "a duct"})
    threading.Timer(0.3, lambda: reader._lines.append("make the inlet 30 mm")).start()
    threading.Timer(0.9, fake_desk.gate.set).start()

    loop.say("mesh it")
    began = time.monotonic()
    loop.run()
    elapsed = time.monotonic() - began

    first = _results(loop, 0)
    assert [b["type"] for b in first] == ["tool_result", "text"]
    cut = describe(first[0]["content"])
    assert "the person wrote, so this answered early" in cut
    assert "running for" in cut, "with where the desk had got to"
    assert first[1]["text"] == "make the inlet 30 mm"
    second = _results(loop, 1)
    assert [b["type"] for b in second] == ["tool_result"]
    assert "answered early" not in describe(second[0]["content"])
    assert "meshed: mesh/constant/polyMesh" in describe(second[0]["content"])
    assert 0.8 <= elapsed < 30, "the first wait ended on the line, the second on the desk"


def test_a_command_typed_mid_wait_is_answered_there_and_the_wait_goes_on(
    loop, ctx, view, backend, store, fake_desk, quick_waits
):
    """`/status` is the harness's to answer, not the model's: it is answered on the
    spot, the wait holds, and the result says nothing about the person."""
    reader = ScriptedReader([])
    _wired(loop, ctx, view, backend, store, reader)
    install_model(loop, [
        message([tool_block("mesh_wait", {"wait_s": 60})], stop_reason="tool_use"),
        message([text_block("done")]),
    ])
    dispatch(ctx, "mesh", {"request": "a duct"})
    threading.Timer(0.2, lambda: reader._lines.append("/status")).start()
    threading.Timer(0.6, fake_desk.gate.set).start()

    loop.say("mesh it")
    loop.run()

    assert view.statuses, "answered while the wait held"
    results = _results(loop, 0)
    assert [b["type"] for b in results] == ["tool_result"]
    assert "answered early" not in describe(results[0]["content"])
    assert "meshed: mesh/constant/polyMesh" in describe(results[0]["content"])


def test_heard_is_what_the_session_wires_the_wait_to(loop, ctx, view, backend, store):
    """Unit: `Loop.heard` drains, keeps words for the model, answers commands, and says
    whether anything is kept -- true until the batch delivers it."""
    reader = ScriptedReader([])
    _wired(loop, ctx, view, backend, store, reader)

    assert loop.heard() is False
    reader._lines.append("/status")
    assert loop.heard() is False and view.statuses
    reader._lines.append("/exit")
    assert loop.heard() is False and reader.poll() == "/exit"
    reader._lines.append("coarser, please")
    assert loop.heard() is True
    assert loop.heard() is True, "still unseen by the model, so still true"
    assert reader.pending() is False, "taken from the reader, so it cannot fire twice"
    assert loop._typed == ["coarser, please"], "and kept for this batch's results, once"


# -- the run's own words -------------------------------------------------------


def test_the_progress_line_and_the_report_are_facts():
    fake = BlockingMesher(a_result(case_rel="Valve_1"))
    run = DeskRun(fake, "a valve", "Valve 1")

    assert run.case_rel == "Valve_1"
    assert run.record() == {"case_rel": "Valve_1", "request": "a valve",
                            "started_at": run.started_wall}
    line = run.progress_line()
    assert "running for" in line and "0 steps" in line

    run.start()
    fake.gate.set()
    assert run.done.wait(5)
    line = run.progress_line()
    assert "finished after" in line and "1 step" in line
    assert "`python3 build.py`" in line and "with a picture" in line
    assert "echo ok" not in line, "the command's first line only"
    report = run.report()
    assert report.startswith("The mesh desk on `Valve_1` has finished.")
    assert "meshed: Valve_1/constant/polyMesh" in report
    assert "/work/study/mesh/renders/mesh_look.png" in report, "where the picture is"
    assert "base64" not in report, "a wake is text; the picture is not attached to it"


def test_a_report_with_no_picture_names_none():
    fake = BlockingMesher(a_result(png=None))
    fake.gate.set()
    run = DeskRun(fake, "a duct").start()
    run.done.wait(5)
    assert "picture of the mesh is at" not in run.report()


# -- watch mode ----------------------------------------------------------------


def test_watch_waits_on_a_live_desk_when_no_job_is_running(backend, store, view):
    fake = BlockingMesher(a_result())
    run = DeskRun(fake, "a duct").start()

    wake = watch(backend, store, view, NullReader(), deadline=time.monotonic() + 0.3, desk=run)

    fake.gate.set()
    assert wake.kind == "timeout", "not idle: the desk is work in flight"
    assert ["mesh desk: mesh"] in view.watched
    assert any("mesh desk on `mesh`" in stage for stage in view.stages), (
        "with no tracker, the stage line carries the desk's progress")


def test_a_finished_desk_wakes_with_its_report(backend, store, view):
    fake = BlockingMesher(a_result())
    run = DeskRun(fake, "a duct").start()
    threading.Timer(0.05, fake.gate.set).start()

    wake = watch(backend, store, view, NullReader(), desk=run)

    assert wake.kind == "desk" and wake.run is run
    assert "has finished" in wake.text and "meshed: mesh/constant/polyMesh" in wake.text


def test_a_typed_line_still_wins_over_a_finished_desk(backend, store, view):
    fake = BlockingMesher(a_result())
    fake.gate.set()
    run = DeskRun(fake, "a duct").start()
    run.done.wait(5)

    wake = watch(backend, store, view, ScriptedReader(["what is it doing?"]), desk=run)

    assert wake.kind == "user" and wake.text == "what is it doing?"


def test_watch_is_idle_without_a_desk_or_a_job(backend, store, view):
    assert watch(backend, store, view, NullReader(), desk=None).kind == "idle"


# -- the session loop ----------------------------------------------------------


def test_a_finished_desk_wakes_the_model_without_a_typed_line(loop, backend, store, view):
    fake = BlockingMesher(a_result())
    run = DeskRun(fake, "a duct").start()
    loop.ctx.desk = run
    store.session.desk = run.record()
    counted = []
    loop.ctx.on_tokens = counted.append
    model = install_model(loop, [message([text_block("the mesh is in; on to the fields")])])
    threading.Timer(0.05, fake.gate.set).start()

    cli._run_interactive(loop, backend, store, view, Browser(backend, store), ScriptedReader([]))

    informed = [m for m in loop.messages if m["role"] in ("system", "user")]
    assert any("has finished" in str(m["content"]) for m in informed), "the model was told"
    assert len(model.calls) == 1, "and took a turn on it"
    assert loop.ctx.desk is None and store.session.desk == {}
    assert counted == [{"input": 10, "output": 5}]


def test_a_finished_desk_is_kept_in_the_thread_while_the_model_is_refusing(
    loop, backend, store, view, monkeypatch
):
    """Same rule as a job's end under `blocked_reason`: a fact worth keeping for
    whenever the session resumes, and no turn, which would be refused like the last."""
    turns = []
    monkeypatch.setattr(loop, "run", lambda: turns.append(1))
    loop.blocked_reason = "The model API returned 402: no budget"
    fake = BlockingMesher(a_result())
    fake.gate.set()
    run = DeskRun(fake, "a duct").start()
    run.done.wait(5)
    loop.ctx.desk = run

    cli._run_interactive(loop, backend, store, view, Browser(backend, store), ScriptedReader([]))

    assert turns == []
    assert any("has finished" in str(m["content"]) for m in loop.messages)
    assert loop.ctx.desk is None


def test_a_one_shot_run_waits_for_the_desk(loop, backend, store, view):
    """`-p` used to return when no job was left; a desk still building would have
    died with the process, its case unreported."""
    fake = BlockingMesher(a_result())
    install_model(loop, [
        message([text_block("the desk is on it; nothing to do until it lands")]),
        message([text_block("done: mesh in, fields written")]),
    ])
    run = DeskRun(fake, "a duct").start()
    loop.ctx.desk = run
    threading.Timer(0.05, fake.gate.set).start()

    outcome = cli._run_one_shot(loop, backend, store, "mesh a duct", view, NullReader())

    assert outcome == "ok"
    assert loop.ctx.desk is None
    assert any("has finished" in str(m.get("content")) for m in loop.messages)


def test_a_bounded_one_shot_says_the_desk_ends_with_the_process(loop, backend, store, view):
    fake = BlockingMesher(a_result())
    install_model(loop, [message([text_block("waiting on the desk")])])
    loop.ctx.desk = DeskRun(fake, "a duct").start()

    outcome = cli._run_one_shot(loop, backend, store, "mesh it", view, NullReader(),
                                max_wait_minutes=0.001)

    fake.gate.set()
    assert outcome == "timeout"
    assert any("ends with this process" in i for i in view.infos)


# -- a resume that finds the record --------------------------------------------


def test_a_resume_that_finds_a_live_desk_record_is_told_once(backend, store):
    store.session.home = "/work/20260824-120000-abcd"
    store.session.desk = {"case_rel": "duct", "request": "a duct 20 mm tall",
                          "started_at": "2026-09-20T08:19:08Z"}
    store.save()
    backend.exec_result = ExecResult(0, "", False, None)

    brief = cli._situation_brief(store, backend, resuming=True, interactive=True,
                                 browser=Browser(backend, store))

    assert "mesh desk was building `duct`" in brief and "interrupted" in brief
    assert "a duct 20 mm tall" in brief and "mesh_look.py duct" in brief
    for pattern in IMPERATIVE_PATTERNS:
        assert re.search(pattern, brief, re.IGNORECASE) is None, pattern

    again = cli._situation_brief(store, backend, resuming=True, interactive=True,
                                 browser=Browser(backend, store))
    assert "interrupted" not in again, "said once"
    assert Store(store.dir.parent, store.session.study_id).session.desk == {}


def test_a_fresh_session_and_a_clean_resume_say_nothing_about_a_desk(backend, store):
    store.session.home = "/work/20260824-120000-abcd"
    backend.exec_result = ExecResult(0, "", False, None)
    for resuming in (False, True):
        brief = cli._situation_brief(store, backend, resuming=resuming, interactive=True,
                                     browser=Browser(backend, store))
        assert "mesh desk" not in brief


def test_a_session_record_written_before_the_desk_field_still_loads(tmp_path):
    root = tmp_path / "studies"
    (root / "old").mkdir(parents=True)
    (root / "old" / "session.json").write_text(
        json.dumps({"study_id": "old", "jobs": {}}), encoding="utf-8")

    assert Store(root, "old").session.desk == {}
