"""The CAD desk: one python cell a step, a picture back, and a script it cannot fake.

What these pin is the harness, not the model. Most of them are the mesh desk's tests
carried across, because most of what the desk owes did not change when the channel did:
a block is taken out of a message and run, older pictures fall out of the thread and
their words do not, the finish token is checked rather than believed, a budget ends a
run that is going nowhere, and the person's words reach the desk while it works.

Three things here are new, and all three are the kernel's doing:

* **the cell log is the artifact.** A cell that ran is not automatically a cell that
  belongs in the script, so the accept gate is tested the way a gate should be -- driven
  randomly, every sequence replayed, not sampled;
* **the picture arrives as bytes**, so the PNG-scrape and the not-sent-twice tests are
  gone with the heuristic they pinned, and a new one takes their place: a cell that
  drew and sent nothing back is told so, because `matplotlib.use("Agg")` succeeds
  silently and is the exact habit a model brings from headless bash scripting;
* **the session can die**, and the log is what survives it.
"""

from __future__ import annotations

import random
import re
import time
from types import SimpleNamespace

import pytest

from openreynolds import images
from openreynolds.backend.base import BackendError, ExecResult
from openreynolds.backend.kernel import CellResult
from openreynolds.cad.agent import (
    KEEP_IMAGES,
    NUDGE_AT_STEP,
    STEP_TIMEOUT_S,
    CadDesk,
    _evict,
    _is_finish,
    parse_action,
)
from openreynolds.cad.brief import CAD_DONE, system_prompt, task_message
from openreynolds.cad.cells import CellLog, bound_names, free_names
from openreynolds.cad.check import Check
from openreynolds.config import Config
from dataclasses import dataclass

from openreynolds.llm import ProviderError, TextBlock, ToolUseBlock, Turn

RAW_PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64

MESHED = "REGION:\nSINGLE:\n"
"""What the workspace says when `constant/polyMesh` is there."""

CHT = "REGION:air\nREGION:solid\n"
"""And when two regions are meshed and no singular mesh is."""


class ScriptedProvider:
    """Replays turns; repeats the last one if the loop keeps asking.

    The thread it is handed is one mutable list the loop keeps appending to, so each
    call is snapshotted -- a test that reads `calls[1]` wants what was sent then.
    """

    name = "anthropic"

    def __init__(self, texts, delay: float = 0.0):
        self.texts = list(texts)
        self.delay = delay
        self.calls: list[dict] = []
        self.made = 0

    def stream(self, **kwargs):
        if self.delay:
            time.sleep(self.delay)
        self.calls.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
        text = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
        if isinstance(text, Exception):
            raise text
        self.made += 1
        return Turn(content=_blocks(text, self.made), provider=self.name,
                    stop_reason="tool_use" if isinstance(text, Cell) else "end_turn",
                    tokens={"input": 10, "output": 5})


@dataclass
class Cell:
    """A scripted turn that calls `run_cell`, and the prose it came with.

    The tests say `block("x = 1")` and mean "a turn that runs this cell". That intent is
    unchanged by the channel; what changed is that the cell now leaves the model as a
    `tool_use` block the API stops on, rather than as a fence in prose that it does not.
    """

    source: str
    prose: str = ""
    calls: int = 1
    """How many `run_cell` calls this turn makes. More than one is the discipline
    failure, not a feature."""
    name: str = "run_cell"


def _blocks(text, made: int) -> list:
    if not isinstance(text, Cell):
        return [TextBlock(text=text)]
    out: list = []
    if text.prose:
        out.append(TextBlock(text=text.prose))
    for index in range(text.calls):
        out.append(ToolUseBlock(id=f"call-{made}-{index}", name=text.name,
                                input={"source": text.source}))
    return out


def block(source: str, prose: str = "") -> Cell:
    return Cell(source=source, prose=prose)


def observed(provider, call: int = 1) -> list:
    """The blocks the desk put in front of the model on the way into `call`.

    A cell's result arrives inside a `tool_result` now, so the blocks that used to sit
    directly in the user message sit one level down. Tests ask what the desk *said*, not
    which envelope it came in, so this unwraps and they do not have to."""
    content = provider.calls[call]["messages"][-1]["content"]
    out: list = []
    for piece in content:
        if isinstance(piece, dict) and piece.get("type") == "tool_result":
            inner = piece.get("content")
            out.extend(inner if isinstance(inner, list)
                       else [{"type": "text", "text": str(inner)}])
        else:
            out.append(piece)
    return out


def said(provider, call: int = 1) -> str:
    """Everything the desk said on the way into `call`, as one string."""
    return "\n".join(b.get("text", "") for b in observed(provider, call)
                      if isinstance(b, dict) and b.get("type") == "text")


DONE = block(f'print("{CAD_DONE}")')


def desk(backend, store, texts, delay: float = 0.0, interject=None, **cfg_kwargs):
    cfg = Config(llm_api_key="k", model="claude-opus-5", **cfg_kwargs)
    made = CadDesk(cfg, backend, store, "/work/study", interject=interject)
    made.provider = ScriptedProvider(texts, delay=delay)
    return made


def says(*lines):
    """An `interject` that hands over one line per call, then nothing."""
    pending = list(lines)
    return lambda: pending.pop(0) if pending else None


def answers(backend, mapping, default=ExecResult(0, "", False, None)):
    """Route exec by a substring of the command, so tests read as intent."""
    real = backend.exec

    def routed(cmd, cwd=None, timeout_s=120, *, background=False):
        real(cmd, cwd, timeout_s, background=background)
        for needle, result in mapping.items():
            if needle in cmd:
                return result
        return default

    backend.exec = routed
    return backend


def kernelled(backend, mapping=None, default=None, raises=None):
    """Give the fake workspace a kernel, and route cells by a substring of the source.

    `raises` is called with the cell and its one-based number and may return an
    exception, which is how a kernel dying under a running cell is staged.
    """
    state = SimpleNamespace(started=[], cells=[], polls=0, interrupts=0, restarts=0,
                            poll_result=None)
    default = default or CellResult(ok=True, stdout="")

    def kernel_start(cwd):
        state.started.append(cwd)
        return "session-under-test"

    def kernel_run(code, timeout_s=STEP_TIMEOUT_S):
        state.cells.append(code)
        if raises is not None:
            boom = raises(code, len(state.cells))
            if boom is not None:
                raise boom
        for needle, result in (mapping or {}).items():
            if needle in code:
                return result
        return default

    def kernel_poll():
        state.polls += 1
        return state.poll_result or CellResult(ok=True, stdout="")

    def kernel_interrupt():
        state.interrupts += 1

    def kernel_restart():
        state.restarts += 1

    backend.kernel_start = kernel_start
    backend.kernel_run = kernel_run
    backend.kernel_poll = kernel_poll
    backend.kernel_interrupt = kernel_interrupt
    backend.kernel_restart = kernel_restart
    backend.kernel = state
    return backend


PASSES = Check(ok=True, cells=3750, faces=15000, points=7600, two_d=True,
               checkmesh="Mesh OK.", regions=[""],
               render="cad/renders/cad_look.png",
               render_abs="/work/study/cad/renders/cad_look.png",
               patches=[{"name": "inlet", "type": "patch", "nFaces": 20,
                         "normal": [-1, 0, 0]}])
REFUSES = Check(ok=False, missing=["nothing has been meshed yet in cad/"])


def checking(monkeypatch, *verdicts):
    """Install the finish check's answers, in order; the last one repeats.

    The check itself is `tests/test_cad_check.py`'s subject. What is under test here is
    what the loop does with a verdict, so the verdict is handed over rather than earned.
    """
    calls: list[dict] = []

    def verify(backend, case_dir, case_rel, request="", script=""):
        calls.append({"case_dir": case_dir, "case_rel": case_rel,
                      "request": request, "script": script})
        return verdicts[min(len(calls) - 1, len(verdicts) - 1)]

    monkeypatch.setattr("openreynolds.cad.agent.verify", verify)
    return calls


def thread(provider, index=-1) -> str:
    """Everything the person's side of the thread said, by the given call.

    Unwraps `tool_result`, because a cell's result is inside one now and the thread is
    still the thread."""
    out: list[str] = []
    for m in provider.calls[index]["messages"]:
        if m.get("role") != "user" or not isinstance(m.get("content"), list):
            continue
        for b in m["content"]:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_result":
                inner = b.get("content")
                if isinstance(inner, list):
                    out.extend(x.get("text", "") for x in inner if isinstance(x, dict))
                else:
                    out.append(str(inner))
            else:
                out.append(b.get("text", ""))
    return "\n".join(out)


# -- reading a message ---------------------------------------------------------


def turn_of(text, made: int = 1):
    """A `Turn` as the provider would hand one over, from the scripted shorthand."""
    return Turn(content=_blocks(text, made), provider="anthropic",
                stop_reason="tool_use" if isinstance(text, Cell) else "end_turn")


def test_the_one_call_is_the_action():
    ids, source, complaint, _ = parse_action(turn_of(block("x = 1", "thinking")))
    assert source == "x = 1" and complaint == "" and len(ids) == 1
    # The prose alongside is not the action and never was.
    assert parse_action(turn_of(block("x = 1")))[1] == "x = 1"


def test_no_call_and_two_calls_are_both_told_what_happened():
    """Both complaints came across from the bash desk verbatim, because neither was ever
    about the language in the fence -- they are about the model sending one action, or
    explaining itself at length instead of acting. The channel changed underneath them
    and the discipline they ask for did not."""
    ids, source, complaint, _ = parse_action(
        turn_of("I will now consider the geometry at length."))
    assert source == "" and ids == [] and "called no tool" in complaint
    ids, source, complaint, _ = parse_action(turn_of(Cell(source="a = 1", calls=2)))
    assert source == ""
    assert "2 tool calls" in complaint and "none of them ran" in complaint
    # Every call is named, because every call has to be answered.
    assert len(ids) == 2


def test_a_call_with_no_source_is_told_so_rather_than_running_nothing():
    ids, source, complaint, _ = parse_action(turn_of(Cell(source="   ")))
    assert source == "" and len(ids) == 1 and "no source" in complaint


def test_a_tool_that_does_not_exist_is_named_in_the_complaint():
    ids, source, complaint, _ = parse_action(turn_of(Cell(source="x = 1", name="bash")))
    assert source == "" and len(ids) == 1 and "bash" in complaint


def test_a_message_with_no_block_costs_a_turn_and_not_a_step(backend, store, monkeypatch):
    """The complaint goes back and the run continues -- it is a nudge, not a failure."""
    checking(monkeypatch, PASSES)
    kernelled(backend)
    made = desk(backend, store, ["no cell here", DONE])
    result = made.run("a duct")
    assert result.ok
    assert result.steps == []
    assert "called no tool" in said(made.provider)


# -- running one --------------------------------------------------------------


def test_a_step_runs_as_a_cell_and_reports_whether_it_raised(backend, store, monkeypatch):
    checking(monkeypatch, PASSES)
    kernelled(backend, {"body =": CellResult(ok=False, stdout="", error="NameError: x",
                                             traceback="Traceback: boom")})
    made = desk(backend, store, [block("body = x"), DONE])
    result = made.run("a duct", case="valve")
    assert result.case_rel == "valve" and result.case_dir == "/work/study/valve"
    assert backend.execs[0] == "mkdir -p /work/study/valve"
    assert backend.kernel.started == ["/work/study/valve"]
    step = result.steps[0]
    assert step.cmd == "body = x" and step.exit_code == 1
    observation = said(made.provider)
    assert observation.startswith("exit 1")
    assert "Traceback: boom" in observation


def test_what_a_cell_drew_comes_back_without_anybody_naming_it(backend, store, monkeypatch):
    """The whole reason the channel is a kernel. Under bash this was a guess -- scrape
    `.png` out of the command text, stat it, hope -- and the guess is gone."""
    checking(monkeypatch, PASSES)
    kernelled(backend, {"plot": CellResult(ok=True, stdout="drawn", images=[RAW_PNG])})
    made = desk(backend, store, [block("fig = 1  # plot the body"), DONE])
    result = made.run("a duct")
    assert result.steps[0].image == "1 picture"
    blocks = observed(made.provider)
    assert [b["type"] for b in blocks] == ["text", "image"]
    assert blocks[1]["source"]["media_type"] == "image/png"
    assert blocks[1]["source"]["data"] == images.attachment(RAW_PNG, "image/png")["source"]["data"]


def test_a_cell_that_drew_forty_times_does_not_put_forty_pictures_in_one_message(
        backend, store, monkeypatch):
    """Eviction counts messages, so a cell that drew in a loop would slip past it whole
    -- forty renders in one observation, and every ceiling the thread has gone at once.
    The channel hands them all over on purpose; what to do about it is decided here."""
    from openreynolds.cad.agent import IMAGES_PER_CELL

    checking(monkeypatch, PASSES)
    kernelled(backend, {"loop": CellResult(ok=True, stdout="", images=[RAW_PNG] * 40)})
    made = desk(backend, store, [block("fig = 1  # loop over sections"), DONE])
    result = made.run("a duct")
    blocks = observed(made.provider)
    assert sum(b["type"] == "image" for b in blocks) == IMAGES_PER_CELL
    assert "40 pictures came back" in result.steps[0].output


def test_a_cell_that_outran_its_window_is_polled_rather_than_killed(backend, store,
                                                                    monkeypatch):
    """A bash timeout lost nothing, because state lived on disk. Killing a cell would
    lose every binding in the session, so the window expiring is news, not an ending."""
    checking(monkeypatch, PASSES)
    kernelled(backend, {"slow": CellResult(ok=True, stdout="working", seconds=241.0,
                                           still_running=True)})
    made = desk(backend, store, [block("body = 1  # slow"), block("n = 1"), DONE])
    result = made.run("a duct")
    assert backend.kernel.polls == 1
    assert "still running after" in thread(made.provider, 1)
    assert "was not killed" in thread(made.provider, 1)
    assert result.ok


def test_a_slow_cell_that_finishes_is_judged_when_it_does(backend, store, monkeypatch):
    """It is not enough to report the late result: a cell that took four minutes and
    worked belongs in the script, or it is the one cell missing from the file."""
    checking(monkeypatch, PASSES)
    kernelled(backend, {"slow": CellResult(ok=True, stdout="", seconds=241.0,
                                           still_running=True)})
    made = desk(backend, store, [block("body = 1  # slow"), block("n = 1"), DONE])
    backend.kernel.poll_result = CellResult(ok=True, stdout="finished", seconds=300.0)
    result = made.run("a duct")
    assert "body = 1  # slow" in result.script
    assert "finished" in thread(made.provider, 2)


def test_a_workspace_that_refuses_a_cell_does_not_end_the_run(backend, store, monkeypatch):
    checking(monkeypatch, PASSES)
    kernelled(backend, raises=lambda code, n: RuntimeError("sandbox gone")
              if "body" in code else None)
    made = desk(backend, store, [block("body = 1"), DONE])
    result = made.run("a duct")
    assert result.steps[0].exit_code == -1
    assert result.ok


def test_an_overloaded_endpoint_is_tried_once_more(backend, store, monkeypatch):
    """A desk five minutes into a build cannot resume -- the next call starts a clean
    thread -- so losing one to a 529 is expensive. A 400 is not retried: it is a fact
    about the request."""
    monkeypatch.setattr("openreynolds.cad.agent.RETRY_PAUSE_S", 0)
    checking(monkeypatch, PASSES)
    kernelled(backend)
    made = desk(backend, store, [ProviderError("529 overloaded", 529), DONE])
    result = made.run("a duct")
    assert result.ok and not result.error

    made = desk(backend, store, [ProviderError("400 bad request", 400), DONE])
    result = made.run("a duct")
    assert result.stopped == "provider" and "400" in result.error


# -- the picture that did not arrive ------------------------------------------


def test_a_cell_that_drew_and_sent_nothing_back_is_told_so(backend, store, monkeypatch):
    """`matplotlib.use("Agg")` is the one line that silently undoes this channel.

    It is what a model writes from habit: in headless bash CFD scripting Agg plus
    `savefig` is mandatory, and there the figure was only ever going to reach anybody as
    a file. In a kernel the cell succeeds, prints nothing unusual, writes its PNG, and
    no image comes back -- so the desk believes it has looked at the shape and has not.
    That is the failure the picture exists to prevent, arriving dressed as a success.
    """
    checking(monkeypatch, PASSES)
    kernelled(backend, {"savefig": CellResult(ok=True, stdout="", images=[])})
    made = desk(backend, store, [
        block('import matplotlib\nimport matplotlib.pyplot as plt\n'
              'matplotlib.use("Agg")\nplt.savefig("look.png")'), DONE])
    result = made.run("a duct")
    said = result.steps[0].output
    assert "no picture came back from this cell" in said
    assert 'matplotlib.use("Agg")' in said
    assert "display(fig)" in said


def test_a_cell_that_drew_and_sent_a_picture_is_left_alone(backend, store, monkeypatch):
    checking(monkeypatch, PASSES)
    kernelled(backend, {"plt": CellResult(ok=True, stdout="", images=[RAW_PNG])})
    made = desk(backend, store, [block("import matplotlib.pyplot as plt\n"
                                       "fig = plt.figure()\nfig"), DONE])
    result = made.run("a duct")
    assert "no picture came back" not in result.steps[0].output


def test_a_cell_that_never_drew_is_not_lectured_about_pictures(backend, store, monkeypatch):
    checking(monkeypatch, PASSES)
    kernelled(backend)
    made = desk(backend, store, [block("PLATE_L_M = 0.120"), DONE])
    result = made.run("a duct")
    assert "no picture came back" not in result.steps[0].output


# -- the finish ---------------------------------------------------------------


def test_the_token_in_a_comment_is_not_a_finish(backend, store, monkeypatch):
    """The brief hands the desk the word, so it writes the word -- in a comment above
    the cell it actually wants run. Matching the word anywhere meant that cell was never
    executed and the check's refusal came back as the answer to it."""
    checking(monkeypatch, PASSES)
    kernelled(backend)
    made = desk(backend, store, [
        block(f'# will print("{CAD_DONE}") once checkMesh passes\nbody = 1'), DONE])
    result = made.run("a duct")
    assert [s.cmd.splitlines()[-1] for s in result.steps] == ["body = 1"]
    assert result.ok


@pytest.mark.parametrize("source,finishes", [
    (f'print("{CAD_DONE}")', True),
    (f"print('{CAD_DONE}')", True),
    (f'  print( "{CAD_DONE}" )  ', True),
    (f'print("{CAD_DONE}"); body = 1', False),
    (f'print("not {CAD_DONE} yet")', False),
    (f'# {CAD_DONE}\nbody = 1', False),
])
def test_what_counts_as_saying_done(source, finishes):
    assert _is_finish(source) is finishes


def test_done_is_checked_not_believed(backend, store, monkeypatch):
    """The whole point of the gate: the desk says done, the machine says otherwise, and
    the run continues with the reason in front of it."""
    calls = checking(monkeypatch, REFUSES, PASSES)
    kernelled(backend)
    made = desk(backend, store, [block(f'print("{CAD_DONE}")', "built it"),
                                 block("body = 1"),
                                 block(f'print("{CAD_DONE}")', "now it is real")])
    result = made.run("a duct")
    assert result.ok
    assert len(calls) == 2
    refusal = said(made.provider)
    assert "did not pass" in refusal and "nothing has been meshed yet" in refusal
    assert result.summary == "now it is real"


def test_the_check_is_handed_the_accepted_cells_as_the_script(backend, store, monkeypatch):
    """The replay criterion is C6's, and this is the seam: what it replays is the log."""
    calls = checking(monkeypatch, PASSES)
    kernelled(backend)
    made = desk(backend, store, [block("PLATE_L_M = 0.120"),
                                 block("width = PLATE_L_M / 2"), DONE])
    result = made.run("a plate")
    assert "PLATE_L_M = 0.120" in calls[0]["script"]
    assert "width = PLATE_L_M / 2" in calls[0]["script"]
    assert calls[0]["script"] == result.script


def test_the_result_carries_the_facts_and_the_picture(backend, store, monkeypatch):
    checking(monkeypatch, PASSES)
    kernelled(backend)
    backend.files["/work/study/cad/renders/cad_look.png"] = RAW_PNG
    made = desk(backend, store, [block(f'print("{CAD_DONE}")', "a 2D valve")])
    result = made.run("a valve")
    assert result.ok and result.check.cells == 3750 and result.check.two_d
    assert result.png == RAW_PNG
    assert result.tokens == {"input": 10, "output": 5}
    joined = "\n".join(result.check.lines())
    assert "inlet" in joined and "Mesh OK." in joined


# -- the nudge ----------------------------------------------------------------


def test_a_desk_that_has_meshed_nothing_is_told_so(backend, store, monkeypatch):
    """The measured failure: on the hardest prompt the desk spent its whole budget
    deriving tangent geometry in closed form and never ran a mesher at all -- 942 s,
    eight commands, nothing on disk."""
    checking(monkeypatch, REFUSES)
    kernelled(backend)
    made = desk(backend, store, [block("body = 1")],
                mesher_max_steps=NUDGE_AT_STEP + 1)
    made.run("a valve")
    said = "\n".join(thread(made.provider, i) for i in range(len(made.provider.calls)))
    assert "nothing in the case directory has been meshed yet" in said
    assert "templates" in said


def test_a_desk_that_has_meshed_is_left_alone(backend, store, monkeypatch):
    checking(monkeypatch, REFUSES)
    kernelled(backend)
    answers(backend, {"polyMesh": ExecResult(0, MESHED, False, None)})
    made = desk(backend, store, [block("body = 1")],
                mesher_max_steps=NUDGE_AT_STEP + 1)
    made.run("a valve")
    said = "\n".join(thread(made.provider, i) for i in range(len(made.provider.calls)))
    assert "has been meshed yet" not in said


def test_a_conjugate_case_with_two_meshed_regions_is_not_nudged(backend, store, monkeypatch):
    """`constant/polyMesh` or `constant/*/polyMesh`. A CHT run that meshed air and solid
    and has no singular mesh has meshed; a nudge here is the same bug the finish check
    and the old command-grep already had in two other places."""
    checking(monkeypatch, REFUSES)
    kernelled(backend)
    answers(backend, {"polyMesh": ExecResult(0, CHT, False, None)})
    made = desk(backend, store, [block("body = 1")],
                mesher_max_steps=NUDGE_AT_STEP + 1)
    made.run("a heat exchanger")
    said = "\n".join(thread(made.provider, i) for i in range(len(made.provider.calls)))
    assert "has been meshed yet" not in said


def test_meshing_on_step_three_is_not_nudged_at_step_twelve(backend, store, monkeypatch):
    """The question is what is on disk, not what the transcript said. In a kernel, prep
    and meshing both look like Python, and `snappyHexMesh` may be inside a `Popen`
    string or not appear at all."""
    checking(monkeypatch, REFUSES)
    kernelled(backend)
    state = {"steps": 0}
    real = backend.exec

    def routed(cmd, cwd=None, timeout_s=120, *, background=False):
        real(cmd, cwd, timeout_s, background=background)
        if "polyMesh" in cmd:
            return ExecResult(0, MESHED if state["steps"] >= 3 else "", False, None)
        return ExecResult(0, "", False, None)

    backend.exec = routed
    made = desk(backend, store, [block("body = 1")], mesher_max_steps=NUDGE_AT_STEP + 1)
    original = made._cell

    def counted(source, reasoning, messages, ids):
        state["steps"] += 1
        return original(source, reasoning, messages, ids)

    made._cell = counted
    made.run("a valve")
    said = "\n".join(thread(made.provider, i) for i in range(len(made.provider.calls)))
    assert "has been meshed yet" not in said


def test_the_nudge_never_runs_a_cell_in_the_desks_kernel(backend, store, monkeypatch):
    """A harness cell would mutate the session the desk is reasoning about, would have
    to be filtered out of the cell log, and would fail outright while the kernel is busy
    with a cell that outran its window. So the loop's own bookkeeping goes through
    `exec` and the kernel sees the desk's cells and nothing else."""
    checking(monkeypatch, REFUSES)
    kernelled(backend)
    made = desk(backend, store, [block("body = 1")], mesher_max_steps=NUDGE_AT_STEP + 1)
    result = made.run("a valve")
    assert backend.kernel.cells == [step.cmd for step in result.steps]
    assert any("polyMesh" in cmd for cmd in backend.execs)


# -- budgets ------------------------------------------------------------------


def test_running_out_of_steps_still_checks_what_is_on_disk(backend, store, monkeypatch):
    """A run that never said done may still have left a mesh; it is looked at either
    way, because the one thing this desk exists to end is an unexamined mesh."""
    calls = checking(monkeypatch, PASSES)
    kernelled(backend)
    made = desk(backend, store, [block("body = 1")], mesher_max_steps=3)
    result = made.run("a duct")
    assert len(result.steps) == 3
    assert result.stopped == "steps"
    assert result.ok and result.check.cells == 3750
    assert len(calls) == 1


def test_running_out_of_time_stops_the_run(backend, store, monkeypatch):
    checking(monkeypatch, REFUSES)
    kernelled(backend)
    made = desk(backend, store, [block("body = 1")], delay=0.02,
                mesher_max_seconds=0.01)
    result = made.run("a duct")
    assert result.stopped == "time" and not result.ok


def test_a_model_failure_is_reported_rather_than_raised(backend, store, monkeypatch):
    checking(monkeypatch, REFUSES)
    kernelled(backend)
    made = desk(backend, store, [ProviderError("429 overloaded", 429)])
    result = made.run("a duct")
    assert not result.ok and result.stopped == "provider"
    assert "429 overloaded" in result.error


def test_a_model_failure_after_the_build_still_reports_the_build(backend, store,
                                                                 monkeypatch):
    """A T10 run built 91,000 cells, hit a 400 on its next model call, and the tool
    result said "nothing was meshed" -- a wrong answer, not a cautious one."""
    checking(monkeypatch, PASSES)
    kernelled(backend)
    made = desk(backend, store, [ProviderError("400 bad request", 400)])
    result = made.run("a duct")
    assert result.ok and result.check.cells == 3750
    assert "400 bad request" in result.error


def test_an_empty_turn_is_not_sent_back_to_the_api(backend, store, monkeypatch):
    """A turn that is all reasoning and no words carries an empty text block, and the
    Messages API refuses the whole next request because of it -- which ended a T09 run
    three steps into a mesh that was going fine."""
    checking(monkeypatch, PASSES)
    kernelled(backend)
    made = desk(backend, store, ["", DONE])
    result = made.run("a duct")
    assert result.ok
    sent = made.provider.calls[1]["messages"]
    assert all(m["role"] != "assistant" for m in sent)
    assert "arrived empty" in sent[-1]["content"][0]["text"]


def test_a_repeated_empty_turn_is_told_it_ran_out_of_room_and_how_often(
    backend, store, monkeypatch
):
    """The dropped turn is the right call and it is not enough on its own.

    An all-reasoning reply cannot be sent back -- the Messages API refuses the empty
    text block -- so the turn is dropped. But then the thread the desk reads carries no
    record that it ever tried, and it starts over: sixteen turns of sixteen on the Tesla
    valve, each re-deriving the same arc geometry, three of them opening "I should first
    explore the environment", none of them aware they had been cut off. Two harness
    faults compounded into a loop with no exit -- the reply budget covering thinking as
    well as words, and the attempt leaving no trace. This pins the second.
    """
    checking(monkeypatch, PASSES)
    kernelled(backend)
    made = desk(backend, store, ["", "", "", DONE])
    result = made.run("a duct")
    assert result.ok

    nudges = [
        block["text"]
        for call in made.provider.calls
        for message in call["messages"]
        if message["role"] == "user"
        for block in message["content"]
        if block.get("type") == "text" and "arrived empty" in block.get("text", "")
    ]
    assert nudges, "an empty turn must still be answered"

    # The first is the plain one; a repeat has to say why and how many times.
    assert "3 times now" in nudges[-1], nudges[-1]
    assert "reasoning" in nudges[-1] and "cut off" in nudges[-1]
    assert "one short cell" in nudges[-1], "it needs something to do differently"

    # And the assistant turn is still dropped, because the API still refuses it.
    for call in made.provider.calls:
        assert all(m["role"] != "assistant" for m in call["messages"])


def test_the_reply_budget_leaves_room_to_speak_after_thinking(backend, store):
    """`thinking={"type": "adaptive"}` spends the reply's own allowance, so a budget
    tight enough for a hard prompt's reasoning leaves nothing for the sentence that
    carries the cell. Measured at 8,000: every reply came back `max_tokens` with
    `block_types=['thinking']` and zero characters of text, so nothing ever ran."""
    from openreynolds.cad import agent

    assert agent.MAX_REPLY_TOKENS >= 16000, (
        "thinking and words share this budget; 8,000 starved the words on hard prompts"
    )


def test_a_turn_with_words_and_an_empty_block_keeps_the_words(backend, store, monkeypatch):
    checking(monkeypatch, REFUSES)
    kernelled(backend)
    made = desk(backend, store, [block("body = 1")], mesher_max_steps=1)

    def stream(**kwargs):
        made.provider.calls.append({**kwargs,
                                    "messages": [dict(m) for m in kwargs["messages"]]})
        return Turn(content=[TextBlock(text=""),
                             ToolUseBlock(id="c1", name="run_cell",
                                          input={"source": "body = 1"})],
                    provider="anthropic", stop_reason="tool_use", tokens={})

    made.provider.stream = stream
    result = made.run("a duct")
    assert [s.cmd for s in result.steps] == ["body = 1"]


# -- the person ---------------------------------------------------------------


def test_the_persons_own_words_travel_with_the_job(backend, store, monkeypatch):
    """The request is the calling agent's paraphrase. What the person typed is on disk
    in the session's own transcript, and a detail dropped in the paraphrase used to be
    one the desk could not recover and did not know was missing."""
    checking(monkeypatch, PASSES)
    kernelled(backend)
    store.append_message("user", "build me a tesla valve, 4 loops")
    store.append_message("assistant", "I will call the CAD desk.")
    store.append_message("user", "the loops must not touch each other")
    made = desk(backend, store, [DONE])
    made.run("a valve with four bypass loops")

    first = made.provider.calls[0]["messages"][0]["content"][0]["text"]
    assert "a valve with four bypass loops" in first          # the job
    assert "build me a tesla valve, 4 loops" in first         # and their words
    assert "the loops must not touch each other" in first
    assert "I will call the CAD desk." not in first           # only the person's
    assert "their own words" in first


def test_the_geometry_it_was_handed_reaches_the_desk_verbatim(backend, store, monkeypatch):
    """A prep job and an authoring job are the same desk; the file is what tells them
    apart, and a path paraphrased is a path that does not open."""
    checking(monkeypatch, PASSES)
    kernelled(backend)
    made = desk(backend, store, [DONE])
    made.run("prepare this", geometry="/work/study/uploads/chassis v2.step")
    first = made.provider.calls[0]["messages"][0]["content"][0]["text"]
    assert "/work/study/uploads/chassis v2.step" in first
    assert "do not redesign it" in first.lower()


def test_a_session_with_no_transcript_still_builds(backend, store, monkeypatch):
    class Broken:
        def recent_messages(self, limit=30):
            raise OSError("no transcript here")

    checking(monkeypatch, PASSES)
    kernelled(backend)
    made = desk(backend, Broken(), [DONE])
    assert made.run("a duct").ok


def test_a_remark_typed_mid_run_reaches_the_desk_at_the_next_step(backend, store,
                                                                  monkeypatch):
    """Before this, a remark waited out the whole call -- up to fifteen minutes -- and
    then went to the calling agent, which had to start the desk again from scratch."""
    checking(monkeypatch, PASSES)
    kernelled(backend)
    made = desk(backend, store, [block("body = 1"), DONE],
                interject=says(None, "actually make it 2 mm wider"))
    result = made.run("a duct")

    assert result.remarks == ["actually make it 2 mm wider"]
    said = thread(made.provider)
    assert "The person watching just said" in said
    assert "actually make it 2 mm wider" in said
    assert "takes precedence" in said


def test_a_remark_arriving_with_done_keeps_the_run_going(backend, store, monkeypatch):
    """Somebody speaking in the same breath as "done" is the newer instruction, so the
    run does not close on a shape that was right one message ago."""
    calls = checking(monkeypatch, PASSES)
    kernelled(backend)
    made = desk(backend, store, [DONE, block("body = 1"), DONE],
                interject=says("no, 3 mm wide"))
    result = made.run("a duct")
    assert result.ok
    assert result.remarks == ["no, 3 mm wide"]
    assert len(calls) == 1                     # the first "done" never reached the check
    assert [s.cmd for s in result.steps] == ["body = 1"]


def test_a_drain_that_throws_does_not_end_the_run(backend, store, monkeypatch):
    def broken():
        raise RuntimeError("the reader is gone")

    checking(monkeypatch, PASSES)
    kernelled(backend)
    made = desk(backend, store, [DONE], interject=broken)
    result = made.run("a duct")
    assert result.ok and result.remarks == []


# -- the thread ---------------------------------------------------------------


def test_old_pictures_leave_the_thread_and_their_words_stay():
    """Unchanged from the bash desk, because it was never about where a picture came
    from: a render is 800-1300 tokens and the fifth-oldest one is answering a question
    that was settled three steps ago."""
    messages = []
    for i in range(4):
        messages.append({"role": "user", "content": [
            {"type": "text", "text": f"exit 0 (step {i})"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": "x"}},
        ]})
        _evict(messages)
    with_images = [m for m in messages
                   if any(b.get("type") == "image" for b in m["content"])]
    assert len(with_images) == KEEP_IMAGES
    assert all(any("exit 0 (step" in b.get("text", "") for b in m["content"])
               for m in messages)
    assert any("no longer shown" in b.get("text", "") for b in messages[0]["content"])


# -- a kernel that died -------------------------------------------------------


def test_a_kernel_death_mid_run_is_survivable(backend, store, monkeypatch):
    """A live process does not survive being preempted, recycled or restarted. A file
    does -- which is the whole argument for the log being the truth and the session
    being the convenience, exercised rather than asserted."""
    checking(monkeypatch, PASSES)
    dead = {"done": False}

    def raises(code, number):
        if "second" in code and not dead["done"]:
            dead["done"] = True
            return BackendError("the kernel stopped while this cell was running",
                                "kernel_gone")
        return None

    kernelled(backend, raises=raises)
    made = desk(backend, store, [block("PLATE_L_M = 0.120  # first"),
                                 block("width = PLATE_L_M / 2  # second"), DONE])
    result = made.run("a plate")
    assert result.ok
    assert backend.kernel.restarts == 1
    # the accepted log went back into the new kernel, then the cell that died ran again
    assert any("PLATE_L_M = 0.120" in code and "cell 1" in code
               for code in backend.kernel.cells)
    assert [s.cmd for s in result.steps] == ["PLATE_L_M = 0.120  # first",
                                             "width = PLATE_L_M / 2  # second"]
    assert "The kernel died and has been replaced" in thread(made.provider)
    assert "width = PLATE_L_M / 2" in result.script


def test_a_cell_still_running_at_the_end_is_stopped_before_anything_is_measured(
        backend, store, monkeypatch):
    """The finish check is about to read the case directory. A cell still writing into
    it would have the check reading a half-written case and the caller told about it as
    though it were finished."""
    checking(monkeypatch, PASSES)
    kernelled(backend, {"slow": CellResult(ok=True, stdout="", seconds=241.0,
                                           still_running=True)})
    made = desk(backend, store, [block("body = 1  # slow")], mesher_max_steps=1)
    made.run("a duct")
    assert backend.kernel.interrupts == 1


# -- the cell log --------------------------------------------------------------


def test_the_static_check_lets_a_legitimately_scoped_cell_through():
    """Names bound by accepted cells, imports, builtins, parameters, comprehension
    targets and locals are all in scope. A check that refused any of these would be
    refusing correct cells, which is worse than the thing it is guarding against."""
    log = CellLog()
    assert log.accept(log.propose("import math\nPLATE_L_M = 0.120")) == ""
    assert log.accept(log.propose(
        "def widths(count, scale=PLATE_L_M):\n"
        "    return [scale * i / count for i in range(count)]\n"
        "all_widths = widths(4)\n"
        "total = sum(all_widths) + math.pi\n"
        "with open('x.txt', 'w') as handle:\n"
        "    handle.write(str(total))\n"
        "try:\n"
        "    body = int(total)\n"
        "except ValueError as exc:\n"
        "    body = str(exc)\n"
    )) == ""
    assert "body" in log.bound() and "widths" in log.bound() and "math" in log.bound()


def test_the_static_check_catches_a_name_no_accepted_cell_binds():
    assert free_names("width = PLATE_L_M / 2", set()) == ["PLATE_L_M"]
    assert free_names("width = PLATE_L_M / 2", {"PLATE_L_M"}) == []
    assert free_names("body = round(len('abc'))", set()) == []
    assert bound_names("import numpy as np\nfrom math import pi") == {"np", "pi"}


def test_an_attribute_or_an_item_assignment_binds_nothing_and_reads_the_base():
    assert free_names("holes[2] = 1", set()) == ["holes"]
    assert free_names("part.label = 'inlet'", set()) == ["part"]


def test_the_cell_that_ran_live_and_fails_in_sequence_is_refused_and_told_it_ran(
        backend, store, monkeypatch):
    """The dangerous case, constructed. A cell errored, so it never entered the log --
    but the name it bound before it errored is still live in the kernel, so the next
    cell using that name runs perfectly. Appending it would put a name in the script
    that the script does not bind, and the replay would fail minutes later with no
    explanation. It is refused; and because the kernel now holds state the log does not,
    the refusal has to say so, or the desk keeps building on a binding that will not
    exist.
    """
    checking(monkeypatch, PASSES)
    kernelled(backend, {"tried": CellResult(ok=False, stdout="", error="ValueError: no",
                                            traceback="ValueError: no")})
    made = desk(backend, store, [block("SCRATCH_M = 0.5  # tried and failed"),
                                 block("width = SCRATCH_M * 2"), DONE])
    result = made.run("a plate")

    assert "SCRATCH_M" not in result.script
    assert "width = SCRATCH_M * 2" not in result.script
    assert result.script == ""
    refusal = thread(made.provider)
    assert "`SCRATCH_M`" in refusal
    assert "That cell ran" in refusal
    assert "in your kernel session" in refusal and "still live there" in refusal
    assert "not** in the script" in refusal
    assert "Re-emit this cell" in refusal


def test_a_star_import_narrows_what_the_check_claims_rather_than_refusing_everything():
    """`from build123d import *` is how build123d is written, and it is the one
    construct that makes "what does this bind" unanswerable from the source. Claiming
    anyway would refuse `Box`, `Pos` and every other correct name; claiming nothing
    would give up the gate. So under a star import the check keeps only what it can
    still be sure of -- a name some earlier cell bound and no accepted cell did, which
    is never something a module's star import supplied."""
    log = CellLog()
    assert log.accept(log.propose("from build123d import *")) == ""
    assert log.accept(log.propose("body = Box(0.02, 0.01, 0.005)")) == ""

    log.propose("SCRATCH_M = 0.5")          # proposed, errored, never accepted
    refusal = log.accept(log.propose("width = SCRATCH_M * 2"))
    assert "`SCRATCH_M`" in refusal and "That cell ran" in refusal


def test_a_shell_magic_is_refused_with_the_reason_it_cannot_be_in_a_file():
    """The brief hands the desk `!blockMesh`, and in the kernel it is right. In the
    file it is a syntax error, so a cell carrying one would make the finish check's
    replay fail minutes later with nothing to read. It is caught where it happened."""
    log = CellLog()
    refusal = log.accept(log.propose("!blockMesh > log.blockMesh 2>&1"))
    assert "`!blockMesh > log.blockMesh 2>&1`" in refusal
    assert "IPython, not python" in refusal
    assert "subprocess.run" in refusal
    assert log.script() == ""
    assert log.accept(log.propose(
        "import subprocess\n"
        "subprocess.run(['blockMesh'], check=True)")) == ""


def test_a_name_only_the_kernel_supplies_is_refused_in_its_own_words():
    """`display` is in the session and not in a script, which is the difference this
    whole gate is about -- so the refusal says which one it is rather than claiming
    nothing binds it."""
    log = CellLog()
    refusal = log.accept(log.propose("fig = 1\ndisplay(fig)"))
    assert "`display`" in refusal
    assert "python3 build.py" in refusal
    assert "from IPython.display import display" in refusal


def test_the_script_carries_the_reasoning_that_came_with_each_cell(backend, store,
                                                                   monkeypatch):
    """The prose is the only record of why this shape is this shape, so it travels with
    the action rather than being thrown away with the turn."""
    checking(monkeypatch, PASSES)
    kernelled(backend)
    made = desk(backend, store, [
        block("PLATE_L_M = 0.120", "120 mm is the plate length the request named."), DONE])
    result = made.run("a plate")
    assert "# 120 mm is the plate length the request named." in result.script
    assert "PLATE_L_M = 0.120" in result.script
    compile(result.script, "build.py", "exec")


# -- the invariant, driven ------------------------------------------------------


@pytest.fixture
def workspace(tmp_path):
    """A real workspace with a real kernel in it. Nothing below is a mock of one: what
    is being asked is whether a script runs from empty, which a fake cannot answer."""
    pytest.importorskip("ipykernel", reason="the invariant is checked in a kernel")
    pytest.importorskip("jupyter_client", reason="the invariant is checked in a kernel")
    from openreynolds.backend.local import LocalBackend

    made = LocalBackend(root=tmp_path / "work", bashrc="")
    made.exec("mkdir -p study/cad")
    made.kernel_start(f"{made.workspace_root}/study/cad")
    yield made
    made.close()


def _sequence(rng: random.Random, index: int) -> list[str]:
    """One randomised drive: cells that bind, cells that poison, cells that borrow.

    Trivial arithmetic on purpose. The invariant is bookkeeping -- does a name the
    script does not bind get into the script -- and a gate that ran real geometry here
    would take minutes a sequence, which is how a gate becomes decorative.
    """
    texts: list[str] = []
    accepted: list[str] = []
    poisoned: list[str] = []
    for step in range(rng.randint(2, 7)):
        name = f"n{index}_{step}"
        choice = rng.random()
        if choice < 0.35 or not (accepted or poisoned):
            texts.append(block(f"{name} = {rng.randint(1, 9)}"))
            accepted.append(name)
        elif choice < 0.55:
            texts.append(block(f"{name} = {rng.randint(1, 9)}\n"
                               f"raise RuntimeError('this cell does not enter the log')"))
            poisoned.append(name)
        elif choice < 0.75 and accepted:
            texts.append(block(f"{name} = {rng.choice(accepted)} + 1"))
            accepted.append(name)
        elif poisoned:
            texts.append(block(f"{name} = {rng.choice(poisoned)} + 1"))
        elif accepted:
            texts.append(block(f"for {name} in range({rng.choice(accepted)} + 2):\n"
                               f"    pass"))
            accepted.append(name)
    texts.append(DONE)
    return texts


def test_the_append_invariant_holds_under_randomised_drive(workspace, store, monkeypatch):
    """Fifty drives, every one of them replayed in a kernel with nothing in it.

    Not sampled, because the thing being tested is a bookkeeping invariant and a
    bookkeeping invariant fails on the sequence nobody looked at. The cells are
    `n = 1`; what varies is which of them are accepted, which errored and left a live
    binding behind, and which later cells reach for one of those.
    """
    checking(monkeypatch, PASSES)
    rng = random.Random(20260911)
    refusals = 0
    for index in range(50):
        made = desk(workspace, store, _sequence(rng, index))
        result = made.run("a plate", case="cad")
        script = result.script
        refusals += sum("That cell ran" in text
                        for call in made.provider.calls
                        for m in call["messages"] if m.get("role") == "user"
                        for b in m["content"] if isinstance(b, dict)
                        for text in [b.get("text", "")])
        workspace.kernel_restart()
        replay = workspace.kernel_run(script or "pass", timeout_s=120)
        assert replay.ok, (f"sequence {index} left a script that does not run from "
                           f"empty: {replay.error}\n{script}")
    assert refusals, "no sequence ever produced a refusal, so nothing was being gated"


def test_a_checkpoint_is_a_cache_and_not_state(workspace, store, monkeypatch):
    """The §3 requirement, as a test. An expensive stage writes its B-rep to a STEP
    file and the cells after it load that instead of rebuilding it -- which is how a
    long build stays inside one step window. The file is a cache the script writes and
    reads, not a second source of truth, so deleting it and running the script from
    empty has to give the same geometry back. A log that only worked because the
    checkpoint survived fails here.
    """
    pytest.importorskip("build123d", reason="the checkpoint is a real STEP file")
    checking(monkeypatch, PASSES)
    case = f"{workspace.workspace_root}/study/cad"
    made = desk(workspace, store, [
        block('from build123d import Box, export_step, import_step\n'
              'PLATE_L_M = 0.020\n'
              'body = Box(PLATE_L_M, 0.010, 0.005)\n'
              'export_step(body, "prep.step")',
              "The import is the expensive stage, so it checkpoints."),
        block('loaded = import_step("prep.step")\n'
              'print(f"volume {loaded.volume:.9f}")'),
        block('open("volume.txt", "w").write(f"{loaded.volume:.9f}")'),
        DONE,
    ])
    result = made.run("a plate", case="cad")
    assert result.ok
    first = workspace.get_file(f"{case}/volume.txt").decode()
    assert float(first) > 0

    # The checkpoint is deleted, and with it the whole session. What is left is the file.
    workspace.exec("rm -f study/cad/prep.step study/cad/volume.txt")
    with pytest.raises(BackendError):
        workspace.stat(f"{case}/prep.step")
    workspace.exec("rm -rf study/replay && mkdir -p study/replay")
    workspace.put_file(f"{workspace.workspace_root}/study/replay/build.py",
                       result.script.encode("utf-8"))
    outcome = workspace.exec("python3 build.py", cwd=f"{workspace.workspace_root}/study/replay",
                             timeout_s=300)
    assert outcome.exit_code == 0, outcome.output
    again = workspace.get_file(f"{workspace.workspace_root}/study/replay/volume.txt").decode()
    assert again == first


# -- the brief ----------------------------------------------------------------


def test_the_task_message_names_the_directory_and_the_request():
    text = task_message("a 20 degree branch", "/work/s/cad", "cad")
    assert "a 20 degree branch" in text and "/work/s/cad" in text


def test_the_brief_the_loop_hands_over_carries_the_loops_own_window():
    assert f"{STEP_TIMEOUT_S} s" in system_prompt(STEP_TIMEOUT_S)


@pytest.mark.parametrize("given,expected", [
    (None, "cad"), ("", "cad"), ("valve", "valve"), ("../escape", "escape"),
    ("/work/etc", "work_etc"), ("a b", "a_b"),
])
def test_the_case_is_a_directory_name_under_the_study(given, expected, backend, store,
                                                      monkeypatch):
    checking(monkeypatch, REFUSES)
    kernelled(backend)
    made = desk(backend, store, [DONE], mesher_max_steps=2)
    assert made.run("x", case=given).case_rel == expected


def test_the_cell_channel_is_a_tool_the_api_enforces(backend, store, monkeypatch):
    """The one property the whole migration exists for.

    In the first baseline sweep the desk wrote fenced blocks into ordinary text and the
    harness regexed them out, so the turn boundary was a request in the brief rather than
    a rule of the protocol -- nothing stopped generation at the closing fence, and twice
    the desk carried on and wrote the cell's output itself. Across the 55 turns of the
    two runs that did it, `stop_reason` was `end_turn` every time and `tool_use` never.

    Sending a real tool moves the boundary into the API: the turn stops at `tool_use`,
    and a `tool_result` is a user-role block the model has no way to author. This asserts
    the tool is actually on the request -- the guarantee is worth nothing if we forget to
    ask for it."""
    checking(monkeypatch, PASSES)
    kernelled(backend)
    made = desk(backend, store, [block("x = 1"), DONE])
    assert made.run("a duct").ok
    for call in made.provider.calls:
        names = [t["name"] for t in call["tools"]]
        assert names == ["run_cell"], names
        schema = call["tools"][0]["input_schema"]
        assert schema["required"] == ["source"]


def test_a_fence_in_the_prose_is_prose():
    """It used to be the channel, which is exactly how a run could be talked into
    believing a cell had run. Now nothing a message *writes* runs -- only what it
    calls -- so a fenced block in the text is no more an action than a sentence is."""
    fenced = "Here is what I would run:\n```python\nblockMesh\n```"
    ids, source, complaint, _ = parse_action(turn_of(fenced))
    assert source == "" and ids == [] and "called no tool" in complaint
    assert not re.search(r"bash", parse_action(turn_of("no block"))[2])


def test_every_turn_is_reported_to_whatever_is_watching_from_outside(
    backend, store, monkeypatch
):
    """The heartbeat is per turn, not per step, and this is why.

    The pathology that went unnoticed for three consecutive runs was a desk producing
    replies and executing zero cells; each burned its full clock and was reported as an
    ordinary budget exhaustion. A step-based heartbeat cannot see it, because there are
    no steps. So a turn that ran nothing -- an all-reasoning reply, a message with no
    fenced block -- still beats, carrying the step count that is not moving.
    """
    checking(monkeypatch, PASSES)
    kernelled(backend)
    seen: list[dict] = []
    made = desk(backend, store, ["", "no block here", block("x = 1"), DONE])
    made.on_turn = lambda **fields: seen.append(fields)
    result = made.run("a duct")
    assert result.ok
    assert [row["turn"] for row in seen] == [1, 2, 3, 4]
    assert [row["fenced"] for row in seen] == [False, False, True, True]
    assert [row["steps"] for row in seen] == [0, 0, 0, 1]
    assert seen[0]["text_chars"] == 0


def test_each_beat_carries_the_running_token_total(backend, store, monkeypatch):
    """The accounting is handed out per turn, not kept until the run returns.

    A killed run never returns its `CadResult`, so a total that exists only there is a
    total nobody can read: the watcher signals the process group, SIGTERM terminates
    without unwinding, and the record keeps its defaults. That is how T5 of the first
    baseline sweep recorded $0.00 against a run `watch.json` clocked at 535 s.
    `replies.jsonl` cannot stand in -- it carries this turn's `output` and never the
    input or cache counts the price needs."""
    checking(monkeypatch, PASSES)
    kernelled(backend)
    seen: list[dict] = []
    made = desk(backend, store, [block("x = 1"), DONE])
    made.on_turn = lambda **fields: seen.append(fields)
    result = made.run("a duct")
    assert result.ok

    totals = [row.get("tokens") for row in seen]
    assert all(isinstance(row, dict) and row for row in totals), totals
    # Cumulative, so no key ever decreases and the last beat agrees with the result.
    for key in result.tokens:
        series = [row.get(key, 0) for row in totals]
        assert series == sorted(series), (key, series)
    assert totals[-1] == result.tokens


def test_a_watcher_that_raises_does_not_end_the_run(backend, store, monkeypatch):
    """The observer reports and is never consulted. A run that could be ended by the
    thing measuring it is a measurement of the measurement."""
    checking(monkeypatch, PASSES)
    kernelled(backend)
    made = desk(backend, store, [block("x = 1"), DONE])

    def explode(**fields):
        raise RuntimeError("the watcher fell over")

    made.on_turn = explode
    assert made.run("a duct").ok


def test_a_desk_can_refuse_and_that_is_a_terminal_not_a_failure(backend, store, monkeypatch):
    """T6 of the first baseline is why this exists.

    Its fixture is a STEP with the `LENGTH_UNIT` declaration emptied. The desk found the
    missing unit, wrote the guess into a constant, and shipped a mesh `checkMesh` passed
    -- and the run recorded `done`, `checkmesh_ok: true`, the corpus's sixth success. The
    brief had named that outcome in advance: "It passed by luck, and the next file is
    inches."

    Nothing was wrong with the desk's honesty. There was nowhere to land: a desk that
    stopped early scored `steps`, indistinguishable from failing, and one that guessed
    scored `done`.
    """
    checking(monkeypatch, PASSES)
    kernelled(backend)
    made = desk(backend, store, [
        block('print("CAD_REFUSED: the STEP declares no length unit")',
              "This file gives no unit and the extents fit both mm and m."),
    ])
    result = made.run("mesh the fluid volume in this STEP")

    assert result.stopped == "refused"
    assert result.error == "the STEP declares no length unit"
    assert not result.ok, "a refusal is not a mesh"
    # No finish check runs: there is nothing to check, and "nothing was meshed" would
    # bury the reason under a complaint about its absence.
    assert result.check is None


def test_the_refusal_token_in_a_comment_is_not_a_refusal(backend, store, monkeypatch):
    """The same rule `_is_finish` learned the hard way: the token is the whole cell, or
    it is a word the desk wrote while explaining itself."""
    from openreynolds.cad.agent import _refusal

    assert _refusal('print("CAD_REFUSED: no unit")') == "no unit"
    assert _refusal('print("CAD_REFUSED")') == "no reason given"
    assert _refusal('# print("CAD_REFUSED: no unit")\nbody = 1') is None
    assert _refusal("body = 1") is None
