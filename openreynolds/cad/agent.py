"""The loop: one python cell a step, in a kernel on the machine with OpenFOAM on it.

Small on purpose, and the same size it was: a thread of messages, a fenced block taken
out of each one, the block's output put back in, and a picture alongside it whenever the
cell drew one. Every mechanism this file does not have was in the thing it replaces, and
none of them were what made a mesh right.

Three things it owns, because they are what the model cannot check about itself:

* **the picture arrives** -- a cell that draws has what it drew attached to its output.
  Under bash this was a guess: scrape `.png` out of the command text, stat it, hope. A
  kernel returns display data, so what the cell drew is what comes back and there is
  nothing for anybody to remember to arrange;
* **the finish is verified** -- `print("CAD_DONE")` does not end the run, `check.py`
  does, and a failed check is handed back as work rather than reported as success;
* **the budget** -- steps and wall clock, so a run that is going nowhere stops going
  there.

And one the kernel adds: **the accepted cells are the script**. A cell that ran is not
automatically a cell that belongs in the log -- the invariant is that the concatenation
still runs, not that the cell worked -- so every cell goes past `CellLog.accept` before
it is appended, and a refusal is handed to the desk as work like any other failure.
`cells.py` carries that argument in full.

What is deliberately not here: any use of the desk's kernel for the harness's own
questions. The nudge and the finish check both ask the workspace through `exec`. A cell
the loop injected would mutate the session the desk is reasoning about, would have to be
kept out of the log by filtering, and would fail outright while the kernel is busy with
a cell that outran its window.
"""

from __future__ import annotations

import re
import shlex
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .. import images
from ..llm import Listener, ProviderError, make_provider
from .brief import CAD_DONE, remark_message, system_prompt, task_message
from .cells import Cell, CellLog
from .check import Check, mesh_regions, verify

MAX_STEPS = 30
"""Enough for a shape, a look, two or three revisions, a mesh and a finish. A run that
has not converged by here is not one step from converging."""

MAX_SECONDS = 900.0
"""Fifteen minutes, and it was briefly twenty, which the measurement then argued back.

Every desk that succeeded finished inside eleven minutes -- 1.9 to 5.2 across the six
2D and 3D acceptance runs, 10.6 for the aerofoil, which is the longest success on
record. The one that failed ran 942 s and eight commands and produced nothing at all.
So past roughly ten minutes this desk is not slow, it is stuck, and the cheapest thing
to do with the caller's time is give it back."""

STEP_TIMEOUT_S = 240
"""How long one cell holds the conversation. It does **not** kill the cell: a cell that
outruns it keeps running and is polled at the next step. Harness configuration, and
nothing a cell can reach -- the point of a step budget is to keep the conversation
moving, not to cap compute."""

RECOVERY_TIMEOUT_S = 900
"""How long the replay of the accepted log is given after a kernel died. Longer than a
step, because it is every accepted cell at once and nobody is waiting on a turn."""

MAX_REPLY_TOKENS = 24000
"""One reply's token budget -- **thinking and words share it**.

`llm/anthropic_api.py` asks for `thinking={"type": "adaptive"}`, so the model decides how
much to reason and that reasoning is spent out of this same allowance. At 8,000 a hard
authoring prompt spent the whole budget thinking and was cut off before it opened a text
block: measured on the Tesla valve, sixteen turns out of sixteen came back
`stop_reason=max_tokens` with `block_types=['thinking']` and **zero characters of text**,
so there was no fenced cell, nothing ran, and the run burned its clock in silence. The
easy prompts never saw it -- a box is transcribed rather than solved, and their replies
sat at 600-2,700 tokens.

Raised so that reasoning cannot starve the sentence that carries the work. It is a
ceiling, not a target: a reply that needs 700 tokens still costs 700.
"""
OUTPUT_CHARS = 6000
"""What comes back from one cell. Mesh logs are long and repetitive, and the news is at
both ends -- the traceback and the summary after it -- so a long output is cut in the
middle rather than truncated."""

TRANSCRIPT_ROWS = 40
"""How far back into the session's transcript to look for the person's own words."""

SAID_LINES = 6
"""How many of them travel with the job. The last few are the ones that are about this
geometry; a whole session's worth would bury the request in an older study's."""

RETRY_STATUSES = {408, 429, 500, 502, 503, 504, 529}
"""Model-API failures worth one more try: overloaded, rate-limited, gateway. A 400 or a
401 says the same thing twice."""

RETRY_PAUSE_S = 5.0

NUDGE_AT_STEP = 12
"""When to say out loud that there is still no mesh.

The failure this addresses, measured: on the hardest prompt the desk spent its whole
budget deriving tangent geometry in closed form and never ran a mesher at all -- 942 s,
eight commands, nothing on disk. Meanwhile the same model with bash alone built the
correct shape by drawing small numbered scripts and checking each one. A desk that has
not meshed anything after a dozen cells is answering the wrong question, and the answer
is not more thinking, it is a coarse mesh to look at."""

KEEP_IMAGES = 2
"""Pictures kept in the thread. Older observations keep their words and lose their
image: a render is 800-1300 tokens and the fifth-oldest one is answering a question
that was settled three steps ago."""

IMAGES_PER_CELL = 4
"""And how many one cell may send at once.

Eviction counts messages, so without this a single cell that drew in a loop puts its
forty figures into one observation and past every ceiling the thread has. The channel
caps at forty and hands them all over deliberately -- what to do with a cell that drew
forty times is the caller's judgement, and here the judgement is to show the first few
and say how many there were."""

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


@dataclass
class Step:
    """One lap, as the record keeps it.

    The field names are the ones the progress renderer already reads. `cmd` is the
    cell's source and `exit_code` is 0 or 1 off `CellResult.ok`, so a step still renders
    as a line of work with a status beside it and nothing downstream had to change.
    """

    cmd: str
    exit_code: int
    seconds: float
    output: str = ""
    image: str = ""
    """What came back as a picture, in words -- there is no path any more, because the
    bytes arrive with the result. Read for whether there was one."""


@dataclass
class CadResult:
    ok: bool = False
    case_rel: str = ""
    case_dir: str = ""
    summary: str = ""
    """The desk's own closing words -- what it built and what it did not check."""
    check: Check | None = None
    png: bytes | None = None
    steps: list[Step] = field(default_factory=list)
    seconds: float = 0.0
    tokens: dict[str, int] = field(default_factory=dict)
    error: str = ""
    remarks: list[str] = field(default_factory=list)
    """What the person said while this ran, in their words. Reported back so the calling
    agent is not the last to hear about a change it did not make."""
    stopped: str = ""
    """Empty when it finished on its own terms; else 'steps', 'time' or 'provider'."""
    script: str = ""
    """The accepted cells, concatenated. The artifact, not a record of it."""


class CadDesk:
    """The CAD desk, wired to a workspace, a kernel and a model.

    One instance per session; `run` is one geometry. It holds no state between runs
    beyond the provider client, so a second call starts from a clean thread and a clean
    log.
    """

    def __init__(self, cfg: Any, backend: Any, store: Any, home: str,
                 on_step: Callable[[Step], None] | None = None,
                 interject: Callable[[], str | None] | None = None,
                 on_turn: Callable[..., None] | None = None):
        self.cfg = cfg
        self.backend = backend
        self.store = store
        """The session's transcript, read for what the person actually said."""
        self.home = (home or backend.workspace_root).rstrip("/")
        self.toolbox = f"{str(backend.workspace_root).rstrip('/')}/.toolbox"
        """Where `cli.py` put the toolbox on *this* backend.

        Read off the backend rather than written down, because the two backends do not
        agree on it: the hosted workspace is `/work` and a local one is wherever it was
        rooted. The brief and the nudge both name this path, and naming it wrongly is
        not an error the desk can see -- it looks like a missing file, and it is
        answered by the desk going looking, which costs steps out of the budget the run
        is judged on."""
        self.on_step = on_step
        self.on_turn = on_turn
        """Told about **every** model turn, whether or not a cell ran.

        The heartbeat the observer outside the run reads (`buildup/heartbeat.py`), and
        the reason it is per turn rather than per step: the failure that went unnoticed
        for three consecutive runs last round was a desk producing replies and executing
        zero cells, and a step-based heartbeat cannot see it because there are no steps.

        It reports; it is never consulted. A raising hook does not end a run."""
        self.interject = interject
        """Drains anything the person has typed since the last call, or None.

        The same callable the main loop uses between its own tool calls. Held here as
        well because this desk holds the thread for minutes at a time, and a remark
        that waits that long is a remark that arrives after the thing it was about."""
        self.provider = make_provider(cfg)
        self.model = cfg.mesher_model or cfg.model
        self.effort = cfg.mesher_effort or "high"
        self.max_steps = int(cfg.mesher_max_steps or MAX_STEPS)
        self.max_seconds = float(cfg.mesher_max_seconds or MAX_SECONDS)
        self.log = CellLog()
        self.case_dir = ""

    # -- the run ---------------------------------------------------------------

    def run(self, request: str, case: str | None = None,
            geometry: str = "") -> CadResult:
        case_rel = _case_name(case)
        case_dir = f"{self.home}/{case_rel}"
        self.case_dir = case_dir
        self.log = CellLog()
        self._pending: Cell | None = None
        """A cell that outran its window and has not been judged yet."""
        started = time.monotonic()
        result = CadResult(case_rel=case_rel, case_dir=case_dir)
        try:
            self.backend.exec(f"mkdir -p {shlex.quote(case_dir)}", timeout_s=60)
        except Exception as exc:  # noqa: BLE001 - the workspace answered badly; say so
            result.error = f"could not make the case directory: {exc}"
            result.seconds = time.monotonic() - started
            return result
        try:
            self.backend.kernel_start(case_dir)
        except Exception as exc:  # noqa: BLE001 - no kernel is no desk
            result.error = f"no kernel on this workspace: {exc}"
            result.seconds = time.monotonic() - started
            return result

        system = self._system()
        messages: list[dict[str, Any]] = [
            {"role": "user",
             "content": [{"type": "text",
                          "text": task_message(request, case_dir, case_rel,
                                               self._said(), geometry)}]}
        ]
        last_text = ""
        turns = 0
        """Every lap counts against the budget, not only the ones that ran a cell: a
        desk that answers with prose, or says done to a check that refuses it, is
        spending the same minutes and would otherwise loop until the clock."""

        empty_turns = 0
        """Consecutive replies that were all reasoning and no words. The dropped turn
        cannot carry this, so `_ran_out_of_room` says it out loud instead."""

        while True:
            if turns >= self.max_steps:
                result.stopped = "steps"
                break
            if time.monotonic() - started >= self.max_seconds:
                result.stopped = "time"
                break
            try:
                turn = self._turn(system, messages)
            except ProviderError as exc:
                result.error = f"the model call failed: {exc}"
                result.stopped = "provider"
                break
            turns += 1
            _add(result.tokens, turn.tokens)
            said = _assistant(turn)
            if said is None:
                # A turn that is all reasoning and no words. Sending it back verbatim
                # is a 400 from the Messages API ("text content blocks must be
                # non-empty") which killed a whole run mid-mesh, so the empty turn is
                # dropped -- but dropping it alone is what made the failure permanent.
                # The thread the desk reads back has no record of the attempt, so it
                # starts over, reasons past the budget again, and says nothing again.
                # Measured: sixteen turns of sixteen, each re-deriving the same geometry
                # from scratch, three of them opening "I should first explore the
                # environment". The count is what the dropped turn cannot carry.
                empty_turns += 1
                self._beat(turns, result, turn, fenced=False)
                _observe(messages, _ran_out_of_room(empty_turns))
                continue
            empty_turns = 0
            messages.append(said)
            last_text = turn.text.strip() or last_text

            remark = self._remark(messages, result)

            source, complaint = parse_action(turn.text)
            self._beat(turns, result, turn, fenced=bool(source))
            if complaint:
                _observe(messages, complaint)
                continue

            if _is_finish(source) and remark:
                # Somebody spoke in the same breath as "done". Their words are the
                # newer instruction, so the run continues rather than closing on a
                # shape that was right one message ago.
                continue

            if _is_finish(source):
                result.summary = _summary(turn.text)
                check = self._verify(case_rel, request, self.log.script())
                result.check = check
                if check.ok:
                    result.ok = True
                    break
                _observe(messages, check.as_refusal())
                continue

            step = self._cell(source, _summary(turn.text), messages)
            result.steps.append(step)
            if len(result.steps) == NUDGE_AT_STEP and not self._has_mesh():
                _observe(messages, self._nudge())
            if self.on_step:
                try:
                    self.on_step(step)
                except Exception:  # noqa: BLE001 - a progress line may not end a run
                    pass
            _evict(messages)

        self._settle()
        result.seconds = time.monotonic() - started
        result.script = self.log.script()
        if not result.ok and result.check is None:
            # The run ended without saying done -- out of steps, out of time, or the
            # model call failed. What is on disk may still be a finished mesh, and a
            # mesh nobody looked at is exactly the failure this desk exists to end.
            # Measured, not assumed: a T10 run built 91,000 cells, hit a 400 on its
            # next model call, and was reported as "nothing was meshed".
            result.check = self._verify(case_rel, request, result.script)
            result.ok = result.check.ok
        if not result.summary:
            result.summary = _summary(last_text)
        result.png = self._render_bytes(result)
        return result

    # -- the three seams a differently-briefed desk replaces ---------------------
    #
    # The build-up phase of `docs/cad-build-up-handoff.md` runs this same loop with a
    # smaller brief, no toolbox and `checkMesh` as the whole of the finish check
    # (`buildup/core.py`). What differs between that desk and this one is only what it is
    # told, what it is nudged with, and what judges it -- so those three are methods, and
    # the loop below does not branch on which desk it is driving.

    def _system(self) -> str:
        return system_prompt(STEP_TIMEOUT_S, toolbox=self.toolbox)

    def _nudge(self) -> str:
        return NUDGE.format(toolbox=self.toolbox)

    def _verify(self, case_rel: str, request: str, script: str) -> Check:
        return verify(self.backend, self.case_dir, case_rel, request, script=script)

    def _mark(self, phase: str, expect_s: float, steps: int = -1) -> None:
        """Tell the watcher that a long, turn-free stretch is starting, and how long.

        The finish check runs `checkMesh` per region and the recovery replay re-runs the
        whole accepted log; neither is a model turn, and both can outlast a beat-age
        threshold that knows only about turns. So the loop declares its own silence rather
        than the supervisor guessing at a constant it cannot see. The turn index does not
        advance -- this is liveness, not progress."""
        if not self.on_turn:
            return
        try:
            self.on_turn(turn=getattr(self, "_turns", 0),
                         steps=steps if steps >= 0 else getattr(self, "_steps", 0),
                         phase=phase, expect_s=float(expect_s))
        except Exception:  # noqa: BLE001 - the watcher is not allowed to end the run
            pass

    def _beat(self, turns: int, result: CadResult, turn: Any, *, fenced: bool) -> None:
        """Report this turn to whatever is watching from outside, and carry on.

        Two call sites, which between them are every path a turn can take: the reply that
        was all reasoning and no words, and everything else. A turn that reported nothing
        would look to the observer exactly like a process that had stopped."""
        if not self.on_turn:
            return
        try:
            self._turns = turns
            self._steps = len(result.steps)
            self.on_turn(turn=turns, steps=len(result.steps),
                         stop_reason=getattr(turn, "stop_reason", ""),
                         output_tokens=int((getattr(turn, "tokens", None) or {}).get("output", 0)),
                         fenced=fenced, text_chars=len(turn.text or ""),
                         thinking_chars=_thinking_chars(turn),
                         text=turn.text or "", block_types=_block_types(turn))
        except Exception:  # noqa: BLE001 - the watcher is not allowed to end the run
            pass

    def _settle(self) -> None:
        """Stop a cell that is still going, before anything is measured or reported.

        The one place an interrupt is not the desk's own decision, and it is not a
        reflex either: the run is over, the finish check is about to read the case
        directory, and a cell still writing into it would have the check reading a
        half-written case and the caller told about it as if it were finished.
        """
        if not self._pending:
            return
        self._pending = None
        try:
            self.backend.kernel_interrupt()
        except Exception:  # noqa: BLE001 - a kernel that will not stop is not a verdict
            pass

    # -- the person ------------------------------------------------------------

    def _said(self) -> list[str]:
        """The last few things the person typed in this session, verbatim.

        Read off the transcript the session is already writing, so nothing new has to be
        plumbed through the calling agent, and what reaches the desk is the person's own
        words rather than a paraphrase of them.
        """
        try:
            rows = self.store.recent_messages(TRANSCRIPT_ROWS)
        except Exception:  # noqa: BLE001 - no transcript is not a reason not to build
            return []
        said = [" ".join(str(row.get("content") or "").split())
                for row in rows if row.get("role") == "user"]
        return [line[:600] for line in said if line][-SAID_LINES:]

    def _remark(self, messages: list[dict[str, Any]], result: CadResult) -> str:
        """Anything the person has typed since the last step, put into the thread."""
        if self.interject is None:
            return ""
        try:
            text = self.interject()
        except Exception:  # noqa: BLE001 - a remark that cannot be read may not end a run
            return ""
        if not (text or "").strip():
            return ""
        _observe(messages, remark_message(text))
        result.remarks.append(text.strip())
        return text.strip()

    def _turn(self, system: str, messages: list[dict[str, Any]]) -> Any:
        """One model call, retried once when the failure is one that passes.

        A desk five minutes into a build is expensive to lose to an overloaded endpoint,
        and the run cannot resume: the next call starts a clean thread. A 400 is not
        retried -- it is a fact about the request, and repeating it costs the same error
        twice."""
        for attempt in (1, 2):
            try:
                return self.provider.stream(
                    model=self.model, system=system, messages=messages, tools=[],
                    effort=self.effort, max_tokens=MAX_REPLY_TOKENS, listener=Listener(),
                )
            except ProviderError as exc:
                if attempt == 2 or exc.status_code not in RETRY_STATUSES:
                    raise
                time.sleep(RETRY_PAUSE_S)
        raise AssertionError("unreachable")

    # -- the kernel ------------------------------------------------------------

    def _cell(self, source: str, reasoning: str,
              messages: list[dict[str, Any]]) -> Step:
        self._catch_up(messages)
        cell = self.log.propose(source, reasoning)
        t0 = time.monotonic()
        try:
            outcome = self.backend.kernel_run(source, timeout_s=STEP_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 - the kernel, not the cell
            outcome = self._retry(source, exc, messages)
            if outcome is None:
                seconds = time.monotonic() - t0
                _observe(messages, f"the cell could not be run: {exc}")
                return Step(cmd=source, exit_code=-1, seconds=seconds, output=str(exc))
        seconds = time.monotonic() - t0
        return self._report(cell, outcome, seconds, messages)

    def _retry(self, source: str, exc: Exception,
               messages: list[dict[str, Any]]) -> Any:
        """The kernel died under a cell. Put one back, replay the log, run it again.

        This is the whole reason the log is the source of truth rather than the session:
        a kernel is a live process and live processes do not survive being preempted,
        recycled or restarted, whereas the accepted cells do, because they are a file.
        """
        if not self._recover(messages, str(exc)):
            return None
        try:
            return self.backend.kernel_run(source, timeout_s=STEP_TIMEOUT_S)
        except Exception:  # noqa: BLE001 - twice is the workspace, not the cell
            return None

    def _recover(self, messages: list[dict[str, Any]], why: str) -> bool:
        try:
            self.backend.kernel_restart()
        except Exception:  # noqa: BLE001 - a kernel too gone to restart gets a new one
            try:
                self.backend.kernel_start(self.case_dir)
            except Exception:  # noqa: BLE001
                return False
        script = self.log.script()
        if script:
            self._mark("recover", RECOVERY_TIMEOUT_S)
            try:
                replay = self.backend.kernel_run(script, timeout_s=RECOVERY_TIMEOUT_S)
            except Exception:  # noqa: BLE001
                return False
            if not replay.ok:
                _observe(messages, RECOVERY_FAILED.format(why=why, error=replay.error))
                return False
        _observe(messages, RECOVERED.format(why=why, cells=len(self.log.cells())))
        return True

    def _catch_up(self, messages: list[dict[str, Any]]) -> None:
        """Where the cell that outran its window got to, asked before sending another.

        A poll reads; it does not execute, so it does not touch the session the desk is
        reasoning about and it works while the kernel is busy. And a slow cell that
        finished correctly belongs in the script, so this is also where it is judged --
        otherwise the one cell that took four minutes is the one missing from the file.
        """
        pending, self._pending = self._pending, None
        if pending is None:
            return
        try:
            outcome = self.backend.kernel_poll()
        except Exception as exc:  # noqa: BLE001 - not knowing is not a verdict
            _observe(messages, f"the earlier cell could not be polled: {exc}")
            return
        if outcome.still_running:
            self._pending = pending
            _observe(messages, STILL_RUNNING.format(seconds=outcome.seconds,
                                                    body=_clip(_streams(outcome), 1000)))
            return
        head = (f"the cell that outran its window has finished "
                f"({outcome.seconds:.0f} s, {'ok' if outcome.ok else 'failed'})")
        _observe(messages, f"{head}\n{_clip(_body(outcome), OUTPUT_CHARS)}".rstrip())
        self._judge(pending, outcome, messages)

    def _report(self, cell: Cell, outcome: Any, seconds: float,
                messages: list[dict[str, Any]]) -> Step:
        """The cell's result into the thread, and the cell into the log or not."""
        body = _clip(_body(outcome), OUTPUT_CHARS)
        head = f"exit {0 if outcome.ok else 1} ({seconds:.0f} s)"
        if outcome.still_running:
            head = (f"still running after {STEP_TIMEOUT_S} s -- it was not killed, and "
                    "the next thing you send is polled against it first")
        if outcome.truncated and outcome.log_path:
            head += f"; output cut, all of it is at {outcome.log_path}"
        note = _no_picture_note(cell.source, outcome)
        if note:
            body = f"{body}\n{note}".strip()
        shown = list(outcome.images or [])[:IMAGES_PER_CELL]
        if len(outcome.images or []) > IMAGES_PER_CELL:
            body = (f"{body}\n[{len(outcome.images)} pictures came back from this cell "
                    f"and the first {IMAGES_PER_CELL} are attached. A cell that draws in "
                    "a loop is worth knowing about, not worth sending: draw the one "
                    "view that answers the question.]").strip()
        blocks: list[dict[str, Any]] = [{"type": "text", "text": f"{head}\n{body}".rstrip()}]
        for data in shown:
            blocks.append(images.attachment(data, "image/png"))
        messages.append({"role": "user", "content": blocks})

        if outcome.still_running:
            self._pending = cell
        else:
            self._judge(cell, outcome, messages)
        return Step(cmd=cell.source, exit_code=0 if outcome.ok else 1, seconds=seconds,
                    output=body, image=_picture_words(outcome))

    def _judge(self, cell: Cell, outcome: Any,
               messages: list[dict[str, Any]]) -> None:
        """Whether this cell goes in the script, and what the desk is told either way.

        A cell that raised is not in the build: the traceback is already in front of the
        desk and there is nothing to add. A cell that ran clean still has to pass the
        static check, and when it does not the refusal says that it ran -- the kernel
        now holds a binding the script will not have, and a desk that was not told would
        keep building on it.
        """
        if not outcome.ok:
            return
        refusal = self.log.accept(cell)
        if refusal:
            _observe(messages, refusal)

    def _has_mesh(self) -> bool:
        """Whether anything in the case directory has actually been meshed.

        Asked of the filesystem, through `exec`, and through the same
        `check.mesh_regions()` the finish check uses. Reading it off the transcript was
        right when every action was a shell command and is not right now: in a kernel,
        prep and meshing both look like Python, and `snappyHexMesh` may be inside a
        `Popen` string or not appear at all. One answer to "is there a mesh here" is
        also what stops the nudge and the check disagreeing about a conjugate case that
        has meshed two regions and no singular mesh.

        A workspace that will not answer is not a desk that has not meshed, so silence
        counts as a mesh: a nudge sent on no evidence is noise in the one place the
        thread can least afford it.
        """
        try:
            return bool(mesh_regions(self.backend, self.case_dir))
        except Exception:  # noqa: BLE001 - not knowing is not a verdict
            return True

    def _render_bytes(self, result: CadResult) -> bytes | None:
        """The picture the tool result carries back: the one the check drew."""
        path = (result.check.render_abs or result.check.render) if result.check else ""
        if not path:
            return None
        try:
            info = self.backend.stat(path)
            if info.size and info.size <= images.MAX_ATTACH_BYTES:
                data = self.backend.get_file(path, limit=info.size)
                return data if len(data) == info.size else None
        except Exception:  # noqa: BLE001 - a missing picture is not a failed mesh
            return None
        return None


# -- reading what the model sent ----------------------------------------------


def parse_action(text: str) -> tuple[str, str]:
    """The one cell in a message, or what to say back about it.

    Returns `(source, "")` or `("", complaint)`. Both halves matter: a model that wrote
    two blocks has to be told which one would have run, and a model that wrote none is
    usually explaining itself at length instead of acting. Neither complaint changed
    when the channel did, because both were about the model's discipline rather than
    about the language in the fence.
    """
    blocks = [b.strip() for b in _FENCE.findall(text or "") if b.strip()]
    if not blocks:
        return "", ("There was no python block in that message, so nothing ran. Send one "
                    "fenced ```python block containing the cell you want run, and "
                    "nothing else that needs running.")
    if len(blocks) > 1:
        return "", (f"That message had {len(blocks)} python blocks and none of them ran. "
                    "One block per message: put the whole cell -- the imports, the "
                    "constants, the measurement -- in a single block.")
    return blocks[0], ""


_FINISH = re.compile(rf"^print\(\s*[\"']{CAD_DONE}[\"']\s*\)$")


def _block_types(turn: Any) -> list[str]:
    """What the reply was made of. `['thinking']` with no text is the starved failure."""
    return [str(getattr(block, "type", "") or (block.get("type", "")
            if isinstance(block, dict) else ""))
            for block in (getattr(turn, "content", None) or [])]


def _thinking_chars(turn: Any) -> int:
    """How much of the reply went to reasoning -- the other half of the budget.

    `starved` is the pair of this and an empty text block: thinking and words are spent
    from one allowance, so a hard prompt can reason past the ceiling and never open the
    sentence that carries the work."""
    total = 0
    for block in getattr(turn, "content", None) or []:
        text = getattr(block, "thinking", None)
        if text is None and isinstance(block, dict):
            text = block.get("thinking")
        total += len(text or "")
    return total


def _is_finish(source: str) -> bool:
    """Whether this block *is* the finish cell, not whether it mentions it.

    It used to be a search for the word anywhere in the block, and a model that writes
    `# print("CAD_DONE") once checkMesh passes` above a perfectly ordinary cell -- which
    is a very normal thing to write, since the brief hands it the token -- had that cell
    silently never run, and got the finish check's refusal as the answer to something it
    had not asked. The token is the whole cell or it is a word in a comment.
    """
    return bool(_FINISH.match(source.strip()))


def _summary(text: str) -> str:
    """The prose around the block: what the desk says it built, and why."""
    without = _FENCE.sub("", text or "").strip()
    return "\n".join(line for line in without.splitlines() if line.strip()).strip()


# -- what came back -----------------------------------------------------------


def _body(outcome: Any) -> str:
    streams = _streams(outcome)
    if outcome.error and outcome.error not in streams:
        streams = f"{streams}\n{outcome.traceback or outcome.error}".strip()
    return streams


def _streams(outcome: Any) -> str:
    return "\n".join(part for part in (outcome.stdout or "", outcome.stderr or "")
                     if part.strip()).strip()


def _picture_words(outcome: Any) -> str:
    count = len(outcome.images or [])
    if not count:
        return ""
    return "1 picture" if count == 1 else f"{count} pictures"


_PLOTTING = re.compile(r"\bplt\.|\bpyplot\b|\bsavefig\b|\bmatplotlib\b|\bshow\(\)")
_NON_INLINE = re.compile(r"matplotlib\.use\(\s*[\"']([A-Za-z_]+)[\"']|"
                         r"\bmpl\.use\(\s*[\"']([A-Za-z_]+)[\"']")


def _no_picture_note(source: str, outcome: Any) -> str:
    """A cell that plotted and sent back no picture, said out loud.

    `matplotlib.use("Agg")` is the habit a model carries over from writing headless CFD
    scripts for bash, where it is mandatory and where the figure is only ever going to
    reach anybody as a file. In a kernel it is the one line that silently undoes this
    channel: the cell succeeds, prints nothing unusual, writes its PNG, and the desk is
    told nothing at all -- so it believes it has looked at the shape when it has not.
    That is precisely the failure the picture is here to prevent, arriving dressed as a
    success, so the harness says what happened rather than leaving it to be noticed.
    """
    if outcome.images or not outcome.ok or outcome.still_running:
        return ""
    if not _PLOTTING.search(source or ""):
        return ""
    backend = next((name for match in _NON_INLINE.findall(source or "")
                    for name in match if name), "")
    if backend and backend.lower() == "module://matplotlib_inline.backend_inline":
        return ""
    blamed = (f"This cell called `matplotlib.use(\"{backend}\")`, which switches the "
              f"figure away from the inline backend that sends it here. " if backend
              else "")
    return (f"[no picture came back from this cell, and it looks like it drew one. "
            f"{blamed}A figure reaches you by being displayed, not by being saved: end "
            "the cell with the figure object, or call `display(fig)`, and do not set a "
            "matplotlib backend -- the kernel is already on the inline one. A `.png` "
            "written to disk is not sent; read it back with "
            "`IPython.display.Image(path)` if you want to see it.]")


# -- the thread ---------------------------------------------------------------


def _assistant(turn: Any) -> dict[str, Any] | None:
    """The turn as a thread entry, with empty blocks left out, or None if nothing is left.

    An empty text block is not a harmless nothing: the Messages API refuses the whole
    request that carries one, so a single empty turn ends the run several steps into a
    build that was going fine.
    """
    message = turn.as_message()
    content = [b for b in (message.get("content") or []) if not _is_empty(b)]
    if not content:
        return None
    message["content"] = content
    return message


def _is_empty(block: Any) -> bool:
    kind = block.get("type") if isinstance(block, dict) else getattr(block, "type", "")
    if kind != "text":
        return False
    text = block.get("text") if isinstance(block, dict) else getattr(block, "text", "")
    return not (text or "").strip()


NUDGE = (
    f"That is {NUDGE_AT_STEP} cells and nothing in the case directory has been meshed "
    "yet. Whatever is left to work out about the shape, work it out in the mesh rather "
    "than before it: build the coarsest version that exists at all, run the mesher, and "
    "look at it. `{toolbox}/templates/` has a prep recipe and a working snappy "
    "dictionary set that go from an outline to a checked polyMesh, and a shape that is "
    "80% right and on disk is worth more than one that is exact and is not.")

STILL_RUNNING = (
    "Your earlier cell is still running ({seconds:.0f} s so far). It was not killed and "
    "nothing here will kill it. Anything you send now queues behind it, so either wait "
    "-- send a cell that only prints, to look again -- or plan for it to run when that "
    "one finishes.\n\nWhat it has printed so far:\n{body}")

RECOVERED = (
    "The kernel died and has been replaced ({why}). Every binding it held is gone, so "
    "the {cells} accepted cells were re-run in the new one and the session is back to "
    "what the script says it is. Anything you bound in a cell that was never accepted "
    "is not there. This is what the script being the artifact buys: a live process does "
    "not survive being recycled, and a file does.")

RECOVERY_FAILED = (
    "The kernel died ({why}) and replaying the accepted cells into its replacement "
    "failed: {error}. The session is not what the script says it is. Re-establish what "
    "you need explicitly before you build on it.")


def _ran_out_of_room(count: int) -> str:
    """What to say to a desk whose reply was all reasoning and no words.

    It cannot see its own dropped turn, so the count goes here or nowhere. Saying *why*
    the message was empty matters as much as saying that it was: a desk told only "that
    arrived empty" reads it as a transport hiccup and sends the same enormous reply
    again, where one told it ran out of room while thinking has something to act on.
    """
    if count == 1:
        return ("That message arrived empty. Send one fenced ```python block with the "
                "cell you want run.")
    return (f"That message arrived empty again -- {count} times now. The whole reply "
            "was spent reasoning and it was cut off before any words were written, so "
            "nothing ran and nothing of it reached this thread. Do not solve the rest "
            "of the problem before answering: send one short cell that makes a little "
            "progress, look at what it prints, and build on it next turn.")


def _observe(messages: list[dict[str, Any]], text: str) -> None:
    messages.append({"role": "user", "content": [{"type": "text", "text": text}]})


def _evict(messages: list[dict[str, Any]]) -> None:
    """Keep the last `KEEP_IMAGES` pictures; older ones become a line of text.

    The words stay, because the words are what the later steps reason from -- the
    measurements, the patch table, the checkMesh line. Only the picture goes. It was
    never about where the pictures came from, so it survived the change of channel
    unaltered.
    """
    seen = 0
    for message in reversed(messages):
        if message.get("role") != "user" or not isinstance(message.get("content"), list):
            continue
        blocks = message["content"]
        if not any(isinstance(b, dict) and b.get("type") == "image" for b in blocks):
            continue
        seen += 1
        if seen <= KEEP_IMAGES:
            continue
        kept = [b for b in blocks if not (isinstance(b, dict) and b.get("type") == "image")]
        kept.append({"type": "text", "text": "(the picture from this step is no longer shown)"})
        message["content"] = kept


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head
    cut = len(text) - limit
    return f"{text[:head]}\n... {cut} characters cut ...\n{text[-tail:]}"


def _add(total: dict[str, int], more: dict[str, int]) -> None:
    for key, value in (more or {}).items():
        total[key] = total.get(key, 0) + int(value or 0)


def _case_name(case: str | None) -> str:
    """A directory name under the study, never a path out of it."""
    name = (case or "").strip().strip("/")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return name or "cad"
