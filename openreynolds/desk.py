"""The desk: the two things said to the person around the main conversation, never in it.

It does two jobs, and both exist because of one rule: **a session's system prompt and
tool list do not change mid-conversation.** A thinking block's signature is bound to the
system prompt and the tools it was made under. A study whose solver was left open used to
start the main agent on a solver-choosing prompt with one tool, `choose_solver`, and swap
both for the chosen solver's the moment it was called; on production, where Claude is
reached through the service's proxy to Bedrock, the request after the swap came back
`400 messages.1.content.0: Invalid signature in thinking block. The block is bound to a
different conversation ... The system prompt and the tools list both differ from when
this block was created`, and the turn ended with nothing said. (Anthropic's own API lets
the same request through when `thinking.block_binding.prefix_mismatch_behavior` is
unset, and refuses it identically with `"error"`; nothing here should depend on which.)

**Routing** (`route`). A study whose solver is `auto` has it settled before the main
agent's first request, by one request of its own: the person's first message, a short
prompt describing the two solvers, and one tool, `choose_solver`. It has no main
conversation to share a cache with, so its own prompt and tools cost nothing the main
thread pays for. The answer -- `openfoam` or `felix` and a sentence why -- fixes the
solver for the study's life, the workspace of that kind is reserved and started, and the
main agent begins on that solver's prompt and tools at turn one, told the choice and
the reason in its briefing. It runs on the study's own model. When it fails, the study is
OpenFOAM's, and that is said to the person and to the agent, not assumed silently.

**The side chat** (`SideChat`). `/btw <question>` asks something without asking the agent
to stop: the question is answered apart from the conversation, and nothing about it is
added to the main thread. It is answered by the main agent's own last request, sent once
more exactly as it was -- the same model, system prompt, tool list, thinking, effort and
output settings, the same messages, thinking blocks included -- with one user message
appended: a harness note saying what this is and what it may do, then the person's words.
That shape is a requirement, not a convenience: it is the only one that reads the whole
conversation from the prompt cache, and the only one whose thinking blocks are still
valid, because the system prompt and the tools they were made under are the ones sent.

The side chat may look but not touch. Of the main agent's tools it runs `read_file` and
`job_check` -- the second with no wait and without moving the job's log offset, which is
the main agent's bookmark -- and `send_to_agent`, which passes the person's own words to
the main agent at its next step; any other call is answered as not run. The tool list it
sends is still the main one, all of it: a different list would miss the cache and
invalidate every thinking block, which is why `send_to_agent` is in the main list from
the first request of every session (`tools.py`).

WHAT THIS REPLACED. A front desk: a second, cheaper model that answered every message
typed while the main agent was busy, from a slice of the transcript. It was built for a
real complaint -- "it still doesn't stop and respond" -- and it did make the screen less
silent. But it answered messages that were meant for the agent, in the agent's place,
from a summary that was not the agent's context, and the person could not tell which of
its answers were knowledge and which were a cheap model's guess. Now a message is the
agent's and reaches it at its next step, with a receipt on screen at once and held waits
ending within a second for it (`tools._hold`); a question for an answer now is a side
question, asked on purpose, answered from the agent's own context.

Why none of this breaks the free-will contract: the contract governs what may influence
the main model's decisions. Routing happens before there is a main model to influence,
and says what it chose and why. The side chat writes nothing into the main thread except
what the person asked it to pass on, and that goes as the person's words, unaltered, with
any note of its own labelled as its own (`loop.forwarded_note`). It cannot stop, approve
or change anything; it says so and points at the controls that can.
"""

from __future__ import annotations

import dataclasses
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from . import solvers
from .llm import Listener, ProviderError
from .llm import base as llm_base
from .loop import MAX_TOKENS, _as_operator_text
from .prompt import system_prompt
from .tools import _clip, describe_job, dispatch, tools_for

# -- routing -------------------------------------------------------------------

ROUTING_SYSTEM = """\
You choose the solver for a CFD study before the engineer who runs it starts. You do \
not run the study, and nothing you say reaches the person except your reason, which is \
shown to them. Read their request and call `choose_solver` once, with the solver and \
one sentence why; write nothing else. The choice holds for the whole study and decides \
which kind of workspace it gets. A request that names a solver settles it. One that \
leaves it open is a question about the physics: choose the solver that can do all of \
what is asked -- OpenFOAM when only it can, or when the request is not yet clear enough \
to tell.

# OpenFOAM

A Linux workspace with OpenFOAM ESI v2512 on CPU cores, billed while it runs. The \
whole library: incompressible and compressible flow, steady and transient, \
multiphase and free-surface flow, heat transfer including conjugate, combustion and \
reacting flow, RANS and LES turbulence models, moving and dynamic meshes. Meshes \
come from blockMesh, snappyHexMesh, cfMesh or gmsh, and a CAD agent turns a shape \
in words or a STEP file into a mesh. Run time grows with the cell count over the \
cores of one machine.

# Felix

A GPU solver for the **incompressible** Navier-Stokes equations on unstructured 3-D \
meshes (tetrahedra, hexahedra, wedges, pyramids), stabilized finite elements with \
implicit time stepping, element order 1 to 5. Optional: a temperature field with \
Boussinesq buoyancy, Spalart-Allmaras turbulence (SA or SA-DDES, wall functions \
available), porous zones, fan zones, volumetric heat sources, and pseudo-transient \
continuation to a steady state. Nothing compressible, no multiphase or free surface, \
no combustion, no moving meshes and no turbulence model other than Spalart-Allmaras. \
A 2-D problem is a one-cell 3-D slab in either solver. Each solve runs on one cloud \
GPU (A100, H100, H200 or B200) billed per second while it runs; on a large mesh one \
GPU is many times faster than the CPU workspace. There is no CAD agent: the mesh is a \
`.vtu` built with gmsh or Python.
"""
"""The routing request's prompt. Most of it is the description the solver-choosing
prompt carried (`SOLVER_CHOICE_PROMPT`, removed): the facts about the two solvers did
not change, only who reads them and when."""

ROUTE_TOOL: dict[str, Any] = {
    "name": "choose_solver",
    "description": (
        "Fix this study's solver: `openfoam` (the OpenFOAM library on the workspace's "
        "CPU cores) or `felix` (an incompressible GPU solver, each solve on a cloud "
        "GPU). Holds for the whole study, resumes included."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "solver": {"type": "string", "enum": list(solvers.SOLVERS)},
            "reason": {
                "type": "string",
                "description": "Why this solver, in one sentence the person can read.",
            },
        },
        "required": ["solver", "reason"],
    },
}
"""The routing request's one tool. Never in the main agent's list."""

ROUTE_MAX_TOKENS = 8_000
"""Room for a short decision and the thinking a model that always thinks does before
it; the answer itself is a sentence and a tool call."""

ROUTE_EFFORT = "low"
"""A two-way choice from a paragraph of text. Its own request, so its effort is its own
and changes nothing the main thread sends."""


@dataclass
class Route:
    """What the routing request decided, or why it could not."""

    solver: str
    reason: str = ""
    failed: str = ""
    """Why no choice came back -- the request failed, or the answer named no solver.
    The solver is then OpenFOAM, and this is what the person and the agent are told."""
    tokens: dict[str, int] = field(default_factory=dict)


ROUTE_WAITS = (5.0, 15.0, 30.0)
"""Seconds before each retry of a routing request that failed transiently
(`llm_base.retry_delay`): under a minute, then OpenFOAM by default, said so."""


def route(provider: Any, model: str, said: str) -> Route:
    """Settle an open solver from what the person asked for. Never raises.

    `said` is their first message. A hosted study can be opened with none -- the page
    starts the session and the person types afterwards -- and the session then waits at
    its prompt, as any interactive one does, so this runs on the first message typed:
    the first request needs a user message anyway, and the solver is the one thing that
    must be known before it. With nothing at all to read (no current path gets here
    without a message), it does not guess: OpenFOAM, said so."""
    if not (said or "").strip():
        return Route(solvers.OPENFOAM, failed="there was no request to choose from")
    attempt = 0
    while True:
        try:
            turn = provider.stream(
                model=model,
                system=ROUTING_SYSTEM,
                messages=[{"role": "user", "content": said}],
                tools=[ROUTE_TOOL],
                effort=ROUTE_EFFORT,
                max_tokens=ROUTE_MAX_TOKENS,
                listener=Listener(),
            )
            break
        except ProviderError as exc:
            # A rate limit here used to settle the solver: OpenFOAM, for the whole
            # study, because the limiter was busy for a second. Asked again first, on
            # a shorter schedule than a turn's -- nothing has started yet, and the
            # person is watching "choosing the solver".
            delay = llm_base.retry_delay(exc, attempt, ROUTE_WAITS)
            if delay is None:
                return Route(solvers.OPENFOAM,
                             failed=f"the routing request failed: {exc.message or exc}")
            attempt += 1
            llm_base.nap(delay)
        except Exception as exc:  # noqa: BLE001 - a study must start whatever this did
            return Route(solvers.OPENFOAM,
                         failed=f"the routing request failed: {type(exc).__name__}: {exc}")
    call = next((c for c in turn.tool_calls if c.name == ROUTE_TOOL["name"]), None)
    chosen = solvers.normalize(str((call.input if call else {}).get("solver") or ""))
    if chosen not in solvers.SOLVERS:
        return Route(solvers.OPENFOAM, failed="the routing request named no solver",
                     tokens=dict(turn.tokens or {}))
    reason = " ".join(str(call.input.get("reason") or "").split())
    return Route(chosen, reason=reason, tokens=dict(turn.tokens or {}))


def solver_name(solver: str) -> str:
    return "Felix" if solver == solvers.FELIX else "OpenFOAM"


def routed_line(chosen: Route) -> str:
    """The briefing's sentence about the choice, for the main agent."""
    name = solver_name(chosen.solver)
    if chosen.failed:
        return (f"This study's solver is {name}, for the whole study. It was left open, "
                f"and choosing it from the person's first message did not work "
                f"({chosen.failed}), so it is OpenFOAM by default; the person has been "
                "told.")
    why = f" The reason given to the person: {chosen.reason}" if chosen.reason else ""
    return (f"This study's solver is {name}, for the whole study. It was left open, and "
            f"was chosen from the person's first message before your first turn.{why}")


def routed_notice(chosen: Route) -> str:
    """What the person is told about the choice."""
    name = solver_name(chosen.solver)
    if chosen.failed:
        return (f"solver: {name}. The solver could not be chosen from your message "
                f"({chosen.failed}), so this study runs on OpenFOAM.")
    return f"solver: {name}, chosen from your first message. {chosen.reason}".strip()


# -- the side chat -------------------------------------------------------------

SIDE_STEPS = 5
"""Requests one side question may make: an answer, or a few looks at a file or a job
and then an answer. A small loop of its own, bounded so a question cannot become a turn."""

SIDE_TOOLS = ("read_file", "job_check", "send_to_agent")
"""What a side question runs. `read_file` reads; `job_check` is answered here without a
wait and without the log offset it would otherwise move (`_side_job_check`). Both are
safe beside a main turn on the same workspace: neither writes a file, starts, stops or
waits on anything, or touches what the main agent keeps (`ctx.echoes`, a job's offset).

Left out, and answered as not run: `bash` (an arbitrary command, which can do anything),
`write_file`, `job_start`, `job_kill`, `fetch` (it writes to the person's machine and to
the capture), `cad` and `mesh_review` (each a model run of its own, minutes long) and
`checkpoint` (a question to the person, which only the main turn may put)."""

SIDE_THREAD_KEEP = 3
"""Earlier side questions carried to a later one once the fork point has moved."""

SIDE_NOTE = (
    "This is a side question from the person, asked with /btw while the work goes on. "
    "It is answered apart from this conversation: nothing in this exchange is added to "
    "it, the work in progress does not see it, and it carries on where it was once you "
    "have answered. Answer in text, briefly, from what this conversation already shows. "
    "Enabled for side questions: read_file, job_check (without wait_s; it reads a job's "
    "status and log and moves nothing) and send_to_agent. Any other tool call is not "
    "run. You are answering beside the work, not as it: say what the conversation "
    "shows, and do not promise what the work will do next. Nothing can be stopped, "
    "approved or changed from here -- the session's Stop and End controls do that (/exit "
    "in a terminal), and a question the agent is waiting on is answered by replying to "
    "it. When the person asks for something to be done, changed or considered, call "
    "send_to_agent: it passes their words, exactly as they wrote them, to the work at "
    "its next step, with a short note of yours if one helps. If you do not know, say so "
    "rather than guess. The person's question follows."
)
"""The harness note in front of a side question, in the form every harness note takes
in a user turn (`loop._as_operator_text`). The system prompt is not touched: it is part
of the cached prefix, and of what the thinking blocks are bound to."""

NOT_ENABLED = (
    "{name} is not enabled in a side question, so it did not run. Enabled here: "
    "read_file, job_check (without waiting) and send_to_agent."
)

NO_THREAD = (
    "There is no conversation to answer a side question from yet: the agent has not "
    "started. A plain message reaches it."
)

OFF = (
    "side questions are off in this session (OPENREYNOLDS_DESK=0); a plain message "
    "reaches the agent at its next step"
)


class SideChat:
    """`/btw <question>`, answered by a fork of the main agent's last request.

    One question at a time, on a thread of its own, so asking never holds the main turn
    or the prompt. The fork point is `Loop.last_request`: what the main thread last
    sent, kept with a messages list of its own. A question is that request, then the
    side thread -- this side chat's own questions and answers since the fork point, so a
    second question builds on the first and still reads everything before it from the
    cache -- then the new question.

    When the main thread has sent another request since, the fork point has moved and
    the earlier side exchange cannot follow as it was: its thinking blocks were made on
    the old prefix and are bound to it. The last few questions and answers ride instead,
    as text, in the new note.

    `deliver(words, note)` is how `send_to_agent` reaches the main agent; the session
    wires it (`cli._deliver_forward`) to `Loop.forward`, or to the prompt when no turn
    is running."""

    def __init__(self, loop: Any, view: Any, deliver: Callable[[str, str], None]):
        self.loop = loop
        self.view = view
        self.deliver = deliver
        self._fork: dict[str, Any] | None = None
        self._side: list[dict[str, Any]] = []
        """The side thread after `_fork`: questions, answers and tool rounds, in order."""
        self._history: list[tuple[str, str]] = []
        self._q: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------------

    def ask(self, question: str, wait: bool = False) -> None:
        """Answer `question` on the side chat's thread -- or here and now, with `wait`."""
        question = (question or "").strip()
        if not question:
            return
        try:
            self.view.side_question(question)
        except Exception:  # noqa: BLE001 - presentation may not end a session
            pass
        if wait:
            self._answer(question)
            return
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="side-chat", daemon=True)
            self._thread.start()
        self._q.put(question)

    def drain(self, timeout: float = 30.0) -> None:
        """Wait for every question asked so far to be answered (tests, close-down)."""
        if self._thread is None:
            return
        done = threading.Event()
        self._q.put(done)  # type: ignore[arg-type]
        done.wait(timeout)

    def stop(self, timeout: float = 3.0) -> None:
        if self._thread is not None:
            self._q.put(None)
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            if isinstance(item, threading.Event):
                item.set()
                continue
            try:
                self._answer(item)
            except Exception as exc:  # noqa: BLE001 - a side answer may never end a session
                _answer_view(self.view, "The side question could not be answered: "
                                        f"{type(exc).__name__}: {exc}", False)

    # -- one question --------------------------------------------------------------

    def _answer(self, question: str) -> None:
        with self._lock:
            base = self._base()
            if base is None:
                _answer_view(self.view, NO_THREAD, False)
                return
            if base is not self._fork:
                # The main thread has moved on since the side thread began: begin it
                # again after the new fork point, carrying what was asked as text.
                self._fork = base
                self._side = []
                carried = self._history[-SIDE_THREAD_KEEP:]
            else:
                carried = []
            prefix = list(base["messages"]) + self._settled_reply(base)
            asked = {"role": "user", "content": side_message(question, carried)}
            messages = prefix + self._side + [asked]
            forwarded = False
            texts: list[str] = []
            failed = ""
            for _ in range(SIDE_STEPS):
                try:
                    turn = self.loop.provider.stream(
                        model=base["model"],
                        system=base["system"],
                        messages=messages,
                        tools=base["tools"],
                        effort=base["effort"],
                        max_tokens=base["max_tokens"],
                        listener=Listener(),
                    )
                except ProviderError as exc:
                    failed = exc.message or str(exc)
                    break
                add = getattr(self.loop, "add_tokens", None)
                if add is not None:
                    add(turn.tokens or {})
                if turn.text.strip():
                    texts.append(turn.text.strip())
                messages.append(turn.as_message())
                calls = turn.tool_calls
                if not calls:
                    break
                results = []
                for call in calls:
                    content, sent = self._run_tool(call, question, forwarded)
                    forwarded = forwarded or sent
                    results.append({"type": "tool_result", "tool_use_id": call.id,
                                    "content": content or "(no output)"})
                messages.append({"role": "user", "content": results})
            if failed:
                # The question stays out of the side thread: it was never answered.
                _answer_view(self.view, "The side question could not be answered: the "
                                        f"model request failed ({failed}).", forwarded)
                return
            answer = "\n\n".join(texts) or "(the side chat gave no answer in words)"
            self._side = messages[len(prefix):]
            self._history.append((question, answer))
            _answer_view(self.view, answer, forwarded)

    def _base(self) -> dict[str, Any] | None:
        """The fork point: the main thread's last request, or -- before it has sent one
        -- the request it would send, which is the same bytes its first one will be."""
        last = getattr(self.loop, "last_request", None)
        if last is not None:
            return last
        ctx = self.loop.ctx
        if getattr(ctx, "solver", None) == solvers.AUTO or not self.loop.messages:
            return None
        if self._fork is not None and self._fork.get("_unsent"):
            return self._fork
        return {
            "model": self.loop.cfg.model,
            "system": system_prompt(ctx.solver),
            "messages": list(self.loop.messages),
            "tools": tools_for(ctx),
            "effort": self.loop.cfg.effort,
            "max_tokens": MAX_TOKENS,
            "_unsent": True,
        }

    def _settled_reply(self, base: dict[str, Any]) -> list[dict[str, Any]]:
        """The main agent's answer to `base`, when its turn ended on it.

        Mid-turn, the reply to the last request asked for tools whose results are not
        in yet, and it cannot be sent; the fork is the request alone, byte for byte.
        Once the turn has ended on words -- "the drag coefficient is 0.31" -- those
        words are what a side question is most often about, so the reply rides after
        the request: the same prefix, its thinking bound to the same prompt and tools,
        and only its own tokens new to the cache."""
        messages = getattr(self.loop, "messages", [])
        sent = base["messages"]
        if len(messages) != len(sent) + 1:
            return []
        if any(a is not b for a, b in zip(messages, sent)):
            return []
        reply = messages[-1]
        if reply.get("role") != "assistant":
            return []
        if any(_kind(b) == "tool_use" for b in reply.get("content") or []):
            return []
        return [reply]

    def _run_tool(self, call: Any, question: str, forwarded: bool) -> tuple[Any, bool]:
        """One tool call from the side chat: (its result, whether it passed the words on)."""
        name = call.name
        if name == "send_to_agent":
            if forwarded:
                return "Already passed on for this question; it was not sent twice.", False
            note = " ".join(str((call.input or {}).get("note") or "").split())
            self.deliver(question, note)
            return ("Passed on: the person's words, as they wrote them, reach the agent at "
                    "its next step" + (", with your note labelled as yours." if note else "."),
                    True)
        if name not in SIDE_TOOLS:
            return NOT_ENABLED.format(name=name), False
        ctx = _side_ctx(self.loop.ctx)
        if name == "job_check":
            try:
                return _side_job_check(ctx, dict(call.input or {})), False
            except Exception as exc:  # noqa: BLE001 - a fact for the side answer
                return f"{type(exc).__name__}: {exc}", False
        content, _ = dispatch(ctx, name, dict(call.input or {}), call_id=call.id)
        return content, False


def side_message(question: str, carried: list[tuple[str, str]] = ()) -> list[dict[str, Any]]:
    """The one message a side question appends: the harness note, then the person's
    words exactly as they typed them."""
    note = SIDE_NOTE
    if carried:
        earlier = "\n".join(f"Q: {q}\nA: {a}" for q, a in carried)
        note += ("\n\nEarlier side questions, asked since the side thread began, with "
                 f"the answers given:\n{earlier}")
    return [
        {"type": "text", "text": _as_operator_text(note)},
        {"type": "text", "text": question},
    ]


def _side_ctx(ctx: Any) -> Any:
    """The main tool context, for a side call: the same workspace and store, none of
    the main thread's own bookkeeping. `echoes` is the main thread's record of what it
    has been shown; a side read entered there would make the main agent's next read of
    the same bytes answer with a pointer to a call it never saw. Nor does a side call
    put a study's spending ceiling to the person (`ceiling.resolve`): that question
    belongs to the conversation, and a side read that meets the pause just says so."""
    if not dataclasses.is_dataclass(ctx):
        return ctx
    return dataclasses.replace(
        ctx, echoes={}, calls=0, view=None, on_render=None, on_fetch=None,
        on_wait_input=None, on_leaving=None, on_tokens=None, on_mode=None,
        approver=None, ceiling=None,
    )


SIDE_LOG_BYTES = 4_000
"""How much of a job's log a side `job_check` shows when no offset is asked for: the
tail, which is what "how is it going" is about."""


def _side_job_check(ctx: Any, args: dict[str, Any]) -> str:
    """`job_check` for a side question: status and log, read and nothing written.

    The main one records where it stopped reading (`log_offset`), so the agent's next
    check continues from there, and it announces the jobs to the view. Neither may
    happen from here -- the agent would miss the lines a side question read -- and a
    wait is the one thing a side question must never do, so `wait_s` is not held."""
    job_id = str(args.get("job_id") or "")
    status = ctx.backend.job_status(job_id)
    size = int(getattr(status, "log_size", 0) or 0)
    offset = args.get("log_offset")
    offset = int(offset) if offset is not None else max(0, size - SIDE_LOG_BYTES)
    data, next_offset, eof = ctx.backend.job_tail(job_id, offset=offset)
    body, _ = _clip(data, ctx.max_output)
    header = describe_job(status) + f"\nlog: bytes {offset}–{next_offset}, eof={eof}"
    if args.get("wait_s"):
        header += "\n[wait_s is not held in a side question; this is the status now]"
    return f"{header}\n\n{body}" if body else header


def _kind(block: Any) -> str:
    return (block.get("type") if isinstance(block, dict) else getattr(block, "type", "")) or ""


def _answer_view(view: Any, text: str, forwarded: bool) -> None:
    """`view.side_answer`, never raising. A view written before side questions inherits
    `View.side_answer`, which shows the answer as the desk's speech."""
    try:
        view.side_answer(text, forwarded)
    except Exception:  # noqa: BLE001 - presentation may not end a session
        pass
