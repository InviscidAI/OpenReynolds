"""The tool surface: ten tools, thin handlers, everything delegating to `Backend`.

There is no `run_gate`, no `amend_spec`, no `ask_user` — asking is just talking. Nothing
here inspects what the model is doing or refuses it on policy grounds. The handlers cap
output and report facts; that is the whole job.

An eleventh, `checkpoint`, exists only when the person chose structured mode
(`modes.py`): it puts a summary in front of them and waits for their answer. It is the
person asking to be consulted, not the harness deciding to consult them, so in full
auto it is not in the list at all.

Three of the ten are the mesh desk's (`mesher/`): `mesh` starts a second agent -- one
bash block a step, on the same workspace -- on a thread of its own and returns at once,
`mesh_note` passes it a remark while it works, and `mesh_wait` holds for its result. The
result otherwise arrives on its own, through the same watch loop that wakes the model
when a job ends (`watch.py`). This module still knows nothing about how that agent
reaches its model.
"""

from __future__ import annotations

import re
import shlex
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import convergence, images
from . import trace
from .mesher.background import DeskRun, mesh_text
from .progress import case_dir_from_cmd, parse_control_dict, phase_from_cmd
from .backend.base import Backend, BackendError, EXEC_MAX_TIMEOUT_S, JobStatus, WORKSPACE_ROOT
from .store import Store

SLOW_COMMAND_S = 10.0
"""Past this, how long a command took is worth saying."""

JOB_WAIT_MAX_S = 300
"""Longest one `job_check` or `mesh_wait` call will hold its answer. Waiting again is free."""

JOB_WAIT_POLL_S = 5.0
"""How often a waiting `job_check` looks at the job."""

DESK_WAIT_POLL_S = 1.0
"""How often a waiting `mesh_wait` looks for typed input. The desk's own end is an
event and needs no polling; this is the latency at which a person is heard."""

READ_TIMEOUT_S = 60.0
"""How long `read_file` waits on the workspace for one file before saying so.

The backend's own default is 300 s, and until 2026-09-07 it was *unbounded* -- one
`stat` was watched sitting for over ten minutes. Measured on a contended workspace,
two of three stats of a file another session was writing timed out at 40 s while the
same file settled answered in 2.8 s, so a read that is going to be slow is usually
going to be very slow, and the model is better told that in a minute than blocked for
five. It reads again if it wants to; a tool call it cannot escape is the expensive
part."""

READ_ATTEMPTS = 2
"""Retries inside one `read_file`. A contended read that failed twice inside a minute
is a fact worth reporting, not one worth waiting out."""

TAIL_HINT_BYTES = 4_000
"""How far back from the end a truncation marker points, so the offered offset lands on
the tail rather than another copy of the head."""


@dataclass
class ToolContext:
    backend: Backend
    store: Store
    max_output: int
    home: str = WORKSPACE_ROOT
    """Where a command runs unless told otherwise: this study's own directory."""
    view: Any = None
    """Optional: told whenever job state changes, so a panel can stay current."""
    on_fetch: Callable[[list[Any]], None] | None = None
    """Called with the local paths `fetch` produced, for artifact capture."""
    on_wait_input: Callable[[], bool] | None = None
    """Whether the person has said something for the model that it has not seen yet.

    A waiting `job_check` or `mesh_wait` ends early on it, so a person who speaks
    during a held call is heard in seconds rather than when the wait runs out. The
    session loop answers it (`Loop.heard`): it drains the inbox as the loop does
    between tool calls, answers commands on the spot, and holds the words for the
    model to ride after this batch's results -- so what ended the wait is in the same
    message as the result that says so, and a line that is nobody's to deliver to the
    model (`/exit`, an EOF, a `/status`) ends nothing. It used to be the reader's
    `pending()`, a peek at the queue, and a put-back `/exit` made every wait for the
    rest of a turn answer at once with `waited 0s` (study 20260921-033019-e1b4)."""
    on_leaving: Callable[[], bool] | None = None
    """Whether the person has ended the session while this turn runs (`Loop.leaving`).

    The other question a held `job_check` or `mesh_wait` asks once a second, after
    `on_wait_input`: a `/exit`, a `/quit` or an EOF met by the drain is nobody's to
    deliver to the model, so it is not "heard" -- but it is the person leaving, and a
    wait held for them is held for nobody. The wait returns at once, saying so, and
    the loop ends the turn after this batch's results rather than at the model's next
    reply. Measured before this (study 20260921-033019-e1b4): End pressed at 03:33:40
    into a turn was honoured when the turn ended on its own, thirty-seven minutes
    later. None when nobody is wired to answer, and then no wait ends on it."""
    cores: int | None = None
    """What `nproc` reported, once it has been asked: hardware threads. See `_core_count`."""
    physical_cores: int | None = None
    """Physical cores, from `lscpu`, asked in the same round trip as `cores`. Zero when
    the machine would not say; then only the thread count is spoken of."""
    started: float = field(default_factory=time.monotonic)
    """When this session began.

    The one quantity nothing in the harness has ever said out loud, and the one the
    person waiting actually cares about. Every cost this file reports -- ranks, write
    times, steps -- is about the run being launched; none of them says what the study
    has already spent. A model told a solve is 20,000 steps and not told the study is
    fifty minutes old will reinvest every saving in more simulated time, which is good
    physics and the wrong trade when somebody is waiting."""
    echoes: dict[str, int] = field(default_factory=dict)
    """Hash of a large tool output -> the call that first produced it.

    `bash` output is 78-82% of everything the model is sent, and the single most
    expensive call measured in one study was `cat .toolbox/notes/*.md | head -200`:
    4,177 tokens that then rode along in all 76 requests after it, 21% of that study's
    whole tool-derived context. It is re-read from scratch in every study, `checkMesh`
    ran three times in one and `--help` was read twice in two pages.

    This forbids nothing. The model may ask again as often as it likes and gets a true
    answer; a byte-identical repeat is answered with a pointer to where the bytes
    already are in the thread, which is the same information without the second copy."""
    calls: int = 0
    """How many tool calls this session has dispatched, so an echo can name one."""
    on_render: Callable[[str], None] | None = None
    """Called with the workspace path whenever the model looks at an image.

    A render the model just examined is exactly the file the user wants on their
    machine right now, not at the next mirror cycle. The hook must not block and
    must not fail the read -- it is a nudge, and the picture matters more."""
    mesher: Any = None
    """The mesh desk (`mesher.Mesher`), when there is a model key to run one with.
    None means the three mesh tools are not offered, and meshing is the caller's own
    work like any other command."""
    desk: Any = None
    """The desk's current background run (`mesher.DeskRun`), or None.

    One at a time: a `mesh` call while this is live starts nothing and answers with
    where the run has got to. Cleared when the result has been handed over -- by
    `mesh_wait`, or by the session loop when the run's end woke the model
    (`take_desk`). Also recorded on the session (`store.session.desk`) so a resume
    can say the run was cut off with the process."""
    on_tokens: Callable[[dict], None] | None = None
    """Called with the model usage a tool spent on the session's behalf -- the mesh
    desk's steps -- so it lands in the same totals as the main loop's."""
    mode: str = "auto"
    """How much the person chose to be consulted (`modes.py`). Read at every tool call,
    so a `/mode` switch applies from the next one."""
    plan_approved: bool = False
    """Structured mode: whether a checkpoint has been approved since the session entered
    the mode. Until one has, `job_start` and `mesh` are held."""
    approver: Any = None
    """The `approval.Approver` that puts a question to the person. None when nobody can
    answer (a one-shot run), which is why a non-auto mode refuses to start there."""
    on_mode: Callable[[str], None] | None = None
    """Switches the session's mode (`Loop.set_mode`), for a checkpoint answered with
    "approve all"."""


def tools_for(ctx: ToolContext) -> list[dict[str, Any]]:
    """The tools this session can actually serve.

    Only the mesh desk's three are conditional: without a desk behind them the tools
    can do nothing but explain that, and a tool in the list that answers "not
    available" costs the model a call to find out. Taking them out of the list is also
    what makes the question answerable -- the same prompt run with the desk and without
    it, which is the only honest way to settle whether a slow natural-language
    sub-agent beats the bash the caller already has.

    `checkpoint` is the other: it is offered only in structured mode. A `/mode` switch
    into or out of structured therefore changes the tool list once, which rewrites the
    prompt cache from position 0 once; that is the price of the person's choice and it
    is paid only when they make it. Sorted by name either way, so the list for a given
    mode is always the same bytes.
    """
    offered = TOOLS if ctx.mesher is not None else [
        tool for tool in TOOLS if tool["name"] not in DESK_TOOLS
    ]
    if ctx.mode != "structured":
        return offered
    return sorted([*offered, CHECKPOINT_TOOL], key=lambda tool: tool["name"])


FRESH_SHELL = (
    "Each call runs in a fresh shell: `cwd` carries between calls, nothing else does "
    "-- variables you export, files you source and shell options are gone by the next "
    "one. The OpenFOAM environment and the tutorials' `RunFunctions` helpers are "
    "loaded again for you on every call, `bash` and `job_start` alike."
)
"""The shell contract, said once and repeated verbatim in both tools that run one.

A fact about the environment, not an instruction: it says what is so and leaves what
to do about it entirely alone. It is here because the only other way to learn it is by
hitting it, and what that produces is `127 command not found` -- a message that reads
as a broken image and points diagnosis at the wrong layer.

The second half is a claim about the sandbox, and it is only true because the exec
wrapper and the job wrapper both source the OpenFOAM bashrc and `RunFunctions` before
anything else. `tests/test_shell_contract.py` checks that against the service rather
than taking it on trust: a description that promises something the environment does not
do is the same class of bug as a listing that does not say it was cut short.

Identical in both descriptions on purpose. They are read independently -- a model
looking at `job_start` alone must get the whole fact -- and one wording repeated is
cheaper to keep true than two that drift."""


TOOLS: list[dict[str, Any]] = [
    {
        "name": "bash",
        "description": (
            "Run a shell command in the workspace and wait for it. "
            f"{FRESH_SHELL} Capped at {EXEC_MAX_TIMEOUT_S} seconds; use "
            "job_start for anything longer. Returns the exit code and the output, "
            "with a pointer to the full log on disk if the output was long."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "The command to run."},
                "cwd": {
                    "type": "string",
                    "description": (
                        "Directory to run in. Defaults to this study's own "
                        "directory, which your briefing names."
                    ),
                },
                "timeout_s": {
                    "type": "integer",
                    "description": f"Seconds to wait, up to {EXEC_MAX_TIMEOUT_S}. Default 120.",
                },
            },
            "required": ["cmd"],
        },
    },
    {
        "name": "fetch",
        "description": (
            "Copy files or directories out of the workspace onto the user's own "
            "machine, and print where they landed. Useful for renders and reports."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Workspace paths to copy out.",
                }
            },
            "required": ["paths"],
        },
    },
    {
        "name": "job_check",
        "description": (
            "Get a job's status together with whatever log has appeared since "
            "log_offset. Cheap to call repeatedly. With wait_s it holds the answer "
            "until the job ends or the wait runs out, ending early if the user says "
            "something (their words then follow this result in the same message; "
            "answer them and call again) - quieter than pacing with sleep in bash, "
            "which counts against the bash time cap."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "wait_s": {
                    "type": "integer",
                    "description": (
                        f"Seconds to hold the answer, up to {JOB_WAIT_MAX_S}, waiting "
                        "for the job to end. The wait is the harness's own and does "
                        "not count against any command timeout."
                    ),
                },
                "log_offset": {
                    "type": "integer",
                    "description": (
                        "Byte offset to read the log from. Defaults to where the last "
                        "check left off."
                    ),
                },
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "job_kill",
        "description": "Stop a running job.",
        "input_schema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    },
    {
        "name": "job_start",
        "description": (
            "Start a long command detached and return a job id immediately. The job "
            "keeps running after your turn ends, and after this session closes. "
            f"{FRESH_SHELL} "
            "A solver started serially holds one core for the whole run, however "
            "many the container has; a case put through `decomposePar` and started "
            "with `mpirun -np N` holds N. What the extra ranks return falls away as "
            "the cells each one holds get small, so there is an N past which they "
            "stop paying. The cells-per-core figures written for shared clusters sit "
            "well above it: those are about not tying up cores someone else could "
            "use, and these are rented, idle and billed either way. "
            "A decomposed run writes one set of files "
            "per rank per write time, and the workspace is a network filesystem that "
            "charges by the file: `-fileHandler collated`, passed to decomposePar, "
            "the solver and reconstructPar alike, writes one set instead of N, which "
            "reconstructs faster and leaves the solve unchanged. "
            "A solver launched into a case that already holds written time steps, with "
            "`startFrom startTime` in its controlDict, would start from t=0 again and "
            "write over them; this tool refuses that launch and says so. To carry a "
            "transient on -- more time, denser writes -- set `startFrom latestTime` "
            "(every write so far is kept and the run continues from the last one); "
            "to start over on purpose, pass overwrite=true or run in a fresh directory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string"},
                "cwd": {
                    "type": "string",
                    "description": (
                        "Directory to run in. Defaults to this study's own "
                        "directory, which your briefing names."
                    ),
                },
                "name": {"type": "string", "description": "A label for your own reference."},
                "kill_on": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Regexes matched against log lines. The first match terminates "
                        "the job and the matching line is reported back. Optional."
                    ),
                },
                "overwrite": {
                    "type": "boolean",
                    "description": (
                        "Allow a solver to restart from startTime in a case that already "
                        "holds written time steps, discarding them. Default false: the "
                        "launch is refused instead."
                    ),
                },
            },
            "required": ["cmd"],
        },
    },
    {
        "name": "mesh",
        "description": (
            "Describe a geometry in words and have an OpenFOAM mesh of it built on the "
            "workspace. A separate agent builds it on this same machine, in the "
            "background — it chooses the mesher (gmsh body-fitted, blockMesh, "
            "snappyHexMesh, cfMesh), writes the geometry as a script, renders the mesh "
            "and measures it, and revises until checkMesh passes and the shape measures "
            "up to what was asked for. This call returns at once with where the desk is "
            "working; the conversation carries on, and the result arrives here as a "
            "message when the desk finishes or gets stuck (typically 2–8 minutes): the "
            "picture's path, the patch table with each patch's area and normal, "
            "checkMesh's verdict, and where the case is. mesh_note passes the person's "
            "remarks about the geometry to the desk while it works; mesh_wait holds for "
            "the result when there is nothing else to do. One desk runs at a time. It is "
            "a MESH only: no fields, no boundary conditions, no solver settings and no "
            "solve — those stay with you. wait=true runs it in the foreground instead "
            "and returns the result directly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": (
                        "The shape in words: its sizes with units, which end is the "
                        "inlet and which the outlet, whether it is a plane (2D) case "
                        "or a volume, and any property that has to be right — an "
                        "angle, a radius, a gap, a count. Anything you leave out is "
                        "the mesh desk's to choose."
                    ),
                },
                "case": {
                    "type": "string",
                    "description": (
                        "Directory name for the case under the study (default 'mesh')."
                    ),
                },
                "wait": {
                    "type": "boolean",
                    "description": (
                        "Hold this call until the desk is done and return the result "
                        "here, as the tool did before it had a background. Default "
                        "false: the desk runs on its own and the result arrives as a "
                        "message."
                    ),
                },
            },
            "required": ["request"],
        },
    },
    {
        "name": "mesh_note",
        "description": (
            "Pass a remark to the mesh desk while it is building: a change to the "
            "shape, a correction, a detail the person added. The desk reads it at its "
            "next command and works to it, and its result says what it heard. Answers "
            "whether a desk was there to hear it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The remark, in the person's words where it is theirs.",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "mesh_wait",
        "description": (
            "Hold the answer until the running mesh desk finishes, up to wait_s "
            f"seconds (at most {JOB_WAIT_MAX_S}), ending early if the user says "
            "something (their words then follow this result in the same message; "
            "answer them and call again). When the desk has finished this returns its "
            "full result — the "
            "picture, the patch table, checkMesh's verdict, where the case is — and "
            "otherwise where it has got to. The result also arrives on its own as a "
            "message when the desk finishes, so this is for when there is nothing "
            "else to do meanwhile."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "wait_s": {
                    "type": "integer",
                    "description": (
                        f"Seconds to hold the answer, up to {JOB_WAIT_MAX_S}; default "
                        f"{JOB_WAIT_MAX_S}. 0 answers at once with where the desk is. "
                        "The wait is the harness's own and does not count against any "
                        "command timeout."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a window of a file as text, or list a directory. Byte offsets, so "
            "arbitrarily large files are readable a piece at a time. A path ending in "
            ".png, .jpg, .gif or .webp comes back as the picture itself, so anything "
            "you render you can also look at."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer", "description": "Byte offset to start at."},
                "limit": {"type": "integer", "description": "Bytes to read."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write text to a path in the workspace. Parent directories are created.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
]

TOOL_NAMES = frozenset(tool["name"] for tool in TOOLS)

DESK_TOOLS = frozenset({"mesh", "mesh_note", "mesh_wait"})
"""The three that exist only when the mesh desk does (`tools_for`)."""

CHECKPOINT_TOOL: dict[str, Any] = {
    "name": "checkpoint",
    "description": (
        "Put where the study stands in front of the person and wait for their answer. "
        "Offered because they chose structured mode: they want to agree a plan, and to "
        "hear from you after each stage. The result says whether they approved, and "
        "carries their words when they asked for changes. job_start and mesh calls are "
        "held until a checkpoint has been approved in this mode."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "stage": {
                "type": "string",
                "description": (
                    "What this checkpoint is about: plan, or a stage of the study "
                    "(geometry, preview, mesh, checkMesh, probe, solve, reconstruct, "
                    "render, animate, report)."
                ),
            },
            "summary": {
                "type": "string",
                "description": "What was done, or what is proposed, in plain words.",
            },
            "next": {
                "type": "string",
                "description": "What happens next if the person approves.",
            },
        },
        "required": ["stage", "summary", "next"],
    },
}
"""Kept out of `TOOLS`: that list is what every mode offers, and this is structured
mode's alone (`tools_for`)."""


ToolResult = str | list[dict[str, Any]]
"""What a handler gives back: text, or content blocks when text cannot carry it."""


ECHO_MIN_BYTES = 2_000
"""Below this, the sentence explaining a repeat costs about what the repeat does."""


def _echo(ctx: ToolContext, body: str) -> str:
    """`body`, or a pointer to the call that already returned exactly these bytes."""
    if len(body) < ECHO_MIN_BYTES:
        return body
    import hashlib

    digest = hashlib.sha256(body.encode("utf-8", "replace")).hexdigest()
    first = ctx.echoes.get(digest)
    if first is None:
        ctx.echoes[digest] = ctx.calls
        return body
    head = body.strip().splitlines()[0][:120] if body.strip() else ""
    return (f"[identical, byte for byte, to what call #{first} returned — it is still "
            f"in this conversation, so it is not repeated here]\n{head}")


def _traced_size(result: ToolResult) -> int:
    """Roughly how much this result costs the next request, for the trace row.

    Cheap on purpose: it is read once per call and only when something is
    listening. A picture is counted as its own bytes rather than rendered, because
    `describe` base64s it and the number wanted here is what the transport carries.
    """
    if isinstance(result, str):
        return len(result)
    total = 0
    for block in result or ():
        if isinstance(block, dict):
            source = block.get("source") or {}
            total += len(block.get("text") or "") + len(source.get("data") or "")
    return total


def dispatch(ctx: ToolContext, name: str, tool_input: dict[str, Any],
             call_id: str | None = None) -> tuple[ToolResult, bool]:
    """Run one tool call. Returns (content, is_error).

    `call_id` is the model's own id for this call, passed through only so a trace
    row can be joined to the turn that asked for it. It is optional because the
    tool has no use for it: a trace with timings and no identity could say a tool
    took nine seconds and not which of the four calls in that turn it was.
    """
    ctx.calls += 1
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"No such tool: {name}", True
    # Read the clock only when something is listening: the handlers below time
    # themselves off the same `time.monotonic`, and tests drive that with a fake.
    started = time.monotonic() if trace.on else 0.0
    ok, size = True, 0
    try:
        result = handler(ctx, tool_input)
        if trace.on:
            size = _traced_size(result)
        # Only text is collapsed, and only when it repeats exactly. An image is already
        # handled by the eviction policy, and an error is never worth collapsing --
        # the same failure twice is a fact about the run, not a duplicate.
        return (_echo(ctx, result) if isinstance(result, str) else result), False
    except BackendError as exc:
        ok = False
        return str(exc), True
    except Exception as exc:  # a harness bug is a fact the model should see
        ok = False
        return f"{type(exc).__name__}: {exc}", True
    finally:
        if trace.on:
            trace.event(
                "tool",
                tool=name,
                seconds=round(time.monotonic() - started, 3),
                cmd=str(tool_input.get("cmd") or tool_input.get("path") or "")[:200],
                tool_use_id=call_id,
                result={"ok": ok, "bytes": size},
            )


# -- handlers ------------------------------------------------------------------


def _bash(ctx: ToolContext, args: dict[str, Any]) -> str:
    asked = int(args.get("timeout_s") or 120)
    started = time.monotonic()
    result = ctx.backend.exec(args["cmd"], cwd=args.get("cwd") or ctx.home, timeout_s=asked)
    elapsed = time.monotonic() - started
    notes = []
    if elapsed >= SLOW_COMMAND_S:
        # A four-minute command and a two-second one read identically otherwise, so
        # the cost of what was just done is invisible to whoever chose to do it.
        notes.append(f"[took {elapsed:.0f}s]")
    mesher = _mesher_note(ctx, args)
    if mesher:
        notes.append(mesher.strip())
    ran_with = min(asked, EXEC_MAX_TIMEOUT_S)
    if asked > EXEC_MAX_TIMEOUT_S:
        # Say so rather than clamping quietly: a command cut off at a ceiling the
        # caller did not know about reads as a command that finished.
        notes.append(
            f"[timeout_s={asked} exceeds the {EXEC_MAX_TIMEOUT_S}s ceiling for a "
            f"synchronous command; it ran with {EXEC_MAX_TIMEOUT_S}s. job_start has "
            "no such limit.]"
        )
    if result.exit_code == -1:
        # The same trap one step further on: a command killed at its timeout reports
        # no status, and a bare "exit_code: -1" reads like a command that merely
        # produced nothing.
        notes.append(
            f"[exit_code -1 means no exit status was reported, which is what happens "
            f"when a command outruns its timeout_s (this one ran with {ran_with}s). "
            "job_start has no time limit.]"
        )
    if getattr(result, "stderr", "").strip():
        # The workspace itself, not the command: a working directory that is gone, a
        # capture-sync that failed. Surfaced so a platform failure is legible as one
        # instead of reading like a command that merely produced nothing and being
        # retried into the same wall.
        notes.append(f"[workspace: {result.stderr.strip()[:300]}]")
    if getattr(result, "job_id", ""):
        # The command ran past the synchronous window and became a detached job. Said
        # plainly so the next step is job_check on this id, not a re-run of a command
        # that is already running.
        notes.append(
            f"[moved to detached job {result.job_id}: it outran the synchronous exec "
            f"window and is running now -- follow it with job_check, do not re-run it]"
        )
    total = None
    if result.truncated and result.log_path:
        try:
            total = ctx.backend.stat(result.log_path).size
        except BackendError:
            total = None

    body, clipped = _clip(result.output, ctx.max_output)
    lines = [f"exit_code: {result.exit_code}"]
    lines.extend(notes)
    if result.truncated or clipped:
        lines.append(_truncation_marker(len(body.encode("utf-8")), total, result.log_path))
    lines.append("")
    lines.append(body)
    return "\n".join(lines)


def _write_file(ctx: ToolContext, args: dict[str, Any]) -> str:
    data = args["content"].encode("utf-8")
    ctx.backend.put_file(args["path"], data)
    return f"wrote {len(data)} bytes to {args['path']}{_written_run_shape(args)}"


def _written_run_shape(args: dict[str, Any]) -> str:
    """How many times a `controlDict` just asked to be written to disk.

    `endTime` over `writeInterval` is the cheapest number in a study to get wrong: it
    is two tokens in a dictionary and it sets how much disk the run needs, how long a
    later `reconstructPar` walks -- that part is serial, however many ranks solved --
    and how much of the workspace the mirror has to carry home. A live run chose 900
    over 0.2, and the four and a half thousand write times that implies cost more in
    reconstruction than the eight extra ranks had saved in solving.

    Said here rather than at `job_start` because this is where the number is chosen,
    and here it costs nothing to say: the content is already in hand, so there is no
    round trip. Arithmetic on two declared values, not a forecast."""
    path = str(args.get("path") or "")
    if not path.endswith("system/controlDict"):
        return ""
    control = parse_control_dict(args.get("content") or "")
    end, every = control.get("endTime"), control.get("writeInterval")
    if end is None or not every:
        return ""
    span = end - (control.get("startTime") or 0.0)
    parts = [f"endTime {end:g} / writeInterval {every:g} = {int(span / every)} write times"]
    step = control.get("deltaT")
    if step and not _ADJUST_DT.search(args.get("content") or ""):
        # The number that actually sets what this run costs, and the one nobody
        # writes down. A live case chose endTime 300 at deltaT 0.005 -- sixty
        # thousand steps, two hours -- with its Courant number already at 0.69, so
        # there was no larger timestep to be had and no way to spend less once the
        # solve began. The write count was visible at this moment and the step
        # count was not, and it was the step count that decided.
        parts.append(f"deltaT {step:g} = {int(span / step)} steps at that timestep")
    return f" [{', '.join(parts)}]"


def _read_file(ctx: ToolContext, args: dict[str, Any]) -> str | list[dict[str, Any]]:
    path = args["path"]
    info = ctx.backend.stat(path, timeout=READ_TIMEOUT_S, max_attempts=READ_ATTEMPTS)
    if info.is_dir:
        listing = "\n".join(info.entries) if info.entries else "(empty)"
        return f"{path} — directory, {len(info.entries)} entries\n\n{listing}"

    media = images.media_type(path)
    if media is not None and not (args.get("offset") or args.get("limit")):
        return _read_image(ctx, path, info, media)

    offset = max(0, int(args.get("offset") or 0))
    limit = int(args.get("limit") or ctx.max_output)
    raw = ctx.backend.get_file(path, offset=offset, limit=limit,
                               timeout=READ_TIMEOUT_S, max_attempts=READ_ATTEMPTS)
    raw, size, moving = _read_again_if_it_moved(ctx, path, info.size, raw, offset, limit)
    text = raw.decode("utf-8", errors="replace")
    body, clipped = _clip(text, ctx.max_output)

    end = offset + len(raw)
    header = f"{path} — bytes {offset}–{end} of {size}"
    if clipped:
        header += f" (shown to {offset + len(body.encode('utf-8'))}; raise offset for more)"
    elif end < size:
        header += f"; {size - end} bytes remain past this window"
    return f"{header}{moving}\n\n{body}"


def _size_now(ctx: ToolContext, path: str) -> int | None:
    """What the path measures at this moment, or nothing if it cannot be asked.

    A file that vanished between the read and the check is a race worth surviving
    quietly: the bytes are in hand and they were real when they were fetched. Turning
    that into a failed tool call would trade a rare inaccuracy for a common one."""
    try:
        return ctx.backend.stat(path, timeout=READ_TIMEOUT_S,
                                max_attempts=READ_ATTEMPTS).size
    except Exception:  # noqa: BLE001 - a check that cannot run makes no claim
        return None


def _read_again_if_it_moved(ctx: ToolContext, path: str, size: int, raw: bytes,
                            offset: int, limit: int) -> tuple[bytes, int, str]:
    """Read once more when the file grew under the read, and say so if it is still growing.

    `_read_file` stats the path, asks for exactly that many bytes and is handed exactly
    that many, so the short-read guard that covers pictures -- `len(data) < info.size`
    in `_read_image` -- cannot fire on anything else. A `postProcessing` forces file, a
    `.dat`, a `.csv` or a solver log read while OpenFOAM is still appending to it comes
    back agreeing with itself all the way up, and a file cut short is a number the model
    will happily average (F-64). Images were given their own answer after a half-written
    PNG ended two sessions -- `images.incomplete` refuses one with no end marker -- and
    the text that the conclusions are actually drawn from had nothing.

    A second stat is the whole check. If the size moved, the writer was mid-flight while
    the bytes were being fetched, and one more read usually lands on a finished file; it
    costs one round trip on a path that already makes several. If the size is still
    moving after that, no number of reads will settle it, so what comes back says which
    of the two it is rather than looking whole either way.
    """
    grown = _size_now(ctx, path)
    if grown is None or grown == size:
        return raw, size, ""
    raw = ctx.backend.get_file(path, offset=offset, limit=limit,
                               timeout=READ_TIMEOUT_S, max_attempts=READ_ATTEMPTS)
    settled = _size_now(ctx, path)
    if settled is None or settled == grown:
        return raw, grown if settled is None else settled, ""
    return raw, settled, (
        f"\nStill being written: {size} bytes when it was measured, {grown} after the "
        f"first read, {settled} after the second. What follows is a snapshot of a file "
        "something is still appending to, so it may stop part-way through a line or a "
        "record; the path read again once the writer has finished gives the whole of it."
    )


def _read_image(ctx: ToolContext, path: str, info: Any, media: str) -> str | list[dict[str, Any]]:
    """Hand back a render as a picture rather than as a description of one.

    An oversized one is reported as a size, not silently dropped: a picture that never
    arrives and a picture of nothing look identical from the inside.
    """
    if info.size > images.MAX_ATTACH_BYTES:
        return (
            f"{path} — {media}, {info.size} bytes. Images are returned as pictures up "
            f"to {images.MAX_ATTACH_BYTES} bytes; this one is larger, so only its size "
            "is reported here."
        )
    # Ask for the whole thing by name. A backend answering an unbounded read with its
    # own page size is the normal case, and a picture cut off at that boundary is not
    # a smaller picture -- it is a corrupt one that still passes every check here.
    data = ctx.backend.get_file(path, limit=info.size, timeout=READ_TIMEOUT_S,
                                max_attempts=READ_ATTEMPTS)
    if len(data) < info.size:
        return (
            f"{path} — {media}, {info.size} bytes, but only {len(data)} came back. "
            "A part of an image is not a smaller image, so it is not attached."
        )
    # The check above catches a read the transport cut short. This one catches a file
    # that was whole when it was measured and half-written when it was made: a figure
    # read back while matplotlib is still saving it stats at its current size, reads
    # back exactly that many bytes, and agrees with itself all the way to the API,
    # which refuses it with a 400 and ends the session. images.incomplete carries the
    # incident. Saying so as text is strictly better than attaching it: the model finds
    # out the picture is not ready and can simply look again.
    why = images.incomplete(data, media)
    if why is not None:
        return (
            f"{path} — {media}, {len(data)} bytes, but not a whole image: {why}. "
            "Nothing is attached, because a partial image is refused by the model API "
            "rather than shown. If something is still writing this file, wait for it "
            "to finish and read the path again."
        )
    if ctx.on_render is not None:
        try:
            ctx.on_render(path)
        except Exception:  # noqa: BLE001 - a nudge may not cost the model its picture
            pass
    shape = images.dimensions(data)
    described = f"{shape[0]}x{shape[1]} " if shape else ""
    # Scaled for transport only. The description keeps the real dimensions and the real
    # size, so what the model is told about the file stays true to the file on disk.
    sent = images.downscale(data, media)
    note = "" if sent is data else f", sent at {images.ATTACH_MAX_EDGE}px"
    return [
        images.attachment(sent, media),
        {"type": "text", "text": f"{path} — {described}{media}, {len(data)} bytes{note}"},
    ]


def _job_start(ctx: ToolContext, args: dict[str, Any]) -> str:
    refusal = _restart_guard(ctx, args)
    if refusal:
        return refusal
    job_id = ctx.backend.job_start(
        args["cmd"],
        cwd=args.get("cwd") or ctx.home,
        name=args.get("name"),
        kill_on=args.get("kill_on") or None,
    )
    ctx.store.record_job(
        job_id, cmd=args["cmd"], name=args.get("name"), cwd=args.get("cwd") or ctx.home
    )
    _announce_jobs(ctx)
    label = f" ({args['name']})" if args.get("name") else ""
    return f"started job {job_id}{label}{_solve_shape(ctx, args)}{_mesher_note(ctx, args)}"


_TIME_DIR = re.compile(r"/(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/?$")
"""A time directory's name at the end of a path: `0.8`, `100`, `1e-05`. Anything else
under a case or a `processors*` directory -- `constant`, `system`, `log.x` -- is not."""


def _written_times(ctx: ToolContext, case: str) -> list[float]:
    """The time steps a case already holds on disk, decomposed or not.

    One `ls` covers the three places OpenFOAM writes them: the case itself, one
    `processorN/` per rank (the default decomposed layout) and `processorsN/` (the
    collated layout, one directory for all ranks). An unreadable case -- no directory,
    a workspace that did not answer -- yields no times, and the guard that reads this
    then does not fire: a guard on bad information would refuse first launches."""
    listing = (
        f"ls -d {shlex.quote(case)}/[0-9]* {shlex.quote(case)}/processor*/[0-9]* "
        f"{shlex.quote(case)}/processors*/[0-9]* 2>/dev/null"
    )
    try:
        out = ctx.backend.exec(listing, timeout_s=30).output or ""
    except Exception:  # noqa: BLE001 - not knowing is not a reason to refuse
        return []
    times: set[float] = set()
    for line in out.splitlines():
        found = _TIME_DIR.search(line.strip())
        if found:
            try:
                times.add(float(found.group(1)))
            except ValueError:
                continue
    return sorted(times)


def _restart_guard(ctx: ToolContext, args: dict[str, Any]) -> str:
    """Refuse a solver launch that would write over a transient's results.

    The loss this stops, measured: a 2D transient had run 22 minutes to t=0.8 when
    the person asked for a gif. The agent raised `endTime`, tightened `writeInterval`
    and relaunched the same command in the same directory -- and the controlDict said
    `startFrom startTime`, so pimpleFoam began again at t=0 and rewrote the
    `processors4/` time directories it had spent those minutes on (study
    20260920-161908-c7ef). Nothing in the harness said a word.

    The rule is the smallest one that catches it: a solving command, a case that
    already holds time steps later than `startTime`, and a controlDict whose
    `startFrom` is not `latestTime` (`startTime` is OpenFOAM's usual choice, and an
    absent entry is refused too, because the dictionary does not parse without one).
    The refusal names the times at stake and the two honest ways forward; a launch that
    truly means to start over says `overwrite=true`. Meshers, post-processing, and the
    first launch of a case (no times yet) are never touched."""
    if args.get("overwrite"):
        return ""
    cmd = args.get("cmd") or ""
    if phase_from_cmd(cmd)[0] != "solving":
        return ""
    case = case_dir_from_cmd(cmd, args.get("cwd") or ctx.home)
    try:
        text = ctx.backend.get_file(f"{case}/system/controlDict").decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - no controlDict: the solver will say so itself
        return ""
    control = parse_control_dict(text)
    if control.get("startFrom") == "latestTime":
        return ""
    start = control.get("startTime") or 0.0
    later = [t for t in _written_times(ctx, case) if t > start]
    if not later:
        return ""
    return (
        f"not started: {case} already holds {len(later)} written time step(s), from "
        f"{later[0]:g} to {later[-1]:g}, and its controlDict says startFrom "
        f"{control.get('startFrom') or 'startTime'} (startTime {start:g}). This launch would "
        "start the solve again from there and write over them. To carry the run on from "
        f"t={later[-1]:g} -- more time, a different writeInterval -- set `startFrom "
        "latestTime;` in system/controlDict and start again (every write so far is kept). "
        "To start over on purpose, call job_start with overwrite=true, or copy the case to "
        "a new directory first so the finished results stay where they are."
    )


_EMPTY_PATCH = re.compile(r"^\s*type\s+empty\s*;", re.M)

_ADJUST_DT = re.compile(r"^\s*adjustTimeStep\s+(yes|on|true)\s*;", re.M | re.I)
"""Whether the solver sets its own timestep. When it does, `deltaT` is only where it
starts and a step count derived from it is a number about nothing -- a live case
wrote deltaT 0.001 under `maxCo 0.8`, and the 200,000 steps that implies was never
going to happen."""


def _is_two_dimensional(ctx: ToolContext, case: str) -> bool:
    """Whether the case's mesh has an `empty` patch, i.e. is one cell thick.

    Read from `constant/polyMesh/boundary` rather than the blockMeshDict, because it
    is what the mesh actually is rather than what a dictionary asked for."""
    try:
        head = ctx.backend.get_file(f"{case}/constant/polyMesh/boundary", limit=8000)
    except Exception:  # noqa: BLE001 - no mesh yet is not a failed launch
        return False
    return bool(_EMPTY_PATCH.search(head.decode("utf-8", "replace")))


def _mesher_shape(ctx: ToolContext, cmd: str, case: str) -> str:
    """What `snappyHexMesh` is about to be asked to do on a one-cell-thick mesh.

    snappy is a three-dimensional mesher. Its snapping phase works by displacing
    points onto the surface and checking mesh quality after each move; in a case
    with an `empty` patch the third direction is pinned, so the displacement it
    wants is the displacement it may not make. It does not fail -- it scales the
    displacement back and tries again, over every cell, indefinitely.

    Observed on this workspace: a 245,805-cell 2D case where the snapping phase
    attracted 0 of 16,896 points to a feature edge, a feature point, or the nearest
    surface, and was still moving the mesh forty minutes later. The agent watching
    it read the phase wrong -- it blamed smoothing, which the log timed at 0.52 s --
    and let it run. So say which phase costs what, at the moment the job starts,
    while it is still cheap to choose the other mesher."""
    if "snappyHexMesh" not in cmd or not _is_two_dimensional(ctx, case):
        return ""
    return (
        " [this mesh has an empty patch, so it is one cell thick. snappyHexMesh is a "
        "3D mesher: in a pinned direction it scales its displacement back and retries "
        "rather than failing, so it stalls instead of stopping. blockMesh and cfMesh's "
        "cartesian2DMesh mesh 2D directly]"
    )


def _mesher_note(ctx: ToolContext, args: dict[str, Any]) -> str:
    """The mesher warning, for whichever tool launched it."""
    cmd = args.get("cmd") or ""
    return _mesher_shape(ctx, cmd, case_dir_from_cmd(cmd, args.get("cwd") or ctx.home))


def _solve_shape(ctx: ToolContext, args: dict[str, Any]) -> str:
    """What the solve that was just started is set to do, read back from its case.

    The same argument as the `[took Ns]` note on `bash`: a run that ends in ninety
    seconds and one that ends in nine hours read identically at the moment they are
    launched, and the choice that separates them was made several turns earlier in a
    file nobody looked at again. `endTime` and `writeInterval` are the two numbers
    that decide it, and the count of writes they imply is arithmetic on them rather
    than a guess about the future -- the bar's estimates stay with the bar
    (`progress.Tracker.facts_for_wake`), which sends the model facts and no forecast.

    Silent whenever it has nothing certain to say: not a solver, no controlDict, no
    `endTime` to speak of. A note that appears only sometimes is a note worth reading.
    """
    cmd = args.get("cmd") or ""
    phase, executable = phase_from_cmd(cmd)
    if phase != "solving":
        return ""
    # A steady solver's residuals on an unsteady flow level off and stay there, and a
    # model not told to expect that read the plateau as a failed run in four studies
    # of five (`convergence`). Said at the launch, because the residuals are the next
    # thing it reads; said for steady solvers only, because a transient's per-step
    # residuals do not have this shape.
    steady = f" [{convergence.STEADY_LAUNCH_NOTE}]" if convergence.is_steady_solver(executable) else ""
    case = case_dir_from_cmd(cmd, args.get("cwd") or ctx.home)
    try:
        text = ctx.backend.get_file(f"{case}/system/controlDict").decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - a missing dict is not a failed launch
        return steady
    control = parse_control_dict(text)
    parts = []
    end, every = control.get("endTime"), control.get("writeInterval")
    if end is not None:
        parts.append(f"endTime {end:g}")
        if every:
            span = end - (control.get("startTime") or 0.0)
            parts.append(f"writeInterval {every:g} ({int(span / every)} write times)")
        step = control.get("deltaT")
        if step and not _ADJUST_DT.search(text):
            # Said here as well as at the `write_file` that chose it, because a
            # dictionary is as often written by a `bash` heredoc, and this reads the
            # file that is actually on disk rather than the one that went past.
            span = end - (control.get("startTime") or 0.0)
            parts.append(f"deltaT {step:g} ({int(span / step)} steps at that timestep)")
    # The two entries that decide whether a transient's results survive the NEXT
    # launch, said at this one while they are still cheap to change: `startFrom
    # latestTime` is what lets a run be carried on, and `purgeWrite N` throws away all
    # but the last N writes as it goes -- a gif made from the "written" times finds N
    # of them.
    if control.get("startFrom"):
        parts.append(f"startFrom {control['startFrom']}")
    purge = control.get("purgeWrite")
    if purge:
        parts.append(f"purgeWrite {purge} (only the last {purge} write times are kept on disk)")
    parts.append(_load_per_rank(ctx, cmd, case))
    parts.append(_study_age(ctx))
    shape = f" [{', '.join(p for p in parts if p)}]" if any(parts) else ""
    return shape + steady


def _study_age(ctx: ToolContext) -> str:
    """How long this study has taken so far. A fact, and the only one about the
    thing the person is actually waiting on. Said at the moment a long run is
    committed to, because that is when spending more of it is chosen."""
    minutes = (time.monotonic() - ctx.started) / 60.0
    if minutes < 1:
        return ""
    return f"this study is {minutes:.0f} min old so far"


_NCELLS = re.compile(r"nCells:\s*(\d+)")


def _cell_count(ctx: ToolContext, case: str) -> int:
    """The mesh's cell count, from the header `blockMesh` and `snappyHexMesh` write.

    `constant/polyMesh/owner` carries it in a `note` line just past the banner --
    around byte 690, and further on a build whose architecture string is longer -- so
    two kilobytes covers it with room to spare and still costs one small read rather
    than a parse of the mesh."""
    try:
        head = ctx.backend.get_file(f"{case}/constant/polyMesh/owner", limit=2000)
    except Exception:  # noqa: BLE001 - no mesh yet is not a failed launch
        return 0
    found = _NCELLS.search(head.decode("utf-8", "replace"))
    return int(found.group(1)) if found else 0


def _load_per_rank(ctx: ToolContext, cmd: str, case: str) -> str:
    """How much mesh each rank is carrying.

    Cells per rank is the number this decision turns on, and it is the one nobody
    has. Reporting cores instead invites an all-or-nothing reading: a live run saw
    "32 cores" against a 20,650-cell mesh, judged -- correctly -- that thirty-two
    ranks would spend more on halo exchange than they saved, and concluded from that
    that it should run on one. The choice was never between 1 and 32.

    So say what each rank would hold, and let the arithmetic be visible. Silent when
    the mesh has not been built yet, or when the machine has one core and there is no
    choice to describe."""
    cores = _core_count(ctx)
    if cores <= 1:
        return ""
    ranks = _ranks_in(cmd)
    cells = _cell_count(ctx, case)
    machine = _machine_line(ctx, ranks)
    if not cells:
        return f"{ranks} rank(s), {machine}"
    return f"{cells} cells on {ranks} rank(s) = {cells // ranks} each; {machine}"


def _machine_line(ctx: ToolContext, ranks: int) -> str:
    """Threads and cores, told apart.

    `nproc` counts hardware threads, and the sentence "8 cores on this machine" was
    built on it. On a machine with two threads a core that sentence is wrong by half,
    and it cost a launch: told 8 cores, an agent decomposed for 6 ranks and Open MPI
    refused the run -- "not enough slots" -- because its default slot count is the
    PHYSICAL cores, four (study 20260920-161908-c7ef, c7i.2xlarge). The agent then
    re-decomposed for 4. So both numbers are said, and what MPI will accept without
    being told otherwise; the extra ranks a thread would add are named for what they
    are, because a CFD solve is bound by memory bandwidth and two ranks on one core
    share it."""
    threads = ctx.cores or 0
    physical = ctx.physical_cores or 0
    if physical and physical < threads:
        line = (f"{threads} hardware threads = {physical} physical cores on this machine; "
                f"mpirun accepts up to {physical} ranks as it is (more needs "
                "--use-hwthread-cpus, and ranks that share a core gain little in CFD)")
        if ranks > physical:
            line += f" -- {ranks} ranks is more than the {physical} cores"
        return line
    return f"{threads} cores on this machine"


_MPIRUN_NP = re.compile(r"\bmpirun\b[^|;&]*?-np\s+(\d+)")


def _ranks_in(cmd: str) -> int:
    """How many ranks the command asked for. One, unless `mpirun -np N` says otherwise."""
    found = _MPIRUN_NP.search(cmd)
    return int(found.group(1)) if found else 1


CORES_PROBE = (
    "nproc; lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l"
)
"""Two numbers in one round trip: hardware threads, then distinct (core, socket) pairs
from lscpu -- the physical cores. The second is 0 where lscpu is missing or refuses."""


def _core_count(ctx: ToolContext) -> int:
    """What `nproc` says, asked once per session, with the physical core count beside it.

    The count is a property of the container and does not change under us, so paying
    a round trip for it on every launch would be paying repeatedly for the same
    answer. Zero means it could not be established, and nothing is said. Returns the
    thread count (what `nproc` has always meant here); `ctx.physical_cores` holds the
    other number for `_machine_line`."""
    if ctx.cores is None:
        try:
            result = ctx.backend.exec(CORES_PROBE, timeout_s=30)
            fields = (result.output or "").split()
            ctx.cores = int(fields[0])
            ctx.physical_cores = int(fields[1]) if len(fields) > 1 and fields[1].isdigit() else 0
        except Exception:  # noqa: BLE001 - not knowing is not a failed launch
            ctx.cores = 0
            ctx.physical_cores = 0
    return ctx.cores


def _job_check(ctx: ToolContext, args: dict[str, Any]) -> str:
    job_id = args["job_id"]
    record = ctx.store.session.jobs.get(job_id)
    offset = args.get("log_offset")
    offset = int(offset) if offset is not None else (record.log_offset if record else 0)

    status = ctx.backend.job_status(job_id)
    asked_wait = int(args.get("wait_s") or 0)
    wait_s = min(asked_wait, JOB_WAIT_MAX_S)
    waited_note = ""
    if wait_s > 0 and status.running:
        began = time.monotonic()
        heard = left = False
        while status.running and time.monotonic() - began < wait_s:
            # `_heard` drains the inbox, which is also how the loop learns of a
            # `/exit`; asked second, the leaving question sees one the drain has just
            # met. Both are remembered, and leaving wins the note below: words typed
            # in the same breath as End ride with this result, but are not answered.
            heard = _heard(ctx)
            left = _leaving(ctx)
            if heard or left:
                break
            remaining = wait_s - (time.monotonic() - began)
            time.sleep(max(0.0, min(JOB_WAIT_POLL_S, remaining)))
            status = ctx.backend.job_status(job_id)
        waited_note = f"[waited {time.monotonic() - began:.0f}s]"
        if asked_wait > JOB_WAIT_MAX_S:
            waited_note += (
                f" [wait_s={asked_wait} exceeds the {JOB_WAIT_MAX_S}s ceiling for "
                "one call; waiting again is free]"
            )
        if left:
            waited_note += " " + _left_note()
        elif heard:
            # Remembered from the loop rather than asked again here: the question
            # drains the inbox, so a second asking would find it empty and drop the
            # one sentence that explains a wait of 0 s.
            waited_note += " " + _heard_note("job_check", "the job is still running")
    data, next_offset, eof = ctx.backend.job_tail(job_id, offset=offset)
    ctx.store.update_job(
        job_id,
        status=status.status,
        end_reason=status.end_reason,
        exit_code=status.exit_code,
        log_offset=next_offset,
    )
    _announce_jobs(ctx)

    body, clipped = _clip(data, ctx.max_output)
    header = describe_job(status)
    if waited_note:
        header += " " + waited_note
    header += _running_on(ctx, record, status)
    header += f"\nlog: bytes {offset}–{next_offset}, eof={eof}"
    if clipped:
        header += f" (this window clipped at {ctx.max_output} bytes; call again from {offset + len(body.encode('utf-8'))})"
    return f"{header}\n\n{body}" if body else header


def _running_on(ctx: ToolContext, record: Any, status: JobStatus) -> str:
    """How much of the machine a still-running solve is using.

    `_solve_shape` says this once, at launch, and a launch is a bad moment to hear
    it: nothing has been spent yet and the number is abstract. A `job_check` is the
    other moment -- the solve is real, the wait is being paid for, and killing it and
    starting again is still cheaper than finishing. So the fact is repeated where the
    decision is, and only while it can still be acted on.

    Reads the command off the job record rather than the workspace, so it costs
    nothing. Silent for a finished job (there is nothing to change), for anything
    that is not a solver, and when the core count could not be established."""
    if not status.running or record is None:
        return ""
    cmd = getattr(record, "cmd", "") or ""
    if phase_from_cmd(cmd)[0] != "solving":
        return ""
    load = _load_per_rank(ctx, cmd, case_dir_from_cmd(cmd, getattr(record, "cwd", "") or ctx.home))
    return f"\n{load}" if load else ""


def _job_kill(ctx: ToolContext, args: dict[str, Any]) -> str:
    status = ctx.backend.job_kill(args["job_id"])
    ctx.store.update_job(
        args["job_id"], status=status.status, end_reason=status.end_reason
    )
    _announce_jobs(ctx)
    return describe_job(status)


def _fetch(ctx: ToolContext, args: dict[str, Any]) -> str:
    paths = list(args["paths"])
    written = ctx.backend.get_tree(paths, ctx.store.fetch_dir())
    if not written:
        return "nothing was copied out"
    if ctx.on_fetch:
        ctx.on_fetch(written)
    listing = "\n".join(f"  {p}" for p in written)
    return f"copied {len(written)} file(s) to the user's machine:\n{listing}"


def _mesh(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Start the mesh desk on a thread of its own and say so, at once.

    Measured before this: one `mesh` call held the loop's thread for 402 s and some
    twenty-five model calls (study 20260920-161908-c7ef), during which the main agent
    answered nothing and every line the person typed was drained into the desk. The
    desk's work was fine; the thread it ran on was the problem. Now the call returns
    with where the desk is working, the conversation carries on, and the finished
    result reaches the model as a wake (`watch.watch`) or through `mesh_wait`.

    `wait: true` is the old shape, whole: the call holds until the desk is done and
    the desk hears the session's inbox directly, as it did before there was a
    background. Kept for callers that want exactly that.
    """
    if ctx.mesher is None:
        return (
            "the mesh desk is not available in this session (it needs a model key of "
            "its own to run); meshing here is yours to do with bash like anything else"
        )
    request = str(args.get("request", ""))
    case = args.get("case")
    if args.get("wait"):
        return mesh_content(_counted(ctx, ctx.mesher.run(request, case=case)))

    running = ctx.desk
    if running is not None and not running.done.is_set():
        # One at a time. Its request is not queued behind the running one either --
        # a queue is a second thing to lose track of, and the caller can call again.
        return (
            f"{running.progress_line()}. One mesh desk runs at a time in a session, so "
            "this call started nothing and its request was not queued. mesh_note passes "
            "a remark to the running desk (a change of shape included); mesh_wait holds "
            "for its result, which also arrives here on its own when it finishes."
        )
    lead: list[dict[str, Any]] = []
    if running is not None:
        # It finished while the caller was working and nothing has handed the result
        # over yet -- the wake would have, at the end of this turn. Delivered here
        # first, so the new run does not bury the old one's answer.
        finished = take_desk(ctx, running)
        lead = _blocks(
            f"The mesh desk on `{running.case_rel}` had finished; its result first:",
            _finished_content(running, finished),
        )
    run = DeskRun(ctx.mesher, request, case).start()
    ctx.desk = run
    ctx.store.session.desk = run.record()
    ctx.store.save()
    started = _desk_started(run)
    return [*lead, {"type": "text", "text": started}] if lead else started


def _desk_started(run: Any) -> str:
    return (
        f"the mesh desk has started on `{run.case_rel}`. It works on its own from here, "
        "and its result arrives in this conversation as a message when it finishes or "
        "gets stuck (typically 2-8 minutes), so nothing here has to wait for it. "
        "Meanwhile: talk with the person, write the case files that do not depend on "
        "the mesh (controlDict, fvSchemes, fvSolution, transportProperties, "
        f"boundary-condition drafts) in a directory of their own rather than in "
        f"`{run.case_rel}`, which is the desk's to write into, and do not poll it with "
        "sleep -- a sleep only holds the turn the result is waiting for. mesh_note "
        "passes the person's remarks about the geometry to the desk; mesh_wait blocks "
        "only when you truly have nothing else to do."
    )


def _mesh_note(ctx: ToolContext, args: dict[str, Any]) -> str:
    """A remark for the running desk. It goes into the desk's own queue, which the
    desk drains at its next step (`DeskRun.take_remarks`); the session's inbox is
    never read on the desk's behalf any more."""
    run = ctx.desk
    text = str(args.get("text") or "").strip()
    if run is None:
        return "no mesh desk is running in this session, so there was nobody to note this for"
    if run.done.is_set():
        return (f"the mesh desk on `{run.case_rel}` has already finished, so this note "
                "reached nobody; mesh_wait returns its result")
    if not text:
        return "nothing was noted: the text was empty"
    run.note(text)
    return f"noted for the mesh desk on `{run.case_rel}`; it reads it at its next command"


def _mesh_wait(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Hold for the running desk, the way a `job_check` with `wait_s` holds for a job:
    bounded, and ending early the moment the person says something for the model."""
    run = ctx.desk
    if run is None:
        return "no mesh desk is running in this session"
    asked = JOB_WAIT_MAX_S if args.get("wait_s") is None else int(args.get("wait_s") or 0)
    wait_s = min(max(asked, 0), JOB_WAIT_MAX_S)
    began = time.monotonic()
    heard = left = False
    while not run.done.is_set():
        elapsed = time.monotonic() - began
        if elapsed >= wait_s:
            break
        # As in `_job_check`: `_heard` drains, the leaving question asked after it
        # sees a `/exit` the drain has just met, and leaving wins the note.
        heard = _heard(ctx)
        left = _leaving(ctx)
        if heard or left:
            break
        run.done.wait(min(DESK_WAIT_POLL_S, wait_s - elapsed))
    notes = []
    if wait_s > 0:
        notes.append(f"[waited {time.monotonic() - began:.0f}s]")
    if asked > JOB_WAIT_MAX_S:
        notes.append(f"[wait_s={asked} exceeds the {JOB_WAIT_MAX_S}s ceiling for one call; "
                     "waiting again is free]")
    if run.done.is_set():
        result = take_desk(ctx, run)
        return _blocks(" ".join(notes), _finished_content(run, result))
    if left:
        notes.append(_left_note())
    elif heard:
        notes.append(_heard_note("mesh_wait", "the desk is still building"))
    return " ".join([run.progress_line(), *notes])


def _heard(ctx: ToolContext) -> bool:
    """Whether the person has said something the model has not seen yet -- asked of the
    loop, which drains the inbox to answer and holds the words for this batch's results
    (`ToolContext.on_wait_input`, `Loop.heard`). False when nobody is wired to answer."""
    return ctx.on_wait_input is not None and bool(ctx.on_wait_input())


def _leaving(ctx: ToolContext) -> bool:
    """Whether the person has ended the session -- asked of the loop, which learned it
    from the same drain `_heard` runs (`ToolContext.on_leaving`, `Loop.leaving`). False
    when nobody is wired to answer, so a wait with no session behind it holds as it
    always did."""
    return ctx.on_leaving is not None and bool(ctx.on_leaving())


def _left_note() -> str:
    """Why a held answer came back before its time when the person ended the session.

    Says only what the tool knows: the wait ended because the person left, and the
    session is on its way down. Not what becomes of the job or the desk -- the
    close-down decides that (`cli._close_down`, `--keep-alive`), and the loop says the
    rest in its own line right after this result (`loop.LEFT_MID_TURN`). The model is
    not asked again, so this is for whoever reads the transcript."""
    return "[the person ended the session, so this answered early; the session is closing down]"


def _heard_note(again: str, still: str) -> str:
    """Why a held answer came back before its time, and what comes next.

    Three facts, because a wait cut short with fewer read as a tool that does not
    wait: the person wrote (the reason); their words are in this same message, after
    the tool results (where to look -- the loop puts them there, `Loop._turns`); and
    what was being waited on is still going, so the same call, made again after
    answering them, holds as it did. Measured without the last two (study
    20260921-033019-e1b4): three returns in nine seconds saying only "the user said
    something", no words following, and the model switched to `bash sleep 115` for
    the rest of the study."""
    return (f"[the person wrote, so this answered early; their words follow this result. "
            f"Answer them, then call {again} again -- {still}]")


def _finished_content(run: Any, result: Any) -> ToolResult:
    """A finished run as a tool result -- or, when the run raised instead of
    returning, the one honest sentence about it."""
    if result is None:
        return (f"the mesh desk on `{run.case_rel}` stopped without a result "
                f"({run.error or 'no reason recorded'}); whatever it built is on disk "
                f"under `{run.case_rel}`")
    return mesh_content(result)


def _blocks(lead: str, content: ToolResult) -> ToolResult:
    """`content` with `lead` in front of its words, whatever shape it came in."""
    lead = lead.strip()
    if not lead:
        return content
    if isinstance(content, str):
        return f"{lead}\n{content}"
    out: list[dict[str, Any]] = []
    for block in content:
        if lead and block.get("type") == "text":
            # The picture stays first; the words after it carry the lead.
            out.append({**block, "text": f"{lead}\n{block.get('text', '')}"})
            lead = ""
        else:
            out.append(block)
    if lead:
        out.append({"type": "text", "text": lead})
    return out


def _counted(ctx: ToolContext, result: Any) -> Any:
    """The desk's tokens into the session's totals, so `/status` shows what the study
    actually spent; the result is handed back for the words."""
    if ctx.on_tokens and result is not None and result.tokens:
        ctx.on_tokens(result.tokens)
    return result


def take_desk(ctx: ToolContext, run: Any) -> Any:
    """Hand a finished run over, once: its tokens into the session's totals, the
    registry and the session's record cleared. Returns the run's `MeshResult`, or None
    when the run raised. Called by `mesh_wait` and by the session loop when the run's
    end woke the model, whichever gets there first; the second caller finds nothing
    left to count."""
    if ctx.desk is run:
        ctx.desk = None
    if ctx.store.session.desk:
        ctx.store.session.desk = {}
        ctx.store.save()
    if run.delivered:
        return run.result
    run.delivered = True
    return _counted(ctx, run.result)


def mesh_content(result: Any) -> ToolResult:
    """The mesh desk's answer as one tool result: the picture first, the words second,
    so that when the picture is later evicted from the thread the caption still carries
    the patch table, the verdict and where the case is."""
    text = mesh_text(result)
    if result.png:
        return [images.attachment(images.downscale(result.png, "image/png"), "image/png"),
                {"type": "text", "text": text}]
    return text


def _checkpoint(ctx: ToolContext, args: dict[str, Any]) -> str:
    """Ask the person, and hand back their answer as a fact.

    Approval is recorded on the context, because structured mode holds `job_start` and
    `mesh` until a plan has been approved (`Loop._consult`). Anything but an approval
    carries the person's own words back as the changes they asked for."""
    stage = str(args.get("stage") or "").strip() or "plan"
    summary = str(args.get("summary") or "").strip()
    after = str(args.get("next") or "").strip()
    if ctx.mode != "structured":
        return (f"The session is in {ctx.mode} mode, not structured mode, so this "
                "checkpoint was not put to the person.")
    if ctx.approver is None:
        return "Nobody is here to answer, so this checkpoint was not put to anyone."
    detail = summary + (f"\n\nnext: {after}" if after else "")
    decision = ctx.approver.ask("checkpoint", f"Checkpoint: {stage}", detail)
    if decision.approved:
        ctx.plan_approved = True
        text = f"The person approved the {stage} checkpoint. Carry on with: {after or 'the next stage'}"
        if decision.all and ctx.on_mode is not None:
            ctx.on_mode("auto")
            text += ("\nThey also asked not to be asked again: the session is now in full "
                     "auto mode and no further checkpoints are put to them.")
        return text
    if decision.note:
        return (f"The person did not approve the {stage} checkpoint. The changes they "
                f"asked for, in their words: {decision.note}")
    return f"The person did not approve the {stage} checkpoint and gave no reason."


_HANDLERS: dict[str, Callable[[ToolContext, dict[str, Any]], ToolResult]] = {
    "bash": _bash,
    "checkpoint": _checkpoint,
    "fetch": _fetch,
    "mesh": _mesh,
    "mesh_note": _mesh_note,
    "mesh_wait": _mesh_wait,
    "job_check": _job_check,
    "job_kill": _job_kill,
    "job_start": _job_start,
    "read_file": _read_file,
    "write_file": _write_file,
}


# -- helpers -------------------------------------------------------------------


def _announce_jobs(ctx: ToolContext) -> None:
    """Job state changed; anything showing it should hear about it now."""
    if ctx.view is not None:
        ctx.view.jobs(list(ctx.store.session.jobs.values()))


def describe(content: ToolResult) -> str:
    """A text rendering of a tool result, for anything that has to store or print one.

    Base64 image data belongs in the request and nowhere else -- a megabyte of it in
    the message log makes the log unreadable and unsearchable for no gain.
    """
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") == "image":
            source = block.get("source", {})
            size = len(source.get("data", "")) * 3 // 4
            parts.append(f"[{source.get('media_type', 'image')}, {size} bytes]")
    return "\n".join(parts)


def describe_job(status: Any) -> str:
    """One factual line about a job. No interpretation."""
    parts = [f"job {status.job_id}"]
    if status.name:
        parts.append(f"name={status.name}")
    parts.append(f"status={status.status}")
    if status.exit_code is not None:
        parts.append(f"exit_code={status.exit_code}")
    if status.end_reason:
        parts.append(f"end_reason={status.end_reason}")
    if status.log_size is not None:
        parts.append(f"log_size={status.log_size}")
    running_for = _running_for(status)
    if running_for:
        parts.append(running_for)
    line = " ".join(parts)
    if status.killed_by:
        # Not "matched kill_on line" any more. That label was right for as long as
        # a `kill_on` regex was the only thing that ever wrote `killed_by`; F-48
        # made the service write it on every terminal state that is a kill, from
        # six producers, and only one of them is a log line. The others are
        # sentences: the container was gone, the process group was gone, the state
        # could not be read, a client sent a signal. Keeping the old label would
        # have told the model that a container which vanished was a regex match --
        # a wrong label on the one field that exists to stop it guessing why its
        # solve died, which is the whole of F-48.
        #
        # `end_reason` already says which kind this is (`kill_on_match` for the
        # log line), so this does not repeat it and does not parse the text: the
        # service's wording changes with what it can measure, and it is meant to
        # be read rather than matched on.
        line += f"\nwhy it ended: {status.killed_by.strip()}"
    return line


def _running_for(status: Any) -> str:
    """How long a job has been going, when the service says enough to work it out.

    Two hours in and one minute in are the same line otherwise, and which of those it
    is changes what anyone would do about it.
    """
    start = _moment(getattr(status, "started_at", None))
    if start is None:
        return ""
    end = _moment(getattr(status, "ended_at", None))
    if end is None:
        if not status.running:
            return ""
        end = time.time()
    seconds = end - start
    if seconds < 0:
        return ""
    verb = "running_for" if status.running else "ran_for"
    return f"{verb}={seconds / 60:.1f}min" if seconds >= 60 else f"{verb}={seconds:.0f}s"


def _moment(value: Any) -> float | None:
    """Seconds since the epoch, from whatever shape the service used."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        from datetime import datetime

        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _clip(text: str, limit: int) -> tuple[str, bool]:
    """Cut text to a byte budget without splitting a character."""
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text, False
    return encoded[:limit].decode("utf-8", errors="ignore"), True


def _truncation_marker(shown_bytes: int, total: int | None, log_path: str | None) -> str:
    """Say which end was kept and how to reach the other.

    The service returns the *head* of a command's log, so on a long solve the part that
    matters most is exactly the part that was cut. The marker has to be explicit about
    that, and hand over an offset that lands on the tail.
    """
    of_total = f" of {total}" if total else ""
    marker = f"[truncated — first {shown_bytes} bytes{of_total} shown"
    if log_path:
        marker += f"; full output at {log_path}"
        if total:
            tail = max(0, total - TAIL_HINT_BYTES)
            marker += f'; read_file(path="{log_path}", offset={tail}) for the tail'
        else:
            marker += "; read_file with an offset to window into it"
    marker += "]"
    return marker
