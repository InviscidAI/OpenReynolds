"""The tool surface: seven tools, thin handlers, everything delegating to `Backend`.

There is no `run_gate`, no `amend_spec`, no `ask_user` — asking is just talking. Nothing
here inspects what the model is doing or refuses it on policy grounds. The handlers cap
output and report facts; that is the whole job.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import images
from . import trace
from .progress import case_dir_from_cmd, parse_control_dict, phase_from_cmd
from .backend.base import Backend, BackendError, EXEC_MAX_TIMEOUT_S, JobStatus, WORKSPACE_ROOT
from .store import Store

SLOW_COMMAND_S = 10.0
"""Past this, how long a command took is worth saying."""

JOB_WAIT_MAX_S = 300
"""Longest one `job_check` call will hold its answer. Waiting again is free."""

JOB_WAIT_POLL_S = 5.0
"""How often a waiting `job_check` looks at the job."""

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
    """Whether the user has said something not yet delivered, without taking it.

    A waiting `job_check` ends early on it, so a person who speaks during a held
    call is heard in seconds rather than when the wait runs out."""
    cores: int | None = None
    """What `nproc` reported, once it has been asked. See `_core_count`."""
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


TOOLS: list[dict[str, Any]] = [
    {
        "name": "bash",
        "description": (
            "Run a shell command in the workspace and wait for it. The OpenFOAM "
            f"environment is sourced. Capped at {EXEC_MAX_TIMEOUT_S} seconds; use "
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
            "something - quieter than pacing with sleep in bash, which counts "
            "against the bash time cap."
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
            "reconstructs faster and leaves the solve unchanged."
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
            },
            "required": ["cmd"],
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


def dispatch(ctx: ToolContext, name: str, tool_input: dict[str, Any]) -> tuple[ToolResult, bool]:
    """Run one tool call. Returns (content, is_error)."""
    ctx.calls += 1
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"No such tool: {name}", True
    # Read the clock only when something is listening: the handlers below time
    # themselves off the same `time.monotonic`, and tests drive that with a fake.
    started = time.monotonic() if trace.on else 0.0
    try:
        result = handler(ctx, tool_input)
        # Only text is collapsed, and only when it repeats exactly. An image is already
        # handled by the eviction policy, and an error is never worth collapsing --
        # the same failure twice is a fact about the run, not a duplicate.
        return (_echo(ctx, result) if isinstance(result, str) else result), False
    except BackendError as exc:
        return str(exc), True
    except Exception as exc:  # a harness bug is a fact the model should see
        return f"{type(exc).__name__}: {exc}", True
    finally:
        if trace.on:
            trace.event(
                "tool",
                tool=name,
                seconds=round(time.monotonic() - started, 3),
                cmd=str(tool_input.get("cmd") or tool_input.get("path") or "")[:200],
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
    info = ctx.backend.stat(path)
    if info.is_dir:
        listing = "\n".join(info.entries) if info.entries else "(empty)"
        return f"{path} — directory, {len(info.entries)} entries\n\n{listing}"

    media = images.media_type(path)
    if media is not None and not (args.get("offset") or args.get("limit")):
        return _read_image(ctx, path, info, media)

    offset = max(0, int(args.get("offset") or 0))
    limit = int(args.get("limit") or ctx.max_output)
    raw = ctx.backend.get_file(path, offset=offset, limit=limit)
    text = raw.decode("utf-8", errors="replace")
    body, clipped = _clip(text, ctx.max_output)

    end = offset + len(raw)
    header = f"{path} — bytes {offset}–{end} of {info.size}"
    if clipped:
        header += f" (shown to {offset + len(body.encode('utf-8'))}; raise offset for more)"
    elif end < info.size:
        header += f"; {info.size - end} bytes remain past this window"
    return f"{header}\n\n{body}"


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
    data = ctx.backend.get_file(path, limit=info.size)
    if len(data) < info.size:
        return (
            f"{path} — {media}, {info.size} bytes, but only {len(data)} came back. "
            "A part of an image is not a smaller image, so it is not attached."
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
    if phase_from_cmd(cmd)[0] != "solving":
        return ""
    case = case_dir_from_cmd(cmd, args.get("cwd") or ctx.home)
    try:
        text = ctx.backend.get_file(f"{case}/system/controlDict").decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - a missing dict is not a failed launch
        return ""
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
    parts.append(_load_per_rank(ctx, cmd, case))
    parts.append(_study_age(ctx))
    return f" [{', '.join(p for p in parts if p)}]" if any(parts) else ""


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
    if not cells:
        return f"{ranks} rank(s), {cores} cores on this machine"
    return (
        f"{cells} cells on {ranks} rank(s) = {cells // ranks} each; "
        f"{cores} cores on this machine"
    )


_MPIRUN_NP = re.compile(r"\bmpirun\b[^|;&]*?-np\s+(\d+)")


def _ranks_in(cmd: str) -> int:
    """How many ranks the command asked for. One, unless `mpirun -np N` says otherwise."""
    found = _MPIRUN_NP.search(cmd)
    return int(found.group(1)) if found else 1


def _core_count(ctx: ToolContext) -> int:
    """What `nproc` says, asked once per session.

    The count is a property of the container and does not change under us, so paying
    a round trip for it on every launch would be paying repeatedly for the same
    answer. Zero means it could not be established, and nothing is said."""
    if ctx.cores is None:
        try:
            result = ctx.backend.exec("nproc", timeout_s=30)
            ctx.cores = int((result.output or "").strip().split()[0])
        except Exception:  # noqa: BLE001 - not knowing is not a failed launch
            ctx.cores = 0
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
        while status.running and time.monotonic() - began < wait_s:
            if ctx.on_wait_input is not None and ctx.on_wait_input():
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
        if status.running and ctx.on_wait_input is not None and ctx.on_wait_input():
            waited_note += " [the user said something, so this answered early]"
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


_HANDLERS: dict[str, Callable[[ToolContext, dict[str, Any]], ToolResult]] = {
    "bash": _bash,
    "fetch": _fetch,
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
        line += f"\nmatched kill_on line: {status.killed_by.strip()}"
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
