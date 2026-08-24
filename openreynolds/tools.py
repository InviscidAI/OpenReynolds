"""The tool surface: seven tools, thin handlers, everything delegating to `Backend`.

There is no `run_gate`, no `amend_spec`, no `ask_user` — asking is just talking. Nothing
here inspects what the model is doing or refuses it on policy grounds. The handlers cap
output and report facts; that is the whole job.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from . import images
from .backend.base import Backend, BackendError, EXEC_MAX_TIMEOUT_S, WORKSPACE_ROOT
from .store import Store

SLOW_COMMAND_S = 10.0
"""Past this, how long a command took is worth saying."""

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
            "log_offset. Cheap to call repeatedly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
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
            "keeps running after your turn ends, and after this session closes."
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


def dispatch(ctx: ToolContext, name: str, tool_input: dict[str, Any]) -> tuple[ToolResult, bool]:
    """Run one tool call. Returns (content, is_error)."""
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"No such tool: {name}", True
    try:
        return handler(ctx, tool_input), False
    except BackendError as exc:
        return str(exc), True
    except Exception as exc:  # a harness bug is a fact the model should see
        return f"{type(exc).__name__}: {exc}", True


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
    return f"wrote {len(data)} bytes to {args['path']}"


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
    data = ctx.backend.get_file(path)
    shape = images.dimensions(data)
    described = f"{shape[0]}x{shape[1]} " if shape else ""
    return [
        images.attachment(data, media),
        {"type": "text", "text": f"{path} — {described}{media}, {len(data)} bytes"},
    ]


def _job_start(ctx: ToolContext, args: dict[str, Any]) -> str:
    job_id = ctx.backend.job_start(
        args["cmd"],
        cwd=args.get("cwd") or ctx.home,
        name=args.get("name"),
        kill_on=args.get("kill_on") or None,
    )
    ctx.store.record_job(job_id, cmd=args["cmd"], name=args.get("name"))
    _announce_jobs(ctx)
    label = f" ({args['name']})" if args.get("name") else ""
    return f"started job {job_id}{label}"


def _job_check(ctx: ToolContext, args: dict[str, Any]) -> str:
    job_id = args["job_id"]
    record = ctx.store.session.jobs.get(job_id)
    offset = args.get("log_offset")
    offset = int(offset) if offset is not None else (record.log_offset if record else 0)

    status = ctx.backend.job_status(job_id)
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
    header += f"\nlog: bytes {offset}–{next_offset}, eof={eof}"
    if clipped:
        header += f" (this window clipped at {ctx.max_output} bytes; call again from {offset + len(body.encode('utf-8'))})"
    return f"{header}\n\n{body}" if body else header


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
