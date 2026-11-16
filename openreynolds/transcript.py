"""A study's message log as a transcript: what a reader of the study is shown.

The agent's message log (`messages.jsonl`, written by `Store`) is the record of what it
did -- every user turn, every reply, every tool call with its input and output, every
fact the harness woke it with. It is append-only and the session writes it as it goes,
so a tail of it *is* the live transcript, and the same reader shows a study that ended
last week.

Every interface that shows a study reads it through here: the MCP servers
(`openreynolds.mcp`) and the hosted web app. So a row reads the same wherever it is
shown. An embedder that words some things its own way passes `say` (applied to the
harness's text before it is shown; return "" to hide a row) and `refusal` (the code of
a service refusal in a text, put on the item as `refusal`).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

OUTPUT_WIRE_BYTES = 12_000
"""Tool output carried on a transcript item. The full row is one `read_row` away."""

TURN_FAILED = "The model request failed, so this turn ended:"
"""How the agent's record of a failed model request begins (`cli._run_turn`): an
`event` row in messages.jsonl, then the reason. Every other event row is the harness
talking to the agent -- a briefing, progress facts, a finished job. This one is the
reason the agent stopped, so it gets a kind of its own (`turn_failed`), and a reader
can tell an agent that failed from one that finished."""

Say = Callable[[str], str]
Refusal = Callable[[str], "str | None"]


def _same(text: str) -> str:
    return text


def _none(text: str) -> str | None:  # noqa: ARG001
    return None


def row_to_item(row: dict[str, Any], *, say: Say = _same, refusal: Refusal = _none) -> dict[str, Any] | None:
    """One `messages.jsonl` row as a transcript item, or None if it shows nothing."""
    role = row.get("role")
    content = row.get("content")
    seq = row.get("seq")
    at = _epoch(row.get("at"))
    base = {"id": f"m{seq}", "seq": seq, "at": at}
    if role == "user":
        text = content if isinstance(content, str) else json.dumps(content, default=str)
        return {**base, "kind": "user", "text": text}
    if role == "assistant":
        text = content if isinstance(content, str) else ""
        if not text.strip():
            return None  # a turn that was only tool calls; the calls show themselves
        return {**base, "kind": "assistant", "text": text}
    if role == "event":
        if isinstance(content, dict) and content.get("desk") == "cad":
            return {**base, **desk_step_item(content)}
        text = content if isinstance(content, str) else json.dumps(content, default=str)
        text = say(text)
        if not text:
            return None
        if text.startswith(TURN_FAILED):
            return _with_refusal({**base, "kind": "turn_failed", "text": text}, text, refusal)
        return {**base, "kind": "event", "text": text, "label": event_label(text)}
    if role == "tool" and isinstance(content, dict):
        output = content.get("output") or ""
        if not isinstance(output, str):
            output = json.dumps(output, default=str)
        output = say(output)
        clipped = len(output.encode("utf-8")) > OUTPUT_WIRE_BYTES
        shown = output.encode("utf-8")[:OUTPUT_WIRE_BYTES].decode("utf-8", "ignore") if clipped else output
        item = {
            **base,
            "kind": "tool",
            "tool": content.get("tool", "?"),
            "input": content.get("input") or {},
            "output": shown,
            "output_clipped": clipped,
            "error": bool(content.get("error")),
        }
        # Only a failed call: a command that merely prints the word is not a refusal.
        return _with_refusal(item, output, refusal) if item["error"] else item
    return None


def _with_refusal(item: dict[str, Any], text: str, refusal: Refusal) -> dict[str, Any]:
    """`refusal` on an item whose text carries a service refusal code; absent, not
    null, on everything else."""
    code = refusal(text)
    return {**item, "refusal": code} if code else item


def desk_step_item(content: dict[str, Any]) -> dict[str, Any]:
    """One step of the CAD desk's work as a transcript shows it.

    The row is `cad.agent.Step.as_event()`, written by `cli._cad_desk_step` while the
    `cad` tool runs: one per cell the desk executes and one per verdict the reviewer
    gives. Its own kind rather than an `event`, because it is not something the agent
    was told; it is what the desk did. `blocking` marks a review that handed work back
    to the desk (`exit_code` 1 on a `review` row)."""
    kind = str(content.get("kind") or "cell")
    exit_code = content.get("exit_code")
    try:
        exit_code = int(exit_code) if exit_code is not None else None
    except (TypeError, ValueError):
        exit_code = None
    try:
        seconds = float(content.get("seconds") or 0.0)
    except (TypeError, ValueError):
        seconds = 0.0
    try:
        step = int(content.get("step") or 0)
    except (TypeError, ValueError):
        step = 0
    return {
        "kind": "desk_step",
        "step_kind": "review" if kind == "review" else "cell",
        "step": step,
        "text": str(content.get("text") or ""),
        "cmd": str(content.get("cmd") or ""),
        "output": str(content.get("output") or ""),
        "exit_code": exit_code,
        "seconds": round(seconds, 1),
        "image": str(content.get("image") or ""),
        "blocking": kind == "review" and exit_code not in (0, None),
    }


def event_label(text: str) -> str:
    head = text.strip().splitlines()[0] if text.strip() else ""
    if head.startswith("Progress facts"):
        return "progress facts"
    if head.startswith("job "):
        return "job finished"
    if "refreshed" in head or "fresh conversation thread" in text:
        return "thread refresh"
    if head.startswith("study "):
        return "briefing"
    return "from the harness"


def _epoch(stamp: Any) -> float | None:
    if not isinstance(stamp, str):
        return None
    try:
        return time.mktime(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
    except ValueError:
        return None


class TranscriptTail:
    """Reads new rows from a study's message log, remembering where it got to."""

    def __init__(self, path: Path, *, say: Say = _same, refusal: Refusal = _none):
        self.path = path
        self.say = say
        self.refusal = refusal
        self._offset = 0
        self._partial = b""
        self.last_seq = -1

    def read_new(self) -> list[dict[str, Any]]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self._offset:
            # Rewritten from scratch (a study directory reused); start over.
            self._offset, self._partial, self.last_seq = 0, b"", -1
        if size == self._offset:
            return []
        try:
            with self.path.open("rb") as fh:
                fh.seek(self._offset)
                chunk = fh.read(size - self._offset)
        except OSError:
            return []
        self._offset = size
        data = self._partial + chunk
        lines = data.split(b"\n")
        self._partial = lines.pop()  # possibly a half-written last line; kept for next time
        items = []
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            seq = row.get("seq")
            if isinstance(seq, int):
                self.last_seq = max(self.last_seq, seq)
            item = row_to_item(row, say=self.say, refusal=self.refusal)
            if item is not None:
                items.append(item)
        return items


def read_items(path: Path, *, say: Say = _same, refusal: Refusal = _none) -> list[dict[str, Any]]:
    """Every item of a study's message log, from the start."""
    return TranscriptTail(path, say=say, refusal=refusal).read_new()


def read_row(path: Path, seq: int) -> dict[str, Any] | None:
    """One full row by sequence number -- for the whole of a clipped output."""
    if not path.is_file():
        return None
    with path.open("rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            if row.get("seq") == seq:
                return row
    return None
