"""The session as newline-delimited JSON, in both directions.

An agent driving this tool was in the same position a script was before
`ONE_SHOT_EXIT_CODES` existed: everything it needed had been said, in rich markup, on
the same stream as everything else. The study id -- the one fact needed to resume --
reached the screen only inside `ConsoleView.header`'s prose, so resuming meant either
scraping a styled line or listing `studies/` behind the tool's back.

This is the fourth interface behind the `View` seam (`ConsoleView`, `TuiView`, the
hosted `WebView`, this), and it is presentation only in the same way they are: it can
change what an agent reads and never what the model does.

Both halves of the protocol live here on purpose. The wire format is one contract --
objects out, objects in -- and splitting it across two modules is how the two halves
drift apart.

OUT: one JSON object per line, each with `v` (schema), `type`, `at` (seconds since the
view was made) and `study`. Every write goes through one lock and is flushed, because
three of the methods below are called from other threads (`progress` from the tracker,
`delivered`/`mirrored` from the mirror, `desk`/`narration` from the concierge). TuiView
solves the same problem with `call_from_thread`; a writer to a stream has to solve it
itself, and a half-written line is not a parse error a reader can recover from -- it
silently rejoins the stream mid-object.

IN: `{"type": "user", "text": "..."}`, one per line. Anything else on the line is
ignored rather than guessed at.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .view import View
from .watch import NOTHING

SCHEMA = 1
"""Bumped when the shape of an event changes in a way that could break a reader.
Adding a field is not that; renaming or removing one is."""

DELTA_FLUSH_CHARS = 240
"""How much model text accumulates before it goes out as its own event.

`text_delta` fires per token. One object per delta costs roughly twenty bytes of
envelope for three bytes of text, and gives a reader nothing that a line does not --
so deltas are coalesced to a line or this many characters, whichever comes first, and
the whole message goes out again as one `message` event at `turn_end`. An agent that
wants to stream reads `text`; an agent that wants the answer reads `message` and
ignores the rest. Buffering only at `turn_end` was the other option and it is worse:
a turn that spends four minutes in tool calls would say nothing at all until it ended.
"""


class _Locked:
    """A write-and-flush that holds the view's lock, for `trace` to share.

    `trace.event` writes one line per call, so one lock acquisition per write makes
    its cost events safe to interleave with view events on the same stream. Without
    this the two writers race and the stream stops being parseable at the first
    collision -- which is the failure that cannot be recovered from downstream.
    """

    def __init__(self, view: JsonView):
        self._view = view

    def write(self, text: str) -> None:
        with self._view.lock:
            self._view.sink.write(text)
            self._view.sink.flush()

    def flush(self) -> None:
        """Already flushed on write; here so any stream-shaped caller is satisfied."""


class JsonView(View):
    """Every session event as one JSON object on one line.

    `sink` is injected rather than assumed to be stdout so a test can read the stream
    back, and so nothing here has an opinion about where the bytes go.
    """

    def __init__(self, sink: Any = None, *, schema: int = SCHEMA):
        self.sink = sink if sink is not None else sys.stdout
        self.lock = threading.Lock()
        self.schema = schema
        self._t0 = time.monotonic()
        self._study = ""
        self._browser: Any = None
        self._text: list[str] = []
        self._said = 0
        """How much of `_text` has already gone out as a `text` event."""
        self._thinking: list[str] = []
        self._thought = 0
        self._progress_key: tuple = ()

    # -- the wire --------------------------------------------------------------

    def emit(self, type_: str, **fields: Any) -> None:
        """One object, one line, flushed.

        Never raises: a view that can break a session is a view that decides
        something, and this one is presentation. A sink that has gone away (a reader
        that closed the pipe) ends the stream, not the study.
        """
        row = {
            "v": self.schema,
            "type": type_,
            "at": round(time.monotonic() - self._t0, 3),
            "study": self._study,
        }
        row.update(fields)
        try:
            line = json.dumps(row, default=str)
        except (TypeError, ValueError):
            line = json.dumps({**{k: row[k] for k in ("v", "type", "at", "study")},
                               "unserialisable": type_})
        try:
            with self.lock:
                self.sink.write(line + "\n")
                self.sink.flush()
        except (OSError, ValueError):
            pass

    def trace_sink(self) -> _Locked:
        """A stream for `trace` to write its cost events to, sharing this lock."""
        return _Locked(self)

    @property
    def origin(self) -> float:
        """The monotonic reading every `at` on this stream is measured from.

        Handed to `trace.to()` so the cost events share it. Two clocks writing one
        field name on one stream is not a schema a reader can be expected to notice:
        it just makes the stream non-monotonic in `at`.
        """
        return self._t0

    # -- session boundaries ----------------------------------------------------

    def header(self, study_id: str, instance_id: str, model: str, mirror: Path) -> None:
        """The first event, and the one that makes a run resumable.

        `--study <id>` has always been able to reopen a run; until this event the id
        existed on screen only as rich markup inside a sentence, so an agent that
        wanted to resume had to scrape a styled line or go behind the tool and read
        the studies directory itself.
        """
        self._study = study_id
        self.emit(
            "session_start",
            study_id=study_id,
            instance_id=instance_id,
            model=model,
            dir=str(mirror),
        )

    def session_end(self, outcome: str | None) -> None:
        """The last event. Not a `View` method -- the protocol has no end-of-session
        call, because a terminal's end of session is the shell prompt coming back.

        `outcome` is what `-p` returns and what `ONE_SHOT_EXIT_CODES` maps, repeated
        here because a reader consuming the stream through a pipe sees the exit code
        last, or not at all.
        """
        self.emit("session_end", outcome=outcome or "ok")

    # -- the model's turn ------------------------------------------------------

    def thinking_begin(self) -> None:
        self.emit("thinking_begin")

    def thinking_delta(self, text: str) -> None:
        self._thinking.append(text)
        self._flush_thinking(force=False)

    def text_delta(self, text: str) -> None:
        self._text.append(text)
        self._flush_text(force=False)

    def turn_end(self) -> None:
        """Flush what is left, then say the whole thing once.

        The whole message is repeated deliberately: an agent that buffered the deltas
        itself has to trust that it saw all of them, and an agent that ignored them
        has one object to read.
        """
        self._flush_text(force=True)
        self._flush_thinking(force=True)
        whole = "".join(self._text)
        thought = "".join(self._thinking)
        self._text, self._said = [], 0
        self._thinking, self._thought = [], 0
        self.emit("message", role="assistant", text=whole, thinking=thought)

    def _flush_text(self, force: bool) -> None:
        pending = "".join(self._text)[self._said:]
        if not pending:
            return
        if not force and len(pending) < DELTA_FLUSH_CHARS and "\n" not in pending:
            return
        self._said += len(pending)
        self.emit("text", text=pending)

    def _flush_thinking(self, force: bool) -> None:
        pending = "".join(self._thinking)[self._thought:]
        if not pending:
            return
        if not force and len(pending) < DELTA_FLUSH_CHARS and "\n" not in pending:
            return
        self._thought += len(pending)
        self.emit("thinking", text=pending)

    # -- what the harness did --------------------------------------------------

    def tool(self, name: str, summary: str) -> None:
        self.emit("tool", name=name, summary=summary)

    def tool_error(self, message: str) -> None:
        self.emit("tool_error", message=message)

    def notice(self, message: str) -> None:
        self.emit("notice", message=message)

    def warn(self, message: str) -> None:
        self.emit("warn", message=message)

    def info(self, message: str) -> None:
        self.emit("info", message=message)

    def usage(self, tokens: int, fraction: float) -> None:
        self.emit("usage", tokens=tokens, fraction=fraction)

    def watching(self, names: list[str]) -> None:
        self.emit("watching", names=list(names))

    def jobs(self, records: list[Any]) -> None:
        """Job state whenever it changes.

        `ConsoleView` inherited this as a silent no-op for its whole life, so a piped
        run dropped every state change -- the one structured fact an agent watching a
        four-hour solve actually wants.
        """
        self.emit("jobs", jobs=[_job(record) for record in records])

    def stage(self, text: str) -> None:
        self.emit("stage", text=text)

    def step(self, number: int, seconds: float, tool_calls: int) -> None:
        self.emit("step", number=number, seconds=round(seconds, 3), tool_calls=tool_calls)

    def interjection(self, text: str) -> None:
        self.emit("interjection", text=text)

    # -- looking at the workspace ----------------------------------------------

    def workspace(self, browser: Any) -> None:
        self._browser = browser
        self.emit("workspace", home=str(getattr(browser, "home", "")))

    def show_files(self, path: str = "", depth: int = 0) -> None:
        if self._browser is None:
            self.emit("files", path=path, depth=depth, entries=[], error="no workspace")
            return
        target = path or self._browser.home
        try:
            entries = (
                self._browser.tree(target, depth) if depth else self._browser.tree(target)
            )
        except Exception as exc:  # noqa: BLE001 - a listing that failed is an answer
            self.emit("files", path=target, depth=depth, entries=[], error=str(exc))
            return
        self.emit(
            "files",
            path=target,
            depth=depth,
            entries=[_entry(item) for item in entries],
            notice=str(getattr(entries, "notice", "") or ""),
        )

    def show_renders(self, renders_dir: Any) -> None:
        directory = Path(renders_dir)
        pics = (
            sorted(
                (p for p in directory.iterdir() if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if directory.is_dir()
            else []
        )
        self.emit("renders", dir=str(directory), files=[str(p) for p in pics])

    def status(self, lines: list[str]) -> None:
        self.emit("status", lines=list(lines))

    def mirrored(self, report: Any) -> None:
        """Called from the mirror's own thread; the lock in `emit` is what makes that
        safe here."""
        self.emit(
            "mirrored",
            pulled=[str(p) for p in getattr(report, "pulled", None) or ()],
            bytes=int(getattr(report, "bytes_pulled", 0) or 0),
            unchanged=int(getattr(report, "unchanged", 0) or 0),
            seconds=round(float(getattr(report, "seconds", 0.0) or 0.0), 3),
            local_dir=str(getattr(report, "local_dir", "")),
            warnings=list(getattr(report, "warnings", None) or ()),
        )

    def prompt(self) -> None:
        """It is the reader's turn. Without this an agent driving the conversational
        shape has no moment to answer at and either talks over a turn or deadlocks."""
        self.emit("prompt")

    def progress(self, snapshot: Any) -> None:
        """Pushed about once a second from the tracker's thread, whether or not a turn
        is in flight.

        Once a second forever is a lot of stream for a reader that mostly wants to
        know a solve is alive, so this says something only when the picture changes:
        the phase, the headline, or the percentage. `ConsoleView` makes the same
        judgement for the same reason and additionally repeats itself on a timer,
        which a reader parsing objects has no use for.
        """
        fraction = getattr(snapshot, "fraction", None)
        percent = None if fraction is None else int(fraction * 100)
        key = (getattr(snapshot, "phase", ""), getattr(snapshot, "headline", ""), percent)
        if key == self._progress_key:
            return
        self._progress_key = key
        self.emit(
            "progress",
            phase=getattr(snapshot, "phase", ""),
            headline=getattr(snapshot, "headline", ""),
            detail=getattr(snapshot, "detail", ""),
            fraction=fraction,
            busy=bool(getattr(snapshot, "busy", False)),
        )

    def narration(self, text: str) -> None:
        self.emit("narration", text=text)

    def desk(self, text: str) -> None:
        self.emit("desk", text=text)

    def delivered(self, event: Any) -> None:
        self.emit(
            "delivered",
            lines=list(event.lines()),
            images=[str(p) for p in getattr(event, "images", None) or ()],
            videos=[str(p) for p in getattr(event, "videos", None) or ()],
        )


def _job(record: Any) -> dict[str, Any]:
    """A job record as plain data. Reads by attribute so a record and a service row
    both work."""
    return {
        "id": str(getattr(record, "job_id", "") or ""),
        "name": getattr(record, "name", None),
        "status": getattr(record, "status", ""),
        "exit_code": getattr(record, "exit_code", None),
        "end_reason": getattr(record, "end_reason", None),
        "cmd": getattr(record, "cmd", ""),
        "cwd": getattr(record, "cwd", ""),
    }


def _entry(item: Any) -> dict[str, Any]:
    """One workspace listing row. The rendered line is kept too: it is what the other
    views show, and an agent comparing its own output with a person's wants both."""
    return {
        "name": str(getattr(item, "name", "") or ""),
        "is_dir": bool(getattr(item, "is_dir", False)),
        "size": getattr(item, "size", None),
        "line": item.line() if hasattr(item, "line") else str(item),
    }


# -- the way in ----------------------------------------------------------------


class JsonReader:
    """Newline-delimited JSON on stdin, read as the user's side of the conversation.

    Same four methods and one attribute as `LineReader` and `NullReader`, and the same
    distinction between the two kinds of nothing: `poll()` answers `NOTHING` when
    nothing has arrived yet and `None` on EOF. They are not interchangeable -- the
    drain between tool calls puts an EOF back for whoever is waiting at the prompt,
    and a reader that returned `None` for "nothing yet" would end every session at its
    first idle moment.

    Lines are read on a background thread so polling stays cheap, exactly as
    `LineReader` does. There is no paste window here: a JSON object is already a whole
    message, so there is nothing to guess about where one ends.
    """

    accepts_input = True

    def __init__(self, stream: Any = None) -> None:
        import queue

        self._queue: queue.Queue[str | None] = queue.Queue()
        self._empty = queue.Empty
        self._stream = stream
        self.ignored = 0
        """Lines that were not an object this reader understands. Counted rather than
        answered: the only stream it could complain on is the one it is not writing."""
        self._thread = threading.Thread(target=self._read, name="stdin-json", daemon=True)
        self._thread.start()

    def _read(self) -> None:
        stream = self._stream if self._stream is not None else sys.stdin
        while True:
            try:
                line = stream.readline()
            except (OSError, ValueError):
                self._queue.put(None)
                return
            if not line:
                self._queue.put(None)
                return
            text = message_text(line)
            if text is None:
                self.ignored += 1
                continue
            self._queue.put(text)

    def get(self, timeout: float | None = None) -> str | None:
        """Next message, or None on EOF. Raises `queue.Empty` on timeout."""
        return self._queue.get(timeout=timeout)

    def poll(self) -> Any:
        try:
            return self._queue.get_nowait()
        except self._empty:
            return NOTHING

    def putback(self, line: str | None) -> None:
        """Return something taken but not used -- including an EOF, which belongs to
        whoever waits at the prompt rather than to whoever polls between tool calls."""
        self._queue.put(line)

    def pending(self) -> bool:
        return not self._queue.empty()


def message_text(line: str) -> str | None:
    """The text of one input line, or None if it is not a message.

    Forgiving about where the text sits -- `text`, `content`, or a content list of
    blocks, which is the shape a model API uses -- and unforgiving about anything
    else. Guessing that a line of plain prose was meant as a message is how a
    malformed stream turns into a study nobody asked for.
    """
    stripped = line.strip()
    if not stripped:
        return None
    try:
        row = json.loads(stripped)
    except ValueError:
        return None
    if not isinstance(row, dict):
        return None
    kind = str(row.get("type") or "user")
    if kind not in ("user", "message", "say"):
        return None
    text = _text_of(row.get("text"))
    if text is None:
        text = _text_of(row.get("content"))
    if text is None:
        inner = row.get("message")
        if isinstance(inner, dict):
            text = _text_of(inner.get("content")) or _text_of(inner.get("text"))
    return text or None


def _text_of(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [
            str(block.get("text") or "")
            for block in value
            if isinstance(block, dict) and block.get("type") in (None, "text")
        ]
        joined = "".join(parts)
        return joined or None
    return None
