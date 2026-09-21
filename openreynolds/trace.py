"""A trace of what a study spent, written only when asked for.

The harness already says what happened; it does not say what it cost. A tool call
under ten seconds is never timed (`tools.SLOW_COMMAND_S`), a turn's latency is
nowhere, and `llm.anthropic_api._context_tokens` sums input, cache reads, cache
writes and output into a single number -- so a cache hit and a cache miss reach the
bar looking identical. None of that is visible from a transcript afterwards.

This writes one JSON object per event, and does nothing at all until somebody asks
for it: `OPENREYNOLDS_TRACE=<path>` in the environment, or `to(sink)` at runtime,
which is how `--output-format stream-json` puts the cost events on the same stream as
everything else it says.

The path used to be read once at import, which meant the only way to turn tracing on
was to have set an environment variable before the process started -- a flag could
not reach it. `on` is still a plain module-level boolean, because three files check
it on every call (`tools.py`, `llm/anthropic_api.py`, `mirror.py`) and the whole point
of the check is that it costs nothing when nothing is being recorded.

Every event carries `v` (schema), `kind`, `type`, `at` (monotonic seconds since this
module loaded, or since the origin `to()` was given), `ts` (wall clock) and `study`. `kind` and the fields each kind already
carried are unchanged: `turn`, `tool` and `mirror` rows written by an older build and a
newer one are the same rows with more identity on them. The identity is the point --
a trace file from a machine running three studies could not say which study a turn
belonged to, and a row with no wall clock could not be lined up against anything else
that happened that afternoon.

It is deliberately outside the session's own machinery: it must not change what the
model sees, what the loop does, or whether a study survives, so every call is wrapped
and a failure here is dropped rather than raised.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

SCHEMA = 1
"""Bumped when a field is renamed or removed. Adding one is not that."""

_lock = threading.Lock()
_IMPORT_T0 = time.monotonic()
"""When this module loaded. The origin `at` is measured from when nobody else's
clock has been offered."""
_t0 = _IMPORT_T0
_path = os.environ.get("OPENREYNOLDS_TRACE") or ""
_sink: Any = None
_study = ""

on = bool(_path)
"""Whether anything is being recorded. Checked by callers to skip the work entirely."""


def begin(study_id: str) -> None:
    """Name the study every later event belongs to.

    Called once a session knows its own id. Harmless when nothing is being recorded,
    so the caller does not have to ask first.
    """
    global _study
    _study = str(study_id or "")


def to(sink: Any, origin: float | None = None) -> None:
    """Send events to a writable object from here on, and turn recording on.

    `sink` needs `write` and may have `flush`. A flush happens after every event: a
    consumer reading the same stream live is the reason this exists, and a trace that
    appears when the operating system feels like it is a trace of nothing.

    `origin` is the monotonic reading that `at` is measured from, and giving it is
    what keeps one stream on one clock. `JsonView` starts counting when it is built --
    after the config load and after the workspace acquire, which against the real
    service is a container cold start -- while this module started counting at import.
    Both write a field called `at`, and a captured run showed the result: a `cost` row
    at 4.125 sitting between two view rows at 2.938. An agent sorting the stream by
    `at` reordered it, and one differencing a `cost` row against its neighbours got a
    tool duration inflated by the whole startup.
    """
    global _sink, on, _t0
    _sink = sink
    if origin is not None:
        _t0 = float(origin)
    on = True


def enable(path: str) -> None:
    """Append events to a file from here on, and turn recording on."""
    global _path, _sink, on
    _path, _sink = str(path), None
    on = bool(_path)


def off() -> None:
    """Drop the runtime sink and go back to whatever the environment asked for.

    Not simply "stop": a session that streamed its cost events must not switch off a
    trace file the person set `OPENREYNOLDS_TRACE` for, and a test that forgot this
    would leave the next test writing into a closed stream.
    """
    global _sink, on, _t0
    _sink = None
    _t0 = _IMPORT_T0
    """The borrowed clock goes back with the sink it came with: a file trace the
    environment asked for is a trace of the process, and the next session in this
    process is not the one whose view lent its origin."""
    on = bool(_path)


def event(kind: str, **fields: Any) -> None:
    """Append one event. Never raises, never blocks on anything but the sink."""
    if not on:
        return
    try:
        row = {
            "v": SCHEMA,
            "type": "cost",
            "kind": kind,
            "at": round(time.monotonic() - _t0, 3),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "study": _study,
            **fields,
        }
        line = json.dumps(row, default=str)
        with _lock:
            sink = _sink
            if sink is not None:
                sink.write(line + "\n")
                flush = getattr(sink, "flush", None)
                if flush is not None:
                    flush()
            elif _path:
                with open(_path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
    except Exception:
        # A trace that breaks the study it is measuring is worse than no trace.
        pass


def usage(response: Any) -> dict[str, int]:
    """The token counts split apart, rather than summed into one number."""
    u = getattr(response, "usage", None)
    if u is None:
        return {}
    return {
        "input": getattr(u, "input_tokens", 0) or 0,
        "output": getattr(u, "output_tokens", 0) or 0,
        "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }
