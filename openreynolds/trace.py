"""A trace of what a study spent, written only when asked for.

The harness already says what happened; it does not say what it cost. A tool call
under ten seconds is never timed (`tools.SLOW_COMMAND_S`), a turn's latency is
nowhere, and `llm.anthropic_api._context_tokens` sums input, cache reads, cache
writes and output into a single number -- so a cache hit and a cache miss reach the
bar looking identical. None of that is visible from a transcript afterwards.

This writes one JSON object per event to `OPENREYNOLDS_TRACE`, and does nothing at
all when that is unset. It is deliberately outside the session's own machinery: it
must not change what the model sees, what the loop does, or whether a study
survives, so every call is wrapped and a failure here is dropped rather than raised.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

_PATH = os.environ.get("OPENREYNOLDS_TRACE") or ""
_lock = threading.Lock()
_t0 = time.monotonic()

on = bool(_PATH)
"""Whether anything is being recorded. Checked by callers to skip the work entirely."""


def event(kind: str, **fields: Any) -> None:
    """Append one event. Never raises, never blocks on anything but the file."""
    if not on:
        return
    try:
        row = {"kind": kind, "at": round(time.monotonic() - _t0, 3), **fields}
        line = json.dumps(row, default=str)
        with _lock, open(_PATH, "a") as fh:
            fh.write(line + "\n")
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
