"""The desk the corpus measures, which as of 2026-09-18 is the desk that ships.

It moved to `openreynolds/cad/core.py`. This module re-exports it, and the move is the
point rather than tidying: `buildup` is the measurement harness -- `probes` imports four
toolbox scripts into the calling process, `isolation` and `supervise` exist only to watch
a run from outside it -- and a product whose entry point imports all of that to construct
its desk has put the harness on the user's machine.

The direction it reversed: `buildup` imported `cad` for the loop, and `cad` named
`buildup` only in comments. Now `cad` owns the desk and `buildup` imports it to measure,
which is the direction that was always meant -- "nothing in this package is synced to a
workspace, imported by the desk, or named in its brief".
"""

from __future__ import annotations

from ..cad.core import (  # noqa: F401
    CHECKMESH_TIMEOUT_S, CORE_NUDGE, CORE_SYSTEM, GATE_TIMEOUT_S, REFERENCE_DIR,
    REFERENCE_FILES, CoreDesk, brief, check_command, read_checkmesh, verify,
)

__all__ = ["CHECKMESH_TIMEOUT_S", "CORE_NUDGE", "CORE_SYSTEM", "GATE_TIMEOUT_S",
           "REFERENCE_DIR", "REFERENCE_FILES", "CoreDesk", "brief", "check_command",
           "read_checkmesh", "verify"]
