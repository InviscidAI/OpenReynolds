"""The CAD desk: CAD and meshing as one job, with its own rules and its own finish line.

Five files and no machinery. `brief.py` is what it is told, `agent.py` is the loop that
gives it one python cell a step in a kernel on the machine with OpenFOAM on it,
`cells.py` decides which of those cells belong in the script the run leaves behind, and
`check.py` is the finish line it does not get to declare for itself. The instruments it
looks through -- `toolbox/mesh_look.py`, `cad_audit.py`, `domain_probe.py` -- live with
the other toolbox scripts, because a person and the main agent run them the same way
this one does.

Why it is separate from the agent that calls it: preparing a shape and meshing it is a
closed task with a checkable answer, and the main agent's contract is written for
open-ended work where nobody knows the right move in advance. Mixing them cost 66-turn
sessions that produced the wrong shape. This desk is allowed to be told what to do.

What is re-exported here is what the harness outside this package uses: `cli.py`
constructs a `CadDesk` and renders its `Step`s, and `tools.py` renders its `CadResult`.
Nothing else above the package reaches inside it.
"""

from __future__ import annotations

from .agent import (
    MAX_SECONDS,
    MAX_STEPS,
    STEP_TIMEOUT_S,
    CadDesk,
    CadResult,
    Step,
    parse_action,
)
from .brief import CAD_DONE, CAD_SYSTEM, remark_message, system_prompt, task_message
from .cells import Cell, CellLog
from .core import CORE_SYSTEM, CoreDesk
from .check import Check, Finding, look_command, mesh_regions, read, verify

__all__ = [
    "CAD_DONE",
    "CAD_SYSTEM",
    "CORE_SYSTEM",
    "CadDesk",
    "CoreDesk",
    "CadResult",
    "Cell",
    "CellLog",
    "Check",
    "Finding",
    "MAX_SECONDS",
    "MAX_STEPS",
    "STEP_TIMEOUT_S",
    "Step",
    "look_command",
    "mesh_regions",
    "parse_action",
    "read",
    "remark_message",
    "system_prompt",
    "task_message",
    "verify",
]
