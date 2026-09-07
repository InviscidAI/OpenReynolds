"""The mesh desk: geometry and meshing as their own agent, with their own rules.

Three files and no machinery. `brief.py` is what it is told, `agent.py` is the loop
that gives it one bash block a step on the machine with OpenFOAM, `check.py` is the
finish line it does not get to declare for itself. The instrument it looks through --
`toolbox/mesh_look.py` -- lives with the other toolbox scripts, because a person and
the main agent run it the same way this one does.

Why it is separate from the agent that calls it: meshing is a closed task with a
checkable answer, and the main agent's contract is written for open-ended work where
nobody knows the right move in advance. Mixing them cost 66-turn sessions that produced
the wrong shape. This desk is allowed to be told what to do.
"""

from __future__ import annotations

from .agent import MAX_SECONDS, MAX_STEPS, MeshResult, Mesher, Step, parse_action
from .brief import MESH_DONE, MESHER_SYSTEM, system_prompt, task_message
from .check import Check, look_command, read, verify

__all__ = [
    "Check",
    "MAX_SECONDS",
    "MAX_STEPS",
    "MESHER_SYSTEM",
    "MESH_DONE",
    "MeshResult",
    "Mesher",
    "Step",
    "look_command",
    "parse_action",
    "read",
    "system_prompt",
    "task_message",
    "verify",
]
