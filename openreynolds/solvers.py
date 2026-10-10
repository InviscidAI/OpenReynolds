"""Which solver a study runs: OpenFOAM on a CPU workspace, or Felix on cloud GPUs.

A study's solver is fixed once and holds for the whole study, resumes included. It is
either given (`--solver openfoam|felix`) or left to the agent (`--solver auto`, the
default), which then makes it its first decision through the `choose_solver` tool; no
workspace is started until it has, because a workspace has one kind and a study binds
only to a workspace of its own kind.

A study recorded before solvers existed has no solver on its record and is an
OpenFOAM study. One that was left to the agent and never chosen is recorded as
`auto`, and is still the agent's to choose when it is resumed.
"""

from __future__ import annotations

OPENFOAM = "openfoam"
FELIX = "felix"
AUTO = "auto"

SOLVERS = (OPENFOAM, FELIX)
CHOICES = (AUTO, OPENFOAM, FELIX)


def normalize(value: str | None) -> str | None:
    """`auto`, `openfoam` or `felix` from any spelling of one, else None."""
    text = (value or "").strip().lower()
    return text if text in CHOICES else None


def chosen(value: str | None) -> bool:
    """Whether `value` names a solver, rather than the choice still to be made."""
    return normalize(value) in SOLVERS
