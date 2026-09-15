"""How much the person wants to be consulted: full auto, ask before compute, structured.

The free-will contract (docs/design.md section 1) holds in the default mode, `auto`,
exactly as it always has: no approvals, no enforced ordering, no tool call ever held.

The other two exist because a person may *choose* to be consulted, and that choice is
theirs to make in the same way `/btw` is theirs (commands.py): the user's own words
about how they want to be heard. When they have chosen, the harness gates exactly what
they asked to have gated and nothing else.

- `partial` puts every `job_start` and every `mesh` call to the person before it runs.
  Those are the calls that spend compute; everything else runs freely.
- `structured` offers a `checkpoint` tool and holds `job_start` and `mesh` until a
  checkpoint (a plan) has been approved since the session entered the mode.

What the modes gate is decided here, in one place, so the loop, the briefing, the help
text and the docs cannot drift apart on it.
"""

from __future__ import annotations

from .toolbox.study_state import PHASES

AUTO = "auto"
PARTIAL = "partial"
STRUCTURED = "structured"

MODES: tuple[str, ...] = (AUTO, PARTIAL, STRUCTURED)

LABELS = {
    AUTO: "Full auto",
    PARTIAL: "Ask before compute",
    STRUCTURED: "Structured",
}

DESCRIPTIONS = {
    AUTO: "The agent decides everything and nothing waits for you.",
    PARTIAL: "Every job start and every mesh build is put to you before it runs.",
    STRUCTURED: "The study goes in stages: you approve a plan, then each stage at a checkpoint.",
}

ALIASES = {
    "full": AUTO,
    "full-auto": AUTO,
    "fullauto": AUTO,
    "free": AUTO,
    "ask": PARTIAL,
    "approve": PARTIAL,
    "approvals": PARTIAL,
    "partial-auto": PARTIAL,
    "plan": STRUCTURED,
    "staged": STRUCTURED,
    "stages": STRUCTURED,
    "guided": STRUCTURED,
}

STAGES: tuple[str, ...] = tuple(PHASES)
"""The stage names structured mode speaks in: the guided pipeline's own phases
(`toolbox/study_state.py`), so the briefing, the checkpoint tool and the toolbox's
`phases.json` all use one vocabulary."""

COMPUTE_TOOLS = frozenset({"job_start", "mesh"})
"""The calls that spend compute, and so the only ones a non-auto mode gates."""

CHECKPOINT = "checkpoint"


def normalize(name: str | None) -> str | None:
    """The canonical mode for a name or an alias, or None when it is not one."""
    if name is None:
        return None
    key = str(name).strip().lower().replace("_", "-").replace(" ", "-")
    if key in MODES:
        return key
    return ALIASES.get(key)


def choices() -> list[str]:
    """Every spelling a command line accepts: the modes, then the aliases."""
    return list(MODES) + sorted(ALIASES)


def label(mode: str) -> str:
    return LABELS.get(normalize(mode) or AUTO, LABELS[AUTO])


def gated(mode: str, tool: str) -> bool:
    """Whether this mode has anything to say about this tool call."""
    return mode in (PARTIAL, STRUCTURED) and tool in COMPUTE_TOOLS


def briefing(mode: str) -> str:
    """The one sentence the session's briefing carries about the mode.

    In the person's terms, like the standing note: what they chose and what the
    harness does about it. Empty in auto, so an auto briefing is byte-identical to
    one written before modes existed."""
    if mode == PARTIAL:
        return (
            "The person chose to be asked before compute is spent: each job_start and "
            "mesh call is put to them and runs once they approve it, and a declined "
            "call comes back with their reason when they give one."
        )
    if mode == STRUCTURED:
        return (
            "The person chose structured mode for this study. They want to agree a plan "
            "with you at a checkpoint and to see a checkpoint after each stage ("
            + ", ".join(STAGES)
            + "); the checkpoint tool puts a summary and what comes next in front of "
            "them and waits for their answer, and job_start and mesh calls are held "
            "until they have approved a plan."
        )
    return ""


def switched(mode: str) -> str:
    """What the model is told, in the harness's voice, when the person switches."""
    if mode == PARTIAL:
        return (
            "The person switched this session to ask-before-compute mode. From the next "
            "tool call, each job_start and mesh call is put to them before it runs; "
            "every other tool runs as before."
        )
    if mode == STRUCTURED:
        return (
            "The person switched this session to structured mode. They want to agree a "
            "plan at a checkpoint and to see a checkpoint after each stage ("
            + ", ".join(STAGES)
            + "). The checkpoint tool is offered from the next request, and job_start "
            "and mesh calls are held until a checkpoint has been approved."
        )
    return (
        "The person switched this session to full auto mode. No tool call is put to "
        "them for approval from here on."
    )


def held(tool: str) -> str:
    """The tool result for a call structured mode holds. Facts, nothing run."""
    return (
        f"This {tool} call was held and did not run. The person chose structured mode, "
        "where job_start and mesh run once a plan has been approved at a checkpoint, and "
        "no checkpoint has been approved since the mode began."
    )


def status_lines(current: str) -> list[str]:
    """The answer to `/mode` with nothing after it."""
    lines = [f"mode: {LABELS.get(current, current)} ({current})"]
    for mode in MODES:
        mark = "*" if mode == current else " "
        lines.append(f" {mark} {mode:<11} {LABELS[mode]}: {DESCRIPTIONS[mode]}")
    lines.append("switch with /mode auto, /mode partial or /mode structured")
    return lines
