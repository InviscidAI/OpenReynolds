"""What a run leaves behind, shaped by the two mistakes that cost the last round time.

**Flat scalars at the top level.** Every number a report reads -- `n_turns`, `n_steps`,
`first_mesh_step`, `mesh_exists`, `checkmesh_ok`, `seconds`, `usd`, `stopped`,
`contaminated` -- sits directly on the record, and the step trace lives under `trace`.
Last round the trace was stored under `steps`, so a naive reader asking for the step
count dumped the whole trace instead, and every quick look at a record was a paging
exercise.

**Every raw reply is persisted, incrementally.** Full text, thinking length,
`stop_reason`, per-turn tokens, block types, whether a fence closed. Without them the
`starved` failure was undiagnosable across three runs; with them it took one look. They
are appended to `replies.jsonl` as they arrive rather than written at the end, because
the runs that most need explaining are the ones that get killed.

**A run that ends without a classification is a bug here, not a result.** `TERMINAL`
is the closed set, and `classify` refuses anything outside it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RECORD = "record.json"
REPLIES = "replies.jsonl"

TERMINAL = ("done", "refused", "steps", "time", "provider", "wedged", "no-progress",
            "starved", "contaminated", "unobserved")
"""Every way a run is allowed to end. `done` and `refused` are the two that are results.

`refused` is the desk saying the request cannot be answered correctly, and why. It was
missing, and its absence is not a gap in the vocabulary so much as a gap in what the
harness could measure: a desk that correctly declined had to land in `steps`, which is
indistinguishable from failing, and a desk that guessed landed in `done`. T6 of the first
baseline guessed a length unit off a STEP that declares none, shipped a mesh `checkMesh`
passed, and was recorded as the corpus's sixth success."""


@dataclass
class Record:
    """One run, flat where a reader looks and nested where a reader does not."""

    run_id: str = ""
    case: str = ""
    arm: str = "core"
    model: str = ""
    workspace: str = ""
    case_dir: str = ""
    started_at: str = ""
    ended_at: str = ""
    seconds: float = 0.0
    n_turns: int = 0
    n_steps: int = 0
    first_mesh_step: int = 0
    """The step at which a polyMesh first existed. 0 means never -- the measure the
    whole build-up phase is comparing arms on."""
    mesh_exists: bool = False
    checkmesh_ok: bool = False
    usd: float = 0.0
    given: list[str] = field(default_factory=list)
    """Filenames this arm was handed on purpose, and which are therefore not evidence of
    contamination when the run is seen touching them.

    The arm declares it here rather than the observer being told at the command line,
    because the observer grades a run directory that may be graded again next week by
    somebody who does not remember which arm wrote it. A run that says what it was given
    can be re-graded correctly from disk alone."""
    tokens: dict[str, int] = field(default_factory=dict)
    stopped: str = ""
    """One of `TERMINAL`, always."""
    expects: str = "done"
    """What this case counts as a pass -- `done` for nearly all of them, `refused` for a
    case whose whole point is that the desk should decline. Read off the prompt, so the
    criterion lives with the case rather than in the thing scoring it."""
    passed: bool = False
    """Whether the run did what its case asked, which is not the same as `checkmesh_ok`.

    For an ordinary case the two agree. For a refusal case they are opposites: T6 ended
    `done` with `checkmesh_ok: true` and failed, and a correct T6 run will end `refused`
    with no mesh at all and pass. Anything scoring a sweep should read this."""
    why: str = ""
    contaminated: bool = False
    completed: bool = False

    trace: list[dict[str, Any]] = field(default_factory=list)
    """The step trace. Under its own key so that asking for a scalar never returns it."""
    beats: list[dict[str, Any]] = field(default_factory=list)
    contamination: dict[str, Any] = field(default_factory=dict)
    probes: list[dict[str, Any]] = field(default_factory=list)
    """The supervisor's silent-failure probes, graded in here and nowhere the run can
    see. §2: detection is separated from delivery, and this is the delivery side."""
    properties: list[dict[str, Any]] = field(default_factory=list)
    """The case's named properties and whether the run measured and printed each. A
    property nobody measured is a property nobody built."""
    declares: list[dict[str, Any]] = field(default_factory=list)
    """Every `declare_complete` the run made, with what the advisory gates said and what
    the desk waived.

    Separate from `probes`, which is the supervisor's independent reading of the same
    case. Holding both is the point: a check the desk waived as a deliberate baffle, and
    which the supervisor then reads as a leak, is a finding that neither column could
    produce alone."""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify(stopped: str) -> str:
    """Hold a terminal state to the closed set, loudly."""
    if stopped not in TERMINAL:
        raise ValueError(
            f"{stopped!r} is not a terminal state. One of {', '.join(TERMINAL)} -- a run "
            "that ends outside the set is a bug in the supervisor, not a result.")
    return stopped


def save(run_dir: Path, record: Record | dict[str, Any]) -> Path:
    """Write the record atomically, so a reader never catches it half-written.

    Temp file and rename: the supervisor polls this file while the run is writing it,
    and a partial JSON read looks exactly like a crashed run."""
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RECORD
    payload = record.as_dict() if isinstance(record, Record) else dict(record)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temp, path)
    return path


def load(run_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((Path(run_dir) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def reply(run_dir: Path, **fields: Any) -> None:
    """Append one raw reply. Called per turn, flushed per turn, for the same reason."""
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fields.setdefault("at", time.time())
    with (directory / REPLIES).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(fields, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def replies(run_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        text = (Path(run_dir) / REPLIES).read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out
