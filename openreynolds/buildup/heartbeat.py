"""A pidfile and a line per model turn -- the whole mechanism, and why it is this one.

The supervision used in the last round failed twice, and both failures are cheap to
repeat, so they are named in the code that replaces them.

**`until ! pgrep -f '<pattern>'` matched the watcher's own command line.** `pgrep`
always found at least itself, the condition was never true, and every watcher spun to
its timeout regardless of what it was watching. So **nothing here matches a process by
its command line.** The child writes its pid to a file; the supervisor calls `kill -0`
on that pid. A pid cannot match itself by accident.

**The pathology that mattered went unseen for three runs.** A run was producing replies
and executing zero cells; each burned its full clock and reported an ordinary budget
exhaustion. A step-based heartbeat cannot see it, because there are no steps. So the
beat is **per model turn**, whether or not a cell ran, and it carries the executed-step
count alongside the turn index -- which is exactly the pair whose divergence is the
failure.

Each beat is one JSON line, flushed and `fsync`-ed before the call returns. A run that
is killed keeps every beat it had written, and the beats are what say why it was killed.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

BEATS = "heartbeat.jsonl"
PIDFILE = "run.pid"


@dataclass(frozen=True)
class Beat:
    """One model turn, as the observer outside the run gets to see it."""

    turn: int
    steps: int
    """Cells executed so far. Flat while `turn` climbs is `no-progress`."""
    stop_reason: str = ""
    output_tokens: int = 0
    fenced: bool = False
    """Whether a python block was found in the reply. False every turn is a run that is
    talking and never acting."""
    text_chars: int = 0
    thinking_chars: int = 0
    phase: str = "turn"
    """`turn` for a model turn, or the long blocking thing the loop is about to do.

    The finish check runs `checkMesh` per region at up to 600 s each and the recovery
    replay is given 900 s, and no model turn happens during either -- so a beat-age alarm
    that knew only about turns would call a run wedged for being checked. Only `turn`
    beats count as progress; the rest are liveness."""
    expect_s: float = 0.0
    """How long that thing may legitimately take. The loop declares its own longest
    silence rather than the supervisor guessing it from a constant it cannot see."""
    """Both halves of the reply budget. `stop_reason=max_tokens` with no text is
    `starved` -- reasoning consumed the sentence that would have carried the work."""
    at: float = 0.0


class Heartbeat:
    """The run's side: a pidfile on `start`, a line on every turn.

    Held by the run, written to a directory the supervisor reads and neither writes into
    the other's file. The run never reads it back -- a run that could see its own
    supervision would be a run that could answer it."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / BEATS
        self.pidfile = self.dir / PIDFILE
        self.turns = 0

    def start(self, pid: int | None = None) -> int:
        """Claim the run. The pid written here is the only handle the supervisor gets."""
        mine = int(pid if pid is not None else os.getpid())
        self.pidfile.write_text(f"{mine}\n", encoding="utf-8")
        return mine

    def beat(self, *, steps: int, turn: int | None = None, stop_reason: str = "",
             output_tokens: int = 0, fenced: bool = False, text_chars: int = 0,
             thinking_chars: int = 0) -> Beat:
        """One turn, on disk before this returns.

        `turn` is the loop's own index when the loop offers it, and this side's count
        otherwise -- the two agree while every turn beats exactly once, and the loop's is
        the one worth believing if they ever stop agreeing."""
        self.turns += 1
        record = Beat(turn=int(turn if turn is not None else self.turns), steps=int(steps), stop_reason=stop_reason or "",
                      output_tokens=int(output_tokens), fenced=bool(fenced),
                      text_chars=int(text_chars), thinking_chars=int(thinking_chars),
                      at=time.time())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record

    def done(self) -> None:
        """Drop the pidfile, so a supervisor polling sees the run end rather than die."""
        try:
            self.pidfile.unlink()
        except OSError:
            pass


def read(run_dir: Path) -> list[Beat]:
    """Every beat written so far. A half-written last line is dropped, not raised on."""
    path = Path(run_dir) / BEATS
    out: list[Beat] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except ValueError:
            continue
        out.append(Beat(**{k: v for k, v in data.items() if k in Beat.__annotations__}))
    return out


def pid_of(run_dir: Path) -> int | None:
    """The run's pid, or None once it has finished or before it has started."""
    try:
        return int((Path(run_dir) / PIDFILE).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def alive(pid: int | None) -> bool:
    """`kill -0`, and nothing else. See this module's first paragraph."""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def attach(desk: Any, run_dir: Path) -> Heartbeat:
    """Wire a desk's per-turn hook to a heartbeat, and claim the run.

    One line at the top of whatever launches a run. It sets `on_turn`, which is the only
    thing the loop owes the observer: the loop reports, it does not consult."""
    pulse = Heartbeat(run_dir)
    pulse.start()
    desk.on_turn = _hook(pulse)
    return pulse


def _hook(pulse: Heartbeat) -> Callable[..., None]:
    """Keep the fields a beat has, and drop the rest.

    The loop's hook carries more than the beat does -- the whole reply text, the block
    types -- because a run record has to be able to explain a `starved` turn afterwards
    and the beat is deliberately small. Filtering here means a caller can add a field to
    the hook without every writer of it growing a keyword."""
    def on_turn(**fields: Any) -> None:
        pulse.beat(**{name: value for name, value in fields.items()
                      if name in Beat.__annotations__})
    return on_turn
