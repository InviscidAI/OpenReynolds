"""One process, outside the run, and the only thing that observes it.

It owns liveness and the three alarms, the silent-failure probes of §2, the
contamination check of §3, and the grading of each case against what the record holds.
Nothing else looks at the run, which is what keeps the measurement out of the thing
being measured.

The boundary is the point, not an implementation detail. A check that lives in the
harness is one refactor away from living in the desk's path, and then the thing being
measured has quietly become the thing doing the measuring. A check that lives here
**cannot** reach the agent by accident: different process, reads the case off disk, no
channel into the conversation. So this module imports the toolbox freely -- that is
allowed here and nowhere the run can see -- and writes only into the run directory, never
into the workspace.

Two things it will not do. It will not match a process by its command line: the pid comes
from the pidfile the run wrote (`heartbeat.py` says what that cost last time). And it
will not let a run end unclassified -- `stopped` is always one of `record.TERMINAL`, and
a run that ends outside the set is a bug here rather than a result.
"""

from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import alarms, heartbeat, isolation, probes, record

WATCHER = "watcher.pid"
"""The watching process's own pid, so a run cannot acquire two of them.

Two watchers on one run is not a harmless duplicate: both evaluate the same alarms, both
may signal the same child, and both write `watch.json` -- so the record's account of how a
run ended depends on which one finished last. It became reachable the moment a sweep driver
started launching watches, because the subagent that drives a single run by hand types the
same command."""

WATCH = "watch.json"
"""Where the watching half leaves what it saw, for the grading half to read.

Found by running the two for real: `watch` and `observe` are separate commands in separate
invocations, and `first_mesh_step` is knowable only to the first. Without this the number
that the whole phase compares arms on reached the record as 0 unless a human copied it
across by hand."""

POLL_S = 5.0
GRACE_S = 10.0
"""Between the SIGTERM that asks a wedged run to stop and the SIGKILL that makes it.

Long enough for the run's own `finally` to write what it had -- the partial record is
usually the whole diagnosis -- and short enough that a wedged process does not go on
holding the workspace."""


@dataclass
class Watch:
    """What the watching half saw. `alarm` is None when the run ended on its own terms."""

    alarm: alarms.Alarm | None = None
    turns: int = 0
    steps: int = 0
    seconds: float = 0.0
    killed: bool = False
    first_mesh_step: int = 0
    """The step count when a polyMesh first appeared, or 0 for never.

    Seen from outside while the run goes, because it cannot be recovered afterwards: the
    case directory at the end says only that a mesh exists, and *when* the desk first got
    one is the number the arms are being compared on. The run is not asked and is not
    told -- the supervisor is polling the pidfile anyway, and looks at the directory on
    the same lap."""
    first_mesh_turn: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"alarm": self.alarm.as_dict() if self.alarm else None,
                "turns": self.turns, "steps": self.steps,
                "seconds": round(self.seconds, 1), "killed": self.killed,
                "first_mesh_step": self.first_mesh_step,
                "first_mesh_turn": self.first_mesh_turn}


def meshed(case_dir: Path | None) -> bool:
    """Is there a polyMesh here yet -- single-region or any region of a conjugate case.

    The same question `cad/check.py:mesh_regions` asks the workspace, asked of the
    filesystem instead, because the supervisor has the case on disk and asking through the
    run's own channel would be observing it from inside."""
    if not case_dir:
        return False
    case = Path(case_dir)
    if (case / "constant" / "polyMesh" / "points").exists():
        return True
    return any(case.glob("constant/*/polyMesh/points"))


class Watched(Exception):
    """Somebody is already watching this run. Raised rather than joining in."""


class Supervisor:
    """The observer. Construct it with the run directory; it needs nothing else."""

    def __init__(self, run_dir: Path, *, case_dir: Path | None = None,
                 stale_s: float = alarms.HEARTBEAT_STALE_S,
                 k: int = alarms.K, poll_s: float = POLL_S,
                 now: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep):
        self.dir = Path(run_dir)
        self.case = Path(case_dir) if case_dir else None
        self.stale_s = float(stale_s)
        self.k = int(k)
        self.poll_s = float(poll_s)
        self.now = now
        self.sleep = sleep

    # -- liveness ---------------------------------------------------------------

    def claim(self) -> None:
        """Take the run's single watch, or refuse because something else holds it."""
        existing = self.dir / WATCHER
        try:
            held = int(existing.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            held = 0
        if held and held != os.getpid() and heartbeat.alive(held):
            raise Watched(
                f"{self.dir} is already being watched by pid {held}. One run has one "
                "observer: a second would evaluate the same alarms, signal the same child "
                "and overwrite the same watch.json.")
        self.dir.mkdir(parents=True, exist_ok=True)
        existing.write_text(f"{os.getpid()}\n", encoding="utf-8")

    def release(self) -> None:
        try:
            (self.dir / WATCHER).unlink()
        except OSError:
            pass

    def watch(self, *, deadline_s: float | None = None) -> Watch:
        """Poll until the run ends or an alarm fires, and kill it if one does."""
        self.claim()
        try:
            return self._watch(deadline_s=deadline_s)
        finally:
            self.release()

    def _steps_on_disk(self) -> int:
        """How many cells the run has executed, read without the heartbeat.

        The runner appends one `# -- cell N` header per executed cell as it goes, so this
        is a second and independent answer to "is anything happening". It exists to tell
        a desk that has hung apart from a watcher that has gone blind: with no beats at
        all, a zero here is `wedged` and a positive one is `unobserved`.

        Deliberately not `record.json`'s `n_steps`: the runner writes that through the
        same `on_turn`/`on_step` path whose failure this is meant to catch, and a check
        that shares a channel with the thing it checks is not a second opinion.
        """
        log = self.dir / "cells.log"
        try:
            return sum(1 for line in log.read_text(encoding="utf-8", errors="replace")
                       .splitlines() if line.startswith("# -- cell "))
        except OSError:
            return 0

    def _watch(self, *, deadline_s: float | None = None) -> Watch:
        started = self.now()
        first_step = first_turn = 0
        while True:
            beats = heartbeat.read(self.dir)
            pid = heartbeat.pid_of(self.dir)
            running = heartbeat.alive(pid)
            # Only while the run is live. A `watch` started after a run has ended sees a
            # finished case directory on its first lap and would record "the mesh appeared
            # at the last step", which is a number nobody measured.
            if not first_turn and beats and running and meshed(self.case):
                first_step, first_turn = beats[-1].steps, beats[-1].turn
            alarm = alarms.evaluate(beats, now=self.now(), started_at=started,
                                    running=running, stale_s=self.stale_s, k=self.k,
                                    steps_seen=self._steps_on_disk())
            seen = Watch(alarm=alarm, turns=beats[-1].turn if beats else 0,
                         steps=beats[-1].steps if beats else 0,
                         seconds=self.now() - started,
                         first_mesh_step=first_step, first_mesh_turn=first_turn)
            if alarm is not None:
                self._leave(seen)
                # The clock is not the diagnosis. A run that is replying and running
                # nothing is aborted with the reason attached rather than left to
                # exhaust its budget and report `time`, which is what happened three
                # times in a row last round.
                seen.killed = self.stop(pid) if running else False
                return seen
            if not running and pid is None:
                self._leave(seen)
                return seen
            if not running and beats:
                self._leave(seen)
                return seen
            if deadline_s is not None and seen.seconds > deadline_s:
                seen.alarm = alarms.Alarm("wedged",
                                          f"the supervisor's own deadline of "
                                          f"{deadline_s:.0f}s passed", seen.turns)
                seen.killed = self.stop(pid) if running else False
                self._leave(seen)
                return seen
            self.sleep(self.poll_s)

    def _leave(self, seen: "Watch") -> None:
        """Write what this watch saw, so `observe` does not have to be told it."""
        try:
            (self.dir / WATCH).write_text(json.dumps(seen.as_dict(), indent=2),
                                          encoding="utf-8")
        except OSError:
            pass

    def group(self) -> int:
        """The run's own process group, if it took one. 0 otherwise.

        Only a group the run declared is signalled. Deriving one with `getpgid` would
        sooner or later name the group this supervisor is in -- under a sweep both are
        children of the same driver -- and killing that is killing the sweep."""
        try:
            claimed = int((self.dir / "run.pgid").read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return 0
        return claimed if claimed and claimed != os.getpgid(0) else 0

    def stop(self, pid: int | None) -> bool:
        """Ask, then insist -- the whole tree where there is one.

        A mesher is a grandchild of the run: the desk runs cells in a kernel process and a
        cell starts `snappyHexMesh` from there. Signalling the runner alone leaves a
        runaway mesh running on the machine the next case is about to use."""
        if not heartbeat.alive(pid):
            return False
        pgid = self.group()
        try:
            if pgid:
                os.killpg(pgid, signal.SIGTERM)
            else:
                os.kill(int(pid), signal.SIGTERM)
        except OSError:
            return False
        waited = 0.0
        while waited < GRACE_S:
            if not heartbeat.alive(pid):
                return True
            self.sleep(min(1.0, GRACE_S))
            waited += 1.0
        try:
            if pgid:
                os.killpg(pgid, signal.SIGKILL)
            else:
                os.kill(int(pid), signal.SIGKILL)
        except OSError:
            pass
        return True

    # -- after the run ----------------------------------------------------------

    def observe(self, *, case_dir: Path | None = None, spec: dict[str, Any] | None = None,
                watch: Watch | None = None, expected: list[str] | None = None,
                given: list[str] | None = None) -> dict[str, Any]:
        """Grade the run: contaminated or not, what the probes found, how it ended.

        Everything here reads the run directory and the case as they stand on disk. It
        writes the record and nothing else, and in particular it writes nothing the run
        could read even if the run were still going."""
        data = record.load(self.dir)
        beats = heartbeat.read(self.dir)
        watch = watch if watch is not None else self.seen()

        # The arm's own declaration when the caller does not override it: a run
        # re-graded from disk next week has to reach the same verdict as this one.
        handed = given if given is not None else list(data.get("given") or [])
        dirt = isolation.scan_run(self.dir, expected=expected or [], given=handed)
        found = probes.run_all(Path(case_dir), dict(spec or {})) if case_dir else []

        data["beats"] = [vars(beat) for beat in beats]
        data["n_turns"] = beats[-1].turn if beats else int(data.get("n_turns") or 0)
        data["n_steps"] = beats[-1].steps if beats else int(data.get("n_steps") or 0)
        data["contamination"] = dirt.as_dict()
        data["contaminated"] = dirt.contaminated
        data["probes"] = [result.as_dict() for result in found]
        if watch is not None:
            data["watch"] = watch.as_dict()
            if watch.first_mesh_step or not data.get("first_mesh_step"):
                data["first_mesh_step"] = watch.first_mesh_step
        data["stopped"] = record.classify(self.ending(data, watch))
        data["why"] = self._why(data, watch)
        data["completed"] = True
        data["ended_at"] = data.get("ended_at") or time.strftime("%Y-%m-%dT%H:%M:%S")
        record.save(self.dir, data)
        return data

    def seen(self) -> Watch | None:
        """What a `watch` in an earlier invocation left behind, if there was one."""
        try:
            data = json.loads((self.dir / WATCH).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        raised = data.get("alarm") or None
        return Watch(
            alarm=alarms.Alarm(raised["alarm"], raised["why"], raised.get("turn", 0))
            if raised else None,
            turns=int(data.get("turns") or 0), steps=int(data.get("steps") or 0),
            seconds=float(data.get("seconds") or 0.0), killed=bool(data.get("killed")),
            first_mesh_step=int(data.get("first_mesh_step") or 0),
            first_mesh_turn=int(data.get("first_mesh_turn") or 0))

    def ending(self, data: dict[str, Any], watch: Watch | None) -> str:
        """The terminal state, in the order that decides which one wins.

        Contamination first, because a contaminated run is not a measurement of anything
        and its own ending is beside the point. Then the alarm, because an aborted run's
        `stopped` field says whatever it had got to. Then the run's own word.

        **The run's own word means all of it.** This tested `stopped in ("steps", "time",
        "provider")` and fell through to `done` for everything else, which silently
        rewrote `refused` -- the one ending only the runner can know, because it is the
        desk declining rather than anything observable from outside. `observe` runs after
        the runner has scored and saves the whole record, so the rewrite landed on disk
        underneath a `passed` the runner had computed from the real value: T6 and T25 of
        the sol corpus each read `stopped: done`, `expects: refused`, `passed: true`,
        which is self-contradictory and re-grades to a failure. The record is supposed to
        be re-gradeable from disk alone.

        So the observer now fills `stopped` only where the runner left it empty. It is
        still the observer's job to override for contamination and for an alarm, because
        those are things the run cannot see about itself."""
        if data.get("contaminated"):
            return "contaminated"
        if watch is not None and watch.alarm is not None:
            return watch.alarm.name
        stopped = str(data.get("stopped") or "")
        if stopped in record.TERMINAL:
            return stopped
        return "done"

    def _why(self, data: dict[str, Any], watch: Watch | None) -> str:
        if data.get("contaminated"):
            return "; ".join(isolation.Contamination(
                True, [isolation.Hit(**hit) for hit in
                       data.get("contamination", {}).get("hits", [])]).lines()[:5])
        if watch is not None and watch.alarm is not None:
            return watch.alarm.why
        return str(data.get("why") or "")


def preflight(workspace: Path) -> dict[str, Any]:
    """§3's before-the-run half, here because the supervisor is what asserts it.

    It runs in this process rather than the run's for the same reason everything else
    here does: a check the run performs on itself is a check the run can be wrong
    about."""
    return isolation.preflight(Path(workspace))
