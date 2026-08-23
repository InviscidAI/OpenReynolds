"""Stopping work, and making sure it actually stopped.

Killing a job signals its process group. A solver launched through `mpirun` puts its
ranks in a different group, so the wrapper dies, the service records the job as killed,
and eight cores keep running on a machine nobody is watching. That happened, and the
first anyone knew of it was the bill.

So stopping is a loop with a check in it: signal, look, escalate, look again, and report
what is still there rather than what was requested.
"""

from __future__ import annotations

import shlex
import time
from dataclasses import dataclass, field

from .backend.base import Backend, BackendError
from .store import Store

SOLVERS = ("simpleFoam", "pimpleFoam", "interFoam", "icoFoam", "buoyantSimpleFoam",
           "potentialFoam", "snappyHexMesh", "blockMesh", "mpirun", "reconstructPar")
"""Processes worth naming when they outlive the job that started them."""

SETTLE_S = 3.0

STOP_PASSES = 4
"""How many times to look again before giving up and saying what is still there."""


@dataclass
class StopReport:
    killed: list[str] = field(default_factory=list)
    escalated: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    survivors: list[str] = field(default_factory=list)
    """Solver processes still running once every job was signalled."""
    passes: int = 0
    """Extra rounds needed, when something kept starting new work."""

    @property
    def clean(self) -> bool:
        return not self.failed and not self.survivors

    def lines(self) -> list[str]:
        out = []
        for name in self.killed:
            out.append(f"stopped {name}")
        for name in self.escalated:
            out.append(f"stopped {name} (it ignored the first signal)")
        for name, why in self.failed:
            out.append(f"could not stop {name}: {why}")
        if self.passes and not self.survivors:
            out.append(
                f"took {self.passes + 1} passes - something kept starting new work"
            )
        if self.survivors:
            out.append(
                "still running after every job was signalled: "
                + ", ".join(sorted(set(self.survivors)))
            )
            out.append("these outlived their job's process group, which is how compute leaks")
        if not out:
            out.append("nothing was running")
        elif self.clean:
            out.append("the instance is idle")
        return out


def running_solvers(backend: Backend) -> list[str]:
    """Solver processes on the instance, by name.

    Matched against `ps` output rather than `pgrep -f`, because a pattern search also
    matches the shell that is doing the searching.
    """
    try:
        result = backend.exec("ps -eo comm=", timeout_s=30)
    except BackendError:
        return []
    names = [line.strip() for line in result.output.splitlines()]
    return [name for name in names if name in SOLVERS]


PKILL_MATCHED = 0
PKILL_NOTHING_MATCHED = 1
"""`pkill` exits 1 when nothing matched, which here means it was already gone."""


def _force_kill(backend: Backend, names: list[str], report: StopReport) -> None:
    """Kill each leftover process by exact name, one call per name.

    `pkill` takes exactly one pattern: a second `-x name` makes it exit 2 having killed
    nothing at all. Sending that to /dev/null and following it with `true` is how a stop
    that stops nothing reports success, which is worse than not having the flag -- the
    user reads "done" and walks away from eight busy cores.
    """
    for name in dict.fromkeys(names):
        try:
            result = backend.exec(f"pkill -9 -x {shlex.quote(name)}", timeout_s=30)
        except BackendError as exc:
            report.failed.append((name, str(exc)))
            continue
        if result.exit_code not in (PKILL_MATCHED, PKILL_NOTHING_MATCHED):
            detail = (result.output or "").strip().splitlines()
            report.failed.append(
                (name, detail[0] if detail else f"pkill exited {result.exit_code}")
            )


def stop_everything(backend: Backend, store: Store, force: bool = False) -> StopReport:
    """Stop every job this study started, and confirm the work actually ended."""
    report = StopReport()

    for record in store.live_jobs():
        label = record.name or record.job_id[:8]
        try:
            backend.job_kill(record.job_id)
            report.killed.append(label)
        except BackendError as exc:
            report.failed.append((label, str(exc)))
            continue
        store.update_job(record.job_id, status="killed", end_reason="killed_by_client")

    if report.killed:
        time.sleep(SETTLE_S)

    survivors = running_solvers(backend)
    if survivors:
        # A second signal, this time one that cannot be ignored.
        for record in list(store.session.jobs.values()):
            if record.job_id in [r.job_id for r in store.live_jobs()]:
                continue
            try:
                backend.job_kill(record.job_id, signal="KILL")
                report.escalated.append(record.name or record.job_id[:8])
            except BackendError:
                pass

        # One pass is not enough. A study is usually driven by a script working
        # through a mesh ladder, and killing the solver it is currently running just
        # frees it to start the next one -- so the check three seconds later finds a
        # brand new simpleFoam and reports failure while everything did in fact die.
        # Keep going until the instance is actually quiet.
        for _attempt in range(STOP_PASSES):
            if force:
                _force_kill(backend, survivors, report)
            time.sleep(SETTLE_S)
            survivors = running_solvers(backend)
            if not survivors:
                break
            report.passes += 1

    report.survivors = survivors
    return report
