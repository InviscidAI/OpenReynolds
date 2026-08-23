"""Stopping work, and making sure it actually stopped.

Killing a job signals its process group. A solver launched through `mpirun` puts its
ranks in a different group, so the wrapper dies, the service records the job as killed,
and eight cores keep running on a machine nobody is watching. That happened, and the
first anyone knew of it was the bill.

So stopping is a loop with a check in it: signal, look, escalate, look again, and report
what is still there rather than what was requested.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .backend.base import Backend, BackendError
from .store import Store

SOLVERS = ("simpleFoam", "pimpleFoam", "interFoam", "icoFoam", "buoyantSimpleFoam",
           "potentialFoam", "snappyHexMesh", "blockMesh", "mpirun", "reconstructPar")
"""Processes worth naming when they outlive the job that started them."""

SETTLE_S = 3.0


@dataclass
class StopReport:
    killed: list[str] = field(default_factory=list)
    escalated: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    survivors: list[str] = field(default_factory=list)
    """Solver processes still running once every job was signalled."""

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
        if force:
            names = " ".join(f"-x {name}" for name in dict.fromkeys(survivors))
            try:
                backend.exec(f"pkill -9 {names} 2>/dev/null; true", timeout_s=60)
            except BackendError as exc:
                report.failed.append(("leftover processes", str(exc)))
        time.sleep(SETTLE_S)
        survivors = running_solvers(backend)

    report.survivors = survivors
    return report
