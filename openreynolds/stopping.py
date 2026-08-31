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

from .backend.base import WORKSPACE_ROOT, Backend, BackendError
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

    Instance-wide, and that is the problem with it: the workspace service caps an
    account at one instance and `acquire()` joins the existing one, so "what is
    running here" is not the same question as "what did this study start". Use
    `own_solvers` for anything that is about to kill something.
    """
    try:
        result = backend.exec("ps -eo comm=", timeout_s=30)
    except BackendError:
        return []
    names = [line.strip() for line in result.output.splitlines()]
    return [name for name in names if name in SOLVERS]


_OWN_PROBE = r"""for d in /proc/[0-9]*; do
c=$(cat "$d/comm" 2>/dev/null) || continue
w=$(readlink "$d/cwd" 2>/dev/null) || continue
case "$w" in %s|%s/*) printf '%%s %%s\n' "${d#/proc/}" "$c" ;; esac
done"""
"""Every process working inside one directory, as `pid name`.

`ps` says what is running and not where it is working, and where it is working is the
only thing that distinguishes this study's solver from somebody else's. `/proc/<pid>/cwd`
answers it for a process in any process group -- which matters, because the reason this
sweep exists at all is mpirun ranks that escape their job's group."""


def own_solvers(backend: Backend, home: str) -> list[tuple[str, str]]:
    """`(pid, name)` for solver processes working inside this study's directory.

    This is the scoped replacement for `running_solvers` on every path that kills.
    A process is this study's if its working directory is under the study's home, which
    holds for a solver launched by a job here and cannot hold for one launched by
    another session in its own study directory.
    """
    probe = _OWN_PROBE % (shlex.quote(home), shlex.quote(home))
    try:
        result = backend.exec(probe, timeout_s=30)
    except BackendError:
        return []
    found = []
    for line in result.output.splitlines():
        pid, _, name = line.strip().partition(" ")
        if pid.isdigit() and name in SOLVERS:
            found.append((pid, name))
    return found


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


def _kill_own(backend: Backend, leftovers: list[tuple[str, str]], report: StopReport) -> None:
    """Kill each leftover by pid, in one call.

    By pid rather than by name, because a name reaches every copy on the instance and
    the pids here were chosen by working directory. A pid that has already gone makes
    `kill` exit non-zero and that is not a failure -- the next sweep is what decides
    whether anything survived.
    """
    if not leftovers:
        return
    pids = " ".join(pid for pid, _ in leftovers)
    try:
        backend.exec(f"kill -9 {pids} 2>/dev/null || true", timeout_s=30)
    except BackendError as exc:
        for _, name in leftovers:
            report.failed.append((name, str(exc)))


def stop_everything(
    backend: Backend, store: Store, force: bool = False, home: str | None = None
) -> StopReport:
    """Stop every job this study started, and confirm the work actually ended.

    `home` is this study's own directory, and giving it is what keeps the sweep to this
    study's work. Without it the survivor check is `ps` across the whole instance and the
    escalation is `pkill -9 -x <name>` -- and because an account is capped at one instance
    and `acquire()` joins the one that is already there, "the whole instance" regularly
    means somebody else's solve. A session that had started no jobs at all still reached
    that branch and killed another study's mpirun; that is what this argument closes.

    With `home`, survivors are the solver processes whose working directory is under it
    and they are killed by pid. The check still happens and orphaned mpirun ranks are
    still caught -- that is why the sweep exists -- it just cannot reach past this study.

    `force` keeps its old meaning for an unscoped call (`openreynolds stop --force`): the
    instance-wide kill by name, a deliberate human escalation. Scoped, the kill is safe
    by construction and needs no flag.
    """
    report = StopReport()
    scoped = bool(home) and str(home).rstrip("/") != WORKSPACE_ROOT
    """A home of `/work` is every study's directory at once (studies that predate homes
    keep the volume root), so it distinguishes nothing and is treated as unscoped."""

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

    def look() -> tuple[list[str], list[tuple[str, str]]]:
        """(names to report, pids to kill). Scoped when this study has a home of its own."""
        if scoped:
            leftovers = own_solvers(backend, str(home))
            return [name for _, name in leftovers], leftovers
        return running_solvers(backend), []

    survivors, leftovers = look()
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
            if scoped:
                _kill_own(backend, leftovers, report)
            elif force:
                _force_kill(backend, survivors, report)
            time.sleep(SETTLE_S)
            survivors, leftovers = look()
            if not survivors:
                break
            report.passes += 1

    report.survivors = survivors
    return report
