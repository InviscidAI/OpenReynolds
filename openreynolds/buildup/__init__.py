"""The build-up phase: the observer outside the run, and the boundary it enforces.

`docs/cad-build-up-handoff.md` is the argument; this package is its infrastructure. The
last round built the desk top-down and measured it against a bare kernel: three times the
steps on eight of eight prompts, no more meshes, and 85% of the discovery spend going to
learn its own surfaces. So the direction reverses -- start from the smallest thing that
works, and add only what a measured failure demands.

Two rules shape every module here, and both are mechanical rather than advisory:

* **One out-of-band observer.** Liveness, the alarms, the silent-failure probes, the
  contamination check and the grading all belong to `supervise.Supervisor`, which runs
  in a different process, reads the case off disk, and has no channel into the
  conversation. Nothing else observes the run.
* **Absence is enforced and then verified.** `isolation` asserts a clean workspace before
  a run and greps the run's own output after it. A run that touched a house surface is
  contaminated and is discarded from the baseline rather than averaged in.

Nothing in this package is synced to a workspace, imported by the desk, or named in its
brief. That is what makes the toolbox imports in `probes` legitimate: they happen here.
"""

from __future__ import annotations

from . import alarms, core, heartbeat, isolation, probes, record, supervise
from .alarms import HEARTBEAT_STALE_S, K, Alarm, evaluate
from .core import CORE_SYSTEM, CoreDesk
from .heartbeat import Beat, Heartbeat, alive, attach, pid_of
from .isolation import Contamination, Dirty, Hit, fresh_workspace, house_names, scan, scan_run
from .probes import REGISTRY, Probe, ProbeResult, run_all
from .record import TERMINAL, Record, classify
from .supervise import Supervisor, Watch, preflight

__all__ = [
    "Alarm", "Beat", "CORE_SYSTEM", "CoreDesk", "core", "Contamination", "Dirty", "HEARTBEAT_STALE_S", "Heartbeat", "Hit",
    "K", "Probe", "ProbeResult", "REGISTRY", "Record", "Supervisor", "TERMINAL", "Watch",
    "alarms", "alive", "attach", "classify", "evaluate", "fresh_workspace", "heartbeat",
    "house_names", "isolation", "pid_of", "preflight", "probes", "record", "run_all",
    "scan", "scan_run", "supervise",
]
