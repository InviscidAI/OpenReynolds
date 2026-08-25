"""What is happening right now, as a bar and a line.

The screen used to have one dim line for everything: a fragment of the model's
reasoning, the name of a tool call, or -- only while the harness was polling -- how
long a job had run and the last line of its log. Whatever happened last overwrote it.
A solve at 2 % and a solve at 98 % looked the same, a five-minute `snappyHexMesh` in a
`bash` call showed nothing but "still running", and the moment the model started
thinking again the job vanished from the screen. A person watching a forty-minute
solve had, as one of them put it, no idea what was going on.

This module keeps the picture instead of the last event. It knows three kinds of fact
and composes them into one `Progress`:

- what the session thread is doing (thinking, writing, a tool call, waiting for you),
- whether the mirror is copying files home,
- what every running job is up to, from its log: solver time against the case's
  `endTime`, residuals, Courant number, the meshing phase.

All of it is derived in plain code from things the harness already reads. Nothing
here reaches the model -- it is presentation, and it may not steer -- and nothing
here is a verdict: an estimate is labelled as one, and a job with no known end is
shown without a percentage rather than with a made-up one.
"""

from __future__ import annotations

import calendar
import posixpath
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backend.base import BackendError

TAIL_BYTES = 6_000
"""How much of a log to read for facts. A pimpleFoam time step is a few hundred
bytes; this is a dozen of them, enough for every field's residual to appear."""

JOB_POLL_S = 20.0
"""How often the tracker looks at running jobs on its own. Watch mode looks on its own
clock as well; whichever asked most recently is the one that counts."""

TOOL_LOG_POLL_S = 10.0
"""How often the log of a running `bash` command is read, when the command says where
it is writing one."""

TICK_S = 1.0
"""How often the bar is redrawn so the elapsed counters move."""

CONTROL_DICT_TTL_S = 120.0
"""How long a case's `endTime` is believed before being read again. The model edits
controlDict mid-run sometimes; a stale end shows a wrong percentage for at most this
long."""

STALE_AFTER_S = 90.0
"""Past this, a job's facts are old enough that the line says so."""

BAR_WIDTH = 24


# -- reading logs --------------------------------------------------------------


TIME = re.compile(r"^Time = ([0-9.eE+-]+)\s*$", re.M)
DELTA_T = re.compile(r"^deltaT = ([0-9.eE+-]+)", re.M)
SOLVING = re.compile(r"Solving for (\S+?),\s*Initial residual = ([0-9.eE+-]+)")
CONTINUITY = re.compile(r"time step continuity errors : sum local = ([0-9.eE+-]+)")
COURANT = re.compile(r"Courant Number mean: ([0-9.eE+-]+) max: ([0-9.eE+-]+)")
CLOCK = re.compile(r"ExecutionTime = ([0-9.eE+-]+) s\s+ClockTime = ([0-9.eE+-]+) s")
END_MARK = re.compile(r"^End\s*$", re.M)

MESH_PHASES = (
    ("Shrinking and layer addition phase", "adding layers"),
    ("Morphing phase", "snapping"),
    ("Refinement phase", "refining"),
)
MESH_ITERATION = re.compile(r"(?:Refinement|Layer addition) iteration (\d+)")
MESH_CELLS = re.compile(r"cells:\s*(\d+)")


@dataclass
class LogFacts:
    """What the tail of a log says, and nothing it does not."""

    sim_time: float | None = None
    delta_t: float | None = None
    residuals: dict[str, float] = field(default_factory=dict)
    continuity: float | None = None
    courant_max: float | None = None
    execution_s: float | None = None
    clock_s: float | None = None
    mesh_phase: str | None = None
    mesh_iteration: int | None = None
    mesh_cells: int | None = None
    last_line: str = ""
    ended: bool = False

    @property
    def is_solver(self) -> bool:
        return self.sim_time is not None or bool(self.residuals)


def parse_log_tail(text: str) -> LogFacts:
    """Facts from the end of an OpenFOAM log -- a solver's or a mesher's.

    Only the last value of each thing: the tail is a window, and the most recent
    time step is the one that describes now."""
    facts = LogFacts()
    times = TIME.findall(text)
    if times:
        facts.sim_time = _float(times[-1])
    deltas = DELTA_T.findall(text)
    if deltas:
        facts.delta_t = _float(deltas[-1])
    # Residuals: the last initial residual per field, from the last time step only,
    # so that a field solved twice per step (a PISO corrector) reports its final pass.
    last_step = text.rsplit("\nTime = ", 1)[-1] if times else text
    for name, value in SOLVING.findall(last_step):
        facts.residuals[name] = _float(value) or 0.0
    cont = CONTINUITY.findall(last_step)
    if cont:
        facts.continuity = _float(cont[-1])
    courant = COURANT.findall(text)
    if courant:
        facts.courant_max = _float(courant[-1][1])
    clocks = CLOCK.findall(text)
    if clocks:
        facts.execution_s = _float(clocks[-1][0])
        facts.clock_s = _float(clocks[-1][1])
    # Meshing: the phase marker that appeared last is the phase it is in.
    best = -1
    for marker, phase in MESH_PHASES:
        at = text.rfind(marker)
        if at > best:
            best, facts.mesh_phase = at, phase
    iterations = MESH_ITERATION.findall(text)
    if iterations:
        facts.mesh_iteration = int(iterations[-1])
    cells = MESH_CELLS.findall(text)
    if cells:
        facts.mesh_cells = int(cells[-1])
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    facts.last_line = lines[-1] if lines else ""
    facts.ended = bool(END_MARK.search(text[-200:]))
    return facts


_DICT_ENTRY = re.compile(
    r"^\s*(startTime|endTime|deltaT|writeInterval)\s+([0-9.eE+-]+)\s*;", re.M
)
_STOP_AT = re.compile(r"^\s*stopAt\s+(\w+)\s*;", re.M)
_COMMENTS = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)


def parse_control_dict(text: str) -> dict[str, Any]:
    """The run's bounds from `system/controlDict`.

    `endTime` is only the end when `stopAt` says so -- `writeNow` and friends mean the
    run ends whenever somebody says, and a percentage against endTime would then be a
    number about nothing."""
    body = _COMMENTS.sub("", text)
    found: dict[str, Any] = {}
    for key, value in _DICT_ENTRY.findall(body):
        number = _float(value)
        if number is not None:
            found[key] = number
    stop = _STOP_AT.findall(body)
    if stop and stop[-1] != "endTime":
        found.pop("endTime", None)
        found["stopAt"] = stop[-1]
    return found


# -- reading commands ----------------------------------------------------------


MESHERS = frozenset(
    {
        "blockMesh", "snappyHexMesh", "cartesianMesh", "cartesian2DMesh", "tetMesh",
        "pMesh", "extrudeMesh", "refineMesh", "gmsh", "gmshToFoam", "fluentMeshToFoam",
        "fluent3DMeshToFoam", "ideasUnvToFoam", "cfx4ToFoam", "star4ToFoam",
        "polyDualMesh", "createPatch", "renumberMesh", "topoSet", "setSet",
        "surfaceFeatureExtract", "surfaceFeatures", "mergeMeshes", "stitchMesh",
        "transformPoints", "createBaffles", "mirrorMesh", "subsetMesh",
    }
)
POST = frozenset(
    {
        "postProcess", "foamToVTK", "sample", "foamListTimes", "pvpython", "pvbatch",
        "foamLog", "foamCalc", "surfaceMeshTriangulate", "foamToEnsight",
    }
)
PHASE_OF = {
    "decomposePar": "decomposing",
    "reconstructPar": "reconstructing",
    "reconstructParMesh": "reconstructing",
    "checkMesh": "checking the mesh",
    "setFields": "setting fields",
    "mapFields": "mapping fields",
    "python": "running python",
    "python3": "running python",
    "pip": "installing",
}
_TOKEN = re.compile(r"[A-Za-z0-9_./-]+")


def phase_from_cmd(cmd: str) -> tuple[str, str]:
    """(phase, executable) for a shell command, from the last thing in it that this
    recognises. A chain like `blockMesh && simpleFoam` reads as solving: it is what
    the command is for, and the log says which part is actually running."""
    phase, exe = "running", ""
    for token in _TOKEN.findall(cmd or ""):
        name = token.rsplit("/", 1)[-1]
        if name in MESHERS:
            phase, exe = "meshing", name
        elif name in PHASE_OF:
            phase, exe = PHASE_OF[name], name
        elif name in POST:
            phase, exe = "post-processing", name
        elif name in ("foamRun", "foamMultiRun") or (
            name.endswith("Foam") and "To" not in name and not name.startswith("foam")
        ):
            phase, exe = "solving", name
    return phase, exe


_CD = re.compile(
    r"(?:^|[;&|]\s*|\bthen\s+|\bdo\s+)\s*cd\s+(?:\"([^\"]+)\"|'([^']+)'|([^\s;&|\"']+))"
)


def case_dir_from_cmd(cmd: str, cwd: str) -> str:
    """Where the command's case lives: the last `cd` in it, resolved against cwd."""
    where = cwd or "/"
    for double, single, bare in _CD.findall(cmd or ""):
        target = double or single or bare
        if target == "-":
            continue
        where = target if target.startswith("/") else posixpath.join(where, target)
    return posixpath.normpath(where)


_REDIRECT = re.compile(r"(?:>>?|\btee\s+(?:-a\s+)?)\s*([\"']?)([^\s\"'&|;>]+)\1")


def log_target_from_cmd(cmd: str, case_dir: str) -> str | None:
    """The file a command redirects to, if it says. `2>&1` is not a file and
    `/dev/null` is not a log."""
    target = None
    for _quote, path in _REDIRECT.findall(cmd or ""):
        if path.startswith("&") or path == "/dev/null":
            continue
        target = path
    if target is None:
        return None
    return target if target.startswith("/") else posixpath.join(case_dir, target)


# -- the picture ---------------------------------------------------------------


@dataclass
class JobProgress:
    """One running job as the bar sees it."""

    job_id: str
    name: str
    phase: str
    executable: str = ""
    facts: LogFacts = field(default_factory=LogFacts)
    start_time: float = 0.0
    end_time: float | None = None
    stop_at: str | None = None
    fraction: float | None = None
    elapsed_s: float | None = None
    eta_s: float | None = None
    log_size: int = 0
    as_of: float = 0.0
    """Monotonic clock at the last successful read."""

    def headline(self, now: float | None = None) -> str:
        parts = [f"{self.phase} {self.name}".strip()]
        facts = self.facts
        if self.phase == "solving" and facts.sim_time is not None:
            when = f"Time {facts.sim_time:g}"
            if self.end_time is not None:
                when += f" / {self.end_time:g} s"
            elif self.stop_at:
                when += f" s (stopAt {self.stop_at})"
            else:
                when += " s"
            parts.append(when)
        elif self.phase == "meshing" and facts.mesh_phase:
            mesh = facts.mesh_phase
            if facts.mesh_iteration is not None:
                mesh += f" iteration {facts.mesh_iteration}"
            parts.append(mesh)
        if self.elapsed_s is not None:
            parts.append(duration(self.elapsed_s))
        if self.eta_s is not None:
            parts.append(f"~{duration(self.eta_s)} left")
        age = (now or time.monotonic()) - self.as_of if self.as_of else 0.0
        if age > STALE_AFTER_S:
            parts.append(f"as of {duration(age)} ago")
        return " · ".join(parts)

    def detail(self) -> str:
        facts = self.facts
        bits = []
        if facts.residuals:
            shown = list(facts.residuals.items())[:4]
            bits.append("  ".join(f"{name} {value:.1e}" for name, value in shown))
        if facts.courant_max is not None:
            bits.append(f"Co max {facts.courant_max:.2f}")
        if facts.delta_t is not None:
            bits.append(f"dt {facts.delta_t:.2e}")
        if facts.continuity is not None:
            bits.append(f"continuity {facts.continuity:.1e}")
        if facts.mesh_cells is not None and self.phase == "meshing":
            bits.append(f"{facts.mesh_cells:,} cells")
        if bits:
            return " · ".join(bits)
        return facts.last_line[:140]


@dataclass
class Activity:
    """What the session thread is doing, as told by whoever is doing it."""

    kind: str
    """thinking | writing | tool | waiting"""
    label: str = ""
    cmd: str = ""
    cwd: str = ""
    since: float = field(default_factory=time.monotonic)
    log_path: str | None = None
    facts: LogFacts | None = None
    end_time: float | None = None
    start_time: float = 0.0


@dataclass(frozen=True)
class Progress:
    """What the bar and its two lines show. Immutable so a view can compare."""

    phase: str = "idle"
    headline: str = ""
    detail: str = ""
    fraction: float | None = None
    busy: bool = False
    """Something is in flight; with no fraction, the bar pulses rather than sits."""
    tick: int = 0
    """A counter that moves every redraw, so a pulse can move with it."""

    def percent(self) -> str:
        return f"{self.fraction * 100:3.0f}%" if self.fraction is not None else "    "


class Tracker:
    """Keeps the picture current, and tells the view every second.

    Three threads talk to it -- the session thread (activity), the mirror thread
    (syncing), and its own (jobs, tool logs, redraws) -- and every read of a job's
    log happens here rather than on the session thread, so a slow instance can
    delay a redraw but never a reply.
    """

    def __init__(
        self,
        view: Any,
        backend: Any = None,
        store: Any = None,
        home: str = "",
        local_dir: Path | None = None,
    ):
        self.view = view
        self.backend = backend
        self.store = store
        self.home = home
        self.local_dir = local_dir
        self.concierge = None
        """The front desk, told when the agent starts and stops working so it knows
        whether the user is waiting on a busy agent. Set by the session; may be None."""
        self._lock = threading.RLock()
        self._activity: Activity | None = None
        self._syncing_since: float | None = None
        self._sync_note = ""
        self._jobs: dict[str, JobProgress] = {}
        self._first_seen: dict[str, tuple[float, float]] = {}
        self._jobs_refreshed = 0.0
        self._tool_polled = 0.0
        self._control: dict[str, tuple[float, dict[str, Any]]] = {}
        self._tick = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- what the session thread says ----------------------------------------

    def begin(self, kind: str, label: str = "", *, cmd: str = "", cwd: str = "") -> None:
        """The session thread started doing something."""
        activity = Activity(kind=kind, label=label, cmd=cmd, cwd=cwd or self.home)
        if kind == "tool" and label == "bash" and cmd:
            case_dir = case_dir_from_cmd(cmd, activity.cwd)
            activity.log_path = log_target_from_cmd(cmd, case_dir)
        with self._lock:
            self._activity = activity
            self._tool_polled = 0.0
        if self.concierge is not None:
            self.concierge.working(kind != "waiting")
        self.push()

    def idle(self) -> None:
        """The session thread finished whatever it was doing."""
        with self._lock:
            self._activity = None
            jobs = bool(self._jobs)
        if self.concierge is not None:
            # A running job still keeps the desk narrating; a bare idle does not.
            self.concierge.working(jobs)
        self.push()

    # -- what the mirror says --------------------------------------------------

    def sync_begin(self) -> None:
        with self._lock:
            self._syncing_since = time.monotonic()
            self._sync_note = ""

    def sync_end(self, report: Any = None) -> None:
        with self._lock:
            self._syncing_since = None
            pulled = getattr(report, "pulled", None)
            self._sync_note = f"{len(pulled)} file(s) arrived" if pulled else ""
        self.push()

    # -- what the jobs say -----------------------------------------------------

    def refresh_jobs(self, force: bool = False) -> list[JobProgress]:
        """Look at every running job. Cheap when looked at recently."""
        if self.backend is None or self.store is None:
            return []
        now = time.monotonic()
        with self._lock:
            if not force and now - self._jobs_refreshed < JOB_POLL_S:
                return list(self._jobs.values())
            self._jobs_refreshed = now
        try:
            records = list(self.store.live_jobs())[:3]
        except RuntimeError:
            return list(self._jobs.values())
        fresh: dict[str, JobProgress] = {}
        for record in records:
            progress = self._read_job(record)
            if progress is not None:
                fresh[record.job_id] = progress
        with self._lock:
            kept = {jid: p for jid, p in self._jobs.items() if any(r.job_id == jid for r in records)}
            kept.update(fresh)
            self._jobs = kept
            for gone in [jid for jid in self._first_seen if jid not in kept]:
                self._first_seen.pop(gone, None)
        return list(self._jobs.values())

    def _read_job(self, record: Any) -> JobProgress | None:
        try:
            status = self.backend.job_status(record.job_id)
        except BackendError:
            return None
        size = status.log_size or 0
        tail = ""
        if size > 0:
            try:
                tail, _next, _eof = self.backend.job_tail(
                    record.job_id, offset=max(0, size - TAIL_BYTES)
                )
            except BackendError:
                tail = ""
        facts = parse_log_tail(tail)
        phase, exe = phase_from_cmd(record.cmd)
        if facts.is_solver:
            phase = "solving"
        elif facts.mesh_phase:
            phase = "meshing"
        progress = JobProgress(
            job_id=record.job_id,
            name=record.name or record.job_id[:8],
            phase=phase,
            executable=exe,
            facts=facts,
            log_size=size,
            elapsed_s=_seconds_since(record.launched_at),
            as_of=time.monotonic(),
        )
        if phase == "solving":
            case_dir = case_dir_from_cmd(record.cmd, getattr(record, "cwd", "") or self.home)
            control = self._control_dict(case_dir)
            progress.start_time = float(control.get("startTime", 0.0) or 0.0)
            progress.end_time = control.get("endTime")
            progress.stop_at = control.get("stopAt")
            self._estimate(progress)
        return progress

    def _estimate(self, progress: JobProgress) -> None:
        """Fraction and time left, from what the log and the controlDict say.

        The rate is measured between this poll and the first one that saw a time,
        once enough wall clock separates them; before that, the solver's own
        ClockTime serves. Either is an estimate, and the line says `~`."""
        t = progress.facts.sim_time
        end = progress.end_time
        if t is None:
            return
        start = progress.start_time
        if end is not None and end > start:
            progress.fraction = max(0.0, min(1.0, (t - start) / (end - start)))
        clock = progress.facts.clock_s
        first = self._first_seen.setdefault(progress.job_id, (t, time.monotonic()))
        rate = None
        wall = time.monotonic() - first[1]
        if wall > 30 and t > first[0]:
            rate = (t - first[0]) / wall
        elif clock and clock > 0 and t > start:
            rate = (t - start) / clock
        if end is not None and rate and rate > 0 and t < end:
            progress.eta_s = (end - t) / rate

    def _control_dict(self, case_dir: str) -> dict[str, Any]:
        """The case's controlDict, from the mirror when it is here, else from the
        instance; remembered for a while either way."""
        now = time.monotonic()
        cached = self._control.get(case_dir)
        if cached and now - cached[0] < CONTROL_DICT_TTL_S:
            return cached[1]
        path = f"{case_dir}/system/controlDict"
        text = self._read_local(path)
        if text is None and self.backend is not None:
            try:
                text = self.backend.get_file(path, limit=32_000).decode("utf-8", "replace")
            except (BackendError, OSError):
                text = None
        parsed = parse_control_dict(text) if text else {}
        self._control[case_dir] = (now, parsed)
        return parsed

    def _read_local(self, path: str) -> str | None:
        if self.local_dir is None:
            return None
        from .mirror import local_for  # local import: mirror is a peer, not a dependency

        try:
            return local_for(self.local_dir, path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def _poll_tool_log(self) -> None:
        """A `bash` command that said where its log goes can be watched while it runs."""
        with self._lock:
            activity = self._activity
        if activity is None or activity.kind != "tool" or not activity.log_path:
            return
        if self.backend is None:
            return
        now = time.monotonic()
        if now - self._tool_polled < TOOL_LOG_POLL_S:
            return
        self._tool_polled = now
        try:
            size = self.backend.stat(activity.log_path).size
            data = self.backend.get_file(
                activity.log_path, offset=max(0, size - TAIL_BYTES), limit=TAIL_BYTES
            )
        except (BackendError, OSError):
            return
        facts = parse_log_tail(data.decode("utf-8", "replace"))
        if facts.is_solver and activity.end_time is None:
            case_dir = case_dir_from_cmd(activity.cmd, activity.cwd)
            control = self._control_dict(case_dir)
            activity.start_time = float(control.get("startTime", 0.0) or 0.0)
            activity.end_time = control.get("endTime")
        with self._lock:
            if self._activity is activity:
                activity.facts = facts

    # -- the picture -----------------------------------------------------------

    def snapshot(self) -> Progress:
        now = time.monotonic()
        with self._lock:
            activity = self._activity
            jobs = list(self._jobs.values())
            syncing = self._syncing_since
            sync_note = self._sync_note
            tick = self._tick
        lead = next((j for j in jobs if j.fraction is not None), jobs[0] if jobs else None)
        if lead is not None:
            headline = lead.headline(now)
            if len(jobs) > 1:
                headline += f"  (+{len(jobs) - 1} more)"
            detail = lead.detail()
            if activity is not None and activity.kind != "waiting":
                detail = f"{_activity_line(activity, now)}   {detail}".strip()
            elif syncing is not None:
                detail = f"syncing files · {duration(now - syncing)}   {detail}".strip()
            return Progress(lead.phase, headline, detail, lead.fraction, True, tick)
        if activity is not None and activity.kind != "waiting":
            phase = activity.kind if activity.kind != "tool" else phase_from_cmd(activity.cmd)[0]
            fraction = _tool_fraction(activity)
            detail = _tool_detail(activity)
            if syncing is not None:
                detail = f"syncing files · {duration(now - syncing)}   {detail}".strip()
            return Progress(phase, _activity_line(activity, now), detail, fraction, True, tick)
        if syncing is not None:
            return Progress("syncing", f"syncing files · {duration(now - syncing)}", "", None, True, tick)
        return Progress("waiting", "waiting for you", sync_note, None, False, tick)

    def push(self) -> None:
        """Tell the view. Never raises: the view may be tearing down."""
        with self._lock:
            self._tick += 1
        try:
            self.view.progress(self.snapshot())
        except Exception:  # noqa: BLE001 - presentation may not end a session
            pass

    def facts_for_wake(self) -> list[str]:
        """Job facts in words, for the narration wake. Facts only: what the solver's
        time is, what the controlDict's end is. No estimate goes to the model."""
        lines = []
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            facts = job.facts
            if facts.sim_time is None:
                continue
            line = f"{job.name}: solver time {facts.sim_time:g}"
            if job.end_time is not None:
                line += f" of endTime {job.end_time:g}"
            if facts.clock_s is not None:
                line += f", ClockTime {facts.clock_s:g} s"
            lines.append(line)
        return lines

    # -- its own clock ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="progress", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(TICK_S):
            try:
                if self.store is not None and self.store.live_jobs():
                    self.refresh_jobs()
                self._poll_tool_log()
            except Exception:  # noqa: BLE001 - a redraw may not end a session
                pass
            self.push()


# -- words and numbers ---------------------------------------------------------


def _activity_line(activity: Activity, now: float) -> str:
    elapsed = duration(now - activity.since)
    if activity.kind == "thinking":
        return f"thinking · {elapsed}"
    if activity.kind == "writing":
        return f"writing · {elapsed}"
    if activity.kind == "tool":
        phase, exe = phase_from_cmd(activity.cmd) if activity.label == "bash" else ("", "")
        what = f"{activity.label}: {exe}" if exe else activity.label
        parts = [what]
        facts = activity.facts
        if facts is not None:
            if facts.is_solver and facts.sim_time is not None:
                when = f"Time {facts.sim_time:g}"
                if activity.end_time is not None:
                    when += f" / {activity.end_time:g} s"
                parts.append(when)
            elif facts.mesh_phase:
                mesh = facts.mesh_phase
                if facts.mesh_iteration is not None:
                    mesh += f" iteration {facts.mesh_iteration}"
                parts.append(mesh)
        parts.append(elapsed)
        return " · ".join(parts)
    return f"{activity.kind} · {elapsed}"


def _tool_fraction(activity: Activity) -> float | None:
    facts = activity.facts
    if facts is None or facts.sim_time is None or activity.end_time is None:
        return None
    span = activity.end_time - activity.start_time
    if span <= 0:
        return None
    return max(0.0, min(1.0, (facts.sim_time - activity.start_time) / span))


def _tool_detail(activity: Activity) -> str:
    facts = activity.facts
    if facts is None:
        return ""
    stub = JobProgress(job_id="", name="", phase="solving" if facts.is_solver else "meshing", facts=facts)
    return stub.detail()


def duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.0f} min"
    hours = int(minutes // 60)
    return f"{hours} h {minutes - hours * 60:.0f} min"


def bar(progress: Progress, width: int = BAR_WIDTH) -> str:
    """The bar as characters: filled to the fraction, a moving pulse when there is
    none but something is happening, and empty when nothing is."""
    if progress.fraction is not None:
        filled = int(round(progress.fraction * width))
        return "█" * filled + "░" * (width - filled)
    if progress.busy:
        span = 4
        at = progress.tick % (width - span + 1)
        return "░" * at + "▓" * span + "░" * (width - span - at)
    return "░" * width


def _float(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _seconds_since(stamp: str) -> float | None:
    try:
        began = calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))
    except (TypeError, ValueError):
        return None
    return max(0.0, time.time() - began)
