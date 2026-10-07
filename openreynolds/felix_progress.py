"""What a Felix job's log says: how far it got, how healthy it is, how it ended.

The OpenFOAM half of `progress.py` reads `Time =` and `Solving for` out of a log.
A Felix solve writes none of that. Its stderr is a strict format (Felix
`docs/user/stderr.md`): one `step=` record per time step, read by field name; a
`ptc step= ... CONVERGED` record when a steady-state run stops; and messages that
start with exactly one of `ERROR:`, `WARNING:` or `NOTE:`, where an unprefixed line
continues the message above it. Through `felix run` the same lines arrive as
felixd's log, with felixd's own `NOTE: felixd: ...` lines among them, and the job
ends with the solve's final status as a JSON object on stdout.

This module turns such a log -- whole, or the tail window the tracker reads -- into
`FelixFacts`, and from those into the two things the harness shows: `LogFacts` for
the progress bar, and `summary_lines` for a person or the model asking what the job
did. Facts only. Whether a residual is low enough, or which GPU to try after an
out-of-memory stop, is the reader's call; the out-of-memory line is parsed so the
number it gives -- the memory the case needs -- is not lost in a paragraph.
"""

from __future__ import annotations

import json
import math
import posixpath
import re
import shlex
from dataclasses import dataclass, field
from typing import Any

from .progress import LogFacts

STEP_FIELDS_INT = ("step", "gcr", "gmres", "cg", "sa_gmres", "energy_gmres", "corrector",
                   "energy_corrector", "sa_corrector")
ITERATION_FIELDS = ("gcr", "gmres", "cg", "sa_gmres", "energy_gmres")
RESIDUAL_FIELDS = ("res0", "resF", "gcr_res", "cg_res", "sa_gmres_res", "energy_gmres_res")

TREND_WINDOW = 10
"""Steps per window when the residual trend compares the latest steps with the ones
before them. Ten steps of a Felix run is a fraction of a second to a few seconds."""

TREND_FACTOR = 1.5
"""A window mean this many times below (above) the previous one reads as falling
(rising); in between, flat. The two numbers are printed beside the word."""

_PREFIXES = ("ERROR:", "WARNING:", "NOTE:")
_PAIR = re.compile(r"(\w+)=(\S+)")

# The three out-of-memory forms of docs/user/stderr.md "GPU memory errors".
_OOM_NEEDS = re.compile(
    r"this case needs at least ([\d.]+) GB of GPU memory.*?the GPU has ([\d.]+) GB "
    r"\(([\d.]+) GB free at start\)")
_OOM_GCR = re.compile(
    r"`gcr\.max: (\d+)` needs ([\d.]+) GB of GPU memory.*?the GPU has ([\d.]+) GB "
    r"\(([\d.]+) GB free at start\)(?:.*?largest gcr\.max that fits is (\d+))?")
_OOM_ALLOC = re.compile(
    r"out of GPU memory: ([\d.]+) GB allocated by this run when a request for ([\d.]+) GB "
    r"failed; the GPU has ([\d.]+) GB \(([\d.]+) GB free at start\)")
_NONFINITE = re.compile(r"non-finite \(NaN/Inf\) (?:at step (\d+)|in the initial state)")
_GPU_MEMORY = re.compile(r"^GPU memory: ([\d.]+) GB used by this run")
_HEALTH_OPENINGS = (
    "the outer corrector used all", "the outer-corrector residual INCREASED",
    "the SA corrector loop used all", "the corrector-pass linear solve",
    "the pressure sub-solve", "the nu_tilde (turbulence) linear solve",
    "the temperature linear solve", "the momentum sub-solve",
    "the wall function's law-of-the-wall solve",
)


@dataclass
class OutOfMemory:
    """An out-of-memory stop: what the case needs against what the GPU had."""

    needed_gb: float
    gpu_gb: float
    free_gb: float
    largest_gcr_max: int | None = None
    line: str = ""


@dataclass
class Trend:
    word: str
    """falling, flat or rising"""
    before: float
    recent: float
    window: int


@dataclass
class FelixFacts:
    step: int | None = None
    sim_time: float | None = None
    dt: float | None = None
    wall_time: float | None = None
    wall_step: float | None = None
    residuals: dict[str, float] = field(default_factory=dict)
    iterations: dict[str, int] = field(default_factory=dict)
    corrector: int | None = None
    trend: Trend | None = None
    steps_seen: int = 0
    ptc_converged_step: int | None = None
    ptc: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    health_notes: list[str] = field(default_factory=list)
    """End-of-run health paragraphs printed as NOTE (the run still advanced)."""
    felixd_notes: list[str] = field(default_factory=list)
    oom: OutOfMemory | None = None
    nonfinite_step: int | None = None
    gpu_memory_gb: float | None = None
    solution_written: bool = False
    warm_start: str | None = None
    status: dict[str, Any] | None = None
    """The final status `felix run` prints on stdout, when the log reached it."""
    summary: dict[str, Any] | None = None
    """The run summary between ===JSON_START=== and ===JSON_END===, when a raw
    solver's stdout landed in the same log."""
    last_line: str = ""

    @property
    def finished(self) -> bool:
        if self.status is not None:
            return self.status.get("state") == "succeeded"
        return self.solution_written and not self.errors

    @property
    def outcome(self) -> str:
        """running, finished, stopped (a non-finite state), failed or killed."""
        state = (self.status or {}).get("state")
        if state == "succeeded":
            return "finished"
        if state == "killed":
            return "killed"
        if state in ("failed", "rejected", "expired"):
            return "stopped" if self.nonfinite_step is not None else "failed"
        if self.nonfinite_step is not None:
            return "stopped"
        if self.errors:
            return "failed"
        if self.solution_written:
            return "finished"
        return "running"


# -- the command ----------------------------------------------------------------------


def _felix_words(cmd: str) -> list[list[str]]:
    """Each simple command in a shell line, as words."""
    out = []
    for part in re.split(r"&&|\|\||;|\|", cmd or ""):
        try:
            words = shlex.split(part)
        except ValueError:
            words = part.split()
        if words:
            out.append(words)
    return out


def felix_subcommand(cmd: str) -> tuple[str, list[str]] | None:
    """(`run`|`continue`, the words after it) for the last `felix run/continue` in a
    shell line, else None. `felix gpus`, `felix-check-mesh` and `felix-tag-mesh` are
    not solves."""
    found = None
    for words in _felix_words(cmd):
        for i, word in enumerate(words[:-1]):
            if word.rsplit("/", 1)[-1] == "felix" and words[i + 1] in ("run", "continue"):
                found = (words[i + 1], words[i + 2:])
    return found


def is_felix_cmd(cmd: str) -> bool:
    return felix_subcommand(cmd) is not None


def case_dir_from_felix_cmd(cmd: str, cwd: str) -> str | None:
    """The case directory a `felix run|continue <case>` names, resolved against the
    last `cd` in the line and then `cwd`."""
    from .progress import case_dir_from_cmd

    found = felix_subcommand(cmd)
    if found is None:
        return None
    _sub, rest = found
    case = None
    skip = False
    for word in rest:
        if skip:
            skip = False
            continue
        if word in (">", ">>", "2>", "<"):
            break
        if word.startswith("--"):
            skip = "=" not in word
            continue
        if word.startswith(">") or word.startswith("2>"):
            break
        case = word
        break
    if case is None:
        return None
    if case.startswith("/"):
        return posixpath.normpath(case)
    return posixpath.normpath(posixpath.join(case_dir_from_cmd(cmd, cwd), case))


_NSTEPS = re.compile(r"^nsteps\s*:\s*(\d+)\s*(?:#.*)?$", re.M)


def nsteps_from_control(text: str) -> int | None:
    """`nsteps:` from a control.yaml. A PTC run may stop before it; a warm start adds
    it to the step it starts from."""
    found = _NSTEPS.findall(text or "")
    return int(found[-1]) if found else None


# -- the log --------------------------------------------------------------------------


def _number(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value


def parse_step_line(line: str) -> dict[str, float]:
    """A `step=` (or `ptc step=`) record as {name: value}. Parsed by name, because
    fields are omitted rather than zeroed when they do not apply; `nan`, `-nan` and
    `inf` are numbers here."""
    record: dict[str, float] = {}
    for name, raw in _PAIR.findall(line):
        value = _number(raw)
        if value is not None:
            record[name] = value
    return record


def _split_json_tail(text: str) -> tuple[str, dict[str, Any] | None]:
    """The log without a trailing JSON object, and that object. `felix run` prints the
    final status last; it may be on one line or indented over many."""
    stripped = text.rstrip()
    if not stripped.endswith("}"):
        return text, None
    starts = [m.start() for m in re.finditer(r"(?m)^\{", stripped)]
    for start in reversed(starts):
        try:
            value = json.loads(stripped[start:])
        except ValueError:
            continue
        if isinstance(value, dict):
            return stripped[:start], value
    return text, None


_JSON_BLOCK = re.compile(r"===JSON_START===\s*(.*?)\s*===JSON_END===", re.S)


def _messages(lines: list[str]) -> list[tuple[str, str]]:
    """(kind, text) per message; `step` for step records. Unprefixed lines continue
    the message above them; a leading fragment with nothing above it (a tail window
    that started mid-line) is dropped."""
    out: list[tuple[str, str]] = []
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("step=") or line.startswith("ptc step="):
            out.append(("ptc" if line.startswith("ptc") else "step", line))
            continue
        for prefix in _PREFIXES:
            if line.startswith(prefix):
                out.append((prefix[:-1], line[len(prefix):].strip()))
                break
        else:
            if out and out[-1][0] in ("ERROR", "WARNING", "NOTE"):
                kind, body = out[-1]
                out[-1] = (kind, f"{body} {line.strip()}")
            else:
                out.append(("other", line.strip()))
    return out


def parse_felix_log(text: str) -> FelixFacts:
    """Facts from a Felix log or the tail of one. Last value wins for the step record;
    every ERROR and WARNING in the window is kept."""
    facts = FelixFacts()
    body, status = _split_json_tail(text or "")
    if status is not None and ("state" in status or "id" in status):
        facts.status = status
    elif status is not None and status.get("kind") == "run_summary":
        facts.summary = status
    block = _JSON_BLOCK.search(body)
    if block:
        try:
            facts.summary = json.loads(block.group(1))
        except ValueError:
            pass
        body = body[:block.start()] + body[block.end():]
    lines = body.splitlines()
    # A tail window starts wherever the byte offset fell: drop a first line that is
    # not one Felix would print in full.
    if lines and not lines[0].startswith(("step=", "ptc step=") + _PREFIXES):
        lines = lines[1:]
    res0: list[float] = []
    for kind, msg in _messages(lines):
        if kind == "step":
            record = parse_step_line(msg)
            if "step" not in record:
                continue
            facts.steps_seen += 1
            facts.step = int(record["step"])
            facts.sim_time = record.get("t")
            facts.dt = record.get("dt")
            facts.wall_time = record.get("wall_time")
            facts.wall_step = record.get("wall_step")
            facts.residuals = {k: record[k] for k in RESIDUAL_FIELDS if k in record}
            facts.iterations = {k: int(record[k]) for k in ITERATION_FIELDS if k in record}
            facts.corrector = int(record["corrector"]) if "corrector" in record else None
            if "res0" in record:
                res0.append(record["res0"])
        elif kind == "ptc":
            record = parse_step_line(msg.replace("ptc ", "", 1))
            if "CONVERGED" in msg and "step" in record:
                facts.ptc_converged_step = int(record["step"])
            facts.ptc = {k: v for k, v in record.items() if k != "step"}
        elif kind == "ERROR":
            facts.errors.append(msg)
            _read_error(facts, msg)
        elif kind == "WARNING":
            if msg.startswith("felixd:"):
                facts.felixd_notes.append(msg)
            facts.warnings.append(msg)
        elif kind == "NOTE":
            _read_note(facts, msg)
    facts.trend = _trend(res0)
    whole = [ln.strip() for ln in lines if ln.strip()]
    facts.last_line = whole[-1] if whole else ""
    return facts


def _read_error(facts: FelixFacts, msg: str) -> None:
    found = _NONFINITE.search(msg)
    if found:
        facts.nonfinite_step = int(found.group(1)) if found.group(1) else 0
        return
    found = _OOM_GCR.search(msg)
    if found:
        facts.oom = OutOfMemory(
            needed_gb=float(found.group(2)), gpu_gb=float(found.group(3)),
            free_gb=float(found.group(4)),
            largest_gcr_max=int(found.group(5)) if found.group(5) else None, line=msg)
        return
    found = _OOM_NEEDS.search(msg)
    if found:
        facts.oom = OutOfMemory(needed_gb=float(found.group(1)), gpu_gb=float(found.group(2)),
                                free_gb=float(found.group(3)), line=msg)
        return
    found = _OOM_ALLOC.search(msg)
    if found:
        allocated, request = float(found.group(1)), float(found.group(2))
        facts.oom = OutOfMemory(needed_gb=round(allocated + request, 6),
                                gpu_gb=float(found.group(3)), free_gb=float(found.group(4)),
                                line=msg)


def _read_note(facts: FelixFacts, msg: str) -> None:
    if msg.startswith("felixd:"):
        facts.felixd_notes.append(msg)
        return
    found = _GPU_MEMORY.match(msg)
    if found:
        facts.gpu_memory_gb = float(found.group(1))
    elif msg.startswith("solution written to"):
        facts.solution_written = True
    elif msg.startswith("warm start loaded from"):
        facts.warm_start = msg[len("warm start loaded from"):].strip()
    elif msg.startswith(_HEALTH_OPENINGS):
        facts.health_notes.append(msg)


def _trend(values: list[float]) -> Trend | None:
    """The latest window of res0 against the one before it, by geometric mean.

    res0 measures how far the previous state is from satisfying this step's
    equations; it falls as the flow approaches a steady state, and on a transient
    run levels off. The word is a description, not a verdict."""
    finite = [v for v in values if v > 0 and math.isfinite(v)]
    window = min(TREND_WINDOW, len(finite) // 2)
    if window < 2:
        return None

    def gmean(xs: list[float]) -> float:
        return math.exp(sum(math.log(x) for x in xs) / len(xs))

    before = gmean(finite[-2 * window:-window])
    recent = gmean(finite[-window:])
    if recent * TREND_FACTOR < before:
        word = "falling"
    elif recent > before * TREND_FACTOR:
        word = "rising"
    else:
        word = "flat"
    return Trend(word=word, before=before, recent=recent, window=window)


# -- what is shown -----------------------------------------------------------------------


def log_facts(facts: FelixFacts) -> LogFacts:
    """The progress bar's view of a Felix job: simulated time, dt, the step's
    nonlinear residuals, wall clock."""
    out = LogFacts()
    out.sim_time = facts.sim_time
    out.delta_t = facts.dt
    out.residuals = {k: v for k, v in facts.residuals.items() if k in ("res0", "resF")}
    out.clock_s = facts.wall_time
    out.execution_s = facts.wall_time
    out.last_line = facts.last_line
    out.ended = facts.outcome != "running"
    return out


def _gb(value: float) -> str:
    return f"{value:g} GB"


def headline(facts: FelixFacts, nsteps: int | None = None) -> str:
    """One line: where the run is."""
    if facts.step is None:
        return "no step completed yet"
    line = f"step {facts.step}"
    if nsteps:
        line += f" / {nsteps}"
    if facts.sim_time is not None:
        line += f", t = {facts.sim_time:g}"
    if facts.dt is not None:
        line += f", dt = {facts.dt:g}"
    if facts.wall_time is not None:
        line += f", {facts.wall_time:g} s of solver wall time"
    return line


def oom_line(oom: OutOfMemory) -> str:
    line = (f"out of GPU memory: the case needs {_gb(oom.needed_gb)}; the GPU has "
            f"{_gb(oom.gpu_gb)} ({_gb(oom.free_gb)} free at start)")
    if oom.largest_gcr_max is not None:
        line += f"; the largest gcr.max that fits is {oom.largest_gcr_max}"
    return line


def summary_lines(facts: FelixFacts, nsteps: int | None = None) -> list[str]:
    """What the job did, in a few lines: state, step, residuals and their trend, how
    it ended, every error, the warnings by their first sentence."""
    lines = [f"felix: {facts.outcome} -- {headline(facts, nsteps)}"]
    if facts.residuals:
        shown = "  ".join(f"{k} {v:.2e}" for k, v in facts.residuals.items())
        lines.append(f"last step: {shown}"
                     + (f", corrector {facts.corrector}" if facts.corrector is not None else ""))
    if facts.trend is not None:
        t = facts.trend
        lines.append(f"res0 {t.word}: {t.before:.2e} -> {t.recent:.2e} "
                     f"(geometric mean of {t.window} steps against the {t.window} before)")
    if facts.ptc_converged_step is not None:
        lines.append(f"PTC converged at step {facts.ptc_converged_step}")
    if facts.nonfinite_step is not None:
        where = f"step {facts.nonfinite_step}" if facts.nonfinite_step else "the initial state"
        lines.append(f"stopped: the solution became non-finite at {where}")
    if facts.oom is not None:
        lines.append(oom_line(facts.oom))
    for error in facts.errors:
        lines.append(f"ERROR: {error}")
    for note in facts.felixd_notes:
        lines.append(f"felixd: {note.split(':', 1)[-1].strip()}")
    solver_warnings = [w for w in facts.warnings if not w.startswith("felixd:")]
    for warning in solver_warnings[:6]:
        lines.append(f"WARNING: {_first_sentence(warning)}")
    if len(solver_warnings) > 6:
        lines.append(f"(+{len(solver_warnings) - 6} more WARNING lines)")
    for note in facts.health_notes[:3]:
        lines.append(f"NOTE: {_first_sentence(note)}")
    if facts.gpu_memory_gb is not None:
        lines.append(f"GPU memory used by the run: {_gb(facts.gpu_memory_gb)}")
    if facts.status is not None:
        lines.append(_status_line(facts.status))
    return lines


def _first_sentence(text: str, limit: int = 220) -> str:
    cut = re.split(r"(?<=\.)\s", text, maxsplit=1)[0]
    return cut if len(cut) <= limit else cut[: limit - 3] + "..."


def _status_line(status: dict[str, Any]) -> str:
    parts = [f"solve {status.get('state', '?')}"]
    if status.get("gpu"):
        parts.append(f"on {status['gpu']}")
    if status.get("exit_code") is not None:
        parts.append(f"exit {status['exit_code']}")
    if status.get("billed_s") is not None:
        parts.append(f"{status['billed_s']:g} s billed")
    if status.get("cost_usd") is not None:
        parts.append(f"${status['cost_usd']:.4f}")
    line = ", ".join(parts)
    if status.get("error"):
        line += f"; error: {status['error']}"
    if status.get("kill_reason"):
        line += f"; killed: {status['kill_reason']}"
    return line


def wake_line(name: str, facts: FelixFacts, nsteps: int | None = None) -> str:
    """The job in one line for the narration wake: where it is and, when it ended
    badly, the error that says why -- the out-of-memory figures in full."""
    line = f"{name}: felix {facts.outcome}, {headline(facts, nsteps)}"
    if facts.oom is not None:
        line += f"; ERROR: {facts.oom.line}"
    elif facts.errors:
        line += f"; ERROR: {facts.errors[-1]}"
    elif facts.ptc_converged_step is not None:
        line += f"; PTC converged at step {facts.ptc_converged_step}"
    return line
