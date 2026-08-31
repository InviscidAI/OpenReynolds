#!/usr/bin/env python3
"""One answer to "what is happening?" -- phase, time reached, pace, and the exact
paths of the pictures that already exist.

The same question comes back every few minutes while something long is running, and
answering it by hand costs a `tail`, an `ls`, a look at `controlDict` and some
arithmetic -- four calls that mostly return things which have not changed. This is
those four collapsed into one. It is read-only on purpose: it renders nothing, starts
nothing, and writes nothing, so it can be called as often as the question is asked,
including while a solve or a frame job is using the machine.

what it puts together

- the phase table from `study_state`, so "where is it" is answered from what the study
  wrote down rather than from what a log happens to look like;
- `Time = ` from the solver log against `endTime` from `system/controlDict`. `stopAt`
  is read as well: when it is not `endTime`, the run ends when somebody says so and a
  percentage against `endTime` would be a number about nothing, so none is shown;
- iterations or time steps completed, which are the same count under two names --
  a steady solver's `Time` is its iteration number;
- the residual trend, as the recent window compared against the window before it. It
  says falling, flat or rising and prints the two numbers behind that word. Whether a
  residual is falling *fast enough* is a judgement about this case, and is yours;
- the latest Courant number, parsed by `log_digest` rather than by a second copy of
  the same regex;
- PNGs on disk in any `*_frames/` directory, against the count expected from a sidecar
  next to them or, failing that, from the number of write times in the case;
- an estimate of the time left with the basis written next to it. A steady run's
  estimate is wall-clock per iteration and tends to hold; a transient with
  `adjustTimeStep` on re-chooses its step whenever the flow does, so the same
  arithmetic deserves less trust and the report says so rather than quietly implying
  the two are equally good;
- the previews registered in the manifest, as absolute paths, because "can I see it?"
  is answered by a path that can go straight into `read_file` and not by a description
  of one;
- ladder rung outcomes recorded by `ladder.py --record`, read from the same manifest.
  A rung climbed in an earlier session is evidence that already exists, and a fresh
  session re-deriving it pays full price for a number the volume was holding all
  along -- which is how a solver choice established in one round has been lost to
  the next.

on reading the log twice: the whole file goes through `log_digest.digest` once for the
residual series, and then the last few hundred kilobytes are read again to pair each
`Time` with its `ExecutionTime`. Pace only means anything over a recent window anyway,
so that second read is bounded however large the log has grown.

    python3 progress_report.py /work/case
    python3 progress_report.py /work/case --json
    python3 progress_report.py /work/case --log /work/case/log.pimpleFoam --window 30
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import log_digest  # noqa: E402  (sibling script, not a package)
import study_state  # noqa: E402

TAIL_BYTES = 400_000
"""How much of the log tail is re-read for the (Time, ExecutionTime) pairs. A time
step prints a few hundred bytes, so this is on the order of a thousand of them --
far more window than any pace estimate should be using, and a fixed cost."""

MAX_WINDOW = 50
"""The largest number of steps either the trend or the pace will look back over.
Longer windows average across whatever the solve was doing an hour ago, which is
exactly the thing the question is not about."""

FLAT_DECADES = 0.1
"""How far the geometric mean has to move between the two windows before the trend
gets a direction rather than being called flat. 0.1 decades is about 26 %, which is
below the step-to-step noise of most residual traces."""

SKIP_DIRS = frozenset({"constant", "system", "postProcessing", "dynamicCode", "VTK", ".reynolds"})
"""Directories not walked when looking for frame directories. Nothing in them is a
frame directory, and `postProcessing` in particular can hold thousands of files."""

FRAME_DEPTH = 3
"""How far below the study root the walk for `*_frames/` goes."""

_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
_COMMENTS = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
_ENTRY = re.compile(r"^\s*(\w+)\s+([^;{}]+?)\s*;")

CONTROL_NUMBERS = ("startTime", "endTime", "deltaT", "writeInterval", "maxCo", "maxDeltaT")
CONTROL_WORDS = ("application", "stopAt", "startFrom", "writeControl", "timeFormat")
TRUTHY = frozenset({"yes", "true", "on", "1"})


# -- controlDict -------------------------------------------------------------------


def parse_control_dict(text: str) -> dict[str, Any]:
    """The run's bounds and its solver, from the text of `system/controlDict`.

    Only top-level entries count. A `functions {}` block routinely contains its own
    `writeControl` and `writeInterval`, and reading those as the run's would report
    the write cadence of one function object as the cadence of the case, so brace
    depth is tracked and anything nested is skipped.

    `stopAt` is carried through as given. When it is not `endTime` the run stops on
    somebody's say-so and `end_time` is reported as unknown, because a fraction
    against a number the run will not reach is worse than no fraction at all.
    """
    body = _COMMENTS.sub("", text or "")
    found: dict[str, Any] = {}
    depth = 0
    for line in body.splitlines():
        if depth == 0:
            entry = _ENTRY.match(line)
            if entry:
                found[entry.group(1)] = entry.group(2).strip()
        depth += line.count("{") - line.count("}")
        depth = max(0, depth)

    parsed: dict[str, Any] = {}
    for key in CONTROL_NUMBERS:
        value = _as_float(found.get(key))
        if value is not None:
            parsed[_snake(key)] = value
    for key in CONTROL_WORDS:
        if key in found:
            parsed[_snake(key)] = found[key]
    if "adjustTimeStep" in found:
        parsed["adjust_time_step"] = found["adjustTimeStep"].lower() in TRUTHY

    stop_at = parsed.get("stop_at")
    if stop_at and stop_at != "endTime" and "end_time" in parsed:
        parsed["end_time_note"] = f"stopAt is {stop_at}, so endTime is not the end"
        parsed["end_time_declared"] = parsed.pop("end_time")
    return parsed


def read_control_dict(case: Path) -> dict[str, Any]:
    """`parse_control_dict` on the case's file, or an empty dict if there is none."""
    path = Path(case) / "system" / "controlDict"
    try:
        return parse_control_dict(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return {}


def _as_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _snake(key: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()


# -- finding the log ---------------------------------------------------------------


def tail_text(path: Path, limit: int = TAIL_BYTES) -> str:
    """The last `limit` bytes of a file, minus the partial line at the front.

    Reading the tail rather than the file is what makes this callable in a loop: a
    solve that has been running for hours has a log measured in tens of megabytes and
    the interesting part of it is always the end.
    """
    path = Path(path)
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > limit:
                handle.seek(size - limit)
            raw = handle.read()
    except OSError:
        return ""
    text = raw.decode("utf-8", errors="replace")
    if size > limit and "\n" in text:
        text = text.split("\n", 1)[1]
    return text


def looks_like_solver_log(path: Path) -> bool:
    """Whether a file's tail has a solver's fingerprint on it.

    Checked by content and not by name: `log.blockMesh`, `log.checkMesh` and
    `log.snappyHexMesh` all sit in the same directory with the same shape of name,
    and none of them has a time loop to report on.
    """
    text = tail_text(path, 16_000)
    if log_digest.SOLVING.search(text):
        return True
    return any(log_digest.TIME.match(line) for line in text.splitlines())


def candidate_logs(case: Path) -> list[Path]:
    """Files in the case (and its `logs/`) that could be a log, newest first."""
    case = Path(case)
    found: list[Path] = []
    for directory in (case, case / "logs"):
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_file():
                continue
            name = entry.name
            if name.startswith("log") or name.endswith(".log") or name.endswith(".out"):
                found.append(entry)
    found.sort(key=lambda p: _mtime(p), reverse=True)
    return found


def find_solver_log(case: Path, application: str = "") -> Path | None:
    """The log the solve is writing, or None.

    `log.<application>` is taken on its name alone when `controlDict` names an
    application: that name came from the case, so the file is the solver's log even
    in the first seconds before any `Time =` has been printed -- and those first
    seconds are exactly when somebody asks what is happening. Requiring a fingerprint
    here would answer "no solver log found" while the log sits in the directory.

    Every other candidate has to earn it by content, because `log.blockMesh`,
    `log.checkMesh` and `log.snappyHexMesh` name themselves the same way and none of
    them has a time loop to report on.
    """
    case = Path(case)
    if application:
        named = case / f"log.{application}"
        if named.is_file():
            return named
    for path in candidate_logs(case):
        if looks_like_solver_log(path):
            return path
    return None


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


# -- pace: (Time, ExecutionTime) pairs ---------------------------------------------


def pace_points(text: str) -> list[tuple[float, float]]:
    """One `(Time, ExecutionTime)` pair per completed step in `text`.

    `log_digest` keeps the times and the last ExecutionTime, which answers "how far"
    but not "how fast" -- the pace needs the two side by side. The pairing is the one
    thing added here: a `Time =` line opens a step and the `ExecutionTime =` line
    closes it, so a step that is half written (the solve is in the middle of it right
    now) contributes nothing and cannot drag the rate down.
    """
    points: list[tuple[float, float]] = []
    pending: float | None = None
    for line in text.splitlines():
        match = log_digest.TIME.match(line)
        if match:
            pending = float(match.group(1))
            continue
        match = log_digest.EXEC_TIME.search(line)
        if match and pending is not None:
            points.append((pending, float(match.group(1))))
            pending = None
    return points


def looks_steady(times: list[float]) -> bool:
    """Whether `Time` is counting iterations rather than seconds.

    A steady solver (`simpleFoam`, `potentialFoam`) numbers its outer iterations 1, 2,
    3 and prints them in the same place a transient prints 0.0002. Whole numbers a
    step apart is the whole test; it decides only which unit the ETA is quoted in.
    """
    tail = [float(t) for t in times[-20:]]
    if len(tail) < 2:
        return False
    if not all(value.is_integer() for value in tail):
        return False
    return all(b - a == 1.0 for a, b in zip(tail, tail[1:]))


def no_eta(why: str, unit: str = "simulated second") -> dict[str, Any]:
    """An ETA that was not computed, in the shape one that was computed has.

    Every caller and every `--json` reader sees the same keys whether or not there
    was anything to estimate from, so a consumer can read `seconds_remaining` without
    first working out which of two shapes it was handed.
    """
    return {
        "basis": None, "unit": unit, "rate_seconds_per_unit": None,
        "units_remaining": None, "seconds_remaining": None,
        "confidence": "none", "window_points": 0, "why": why,
    }


def estimate_eta(
    points: list[tuple[float, float]],
    target: float | None,
    *,
    steady: bool = False,
    adjustable_dt: bool = False,
    window: int | None = None,
) -> dict[str, Any]:
    """Wall-clock left, the basis it was computed on, and how much to trust it.

    The arithmetic is the same either way -- wall seconds divided by units covered,
    multiplied by units remaining -- and the honest part is the label. Per iteration
    on a steady run is a rate that barely moves. Per simulated second on a transient
    with an adjustable step is a rate that changes whenever the Courant limit does, so
    the confidence is capped and the reason is written out rather than left for the
    reader to remember.
    """
    unit = "iteration" if steady else "simulated second"
    basis = f"wall-clock per {unit}"
    blank = no_eta("", unit)

    if len(points) < 2:
        blank["why"] = "fewer than two completed steps in the log"
        return blank

    size = max(2, min(window or MAX_WINDOW, len(points)))
    used = points[-size:]
    span_units = used[-1][0] - used[0][0]
    span_wall = used[-1][1] - used[0][1]
    if span_units <= 0:
        blank["why"] = f"the reported time did not advance over the last {len(used)} steps"
        return blank
    if span_wall <= 0:
        blank["why"] = "ExecutionTime did not advance, so there is no wall-clock to divide"
        return blank

    rate = span_wall / span_units
    rates = [
        (b[1] - a[1]) / (b[0] - a[0])
        for a, b in zip(used, used[1:])
        if b[0] > a[0] and b[1] >= a[1]
    ]
    spread = 0.0
    if len(rates) >= 2:
        mean = statistics.fmean(rates)
        if mean > 0:
            spread = statistics.pstdev(rates) / mean

    if len(used) >= 8 and spread <= 0.15:
        confidence = "high"
    elif len(used) >= 4 and spread <= 0.40:
        confidence = "medium"
    else:
        confidence = "low"

    reasons = [f"{len(used)} steps in the window", f"pace varied by {spread * 100:.0f}%"]
    if adjustable_dt:
        reasons.append("adjustTimeStep is on, so the step size and the pace can still change")
        if confidence == "high":
            confidence = "medium"

    result = dict(blank)
    result.update({
        "basis": basis,
        "rate_seconds_per_unit": rate,
        "confidence": confidence,
        "window_points": len(used),
        "why": "; ".join(reasons),
    })

    if target is None:
        result["why"] = "; ".join([*reasons, "no end is known, so only the rate is reported"])
        return result

    remaining = max(0.0, float(target) - used[-1][0])
    result["units_remaining"] = remaining
    result["seconds_remaining"] = remaining * rate
    if remaining == 0.0:
        result["why"] = "; ".join([*reasons, "the log has already reached the target"])
    return result


# -- residual trend ----------------------------------------------------------------


def geometric_mean(values: list[float]) -> float | None:
    """The geometric mean of positive values, ignoring the rest.

    Geometric and not arithmetic because a residual trace lives on a log axis: the
    thing that matters is the ratio between two windows, and one 1e-1 outlier would
    otherwise swamp fifty 1e-6 samples and report a rise that is not there.
    """
    logs = [math.log10(value) for value in values if isinstance(value, (int, float)) and value > 0]
    if not logs:
        return None
    return 10.0 ** statistics.fmean(logs)


def trend_of(values: list[float], window: int) -> dict[str, Any]:
    """One field's recent window against the window before it.

    The word is a description of two numbers that are also printed, not a verdict:
    "falling" says the geometric mean went down, and nothing about whether it went
    down far enough for this case.
    """
    size = min(max(1, window), len(values) // 2)
    if size < 2:
        # Two different reasons land here and they are not interchangeable: a series
        # too short to halve, and a `--window` too small to compare two windows of.
        why = (
            f"fewer than four steps recorded ({len(values)})"
            if len(values) < 4
            else f"a window of {window} is too short to compare two windows"
        )
        return {"trend": "unknown", "why": why, "window": size}
    recent = geometric_mean(values[-size:])
    earlier = geometric_mean(values[-2 * size:-size])
    if recent is None or earlier is None:
        return {"trend": "unknown", "why": "no positive residuals in the window", "window": size}
    decades = math.log10(recent / earlier)
    if decades <= -FLAT_DECADES:
        word = "falling"
    elif decades >= FLAT_DECADES:
        word = "rising"
    else:
        word = "flat"
    return {
        "trend": word,
        "earlier": earlier,
        "recent": recent,
        "decades": decades,
        "window": size,
    }


def residual_trend(residuals: dict[str, list], window: int | None = None) -> dict[str, Any]:
    """Every field's trend, plus one word for all of them together.

    `mixed` is its own answer and not a rounding of the others: pressure creeping up
    while velocity falls is a different situation from everything falling, and
    collapsing the two would hide the only interesting part.
    """
    longest = max((len(series) for series in residuals.values()), default=0)
    requested = window or min(MAX_WINDOW, max(3, longest // 4))
    fields = {
        field: trend_of([value for _step, value in series], requested)
        for field, series in sorted(residuals.items())
    }
    words = {row["trend"] for row in fields.values() if row["trend"] != "unknown"}
    if not words:
        overall = "unknown"
    elif len(words) == 1:
        overall = words.pop()
    else:
        overall = "mixed"
    # The window that was asked for and the one there was room for are not always the
    # same -- a series of twelve steps cannot be split into two windows of ten. What
    # is reported is what was used, because the sentence built from it names it.
    used = max(
        (row.get("window", 0) for row in fields.values() if row["trend"] != "unknown"),
        default=0,
    )
    return {
        "overall": overall,
        "fields": fields,
        "window": used or requested,
        "window_requested": requested,
    }


# -- frames ------------------------------------------------------------------------


def write_times(case: Path) -> list[float]:
    """The case's write times, from the names of its time directories.

    Read from the directory names and not from a reader, because opening the case in
    pyvista costs seconds and this has to stay cheap. `constant`, `system` and the
    `processor*` directories are not numbers and fall out by themselves.
    """
    times: list[float] = []
    try:
        entries = list(Path(case).iterdir())
    except OSError:
        return times
    for entry in entries:
        if entry.is_dir() and _NUMBER.match(entry.name):
            times.append(float(entry.name))
    return sorted(times)


def read_frame_sidecar(directory: Path) -> int | None:
    """The frame count a renderer wrote down next to its frames, if it wrote one.

    Read tolerantly -- any of several filenames, any of several key names -- because
    the sidecar is a courtesy from whatever produced the frames and not a format this
    script gets to impose. When there is none, the count is unknown here and the
    caller falls back to the write times.
    """
    directory = Path(directory)
    candidates = [
        directory / "frames.json",
        directory / ".frames.json",
        directory / "manifest.json",
        directory.with_name(directory.name + ".json"),
    ]
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        for key in ("expected", "n_frames", "frames", "count", "total"):
            value = data.get(key)
            if isinstance(value, list):
                return len(value)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
    return None


def find_frame_dirs(root: Path, max_depth: int = FRAME_DEPTH) -> list[Path]:
    """Frame directories under `root`, without walking the whole tree.

    A study directory contains a case, and a case contains a time directory per write
    and possibly one per processor; an unbounded `rglob` there is the opposite of
    cheap. So the walk is depth-limited and skips the directories that are known not
    to hold frames.
    """
    found: list[Path] = []

    def walk(directory: Path, depth: int) -> None:
        try:
            entries = sorted(p for p in directory.iterdir() if p.is_dir())
        except OSError:
            return
        for entry in entries:
            name = entry.name
            if name.endswith("_frames") or name == "frames":
                found.append(entry)
                continue
            if depth >= max_depth:
                continue
            if name.startswith(".") or name.startswith("processor") or name in SKIP_DIRS:
                continue
            if _NUMBER.match(name):
                continue
            walk(entry, depth + 1)

    walk(Path(root), 0)
    return found


def frame_progress(directory: Path, fallback_expected: int | None = None) -> dict[str, Any]:
    """PNGs on disk in one frame directory, against however many are expected."""
    directory = Path(directory)
    try:
        count = sum(1 for entry in directory.iterdir() if entry.suffix.lower() == ".png")
    except OSError:
        count = 0
    expected = read_frame_sidecar(directory)
    source = "sidecar"
    if expected is None:
        expected = fallback_expected if fallback_expected else None
        source = "write times" if expected else "unknown"
    row: dict[str, Any] = {
        "dir": str(directory),
        "count": count,
        "expected": expected,
        "expected_from": source,
        "fraction": None,
        "newest_age_seconds": None,
    }
    if expected:
        row["fraction"] = min(1.0, count / expected)
    newest = 0.0
    try:
        for entry in directory.iterdir():
            if entry.suffix.lower() == ".png":
                newest = max(newest, _mtime(entry))
    except OSError:
        pass
    if newest:
        row["newest_age_seconds"] = max(0.0, time.time() - newest)
    return row


# -- previews ----------------------------------------------------------------------


def previews(root: Path) -> list[dict[str, Any]]:
    """The newest artifact of each kind that is still on disk, as absolute paths.

    Newest per kind rather than everything: after a long session the manifest holds
    six versions of the same slice, and "can I see it?" wants the current one. The
    older rows are still in the manifest for anyone who wants them.
    """
    newest: dict[str, dict[str, Any]] = {}
    for row in study_state.artifacts(root=root, exists=True):
        if str(row.get("kind", "")) == study_state.RUNG_KIND:
            continue  # rung evidence is not a picture; it gets its own section
        newest[str(row.get("kind", "other"))] = row
    ordered = sorted(newest.items(), key=lambda item: study_state.KINDS.index(item[0])
                     if item[0] in study_state.KINDS else len(study_state.KINDS))
    return [
        {
            "kind": kind,
            "path": row.get("abspath", row.get("path", "")),
            "label": row.get("label", ""),
            "case": row.get("case", ""),
            "at": row.get("at", ""),
        }
        for kind, row in ordered
    ]


def ladder_evidence(root: Path) -> list[dict[str, Any]]:
    """Rung outcomes recorded by `ladder.py --record`, one row per class and rung.

    Read from the same manifest as the previews, and surfaced for the same reason:
    a rung an earlier session climbed is a fact the study already paid for, and a
    fresh session that does not see it re-derives it at full price. Rows are facts,
    not advice -- what a recorded fail means for today's plan is the reader's call.
    """
    rows: list[dict[str, Any]] = []
    for row in study_state.rung_evidence(root=root):
        meta = row.get("meta") or {}
        rows.append({
            "class": str(meta.get("class", "")),
            "rung": meta.get("rung"),
            "name": str(meta.get("name", "")),
            "status": str(meta.get("status", "")),
            "value": meta.get("value"),
            "note": str(meta.get("note", "")),
            "case": str(row.get("case", "")),
            "at": str(row.get("at", "")),
        })
    return rows


# -- the phase table ---------------------------------------------------------------


def current_phase(root: Path) -> dict[str, Any]:
    """Which phase the study says it is in.

    A phase marked `running` wins over the first unsettled one: a solve that has been
    started and a solve that has not are both "not done", and the table is the only
    place that knows which of the two this is.
    """
    table = study_state.load_phases(root)
    rows = [row for row in table.get("phases", []) if isinstance(row, dict)]
    running = [row for row in rows if row.get("status") == "running"]
    failed = [row for row in rows if row.get("status") == "failed"]
    chosen = running[-1] if running else (failed[-1] if failed else None)
    upcoming = study_state.next_phase(root)
    if chosen is None:
        chosen = next((row for row in rows if row.get("name") == upcoming), None)
    return {
        "current": str((chosen or {}).get("name") or ""),
        "status": str((chosen or {}).get("status") or "pending"),
        "note": str((chosen or {}).get("note") or ""),
        "next_unsettled": upcoming,
        "case": str(table.get("case") or ""),
        "updated_at": str(table.get("updated_at") or ""),
        "table": [
            {"name": row.get("name", ""), "status": row.get("status", ""), "note": row.get("note", "")}
            for row in rows
        ],
    }


# -- putting it together -----------------------------------------------------------


def collect(case: Path, *, log: Path | None = None, window: int | None = None) -> dict[str, Any]:
    """Every fact this reports, as plain data. The text and the JSON are two renderings
    of this one dict, so they can never disagree about what was read."""
    case = Path(case).resolve()
    root = study_state.find_root(case)
    control = read_control_dict(case)
    notes: list[str] = []

    report: dict[str, Any] = {
        "case": str(case),
        "case_name": case.name,
        "study": str(root),
        "study_name": root.name,
        "generated_at": study_state.now_iso(),
        "phase": current_phase(root),
        "control": control,
        "solve": {},
        "residuals": {"overall": "unknown", "fields": {}, "window": 0, "window_requested": 0},
        "eta": no_eta("no solver log was read"),
        "frames": [],
        "previews": previews(root),
        "ladder": ladder_evidence(root),
        "notes": notes,
    }

    if "end_time_note" in control:
        notes.append(control["end_time_note"])

    solver_log = Path(log) if log else find_solver_log(case, str(control.get("application", "")))
    if solver_log is None or not Path(solver_log).is_file():
        report["solve"] = {"log": None, "why": f"no solver log found in {case}"}
        report["eta"] = no_eta("no solver log found")
        report["frames"] = _frames_for(root, case)
        return report

    solver_log = Path(solver_log)
    data = log_digest.digest(solver_log)
    times = data["times"]
    steady = looks_steady(times)
    target = control.get("end_time")

    solve: dict[str, Any] = {
        "log": str(solver_log),
        "log_age_seconds": max(0.0, time.time() - _mtime(solver_log)),
        "application": control.get("application", ""),
        "steady": steady,
        "unit": "iteration" if steady else "simulated second",
        "steps": len(times),
        "time": times[-1] if times else None,
        "first_time": times[0] if times else None,
        "start_time": control.get("start_time"),
        "end_time": target,
        "stop_at": control.get("stop_at", ""),
        "adjust_time_step": bool(control.get("adjust_time_step", False)),
        "exec_time": data["exec_time"],
        "courant": None,
        "continuity": None,
        "bounding": data["bounding"],
        "fraction": None,
    }
    if data["courant"]:
        solve["courant"] = {"mean": data["courant"][0], "max": data["courant"][1]}
    if data["continuity"]:
        local, glob, cumulative = data["continuity"]
        solve["continuity"] = {"sum_local": local, "global": glob, "cumulative": cumulative}

    if times and target is not None:
        origin = control.get("start_time")
        if origin is None or origin > times[0]:
            origin = times[0]
        span = float(target) - float(origin)
        if span > 0:
            solve["fraction"] = min(1.0, max(0.0, (times[-1] - origin) / span))

    if solve["log_age_seconds"] > 120 and report["phase"]["status"] == "running":
        notes.append(
            f"the log has not been written to for {duration(solve['log_age_seconds'])}, "
            "while the phase table still says running"
        )

    report["solve"] = solve
    report["residuals"] = residual_trend(data["residuals"], window)
    report["eta"] = estimate_eta(
        pace_points(tail_text(solver_log)),
        target,
        steady=steady,
        adjustable_dt=bool(control.get("adjust_time_step", False)),
        window=window,
    )
    report["frames"] = _frames_for(root, case, len(write_times(case)))
    return report


def _frames_for(root: Path, case: Path, expected: int | None = None) -> list[dict[str, Any]]:
    if expected is None:
        expected = len(write_times(case))
    directories = find_frame_dirs(root)
    for extra in find_frame_dirs(Path(case).parent, max_depth=1):
        if extra not in directories:
            directories.append(extra)
    return [frame_progress(directory, expected) for directory in sorted(set(directories))]


# -- rendering ---------------------------------------------------------------------


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.0f} min"
    hours = minutes / 60.0
    if hours < 36:
        whole = int(hours)
        rest = round(minutes - whole * 60)
        if rest >= 60:  # 2.999 h is two hours and sixty minutes, which nobody says
            whole, rest = whole + 1, 0
        return f"{whole} h {rest} min"
    days = int(hours // 24)
    rest = round(hours - days * 24)
    if rest >= 24:
        days, rest = days + 1, 0
    return f"{days} d {rest} h"


def _row(label: str, text: str, width: int = 12) -> str:
    return f"{label:<{width}}{text}"


def _cont(text: str, width: int = 12) -> str:
    return f"{'':<{width}}{text}"


def _wrapped(text: str, width: int = 12, columns: int = 96) -> list[str]:
    """A sentence wrapped into the continuation column. Only prose goes through here:
    a path wrapped across two lines is a path nobody can copy."""
    body = textwrap.wrap(text, width=max(20, columns - width)) or [""]
    return [_cont(line, width) for line in body]


def render_text(report: dict[str, Any]) -> str:
    """The report as something to read aloud. Facts and the arithmetic behind them;
    nothing here says whether any of it is good news."""
    lines: list[str] = []
    phase = report["phase"]
    lines.append(f"progress: {report['study_name']}/{report['case_name']}    {report['generated_at']}")
    upcoming = phase["next_unsettled"] or "none, every phase is settled"
    lines.append(_row(
        "phase", f"{phase['current'] or '-'} ({phase['status']})    first unsettled phase: {upcoming}"
    ))
    if phase["note"]:
        lines.append(_cont(phase["note"]))

    solve = report["solve"]
    lines.append("")
    if not solve.get("log"):
        lines.append(_row("solve", solve.get("why", "nothing read")))
    else:
        age = duration(solve.get("log_age_seconds"))
        lines.append(_row("solve", f"{Path(solve['log']).name}, last written {age} ago"))
        shape = "steady (Time counts iterations)" if solve["steady"] else "transient"
        application = solve.get("application") or "solver not named in controlDict"
        adjust = ", adjustTimeStep on" if solve["adjust_time_step"] else ""
        lines.append(_cont(f"{application}, {shape}{adjust}"))
        if solve["time"] is None:
            lines.append(_cont("no Time = line in the log yet: the time loop has not started"))
        else:
            unit = "iterations" if solve["steady"] else "time steps"
            end = f" of endTime {solve['end_time']:g}" if solve["end_time"] is not None else ""
            fraction = f"    {solve['fraction'] * 100:.0f}%" if solve["fraction"] is not None else ""
            lines.append(_cont(f"Time = {solve['time']:g}{end}{fraction}    ({solve['steps']} {unit})"))
        if solve.get("exec_time") is not None:
            lines.append(_cont(f"ExecutionTime {duration(solve['exec_time'])}"))
        if solve.get("courant"):
            lines.append(_cont(f"Courant  mean {solve['courant']['mean']:g}  max {solve['courant']['max']:g}"))
        if solve.get("continuity"):
            cont = solve["continuity"]
            lines.append(_cont(
                f"continuity  sum local {cont['sum_local']:.3e}  "
                f"global {cont['global']:.3e}  cumulative {cont['cumulative']:.3e}"
            ))
        if solve.get("bounding"):
            summary = ", ".join(f"{field} x{count}" for field, count in sorted(solve["bounding"].items()))
            lines.append(_cont(f"bounding messages  {summary}"))

    residuals = report["residuals"]
    lines.append("")
    if not residuals["fields"]:
        lines.append(_row("residuals", "none parsed"))
    else:
        lines.append(_row(
            "residuals",
            f"{residuals['overall']} overall, last {residuals['window']} steps "
            f"against the {residuals['window']} before them",
        ))
        for field, row in residuals["fields"].items():
            if row["trend"] == "unknown":
                lines.append(_cont(f"{field:<6} {row.get('why', 'not enough data')}"))
                continue
            lines.append(_cont(
                f"{field:<6} {row['earlier']:.3e} -> {row['recent']:.3e}   "
                f"{row['trend']} ({row['decades']:+.2f} decades)"
            ))

    eta = report["eta"]
    lines.append("")
    if not eta.get("basis"):
        lines.append(_row("eta", f"not estimated -- {eta.get('why', 'no basis')}"))
    else:
        if eta.get("seconds_remaining") is None:
            lines.append(_row("eta", "no end to count down to; the rate is below"))
        else:
            unit = eta["unit"]
            plural = f"{eta['units_remaining']:g} {unit}" + ("s" if eta["units_remaining"] != 1 else "")
            lines.append(_row("eta", f"about {duration(eta['seconds_remaining'])} left ({plural} to go)"))
        lines.append(_cont(
            f"basis: {eta['rate_seconds_per_unit']:.4g} s of wall-clock per {eta['unit']}"
        ))
        lines.extend(_wrapped(f"confidence {eta['confidence']} -- {eta['why']}"))

    lines.append("")
    if not report["frames"]:
        lines.append(_row("frames", "no *_frames/ directory yet"))
    else:
        for index, row in enumerate(report["frames"]):
            expected = f" of {row['expected']} (from {row['expected_from']})" if row["expected"] else ""
            age = f", newest {duration(row['newest_age_seconds'])} old" if row["newest_age_seconds"] else ""
            text = f"{row['count']} PNG{'' if row['count'] == 1 else 's'}{expected}{age}"
            lines.append(_row("frames", text) if index == 0 else _cont(text))
            lines.append(_cont(f"  {row['dir']}"))

    lines.append("")
    if not report["previews"]:
        lines.append(_row("previews", "nothing registered in the manifest yet"))
    else:
        width = max(len(row["kind"]) for row in report["previews"])
        for index, row in enumerate(report["previews"]):
            label = f"   {row['label']}" if row["label"] else ""
            text = f"{row['kind']:<{width}}  {row['path']}{label}"
            lines.append(_row("previews", text) if index == 0 else _cont(text))

    ladder = report.get("ladder") or []
    if ladder:
        lines.append("")
        for index, row in enumerate(ladder):
            value = ""
            if row.get("value") is not None:
                shown = row["value"]
                value = f"   value {shown:g}" if isinstance(shown, (int, float)) else f"   value {shown}"
            note = f"   {row['note']}" if row.get("note") else ""
            text = (
                f"rung {row['rung']} ({row['name']})  {row['status']}{value}"
                f"   {row['at']}{note}"
            )
            lines.append(_row("ladder", text) if index == 0 else _cont(text))

    if report["notes"]:
        lines.append("")
        for index, note in enumerate(report["notes"]):
            wrapped = _wrapped(note)
            first = wrapped[0].strip()
            lines.append(_row("notes", first) if index == 0 else _cont(first))
            lines.extend(wrapped[1:])

    return "\n".join(lines)


# -- the command line --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("case", type=Path, help="The case directory (or anything inside the study).")
    parser.add_argument("--json", action="store_true", help="The same facts as JSON.")
    parser.add_argument("--log", type=Path, default=None, help="Read this log instead of looking for one.")
    parser.add_argument(
        "--window", type=int, default=None,
        help=f"Steps in the recent window for the trend and the pace (default up to {MAX_WINDOW}).",
    )
    args = parser.parse_args(argv)

    report = collect(args.case, log=args.log, window=args.window)
    print(json.dumps(report, indent=2, default=str) if args.json else render_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
