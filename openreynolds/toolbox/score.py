#!/usr/bin/env python3
"""One candidate of a parametric study scored into `metrics.json` -- the only writer of it.

    python3 score.py CASE_DIR --lock goal.lock.json --out metrics.json
    python3 score.py CASE_DIR --lock goal.lock.json --out metrics.json --mesh-only
    python3 score.py CASE_DIR --lock goal.lock.json --out metrics.json --label timeout --detail "killed at 3x median"

`parametric.py` runs this last, after the build, the mesh, the dressing and the solve,
and nothing else writes the file: a `hooks.py` cannot leave a number behind that ends up
in the leaderboard, because this reads the case -- `postProcessing/`, the solver log, the
mesh -- and the lock, and nothing a hook wrote. What a candidate is worth is decided in
one place, on measurements, the same way for every point of the study.

what it measures, and how

    A pressure drop is the mean over the same window of `p_inlet - p_outlet`, read off
    the two `surfaceFieldValue` records `case_gen.py --dp` leaves
    (`postProcessing/pInlet/`, `postProcessing/pOutlet/`), aligned by time so a restart
    cannot subtract two different iterations. It is reported as `dp`, in the solver's
    own pressure units.

    Force coefficients are the mean over the last `fidelity.window` fraction of the
    `forceCoeffs.dat` rows (default the last 20%), never the last row: on a steady
    solver a bluff section's coefficients oscillate about their mean, and a point read
    is whichever phase the final write landed on. `window` in the output is the number
    of rows that mean was taken over, so a ten-row window on a 1000-iteration run is
    visible as one. `converged` is `log_digest.residual_shape` reading `levelled` or
    `falling`, or the solver's own "solution converged" line. y+ comes from the `yPlus`
    function object's output when the case has one (`case_gen.py --yplus`), the body
    patch's own row at the last write, min and max. The cell count is read off
    `constant/polyMesh/owner`'s header, which is the mesh's own statement of it.

a transient case

    A lock whose fidelity says `solver_mode: "transient"` (with `end_time`,
    `write_interval` and `average_from` in convective times H/U, `transient_of`) is a
    shedding body, and its coefficients are a mean over TIME from `average_from` to the
    end of the record -- a trapezoid, because the adjustable step writes uneven rows --
    not over the last rows. Two more metrics come with it: `St`, the lift's dominant
    frequency on the lock's `--length` and `--speed`, and `Cd_rms`, the drag's rms about
    its mean. `converged` is then not the residual shape (a shedding run's residuals
    oscillate with the flow and never level) but whether the window spans
    `fidelity.min_periods` (default 5) periods of that lift; a shorter one is
    `unconverged` with the count it saw, and `shedding` in the output carries the
    frequency, the period count and the window. A lift that is not shedding -- its rms
    under `fidelity.min_lift_rms` (default `MIN_LIFT_RMS`), or crossings that say a
    Strouhal number no bluff body has -- counts no periods at all, so solver noise on a
    symmetric wake is `unconverged`, not a frequency. A steady lock never sees any of this.

the label

    One word the optimizer reads, in this order of precedence, the first that applies:
    `mesh` (no polyMesh, or `checkMesh` failed), `diverged` (a FOAM FATAL in the log, a
    residual climbing off its own best, or a field held at its bound for most of the
    run), `unconverged` (a residual shape that is neither levelled nor falling, or no
    residuals at all), `yplus` (outside `fidelity.yplus`), `cells` (outside
    `fidelity.cells`), else `ok`. `timeout` is never decided here -- this cannot know how
    long a run was allowed -- so the caller that killed the run passes `--label timeout`,
    and a stated label wins over everything measured.

several cases per candidate

    A lock may declare `cases` -- `[{"name": "forward", "case_gen_args": [...]},
    {"name": "reverse", "case_gen_args": [...]}]` -- and then one candidate is several
    solves off one mesh, each in `CANDIDATE_DIR/<name>/`, and this writes ONE
    `metrics.json` for the candidate. Every case is scored on its own and its metrics
    arrive namespaced: `forward.dp`, `reverse.dp`, `forward.Cd`. The candidate's label
    is the worst of theirs (`SEVERITY`), so a diverged reverse solve is not averaged
    away by a converged forward one, and `cases` in the output carries each case's own
    label, detail, window and case_gen arguments.

    `derived` in the lock names metrics computed from those: `{"diodicity":
    "reverse.dp / forward.dp"}`. The expression is arithmetic over metric names and
    numbers -- `+ - * /`, parentheses, nothing else, checked as a syntax tree rather
    than trusted to `eval` -- because the lock is data that arrives from a goal file and
    an objective is not a place to run code. A derived metric whose inputs are missing,
    or whose arithmetic divides by zero, is left out rather than written as a number.

The schema is a contract with whatever reads the leaderboard; the keys stay.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import log_digest  # noqa: E402
import results  # noqa: E402

LABELS = ("ok", "mesh", "diverged", "timeout", "unconverged", "yplus", "cells")
"""Every label this can write. `timeout` only ever arrives through `--label`."""

SEVERITY = ("mesh", "diverged", "timeout", "unconverged", "yplus", "cells", "ok")
"""The labels worst first, which is how a multi-case candidate gets one: the worst of
its cases. Same order as the precedence a single case is decided by, so "the candidate's
label" and "the label this case would have had on its own" mean the same thing."""

DEFAULT_WINDOW = 0.2
"""The fraction of the force record the coefficients are averaged over when the lock
does not say. `results.py` quotes the last quarter; a design study wants the flatter
tail of a run that was sized to converge, so a fifth."""

DP_FUNCTIONS = ("pInlet", "pOutlet")
"""The two function objects `case_gen.py --dp` writes (`PRESSURE_FUNCTIONS` there), in
the order the drop subtracts them: `dp = p_inlet - p_outlet`."""

COEFFICIENTS = ("Cd", "Cl", "CmPitch")
"""The whole-body coefficients reported under their forceCoeffs names. `Cm` is what
the same moment was called before v2006 and is reported as `CmPitch` when that is the
column the file has."""

BOUND_FRACTION = 0.5
"""A field whose `bounding` message appears on at least this fraction of the parsed
time steps, and whose clip is material (`BOUND_DEPTH`), is being held at its bound
rather than solved for, and that is a run that is failing slowly. Early bounding on k
or omega is ordinary and is not this."""

BOUND_DEPTH = 0.1
"""How far below zero the field is being clipped from, as a fraction of its mean
(`log_digest`'s `bounding_depth`: the median over the run of `|min| / average`),
before the count above means anything.

The count alone labelled a converged case diverged. The ground-effect section at its
baseline (2026-09-22, 26,004 cells, 1500 iterations of simpleFoam on 4 ranks): every
residual at 1e-5 and flat, the force coefficients steady to six figures, and
`bounding k, min: -9e-4 max: 1.98 average: 0.094` on 1355 of the 1500 steps -- a
handful of cells at the trailing edge dipping under zero by half a percent of the
field's mean. That is k-omega SST's ordinary housekeeping and would have been on
every candidate of the study. A field genuinely held at its bound reads the other
way -- `bounding omega, min: -5.6e3 max: 8.1e6 average: 22` -- a clip many times the
mean. One tenth of the mean is a decade above the healthy case and two below the
failing ones."""

MIN_PERIODS = 5
"""How many shedding periods a transient case's averaging window must span, when the
lock does not say (`fidelity.min_periods`). A mean over a fraction of a cycle is a
phase, not a coefficient; over two it is still a few percent off a bluff body's Cd from
where the window happened to start and stop. Five is the usual floor for quoting one."""

HYSTERESIS = 0.25
"""The lift must swing this many rms fluctuations below its mean before an upward
crossing counts (`shedding_frequency`): solver noise about the mean is not a cycle."""

MIN_LIFT_RMS = 1e-2
"""The smallest rms fluctuation of Cl about its mean that is a shedding wake, when the
lock does not say (`fidelity.min_lift_rms`). The hysteresis is relative to the signal's
own rms, so on its own it counts solver noise on a symmetric wake's lift as thousands of
cycles; this floor is absolute. A shedding bluff body's lift swings by tenths (a plate's
rms Cl is of order 1, a cylinder's at Re 1e5 about 0.3 on URANS); a wake that has not
broken symmetry carries 1e-6 of round-off. A hundredth sits two decades from either."""

MAX_STROUHAL = 1.0
"""The highest lift frequency, as a Strouhal number on the lock's length and speed, that
is read as shedding. Bluff bodies shed at 0.1-0.3 on their frontal height; a frequency
that says St 3 is noise loud enough to clear the floor, crossing its mean every few rows."""


# -- the pieces --------------------------------------------------------------------


def load_lock(path: Path) -> dict:
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(lock, dict):
        raise SystemExit(f"{path} is not a JSON object")
    return lock


def fidelity_of(lock: dict) -> dict:
    fidelity = lock.get("fidelity") or {}
    return dict(fidelity) if isinstance(fidelity, dict) else {}


def band(spec) -> tuple[float | None, float | None]:
    """`{"min": a, "max": b}` as a pair, either side optional and None when absent."""
    if not isinstance(spec, dict):
        return (None, None)
    low, high = spec.get("min"), spec.get("max")
    return (float(low) if low is not None else None, float(high) if high is not None else None)


def outside(value, limits: tuple[float | None, float | None]) -> bool:
    low, high = limits
    if value is None:
        return False
    return (low is not None and value < low) or (high is not None and value > high)


def cell_count(case: Path) -> int:
    """Cells, from the `note` line in `constant/polyMesh/owner`'s header.

    OpenFOAM writes `nCells:` there in every format, ascii or binary, so the count is
    read without a reader and without `checkMesh`. `look.json` from `mesh_look.py` is
    the fallback for a mesh whose owner file has no note.
    """
    owner = case / "constant" / "polyMesh" / "owner"
    if owner.is_file():
        try:
            with owner.open("rb") as handle:
                head = handle.read(4000).decode("utf-8", errors="replace")
        except OSError:
            head = ""
        match = re.search(r"nCells:\s*(\d+)", head)
        if match:
            return int(match.group(1))
    look = case / "look.json"
    if look.is_file():
        try:
            return int(json.loads(look.read_text(encoding="utf-8")).get("cells") or 0)
        except (OSError, ValueError):
            return 0
    return 0


def mesh_verdict(case: Path) -> tuple[bool, str]:
    """(mesh present and checkMesh passed, why not).

    `look.json` is what `parametric.py` leaves from its `mesh_look.py` call and carries
    `checkmesh_ok`; without it the presence of `constant/polyMesh/points` is all this can
    say, and it says so.
    """
    if not (case / "constant" / "polyMesh" / "points").exists():
        return False, "no constant/polyMesh in the case"
    look = case / "look.json"
    if look.is_file():
        try:
            payload = json.loads(look.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False, "look.json could not be read"
        if not payload.get("checkmesh_ok"):
            return False, f"checkMesh: {payload.get('checkmesh') or 'no verdict'}"
    return True, ""


def force_metrics(case: Path, fraction: float = DEFAULT_WINDOW) -> tuple[dict, int]:
    """The whole-body coefficients averaged over the tail of the record, and the number
    of rows the average was taken over. Empty and zero when the case logged no forces."""
    grouped = results.find_force_files(case)
    if not grouped:
        return {}, 0
    series = results.choose_force_series(grouped)
    history = results.read_history(grouped[series])
    if not history["rows"]:
        return {}, 0
    columns = list(history["columns"])
    keep = max(1, int(round(len(history["rows"]) * fraction)))
    out: dict = {}
    for name in COEFFICIENTS:
        column = name
        if name not in columns and name == "CmPitch" and "Cm" in columns:
            column = "Cm"
        values = results.column(history, column)
        if values.size:
            out[name] = results.tail_mean(values, fraction)
    return out, keep


def _flag(args: list[str], flag: str) -> str | None:
    """One `case_gen.py` flag's value off an argument list, `--flag=value` included."""
    for index, item in enumerate(args):
        if item.startswith(f"{flag}="):
            return item.split("=", 1)[1]
        if item == flag and index + 1 < len(args):
            return args[index + 1]
    return None


def transient_of(lock: dict) -> dict | None:
    """The lock's transient fidelity in seconds, or None for the steady default.

    `fidelity.solver_mode: "transient"` says the answer is a time average, and the lock
    states its clock in convective times (H/U) -- `end_time`, `write_interval`,
    `average_from` -- because that is the unit a shedding period is known in (St 0.2 is
    five of them) and it does not move when a goal changes its speed. The seconds follow
    from `--length` and `--speed` in the lock's `case_gen_args`, which is why a lock
    without both is refused here rather than run on a clock nobody can read: this is
    called by `parametric.py` before anything is paid for, and by `score` after.

    `min_periods` (default `MIN_PERIODS`) is how many shedding periods the averaging
    window must span before its mean is a number, and `min_lift_rms` (default
    `MIN_LIFT_RMS`) the smallest lift fluctuation that is a shedding wake at all.
    """
    fidelity = fidelity_of(lock)
    if fidelity.get("solver_mode", "steady") == "steady":
        return None
    if fidelity["solver_mode"] != "transient":
        raise SystemExit(f"fidelity.solver_mode: {fidelity['solver_mode']!r}; steady or transient")
    args = [str(a) for a in (lock.get("case_gen_args") or [])]
    if (_flag(args, "--study") or "transient") != "transient":
        raise SystemExit("a transient fidelity with --study " + str(_flag(args, "--study"))
                         + " in case_gen_args: the lock says two different things")
    if str(lock.get("solver") or "simpleFoam") == "simpleFoam":
        raise SystemExit("a transient fidelity with the solver simpleFoam: a steady solver "
                         "has no time to average over (the lock's solver is pimpleFoam)")
    numbers = {}
    for key in ("end_time", "write_interval", "average_from"):
        value = fidelity.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise SystemExit(f"fidelity.{key}: a positive number of convective times, got {value!r}")
        numbers[key] = float(value)
    floor = fidelity.get("min_lift_rms", MIN_LIFT_RMS)
    if isinstance(floor, bool) or not isinstance(floor, (int, float)) or floor <= 0:
        raise SystemExit(f"fidelity.min_lift_rms: a positive rms of Cl, got {floor!r}")
    if numbers["average_from"] >= numbers["end_time"]:
        raise SystemExit(f"fidelity.average_from ({numbers['average_from']:g}) is not before "
                         f"end_time ({numbers['end_time']:g}): there is no window to average over")
    scale = {}
    for flag in ("--length", "--speed"):
        value = _flag(args, flag)
        if value is None:
            raise SystemExit(f"a transient fidelity needs {flag} in case_gen_args: the clock is "
                             "in convective times H/U and that is half of it")
        scale[flag] = float(value)
    convective = scale["--length"] / scale["--speed"]
    return {"convective_s": convective,
            "end_s": numbers["end_time"] * convective,
            "average_from_s": numbers["average_from"] * convective,
            "writes": max(1, int(round(numbers["end_time"] / numbers["write_interval"]))),
            "min_periods": int(fidelity.get("min_periods") or MIN_PERIODS),
            "min_lift_rms": float(floor)}


def shedding_frequency(times, values) -> float | None:
    """The dominant frequency of a signal, from its upward crossings of its own mean.

    A Schmitt trigger rather than a bare sign change: a crossing counts only when the
    signal climbs from below `-HYSTERESIS` x its rms fluctuation to above `+HYSTERESIS`
    x it, so solver noise riding on the mean is not counted as extra cycles. Zero
    crossings rather than an FFT because the adjustable time step writes uneven rows and
    a crossing needs no resampling. None under two crossings -- no period is measured.
    """
    import numpy as np

    t = np.asarray(times, dtype=float)
    x = np.asarray(values, dtype=float)
    x = x - np.mean(x)
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    if rms <= 1e-9:
        return None
    level = HYSTERESIS * rms
    ups: list[float] = []
    armed = False
    for i in range(1, x.size):
        if x[i] < -level:
            armed = True
        if armed and x[i - 1] < 0 <= x[i]:
            ups.append(t[i - 1] + (t[i] - t[i - 1]) * (-x[i - 1]) / (x[i] - x[i - 1]))
            armed = False
    if len(ups) < 2:
        return None
    return (len(ups) - 1) / (ups[-1] - ups[0])


def transient_force_metrics(case: Path, window: dict) -> tuple[dict, int, dict]:
    """(coefficients, rows in the window, the shedding record) for a transient case.

    The mean is over TIME, a trapezoid over `[average_from, end of record]`: the step is
    adjustable, so a mean over rows weighs whichever part of the cycle the Courant
    number made the step small in. `Cd_rms` is the rms of Cd about that mean, and `St`
    the lift's dominant frequency on the lock's length and speed. `periods` is the
    window's length times that frequency, which is what `score` holds to
    `min_periods`.

    A lift whose rms about its mean is under `min_lift_rms`, or whose crossings say a
    Strouhal number over `MAX_STROUHAL`, is not shedding: no `St`, no periods, and
    `why` in the record says which, for `unshed`.
    """
    import numpy as np

    shed = {"periods": 0.0, "frequency_hz": None, "window_s": None,
            "min_periods": window["min_periods"], "average_from_s": window["average_from_s"]}
    grouped = results.find_force_files(case)
    if not grouped:
        return {}, 0, shed
    history = results.read_history(grouped[results.choose_force_series(grouped)])
    if not history["rows"]:
        return {}, 0, shed
    t = np.array([row[0] for row in history["rows"]], dtype=float)
    keep = t >= window["average_from_s"]
    if keep.sum() < 2:
        shed["last_s"] = float(t[-1])
        return {}, 0, shed
    tw = t[keep]
    span = float(tw[-1] - tw[0])
    shed["window_s"] = [float(tw[0]), float(tw[-1])]

    def mean(values):
        return float(np.trapezoid(values, tw) / span) if hasattr(np, "trapezoid") \
            else float(np.trapz(values, tw) / span)

    columns = list(history["columns"])
    out: dict = {}
    for name in COEFFICIENTS:
        column = "Cm" if name == "CmPitch" and name not in columns and "Cm" in columns else name
        values = results.column(history, column)
        if values.size:
            out[name] = mean(values[keep])
    if "Cd" in out:
        cd = results.column(history, "Cd")[keep]
        out["Cd_rms"] = float(np.sqrt(mean((cd - out["Cd"]) ** 2)))
    if "Cl" in columns:
        cl = results.column(history, "Cl")[keep]
        rms = float(np.std(cl))
        shed["lift_rms"] = rms
        floor = window.get("min_lift_rms", MIN_LIFT_RMS)
        frequency = shedding_frequency(tw, cl) if rms >= floor else None
        strouhal = frequency * window["convective_s"] if frequency else None
        if rms < floor:
            shed["why"] = (f"the lift's rms fluctuation {rms:.3g} is under {floor:g}: "
                           "the wake is not shedding")
        elif strouhal is not None and strouhal > MAX_STROUHAL:
            shed["why"] = (f"the lift's crossings say St {strouhal:.3g}, over {MAX_STROUHAL:g}: "
                           "noise, not shedding")
        elif frequency:
            shed["frequency_hz"] = frequency
            shed["periods"] = span * frequency
            out["St"] = strouhal
    return out, int(keep.sum()), shed


def pressure_series(case: Path, function: str) -> dict[float, float]:
    """`{time: areaAverage(p)}` off one `surfaceFieldValue` function object's record.

    Every time directory of the object is read and merged, later writes winning, the
    way `results.read_history` merges a restarted force record. The value column is the
    one whose header names an operation on a field (`areaAverage(p)`); failing that,
    the second column, which is where a one-field object puts it.
    """
    base = case / "postProcessing" / function
    if not base.is_dir():
        return {}
    out: dict[float, float] = {}
    for time_dir in sorted((d for d in base.iterdir() if d.is_dir()), key=lambda d: results.time_key(d.name)):
        for dat in sorted(time_dir.glob("*.dat")):
            try:
                data = results.parse_dat(dat.read_text(errors="replace"))
            except OSError:
                continue
            if not data["rows"]:
                continue
            columns = list(data["columns"])
            index = next((i for i, name in enumerate(columns)
                          if i and "(" in name and ")" in name), 1)
            for row in data["rows"]:
                if len(row) > index:
                    out[float(row[0])] = float(row[index])
    return out


def pressure_drop(case: Path, fraction: float = DEFAULT_WINDOW) -> tuple[float | None, int]:
    """(`dp` over the tail of the record, rows the mean was taken over).

    `p_inlet - p_outlet` is formed per time step, on the times both records have, and
    only then averaged: subtracting two tail means would be the same number on a clean
    run and a silently wrong one on a case where the two objects wrote different
    iterations. (None, 0) when either record is missing -- a case run without
    `case_gen.py --dp` has no pressure drop, and zero is not one.
    """
    inlet, outlet = (pressure_series(case, name) for name in DP_FUNCTIONS)
    times = sorted(set(inlet) & set(outlet))
    if not times:
        return None, 0
    keep = max(1, int(round(len(times) * fraction)))
    window = times[-keep:]
    return sum(inlet[t] - outlet[t] for t in window) / len(window), len(window)


# -- derived metrics: arithmetic over metric names, checked as a tree ----------------


_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
           ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b}


class DerivedError(ValueError):
    """The expression is not arithmetic over metric names; the message says what was in it."""


def dotted(node: ast.AST) -> str | None:
    """`reverse.dp` out of the tree `Attribute(Name('reverse'), 'dp')`, or None.

    Namespaced metric names are not Python identifiers, so they parse as attribute
    access; reading them back as one dotted string is what lets the lock write
    `reverse.dp / forward.dp` and mean the two numbers `cases` put in `metrics`.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = dotted(node.value)
        return f"{head}.{node.attr}" if head else None
    return None


def derived_names(expression: str) -> list[str]:
    """Every metric name an expression reads, after refusing anything that is not
    arithmetic over names and numbers. Raises DerivedError with what it found."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise DerivedError(f"{expression!r} does not parse: {exc.msg}") from exc
    names: list[str] = []

    def walk(node: ast.AST) -> None:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise DerivedError(f"{expression!r} has the constant {node.value!r}; only numbers")
            return
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = dotted(node)
            if name is None:
                raise DerivedError(f"{expression!r} reads something that is not a metric name")
            names.append(name)
            return
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            walk(node.left)
            walk(node.right)
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            walk(node.operand)
            return
        raise DerivedError(
            f"{expression!r} is not arithmetic over metric names: it has a "
            f"{type(node).__name__}. Only names, numbers, + - * / and parentheses."
        )

    walk(tree.body)
    return names


def evaluate_derived(expression: str, values: dict[str, float]) -> float | None:
    """The expression's value, or None when a name it reads is missing or the
    arithmetic will not stand (a division by zero, an overflow)."""
    names = derived_names(expression)
    if any(name not in values or values[name] != values[name] for name in names):
        return None

    def run(node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, (ast.Name, ast.Attribute)):
            return float(values[dotted(node)])
        if isinstance(node, ast.UnaryOp):
            value = run(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        return _BINOPS[type(node.op)](run(node.left), run(node.right))  # type: ignore[attr-defined]

    try:
        out = float(run(ast.parse(expression.strip(), mode="eval").body))
    except (ZeroDivisionError, OverflowError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def derived_metrics(values: dict[str, float], spec: Any) -> dict[str, float]:
    """The lock's `derived` block evaluated against the metrics measured so far.

    In the order the lock wrote them, so one derived metric may read another. A
    refused expression raises; a merely unanswerable one is left out.
    """
    if not isinstance(spec, dict):
        return {}
    out: dict[str, float] = {}
    known = dict(values)
    for name, expression in spec.items():
        value = evaluate_derived(str(expression), known)
        if value is not None:
            out[name] = value
            known[name] = value
    return out


def cases_of(lock: dict) -> list[dict]:
    """The lock's `cases`, checked: a name that can be a directory and a namespace, and
    a list of `case_gen.py` flags. Empty for the one-case lock, which is most of them."""
    raw = lock.get("cases") or []
    if not isinstance(raw, list):
        raise SystemExit("cases: expected a list of {name, case_gen_args}")
    out: list[dict] = []
    seen: set[str] = set()
    for i, case in enumerate(raw):
        if not isinstance(case, dict):
            raise SystemExit(f"cases[{i}]: expected an object with name and case_gen_args")
        name = str(case.get("name") or "")
        if not name.isidentifier():
            raise SystemExit(f"cases[{i}].name: {name!r} must be an identifier -- it is a "
                             "directory name and a metric namespace")
        if name in seen:
            raise SystemExit(f"cases[{i}].name: {name!r} twice")
        seen.add(name)
        args = case.get("case_gen_args") or []
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise SystemExit(f"cases[{i}].case_gen_args: expected a list of strings")
        out.append({"name": name, "case_gen_args": [str(a) for a in args]})
    return out


def solver_log(case: Path, solver: str) -> Path | None:
    """`log.<solver>` when it is there -- that is the file `parametric.py` writes -- else
    whatever `results.find_solver_log` picks out of the case."""
    named = case / f"log.{solver}"
    if named.is_file():
        return named
    return results.find_solver_log(case)


def read_log(log: Path | None) -> dict:
    """What the solver log says about the run: shape, convergence, divergence."""
    out = {"shape": "", "converged": False, "diverged": "", "steps": 0}
    if log is None or not log.is_file():
        return out
    data = log_digest.digest(log)
    shape = log_digest.residual_shape(data["residuals"])["shape"]
    out["shape"] = shape
    out["steps"] = len(data["times"])
    out["converged"] = data["converged_at"] is not None or shape in ("levelled", "falling")
    if data["fatal"]:
        out["diverged"] = data["fatal"]
    elif shape == "diverging":
        out["diverged"] = "a residual is climbing off its own best"
    else:
        steps = max(1, out["steps"])
        depth = data.get("bounding_depth") or {}
        held = [field for field, count in sorted(data["bounding"].items())
                if count >= BOUND_FRACTION * steps and count >= log_digest.MIN_STEPS
                and depth.get(field, 0.0) >= BOUND_DEPTH]
        if held:
            out["diverged"] = (f"{', '.join(held)} held at its bound on "
                               f"{', '.join(str(data['bounding'][f]) for f in held)} of "
                               f"{out['steps']} steps (clipped from "
                               f"{', '.join(f'{depth[f]:.2g}' for f in held)} x the mean below zero)")
    if not data["residuals"] and not data["fatal"]:
        out["shape"] = "short"
    return out


def yplus_range(case: Path, patch: str = "") -> dict | None:
    """`{"min": .., "max": ..}` off the yPlus function object's last write, or None.

    The file is `postProcessing/<name>/<time>/yPlus.dat`, one row per patch per write:
    `time patch min max average`. The body patch's row is the one that matters for a
    wall-function band; without a named patch every wall row at the last write counts.
    """
    base = case / "postProcessing"
    if not base.is_dir():
        return None
    rows: list[tuple[float, str, float, float]] = []
    for dat in sorted(base.rglob("yPlus*.dat")):
        try:
            text = dat.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 4 or parts[0].startswith("#"):
                continue
            try:
                rows.append((float(parts[0]), parts[1], float(parts[2]), float(parts[3])))
            except ValueError:
                continue
    if patch:
        rows = [row for row in rows if row[1] == patch]
    if not rows:
        return None
    last = max(row[0] for row in rows)
    final = [row for row in rows if row[0] == last]
    return {"min": min(row[2] for row in final), "max": max(row[3] for row in final)}


def build_sha256(case: Path) -> str:
    """The hash of the `build.py` that produced this mesh, so a leaderboard entry names
    its parameterization. Empty when the case has no build script."""
    script = case / "build.py"
    if not script.is_file():
        return ""
    return hashlib.sha256(script.read_bytes()).hexdigest()


def versions() -> dict:
    """What produced the numbers, best effort: the environment's OpenFOAM, and the two
    Python libraries a build script is likely to have used. `unknown` where absent."""
    out = {"foam": os.environ.get("WM_PROJECT_VERSION") or "unknown"}
    for name in ("gmsh", "build123d"):
        try:
            module = __import__(name)
            out[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # noqa: BLE001 - absent, broken, either way not here
            out[name] = "unknown"
    return out


# -- the verdict ---------------------------------------------------------------------


def score(case: Path, lock: dict, run: dict | None = None, *, label: str = "",
          detail: str = "", mesh_only: bool = False) -> dict:
    """Everything `metrics.json` carries, from the case and the lock.

    `run` is what `parametric.py` knows and the case does not: the `case_gen.py`
    arguments it used, the wall time, the trim record. `label` when given is the
    caller's verdict and wins -- that is how a mesh failure keeps its reason and how a
    killed run is marked `timeout`.
    """
    case = Path(case)
    run = dict(run or {})
    fidelity = fidelity_of(lock)
    fraction = float(fidelity.get("window") or DEFAULT_WINDOW)
    solver = str(lock.get("solver") or "simpleFoam")
    body = str(lock.get("body_patch") or "")
    transient = transient_of(lock)

    mesh_ok, mesh_why = mesh_verdict(case)
    cells = cell_count(case)
    metrics: dict = {}
    window = 0
    converged = False
    shape = ""
    yplus = None
    diverged = ""
    shed = None
    if not mesh_only and mesh_ok:
        if transient:
            metrics, window, shed = transient_force_metrics(case, transient)
        else:
            metrics, window = force_metrics(case, fraction)
        dp, dp_window = pressure_drop(case, fraction)
        if dp is not None:
            metrics["dp"] = dp
            window = window or dp_window
        log = read_log(solver_log(case, solver))
        converged, shape, diverged = log["converged"], log["shape"], log["diverged"]
        if transient:
            # A shedding run's residuals oscillate with the flow and never level; what
            # says its mean is a number is whether the window spans whole cycles.
            converged = shed["periods"] >= transient["min_periods"]
        yplus = yplus_range(case, body)

    yplus_band = band(fidelity.get("yplus"))
    yplus_bad = yplus is not None and (outside(yplus["min"], yplus_band)
                                       or outside(yplus["max"], yplus_band))
    # The cell band applies to a mesh probe too: a mesh that lands outside it is known
    # before a solve is paid for on it, under the label the solve would have earned.
    cells_bad = bool(cells) and outside(cells, band(fidelity.get("cells")))

    decided, why = label, detail
    if not decided:
        if not mesh_ok:
            decided, why = "mesh", mesh_why
        elif not mesh_only and diverged:
            decided, why = "diverged", diverged
        elif not mesh_only and not converged and shed is not None:
            decided, why = "unconverged", unshed(shed)
        elif not mesh_only and not converged:
            decided, why = "unconverged", (f"residuals {shape}" if shape else "no residuals in the log")
        elif not mesh_only and yplus_bad:
            decided, why = "yplus", f"y+ {yplus['min']:.3g}..{yplus['max']:.3g} against {fidelity.get('yplus')}"
        elif cells_bad:
            decided, why = "cells", f"{cells} cells against {fidelity.get('cells')}"
        else:
            decided, why = "ok", ""

    out = {
        "label": decided,
        "detail": why,
        "fidelity": fidelity,
        "template_version": lock.get("template_version"),
        "build_sha256": build_sha256(case),
        "case_args": list(run.get("case_args") or []),
        "trim": run.get("trim"),
        "metrics": metrics,
        "window": window,
        "converged": bool(converged),
        "residual_shape": shape,
        "yplus": yplus,
        "cells": int(cells),
        "wall_seconds": float(run.get("wall_seconds") or 0.0),
        "versions": versions(),
    }
    if shed is not None:
        out["shedding"] = shed
    return out


def unshed(shed: dict) -> str:
    """Why a transient case's window is not a mean yet, in the numbers it saw."""
    if shed["window_s"] is None:
        if "last_s" in shed:
            return (f"the force record ends at {shed['last_s']:.4g} s, before the averaging "
                    f"window opens at {shed['average_from_s']:.4g} s")
        return "no force record in the case"
    t0, t1 = shed["window_s"]
    said = (f"{shed['periods']:.1f} shedding periods in the averaging window "
            f"({t0:.4g}..{t1:.4g} s); {shed['min_periods']} needed")
    return f"{said} -- {shed['why']}" if shed.get("why") else said


def worst(labels: list[str]) -> str:
    """The worst of several labels by `SEVERITY`; `ok` when there are none."""
    ranked = [label for label in SEVERITY if label in labels]
    if ranked:
        return ranked[0]
    return labels[0] if labels else "ok"


def score_cases(candidate: Path, lock: dict, run: dict | None = None, *, label: str = "",
                detail: str = "", mesh_only: bool = False) -> dict:
    """One `metrics.json` for a candidate the lock gave several cases.

    Each case is `CANDIDATE_DIR/<name>/` -- the same mesh, dressed and solved under its
    own `case_gen.py` arguments -- and is scored by `score` as if it were the whole
    candidate. What comes out is one record: the cases' metrics namespaced by case name,
    the lock's `derived` arithmetic over them, and the worst of their labels, because a
    candidate is only as good as its worst case.

    `mesh_only` scores the mesh in the candidate root and no case: a probe meshes once
    and there is nothing per-case about it.
    """
    candidate = Path(candidate)
    run = dict(run or {})
    cases = cases_of(lock)
    if mesh_only or not cases:
        return score(candidate, lock, run, label=label, detail=detail, mesh_only=mesh_only)

    per_case: dict[str, Any] = {}
    values: dict[str, float] = {}
    labels: list[str] = []
    details: list[str] = []
    windows: list[int] = []
    ran = dict(run.get("cases") or {})
    for case in cases:
        name = case["name"]
        directory = candidate / name
        case_args = list((ran.get(name) or {}).get("case_args") or case["case_gen_args"])
        if not directory.is_dir():
            per_case[name] = {"label": "mesh", "detail": f"{name}/ is not there: the case did not run",
                              "metrics": {}, "window": 0, "converged": False, "case_args": case_args}
            labels.append("mesh")
            details.append(f"{name}: the case did not run")
            continue
        one = score(directory, lock, {"case_args": case_args}, mesh_only=False)
        per_case[name] = {
            "label": one["label"], "detail": one["detail"], "metrics": one["metrics"],
            "window": one["window"], "converged": one["converged"],
            "residual_shape": one["residual_shape"], "yplus": one["yplus"],
            "case_args": case_args,
        }
        labels.append(one["label"])
        if one["detail"]:
            details.append(f"{name}: {one['detail']}")
        windows.append(int(one["window"]))
        for metric, value in one["metrics"].items():
            values[f"{name}.{metric}"] = value
    values.update(derived_metrics(values, lock.get("derived")))

    # The mesh is the candidate's, not a case's: it was built and measured once in the
    # candidate root and every case is a copy of it, so the mesh verdict and the cell
    # band are read there -- and they count among the labels, or a candidate whose mesh
    # is outside the band would pass on two happy solves.
    root = score(candidate, lock, run, mesh_only=True)
    if root["label"] != "ok":
        labels.append(root["label"])
        if root["detail"]:
            details.append(f"mesh: {root['detail']}")
    decided = label or worst(labels)
    why = detail or ("; ".join(details) if decided != "ok" else "")
    root.update({
        "label": decided, "detail": why, "metrics": values, "cases": per_case,
        "window": min(windows) if windows else 0,
        "converged": all(bool(c.get("converged")) for c in per_case.values()) if per_case else False,
        "residual_shape": "/".join(str(c.get("residual_shape") or "") for c in per_case.values()),
        "yplus": next((c["yplus"] for c in per_case.values() if c.get("yplus")), None),
        "wall_seconds": float(run.get("wall_seconds") or 0.0),
    })
    return root


def write(metrics: dict, out: Path) -> Path:
    """The one place `metrics.json` is written. Atomic, so a reader polling for the
    file never sees half of it."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(metrics, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    return out


def read_run(case: Path, path: Path | None) -> dict:
    """`run.json`, which `parametric.py` leaves beside the candidate as it works:
    case_args, wall_seconds, trim. Absent is fine -- the fields come back empty."""
    target = Path(path) if path else case / "run.json"
    if not target.is_file():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", type=Path, help="the case directory that was built and solved")
    ap.add_argument("--lock", type=Path, required=True, help="goal.lock.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="where metrics.json goes (default <case>/metrics.json)")
    ap.add_argument("--mesh-only", action="store_true", dest="mesh_only",
                    help="a mesh probe: label mesh, cells or ok, no coefficients")
    ap.add_argument("--label", default="", choices=["", *LABELS],
                    help="the caller's verdict, which wins over what is measured")
    ap.add_argument("--detail", default="", help="why, in a sentence, beside --label")
    ap.add_argument("--run", type=Path, default=None,
                    help="parametric.py's run.json (default <case>/run.json)")
    args = ap.parse_args(argv)

    case = args.case.resolve()
    lock = load_lock(args.lock)
    run = read_run(case, args.run)
    if not run.get("wall_seconds") and run.get("started"):
        run["wall_seconds"] = max(0.0, time.time() - float(run["started"]))
    # A lock with `cases` means the candidate is several solves off one mesh, and
    # `score_cases` falls back to `score` when it has none, so one call serves both.
    metrics = score_cases(case, lock, run, label=args.label, detail=args.detail,
                          mesh_only=args.mesh_only)
    out = write(metrics, args.out or case / "metrics.json")
    print(f"{metrics['label']}" + (f"  ({metrics['detail']})" if metrics["detail"] else ""))
    for name, value in sorted(metrics["metrics"].items()):
        print(f"  {name} = {value:.5g}   (mean of the last {metrics['window']} rows)")
    for name, one in (metrics.get("cases") or {}).items():
        print(f"  [{name}] {one['label']}" + (f": {one['detail']}" if one["detail"] else ""))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
