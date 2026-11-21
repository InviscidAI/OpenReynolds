#!/usr/bin/env python3
"""A cheap stand-in for the solver, fitted to runs that have already finished.

    python3 surrogate.py points.csv --objective Cd
    python3 surrogate.py points.csv --objective Cd --design alpha,camber --png fit.png
    python3 surrogate.py --candidates round1/ --objective metrics.body.Cd --json s.json
    python3 surrogate.py points.csv --objective Cd --at alpha=4.5,camber=0.03

A parametric sweep leaves a table: a handful of design variables and one number per
row that cost a solve. The question afterwards is always the same -- what does the
objective look like between the points that were paid for, and where is its best
value -- and answering it by eye on more than two variables does not work.

This fits a polynomial response surface to that table and REPORTS HOW WRONG IT IS
BY LEAVING EACH POINT OUT.

on the one number that makes this honest

A surrogate graded on the points it was fitted to is not graded at all. Enough terms
and a least-squares fit passes within round-off of every sample, so its residual is
nearly zero on a response it has learned nothing about, and that near-zero is what
gets quoted. The measured version, on `sin(2.4x) exp(-1.3y) + 0.35xy` over the unit
square (`tests/test_toolbox_surrogate.py::test_the_training_error_of_a_rich_fit_is
_a_lie`, which prints every number below): a degree-4 surface, 15 terms through 16
points, highest leverage 0.999993.

    rmse on its own 16 points            0.000109
    rmse on 2500 points it never saw     0.005236   -- 48x the first number
    rmse this script reports (leave-one-out)  0.044015

So every fit here is reported with both numbers, and the one chosen between degrees is
the leave-one-out one. For a model that is linear in its coefficients -- which a
polynomial fitted by least squares is -- leave-one-out needs no refitting: with the
projection `H = X X^+`, the fit residual `r_i` and the leverage `h_ii = H_ii`,

    residual with point i left out = r_i / (1 - h_ii)

exactly, for every point, from the one fit. That identity is checked against twenty
actual refits rather than trusted (`test_the_closed_form_leave_one_out_equals_
refitting_without_the_point`). It is also why a degree whose basis has as many terms
as there are rows is refused by name rather than fitted. At `p = n` every `h_ii` is 1,
the division is by zero, and the surface interpolates: there is no information left
with which to grade it. `--max-degree` will not talk you past that.

The cross-validated error is itself checked rather than asserted. On the same closed
form sampled at 25 points, the chosen degree-3 fit reports

    rmse on its own 25 points            0.004610   -- 3.1x low
    rmse_loo, what this script reports   0.010449   -- 1.35x low
    rmse on 2500 points it never saw     0.014102

so the honest number came back 26% optimistic where the fit error came back a factor
of three optimistic, and on the overfitted case above it came back 8x PESSIMISTIC.
Leave-one-out on tens of points is a noisy estimator and that is the size of the
noise; what it does not do is miss by an order of magnitude in the flattering
direction, which is the single failure mode this script exists to prevent. Both
comparisons are against a response whose value is known everywhere, which is the only
kind of reference a surrogate can honestly be graded against -- there is no published
number for how well a cubic fits somebody's sweep.

on why a polynomial and not a Gaussian process

scipy is not on the image (`ENVIRONMENT.md`) and sklearn cannot be added, so both
would be hand-written numpy either way. The polynomial wins on what can be checked: it
is linear in its coefficients, so the leave-one-out identity above is exact and free,
while a GP's hyperparameters are themselves fitted and a GP graded with its
hyperparameters tuned on the full sample has the same lie built back in one level up.
With the handful of points a CFD sweep can afford -- tens, not thousands -- a quadratic
or cubic in a few variables is also the model that sample size supports. If a response
is known to be sharply non-polynomial, this says so the only way it honestly can: the
leave-one-out error comes back large and no degree improves it.

on the best point it reports

`optimum` is the fitted surface's own extreme inside the box the samples span, found
by a dense quasi-random search over that box, and it is A PREDICTION. Its error bar is
`rmse_loo`, which is printed beside it, and the objective there has not been solved
for. The box is the sampled range in every variable and nothing extrapolates past it:
a polynomial a little way outside its samples is not a model of anything. When the
predicted optimum sits on a face of that box, that is said out loud, because it means
the sweep was bounded before the response was.

on the inputs

`points.csv` (or `.json`): one row per design point, one column per design variable
plus the objective. `--design` names the variables when the table carries columns that
are not any (a label, a wall time, a second objective); without it every numeric
column that is not the objective is a design variable.

`--candidates DIR` reads the layout a parametric round leaves on disk instead: one
subdirectory per candidate, each with a `design_constants.py` of `NAME = <number>`
lines and a `metrics.json`. The design variables are those names, the objective is a
dotted path into the JSON (`metrics.body.Cd`), and a candidate missing either file, or
whose `label` is not `ok`, is NAMED AND SKIPPED rather than fitted as a zero. The
constants file is read with a regex and never executed: it is somebody else's data.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

CONSTANT = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*([-+0-9.eE]+)\s*(?:#.*)?$", re.M)
"""A `NAME = <number>` line of a `design_constants.py`. Read, not executed."""

LEVERAGE_FLOOR = 1e-9
"""How close `1 - h_ii` may come to zero before the leave-one-out residual at that
point is meaningless. At `h_ii = 1` the surface passes exactly through point i whether
or not it was in the fit, so there is nothing to divide and nothing to learn."""

SEARCH_POINTS = 20000
"""Points in the box the optimum search uses, on a fixed seed so two runs of this
script on one table report the same optimum."""


class Unavailable(Exception):
    """Not enough to fit anything. A refusal with a reason, not a crash."""


# -- reading a table ---------------------------------------------------------------


def _number(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def read_table(path: Path) -> list[dict[str, Any]]:
    """Rows from a `.csv` or a `.json` (a list of objects, or an object of columns)."""
    path = Path(path)
    if not path.is_file():
        raise Unavailable(f"no such table: {path}")
    if path.suffix.lower() == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            keys = list(loaded)
            length = {len(v) for v in loaded.values() if isinstance(v, list)}
            if len(length) != 1:
                raise Unavailable(f"{path.name} is an object whose columns differ in length")
            return [{k: loaded[k][i] for k in keys} for i in range(length.pop())]
        if not isinstance(loaded, list):
            raise Unavailable(f"{path.name} is neither a list of rows nor columns")
        return [dict(row) for row in loaded]
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def dig(payload: Any, dotted: str) -> Any:
    """`metrics.body.Cd` out of nested JSON, or None where the path stops."""
    cursor = payload
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def read_candidates(root: Path, objective: str) -> tuple[list[dict[str, Any]], list[str]]:
    """One row per candidate directory, plus the ones that were skipped and why.

    The layout a parametric round leaves: `<candidate>/design_constants.py` and
    `<candidate>/metrics.json`. A candidate with no metrics has not been scored and a
    candidate whose `label` is not `ok` was scored as a failure -- a diverged run's
    coefficient is not a sample of the response, and averaging it in is how a surrogate
    learns that a bad mesh is a good design.
    """
    root = Path(root)
    if not root.is_dir():
        raise Unavailable(f"no such directory: {root}")
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for candidate in sorted(p for p in root.iterdir() if p.is_dir()):
        constants = candidate / "design_constants.py"
        metrics_file = candidate / "metrics.json"
        if not constants.is_file():
            skipped.append(f"{candidate.name}: no design_constants.py")
            continue
        if not metrics_file.is_file():
            skipped.append(f"{candidate.name}: not scored (no metrics.json)")
            continue
        try:
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            skipped.append(f"{candidate.name}: metrics.json unreadable ({exc})")
            continue
        label = metrics.get("label")
        if label is not None and label != "ok":
            skipped.append(f"{candidate.name}: label {label}")
            continue
        value = _number(str(dig(metrics, objective)))
        if value is None:
            skipped.append(f"{candidate.name}: no {objective} in metrics.json")
            continue
        row: dict[str, Any] = {
            name: float(text)
            for name, text in CONSTANT.findall(constants.read_text(encoding="utf-8"))
        }
        if not row:
            skipped.append(f"{candidate.name}: design_constants.py names no numbers")
            continue
        row[objective] = value
        row["candidate"] = candidate.name
        rows.append(row)
    return rows, skipped


def as_samples(rows: Sequence[dict[str, Any]], objective: str,
               design: Sequence[str] | None = None) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Design names, the `n x k` design matrix and the `n` objectives.

    A row missing the objective, or carrying it as something that is not a number, is
    dropped: an empty cell in a hand-edited sweep table is a point that was not run.
    """
    if not rows:
        raise Unavailable("the table has no rows")
    if objective not in rows[0]:
        raise Unavailable(
            f"no column {objective!r}; the table has {', '.join(sorted(rows[0]))}")
    if design:
        names = list(design)
        missing = [n for n in names if n not in rows[0]]
        if missing:
            raise Unavailable(f"no column(s) {', '.join(missing)} in the table")
    else:
        names = [key for key in rows[0]
                 if key != objective and _number(str(rows[0][key])) is not None]
    if not names:
        raise Unavailable("no numeric design variable beside the objective")
    points, values = [], []
    for row in rows:
        y = _number(str(row.get(objective)))
        xs = [_number(str(row.get(name))) for name in names]
        if y is None or any(x is None for x in xs):
            continue
        points.append(xs)
        values.append(y)
    if not points:
        raise Unavailable("every row was missing the objective or a design variable")
    return names, np.asarray(points, dtype=float), np.asarray(values, dtype=float)


# -- the basis ---------------------------------------------------------------------


def exponents(variables: int, degree: int) -> Iterator[tuple[int, ...]]:
    """Every exponent tuple of total degree at most `degree`, lowest degree first."""
    if variables == 0:
        yield ()
        return
    for power in range(degree + 1):
        for rest in exponents(variables - 1, degree - power):
            yield (power,) + rest


def terms(variables: int, degree: int) -> list[tuple[int, ...]]:
    return sorted(exponents(variables, degree), key=lambda e: (sum(e), e))


def design_matrix(scaled: np.ndarray, basis: Sequence[tuple[int, ...]]) -> np.ndarray:
    """Each column is one monomial of the scaled coordinates, evaluated at every row."""
    columns = [np.prod(scaled ** np.asarray(term, dtype=float), axis=1) for term in basis]
    return np.column_stack(columns) if columns else np.empty((len(scaled), 0))


def scaling(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Per-variable `lo`, `hi`, and which variables are constant across the sample.

    The basis is built on coordinates mapped to [-1, 1] rather than on the raw
    numbers. A sweep over a chord in millimetres and a camber in fractions spans five
    orders of magnitude, and a cubic of the raw values has columns spanning fifteen:
    the normal equations of that are singular to single precision long before the
    sample runs out. A constant variable cannot be scaled at all -- it is named and
    dropped, because a column of ones repeated is the intercept twice over.
    """
    lo, hi = points.min(axis=0), points.max(axis=0)
    flat = [i for i in range(points.shape[1]) if hi[i] <= lo[i]]
    return lo, hi, flat


def to_unit(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    span = np.where(hi > lo, hi - lo, 1.0)
    return 2.0 * (points - lo) / span - 1.0


# -- the fit and its leave-one-out error -------------------------------------------


def fit_degree(scaled: np.ndarray, values: np.ndarray, degree: int) -> dict[str, Any]:
    """One least-squares polynomial fit, with its fit error and its honest error.

    `rmse_loo` is the exact leave-one-out error of this very fit, from the leverages
    of the projection, not an approximation of it and not a second fit.
    """
    basis = terms(scaled.shape[1], degree)
    n, p = len(values), len(basis)
    row: dict[str, Any] = {"degree": degree, "n_terms": p, "n_points": n}
    if p >= n:
        row["refused"] = (f"{p} terms for {n} points: the surface interpolates, so there "
                          f"is nothing left to grade it on")
        return row
    matrix = design_matrix(scaled, basis)
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, values, rcond=None)
    if rank < p:
        row["refused"] = (f"the sample is degenerate for this basis: rank {rank} of {p} "
                          f"terms, so these points do not determine the surface")
        return row
    predicted = matrix @ coefficients
    residual = values - predicted
    leverage = np.einsum("ij,ji->i", matrix, np.linalg.pinv(matrix))
    denominator = 1.0 - leverage
    if np.any(denominator < LEVERAGE_FLOOR):
        worst = int(np.argmin(denominator))
        row["refused"] = (f"point {worst} has leverage {leverage[worst]:.6f}: the fit is "
                          f"pinned to it, so leaving it out is not a test")
        return row
    loo_residual = residual / denominator
    spread = float(np.sum((values - values.mean()) ** 2))
    row.update({
        "coefficients": [float(c) for c in coefficients],
        "terms": [list(term) for term in basis],
        "rmse_fit": float(np.sqrt(np.mean(residual ** 2))),
        "rmse_loo": float(np.sqrt(np.mean(loo_residual ** 2))),
        "max_loo": float(np.max(np.abs(loo_residual))),
        "q2_loo": float(1.0 - np.sum(loo_residual ** 2) / spread) if spread > 0 else None,
        "loo_predicted": [float(v) for v in values - loo_residual],
        "max_leverage": float(np.max(leverage)),
    })
    return row


def predict(fit: dict[str, Any], points: np.ndarray) -> np.ndarray:
    """The fitted surface at raw (unscaled) coordinates."""
    lo = np.asarray(fit["lo"], dtype=float)
    hi = np.asarray(fit["hi"], dtype=float)
    basis = [tuple(term) for term in fit["terms"]]
    matrix = design_matrix(to_unit(np.atleast_2d(points), lo, hi), basis)
    return matrix @ np.asarray(fit["coefficients"], dtype=float)


def best_in_box(fit: dict[str, Any], minimise: bool = True) -> dict[str, Any]:
    """The fitted surface's extreme inside the box the samples span.

    A fixed-seed sample of the box rather than a solve: the surface is cheap, the box
    is low-dimensional in every case this is for, and a search that reports the same
    answer twice is worth more here than the last decimal of the extreme. Reported
    with which coordinates sit on a face of the box, because an optimum on a face is
    the sweep's bound and not the response's.
    """
    lo = np.asarray(fit["lo"], dtype=float)
    hi = np.asarray(fit["hi"], dtype=float)
    rng = np.random.default_rng(0)
    trial = lo + (hi - lo) * rng.random((SEARCH_POINTS, len(lo)))
    values = predict(fit, trial)
    index = int(np.argmin(values) if minimise else np.argmax(values))
    at = trial[index]
    span = np.where(hi > lo, hi - lo, 1.0)
    on_edge = [fit["design"][i] for i in range(len(at))
               if min(at[i] - lo[i], hi[i] - at[i]) < 0.02 * span[i]]
    return {
        "at": {name: float(value) for name, value in zip(fit["design"], at)},
        "predicted": float(values[index]),
        "sense": "minimum" if minimise else "maximum",
        "on_the_box_edge": on_edge,
    }


def build(names: Sequence[str], points: np.ndarray, values: np.ndarray,
          max_degree: int = 3) -> dict[str, Any]:
    """Fit every degree up to `max_degree` and keep the one with the lowest honest
    error. Every degree tried is reported, refusals included: which degrees could not
    be graded on this sample is as much of the answer as the one that could."""
    lo, hi, flat = scaling(points)
    dropped = [names[i] for i in flat]
    keep = [i for i in range(points.shape[1]) if i not in set(flat)]
    if not keep:
        raise Unavailable("every design variable is constant across the sample")
    names = [names[i] for i in keep]
    points, lo, hi = points[:, keep], lo[keep], hi[keep]
    scaled = to_unit(points, lo, hi)
    tried = [fit_degree(scaled, values, degree) for degree in range(1, max_degree + 1)]
    graded = [row for row in tried if "rmse_loo" in row]
    if not graded:
        raise Unavailable(
            f"{len(values)} points support no degree up to {max_degree}: "
            + "; ".join(row["refused"] for row in tried))
    # An exact fit's leave-one-out error is round-off, so choosing between two of them
    # by `min` compares noise: on a quadratic sampled without error, degree 2 and
    # degree 3 both land near 1e-16 and which is smaller depends on the BLAS. That is
    # how this picked a cubic on Windows and a quadratic on Linux for the same points.
    # Anything at or below the floor is exact, and the lowest exact degree is the model
    # -- the extra terms of a richer basis are fitting round-off, and the fit is what
    # `--at` evaluates afterwards. The floor is relative to the data's own scale
    # because a residual's round-off is relative to the magnitudes being differenced,
    # and 32 eps is slack for the factorisation and the leverage divide rather than for
    # a single operation.
    scale = float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0
    exact_floor = 32.0 * float(np.finfo(float).eps) * max(scale, 1.0)
    exact = [row for row in graded if row["rmse_loo"] <= exact_floor]
    if exact:
        chosen = min(exact, key=lambda row: row["degree"])
    else:
        chosen = min(graded, key=lambda row: row["rmse_loo"])
    fit = dict(chosen)
    fit.update({
        "exact_floor": exact_floor,
        "exact_degrees": [row["degree"] for row in exact],
        "design": list(names),
        "objective_mean": float(values.mean()),
        "objective_range": [float(values.min()), float(values.max())],
        "lo": [float(v) for v in lo],
        "hi": [float(v) for v in hi],
        "dropped_as_constant": dropped,
        "observed": [float(v) for v in values],
    })
    return {"fit": fit, "degrees": [{k: v for k, v in row.items()
                                     if k not in ("loo_predicted", "coefficients", "terms")}
                                    for row in tried]}


# -- saying it ---------------------------------------------------------------------


def report(found: dict[str, Any]) -> str:
    fit = found["fit"]
    lines = [f"design variables: {', '.join(fit['design'])}  ({len(fit['observed'])} points)"]
    if fit["dropped_as_constant"]:
        lines.append(f"constant across the sample, dropped: "
                     f"{', '.join(fit['dropped_as_constant'])}")
    lines.append(f"objective spans {fit['objective_range'][0]:.6g} to "
                 f"{fit['objective_range'][1]:.6g}")
    lines.append("")
    lines.append("degree  terms   rmse_fit    rmse_loo    Q2_loo")
    for row in found["degrees"]:
        if "refused" in row:
            lines.append(f"{row['degree']:>6}  {row['n_terms']:>5}   refused: {row['refused']}")
            continue
        q2 = "--" if row["q2_loo"] is None else f"{row['q2_loo']:>8.4f}"
        lines.append(f"{row['degree']:>6}  {row['n_terms']:>5}  {row['rmse_fit']:>9.4g}  "
                     f"{row['rmse_loo']:>9.4g}  {q2}")
    lines.append("")
    lines.append(f"kept degree {fit['degree']}: leave-one-out RMSE {fit['rmse_loo']:.4g}, "
                 f"worst single point {fit['max_loo']:.4g}")
    lines.append(f"  the fit RMSE of {fit['rmse_fit']:.4g} is what it scores on points it "
                 f"was fitted to, and is not its error")
    best = found.get("optimum")
    if best:
        where = ", ".join(f"{k}={v:.6g}" for k, v in best["at"].items())
        lines.append("")
        lines.append(f"predicted {best['sense']} {best['predicted']:.6g} at {where}")
        lines.append(f"  predicted, not solved: carry +/- {fit['rmse_loo']:.4g} on it")
        if best["on_the_box_edge"]:
            lines.append(f"  on the edge of the sampled box in "
                         f"{', '.join(best['on_the_box_edge'])}: that bound is the sweep's, "
                         f"not the response's")
    for note in found.get("skipped", []):
        lines.append(f"skipped {note}")
    return "\n".join(lines)


def plot(found: dict[str, Any], target: Path) -> Path:
    """Two panels: what the surface predicts for each point when that point is left
    out against what the solver measured, and the two error curves against degree."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit = found["fit"]
    observed = np.asarray(fit["observed"], dtype=float)
    held_out = np.asarray(fit["loo_predicted"], dtype=float)
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6))
    limits = [min(observed.min(), held_out.min()), max(observed.max(), held_out.max())]
    pad = 0.05 * (limits[1] - limits[0] or 1.0)
    limits = [limits[0] - pad, limits[1] + pad]
    left.plot(limits, limits, color="0.6", lw=1)
    left.scatter(observed, held_out, s=28, color="#1f77b4")
    left.set_xlim(limits)
    left.set_ylim(limits)
    left.set_xlabel("measured")
    left.set_ylabel("predicted with this point left out")
    left.set_title(f"degree {fit['degree']}: leave-one-out RMSE {fit['rmse_loo']:.4g}")
    left.grid(alpha=0.3)

    graded = [row for row in found["degrees"] if "rmse_loo" in row]
    if graded:
        degrees = [row["degree"] for row in graded]
        right.plot(degrees, [row["rmse_fit"] for row in graded], "o-", label="fit (not an error)")
        right.plot(degrees, [row["rmse_loo"] for row in graded], "s-", label="leave-one-out")
        right.set_yscale("log")
        right.set_xticks(degrees)
    refused = [row["degree"] for row in found["degrees"] if "refused" in row]
    if refused:
        right.set_title(f"degree{'s' if len(refused) > 1 else ''} "
                        f"{', '.join(str(d) for d in refused)} refused: too few points",
                        fontsize=9)
    right.set_xlabel("polynomial degree")
    right.set_ylabel("RMSE")
    right.legend(fontsize=8)
    right.grid(alpha=0.3)
    figure.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=130)
    plt.close(figure)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A polynomial surrogate for a finished parametric sweep, reported "
                    "with its leave-one-out error rather than its fit error.")
    parser.add_argument("table", type=Path, nargs="?", default=None,
                        help="the sweep as a .csv or .json: one row per design point")
    parser.add_argument("--candidates", type=Path, default=None,
                        help="instead of a table, a directory of candidate "
                             "subdirectories each holding design_constants.py and "
                             "metrics.json")
    parser.add_argument("--objective", required=True,
                        help="the column to fit; with --candidates a dotted path into "
                             "metrics.json, e.g. metrics.body.Cd")
    parser.add_argument("--design", default="",
                        help="comma-separated design variables; without it, every "
                             "numeric column that is not the objective")
    parser.add_argument("--max-degree", type=int, default=3,
                        help="highest polynomial degree to try (default 3). A degree "
                             "with as many terms as there are points is refused, not "
                             "fitted")
    parser.add_argument("--maximise", action="store_true",
                        help="report the predicted maximum rather than the minimum")
    parser.add_argument("--at", default="",
                        help="evaluate the surrogate here, e.g. alpha=4.5,camber=0.03")
    parser.add_argument("--png", type=Path, default=None, help="write the figure here")
    parser.add_argument("--json", type=Path, default=None, help="write the findings here")
    args = parser.parse_args(argv)

    try:
        skipped: list[str] = []
        if args.candidates is not None:
            rows, skipped = read_candidates(args.candidates, args.objective)
        elif args.table is not None:
            rows = read_table(args.table)
        else:
            parser.error("give a table or --candidates")
            return 2
        names, points, values = as_samples(
            rows, args.objective,
            [n.strip() for n in args.design.split(",") if n.strip()] or None)
        found = build(names, points, values, max_degree=max(1, args.max_degree))
    except Unavailable as exc:
        # A refusal is a reading, like `spectra.py` and `layer_report.py`: the caller
        # asked what the response looks like and the answer is that this sample cannot
        # say, which is worth exit 0 and a sentence.
        print(f"not fitted: {exc}")
        return 0
    found["skipped"] = skipped
    found["optimum"] = best_in_box(found["fit"], minimise=not args.maximise)
    if args.at:
        asked = dict(piece.split("=", 1) for piece in args.at.split(",") if "=" in piece)
        missing = [n for n in found["fit"]["design"] if n not in asked]
        if missing:
            print(f"--at is missing {', '.join(missing)}")
            return 2
        where = np.array([[float(asked[n]) for n in found["fit"]["design"]]])
        lo = np.asarray(found["fit"]["lo"])
        hi = np.asarray(found["fit"]["hi"])
        outside = [n for n, v, a, b in zip(found["fit"]["design"], where[0], lo, hi)
                   if v < a or v > b]
        value = float(predict(found["fit"], where)[0])
        found["at"] = {"where": asked, "predicted": value, "outside_the_sampled_box": outside}
        print(f"predicted {args.objective} = {value:.6g} "
              f"+/- {found['fit']['rmse_loo']:.4g} (leave-one-out)")
        if outside:
            print(f"  outside the sampled range in {', '.join(outside)}: "
                  f"a polynomial past its samples is not a model of anything")
    print(report(found))
    if args.png is not None:
        written = plot(found, Path.cwd() / args.png if not args.png.is_absolute() else args.png)
        print(f"\nfigure: {written}")
    if args.json is not None:
        target = Path.cwd() / args.json if not args.json.is_absolute() else args.json
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(found, indent=2, default=float), encoding="utf-8")
        print(f"json: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
