#!/usr/bin/env python3
"""Grade an `mhdFoam` Hartmann-flow run against the closed-form velocity profile.

    python benchmarks/mhd/grade.py hartmann <case> --hartmann 20 \\
        --mean-velocity 1 --half-height 1 [--json]

`hartmann`: laminar flow of a conducting fluid along a channel with a uniform magnetic
field across it. The Lorentz force on a fully developed flow in a transverse field is a
linear drag, so the profile is a constant minus a cosh rather than a parabola:

    u(y)/u_c   = ( cosh(Ha) - cosh(Ha*y/L) ) / ( cosh(Ha) - 1 )            (2)
    u_c/u_mean = ( cosh(Ha) - 1 ) / ( cosh(Ha) - sinh(Ha)/Ha )             (3)
    Ha         = B0*L*sqrt(sigma/(rho*nu))

Both are exact, both are derived in `README.md` beside this file and again in
`openreynolds/toolbox/templates/mhd_hartmann.py`, which is the template that writes the
case, and the derivation starts from `mhdFoam`'s own two equations rather than from a
textbook's. There is no reference paper to misquote here and no reference uncertainty to
hide inside, which is the whole reason this case was chosen over a liquid-metal
experiment: `benchmarks/README.md` records that of its eleven published reference
numbers, re-checking two against their actual sources changed them, one by a factor of
two.

Ha, the mean velocity and the half-height are **required** rather than inferred. Ha
needs B0, sigma, rho and nu together; the mean velocity is the inlet condition; and the
half-height cannot be read off the sampled line, because the line the template writes
stops a thousandth of a half-height short of each wall and at Ha 20 the profile's slope
there is about 20 u_c per L, so inferring L from the samples would be a 2% error exactly
where the check is tightest. `heat.py` and `balance.py` refuse a missing constant by the
flag that would supply it for the same reason, and a profile graded at the wrong Ha
looks as measured as one graded at the right Ha.

What is read: the `raw` profile `postProcess -func sample` wrote under
`postProcessing/`, and nothing else. Never the run's own analysis.

The checks, and what each one is responsible for on its own:

* `samples` -- enough points to say anything, and a y range that reaches both walls.
* `symmetry` -- u(y) against u(-y). A flow still developing, or one sampled too near
  the inlet, is asymmetric about the centreline while its shape error can still be
  small; this separates "not developed" from "wrong profile".
* `shape_rms` / `shape_max` -- the sampled profile divided by its own centreline value
  against (2). No constant is fitted anywhere, so this is a check on the shape alone
  and passes or fails independently of the flow rate.
* `centreline` -- the measured centreline speed against `u_mean * (3)`, which is
  absolute. The template's inlet is a plug at u_mean and the case is 2D with both z
  faces `empty`, so continuity fixes the mean at every station exactly and this number
  has no fitted constant in it either.
* `flow_rate` -- the trapezoidal mean of the sampled profile against the stated
  u_mean, which is the assumption `centreline` rests on, checked rather than assumed.
* `parabola_contrast` -- the same sampled profile against the Ha = 0 parabola, which
  has to be very much worse. Without it, `shape_rms` is a check that could be passed by
  any plausible channel profile at low Ha, and `benchmarks/lbm/` makes the same
  argument about its own exactness claim: a bound met by the wrong physics too is not
  evidence. Reported without a verdict where Ha is small enough that the two curves are
  within the threshold of each other, since there the contrast does not exist to be
  measured.

Exit status: 0 when every check passes, 1 when one fails, 2 when the case cannot be
read.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

THRESHOLDS = {
    # Of the centreline speed. The profile is exact, so this is a discretisation bound
    # and not a reference uncertainty: the template's mesh is uniform at the ESI
    # tutorial's own dy of 0.05, which puts four cells inside the L/Ha Hartmann layer at
    # the default Ha of 5, and the scheme is second order. (An earlier version of this
    # comment said seven cells, which described the GRADED mesh -- the one that diverged,
    # because mhdFoam's explicit magnetic stresses grow as the cell shrinks.) The bound
    # is set where it is because the Ha = 0 parabola sits 0.366 RMS from the Ha = 20
    # profile -- twenty-four times this bound -- so a number this side of it cannot be
    # reached by the wrong physics.
    "shape_rms": 1.5e-2,
    "shape_max": 5.0e-2,
    # u_c/u_mean against (3), relative.
    "centreline_rel": 2.0e-2,
    # The sampled profile's own mean against the stated inlet mean, relative. Looser
    # than the shape because it is a trapezoidal integral of 400-odd interpolated
    # points over a graded mesh, so part of it is the sampler's quadrature.
    "flow_rate_rel": 3.0e-2,
    # max|u(y) - u(-y)| over the profile, as a fraction of u_c.
    "symmetry": 2.0e-2,
    # How much worse the parabola has to be than the closed form, as a ratio of RMS.
    "contrast_factor": 10.0,
    # Below this separation between the two curves there is no contrast to measure and
    # the check is reported rather than graded.
    "contrast_floor": 0.15,
}

MIN_SAMPLES = 50


class GradeError(Exception):
    """The case cannot be read, which is a different thing from failing."""


# -- the closed form ------------------------------------------------------------


def hartmann_profile(eta: np.ndarray, hartmann: float) -> np.ndarray:
    """u/u_c at y/L = eta, equation (2). Exact.

    The Ha -> 0 limit is written out rather than left to cancel: (2) is 0/0 there and
    evaluates to a division by zero in floating point rather than to the parabola it
    tends to.
    """
    eta = np.asarray(eta, dtype=float)
    if hartmann == 0.0:
        return 1.0 - eta * eta
    cosh_ha = math.cosh(hartmann)
    return (cosh_ha - np.cosh(hartmann * eta)) / (cosh_ha - 1.0)


def centreline_over_mean(hartmann: float) -> float:
    """u_c/u_mean, equation (3). 3/2 at Ha = 0, tending to Ha/(Ha-1) for large Ha."""
    if hartmann == 0.0:
        return 1.5
    cosh_ha = math.cosh(hartmann)
    return (cosh_ha - 1.0) / (cosh_ha - math.sinh(hartmann) / hartmann)


# -- reading the run ------------------------------------------------------------


def read_profile(case: Path) -> dict[str, Any]:
    """`{y, ux, time, path}` out of the newest `raw` profile `postProcess` wrote.

    The `raw` set writer puts a `#` header naming the columns on the first line and one
    row per sample point after it. `system/sample` in the template asks for `axis y` and
    the field `U`, so the columns are y and the three velocity components -- the column
    count is checked rather than assumed, because guessing at a column is how a grader
    comes to compare the wrong field.
    """
    root = case / "postProcessing"
    found = sorted(root.rglob("*.xy")) if root.is_dir() else []
    if not found:
        raise GradeError(
            f"no {root}/**/*.xy: the profile is written by\n"
            "    postProcess -func sample -noZero\n"
            "run in the case after the solver, with the `sample` dictionary the\n"
            "mhd_hartmann.py template writes into system/")

    def when(path: Path) -> float:
        try:
            return float(path.parent.name)
        except ValueError:
            return -1.0

    path = max(found, key=lambda p: (when(p), len(p.name)))
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(line.split())
    if not rows:
        raise GradeError(f"{path} has a header and no rows")
    width = len(rows[0])
    if width != 4:
        raise GradeError(
            f"{path} has {width} columns and a sampled vector along one axis has 4 "
            "(y, Ux, Uy, Uz). `fields ( U )` with `axis y` is what the template asks "
            "for; a case sampling `Ux` alone, or along `distance`, is a different file")
    data = np.array([[float(value) for value in row] for row in rows], dtype=float)
    order = np.argsort(data[:, 0])
    return {"y": data[order, 0], "ux": data[order, 1],
            "time": when(path), "path": str(path)}


# -- grading --------------------------------------------------------------------


def _check(name: str, value: Any, reference: Any, passed: bool | None,
           note: str = "") -> dict:
    """`passed=None` is "reported, not graded", which `benchmarks/sph/` uses for the
    dam break and which is the honest verdict when there is nothing to compare to."""
    return {"name": name, "value": value, "reference": reference,
            "passed": None if passed is None else bool(passed), "note": note}


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(math.sqrt(float(np.mean(values * values))))


def grade_hartmann(case: Path, hartmann: float, mean_velocity: float,
                   half_height: float) -> dict[str, Any]:
    """Every check, against the closed form and against the parabola beside it."""
    run = read_profile(case)
    y, ux = run["y"], run["ux"]
    checks: list[dict] = []

    checks.append(_check("samples", int(y.size), f">= {MIN_SAMPLES}",
                         y.size >= MIN_SAMPLES,
                         f"from {run['path']} at t = {run['time']:g}"))
    if y.size < MIN_SAMPLES:
        return {"case": str(case), "hartmann": hartmann, "checks": checks,
                "passed": False}

    eta = y / half_height
    # The *smaller* of the two reaches, not the largest |y|: half a channel reaches one
    # wall perfectly and is symmetric about its own end, so a max would pass it and
    # every check after it would then be grading half a profile against a whole closed
    # form. Capped above at 1 as well, since a line outside the channel means the
    # half-height supplied is not this case's.
    reach = min(-float(eta.min()), float(eta.max()))
    checks.append(_check("y_range", round(reach, 6), "0.95 to 1.0",
                         0.95 <= reach and float(np.max(np.abs(eta))) <= 1.0,
                         "the smaller of the two reaches, so half a channel fails here "
                         "rather than being graded as a whole one"))

    # The centreline value, by interpolation at eta = 0 rather than by taking a maximum:
    # the maximum of a sampled profile is the largest sample, which sits half a sample
    # spacing off the centre and is biased low by exactly the curvature there.
    centre = float(np.interp(0.0, eta, ux))
    if centre <= 0.0:
        return {"case": str(case), "hartmann": hartmann,
                "checks": checks + [_check("centre", centre, "> 0", False,
                                           "the centreline velocity is not positive, "
                                           "so there is no profile to normalise")],
                "passed": False}

    # Symmetry: u(y) against u(-y), on the sampled grid by reflection.
    mirrored = np.interp(-eta, eta, ux)
    asymmetry = float(np.max(np.abs(ux - mirrored))) / centre
    checks.append(_check("symmetry", round(asymmetry, 6),
                         f"< {THRESHOLDS['symmetry']:g}",
                         asymmetry < THRESHOLDS["symmetry"],
                         "max|u(y) - u(-y)|/u_c; an undeveloped flow fails here first"))

    exact = hartmann_profile(eta, hartmann)
    residual = ux / centre - exact
    shape_rms, shape_max = rms(residual), float(np.max(np.abs(residual)))
    checks.append(_check("shape_rms", round(shape_rms, 8),
                         f"< {THRESHOLDS['shape_rms']:g}",
                         shape_rms < THRESHOLDS["shape_rms"],
                         "u/u_c against (cosh Ha - cosh(Ha y/L))/(cosh Ha - 1), "
                         "no constant fitted"))
    checks.append(_check("shape_max", round(shape_max, 8),
                         f"< {THRESHOLDS['shape_max']:g}",
                         shape_max < THRESHOLDS["shape_max"]))

    # The flow rate the shape check cannot see, and the absolute centreline speed that
    # rests on it.
    # The trapezoid rule written out rather than called: `numpy.trapezoid` only exists
    # from numpy 2.0 and `numpy.trapz` is removed in it, and this repository supports
    # numpy >= 1.24 (`pyproject.toml:51`). `test_toolbox_panel.py:75` records the same
    # trap.
    measured_mean = float(np.sum(0.5 * (ux[1:] + ux[:-1]) * np.diff(eta))) / (
        eta[-1] - eta[0])
    rate_error = abs(measured_mean - mean_velocity) / mean_velocity
    checks.append(_check("flow_rate", round(measured_mean, 6), mean_velocity,
                         rate_error < THRESHOLDS["flow_rate_rel"],
                         "trapezoidal mean of the sampled profile against the inlet "
                         f"plug ({100.0 * rate_error:+.3g}%)"))

    expected_centre = mean_velocity * centreline_over_mean(hartmann)
    centre_error = abs(centre - expected_centre) / expected_centre
    checks.append(_check("centreline", round(centre, 6), round(expected_centre, 6),
                         centre_error < THRESHOLDS["centreline_rel"],
                         f"u_mean*(cosh Ha - 1)/(cosh Ha - sinh Ha/Ha), "
                         f"{100.0 * centre_error:+.3g}%"))

    # The contrast that stops the shape check being a tautology.
    parabola = hartmann_profile(eta, 0.0)
    parabola_rms = rms(ux / centre - parabola)
    separation = rms(exact - parabola)
    if separation < THRESHOLDS["contrast_floor"]:
        checks.append(_check(
            "parabola_contrast", round(parabola_rms, 6), None, None,
            f"reported, not graded: at Ha = {hartmann:g} the closed form is only "
            f"{separation:.4f} RMS from the parabola, so there is no contrast to "
            "measure. Ha = 20 puts them 0.366 apart"))
    else:
        factor = parabola_rms / max(shape_rms, 1e-15)
        checks.append(_check(
            "parabola_contrast", round(factor, 3),
            f"> {THRESHOLDS['contrast_factor']:g}",
            factor >= THRESHOLDS["contrast_factor"],
            f"the same profile is {parabola_rms:.4f} RMS from the Ha = 0 parabola "
            f"against {shape_rms:.6f} from the closed form, {factor:.1f}x"))

    graded = [check for check in checks if check["passed"] is not None]
    return {
        "case": str(case),
        "hartmann": hartmann,
        "half_height": half_height,
        "mean_velocity": mean_velocity,
        "time": run["time"],
        "samples": int(y.size),
        "centreline": centre,
        "centreline_over_mean": centre / measured_mean if measured_mean else None,
        "checks": checks,
        "profile": [{"eta": float(e), "u_over_uc": float(u / centre),
                     "exact": float(x)}
                    for e, u, x in zip(eta[::40], ux[::40], exact[::40])],
        "passed": all(check["passed"] for check in graded),
    }


# -- output ---------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def print_table(result: dict[str, Any]) -> None:
    print(f"hartmann: mhdFoam, Ha = {result['hartmann']:g}, "
          f"{result.get('samples', 0)} samples at t = {result.get('time', 0):g}")
    for check in result["checks"]:
        mark = {True: "PASS", False: "FAIL", None: "----"}[check["passed"]]
        line = f"  {mark}  {check['name']}: {_fmt(check['value'])}"
        if check["reference"] is not None:
            line += f" (reference {_fmt(check['reference'])})"
        if check.get("note"):
            line += f"  -- {check['note']}"
        print(line)
    if "profile" in result:
        print("  y/L       u/u_c     closed form")
        for point in result["profile"]:
            print(f"  {point['eta']:+.4f}   {point['u_over_uc']:.6f}  "
                  f"{point['exact']:.6f}")
    print("PASSED" if result["passed"] else "FAILED")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="case_name", required=True)
    hartmann = sub.add_parser("hartmann")
    hartmann.add_argument("case", type=Path)
    hartmann.add_argument("--hartmann", type=float, required=True,
                          help="B0*L*sqrt(sigma/(rho*nu)); not inferred, see the "
                               "module docstring")
    hartmann.add_argument("--mean-velocity", type=float, required=True,
                          dest="mean_velocity",
                          help="the inlet plug speed, m/s")
    hartmann.add_argument("--half-height", type=float, required=True,
                          dest="half_height", help="L, metres")
    hartmann.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = grade_hartmann(args.case, args.hartmann, args.mean_velocity,
                                args.half_height)
    except GradeError as error:
        print(f"grade.py: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_table(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
