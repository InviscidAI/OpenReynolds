#!/usr/bin/env python3
"""Grade the smoothed-particle hydrodynamics solver in `openreynolds/toolbox/sph.py`.

    python benchmarks/sph/grade.py hydrostatic
    python benchmarks/sph/grade.py sloshing
    python benchmarks/sph/grade.py dambreak --json

Three cases, two of them graded against closed-form answers and the third reported
without a verdict. Which is which, and why, is the whole subject of
`benchmarks/sph/README.md`; the short version is that the canonical dam-break reference
could not be read, so it is not quoted here as though it had been.

`hydrostatic`: still water, p = rho g (h - y), exact. `sloshing`: the first-mode
standing wave in a closed tank, period 2 pi / sqrt(g k tanh(k h)) with k = pi/L, the
linear water-wave dispersion relation (Lamb, *Hydrodynamics*, 6th ed., art. 228).
`dambreak`: a collapsing column, whose surge front is reported in Martin & Moyce's
dimensionless form and graded only on the two conditions that decide whether the
*scheme* was valid during the run -- that the walls contained the fluid, and that the
weakly compressible assumption held.

Like `lbm/grade.py` beside it, this grader solves rather than reading a finished run:
the solver is in this repository, costs seconds to minutes, and committing its output
would replace a check that re-derives the number with a number nobody can re-derive.

Exit status: 0 when every check passes, 1 when one fails, 2 when the solver cannot be
loaded.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

THRESHOLDS = {
    # Set from the measurement, not from taste: the figure each run produced is in the
    # note beside its check and in benchmarks/sph/README.md, and each bound here is the
    # round number above it. They are loose compared with the lattice Boltzmann grades
    # next door, and that is the honest statement about the method -- a 2D weakly
    # compressible SPH with dynamic boundary particles at dx = h/20 is a percents-level
    # scheme, and a tighter threshold would be a threshold tuned to one run on one
    # machine rather than a property of the solver.
    "hydrostatic_rms": 0.05,     # RMS pressure error as a fraction of rho g h
    "hydrostatic_max": 0.12,     # and the worst single particle
    "period_rel": 0.08,          # sloshing period against the linear dispersion relation
    "containment": 1.0,          # fluid may not pass a wall by more than one dx
    "density": 0.05,             # max rho/rho0 - 1 while the run is weakly compressible
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - a missing file
        raise SystemExit(f"grade.py: cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def solver():
    return _load(ROOT / "openreynolds" / "toolbox" / "sph.py", "graded_sph")


def _check(name: str, value: Any, reference: Any, passed: bool, note: str = "") -> dict:
    return {"name": name, "value": value, "reference": reference, "passed": bool(passed),
            "note": note}


def _cost(result: dict[str, Any]) -> str:
    return (f"{result['fluid_particles']:,} fluid + "
            f"{result['particles'] - result['fluid_particles']:,} wall particles, "
            f"{result['steps']:,} steps of {result['dt']:.2e} s in "
            f"{result['seconds']:.1f} s ({result['steps_per_second']:.0f} steps/s)")


def _density_check(sph, result: dict[str, Any]) -> dict:
    """Whether the run stayed inside its own assumption.

    Tait's equation of state is a linearisation about rho0 dressed up as a seventh
    power; at a few percent compression the pressure it returns is still the pressure of
    water, and well past that it is the pressure of a numerical fluid. A run that
    exceeded this did not fail to match a reference -- it stopped being a simulation of
    water, and no agreement it reports afterwards means anything.
    """
    excess = max(row["rho_max"] for row in result["history"]) / sph.RHO0 - 1.0
    return _check("max density excess rho/rho0 - 1", excess,
                  f"<= {THRESHOLDS['density']}", excess <= THRESHOLDS["density"],
                  "beyond a few percent the equation of state is not water's any more")


def _containment_check(sph, result: dict[str, Any], floor: float = 0.0) -> dict:
    """Whether the walls held. Dynamic boundary particles are the cheap boundary
    treatment, and this is the failure they have: a fast particle passes through a wall
    that is thinner than the kernel support, and the run continues without complaint."""
    fluid = result["pos"][result["moving"]]
    depth = float(floor - fluid[:, 1].min()) / result["dx"]
    through_left = float(-fluid[:, 0].min()) / result["dx"]
    worst = max(depth, through_left)
    return _check("deepest wall penetration, in particle spacings", worst,
                  f"<= {THRESHOLDS['containment']}", worst <= THRESHOLDS["containment"],
                  "three wall layers against a 2h = 2.6 dx kernel support")


# -- hydrostatic --------------------------------------------------------------------


def grade_hydrostatic(dx: float | None = None, end: float = 1.0) -> dict[str, Any]:
    """Still water. Everything this case produces is error, which is what makes it the
    sharpest cheap check of the pressure in the whole solver."""
    sph = solver()
    run = sph.hydrostatic(dx=dx, end=end, progress=False)
    error = np.abs(run["p_measured"] - run["p_exact"]) / run["p_scale"]
    rms = float(math.sqrt(float((error ** 2).mean())))
    worst = float(error.max())
    speed = float(np.abs(run["vel"][run["moving"]]).max())

    checks = [
        _check("RMS pressure error / rho g h", rms,
               f"<= {THRESHOLDS['hydrostatic_rms']}", rms <= THRESHOLDS["hydrostatic_rms"],
               f"{run['sampled']} particles in the middle half of the tank"),
        _check("worst particle pressure error / rho g h", worst,
               f"<= {THRESHOLDS['hydrostatic_max']}", worst <= THRESHOLDS["hydrostatic_max"],
               "the worst is at the free surface, where the kernel support is "
               "truncated and there is no fluid above to sum"),
        _density_check(sph, run),
        _containment_check(sph, run),
    ]
    return {
        "case": "hydrostatic", "passed": all(c["passed"] for c in checks), "checks": checks,
        "notes": [
            _cost(run),
            f"depth {run['depth']:g} m, tank {run['length']:g} m wide, "
            f"dx = {run['dx']:g} m, c0 = {run['c0']:.2f} m/s",
            f"the exact answer is rest, and the largest residual velocity anywhere in "
            f"the fluid is {speed:.4f} m/s against sqrt(g h) = "
            f"{math.sqrt(sph.G * run['depth']):.3f} m/s",
            "the middle half of the tank only: the ends carry the wall's own error, "
            "which the sloshing case measures as a period shift instead",
        ],
    }


# -- sloshing -----------------------------------------------------------------------


def grade_sloshing(dx: float | None = None, periods: float = 3.0) -> dict[str, Any]:
    """The linear standing-wave period, which is closed form and is the one dynamic
    check here with an exact answer.

    It is a real test of the boundary treatment and not only of the time integration:
    the period depends on k = pi/L, so a wall that holds the fluid a fraction of a
    particle spacing away from where the tank says it is changes L and changes the
    period. That is the error dynamic boundary particles have, and this measures it.
    """
    sph = solver()
    run = sph.sloshing(dx=dx, periods=periods, progress=False)
    measured, exact = run["period_measured"], run["period_exact"]
    relative = abs(measured / exact - 1.0) if math.isfinite(measured) else math.inf

    checks = [
        _check("period against sqrt(g k tanh(k h))", measured, exact,
               relative <= THRESHOLDS["period_rel"],
               f"{relative * 100:+.2f}% over {run['cycles']:.2f} cycles in the window"
               if math.isfinite(measured) else "no oscillation was measured"),
        _density_check(sph, run),
        _containment_check(sph, run),
    ]
    return {
        "case": "sloshing", "passed": all(c["passed"] for c in checks), "checks": checks,
        "notes": [
            _cost(run),
            f"tank {run['length']:g} m long, {run['depth']:g} m deep, first mode "
            f"k = pi/L = {run['k']:.4f} 1/m, omega = {run['omega_exact']:.4f} rad/s, "
            f"period {exact:.5f} s",
            f"amplitude {run['amplitude']:g} m, which is {run['amplitude'] / run['depth'] * 100:.1f}% "
            "of the depth: the reference is linear theory, so an amplitude large enough "
            "to be easy to measure would be compared against the wrong formula",
            "released from the mode's own velocity field with a flat surface, so the "
            "particle lattice starts uniform; displacing the surface instead would "
            "measure the disturbed lattice as well as the period",
            "measured from the fluid's horizontal centroid rather than a surface probe "
            "-- every particle contributes to it, and it does not care which particle "
            "happens to be highest when the sample is taken",
        ],
    }


# -- the dam break ------------------------------------------------------------------


def grade_dambreak(dx: float | None = None) -> dict[str, Any]:
    """The requested case, graded only on whether the scheme stayed valid.

    The surge front is the quantity a dam break is for and it is reported here without a
    verdict, because the reference it would be graded against could not be read. See
    `benchmarks/sph/README.md`. `benchmarks/README.md` already carries two rows that do
    this -- an amplitude with no published value for the configuration, and a lift
    coefficient that needs resolution the drag does not -- and one row whose reference
    was quoted from a secondary source and turned out to be wrong by a factor of two,
    which is the outcome this is avoiding.
    """
    sph = solver()
    run = sph.dambreak(dx=dx, progress=False)
    checks = [_containment_check(sph, run), _density_check(sph, run)]
    front = run["front"]
    # A handful of stations rather than all 400 samples: the curve is in the png and the
    # numbers worth reading in a log are the ones a reader can compare to a figure.
    stations = [row for index, row in enumerate(front)
                if index % max(1, len(front) // 8) == 0 or index == len(front) - 1]
    return {
        "case": "dambreak", "passed": all(c["passed"] for c in checks), "checks": checks,
        "front": [{"T": round(row["T"], 4), "Z": round(row["Z"], 4)} for row in stations],
        "notes": [
            _cost(run),
            f"column {run['column']:g} m wide and {run['aspect']:g} times as tall "
            f"(Martin & Moyce's n = 2), tank {run['tank_length']:g} m long",
            "Z = x/a against T = t sqrt(2g/a) is reported above and NOT graded: the "
            "experimental reference for it is Martin & Moyce (1952), Phil. Trans. R. "
            "Soc. A 244, 312-324, and the publisher refused every attempt to read it, "
            "so there is no primary source here to check a number against",
            "the run stops before the front reaches the far wall, so no reflection is "
            "in the reported curve",
        ],
    }


# -- output -------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def print_table(result: dict[str, Any]) -> None:
    print(f"{result['case']}: sph.py")
    for check in result["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        line = (f"  {mark}  {check['name']}: {_fmt(check['value'])} "
                f"(reference {_fmt(check['reference'])})")
        if check.get("note"):
            line += f"  -- {check['note']}"
        print(line)
    for note in result.get("notes", []):
        print(f"  note: {note}")
    if "front" in result:
        print("  surge front, reported and not graded:")
        print("    T = t sqrt(2g/a)    Z = x/a")
        for row in result["front"]:
            print(f"    {row['T']:>16.4f}    {row['Z']:.4f}")
    print("PASSED" if result["passed"] else "FAILED")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="which", required=True)
    for name in ("hydrostatic", "sloshing", "dambreak"):
        case = sub.add_parser(name)
        case.add_argument("--dx", type=float, default=None,
                          help="particle spacing in metres (each case has its own default)")
        case.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    graders = {"hydrostatic": grade_hydrostatic, "sloshing": grade_sloshing,
               "dambreak": grade_dambreak}
    result = graders[args.which](dx=args.dx)
    if args.json:
        print(json.dumps(result, indent=2, default=float))
    else:
        print_table(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
