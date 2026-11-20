#!/usr/bin/env python3
"""Grade the lattice Boltzmann solver in `openreynolds/toolbox/lbm.py`.

    python benchmarks/lbm/grade.py poiseuille
    python benchmarks/lbm/grade.py cavity --n 128
    python benchmarks/lbm/grade.py cavity --json

`poiseuille`: plane Poiseuille flow against the closed-form parabola, which is exact and
needs no paper. `cavity`: the lid-driven cavity at Re 1000 against Ghia, Ghia & Shin
(1982), *J. Comput. Phys.* **48**, 387-411, Table I (u along the vertical centreline) and
Table II (v along the horizontal centreline).

This grader solves, where `felix/grade.py` and `moving_mesh/grade.py` read a finished
run. That is a difference worth stating rather than hiding: those two grade an OpenFOAM
or Felix case that cost a cluster an hour, so the run is done once and trimmed output is
committed under `tests/data/`. This solver is 400 lines of numpy in this repository and
both cases finish on the machine running the tests, so committing its output would be
committing a number nobody can re-derive, in place of a check that re-derives it every
time. Nothing here reads a case directory and nothing is cached.

Ghia's table is imported from `benchmarks/felix/grade.py` rather than typed again. Two
copies of a reference table is two things to keep right, and the one already in this
repository has been through the re-checking described in `benchmarks/README.md` -- which
changed two of the eleven reference values in that file, so it is not a hypothetical
risk. Importing it also means the two solvers are graded against the same 34 numbers
with the same threshold, which is the only way the comparison between them means
anything.

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
    # The same bound `felix/grade.py` applies, deliberately: RMS of (u - u_Ghia) over
    # Ghia's 17 points as a fraction of the lid speed, 2%. A new solver that invented
    # its own threshold would be reporting that it had met a bar it chose.
    "cavity_rms": 0.02,
    # Re is a setting, not a result.
    "reynolds_rel": 0.01,
    # Poiseuille is exact for this scheme, so the bar is not a discretisation error --
    # it is "indistinguishable from the closed form". The run below measures around
    # 1e-10 of u_max, four orders inside this, and the same scheme with one relaxation
    # time instead of two measures worse than 1e-4 at the same resolution. 1e-6
    # separates those two without being a number tuned to the answer.
    "poiseuille_exact": 1e-6,
    # The BGK contrast, which is what stops the check above being a tautology: a
    # one-relaxation-time run away from its one exact viscosity must be measurably
    # wrong, or the exactness above is not evidence of anything.
    "bgk_wrong": 1e-4,
    # A run that stopped on its step ceiling rather than on its residual is not a
    # converged answer and is not graded as one.
    "residual": 1e-6,
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
    return _load(ROOT / "openreynolds" / "toolbox" / "lbm.py", "graded_lbm")


def ghia() -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    """Ghia's Table I and Table II, from the copy `felix/grade.py` already carries."""
    felix = _load(ROOT / "benchmarks" / "felix" / "grade.py", "graded_felix")
    return felix.GHIA_RE1000_U, felix.GHIA_RE1000_V


def _check(name: str, value: Any, reference: Any, passed: bool, note: str = "") -> dict:
    return {"name": name, "value": value, "reference": reference, "passed": bool(passed),
            "note": note}


def rms(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(math.sqrt(float(np.mean(d * d))))


def _cost(result: dict[str, Any]) -> str:
    return (f"{result['steps']:,} steps in {result['seconds']:.1f} s at "
            f"{result['mlups']:.2f} million lattice updates per second "
            f"(residual {result['residual']:.2e})")


# -- Poiseuille ---------------------------------------------------------------------


def grade_poiseuille(n: int = 64, viscosities: tuple[float, ...] = (0.02, 0.1, 0.4)
                     ) -> dict[str, Any]:
    """The exact parabola, at three viscosities a factor of twenty apart.

    Three rather than one because the property being tested is that the error does *not*
    depend on the viscosity. One viscosity cannot distinguish "exact" from "exact at this
    tau", which is precisely the difference between the two relaxation schemes.
    """
    lbm = solver()
    checks: list[dict] = []
    notes: list[str] = []
    limit = THRESHOLDS["poiseuille_exact"]

    for nu in viscosities:
        run = lbm.poiseuille(n=n, nu=nu, tolerance=1e-12, check_every=200, progress=False)
        error = float(np.abs(run["u_profile"] - run["u_exact"]).max() / run["u_max_exact"])
        checks.append(_check(
            f"max |u - u_exact| / u_max, nu = {nu:g} (TRT)", error, f"<= {limit}",
            error <= limit and run["converged"],
            f"tau_plus {1.0 / run['omega_plus']:.4f}, {_cost(run)}"))
        notes.append(f"nu = {nu:g}: {_cost(run)}")

    # The contrast. BGK satisfies the 3/16 condition at exactly one viscosity,
    # nu = sqrt(3)/12 = 0.14434, and the run four times away from it has moved its wall
    # off the link midpoint. Without this the check above would pass on a grader that
    # compared the solver to itself.
    magic = math.sqrt(3.0) / 12.0
    bgk_off = lbm.poiseuille(n=n, nu=4.0 * magic, tolerance=1e-12, check_every=200,
                             scheme="bgk", progress=False)
    off_error = float(np.abs(bgk_off["u_profile"] - bgk_off["u_exact"]).max()
                      / bgk_off["u_max_exact"])
    bgk_at = lbm.poiseuille(n=n, nu=magic, tolerance=1e-12, check_every=200,
                            scheme="bgk", progress=False)
    at_error = float(np.abs(bgk_at["u_profile"] - bgk_at["u_exact"]).max()
                     / bgk_at["u_max_exact"])
    checks += [
        _check("max error, one relaxation time at nu = 4*sqrt(3)/12", off_error,
               f">= {THRESHOLDS['bgk_wrong']}", off_error >= THRESHOLDS["bgk_wrong"],
               "BGK away from its one exact viscosity has to be wrong, or the TRT "
               "result above is not evidence"),
        _check("max error, one relaxation time at nu = sqrt(3)/12", at_error,
               f"<= {limit}", at_error <= limit,
               "where (tau - 1/2)^2 = 3/16, BGK is TRT and is exact too"),
    ]
    return {
        "case": "poiseuille", "n": n,
        "passed": all(c["passed"] for c in checks), "checks": checks,
        "notes": notes + [
            "u(y) = G/(2 mu) y (H - y) exactly, with G the body force per unit volume "
            "and mu = rho nu; the force is set so u_max is 0.05 in lattice units, which "
            "keeps the scheme's O(Ma^2) compressibility error small",
            "the walls are at the link midpoints, half a cell outside the first and "
            "last fluid row, which is what `lbm.centres` encodes",
        ],
    }


# -- the cavity ---------------------------------------------------------------------


def centreline_profiles(run: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """u at Ghia's 17 y-stations and v at his 17 x-stations, as fractions of the lid.

    The lattice carries node values at cell centres and the walls carry the boundary
    condition, so the profile handed to the comparison is the nodes with the two wall
    values on the ends -- u = 1 at the lid and 0 at the stationary wall, v = 0 at both
    side walls -- and is then interpolated to Ghia's stations. That is the same
    comparison `felix/grade.py` makes, which samples its continuous field a nanometre
    inside the wall and so reads the boundary condition there too. The interpolation is
    linear and the only interval where that is a real approximation is the half cell
    between the last node and the wall: 0.4% of the cavity at n = 128.
    """
    table_u, table_v = ghia()
    nodes = run["coordinate"]
    y = np.concatenate([[0.0], nodes, [1.0]])
    u = np.concatenate([[0.0], run["u_centreline"], [1.0]])
    v = np.concatenate([[0.0], run["v_centreline"], [0.0]])
    stations_u = np.array([point for point, _ in table_u])
    stations_v = np.array([point for point, _ in table_v])
    return np.interp(stations_u, y, u), np.interp(stations_v, y, v)


def grade_cavity(n: int = 128, re: float = 1000.0, tolerance: float = 1e-6
                 ) -> dict[str, Any]:
    lbm = solver()
    table_u, table_v = ghia()
    run = lbm.cavity(n=n, re=re, tolerance=tolerance, check_every=500, progress=False)
    u, v = centreline_profiles(run)
    u_ref = np.array([value for _, value in table_u])
    v_ref = np.array([value for _, value in table_v])
    limit = THRESHOLDS["cavity_rms"]
    u_err, v_err = rms(u, u_ref), rms(v, v_ref)
    reynolds = run["u_lid"] * n / run["nu"]

    checks = [
        _check("Reynolds number", reynolds, re,
               abs(reynolds - re) <= THRESHOLDS["reynolds_rel"] * re),
        _check("converged", run["residual"], f"<= {THRESHOLDS['residual']}",
               bool(run["converged"]) and run["residual"] <= THRESHOLDS["residual"],
               _cost(run)),
        _check("u RMS error (vertical centreline)", u_err, f"<= {limit}", u_err <= limit,
               f"max |du| {np.abs(u - u_ref).max():.4f}"),
        _check("v RMS error (horizontal centreline)", v_err, f"<= {limit}", v_err <= limit,
               f"max |dv| {np.abs(v - v_ref).max():.4f}"),
    ]
    return {
        "case": "cavity", "n": n, "re": re,
        "passed": all(c["passed"] for c in checks), "checks": checks,
        "profiles": {
            "u": [{"y": point, "ghia": value, "lbm": float(mine)}
                  for (point, value), mine in zip(table_u, u)],
            "v": [{"x": point, "ghia": value, "lbm": float(mine)}
                  for (point, value), mine in zip(table_v, v)],
        },
        "notes": [
            _cost(run),
            f"nu = u_lid*n/Re = {run['nu']:.6g} lattice units, tau_plus = "
            f"{1.0 / run['omega_plus']:.4f}; the scheme is TRT because at this tau a "
            "single relaxation time is close to the 1/2 where it loses stability",
            "Ghia's own error is larger than the agreement being measured here: his "
            "Re 1000 centreline extrema sit 1.46% to 2.20% from Botella & Peyret's "
            "(1998) spectral benchmark, so agreement much inside 2% is agreement to "
            "inside his uncertainty and part of it is luck. The same caveat applies to "
            "the Felix row in benchmarks/README.md and for the same reason",
            f"{n} cells across, against Ghia's 129x129 grid, so neither side of this "
            "comparison is the converged answer",
        ],
    }


# -- output -------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def print_table(result: dict[str, Any]) -> None:
    print(f"{result['case']}: lbm.py, n = {result['n']}")
    for check in result["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        line = (f"  {mark}  {check['name']}: {_fmt(check['value'])} "
                f"(reference {_fmt(check['reference'])})")
        if check.get("note"):
            line += f"  -- {check['note']}"
        print(line)
    for note in result.get("notes", []):
        print(f"  note: {note}")
    if "profiles" in result:
        print("  y       u Ghia    u lbm    |  x       v Ghia    v lbm")
        for pu, pv in zip(result["profiles"]["u"], result["profiles"]["v"]):
            print(f"  {pu['y']:.4f}  {pu['ghia']:+.5f}  {pu['lbm']:+.5f}  |  "
                  f"{pv['x']:.4f}  {pv['ghia']:+.5f}  {pv['lbm']:+.5f}")
    print("PASSED" if result["passed"] else "FAILED")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="which", required=True)
    poiseuille = sub.add_parser("poiseuille")
    poiseuille.add_argument("--n", type=int, default=64)
    poiseuille.add_argument("--json", action="store_true")
    cavity = sub.add_parser("cavity")
    cavity.add_argument("--n", type=int, default=128)
    cavity.add_argument("--re", type=float, default=1000.0)
    cavity.add_argument("--tolerance", type=float, default=1e-6)
    cavity.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.which == "poiseuille":
        result = grade_poiseuille(n=args.n)
    else:
        result = grade_cavity(n=args.n, re=args.re, tolerance=args.tolerance)
    if args.json:
        print(json.dumps(result, indent=2, default=float))
    else:
        print_table(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
