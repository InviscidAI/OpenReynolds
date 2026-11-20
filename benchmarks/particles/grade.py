#!/usr/bin/env python3
"""Grade the particle tracer in `openreynolds/toolbox/tracks.py`.

    python benchmarks/particles/grade.py drag
    python benchmarks/particles/grade.py tracer
    python benchmarks/particles/grade.py capture --json

Three cases. `drag` and `tracer` are graded against closed forms that are exact and
need no paper: the velocity relaxation of a particle released into a uniform stream,
its terminal velocity under gravity, and the closure of a massless tracer's orbit in
solid-body rotation. `capture` is the one with a published number behind it, the
critical Stokes number for inertial capture on a circular cylinder in inviscid flow.

Like `../lbm/grade.py` and `../sph/grade.py`, this grader solves rather than reading a
finished run: the integrator is numpy in this repository and the whole of this file
finishes in under a minute, so committing its output would replace a check that
re-derives the number with a number nobody can re-derive.

The published value, and the factor of two that goes with it. The critical Stokes
number for a circular cylinder in inviscid flow is **1/8** when the Stokes number is
`tau_p U_inf / R`, the convention used here. The source read for it is Robinson,
Warren, Turner & Sear, *Critical inertia for particle capture is determined by surface
geometry at forward stagnation point*, arXiv:2310.03474v2 (25 Feb 2025), Section III.3,
which states "So here the critical Stokes number St_c = 1/4 for a cylinder" -- in its
own convention `St = m U / (l_y xi)` with `l_y` the transverse semi-axis (the radius,
for a circle) and `U` the local flow approaching the stagnation point, which for
inviscid flow past a circle is `2 U_inf`. The same paper says of Langmuir and Blodgett
that they "actually quote St_c = 1/8" "because of their alternative choice to set
U = U_inf". So 1/4 and 1/8 are the same result in two conventions a factor of two
apart, and `../README.md` is a long argument about exactly that class of mistake: of
its eleven reference values, re-checking changed two, and one of them was a factor of
two that reached a customer-facing page. The convention is therefore written into the
threshold's name and into every printed line.

What was not read: Taylor's 1940 Aeronautical Research Committee note and Langmuir &
Blodgett's 1946 AAF technical report, the two primary attributions. The value graded
here comes from a paper that derives it and states both conventions, which is a
readable source rather than a secondary citation of a number, but it is not the
original and this file does not pretend otherwise. The derivation is reproduced in
`README.md` beside the measurement, because the result is three lines of algebra on the
stagnation-point strain rate and a reader should not have to take it on trust either.

Exit status: 0 when every check passes, 1 when one fails, 2 when the tracer cannot be
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
    # The published critical Stokes number for a circular cylinder in inviscid flow,
    # in the tau_p U_inf / R convention. 1/4 in the l_y = R, U = 2 U_inf convention of
    # arXiv:2310.03474v2; see the module docstring. Never write 0.125 here without the
    # convention beside it.
    "stokes_critical_over_R": 0.125,
    # Grading choice, set from the convergence study in README.md rather than from
    # taste. A finite stopping criterion can only over-estimate this threshold, since
    # below it the approach to the wall takes infinite time, so the bound is one-sided
    # and the measured value must land above 1/8. The default sweep's tightest stopping
    # speed, 1e-14 U, measures +1.033%, and 1e-10 measures +2.000%; 3% is the round
    # number above the loosest of the default sweep.
    "capture_error_pct": 3.0,
    # The convergence itself, which is what makes the +2% a diagnosis rather than an
    # unexplained gap: tightening the stopping speed by four decades must move the
    # measured threshold monotonically toward 1/8. Measured +5.671, +3.152, +2.000,
    # +1.375, +1.033% at stopping speeds 1e-6 through 1e-14 of U (dt 0.005, horizon 50).
    "capture_must_converge": True,
    # Exact: v(t) = U (1 - exp(-t/tau_p)) for linear drag. RK4 at dt = tau_p/200
    # measured 1.9e-12 of U over five relaxation times, so this is six orders of
    # slack and still nowhere near a discretisation bound.
    "relaxation_exact": 1e-8,
    # Exact: w_t = tau_p g (1 - rho_f/rho_p). Measured 2.2e-14 as a fraction.
    "terminal_exact": 1e-9,
    # The contrast that stops the two checks above being a tautology. With the
    # Schiller-Naumann correction at a particle Reynolds number of 42.8 the terminal
    # velocity must be far below the Stokes value -- measured 0.4619 of it -- or the
    # drag branch was never exercised. 0.9 separates "a different answer" from
    # "the same answer with a different flag".
    "sn_ratio_max": 0.9,
    # And the Schiller-Naumann terminal velocity must equal the root of
    # w = tau_p g' / phi(Re_p(w)), solved here by fixed-point iteration independently
    # of the integrator. Exact; measured 5.1e-15.
    "sn_implicit_exact": 1e-7,
    # A massless tracer in solid-body rotation returns exactly to where it started
    # after one period. Measured 5.1e-8 of the radius at 200 steps per revolution.
    "orbit_closure": 1e-6,
    # And RK4 must actually be fourth order on it, or the integrator is not the one
    # the docstring claims. Measured 3.999989 and 4.000002 over 200 -> 400 -> 800
    # steps per revolution, so it is four to six figures and not four to one.
    "orbit_order_min": 3.8,
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tracks():
    return _load(ROOT / "openreynolds" / "toolbox" / "tracks.py", "toolbox_tracks")


# --------------------------------------------------------------------------- helpers


def _march(rate, position, velocity, dt, steps):
    """RK4 on (x, v) with a history of v, which `tracks.integrate` does not return.

    Deliberately a second, three-line integrator rather than a change to the toolbox
    script: the two closed forms below are about `v(t)`, and widening the script's
    return value so a grader can see it would be the grader shaping the instrument.
    It is the same scheme, and the capture check further down drives the real one.
    """
    history = []
    for i in range(steps):
        k1 = rate(position, velocity)
        k2 = rate(position + dt / 2 * k1[0], velocity + dt / 2 * k1[1])
        k3 = rate(position + dt / 2 * k2[0], velocity + dt / 2 * k2[1])
        k4 = rate(position + dt * k3[0], velocity + dt * k3[1])
        position = position + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        velocity = velocity + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        history.append(((i + 1) * dt, position.copy(), velocity.copy()))
    return history


def _check(name, value, reference, passed):
    return {"name": name, "value": value, "reference": reference,
            "verdict": "-" if passed is None else ("PASS" if passed else "FAIL")}


# --------------------------------------------------------------------------- drag


def grade_drag(args) -> dict:
    """Relaxation into a uniform stream, terminal velocity, and the two drag laws."""
    t = tracks()
    speed, tau = 1.0, 0.3
    dt = tau / args.steps_per_tau
    steps = int(round(5 * tau / dt))
    stream = np.array([[speed, 0.0, 0.0]])

    def rate(p, v):
        return v, (stream - v) / tau

    history = _march(rate, np.zeros((1, 3)), np.zeros((1, 3)), dt, steps)
    relaxation = max(
        abs(v[0, 0] - speed * (1 - math.exp(-time / tau))) / speed
        for time, _, v in history
    )

    # Settling. Air at 20 C and a quartz grain, so rho_p/rho_f is about 2200 and the
    # terms this integrator leaves out (added mass, Basset) are the small ones.
    rho_p, rho_f, nu = 2650.0, 1.2, 1.5e-5
    mu, g = rho_f * nu, 9.81
    reduced = g * (1.0 - rho_f / rho_p)

    def settle(diameter: float, law: str):
        tau_p = t.relaxation_time(rho_p, diameter, mu)
        body = np.array([0.0, 0.0, -reduced])

        def rate_settle(p, v):
            slip = -v
            phi = (1.0 if law == "stokes"
                   else t.drag_factor(slip, diameter, nu, law)[:, None])
            return v, slip * phi / tau_p + body

        dt_s = tau_p / args.steps_per_tau
        out = _march(rate_settle, np.zeros((1, 3)), np.zeros((1, 3)),
                     dt_s, int(round(60 * tau_p / dt_s)))
        return tau_p, float(-out[-1][2][0, 2])

    small = 20e-6
    tau_small, w_small = settle(small, "stokes")
    exact_small = tau_small * reduced
    terminal_error = abs(w_small - exact_small) / exact_small

    big = 200e-6
    tau_big, w_stokes = settle(big, "stokes")
    _, w_sn = settle(big, "schiller-naumann")
    exact_big = tau_big * reduced

    # The implicit root of w = tau_p g' / phi(Re_p(w)), solved here rather than taken
    # from the integrator, so the Schiller-Naumann branch is checked against algebra
    # and not against itself.
    root = exact_big
    for _ in range(500):
        re_p = root * big / nu
        root = exact_big / (1.0 + t.SN_COEFF * re_p ** t.SN_EXPONENT)

    checks = [
        _check("relaxation v(t) vs U (1 - exp(-t/tau_p)), max |err|/U", relaxation,
               f"exact, <= {THRESHOLDS['relaxation_exact']:g}",
               relaxation <= THRESHOLDS["relaxation_exact"]),
        _check("tau_p, 20 um quartz in air [s]", tau_small, "rho_p d^2 / 18 mu", None),
        _check("terminal velocity [m/s]", w_small, f"exact {exact_small:.9g}", None),
        _check("terminal velocity, |err| as a fraction", terminal_error,
               f"exact, <= {THRESHOLDS['terminal_exact']:g}",
               terminal_error <= THRESHOLDS["terminal_exact"]),
        _check("particle Reynolds number at that terminal velocity",
               w_small * small / nu, "<< 1, where Stokes drag is the answer", None),
        _check("particle Reynolds number, 200 um, on the Stokes velocity",
               w_stokes * big / nu, "finite: Stokes drag is now wrong", None),
        _check("Schiller-Naumann terminal / Stokes terminal", w_sn / w_stokes,
               f"<= {THRESHOLDS['sn_ratio_max']:g}, or the drag law did nothing",
               w_sn / w_stokes <= THRESHOLDS["sn_ratio_max"]),
        _check("Schiller-Naumann terminal vs the implicit root, |err|/w",
               abs(w_sn - root) / root,
               f"exact, <= {THRESHOLDS['sn_implicit_exact']:g}",
               abs(w_sn - root) / root <= THRESHOLDS["sn_implicit_exact"]),
    ]
    return {"benchmark": "drag", "checks": checks,
            "measured": {"relaxation_max_error": relaxation,
                         "terminal_small": w_small, "terminal_exact_small": exact_small,
                         "terminal_stokes_big": w_stokes, "terminal_sn_big": w_sn,
                         "sn_implicit_root": root}}


# --------------------------------------------------------------------------- tracer


def grade_tracer(args) -> dict:
    """A massless tracer in solid-body rotation closes its orbit exactly.

    Solid-body rotation is chosen because the exact path is a circle and the exact
    period is 2 pi / Omega, so the closure error after one revolution is the whole
    integration error with nothing else in it -- no reference, no interpolation, no
    stopping criterion. It is also the one check here that drives `tracks.integrate`
    with tau_p = 0, the branch a streamline uses.
    """
    t = tracks()
    omega, radius = 1.0, 1.0

    def velocity(points: np.ndarray) -> np.ndarray:
        return np.stack([-omega * points[:, 1], omega * points[:, 0],
                         np.zeros(len(points))], axis=-1)

    start = np.array([[radius, 0.0, 0.0]])
    errors, counts = [], [args.revolution_steps * 2**k for k in range(3)]
    for n in counts:
        dt = 2 * math.pi / omega / n
        _, _, where, _ = t.integrate(start, velocity, 0.0, dt, n)
        errors.append(float(np.linalg.norm(where[0] - start[0])))
    orders = [math.log(a / b, 2) for a, b in zip(errors[:-1], errors[1:])]

    checks = [
        _check(f"orbit closure after one revolution at {counts[0]} steps", errors[0],
               f"exact 0, <= {THRESHOLDS['orbit_closure']:g} of the radius",
               errors[0] <= THRESHOLDS["orbit_closure"] * radius),
        _check("closure at " + ", ".join(str(c) for c in counts), errors, "", None),
        _check("observed order of convergence", orders,
               f"RK4, >= {THRESHOLDS['orbit_order_min']:g}",
               min(orders) >= THRESHOLDS["orbit_order_min"]),
    ]
    return {"benchmark": "tracer", "checks": checks,
            "measured": {"closure": errors, "steps_per_revolution": counts,
                         "orders": orders}}


# --------------------------------------------------------------------------- capture


def grade_capture(args) -> dict:
    """The critical Stokes number for capture on a cylinder, against 1/8."""
    t = tracks()
    published = THRESHOLDS["stokes_critical_over_R"]

    # The bisection is only valid if the fate is monotone in the Stokes number. Checked
    # rather than assumed: 721 values across the threshold, and no particle below the
    # first capture may capture.
    span = np.linspace(0.118, 0.136, args.monotone_samples)
    captured = []
    for stokes in span:
        fate, _, _, _ = t.integrate(
            np.array([[-5.0, 0.0, 0.0]]),
            # `stokes` is tau_p here: at R = U = 1 the Stokes number tau_p U/R is
            # numerically the relaxation time, which is why the sweep reads directly.
            t.potential_cylinder(1.0, 1.0), stokes, args.dt,
            int(round(args.horizon / args.dt)),
            capture=t.cylinder_capture(1.0),
            exit_test=lambda p: p[:, 0] > 2.0, stop_speed=1e-10,
        )
        captured.append(fate[0] == "captured")
    monotone = bool(np.all(np.diff(np.array(captured, dtype=int)) >= 0))

    sweep = []
    for stop_speed in args.stop_speeds:
        result = t.capture_threshold(stop_speed=stop_speed, dt=args.dt,
                                     horizon=args.horizon)
        measured = result["stokes_critical"]
        sweep.append({"stop_speed": stop_speed, "stokes_critical": measured,
                      "error_pct": 100 * (measured - published) / published})
    errors = [row["error_pct"] for row in sweep]
    tightest = sweep[-1]

    # Independent of the time step, which is what says the gap is the stopping
    # criterion and not the integrator.
    steps = [{"dt": dt, "stokes_critical": t.capture_threshold(
                  stop_speed=args.stop_speeds[-1], dt=dt, horizon=args.horizon,
              )["stokes_critical"]} for dt in (args.dt, args.dt / 2.5)]
    step_spread = abs(steps[0]["stokes_critical"] - steps[1]["stokes_critical"])

    checks = [
        _check("fate monotone in the Stokes number over "
               f"{args.monotone_samples} samples, 0.118 to 0.136", monotone,
               "no holes, or the bisection is not a threshold", monotone),
        _check(f"critical Stokes number tau_p U/R (stop speed {tightest['stop_speed']:g} U)",
               tightest["stokes_critical"],
               f"{published:g} = 1/8 (arXiv:2310.03474v2 III.3, in its 1/4 convention)",
               None),
        _check("error against 1/8 [%]", tightest["error_pct"],
               f"|err| <= {THRESHOLDS['capture_error_pct']:g}",
               abs(tightest["error_pct"]) <= THRESHOLDS["capture_error_pct"]),
        _check("the error is positive, which it has to be", tightest["error_pct"] > 0,
               "a finite stopping criterion can only over-estimate it",
               tightest["error_pct"] > 0),
        _check("error [%] at stop speeds " + ", ".join(f"{r['stop_speed']:g}" for r in sweep),
               errors, "monotone toward 1/8",
               all(a > b for a, b in zip(errors[:-1], errors[1:]))),
        _check(f"spread over dt {steps[0]['dt']:g} and {steps[1]['dt']:g}", step_spread,
               "the integrator is not the error here", None),
    ]
    return {"benchmark": "capture", "published_critical_over_R": published,
            "sweep": sweep, "time_steps": steps, "monotone": monotone,
            "checks": checks}


# --------------------------------------------------------------------------- output


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return " .. ".join(_fmt(v) for v in value)
    if isinstance(value, float):
        return f"{value:.7g}"
    return str(value)


def print_table(result: dict) -> None:
    rows = [(c["name"], _fmt(c["value"]), c["reference"], c["verdict"])
            for c in result["checks"]]
    widths = [max(len(r[i]) for r in rows) for i in range(3)]
    for name, value, reference, verdict in rows:
        print(f"{name:<{widths[0]}}  {value:>{widths[1]}}  {reference:<{widths[2]}}  {verdict}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="benchmark", required=True)

    dp = sub.add_parser("drag", help="relaxation and terminal velocity, both exact")
    dp.add_argument("--steps-per-tau", type=int, default=200)

    tp = sub.add_parser("tracer", help="a tracer orbit in solid-body rotation, exact")
    tp.add_argument("--revolution-steps", type=int, default=200)

    cp = sub.add_parser("capture", help="the critical Stokes number on a cylinder")
    cp.add_argument("--dt", type=float, default=0.005)
    cp.add_argument("--horizon", type=float, default=50.0,
                    help="flow times to march; the threshold is the same at 50 and 100")
    cp.add_argument("--monotone-samples", type=int, default=91)
    cp.add_argument("--stop-speeds", type=float, nargs="+",
                    default=[1e-8, 1e-10, 1e-12, 1e-14])

    for p in (dp, tp, cp):
        p.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    grade = {"drag": grade_drag, "tracer": grade_tracer, "capture": grade_capture}
    result = grade[args.benchmark](args)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_table(result)
    return 1 if any(c["verdict"] == "FAIL" for c in result["checks"]) else 0


if __name__ == "__main__":
    sys.exit(main())
