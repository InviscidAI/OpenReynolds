#!/usr/bin/env python3
"""Particle and tracer paths through a finished Eulerian field, and their fates.

    python3 tracks.py cylinder --stokes 0.2 --seeds 41 --out tracks.png
    python3 tracks.py cylinder --threshold --stop-speed 1e-10 --json
    python3 tracks.py case <case> --diameter 50e-6 --density 2650 --seeds 200 --json

Two sources of velocity, one integrator. `cylinder` is the closed-form 2D potential
flow past a circular cylinder, which is here because it is the one particle-tracking
configuration with a published capture threshold to grade against (see
`benchmarks/particles/`). `case` samples a finished OpenFOAM case through the same
reader `render.py` uses, so this is post-processing over a solved Eulerian field and
needs nothing in the image that is not already there.

Why post-processing and not a parcel cloud. The workspace image does carry Lagrangian
solvers -- `DPMFoam`, `DPMDyMFoam`, `MPPICFoam`, `MPPICDyMFoam` are in `$FOAM_APPBIN`
on the instance (measured: `studies/20260823-213712-babc/messages.jsonl` seq 14, an
`ls $FOAM_APPBIN | head -40`) and `tutorials/lagrangian/` carries at least
`kinematicParcelFoam`, `icoUncoupledKinematicParcelDyMFoam`,
`uncoupledKinematicParcelDyMFoam` and `MPPICDyMFoam`. What could not be measured from
here is the exact spelling of the uncoupled kinematic solver in v2512 or whether a
`kinematicCloudProperties` written against an older release still parses, and a case
generator naming a binary nobody has run would be a generator that fails on the
instance. One-way coupling for a dilute cloud is the same physics either way: the
parcels do not change the carrier field, so integrating them afterwards over the field
the Eulerian solver already wrote costs nothing and is checkable here.

The particle equation is drag plus buoyancy-corrected gravity and nothing else:

    dv/dt = (u(x) - v) / tau_p * phi(Re_p) + g (1 - rho_f/rho_p)

with tau_p = rho_p d^2 / (18 mu) the Stokes relaxation time. `--drag stokes` sets
phi = 1, which is exact as Re_p -> 0; `--drag schiller-naumann` sets
phi = 1 + 0.15 Re_p^0.687, the standard finite-Reynolds correction, with
Re_p = |u - v| d / nu. Left out on purpose and worth knowing about: added mass, the
Basset history term, lift, particle-particle collisions, and any feedback on the fluid.
For a heavy particle in a gas (rho_p/rho_f >> 1) those are small; for a bubble or a
neutrally buoyant particle they are not, and this script is the wrong instrument.

A massless tracer (`--stokes 0`) integrates dx/dt = u(x) instead, which is a streamline
of the frozen field, not a pathline of the unsteady one. On a time-dependent case they
are different curves; the field read here is one write time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Schiller & Naumann's finite-Reynolds drag correction, phi = 1 + 0.15 Re_p^0.687.
# Named here rather than inlined because it is the one modelling choice in the
# integrator and the grader contrasts it against phi = 1 deliberately: a result that
# is the same either way is not evidence that either branch was exercised.
SN_COEFF, SN_EXPONENT = 0.15, 0.687


# --------------------------------------------------------------------------- fields


def potential_cylinder(radius: float, speed: float):
    """Velocity of 2D potential flow past a circular cylinder, as a callable.

    psi = U y (1 - R^2/r^2), so u = U (1 - R^2/r^2 + 2 R^2 y^2 / r^4) and
    v = -2 U R^2 x y / r^4. On the upstream axis u = U (1 - R^2/x^2), which gives the
    stagnation-point strain rate 2U/R that the capture threshold turns on.

    Inside the cylinder the expression is meaningless; points there are the ones the
    capture test has already stopped, so it is left alone rather than masked.
    """

    def velocity(points: np.ndarray) -> np.ndarray:
        x, y = points[:, 0], points[:, 1]
        r2 = np.maximum(x * x + y * y, 1e-300)
        out = np.zeros_like(points)
        out[:, 0] = speed * (1.0 - radius**2 / r2 + 2.0 * radius**2 * y * y / r2**2)
        out[:, 1] = -2.0 * speed * radius**2 * x * y / r2**2
        return out

    return velocity


def case_field(case: Path, time: float | None, field: str = "U"):
    """Velocity of a finished OpenFOAM case, as a callable, plus its bounding box.

    Read through `render.open_case`, which is the path that already handles the
    `0/U` files pyvista's own reader chokes on (a `$variable` or an `#includeEtc`
    directive) by falling back to `foamToVTK`. Reusing it rather than opening the case
    again here means a case that renders is a case that can be tracked.

    Sampling is pyvista's `sample`, which is cell-wise linear interpolation and sets
    `vtkValidPointMask` to 0 outside the mesh. That mask is what tells a particle it
    has left the domain, so no geometry of the walls is needed: a particle that stops
    being inside the mesh has either hit something or gone out of an opening, and
    which one it was is decided from where it stopped.
    """
    import pyvista as pv

    import render

    block, chosen, _ = render.open_case(case, time)
    mesh = render.internal_mesh(block)
    if mesh is None:
        raise SystemExit(f"no internalMesh in {case}")
    if field not in mesh.point_data:
        if field in mesh.cell_data:
            mesh = mesh.cell_data_to_point_data()
        else:
            raise SystemExit(
                f"no field '{field}' in {case} at t = {chosen:g}; it has "
                f"{sorted(mesh.point_data.keys())}"
            )

    def velocity(points: np.ndarray) -> np.ndarray:
        probe = pv.PolyData(np.ascontiguousarray(points, dtype=float))
        sampled = probe.sample(mesh, pass_point_data=True)
        values = np.asarray(sampled[field], dtype=float)
        valid = np.asarray(sampled["vtkValidPointMask"], dtype=bool)
        values[~valid] = np.nan
        return values[:, :3] if values.ndim == 2 else values

    return velocity, np.asarray(mesh.bounds, dtype=float), chosen


# --------------------------------------------------------------------------- integrate


def relaxation_time(density: float, diameter: float, viscosity: float) -> float:
    """tau_p = rho_p d^2 / (18 mu), the Stokes relaxation time, with mu dynamic."""
    return density * diameter**2 / (18.0 * viscosity)


def drag_factor(slip: np.ndarray, diameter: float, nu: float, law: str) -> np.ndarray:
    if law == "stokes":
        return np.ones(len(slip))
    re_p = np.linalg.norm(slip, axis=1) * diameter / nu
    return 1.0 + SN_COEFF * np.power(np.maximum(re_p, 0.0), SN_EXPONENT)


def integrate(
    seeds: np.ndarray,
    velocity,
    tau: float,
    dt: float,
    steps: int,
    *,
    capture=None,
    exit_test=None,
    gravity: np.ndarray | None = None,
    drag: str = "stokes",
    diameter: float = 0.0,
    nu: float = 0.0,
    stop_speed: float = 0.0,
    record: int = 0,
):
    """March every seed with RK4 until it is captured, exits, or stops approaching.

    Returns `(fate, time, position, paths)`: one string and one time per seed, the
    final position, and (if `record` is non-zero) every `record`-th position as an
    (nsample, nseed, 3) array for drawing.

    Vectorised over seeds, with one shared step: the capture question this is graded
    on is a bifurcation, and a per-particle adaptive step makes the threshold a
    function of the step controller rather than of the physics. Measured on the
    potential-flow cylinder: the threshold is identical to four digits at dt = 0.01,
    0.005 and 0.002 R/U (`benchmarks/particles/README.md`), which is what says the
    remaining error is the stopping criterion and not the integrator.

    `stop_speed` is why this terminates at all. Below the critical Stokes number a
    particle on the stagnation streamline approaches the wall exponentially and never
    reaches it, so a distance test alone runs forever and eventually reports a capture
    that is a floating-point underflow. A particle whose approach speed toward the
    nearest capture surface has fallen below `stop_speed` is recorded as `stopped`:
    it is a finite statement about an infinite-time question, and it biases the
    measured threshold upward by a known amount that falls as `stop_speed` falls.
    """
    position = np.array(seeds, dtype=float).reshape(-1, 3)
    n = len(position)
    sampled = velocity(position)
    if tau <= 0.0:
        sampled = np.nan_to_num(sampled)
    particle = sampled.copy()
    body = np.zeros(3) if gravity is None else np.asarray(gravity, dtype=float)

    fate = np.array(["running"] * n, dtype=object)
    finished_at = np.full(n, np.nan)
    live = np.ones(n, dtype=bool)
    paths = [position.copy()] if record else []

    def rate(p: np.ndarray, v: np.ndarray):
        """(dx/dt, dv/dt, u) -- the carrier velocity comes back so the loop does not
        have to sample the field a second time for the exit test."""
        u = velocity(p)
        if tau <= 0.0:
            clean = np.nan_to_num(u)
            return clean, np.zeros_like(v), u
        slip = u - v
        phi = 1.0 if drag == "stokes" else drag_factor(slip, diameter, nu, drag)[:, None]
        return v, np.nan_to_num(slip * phi / tau + body), u

    for step in range(steps):
        if not live.any():
            break
        k1p, k1v, u1 = rate(position, particle)
        k2p, k2v, _ = rate(position + dt / 2 * k1p, particle + dt / 2 * k1v)
        k3p, k3v, _ = rate(position + dt / 2 * k2p, particle + dt / 2 * k2v)
        k4p, k4v, _ = rate(position + dt * k3p, particle + dt * k3v)
        moved = position + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        sped = particle + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        position = np.where(live[:, None], moved, position)
        # A massless tracer has no velocity of its own: its transport velocity is the
        # field at where it now is, and the capture test needs that rather than the
        # field at the seed, which is what a carried-forward state would keep.
        particle = (np.nan_to_num(velocity(position)) if tau <= 0.0
                    else np.where(live[:, None], sped, particle))
        now = (step + 1) * dt

        if capture is not None:
            distance, approach = capture(position, particle)
            hit = live & (distance <= 0.0)
            halted = live & ~hit & (approach < stop_speed)
        else:
            hit = np.zeros(n, dtype=bool)
            halted = np.zeros(n, dtype=bool)
        left = live & ~hit & ~halted & (
            exit_test(position) if exit_test is not None
            else ~np.isfinite(u1).all(axis=1)
        )
        for mask, name in ((hit, "captured"), (halted, "stopped"), (left, "exited")):
            if mask.any():
                fate[mask] = name
                finished_at[mask] = now
                live &= ~mask
        if record and (step + 1) % record == 0:
            paths.append(position.copy())

    finished_at[live] = steps * dt
    return fate, finished_at, position, (np.array(paths) if record else None)


def cylinder_capture(radius: float):
    """Signed distance to a cylinder of radius `radius` at the origin, and the
    approach speed toward it -- the pair `integrate` wants as its capture test."""

    def capture(position: np.ndarray, particle: np.ndarray):
        r = np.hypot(position[:, 0], position[:, 1])
        radial = -(position[:, 0] * particle[:, 0] + position[:, 1] * particle[:, 1])
        return r - radius, radial / np.maximum(r, 1e-300)

    return capture


def capture_threshold(
    radius: float = 1.0,
    speed: float = 1.0,
    *,
    low: float = 0.1,
    high: float = 0.2,
    dt: float = 0.005,
    horizon: float = 100.0,
    start: float = -5.0,
    stop_speed: float = 1e-10,
    tolerance: float = 1e-5,
) -> dict:
    """The smallest Stokes number tau_p U/R that captures a particle on the axis.

    Bisected, because the fate is monotone in the Stokes number on this flow (checked
    over 721 values between 0.118 and 0.136 in `benchmarks/particles/grade.py`: no
    holes). Released on the stagnation streamline at `start` radii upstream, moving
    with the local flow, which is the limit the published threshold is stated in; the
    measured value is independent of `start` over -3 to -20 R, as the inviscid
    threshold is a property of the stagnation point alone.
    """
    velocity = potential_cylinder(radius, speed)
    capture = cylinder_capture(radius)
    steps = int(round(horizon / dt))

    def captured(stokes: float) -> bool:
        tau = stokes * radius / speed
        seed = np.array([[start * radius, 0.0, 0.0]])
        fate, _, _, _ = integrate(
            seed, velocity, tau, dt, steps, capture=capture,
            exit_test=lambda p: p[:, 0] > 2 * radius, stop_speed=stop_speed * speed,
        )
        return bool(fate[0] == "captured")

    if captured(low):
        raise SystemExit(f"the bracket is wrong: St = {low:g} already captures")
    if not captured(high):
        raise SystemExit(f"the bracket is wrong: St = {high:g} does not capture")
    iterations = 0
    while high - low > tolerance:
        mid = 0.5 * (low + high)
        if captured(mid):
            high = mid
        else:
            low = mid
        iterations += 1
    return {
        "stokes_critical": 0.5 * (low + high),
        "bracket": [low, high],
        "iterations": iterations,
        "dt": dt,
        "horizon": horizon,
        "start_radii": start,
        "stop_speed": stop_speed,
    }


# --------------------------------------------------------------------------- figure


def draw(paths: np.ndarray, fate, out: Path, radius: float | None, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colour = {"captured": "#b2182b", "exited": "#2166ac", "stopped": "#777777",
              "running": "#777777"}
    for j in range(paths.shape[1]):
        ax.plot(paths[:, j, 0], paths[:, j, 1], lw=0.7,
                color=colour.get(fate[j], "#777777"))
    if radius:
        ax.add_patch(plt.Circle((0, 0), radius, color="#333333", zorder=5))
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, fontsize=10)
    handles = [plt.Line2D([], [], color=c, label=k) for k, c in colour.items()
               if k in set(fate)]
    if handles:
        ax.legend(handles=handles, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- main


def _cylinder(args) -> dict:
    if args.threshold:
        result = capture_threshold(
            args.radius, args.speed, dt=args.dt, horizon=args.horizon,
            start=args.start, stop_speed=args.stop_speed,
        )
        result["published_critical"] = 0.125
        result["error_pct"] = 100 * (result["stokes_critical"] - 0.125) / 0.125
        return result

    velocity = potential_cylinder(args.radius, args.speed)
    offsets = np.linspace(-args.span, args.span, args.seeds) * args.radius
    seeds = np.stack([np.full(args.seeds, args.start * args.radius), offsets,
                      np.zeros(args.seeds)], axis=1)
    steps = int(round(args.horizon / args.dt))
    fate, when, where, paths = integrate(
        seeds, velocity, args.stokes * args.radius / args.speed, args.dt, steps,
        capture=cylinder_capture(args.radius),
        exit_test=lambda p: p[:, 0] > 2 * args.radius,
        stop_speed=args.stop_speed * args.speed,
        record=max(1, steps // 400),
    )
    captured = [i for i, f in enumerate(fate) if f == "captured"]
    width = (2 * args.span * args.radius)
    if args.out and paths is not None:
        draw(paths, fate, Path(args.out), args.radius,
             f"potential flow past a cylinder, St = {args.stokes:g}")
    return {
        "flow": "potential_cylinder", "radius": args.radius, "speed": args.speed,
        "stokes": args.stokes, "seeds": int(args.seeds),
        "captured": len(captured),
        # Single-cylinder collection efficiency: the swept width of the seeds that hit,
        # over the cylinder's own width. Read off the seed spacing, so it is resolved
        # to 1/(seeds - 1) of the seeded band and no better -- `seed_resolution_over_D`
        # is that bound, printed beside it so the number is not read as tighter.
        "efficiency": (
            float(abs(offsets[captured[-1]] - offsets[captured[0]]) / (2 * args.radius))
            if captured else 0.0
        ),
        "seed_resolution_over_D": float(width / (args.seeds - 1) / (2 * args.radius)),
        "fates": {name: int(sum(f == name for f in fate))
                  for name in ("captured", "exited", "stopped", "running")},
    }


def _case(args) -> dict:
    velocity, bounds, when = case_field(Path(args.case), args.time, args.field)
    lo, hi = bounds[0::2], bounds[1::2]
    rng = np.random.default_rng(args.seed)
    box = args.seed_box
    if box:
        lo_seed, hi_seed = np.array(box[0::2], dtype=float), np.array(box[1::2], dtype=float)
    else:
        lo_seed, hi_seed = lo.copy(), hi.copy()
        lo_seed[0] = hi_seed[0] = lo[0] + 0.02 * (hi[0] - lo[0])
    seeds = lo_seed + rng.random((args.seeds, 3)) * (hi_seed - lo_seed)

    tau = (relaxation_time(args.density, args.diameter, args.density_fluid * args.nu)
           if args.diameter > 0 else 0.0)
    gravity = np.array(args.gravity, dtype=float) * (
        1.0 - args.density_fluid / args.density if args.density > 0 else 0.0)
    steps = int(round(args.horizon / args.dt))
    fate, when_done, where, paths = integrate(
        seeds, velocity, tau, args.dt, steps,
        gravity=gravity, drag=args.drag, diameter=args.diameter, nu=args.nu,
        record=max(1, steps // 400) if args.out else 0,
    )
    if args.out and paths is not None:
        draw(paths, fate, Path(args.out), None,
             f"{Path(args.case).name} at t = {when:g}, tau_p = {tau:.3e} s")
    residence = [t for t, f in zip(when_done, fate) if f == "exited"]
    return {
        "case": str(args.case), "time": when, "field": args.field,
        "relaxation_time_s": tau, "drag": args.drag, "seeds": int(args.seeds),
        "fates": {name: int(sum(f == name for f in fate))
                  for name in ("captured", "exited", "stopped", "running")},
        "mean_residence_s": float(np.mean(residence)) if residence else None,
        "median_residence_s": float(np.median(residence)) if residence else None,
        "bounds": [float(v) for v in bounds],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="flow", required=True)

    cy = sub.add_parser("cylinder", help="closed-form potential flow past a cylinder")
    cy.add_argument("--radius", type=float, default=1.0)
    cy.add_argument("--speed", type=float, default=1.0)
    cy.add_argument("--stokes", type=float, default=0.2, help="tau_p U / R")
    cy.add_argument("--seeds", type=int, default=41)
    cy.add_argument("--span", type=float, default=1.0, help="seed band half-width in radii")
    cy.add_argument("--threshold", action="store_true",
                    help="bisect the critical Stokes number instead of tracking a band")

    cs = sub.add_parser("case", help="a finished OpenFOAM case")
    cs.add_argument("case")
    cs.add_argument("--time", type=float, default=None)
    cs.add_argument("--field", default="U")
    cs.add_argument("--diameter", type=float, default=0.0, help="particle diameter [m]; 0 is a tracer")
    cs.add_argument("--density", type=float, default=1000.0, help="particle density [kg/m3]")
    cs.add_argument("--density-fluid", type=float, default=1.0, help="fluid density [kg/m3]")
    cs.add_argument("--nu", type=float, default=1.5e-5, help="fluid kinematic viscosity [m2/s]")
    cs.add_argument("--gravity", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    cs.add_argument("--seeds", type=int, default=200)
    cs.add_argument("--seed-box", type=float, nargs=6, default=None,
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    cs.add_argument("--seed", type=int, default=0, help="seed of the seed positions")

    for p in (cy, cs):
        p.add_argument("--dt", type=float, default=0.005)
        p.add_argument("--horizon", type=float, default=100.0)
        p.add_argument("--stop-speed", type=float, default=1e-10)
        p.add_argument("--drag", choices=("stokes", "schiller-naumann"), default="stokes")
        p.add_argument("--out", help="write a PNG of the paths here")
        p.add_argument("--json", action="store_true")
    cy.add_argument("--start", type=float, default=-5.0, help="release plane, in radii")

    args = parser.parse_args(argv)
    result = _cylinder(args) if args.flow == "cylinder" else _case(args)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for key, value in result.items():
            print(f"{key:<26} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
