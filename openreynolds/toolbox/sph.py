#!/usr/bin/env python3
"""Smoothed-particle hydrodynamics in numpy: 2D, weakly compressible, nothing compiled.

    python3 sph.py hydrostatic --dx 0.005
    python3 sph.py sloshing --out sloshing.png
    python3 sph.py dambreak --dx 0.0025 --out dambreak.png --json

Why this is a toolbox script and not a layer in the image: the same reason as `lbm.py`
beside it. The image is already 1.2 GB compressed with only about a third of that
OpenFOAM, both backends run the *image's* `python3` so a pip install on the host never
reaches the solve, and `cli.py` copies this directory into `/work/.toolbox` at session
start -- so a toolbox script reaches every workspace for zero megabytes and arrives with
a `pip install -U openreynolds`. DualSPHysics and SPlisHSPlasH are compiled C++ with
XML or CMake front ends; PySPH installs from pip but only as a source distribution that
compiles some forty Cython extensions and then generates and compiles more at run time.

What it is for. A free surface that breaks, folds or separates: a column collapsing, a
tank sloshing, a wave hitting something. interFoam solves the same physics and is the
better tool once the mesh exists, and `templates/free_surface.py` writes that case. What
it does not need is the mesh: there are no cells here, so there is nothing to collapse
in a violent deformation and nothing to resolve an interface onto. What it is not for:
anything thin, anything steady, anything where the boundary layer is the answer. The
boundary treatment below is the cheap one, and it has a measured wall error.

The scheme, stated so the choices can be argued with

    Wendland C2 kernel, support 2h, h = 1.3*dx. Density by the continuity equation
    rather than by summation, because a summation density reads low at a free surface --
    there is no fluid above it to sum -- which is exactly where these cases live.

    Tait's equation of state, p = (c0^2 rho0 / 7) ((rho/rho0)^7 - 1), with c0 an
    artificial sound speed about ten times the fastest flow. That is what "weakly
    compressible" means and it is the scheme's cost: the time step is acoustic, dt ~
    0.2 h / c0, so a second of a dam break is tens of thousands of steps no matter how
    few particles there are.

    Monaghan's artificial viscosity, which damps approach and does nothing to separation.
    `--alpha` is it, and it is a numerical parameter and not a physical viscosity: these
    cases are gravity-driven and inertia-dominated, and nothing here claims a Reynolds
    number.

    Dynamic boundary particles: the walls are particles that never move, whose density
    still evolves and whose pressure comes from the same equation of state, so an
    approaching fluid particle raises their pressure and is pushed back. It is three
    lines rather than thirty and its error is a sub-particle gap at the wall, which
    `sloshing` below measures rather than assumes.

    Symplectic (velocity-then-position) Euler. One line each way, and it does not pump
    energy into an oscillation the way the other Euler does, which matters because one of
    the graded cases *is* an oscillation.

The neighbour search is the part that decides whether this runs at all. Particles are
bucketed into a uniform grid of cells one support radius across, and the buckets of the
nine cells around each particle are gathered into one dense `(particles, candidates)`
index table. Every force is then a masked sum along one axis -- no Python loop over
pairs, no variable-length lists. The table costs more memory than a compacted pair list
and is the reason this is numpy rather than C.

What it costs. Every run prints its own particle count, step count, wall-clock time and
steps per second, which is the only honest answer to "will this finish on my laptop":
it depends on the machine and on what else is running there, and a table in a docstring
cannot know either. `benchmarks/sph/README.md` carries what the graded runs measured on a
GitHub runner -- a slow, consistent two-core machine, so a floor rather than a flattering
number.

The knob is `--dx`, and it is expensive in both directions at once: halving it doubles
the particles per metre in each of two directions and halves the time step, so the cost
of a fixed physical duration goes as 1/dx^3. The step count itself is acoustic and not
physical -- dt = 0.2*h/c0 with c0 ten to twenty times the fastest flow -- so a tenth of a
second of a dam break is thousands of steps however few particles there are, and that
floor is the scheme's and not this implementation's.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

GAMMA = 7.0
"""Tait's exponent for water. 7 is the standard value and the one c0 is calibrated
against; it is not a free parameter here."""

G = 9.81
"""Standard gravity to three figures (9.80665), pointing along -y, the same figure and
the same reason as `templates/free_surface.py`."""

RHO0 = 1000.0
"""Reference density. Plain 1000 rather than water's 998.2 at 20 C: every quantity these
cases are graded on is either a ratio or a length, so the third figure cancels, and the
pressure check below is against rho0*g*h with this same rho0."""


# -- the kernel ---------------------------------------------------------------------


def kernel(r: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """(W, dW/dr) for the Wendland C2 kernel in 2D, zero outside the 2h support.

        W(q) = 7/(4 pi h^2) (1 - q/2)^4 (2q + 1),  q = r/h,  0 <= q <= 2
        dW/dq = 7/(4 pi h^2) (1 - q/2)^3 (-5q)

    Wendland rather than the cubic spline because the cubic spline's second derivative
    is discontinuous at q = 1, which shows up as pairing instability in exactly the
    free-surface cases this is for.
    """
    q = r / h
    inside = q < 2.0
    t = np.where(inside, 1.0 - 0.5 * q, 0.0)
    t3 = t * t * t
    alpha = 7.0 / (4.0 * math.pi * h * h)
    w = np.where(inside, alpha * t3 * t * (2.0 * q + 1.0), 0.0)
    dwdr = np.where(inside, alpha * t3 * (-5.0 * q) / h, 0.0)
    return w, dwdr


# -- the neighbour table ------------------------------------------------------------


def candidates(pos: np.ndarray, radius: float) -> np.ndarray:
    """A dense (particles, 9*per_cell) table of possibly-neighbouring particle indices.

    Cells are one support radius across, so a particle's neighbours are all in its own
    cell or the eight around it. The grid is padded by a cell on every side, which is
    what lets the nine offsets be added to a flat cell index without a bounds check.
    Empty slots hold -1; the callers mask on the distance anyway, and -1 indexes the
    last particle rather than raising, which is harmless once masked.
    """
    if not np.isfinite(pos).all():
        raise ValueError("particle positions are not finite: the run has diverged, and "
                         "a smaller time step (a smaller --dx) is the usual cause")
    low = pos.min(axis=0) - 2.0 * radius
    cell = ((pos - low) / radius).astype(np.int64)
    shape = cell.max(axis=0) + 3
    # In Python ints, not numpy's: the whole point of the guard below is to catch a cell
    # count of 1e20, and an int64 product of two 1e10 axes wraps round silently and can
    # come out small enough to pass. `flat` is computed after the guard for the same
    # reason -- the multiply in it overflows on the same input.
    cells = int(shape[0]) * int(shape[1])
    # A divergence that stays finite is the dangerous one. The grid is sized from the
    # bounding box, so particles thrown to 1e8 ask for an array of 1e20 cells, and numpy
    # does not refuse that quickly -- it takes the machine into swap and the run looks
    # like a hang rather than an instability. A sane 2D packing is a few particles per
    # cell, so a grid two orders of magnitude emptier than that is not geometry.
    if cells > 256 * pos.shape[0] + 4096:
        span = pos.max(axis=0) - pos.min(axis=0)
        raise ValueError(
            f"{pos.shape[0]:,} particles span {span[0]:.3g} x {span[1]:.3g} at a "
            f"{radius:.3g} cell size, which is {cells:,} cells: the run has diverged")
    flat = cell[:, 0] + cell[:, 1] * shape[0]

    order = np.argsort(flat, kind="stable")
    ordered = flat[order]
    within = np.arange(order.size) - np.searchsorted(ordered, ordered, side="left")
    per_cell = int(np.bincount(flat, minlength=cells).max())
    bucket = np.full((cells, per_cell), -1, dtype=np.int64)
    bucket[ordered, within] = order

    offsets = np.array([-shape[0] - 1, -shape[0], -shape[0] + 1, -1, 0, 1,
                        shape[0] - 1, shape[0], shape[0] + 1])
    return bucket[flat[:, None] + offsets[None, :]].reshape(order.size, -1)


# -- one step -----------------------------------------------------------------------


def pressure(rho: np.ndarray, c0: float) -> np.ndarray:
    """Tait. The bulk modulus is c0^2*rho0/gamma, so c0 sets both the stiffness and,
    through the time step, the whole cost of the run."""
    return (c0 * c0 * RHO0 / GAMMA) * ((rho / RHO0) ** GAMMA - 1.0)


def accelerations(pos: np.ndarray, vel: np.ndarray, rho: np.ndarray, mass: np.ndarray,
                  table: np.ndarray, h: float, c0: float, alpha: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """(dv/dt, drho/dt) for every particle, as masked sums over the neighbour table."""
    p = pressure(rho, c0)
    dx = pos[:, None, :] - pos[table]
    r = np.sqrt((dx * dx).sum(axis=2))
    live = (table >= 0) & (r < 2.0 * h) & (r > 0.0)
    r = np.where(live, r, 1.0)  # keep the divisions finite; `live` masks the result

    _w, dwdr = kernel(r, h)
    grad = (dwdr / r)[:, :, None] * dx  # grad_i W_ij, pointing from i towards j
    grad = np.where(live[:, :, None], grad, 0.0)

    dv = vel[:, None, :] - vel[table]
    rho_j = rho[table]
    mass_j = mass[table]

    # Monaghan's symmetric pressure term: equal and opposite between a pair, so momentum
    # is conserved to round-off whatever the density field does.
    coefficient = (p[:, None] + p[table]) / (rho[:, None] * rho_j)

    # Artificial viscosity, on approach only. dv.dx < 0 is approach, and there the term
    # is a repulsion; separation gets nothing, which is what stops it damping the
    # free-surface motion the cases are about.
    vdotr = (dv * dx).sum(axis=2)
    rho_bar = 0.5 * (rho[:, None] + rho_j)
    viscous = np.where(vdotr < 0.0,
                       -alpha * c0 * h * vdotr / (rho_bar * (r * r + 0.01 * h * h)),
                       0.0)

    weight = (mass_j * (coefficient + viscous))[:, :, None]
    acceleration = -(weight * grad).sum(axis=1)
    acceleration[:, 1] -= G

    drho = (mass_j * (dv * grad).sum(axis=2)).sum(axis=1)
    return acceleration, drho


def run(pos: np.ndarray, moving: np.ndarray, dx: float, *, end: float,
        c0: float, alpha: float = 0.02, cfl: float = 0.2, h_factor: float = 1.3,
        samples: int = 400, progress: bool = True,
        initial_velocity: np.ndarray | None = None) -> dict[str, Any]:
    """Integrate to `end` seconds and report the state plus a time history.

    `moving` is the particle mask that is fluid: everything else is a wall particle,
    which keeps its position and its velocity of zero and takes part in everything else.
    A wall particle's density is held at or above rho0, which is what stops a wall from
    ever pulling fluid into it -- the one place dynamic boundary particles need a guard.

    The history is `samples` evenly spaced records of the things the cases are graded
    on, which is cheaper and far less noisy than writing every particle every step.
    """
    pos = np.array(pos, dtype=float)
    moving = np.asarray(moving, dtype=bool)
    h = h_factor * dx
    mass = np.full(pos.shape[0], RHO0 * dx * dx)
    rho = np.full(pos.shape[0], RHO0)
    vel = np.zeros_like(pos)
    if initial_velocity is not None:
        vel[:] = initial_velocity
        vel[~moving] = 0.0  # a wall particle never moves, whatever was asked for
    dt = cfl * h / c0
    steps = int(math.ceil(end / dt))
    every = max(1, steps // samples)
    fluid_mass = float(mass[moving].sum())

    history: list[dict[str, float]] = []
    started = time.monotonic()
    for step in range(steps + 1):
        table = candidates(pos, 2.0 * h)
        if step % every == 0:
            fluid = pos[moving]
            history.append({
                "t": step * dt,
                "front": float(fluid[:, 0].max()),
                "surface": float(fluid[:, 1].max()),
                "centroid_x": float((mass[moving] * fluid[:, 0]).sum() / fluid_mass),
                "kinetic": float(0.5 * (mass[moving] * (vel[moving] ** 2).sum(axis=1)).sum()),
                "rho_max": float(rho[moving].max()),
            })
            if progress and step % (every * 40) == 0:
                print(f"  t = {step * dt:7.4f} s  step {step:>7,}/{steps:,}",
                      file=sys.stderr)
        if step == steps:
            break

        acceleration, drho = accelerations(pos, vel, rho, mass, table, h, c0, alpha)
        rho += dt * drho
        np.maximum(rho, RHO0, out=rho, where=~moving)
        vel[moving] += dt * acceleration[moving]
        pos[moving] += dt * vel[moving]

    elapsed = time.monotonic() - started
    return {
        "pos": pos, "vel": vel, "rho": rho, "moving": moving,
        "pressure": pressure(rho, c0), "mass": mass,
        "dx": dx, "h": h, "dt": dt, "c0": c0, "alpha": alpha,
        "particles": int(pos.shape[0]), "fluid_particles": int(moving.sum()),
        "steps": steps, "end": end, "history": history, "seconds": elapsed,
        "steps_per_second": steps / elapsed if elapsed > 0 else 0.0,
    }


# -- geometry -----------------------------------------------------------------------


def block(x0: float, y0: float, width: float, height: float, dx: float) -> np.ndarray:
    """A staggered-free rectangular lattice of particles, first one half a spacing in.

    Half a spacing is not cosmetic: it makes the block's edge the surface the fluid
    occupies rather than a row of particles sitting on it, so a column of water of
    height H has its free surface at H and not at H - dx/2.
    """
    xs = np.arange(x0 + 0.5 * dx, x0 + width - 1e-9, dx)
    ys = np.arange(y0 + 0.5 * dx, y0 + height - 1e-9, dx)
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)
    return grid.reshape(-1, 2)


def tank(length: float, height: float, dx: float, layers: int = 3) -> np.ndarray:
    """Floor and two side walls, `layers` particles thick, outside the fluid domain.

    Three layers because the kernel support is 2h = 2.6*dx: fewer than three and the
    wall is transparent to a fast particle, which shows up as fluid leaking through the
    floor in the dam break and nowhere else.
    """
    floor = block(-layers * dx, -layers * dx, length + 2 * layers * dx, layers * dx, dx)
    left = block(-layers * dx, 0.0, layers * dx, height, dx)
    right = block(length, 0.0, layers * dx, height, dx)
    return np.concatenate([floor, left, right])


def _assemble(fluid: np.ndarray, walls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.concatenate([fluid, walls])
    moving = np.zeros(pos.shape[0], dtype=bool)
    moving[: fluid.shape[0]] = True
    return pos, moving


# -- the cases ----------------------------------------------------------------------


def hydrostatic(depth: float = 0.1, dx: float | None = None, end: float = 1.0,
                **kwargs: Any) -> dict[str, Any]:
    """Still water in a tank. The answer is p = rho*g*(depth - y), exact.

    Nothing moves in the exact solution, so every number this produces is error. It is
    the sharpest cheap check of the equation of state, the density evolution and the
    wall treatment at once, and the one the brief's dam break cannot be: a dam break's
    published reference is an experiment and this one is arithmetic.
    """
    dx = depth / 20.0 if dx is None else dx
    length = 2.0 * depth
    pos, moving = _assemble(block(0.0, 0.0, length, depth, dx),
                            tank(length, 1.5 * depth, dx))
    result = run(pos, moving, dx, end=end, c0=20.0 * math.sqrt(G * depth), **kwargs)
    result["case"] = "hydrostatic"
    result["depth"] = depth
    result["length"] = length
    # The middle half of the tank only: the ends carry the wall's own error, which is a
    # separate quantity and is what `sloshing` measures.
    fluid = result["moving"]
    middle = fluid & (result["pos"][:, 0] > 0.25 * length) & (result["pos"][:, 0] < 0.75 * length)
    y = result["pos"][middle, 1]
    result["y_measured"] = y
    result["p_measured"] = result["pressure"][middle]
    result["p_exact"] = RHO0 * G * np.maximum(depth - y, 0.0)
    result["p_scale"] = RHO0 * G * depth
    result["sampled"] = int(middle.sum())
    return result


def sloshing(length: float = 1.0, depth: float = 0.5, amplitude: float = 0.01,
             dx: float | None = None, periods: float = 3.0, **kwargs: Any) -> dict[str, Any]:
    """A first-mode standing wave in a closed tank, released from its velocity field.

    The answer is the linear dispersion relation for a standing wave in water of finite
    depth, which is closed form and in every textbook (Lamb, *Hydrodynamics*, 6th ed.,
    art. 228): for a tank of length L the first mode has k = pi/L and

        omega^2 = g k tanh(k h)

    The initial condition is the mode's own velocity field at the instant the surface is
    flat, so the particle spacing starts uniform -- displacing the surface instead would
    start the run with a disturbed lattice and measure that as well as the period.

    What is measured is the horizontal centroid of the fluid, not a surface probe. The
    centroid oscillates at the mode frequency, every particle contributes to it, and it
    does not care which particle happens to be highest at the moment of sampling.
    """
    dx = depth / 20.0 if dx is None else dx
    k = math.pi / length
    omega = math.sqrt(G * k * math.tanh(k * depth))
    fluid = block(0.0, 0.0, length, depth, dx)
    pos, moving = _assemble(fluid, tank(length, 1.6 * depth, dx))
    result_period = 2.0 * math.pi / omega

    # phi = (a*omega/k) cos(kx) cosh(k(z+h))/sinh(kh) cos(omega t), z from the surface.
    z = fluid[:, 1] - depth
    scale = amplitude * omega / math.sinh(k * depth)
    vx = -scale * np.sin(k * fluid[:, 0]) * np.cosh(k * (z + depth))
    vy = scale * np.cos(k * fluid[:, 0]) * np.sinh(k * (z + depth))

    pos = np.array(pos)
    initial = np.zeros_like(pos)
    initial[: fluid.shape[0], 0] = vx
    initial[: fluid.shape[0], 1] = vy
    result = run(pos, moving, dx, end=periods * result_period,
                 c0=10.0 * math.sqrt(G * depth), initial_velocity=initial, **kwargs)
    result["case"] = "sloshing"
    result["length"] = length
    result["depth"] = depth
    result["amplitude"] = amplitude
    result["k"] = k
    result["omega_exact"] = omega
    result["period_exact"] = result_period
    result["period_measured"], result["crossings"] = _period(result["history"])
    return result


def _period(history: list[dict[str, float]]) -> tuple[float, int]:
    """The oscillation period from the zero crossings of the centroid's displacement.

    Two crossings make half a period, so the mean spacing of all of them doubled is the
    period, and using every crossing rather than the first and last averages the sample
    jitter instead of putting all of it in one interval. Fewer than three crossings is
    not a period and says so by returning nan.
    """
    t = np.array([row["t"] for row in history])
    signal = np.array([row["centroid_x"] for row in history])
    signal = signal - signal.mean()
    sign_change = np.flatnonzero(np.diff(np.sign(signal)) != 0)
    if sign_change.size < 3:
        return float("nan"), int(sign_change.size)
    # Linear interpolation to each crossing, which is worth having: the sample interval
    # is a few percent of the period and a nearest-sample crossing would quantise the
    # answer at that size.
    crossings = []
    for index in sign_change:
        a, b = signal[index], signal[index + 1]
        crossings.append(t[index] + (t[index + 1] - t[index]) * a / (a - b))
    return float(2.0 * np.mean(np.diff(crossings))), int(sign_change.size)


def dambreak(column: float = 0.1, aspect: float = 2.0, dx: float | None = None,
             tank_length: float | None = None, end: float | None = None,
             **kwargs: Any) -> dict[str, Any]:
    """A water column of width `column` and height `aspect*column`, released on a floor.

    Martin & Moyce's aspect ratio n = 2 by default, which is what their n = 2 series and
    `templates/free_surface.py` both use. The quantity is the surge front position in
    their dimensionless form, Z = x/a against T = t*sqrt(2g/a); see
    `benchmarks/sph/README.md` for why it is reported here and not graded.
    """
    dx = column / 20.0 if dx is None else dx
    height = aspect * column
    tank_length = 5.0 * column if tank_length is None else tank_length
    # sqrt(2g/a) is Martin & Moyce's time scale; 3 of those is past the point the front
    # reaches the far wall at this tank length, so the default stops before the
    # reflection arrives and contaminates the front.
    end = 3.0 / math.sqrt(2.0 * G / column) if end is None else end
    pos, moving = _assemble(block(0.0, 0.0, column, height, dx),
                            tank(tank_length, 1.2 * height, dx))
    result = run(pos, moving, dx, end=end, c0=20.0 * math.sqrt(G * height), **kwargs)
    result["case"] = "dambreak"
    result["column"] = column
    result["aspect"] = aspect
    result["tank_length"] = tank_length
    timescale = math.sqrt(2.0 * G / column)
    result["front"] = [{"T": row["t"] * timescale, "Z": row["front"] / column,
                        "t": row["t"], "x": row["front"]}
                       for row in result["history"]]
    return result


CASES = {"hydrostatic": hydrostatic, "sloshing": sloshing, "dambreak": dambreak}


# -- output -------------------------------------------------------------------------


def figure(result: dict[str, Any], path: Path) -> None:
    """The particles, and the quantity the case is graded on, in one png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    pos, moving = result["pos"], result["moving"]
    left.scatter(pos[~moving, 0], pos[~moving, 1], s=2, c="0.65", label="wall")
    dots = left.scatter(pos[moving, 0], pos[moving, 1], s=3,
                        c=result["pressure"][moving] / 1000.0, cmap="viridis")
    fig.colorbar(dots, ax=left, label="p (kPa)")
    left.set_aspect("equal")
    left.set_title(f"{result['case']}: t = {result['end']:.3f} s, "
                   f"{result['fluid_particles']} fluid particles")

    if result["case"] == "hydrostatic":
        depth = result["depth"]
        y = np.linspace(0.0, depth, 50)
        right.plot(RHO0 * G * (depth - y) / 1000.0, y, label="rho g (h - y), exact")
        right.scatter(result["p_measured"] / 1000.0, result["y_measured"], s=4,
                      label="particles, middle half")
        right.set_xlabel("p (kPa)")
        right.set_ylabel("y (m)")
    elif result["case"] == "sloshing":
        t = [row["t"] for row in result["history"]]
        signal = np.array([row["centroid_x"] for row in result["history"]])
        right.plot(t, (signal - signal.mean()) * 1000.0)
        right.set_xlabel("t (s)")
        right.set_ylabel("fluid centroid, x - mean (mm)")
        right.set_title(f"period {result['period_measured']:.4f} s against "
                        f"{result['period_exact']:.4f} s exact")
    else:
        right.plot([row["T"] for row in result["front"]],
                   [row["Z"] for row in result["front"]], label="this run")
        right.set_xlabel("T = t sqrt(2g/a)")
        right.set_ylabel("Z = x/a")
        right.set_title("surge front")
    right.grid(alpha=0.3)
    right.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def report(result: dict[str, Any]) -> str:
    lines = [f"# {result['case']}",
             f"  particles:  {result['fluid_particles']:,} fluid, "
             f"{result['particles'] - result['fluid_particles']:,} wall  "
             f"(dx = {result['dx']:g} m, h = {result['h']:g} m)",
             f"  stepping:   {result['steps']:,} steps of {result['dt']:.3e} s to "
             f"t = {result['end']:.4f} s  (c0 = {result['c0']:.2f} m/s, "
             f"alpha = {result['alpha']:g})",
             f"  cost:       {result['seconds']:.1f} s at "
             f"{result['steps_per_second']:.0f} steps/s",
             f"  density:    max rho/rho0 = "
             f"{max(row['rho_max'] for row in result['history']) / RHO0:.5f} "
             f"over the run"]
    if result["case"] == "hydrostatic":
        error = np.abs(result["p_measured"] - result["p_exact"])
        lines += [f"  pressure:   {result['sampled']} particles in the middle half",
                  f"              RMS error {np.sqrt((error**2).mean()) / result['p_scale'] * 100:.3f}% "
                  f"of rho g h, max {error.max() / result['p_scale'] * 100:.3f}%"]
    elif result["case"] == "sloshing":
        measured, exact = result["period_measured"], result["period_exact"]
        lines += [f"  period:     {measured:.5f} s measured over "
                  f"{result['crossings']} crossings, {exact:.5f} s exact "
                  f"({(measured / exact - 1) * 100:+.3f}%)"]
    else:
        reached = result["front"][-1]
        lines += [f"  front:      Z = {reached['Z']:.3f} at T = {reached['T']:.3f} "
                  f"(x = {reached['x']:.4f} m at t = {reached['t']:.4f} s)",
                  "              reported, not graded: see benchmarks/sph/README.md"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--dx", type=float, default=None, help="particle spacing in metres")
    parser.add_argument("--alpha", type=float, default=0.02,
                        help="Monaghan's artificial viscosity coefficient")
    parser.add_argument("--end", type=float, default=None, help="stop at this many seconds")
    parser.add_argument("--out", type=Path, default=None, help="write a png here")
    parser.add_argument("--npz", type=Path, default=None, help="write the particles here")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    options: dict[str, Any] = {"alpha": args.alpha, "progress": not args.quiet}
    if args.dx is not None:
        options["dx"] = args.dx
    if args.end is not None:
        options["end"] = args.end
    result = CASES[args.case](**options)

    if args.out:
        figure(result, args.out)
    if args.npz:
        np.savez_compressed(args.npz, **{key: value for key, value in result.items()
                                         if isinstance(value, np.ndarray)})
    if args.json:
        print(json.dumps({key: (value.tolist() if isinstance(value, np.ndarray) else value)
                          for key, value in result.items()}, indent=2))
    else:
        print(report(result))
        if args.out:
            print(f"  figure:     {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
