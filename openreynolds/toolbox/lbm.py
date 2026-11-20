#!/usr/bin/env python3
"""A lattice Boltzmann solver in numpy: D2Q9, two relaxation times, nothing compiled.

    python3 lbm.py poiseuille --n 32
    python3 lbm.py cavity --n 128 --re 1000 --out cavity.png
    python3 lbm.py cavity --n 64 --re 1000 --json

Why this is a toolbox script and not a layer in the image. The workspace image is
already 1.2 GB compressed and only about a third of that is OpenFOAM; every solver
added to it is paid for by every person who pulls it, on every machine, whether or not
they ever run that solver. OpenLB, Palabos and waLBerla are all compiled C++ and all of
them would want a per-case recompile, which in a product where the agent writes Python
into a kernel is the wrong shape as well as the wrong size. lbmpy and XLB are pip
installable and would still not reach anybody: the Docker and hosted backends both run
the *image's* `python3`, so a package installed on the host is invisible to the solve.
The toolbox is the one thing that does reach every workspace without an image --
`cli.py`'s `TOOLBOX_SOURCE` is copied to `/work/.toolbox` at session start -- and numpy
is already in the image, so this file costs zero megabytes and arrives with a
`pip install -U openreynolds`.

What it is for. LBM's useful property here is not speed, it is that the geometry is a
boolean array: a solid mask streams and bounces with no mesh, no snappyHexMesh, no
checkMesh and no cell quality to argue about. For a flow in an awkward passage at a
Reynolds number in the hundreds, that removes the step this corpus loses the most time
in. What it is not for: it is explicit, weakly compressible and second order, the whole
lattice lives in memory, and there is no turbulence model in here at all. A transonic
aerofoil or a 20-million-cell hull is OpenFOAM's and HiSA's work, and `advise.py` says
so beside this.

The numerics, and why each choice is the one that gets measured

    Two relaxation times (TRT) rather than one (BGK), because the two validations in
    `benchmarks/lbm/` both turn on it. Collision is split into the parts of the
    distribution even and odd under c -> -c and each relaxes at its own rate: tau_plus
    sets the viscosity, tau_minus is free. Fixing the product

        LAMBDA = (tau_plus - 1/2) * (tau_minus - 1/2) = 3/16

    puts the no-slip wall of a halfway bounce-back exactly on the lattice link midpoint
    for a straight wall, at any viscosity. BGK is the special case tau_minus = tau_plus,
    which satisfies that at exactly one viscosity (tau = 1/2 + sqrt(3)/4 = 0.9330) and
    nowhere else, so a BGK channel has a wall that moves with the Reynolds number and a
    parabola that is never quite the right one. The result is Ginzburg & d'Humieres,
    Phys. Rev. E 68, 066614 (2003), and `--relaxation bgk` is here so the difference can
    be measured rather than believed: at Re 1000 on a 128 cavity BGK runs at tau 0.538,
    which is also close enough to the 1/2 where BGK stops being stable to matter.

    Guo forcing for the body force (Guo, Zheng & Shi, Phys. Rev. E 65, 046308 (2002)),
    with the half-force already in the velocity the equilibrium is built from, and its
    odd and even parts given the odd and even relaxation rates -- which is what makes it
    the TRT force rather than the BGK one bolted onto a TRT collision.

    Bounce-back held on the solid nodes themselves, which is the two-line form of
    halfway bounce-back: a population that streams into a solid node is reflected in
    place and the next streaming step carries it back out. A wall that moves adds
    6*w_i*(c_i.u_wall) to the reflected population, which is the Ladd term.

What it costs. Every run prints its own step count, wall-clock time and rate in million
lattice updates per second, because that is the only honest way to answer "will this
finish on my laptop" -- the answer depends on the machine and on what else is running on
it, and a table in a docstring cannot know either. `benchmarks/lbm/README.md` carries the
rates the two graded runs measured on a GitHub runner, which is a slow and consistent
two-core machine and so a reasonable floor rather than a flattering number.

The cost is dominated by the step count, which scales with the lattice: the residual has
to fall across the whole cavity, and the number of steps to do that grows roughly
linearly in cells across while the cost of each step grows as the square. Halving `--n`
is therefore about eight times cheaper and is the knob to reach for first; `--re` is
nearly free in itself, and is limited instead by stability, since nu = u_lid*n/Re drives
tau_plus towards 1/2 as Re rises.

The lattice is in lattice units throughout -- dx = dt = 1, c_s^2 = 1/3 -- and the only
dimensionless group that reaches the answer is the Reynolds number. `u_lid` and the body
force are kept small because the compressibility error of this scheme is O(Ma^2);
0.1 is the usual compromise and is what both cases default to.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# D2Q9: 0 is the rest population, 1-4 the axes, 5-8 the diagonals.
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=float)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=float)
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
"""`OPP[i]` is the direction opposite `i`, which is what bounce-back reflects into."""

PAIRS = ((1, 3), (2, 4), (5, 7), (6, 8))
"""The four opposite pairs, which is the shape the collision is actually written in.

Not a presentation choice. Writing the collision over whole (9, ny, nx) arrays is
shorter and was the first version here, and it is memory-bound: each temporary is 1.2 MB
at n = 128 and a dozen are live at once, which is past any laptop's L2 and most of its
L3, so the step cost grows faster than the lattice does. Per pair the working set is one
(ny, nx) plane -- 135 kB at the same n -- the equilibrium's even half is computed once
for both members of the pair instead of twice, and `f[OPP]`, a whole-lattice gather
three times a step, becomes a loop index. The rate either version achieves is printed by
every run rather than asserted here."""

LAMBDA = 3.0 / 16.0
"""The TRT "magic parameter". See the module docstring: this value is what puts a
straight no-slip wall on the link midpoint independently of the viscosity."""

CS2 = 1.0 / 3.0

STIFF_TAU = 0.52
"""Below this tau_plus a run is reported as too stiff for its lattice rather than left
to look like a defect.

The number is where the trouble starts and not a cliff: tau_plus = nu/c_s^2 + 1/2, so
tau_plus -> 1/2 is nu -> 0 and the collision stops relaxing anything. The graded cavity
sits at 0.538 and converges; a 16-cell cavity at the same Reynolds number sits at 0.5048
and does not converge at all, which is a configuration that is easy to ask for by
halving `--n` and hard to diagnose from the output. Nothing is refused -- `solve` runs
whatever it is given and the residual it reports is the honest answer."""

_AX = (slice(None), None, None)  # broadcast a per-direction vector over the lattice


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """The second-order Maxwellian the collision relaxes towards, shape (9, ny, nx)."""
    cu = (CX[_AX] * ux + CY[_AX] * uy) / CS2
    usq = 0.5 * (ux * ux + uy * uy) / CS2
    return W[_AX] * rho * (1.0 + cu + 0.5 * cu * cu - usq)


def rates(nu: float, scheme: str = "trt") -> tuple[float, float]:
    """(omega_plus, omega_minus) for a kinematic viscosity in lattice units.

    `nu` fixes the even rate and nothing else: tau_plus = nu/c_s^2 + 1/2. The odd rate
    is free, and the two schemes are the two ways of spending it -- `bgk` sets it equal
    (one relaxation time, the wall location then depends on nu) and `trt` sets it from
    LAMBDA (the wall location does not).
    """
    tau_plus = nu / CS2 + 0.5
    if scheme == "bgk":
        tau_minus = tau_plus
    elif scheme == "trt":
        tau_minus = 0.5 + LAMBDA / (tau_plus - 0.5)
    else:
        raise ValueError(f"relaxation scheme {scheme!r} is neither 'trt' nor 'bgk'")
    return 1.0 / tau_plus, 1.0 / tau_minus


def solve(
    solid: np.ndarray,
    nu: float,
    *,
    wall_velocity: tuple[np.ndarray, np.ndarray] | None = None,
    force: tuple[float, float] = (0.0, 0.0),
    scheme: str = "trt",
    reference_velocity: float = 1.0,
    steps: int | None = None,
    tolerance: float = 1e-7,
    check_every: int = 100,
    max_steps: int = 4_000_000,
    progress: bool = False,
) -> dict[str, Any]:
    """Run the lattice until it stops moving, and report the field and the cost.

    `solid` is the geometry: a boolean array, True where the fluid is not. Anything that
    can be voxelised into one can be solved, which is the reason to have this at all.
    The domain is periodic in both directions because streaming is `np.roll`; a case
    that should not wrap says so by being walled, as both cases below are in y.

    `wall_velocity` is a pair of arrays over the same shape, read only on solid nodes:
    that is where the lid of a cavity lives. `force` is a body force per unit volume,
    uniform, which is how a periodic channel is driven.

    Stops at `steps` if given, otherwise when the largest velocity change over
    `check_every` steps falls below `tolerance * reference_velocity`. The residual it
    stopped at is reported either way, because a run that hit `max_steps` and a run that
    converged are not the same answer and the difference is one number.
    """
    solid = np.asarray(solid, dtype=bool)
    fluid = ~solid
    if not fluid.any():
        raise ValueError("every node is solid: there is no fluid to solve for")
    omega_plus, omega_minus = rates(nu, scheme)

    uwx, uwy = (np.zeros_like(solid, float), np.zeros_like(solid, float))
    if wall_velocity is not None:
        uwx, uwy = (np.asarray(component, float) + np.zeros_like(solid, float)
                    for component in wall_velocity)
    # The boundary work is O(wall nodes), not O(lattice): the solid nodes are gathered
    # once, by flat index, and the bounce-back touches only those columns.
    wall = np.flatnonzero(solid.ravel())
    # The Ladd term for a moving wall: reflecting into direction j picks up
    # 6*w_j*(c_j.u_wall), because OPP flips the sign of c and leaves w alone.
    ladd = (6.0 * W[_AX] * (CX[_AX] * uwx + CY[_AX] * uwy)).reshape(9, -1)[:, wall].copy()

    fx, fy = float(force[0]), float(force[1])
    driven = bool(fx or fy)
    f = equilibrium(np.ones_like(solid, float), np.zeros(solid.shape), np.zeros(solid.shape))
    streamed = np.empty_like(f)
    ux = np.zeros(solid.shape)
    uy = np.zeros(solid.shape)

    limit = steps if steps is not None else max_steps
    stop = tolerance * abs(reference_velocity)
    previous_ux, previous_uy = ux.copy(), uy.copy()
    residual = float("inf")
    done = 0
    started = time.monotonic()

    while done < limit:
        held = f.reshape(9, -1)[:, wall].copy()  # solid nodes do not collide

        rho = f.sum(axis=0)
        # Guo's velocity: the body force contributes half a step to the momentum the
        # equilibrium is built from. Without it the force is first-order accurate.
        ux = (f[1] - f[3] + f[5] - f[6] - f[7] + f[8] + 0.5 * fx) / rho
        uy = (f[2] - f[4] + f[5] + f[6] - f[7] - f[8] + 0.5 * fy) / rho

        # The part of the equilibrium that does not depend on the direction, and the
        # part of Guo's source that is even in c, both shared across a pair.
        base = 1.0 - 1.5 * (ux * ux + uy * uy)
        force_dot_u = 3.0 * (ux * fx + uy * fy) if driven else 0.0

        feq = W[0] * rho * base
        f[0] -= omega_plus * (f[0] - feq)
        if driven:
            f[0] -= (1.0 - 0.5 * omega_plus) * (W[0] * force_dot_u)

        for i, j in PAIRS:
            cu = CX[i] * ux + CY[i] * uy
            shared = W[i] * rho * (base + 4.5 * cu * cu)
            swing = W[i] * rho * (3.0 * cu)
            # feq[i] = shared + swing, feq[j] = shared - swing: c_j = -c_i, w_j = w_i.
            di = f[i] - (shared + swing)
            dj = f[j] - (shared - swing)
            even = 0.5 * (di + dj)
            odd = 0.5 * (di - dj)
            f[i] -= omega_plus * even + omega_minus * odd
            f[j] -= omega_plus * even - omega_minus * odd
            if driven:
                cf = CX[i] * fx + CY[i] * fy
                # Guo's source split by parity: 3*w*(c.F) is odd in c and takes the odd
                # rate; the rest is even in c and takes the even one.
                odd_source = (1.0 - 0.5 * omega_minus) * (3.0 * W[i] * cf)
                even_source = (1.0 - 0.5 * omega_plus) * (
                    W[i] * (9.0 * cu * cf - force_dot_u))
                f[i] += odd_source + even_source
                f[j] += even_source - odd_source

        f.reshape(9, -1)[:, wall] = held

        for i in range(9):
            shifted = f[i]
            if CY[i]:
                shifted = np.roll(shifted, int(CY[i]), axis=0)
            if CX[i]:
                shifted = np.roll(shifted, int(CX[i]), axis=1)
            streamed[i] = shifted

        # Halfway bounce-back, on the solid node: what streamed in is reflected in
        # place, and the next streaming step carries it back into the fluid.
        view = streamed.reshape(9, -1)
        arrived = view[:, wall]
        view[:, wall] = arrived[OPP] + ladd

        f, streamed = streamed, f
        done += 1
        if steps is None and done % check_every == 0:
            residual = float(max(np.abs(ux - previous_ux)[fluid].max(),
                                 np.abs(uy - previous_uy)[fluid].max()))
            if progress:
                print(f"  {done:>9,} steps  residual {residual:.3e}", file=sys.stderr)
            if residual < stop:
                break
            previous_ux, previous_uy = ux.copy(), uy.copy()

    elapsed = time.monotonic() - started
    masked = np.where(fluid, 1.0, np.nan)
    return {
        "ux": ux * masked,
        "uy": uy * masked,
        "rho": rho * masked,
        "solid": solid,
        "steps": done,
        "converged": bool(steps is None and residual < stop),
        "residual": residual,
        "nu": nu,
        "scheme": scheme,
        "omega_plus": omega_plus,
        "omega_minus": omega_minus,
        "seconds": elapsed,
        "mlups": fluid.sum() * done / elapsed / 1e6 if elapsed > 0 else 0.0,
    }


# -- the two cases whose answers are known from outside a solver --------------------
#
# Both are set up so that the walls sit on the link midpoints, which is where halfway
# bounce-back puts them: with `n` fluid rows between two solid rows the channel or
# cavity is exactly `n` lattice units across, and fluid row j (counting the solid row
# as 0) has its centre at y = (j - 0.5) / n of the way across. Getting that half cell
# wrong is a 1/(2n) error in every length in the problem and it looks like a solver
# defect, so `centres` is the one place it is written down.


def centres(n: int) -> np.ndarray:
    """The fluid node centres across a wall-to-wall span of 1, as fractions of it."""
    return (np.arange(n) + 0.5) / n


def poiseuille(n: int = 32, nu: float = 0.1, u_max: float = 0.05, nx: int = 4,
               **kwargs: Any) -> dict[str, Any]:
    """Plane Poiseuille flow: a periodic channel `n` cells tall, driven by a body force.

    The answer is exact and needs no paper. For a force per unit volume G across a
    channel of height H with dynamic viscosity mu = rho*nu,

        u(y) = G/(2*mu) * y * (H - y),   u_max = G*H^2/(8*mu)

    and G is chosen here so that u_max comes out at `u_max`, which keeps the Mach number
    where the scheme's compressibility error is small.
    """
    solid = np.zeros((n + 2, nx), dtype=bool)
    solid[0, :] = True
    solid[-1, :] = True
    force = 8.0 * u_max * nu / n**2
    result = solve(solid, nu, force=(force, 0.0),
                   reference_velocity=u_max, **kwargs)
    y = centres(n) * n
    result["case"] = "poiseuille"
    result["height"] = n
    result["force"] = force
    result["u_max_exact"] = force * n**2 / (8.0 * nu)
    result["y"] = y
    result["u_profile"] = result["ux"][1:-1, nx // 2]
    result["u_exact"] = force / (2.0 * nu) * y * (n - y)
    return result


def cavity(n: int = 128, re: float = 1000.0, u_lid: float = 0.1,
           **kwargs: Any) -> dict[str, Any]:
    """The lid-driven cavity: a square box, the top wall sliding at `u_lid`.

    The viscosity follows from the Reynolds number and the box: nu = u_lid * n / Re,
    with the cavity exactly `n` lattice units on a side. At Re 1000 and n 128 that is
    nu = 0.0128, so tau_plus = 0.538 -- close to the 1/2 at which a single-relaxation
    scheme loses stability, which is why `trt` is the default here.
    """
    solid = np.zeros((n + 2, n + 2), dtype=bool)
    solid[0, :] = solid[-1, :] = True
    solid[:, 0] = solid[:, -1] = True
    uwx = np.zeros_like(solid, float)
    uwx[-1, :] = u_lid  # the lid is the top row of solid nodes
    nu = u_lid * n / re
    result = solve(solid, nu, wall_velocity=(uwx, np.zeros_like(uwx)),
                   reference_velocity=u_lid, **kwargs)
    result["case"] = "cavity"
    result["n"] = n
    result["re"] = re
    result["u_lid"] = u_lid
    # Ghia's two centrelines, in his normalisation: u/u_lid along x = 0.5 against y, and
    # v/u_lid along y = 0.5 against x. 0.5 falls between two nodes for even n, so both
    # are the mean of the two straddling columns/rows rather than a nearest-node pick.
    interior = (slice(1, -1), slice(1, -1))
    ux, uy = result["ux"][interior], result["uy"][interior]
    mid = n // 2
    columns = slice(mid - 1, mid + 1) if n % 2 == 0 else slice(mid, mid + 1)
    result["coordinate"] = centres(n)
    result["u_centreline"] = ux[:, columns].mean(axis=1) / u_lid
    result["v_centreline"] = uy[columns, :].mean(axis=0) / u_lid
    return result


CASES = {"poiseuille": poiseuille, "cavity": cavity}


# -- a picture, because a field is not readable as numbers -------------------------


def figure(result: dict[str, Any], path: Path) -> None:
    """Velocity magnitude and the profile the case is graded on, in one png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))
    speed = np.hypot(result["ux"], result["uy"])
    left.imshow(speed, origin="lower", cmap="viridis")
    left.set_title(f"{result['case']}: |u| in lattice units")
    left.set_xticks([])
    left.set_yticks([])

    if result["case"] == "cavity":
        right.plot(result["u_centreline"], result["coordinate"], label="u on x = 0.5")
        right.plot(result["coordinate"], result["v_centreline"], label="v on y = 0.5")
        right.set_xlabel("u/u_lid, or x")
        right.set_ylabel("y, or v/u_lid")
        right.set_title(f"Re {result['re']:g}, {result['n']} cells")
    else:
        right.plot(result["u_exact"], result["y"], label="exact parabola")
        right.plot(result["u_profile"], result["y"], "o", ms=3, label="lattice")
        right.set_xlabel("u (lattice units)")
        right.set_ylabel("y (cells)")
        right.set_title("Poiseuille, force-driven")
    right.legend(fontsize=8)
    right.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def report(result: dict[str, Any]) -> str:
    tau_plus = 1.0 / result["omega_plus"]
    lines = [f"# {result['case']}",
             f"  scheme:     {result['scheme']}  "
             f"omega_plus {result['omega_plus']:.6f}  "
             f"omega_minus {result['omega_minus']:.6f}  "
             f"tau_plus {tau_plus:.5f}",]
    if tau_plus < STIFF_TAU:
        # Stated rather than refused: this is where a run stops converging and starts
        # looking like a solver bug. nu = u_lid*n/Re, so the way out is a finer lattice
        # or a larger u_lid, and a case run here is cheap to re-run and expensive to
        # misread.
        lines.append(f"  NOTE:       tau_plus is {tau_plus:.5f}, within "
                     f"{STIFF_TAU - 0.5:g} of the 1/2 where the collision stops "
                     "relaxing. nu = u_lid*n/Re is too small for this lattice: a run "
                     "here converges slowly or not at all, and the residual below says "
                     "which")
    lines += [
             f"  nu:         {result['nu']:.6g} (lattice units)",
             f"  steps:      {result['steps']:,}"
             f"  ({'converged' if result['converged'] else 'stopped'}, "
             f"residual {result['residual']:.3e})",
             f"  cost:       {result['seconds']:.1f} s at "
             f"{result['mlups']:.2f} million lattice updates per second"]
    if result["case"] == "poiseuille":
        measured = float(np.nanmax(result["u_profile"]))
        exact = result["u_max_exact"]
        error = np.abs(result["u_profile"] - result["u_exact"])
        lines += [f"  u_max:      {measured:.9f} against the exact "
                  f"{exact:.9f} ({(measured / exact - 1) * 100:+.4f}%)",
                  f"  profile:    max |u - u_exact| / u_max = "
                  f"{error.max() / exact:.3e} over {len(error)} rows"]
    else:
        lines += [f"  u range:    {np.nanmin(result['u_centreline']):+.5f} to "
                  f"{np.nanmax(result['u_centreline']):+.5f} (u/u_lid on x = 0.5)",
                  f"  v range:    {np.nanmin(result['v_centreline']):+.5f} to "
                  f"{np.nanmax(result['v_centreline']):+.5f} (v/u_lid on y = 0.5)"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--n", type=int, default=None, help="cells across the channel or cavity")
    parser.add_argument("--re", type=float, default=1000.0, help="Reynolds number (cavity)")
    parser.add_argument("--nu", type=float, default=0.1,
                        help="kinematic viscosity in lattice units (poiseuille)")
    parser.add_argument("--relaxation", choices=("trt", "bgk"), default="trt")
    parser.add_argument("--steps", type=int, default=None,
                        help="run exactly this many steps instead of to a residual")
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--out", type=Path, default=None, help="write a png here")
    parser.add_argument("--npz", type=Path, default=None, help="write the fields here")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="no per-check residual line")
    args = parser.parse_args(argv)

    options: dict[str, Any] = {"scheme": args.relaxation, "steps": args.steps,
                               "tolerance": args.tolerance, "progress": not args.quiet}
    if args.n is not None:
        options["n"] = args.n
    if args.case == "cavity":
        options["re"] = args.re
    else:
        options["nu"] = args.nu
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
