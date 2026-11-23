"""DFSPH on NVIDIA Warp: the particle backend, for a free surface that actually moves.

`docs/sph-contract.md` is the interface and it was written before this file; this is the
implementation of it and nothing here is allowed to widen it.

## Written from the papers, not ported

The scheme is Bender and Koschier, "Divergence-Free Smoothed Particle Hydrodynamics"
(SCA 2015) and its journal version, "Divergence-Free SPH for Incompressible and Viscous
Fluids" (IEEE TVCG 23(3), 2017); the rigid coupling is Akinci, Ihmsen, Akinci, Solenthaler
and Teschner, "Versatile Rigid-Fluid Coupling for Incompressible SPH" (SIGGRAPH 2012);
the kernel is Monaghan's cubic spline and the viscosity is Monaghan (1992) with a
physical nu.

**It is not a port of SPlisHSPlasH.** SPlisHSPlasH is MIT and porting it would be
permitted, and its source is not available in this environment, so every equation below
was re-derived from the papers and checked dimensionally. There is therefore no
SPlisHSPlasH copyright notice in this file and no entry owed in a `LICENSES` file, and
that is a statement about provenance rather than about style: the alternative -- a notice
for code nobody copied -- is a false claim about where code came from.

The one thing taken from somebody else's code is the shape of three calls:
`wp.hash_grid_point_id`, `wp.hash_grid_query` and the `for index in query` loop, which is
the call pattern in Warp's own `examples/core/example_sph.py` (Apache-2.0, NVIDIA). That
is the documented API for `wp.HashGrid` and there is one way to write it; no expression
from that file is reproduced here.

## Why SPH at all, when there is a lattice next door

`freesurface.py` is a Korner-style free surface on the lattice and `freesurface.FROZEN`
says why it stopped. The short version is that a lattice free surface carries a one-cell
interface layer whose mass exchange has to be repaired every step, and a hull at a
waterline broke it. A particle free surface has no interface layer at all: the surface is
wherever the particles stop, there is nothing to reconstruct, and the hydrostatic
pressure on a floating body comes out of the same pressure solve as everything else.

That is also why the Akinci boundary coupling below is the part this file exists for. The
`wrench(body)` it produces is the force **on the body**, in newtons, the lattice
backend's sign convention (`lattice.stream`'s momentum exchange, which accumulates
`cj * (f_in + reflected)` with `cj` pointing from the fluid into the solid). Getting that
backwards has happened twice in this repository and
`tests/test_live_sph.py::test_the_floor_of_a_tank_carries_the_weight_of_the_water` is the
standing check on it: the sign AND the magnitude in one number.

## SI in, SI out, and no `units.Scaling` anywhere

The lattice backend has `units.py` because a lattice Boltzmann scheme has a fixed lattice
speed of sound and three numbers that have to be made to agree. DFSPH has none of that:
it is dimensionally consistent in metres, seconds and kilograms, there is no artificial
sound speed (the whole point of a projection-style solver), and the only scale check is a
CFL condition on the timestep. So everything in this file is SI, including the arrays on
the device, and `UnphysicalScaling` is still the refusal because it is the repository's
name for "this run cannot be set up as asked, and here is the knob".

Single precision throughout, like `lattice.py`, and for a less interesting reason: the
numbers are all O(1) to O(1e9) in SI at the resolutions this runs at (a mass of 1 mg, a
kernel gradient of 1.6e7 m^-4, the stiffness denominator at 3e9 kg^2/m^8), which is the
middle of float32's range rather than either end.

## The step, in order, and why that order

One substep is the 2015 paper's Algorithm 1, rotated by one so that the neighbour search
happens once instead of twice:

    1  pose        boundary particles follow their bodies (device, from `body_q`)
    2  grid        wp.HashGrid rebuilt over fluid AND boundary particles
    3  densities   rho_i = sum_j m_j W_ij + sum_b psi_b W_ib     (Akinci)
    4  factors     alpha_i, the DFSPH stiffness coefficient
    5  divergence  the divergence-free solver, on this substep's own neighbourhood
    6  nonpressure gravity and viscosity, as an acceleration, then v += dt a
    7  density     the constant-density solver
    8  advect      x += dt v

The paper does 5 at the *end* of the step, after advection, with a second neighbour
search and a second density/alpha pass to go with it. Steps 1 to 4 of the next substep
are exactly that second pass, so running the divergence solver at the top of the substep
instead of the bottom is the same sequence of operations with one neighbour search per
substep rather than two. The only difference is at the ends of a run: the first substep
divergence-corrects a velocity field that is usually zero, and the last substep leaves
its advected velocity uncorrected until the next `step()`.

Not timed against the paper's literal order, because writing that order out in order to
time it would mean carrying a second step loop; what is stated instead is the operation
count, which is the whole of the saving. The paper's order runs `grid.build`, `densities`,
`factors` and `boundary_volumes` twice a substep and this runs them once; measured on the
2,394-particle tank those four are 19.9 ms of a 143 ms substep, so the literal order would
add about 14% and no accuracy that the divergence solver's own residual can see.

## What `NewtonCoupling` still needs, which is not in this file

`coupling_style = "particles"` is declared and `couple.py` branches on it, so shape
scales, poses and the wrench are no longer divided by a `dx` this backend does not have.
Two things are still open and both of them are in `couple.py`, so they are reported rather
than worked around here:

* `couple.py:515` and `:565` read `world.scaling` unconditionally. There is no
  `units.Scaling` on an `SphWorld` and there should not be -- that class exists to make
  a lattice's three numbers agree and SPH has none of them -- so
  `NewtonCoupling(model, solver, sph_world)` raises
  `AttributeError: 'SphWorld' object has no attribute 'scaling'` from `_push_poses`
  before it has stepped anything. Both lines want `self.world.scaling.dt` to be 1.0 on
  the particle path, which is the same `_style` guard the wrench scaling already has.
* `couple.py:398` hands the world a device shape table (`_newton_shapes`) for the lattice
  to voxelise every step. This backend samples boundary particles on the **host**, at
  `add_shape` time, because the sampling is a one-off: a rigid shape's particles never
  change in its own frame and `pose_boundary` moves them from `body_q` on the device. So
  the table is stored and ignored, and a Newton model's shapes never become boundary
  particles -- an `SphWorld` driven by `NewtonCoupling` today has bodies with no wetted
  surface and therefore a wrench of exactly zero, which reads as a coupling that is not
  wired up. The bridge is a host-side loop at coupling construction calling
  `SphWorld.add_shape` once per Newton shape, which is the right shape of fix and belongs
  beside `_static_shape_arrays` rather than here.

## There is no periodic axis, and this is why rather than an oversight

`docs/sph-contract.md` asks for periodicity on a chosen axis, so that a quasi-2D case
does not pay for a third dimension it does not use, and **it is not here.** The reason is
not that it is hard physics; it is that it is not local to this file.

`wp.HashGrid` has no periodic mode. The two ways to get one are a wrapped query, which
would mean not using `wp.HashGrid` (the contract names it, and its query is a C++ builtin
with no hook for a wrap), or ghost particles: duplicate every particle within one support
of each periodic face onto the far side, every substep, on the device. Ghosts are the
standard answer and they are cheap to *create*. What they are not cheap to get right is
everything downstream of the particle index:

* `hash_grid_point_id` returns an index into the array the grid was built over, so the
  array has to grow by the ghost count and the ghost count changes as particles move --
  which means either a reallocation inside the step loop or a fixed-capacity slab and a
  refusal when it overflows;
* `i < n_fluid` is how every kernel here tells fluid from boundary, and a ghost of a
  fluid particle and a ghost of a boundary particle land on opposite sides of that test;
* `apply_pressure` writes `v[i]`, and a ghost must write to its *original*, which is an
  indirection and an atomic rather than a store;
* the Akinci wrench accumulates per boundary particle, and a ghost boundary particle
  would accumulate its body's force a second time at the wrong arm.

That is four invariants and an allocation policy, in the file that holds the wrench the
whole redirect exists for. Built in a hurry it would be a plausible-looking run with a
double-counted force in it, which is exactly the failure this project has already paid
for twice. It is a day of work done properly, with its own test for each of the four, and
it is better asked for as such than slipped in.

**The cheap substitute, which is what a quasi-2D case should use today:** make the thin
axis two kernel supports wide -- four particle spacings -- and wall both its faces. The
Akinci walls then hold the fluid in a slab whose interior particles have a full
neighbourhood. The cost against a true periodic axis is the walls themselves: four fluid
planes plus two boundary layers each side is eight planes where a periodic axis would
carry four. A factor of two on one axis, not a factor of the third dimension, and the
walls are also what a real flume has.

## The one host read inside the step, named rather than hidden

The repository's rule is no host round trips in the step loop, and both DFSPH solvers
iterate *to a tolerance*, which means somebody has to look at the error. So each solver
iteration copies the residual field back and takes its mean in numpy. That is a device
synchronise per solver iteration -- three to six per substep in practice -- and it is
the standard DFSPH implementation shape, but it is a host round trip and calling it
anything else would be a lie. `check_every=0` turns it off and runs `max_iterations`
blind, which is the setting for a GPU coupled run where the sync is the cost;
`report()` says which was in force.

**It used to be eight bytes, and the eight bytes were wrong.** A `wp.atomic_add` summed
`n_fluid` float32 residuals on the device and the host read a two-float array. float32
addition is not associative and a CUDA atomic adds in retirement order, so that mean
moved in its sixth digit from run to run on byte-identical code -- and the comparison it
feeds is a *branch*. A sixth-digit difference therefore bought one run an extra solver
iteration that the next run did not take, after which the two velocity fields had nothing
to do with each other. Four bytes of ordering noise through a branch was a measurable
fraction of the sloshing period; `tests/test_live_sph.py` and this module's history have
the numbers. numpy's pairwise sum over a fixed-length array is the same arithmetic every
time, so the convergence decision is now a function of the field and nothing else.

What that costs, measured on the 3,840-particle sloshing tank on an H100, five runs each
side: 15 KiB a read instead of 8 bytes, at 4.8 density plus 2.0 divergence iterations a
substep, for a step time of **6.890 ms against 6.756 ms** -- 2.0%, and the two ranges
overlap (6.54 to 7.26 ms against 6.55 to 7.02). The cost of this read is the synchronise,
not the bytes: the 35 microseconds the old eight were measured at is already more than a
15 KiB DMA. On a CPU the sync is free. A case large enough for the transfer to matter has
`check_every` to raise and `check_every=0` to turn it off.

Everything else -- `positions()`, `densities()`, `wrench()`, `density_error()` -- is a
host read for reports and figures, outside the loop, same as the lattice backend.


**This backend ships as beta in v0.1** (`DECISIONS_LIVE.md` D21), beside the
lattice backend for single-phase flow, which is stable. What beta means concretely:
the fluid-only cases are bit-for-bit reproducible, the floating ones are not, and
the per-step force noise on a freely floating hull is 101.5% RMS against 23.0% held
with the mean correct in both. No floating-body number should come from one run.

**The force noise has a cause and a fix, and the fix is the substep.** `CFL`'s docstring
has the measurement: a step of `CFL/4` takes the per-step RMS on a freely floating hull
from 99-115% to **1.3 to 8.3%** across a 1.67x refinement, falling as the resolution
rises -- which the 101.5%-to-114.9% baseline did the opposite of. The same step takes a
*held* hull from 23.0% to 1.3%, so it is the pressure field converging in time and not
the explicit coupling's lag. It is not the DFSPH tolerance (100x tighter buys 13% at
6.9x the wall clock, with a 50x better density field) and it is not spurious density
oscillation (delta-SPH buys nothing free and makes the held hull worse). The default is
unchanged because `CFL` is also the fluid's own stability number and a fluid-only case
has no wrench to be noisy; `SphWorld.wrench` has the optional filter for a caller who
cannot afford the shorter step.
"""

from __future__ import annotations

import math
import time
from typing import Any, Sequence

import numpy as np
import warp as wp

from . import lattice as L
from .units import UnphysicalScaling, fluid as resolve_fluid, require, version_note
from .world import FACE_ORDER
# `lattice` and `world` for the shape codes, the face codes and the face *order*, which
# are conventions and not values: a second copy of `("-x", "+x", "-y", ...)` here is how
# a tank ends up with its floor on the ceiling. Neither import is expensive -- Warp
# compiles a kernel on its first launch, not on import, and no D3Q19 kernel is ever
# launched from this file.

__all__ = ["SphWorld", "SOLVERS", "SUPPORT_FACTOR", "CFL"]

wp.set_module_options({
    # Same reasoning as `lattice.py`: nothing in this extension differentiates through a
    # fluid, and the adjoint of every kernel below is generated otherwise. Measured on
    # a cold kernel cache (`wp.config.kernel_cache_dir` pointed at a fresh temporary
    # directory, `wp.load_module` timed): 1.92 s with it off against 6.21 s with it on,
    # so 3.2x the codegen and compile for code no caller can reach.
    "enable_backward": False,
    # There is no `Q` loop here -- every loop below is a neighbour loop with a runtime
    # bound, which Warp cannot unroll at all -- so `max_unroll` is left at its default.
    # The 7.9-to-29.8 MLUPS effect `lattice._max_unroll` measures has no analogue in a
    # particle kernel, and raising it would only be cargo.
})

DTYPE = wp.float32

SOLVERS = ("dfsph",)
"""`solver=` accepts one name today.

A WCSPH path is in `docs/sph-contract.md` as optional and is not here. It would be a
useful *reference* -- Tait's equation of state is twenty lines and
`toolbox/sph.py` already has a 2D one to check against -- and it is not a product: its
timestep is acoustic, `dt ~ 0.2 h / c0`, so a second of a dam break is tens of thousands
of steps where DFSPH takes hundreds. `toolbox/sph.py`'s own docstring carries that
measurement. Adding it here would mean a second pressure path to keep right for
something no case would ever run.
"""

SUPPORT_FACTOR = 4.0
"""Kernel support radius in units of the particle radius, so `h = 2 * spacing`.

Not a taste parameter. The cubic spline's summation density on the cubic lattice this
file fills is a pure function of this number, and it has one good value:

    h / spacing   neighbours in support   sum_j m_j W_ij / rho0
        1.75              26                   1.004537
        2.00              26                   0.999972
        2.50              80                   0.998517

At `h = 2 * spacing` a bulk lattice of particles of mass `rho0 * spacing^3` reads its own
rest density to 28 parts per million, which is what lets this file ship with no mass
calibration factor at all -- the fluid starts at rest *and* at rho0, so a column does not
settle on the first few hundred steps while the solver pushes it to a density the initial
condition never had. Widening to 2.5 buys 80 neighbours instead of 26 (smoother
gradients, three times the neighbour work) and costs a 0.15% standing density bias, which
is 50 times the bias here. 1.75 is worse on both counts.

26 is on the low side for a second-order SPH gradient and that is the honest cost of the
choice; the `density_error` numbers in `tests/test_live_sph.py` are what it measures to.
"""

CFL = 0.4
"""Courant number for the substep: a particle moves at most 0.4 of a spacing per substep.

0.4 and not 0.25 because DFSPH is not acoustically limited -- there is no sound speed in
the timestep -- so the only thing the step has to beat is the neighbourhood changing
faster than the solver can follow. 0.4 is the DFSPH papers' own figure. The peak speed it
is applied to is an estimate, not a measurement (`SphWorld._substep`), because measuring
it needs a host read of the velocity field every step.

## **0.4 is right for the fluid's motion and three times too long for its wrench**

This is the whole of the v0.1 force-noise finding: **0.4 converges where the particles
go and does not converge the pressure field that `wrench` integrates. A case that reads
a per-step force needs `CFL/3` or smaller.** `benchmarks/live/force_noise.py` is the
measurement -- one settle, then every candidate released from the identical settled
state, two independent settles of each -- and this is it, as the per-step RMS of the
vertical fluid force on a 64 x 64 x 32 mm hull at 0.5 of water's density in 60 mm of
water, over the last 40% of a six-heave-period release, on an H100:

    effective CFL     2.5 mm radius      2.0 mm            1.5 mm
      0.4   (dt)      92.9 / 94.0%    104.6 / 99.2%    114.9 / 115.0%
      0.2   (dt/2)    92.1 / 89.8%     97.3 / 97.2%     90.8 /  89.4%
      0.133 (dt/3)     5.5 /  5.3%          --               --
      0.1   (dt/4)     5.1 /  5.4%      1.8 /  2.6%      1.3 /   1.3%
      0.067 (dt/6)     6.9 /  5.9%          --               --
      0.05  (dt/8)     5.8 /  5.4%          --           1.4 /   1.6%

**It is a cliff and not a slope.** Halving dt does nothing -- 97.3% against 104.6% at
2 mm -- and thirding it takes 93% to 5.4%. Between `CFL/2` and `CFL/3` the per-step force
stops being noise with a mean in it and becomes a force: at 1.5 mm and `CFL/4` the
vertical force stays inside **0.955 to 1.040 `m g` on every single step**, against 0.004
to 2.65 at `CFL 0.4`, and the steps where the hull felt essentially nothing -- 27% of
them at 1.5 mm -- go to **zero**. `CFL/6` and `CFL/8` are no better than `CFL/4`, so
`CFL/4` is the floor and not a point on the way down.

**And it is a ratio, not an absolute step**, which is the one thing that had to be
measured rather than assumed because the two are indistinguishable at one resolution:
476 us is quiet at a 2.5 mm radius and 428 us is *noisy* at 1.5 mm. A fixed millisecond
figure would have the sign backwards. `CFL` is the knob.

**It is this file's step and not the coupling's, which is also measured rather than
assumed.** The obvious story is the explicit coupling -- the fluid sees `body_vel` from
the start of the step and the body sees the wrench from the end of it -- and it is wrong,
because the same cliff is there on a hull Newton **never steps**. Held at the Archimedes
draft, where `body_vel` is identically zero and there is no lag to shorten, `CFL/4` takes
the per-step RMS from **22.0% to 1.3%**. Shortening the step is converging the pressure
field in time, and the coupling lag is a second and smaller term on top of it: freezing
the boundary particles' velocity on a hull that is being integrated (not a remedy -- a
hull with no surface velocity has no added mass) takes 99-115% down to only 42-66%,
about half the free-minus-held excess and all of the near-zero steps.

What it costs, measured as wall clock per simulated second on the same hardware: **1.9x
at `CFL/3` and 2.3 to 2.5x at `CFL/4`**, and not the 3x and 4x the step count suggests,
because the constant-density solver's iteration count falls from 8.7-9.7 a substep to
exactly **2.0** -- the papers' floor -- once the step is small enough that one substep's
density error is already inside the tolerance.

**The spread, because one run settles nothing on a floating body.** Every figure above
is two independent settles, and `CFL/4` reproduced to 0.8 percentage points or better at
2.0 and 1.5 mm. At 2.5 mm -- 3.2 particle spacings through the draft, the coarse end --
a third settle read **8.3%** against the 5.1% and 5.4% of the first two, so the spread
there is 3.2 points and the margin to a 10% target is thin. The margin *grows* with
refinement, which is the opposite of the baseline's behaviour and is the thing that makes
this a convergent result rather than a lucky setting.

**Nothing here changes the default,** and that is deliberate: `CFL` is also the number
that keeps the neighbourhood from outrunning the solver, a fluid-only case has no wrench
to be noisy, and `docs/sph-contract.md` sets this constant. A case that reads a per-step
force asks for the shorter step through `dt=`, which `_substep` already accepts and
already refuses an inconsistent `substeps` against:

    world = SphWorld(..., particle_radius=r)      # dt is one CFL substep
    world = SphWorld(..., particle_radius=r, dt=world_dt / 4.0)   # or a quarter of it

and `openreynolds/live/sph_cases.py`'s floating runners are where that choice belongs,
which is not this file. The three knobs that do NOT fix it are written down where each
one lives: `diffusivity` and `diffuse_density` for delta-SPH, `SphWorld.wrench` for the
filter, and the tolerance in this list -- `tolerance=1e-5` drives the mean compression
error from 0.097% to 0.002%, a 50x better density field, and buys **13%** off the force
RMS at 6.9x the wall clock. The noise is not under-converged pressure; it is pressure
converged at the wrong instant.
"""

_EPS = 1.0e-9

DIFFUSION_EIGENVALUE = 23.2963
"""`lambda_max h^2` for `diffuse_density`'s Laplacian on this file's own lattice fill.

**Measured, not assumed, and the assumption it replaces was wrong by 3x.** The density
diffusion is an explicit Laplacian, so a substep multiplies a mode by
`1 - D dt lambda`, and the sharpest mode a cubic lattice at spacing `h/2` carries is the
checkerboard. Launching `diffuse_density` once on a checkerboard of unit amplitude and
reading the amplitude back gives `lambda h^2` directly, and it is the same number at
every resolution because it is a property of the kernel and the lattice and not of the
run: **23.2963 at particle radii of 2, 2.5 and 5 mm, to four decimals.**

The limit `_check` holds is therefore `D dt lambda <= 1`, which is monotone decay of
that mode. Stability alone would allow `<= 2`, and the band between the two is a
diffusion that *alternates the sign* of the noise it is removing every substep -- which
is not a diffusion, and on a one-sided density constraint it is a rectifier. Measured at
`D dt lambda = 2.209`: a unit checkerboard came back at **-1.209**, so it had grown.

Taking the viscous term's `0.125 h^2/nu` for this operator instead -- which is what
writing it by analogy gives, and what this file did first -- allows
`D dt lambda = 2.9` and is past even the stability edge. `0.125` is Monaghan's figure
for Monaghan's viscosity, which carries a `2(d+2) = 10` and a different normalisation;
it is not a property of SPH Laplacians in general. One checkerboard launch is cheaper
than the argument, and `tests/test_live_sph.py` keeps it.
"""

# -- the cubic spline ------------------------------------------------------------------
#
# `sigma = 8 / (pi h^3)` and `q = r/h` with the support at `q = 1`, which is Monaghan's
# cubic spline written with the support radius as the length scale rather than the
# "smoothing length" h' = h/2. Both conventions are in the literature and they are the
# same kernel; this one is used because every neighbour query below is in terms of the
# support, so a second length that is half of it is a second thing to get wrong.
#
#     W(q) = sigma * ( 6(q^3 - q^2) + 1 )   q <= 1/2
#            sigma * ( 2(1 - q)^3 )         1/2 < q <= 1
#            0                              q > 1
#
# Normalisation checked by quadrature, not by trust: 4 pi int_0^h W r^2 dr = 1.000000.
# `tests/test_live_sph.py::test_the_kernel_integrates_to_one_over_its_support` holds it.


@wp.func
def _spline(r: DTYPE, h: DTYPE, sigma: DTYPE) -> DTYPE:
    q = r / h
    if q >= DTYPE(1.0):
        return DTYPE(0.0)
    if q <= DTYPE(0.5):
        return sigma * (DTYPE(6.0) * (q * q * q - q * q) + DTYPE(1.0))
    u = DTYPE(1.0) - q
    return sigma * DTYPE(2.0) * u * u * u


@wp.func
def _spline_grad(d: wp.vec3, r: DTYPE, h: DTYPE, sigma: DTYPE) -> wp.vec3:
    """grad_i W(x_i - x_j), so it points from i towards j for a decaying kernel.

    `d` is `x_i - x_j` and `r` its length. The `r > eps` guard is the self term: a
    neighbour query returns the particle itself and `d/r` is not defined there. Returning
    zero is correct and not a fudge -- the kernel gradient at the origin genuinely is
    zero by symmetry.
    """
    q = r / h
    if q >= DTYPE(1.0) or r < DTYPE(_EPS):
        return wp.vec3(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
    scale = sigma / h
    if q <= DTYPE(0.5):
        scale *= DTYPE(6.0) * (DTYPE(3.0) * q * q - DTYPE(2.0) * q)
    else:
        u = DTYPE(1.0) - q
        scale *= DTYPE(-6.0) * u * u
    return d * (scale / r)


# -- boundary particles follow their bodies --------------------------------------------


@wp.kernel
def pose_boundary(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    local: wp.array(dtype=wp.vec3),
    body: wp.array(dtype=wp.int32),
    body_pose: wp.array(dtype=wp.transform),
    body_vel: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    n_fluid: wp.int32,
):
    """Boundary particle positions and velocities from the body table, on the device.

    `local` is the particle in the body's own frame, in metres, captured when the shape
    was sampled. This is the whole of "boundary particles follow `body_q` on the device"
    from the contract: one transform per particle, no host round trip, and a body with
    `body < 0` (scenery -- the tank, a ground plane) is left exactly where it was sampled.

    `body_vel` is Newton's `body_qd` packing, linear then angular in the world frame
    about the centre of mass, which is what `lattice._wall_velocity` reads too.
    """
    tid = wp.tid()
    i = tid + n_fluid
    b = body[i]
    if b < 0:
        return
    pose = body_pose[b]
    x[i] = wp.transform_point(pose, local[i])
    vel = body_vel[b]
    linear = wp.vec3(vel[0], vel[1], vel[2])
    angular = wp.vec3(vel[3], vel[4], vel[5])
    v[i] = linear + wp.cross(angular, x[i] - body_com[b])


# -- Akinci boundary volumes -----------------------------------------------------------


@wp.kernel
def boundary_volumes(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    psi: wp.array(dtype=DTYPE),
    n_fluid: wp.int32,
    h: DTYPE,
    sigma: DTYPE,
    rho0: DTYPE,
):
    """`psi_b = rho0 / sum_{b'} W(x_b - x_b')`, Akinci et al. 2012 eq. (3) and (4).

    The sum is over boundary particles only, including `b` itself, and it is the whole
    idea of that paper: a boundary particle's contribution to a fluid particle's density
    is weighted by how crowded its *own* neighbourhood is, so a surface sampled twice as
    densely does not push twice as hard. Without it the boundary force depends on how
    the sampler happened to lay particles down, which is not a physical parameter.

    `psi` has units of kilograms -- it is a pseudo-mass, `rho0 * V_b` -- and the
    consistency check is that for a boundary region sampled on the same lattice as the
    fluid, `sum W = 1/spacing^3` to 28 ppm (see `SUPPORT_FACTOR`), so `psi_b` comes out
    equal to a fluid particle's mass. A single surface layer gets roughly twice that,
    which is the correction doing its job.

    Rebuilt every step rather than cached, because a body that rotates into a corner of
    the tank has boundary neighbours it did not have last step, and a cached psi would
    make the corner push harder than the wall.

    **It is the most expensive single sweep here and the price is known**: 10.23 ms of a
    143 ms substep at 2,394 particles, 7.2%, because it loops over every boundary particle
    and queries every particle. All of that is wasted on a scene whose only boundaries are
    the tank -- walls do not move, so their psi is a constant -- and the guard is five
    lines: skip it unless `_device_bodies_dirty` was set since the last one. It is not
    here because 7% of the static case is not worth a staleness bug in the term that
    scales every boundary force in the run, and the usual coupled case has a body moving
    every step and would recompute anyway. Written down rather than taken.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i < n_fluid:
        return
    xi = x[i]
    total = DTYPE(0.0)
    for j in wp.hash_grid_query(grid, xi, h):
        if j >= n_fluid:
            total += _spline(wp.length(xi - x[j]), h, sigma)
    psi[i] = rho0 / wp.max(total, DTYPE(_EPS))


# -- densities and the DFSPH factor ----------------------------------------------------


@wp.kernel
def densities(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    psi: wp.array(dtype=DTYPE),
    rho: wp.array(dtype=DTYPE),
    mass: DTYPE,
    n_fluid: wp.int32,
    h: DTYPE,
    sigma: DTYPE,
):
    """`rho_i = sum_j m_j W_ij + sum_b psi_b W_ib`, fluid particles only.

    Summation density and not the continuity equation, which is the opposite of the
    choice `toolbox/sph.py` made and for the opposite reason. That script is weakly
    compressible: its pressure comes from Tait's equation of state, so a density that
    reads low at a free surface is a *negative pressure* at the free surface and the
    surface tears. DFSPH has no equation of state -- the density only ever enters as the
    constraint `rho = rho0` and as the error the solver drives down -- and a constraint
    needs the actual density, not an integrated history of it that can drift away from
    the particle positions it is supposed to describe.

    The free-surface deficit is still there and is still real: a surface particle has no
    fluid above it and reads perhaps 70% of rho0. That is why the constant-density solver
    clamps (`density_adv`) and why `density_error()` reports the compression error.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i >= n_fluid:
        return
    xi = x[i]
    total = DTYPE(0.0)
    for j in wp.hash_grid_query(grid, xi, h):
        w = _spline(wp.length(xi - x[j]), h, sigma)
        if j < n_fluid:
            total += mass * w
        else:
            total += psi[j] * w
    rho[i] = total


@wp.kernel
def diffuse_density(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    rho: wp.array(dtype=DTYPE),
    rho_out: wp.array(dtype=DTYPE),
    mass: DTYPE,
    n_fluid: wp.int32,
    h: DTYPE,
    sigma: DTYPE,
    coefficient: DTYPE,
):
    """delta-SPH: one substep of density diffusion, into the field the constraint reads.

        rho_c_i = rho_i + dt * delta * h' * c0 * 2 sum_j (m_j/rho_j) (rho_i - rho_j)
                                                    (x_ij . grad_i W_ij)/(|x_ij|^2 + 0.01 h^2)

    which is Molteni and Colagrossi (2009) eq. (14), the diffusive term Antuono et al.
    (2010) and Marrone et al. (2011) then build the renormalised delta-SPH on.
    `coefficient` is the whole prefactor `dt * delta * h' * c0`, formed on the host so
    this kernel carries no units it has to agree with anything about.

    **The one adaptation, and it is the reason for the `_c`.** Those papers add this term
    to the *continuity equation*, because a weakly compressible scheme integrates
    `Drho/Dt` and the noise accumulates in the integral. This solver has no continuity
    equation: `densities` re-sums `rho_i` from the particle positions every substep, by
    the deliberate choice its docstring argues for. The same operator applied to the same
    field is therefore a one-substep diffusion of the summation density, and where it goes
    is the constant-density residual's `rho_i/rho0 - 1` term -- the one place the density
    enters as a constraint rather than as a normalisation. `factors` and `apply_pressure`
    keep the raw `rho_i`, because there it divides a stiffness and is not being asked to
    be true; `densities()` and `density_error()` keep the raw one too, so **the reported
    density error is the field as summed and not the field as smoothed.** A diffusion
    term that also marked its own homework would make every graded density number here
    better by construction.

    `h'` in the prefactor is the papers' *smoothing length*, which is half this file's
    support radius (`SUPPORT_FACTOR`'s docstring on the two conventions). The host passes
    `self.support / 2`, and `report()` prints the resulting diffusivity in m^2/s so that
    nobody has to work out which `h` a published `delta` was quoted against.

    **Boundary neighbours are skipped.** A boundary particle has no density: `psi_b` is a
    pseudo-mass from Akinci's own normalisation and `rho[b]` is not even allocated. The
    published wall treatments for this term (Marrone's included) extrapolate a wall
    density from the fluid, which is a second scheme with its own ghost-node machinery;
    without it, `rho_j` for a wall is a number somebody would have to invent, and
    inventing it is how a diffusion term becomes a wall attraction. So the operator is
    the interior one, which is the standard practice, and the cost is that it does
    nothing in the outermost fluid layer -- the same layer the free surface already
    truncates.

    Double-buffered: it reads `rho` for every neighbour and writes `rho_out`, which is a
    different array. In place it would be a race with no symptom, since the term is small
    and a neighbour's half-updated density still looks like a density. **`SphWorld.rho_c`
    is only a separate array when this term is on** -- `_commit` does not allocate one
    otherwise -- so a caller launching this kernel by hand with the term off would be
    passing `rho` as both. Measured, because it is what a test did: the sweep becomes
    Gauss-Seidel in grid order and the operator's apparent eigenvalue drifts from 23.19
    down to 20.53 with the diffusion number, which reads exactly like a nonlinearity.

    The sign is the thing to get right and it is checkable rather than arguable:
    `x_ij . grad_i W_ij` is negative everywhere in the support (the kernel falls away
    from the particle), so with `rho_i > rho_j` the whole term is negative and `rho_i`
    comes down towards its neighbours. Backwards, it is an anti-diffusion that sharpens
    the noise it was added to remove, and it would still produce a plausible-looking run.
    `openreynolds/toolbox/sph.py::accelerations` carries the same term with the same
    sign check, and
    `tests/test_live_sph.py::test_density_diffusion_flattens_a_bump_and_spares_a_gradient`
    is the assertion: a bump flattens, a linear ramp does not move.

    ## It does not reduce the force noise on a floating hull, and it is off by default

    This term was written to attack the per-step force noise `CFL`'s docstring measures,
    and **it does not touch it.** `benchmarks/live/force_noise.py`, two independent
    settles at each of two resolutions, per-step RMS of the vertical force:

        delta        free, 2.0 mm       free, 2.5 mm      held, 2.0 mm   held, 2.5 mm
        0 (off)      104.6 / 99.2%     93.9 / 95.8%       23.0%          22.0%
        0.004        109.9 / 110.2%    97.8 / 96.8%        --             --
        0.012        102.3 / 105.8%   100.6 / 102.7%      31.2%          23.0%

    0.012 is 88% of the ceiling `SphWorld.diffusivity` derives and 0.004 is 29% of it, so
    the range is bracketed rather than sampled. **The held rows are the clean ones** -- a
    hull Newton never steps is deterministic here, and both of those numbers reproduced
    bit for bit across the two runs -- and they say it plainly: at 2 mm the diffusion
    makes the fluid's own force noise **worse, 23.0% to 31.2%**, and it raises the worst
    step from 1.41 to 2.44 `m g`.

    Why, stated so it can be argued with rather than as a shrug: delta-SPH removes the
    *accumulated* density error of a scheme that integrates `Drho/Dt`, and this one has
    none to remove -- `densities` re-sums the field from the positions every substep, so
    there is no history for a diffusion to clean. What the operator does instead is blur
    the real short-wavelength structure that is there, and the shortest-wavelength real
    structure in this problem is the pressure in the one or two fluid layers against the
    hull, which is exactly what the wrench integrates. The reported compression error
    agrees: it goes *up* with delta, 0.097% to 0.136% at 2 mm, because the solver is now
    driving a smoothed field to rho0 and leaving the summed one further away.

    It is kept, off, because it is correct, cheap when off (no allocation and no launch),
    checked, and the right tool for a weakly compressible path if one is ever added --
    `toolbox/sph.py` measures 67% to 2.1% pressure error from exactly this term. On this
    solver it is not the remedy, and a knob whose measurement says so is more useful than
    a knob that was never tried.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i >= n_fluid:
        return
    xi = x[i]
    rho_i = rho[i]
    soft = DTYPE(0.01) * h * h
    total = DTYPE(0.0)
    for j in wp.hash_grid_query(grid, xi, h):
        if j >= n_fluid:
            continue
        d = xi - x[j]
        r = wp.length(d)
        if r < DTYPE(_EPS) or r >= h:
            continue
        g = _spline_grad(d, r, h, sigma)
        total += (mass / rho[j]) * DTYPE(2.0) * (rho_i - rho[j]) \
            * wp.dot(d, g) / (wp.dot(d, d) + soft)
    rho_out[i] = rho_i + coefficient * total


@wp.kernel
def factors(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    psi: wp.array(dtype=DTYPE),
    rho: wp.array(dtype=DTYPE),
    alpha: wp.array(dtype=DTYPE),
    mass: DTYPE,
    n_fluid: wp.int32,
    h: DTYPE,
    sigma: DTYPE,
):
    """`alpha_i = rho_i / ( |G_i|^2 + sum_j m_j^2 |grad W_ij|^2 )`, the DFSPH coefficient.

    `G_i = sum_j m_j grad W_ij + sum_b psi_b grad W_ib` is the full gradient sum, fluid
    and boundary together. The derivation, because the published form is stated for a
    fluid with no boundary in it and the boundary term is not a free choice:

    A stiffness `k_i` produces, through the pressure acceleration, a velocity change
    `dv_i = -dt (k_i/rho_i) G_i - dt sum_j m_j (k_j/rho_j) grad W_ij`, where the first
    term collects everything `k_i` drives (its own neighbours and the boundary, which
    cannot move in response) and the second is what the neighbours' own stiffnesses do.
    Differentiating `rho*_i = rho_i + dt sum_j m_j (v_i - v_j).grad W_ij + dt sum_b psi_b
    (v_i - v_b).grad W_ib` with respect to `k_i` alone, and using `grad W_ji = -grad
    W_ij`, gives

        d rho*_i / d k_i = -dt^2 (1/rho_i) ( |G_i|^2 + m_i sum_j m_j |grad W_ij|^2 )

    and with equal particle masses -- which this file guarantees, every fluid particle is
    `rho0 * spacing^3` -- the second term is the `sum_j |m_j grad W_ij|^2` the papers
    write. So `k_i = (rho*_i - rho0) alpha_i / dt^2` inverts it exactly, and the boundary
    appears inside `|G_i|^2` and nowhere else. Dimensionally `k` is m^2/s^2, which is
    `p/rho`, and that is where `pressures()` gets its pascals from.

    The `max(., eps)` floor is for an isolated particle with no neighbours at all: `G_i`
    and the second sum are both zero, alpha is infinite, and one splash droplet would
    otherwise produce a non-finite stiffness and take the whole run with it. Floored
    rather than branched because a particle with a zero gradient sum also has a zero
    pressure force, so the value of alpha there never reaches the answer.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i >= n_fluid:
        return
    xi = x[i]
    total = wp.vec3(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
    squares = DTYPE(0.0)
    for j in wp.hash_grid_query(grid, xi, h):
        d = xi - x[j]
        r = wp.length(d)
        g = _spline_grad(d, r, h, sigma)
        if j < n_fluid:
            total += g * mass
            squares += mass * mass * wp.dot(g, g)
        else:
            total += g * psi[j]
    alpha[i] = rho[i] / wp.max(wp.dot(total, total) + squares, DTYPE(_EPS))


# -- gravity and viscosity -------------------------------------------------------------


@wp.kernel
def nonpressure(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    psi: wp.array(dtype=DTYPE),
    rho: wp.array(dtype=DTYPE),
    body: wp.array(dtype=wp.int32),
    body_com: wp.array(dtype=wp.vec3),
    wrench: wp.array2d(dtype=DTYPE),
    accel_out: wp.array(dtype=wp.vec3),
    mass: DTYPE,
    n_fluid: wp.int32,
    h: DTYPE,
    sigma: DTYPE,
    gravity: wp.vec3,
    nu: DTYPE,
    rho0: DTYPE,
):
    """Gravity plus Monaghan's viscosity, as an acceleration, with the wall reaction.

    **It writes an acceleration and not a velocity, and that is a bug fix with a number
    on it.** This kernel used to end `v[i] = vi + accel * dt`, in the same launch whose
    neighbour loop reads `v[j]`. Every thread therefore raced every neighbour it looked
    at: `v[j]` was the old velocity if thread `j` had not reached its store yet and the
    new one if it had, so `dv = vi - v[j]` carried a spurious `-gravity * dt` for some
    arbitrary subset of the neighbourhood -- arbitrary meaning "whatever the scheduler
    did this time", which is different on the next run of the same code.

    The size of it, which is why this was not a rounding error. Gravity's increment is
    `g dt`, uniform over every particle, so it cancels exactly out of `vi - v[j]` when
    both sides are from the same instant and survives in full when they are not. At the
    sloshing tank's numbers -- `dt` 752 us, so `g dt` = 7.4 mm/s -- that contamination is
    about five times the real neighbour velocity difference (a 32 mm/s mode over a 5 mm
    support), so the viscous term was mostly made of it. Measured on a rigidly
    translating block at a 2.5 mm spacing, where the true viscous acceleration is
    identically zero: **1.07e-5 m/s of spurious velocity in one substep**, 0.011% of the
    block's own speed, injected afresh every substep and in a different pattern each run.
    `tests/test_live_sph.py::test_the_viscous_term_cannot_see_a_velocity_another_thread_wrote`
    is that measurement as an assertion.

    So the acceleration goes to `accel_out` and `SphWorld.step` adds it with a second
    launch -- which is the `advect` kernel again, since `v += a dt` and `x += v dt` are
    the same three lines. The neighbour loop then reads one velocity field, the one at
    the start of the substep, which is also what "explicit viscosity" means.

    What it bought, end to end: `sph_sloshing` run five times on byte-identical code on
    an H100 read a first-mode period of 374.4, 375.8, 417.6, 418.4 and 419.3 ms against
    a 417.09 ms theory with a +/-5% gate, and its two period estimators disagreed by
    1.7% to 26.8% against a 5% gate -- 3/5 runs passing, which is a solver that cannot
    settle anything. With this kernel and the residual reduction below fixed, the same
    five runs return **one** trajectory: 421.376 ms every time, estimators 2.768% apart,
    5/5 passing, and the sampled surface trace identical byte for byte across all five.
    The cost of both fixes together is 6.890 ms a step against 6.756 ms, 2.0%, with the
    two ranges overlapping.

    **Why the standard viscosity and not XSPH.** XSPH is a velocity smoothing,
    `dv_i = -xi sum_j (m_j/rho_j)(v_i - v_j) W_ij`, it is unconditionally stable, and its
    one parameter is dimensionless -- so a run using it has no viscosity in m^2/s and
    therefore no Reynolds number, which is the thing this repository grades everything
    against. Monaghan (1992) in the Cleary-Gingold form,

        a_i = nu sum_j (m_j/rho_j) * 2(d+2) * (v_ij.x_ij)/(|x_ij|^2 + 0.01 h^2) grad W_ij

    takes `nu` from `units.Fluid` directly, so `fluid="water"` means water. Its cost is
    an explicit stability limit, `dt < 0.125 h^2 / nu`, and the reason that cost is zero
    here is arithmetic: water at nu = 1.0034e-6 m^2/s with h = 20 mm gives 49.8 s, and
    the CFL substep at that resolution is 2.61 ms (a 120 mm tank, 5 mm particles, which
    is what `report()` printed for it). Four orders of magnitude of margin,
    so XSPH's unconditional stability buys literally nothing and the dimensionless
    parameter costs the Reynolds number. `SphWorld._check` is where that margin is
    actually checked rather than assumed, because a caller passing a 10,000x eddy
    viscosity would land the other side of it.

    `2(d+2) = 10` in 3D. The `0.01 h^2` is the usual singularity softener for `r -> 0`.

    The boundary term uses `psi_b / rho0` as the neighbour's volume and the body's own
    surface velocity, which is what makes a moving wall drag the fluid and a stationary
    one hold it -- and the equal and opposite force goes onto the body here, in the same
    loop, because a viscous wall force the body does not feel is not a coupling.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i >= n_fluid:
        return
    xi = x[i]
    vi = v[i]
    accel = gravity
    soft = DTYPE(0.01) * h * h
    for j in wp.hash_grid_query(grid, xi, h):
        d = xi - x[j]
        r = wp.length(d)
        if r < DTYPE(_EPS) or r >= h:
            continue
        g = _spline_grad(d, r, h, sigma)
        dv = vi - v[j]
        shear = nu * DTYPE(10.0) * wp.dot(dv, d) / (wp.dot(d, d) + soft)
        if j < n_fluid:
            accel += g * (shear * mass / rho[j])
        else:
            # The same term with the boundary particle's Akinci volume in place of
            # `m_j/rho_j`, and its reaction on the body.
            volume = psi[j] / rho0
            contribution = g * (shear * volume)
            accel += contribution
            b = body[j]
            if b >= 0:
                force = contribution * (-mass)
                arm = x[j] - body_com[b]
                _accumulate(wrench, b, force, wp.cross(arm, force))
    accel_out[i] = accel


# -- the two solvers -------------------------------------------------------------------


@wp.kernel
def residual(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    psi: wp.array(dtype=DTYPE),
    rho: wp.array(dtype=DTYPE),
    adv: wp.array(dtype=DTYPE),
    mass: DTYPE,
    n_fluid: wp.int32,
    h: DTYPE,
    sigma: DTYPE,
    rho0: DTYPE,
    dt: DTYPE,
    use_density: wp.int32,
):
    """Both solvers' residual, as a fraction of the rest density. One kernel, one flag.

        constant density   adv_i = max( (rho_i/rho0 - 1) + dt (Drho_i/Dt)/rho0 , 0 )
        divergence free    adv_i = max(                    dt (Drho_i/Dt)/rho0 , 0 )

    The two differ by one term, and `D rho_i/Dt = sum_j m_j (v_i - v_j).grad W_ij +
    sum_b psi_b (v_i - v_b).grad W_ib` -- the whole neighbour sweep -- is the same in
    both. Two kernels would be two copies of that sweep, and the Akinci boundary term
    inside it is exactly the kind of thing that gets fixed in one copy and not the other.

    **The divergence residual is multiplied by `dt` on purpose.** Raw, it has units of
    1/s, so a tolerance on it would mean something different at every timestep and a
    caller would have to hold two tolerances in their head. `dt Drho/Dt / rho0` is "the
    density error this divergence would produce in one substep", which is the same
    quantity and the same units as the constant-density residual, so `tolerance=` means
    one thing for both solvers. The `1/dt` comes back when the stiffness is formed.

    **Clamped at zero, and the free surface is why.** A surface particle's summation
    density is 20 to 30% below rho0 because there is no fluid above it, so an unclamped
    residual asks the solver to *compress* it back up to rho0 -- a negative pressure,
    which pulls the surface particles inward, and the result is a free surface that
    behaves like a cohesive blob. Clamping makes the constraint one-sided, `rho <= rho0`,
    which is the physically meaningful half: a liquid resists compression and does not
    resist being pulled apart. It is also what puts a zero pressure on the free surface,
    which is the boundary condition the hydrostatic test measures against.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i >= n_fluid:
        return
    xi = x[i]
    vi = v[i]
    rate = DTYPE(0.0)
    for j in wp.hash_grid_query(grid, xi, h):
        d = xi - x[j]
        r = wp.length(d)
        g = _spline_grad(d, r, h, sigma)
        if j < n_fluid:
            rate += mass * wp.dot(vi - v[j], g)
        else:
            rate += psi[j] * wp.dot(vi - v[j], g)
    error = dt * rate / rho0
    if use_density != 0:
        error += rho[i] / rho0 - DTYPE(1.0)
    adv[i] = wp.max(error, DTYPE(0.0))


@wp.kernel
def stiffness(
    adv: wp.array(dtype=DTYPE),
    alpha: wp.array(dtype=DTYPE),
    kappa: wp.array(dtype=DTYPE),
    kappa_sum: wp.array(dtype=DTYPE),
    rho0: DTYPE,
    dt: DTYPE,
    accumulate: wp.int32,
):
    """`kappa_i = adv_i rho0 alpha_i / dt^2`, this iteration's stiffness.

    One kernel of its own rather than folded into `apply_pressure`, because that kernel
    reads `kappa_j` for every neighbour and a value computed per-thread would be read
    before it was written. Both solvers use this: the divergence residual arrives already
    multiplied by `dt` (see `divergence_adv`), so dividing by `dt^2` here gives the
    `Drho/Dt alpha / dt` the paper writes for `k^v`. Same line, both solvers, no branch.

    `kappa_sum` is only accumulated for the constant-density solver (`accumulate`), and
    it is what `pressures()` reports as `kappa_sum * rho`. The divergence solver's
    stiffness is a stabilisation and not a pressure -- it has the units but it is the
    correction that removes a velocity divergence, not a thermodynamic state -- so
    folding it into the reported pressure would make `pressures()` a number with two
    meanings in it.
    """
    i = wp.tid()
    k = adv[i] * rho0 * alpha[i] / (dt * dt)
    kappa[i] = k
    if accumulate != 0:
        kappa_sum[i] = kappa_sum[i] + k


@wp.func
def _accumulate(wrench: wp.array2d(dtype=DTYPE), b: wp.int32, force: wp.vec3,
                torque: wp.vec3):
    """Six atomic adds into the body's row. Force then torque, `lattice.stream`'s order.

    A `wp.func` and not six inline lines because it is called from two kernels and the
    ordering of the six components is exactly the kind of thing that gets transcribed
    wrong the second time.

    **These six are the backend's one remaining run-to-run variation, and it is bounded
    rather than removed.** float32 addition is not associative and an atomic adds in
    whatever order the blocks retire, so the same set of boundary contributions sums to a
    slightly different total each run. The rest of the file is now deterministic -- same
    input, same field, bit for bit -- so that variation can be measured on its own, and
    it was: `sph_force_noise` on an H100 at a shortened settle, twice, gave a mean force
    on a held submerged box of 311.5991608 against 311.5991652 mN, **1.4e-8 relative**,
    and 3.2e-7 on the most derived quantity in that case (the noise reduction ratio).
    Against the 5% the case grades that mean force to, that is six millionths of the
    tolerance.

    It is left in because removing it properly means a fixed summation order over the
    fluid-boundary pairs -- a materialised neighbour list and a segmented reduction, not
    a dtype -- and a float64 accumulator would only move the figure to about 1e-16
    without making it deterministic. The bound above is what a caller may rely on, with
    one caveat that is in `SphWorld.wrench`: it is a bound on the force *out*, measured
    on a body that does not move. A body whose motion is integrated from this force is a
    dynamical system with this as its input noise, and nothing here bounds what that
    grows to over a long coupled run.
    """
    wp.atomic_add(wrench, b, 0, force[0])
    wp.atomic_add(wrench, b, 1, force[1])
    wp.atomic_add(wrench, b, 2, force[2])
    wp.atomic_add(wrench, b, 3, torque[0])
    wp.atomic_add(wrench, b, 4, torque[1])
    wp.atomic_add(wrench, b, 5, torque[2])


@wp.kernel
def apply_pressure(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    psi: wp.array(dtype=DTYPE),
    rho: wp.array(dtype=DTYPE),
    kappa: wp.array(dtype=DTYPE),
    body: wp.array(dtype=wp.int32),
    body_com: wp.array(dtype=wp.vec3),
    wrench: wp.array2d(dtype=DTYPE),
    mass: DTYPE,
    n_fluid: wp.int32,
    h: DTYPE,
    sigma: DTYPE,
    dt: DTYPE,
):
    """One solver iteration's velocity correction, and the force it puts on the bodies.

        dv_i = -dt (k_i/rho_i) sum_b psi_b grad W_ib
               -dt sum_j m_j (k_i/rho_i + k_j/rho_j) grad W_ij

    which is the same `dv` the `factors` derivation differentiated, so the two cannot
    drift apart: if one is changed the other is wrong and the density error says so.

    **The body half, which is the reason this whole file exists.** The boundary term is a
    force on the fluid particle, `F_i = m_i dv_i^bdy / dt = -m_i psi_b (k_i/rho_i) grad
    W_ib`, and Newton's third law puts `-F_i` on the body that boundary particle belongs
    to, applied at the boundary particle's own position:

        F_body += + m_i psi_b (k_i/rho_i) grad W_ib
        T_body += (x_b - com) x F_body

    Check the sign on the case that matters, because getting it backwards has happened
    twice here. Fluid particle `i` above a floor particle `b`: `x_i - x_b` points up, the
    spline's radial derivative is negative, so `grad W_ib` points *down*, from the fluid
    towards the wall. With `k_i > 0` (compressed fluid) the fluid gets `-(+)(down)`, which
    is up -- the floor holds the water up -- and the body gets the opposite, which is
    down. Water pushes down on the floor it sits on. That is the force **on the body**,
    which is `lattice.stream`'s convention and the contract's.

    The torque arm is to the body's centre of mass and not its origin, for the reason
    `world.rotor_reaction` writes down: `body_f` is documented as a wrench about the
    centre of mass, and the two differ on any body whose origin is not its centroid.

    Accumulated across every solver iteration and every substep, then divided by the
    substep count in `wrench()` -- the mean over the interval Newton is about to
    integrate, which is what the lattice backend hands over too.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if i >= n_fluid:
        return
    xi = x[i]
    ki = kappa[i] / rho[i]
    delta = wp.vec3(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
    for j in wp.hash_grid_query(grid, xi, h):
        d = xi - x[j]
        r = wp.length(d)
        if r < DTYPE(_EPS) or r >= h:
            continue
        g = _spline_grad(d, r, h, sigma)
        if j < n_fluid:
            delta -= g * (mass * (ki + kappa[j] / rho[j]))
        else:
            step = g * (psi[j] * ki)
            delta -= step
            b = body[j]
            if b >= 0:
                force = step * mass
                arm = x[j] - body_com[b]
                _accumulate(wrench, b, force, wp.cross(arm, force))
    v[i] = v[i] + delta * dt


@wp.kernel
def advect(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), n_fluid: wp.int32,
           dt: DTYPE):
    """`x += dt v`, fluid only. Semi-implicit Euler: the velocity is already the new one.

    Symplectic in the sense that matters -- it does not pump energy into a sloshing
    oscillation the way forward Euler does, which is the same argument `toolbox/sph.py`
    makes for the same integrator. Boundary particles are posed from their bodies and are
    never advected.

    **Launched twice a substep, on two different pairs of arrays**, because `y += dt dy`
    is the same three lines whether `y` is a position and `dy` a velocity or `y` is a
    velocity and `dy` the acceleration `nonpressure` just wrote. The second use is what
    keeps that kernel's neighbour loop off the field it is updating; see its docstring.
    A second kernel with the same body and different parameter names would be the same
    three lines written twice.
    """
    i = wp.tid()
    if i >= n_fluid:
        return
    x[i] = x[i] + v[i] * dt


@wp.func
def _reflect(p: DTYPE, u: DTYPE, lo: DTYPE, hi: DTYPE, damping: DTYPE) -> wp.vec2:
    """One axis of the domain clamp: the clamped coordinate and the kept velocity.

    A `wp.vec2` return rather than three copies of the branch in the kernel, and
    rather than writing `p[a]` inside a `range(3)` loop -- a dynamic index write into a
    local vector is the one Warp construct `lattice.stream` is also written around.
    """
    if p < lo:
        return wp.vec2(lo, wp.where(u < DTYPE(0.0), -u * damping, u))
    if p > hi:
        return wp.vec2(hi, wp.where(u > DTYPE(0.0), -u * damping, u))
    return wp.vec2(p, u)


@wp.kernel
def clamp_to_domain(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    low: wp.vec3,
    high: wp.vec3,
    caught: wp.array(dtype=wp.int32),
    n_fluid: wp.int32,
    damping: DTYPE,
):
    """A last-resort reflection at the domain box, so one escaped particle is not a crash.

    This is **not** the boundary condition -- the walls are Akinci boundary particles and
    they are what holds the fluid in. This catches the particle that got through anyway:
    a splash in the corner where two walls meet and the sampling is thinnest, or a
    particle that took one bad step during a transient. Without it that particle leaves
    the region the hash grid was sized for, its neighbour query returns nothing, its
    alpha hits the floor in `factors`, and from then on it is a permanent source of
    garbage in every reduction.

    `caught` counts how often it fires, and `report()` prints it, because a wall that
    leaks every step is a sampling problem and not something a clamp should be quietly
    papering over.
    """
    i = wp.tid()
    if i >= n_fluid:
        return
    p = x[i]
    u = v[i]
    ax = _reflect(p[0], u[0], low[0], high[0], damping)
    ay = _reflect(p[1], u[1], low[1], high[1], damping)
    az = _reflect(p[2], u[2], low[2], high[2], damping)
    q = wp.vec3(ax[0], ay[0], az[0])
    if q[0] != p[0] or q[1] != p[1] or q[2] != p[2]:
        wp.atomic_add(caught, 0, 1)
    x[i] = q
    v[i] = wp.vec3(ax[1], ay[1], az[1])


# -- host-side signed distances --------------------------------------------------------
#
# The same five SDFs as `lattice.py`'s `_sdf_*`, in numpy and in metres instead of Warp
# and lattice units, and in the same local frames (capsule and cylinder along local z,
# plane the local z = 0 half-space with +z outward -- Newton's conventions, which
# `couple.py` reads off the model rather than assuming).
#
# Duplicated rather than shared because `lattice.py`'s are `@wp.func` and only callable
# from a kernel, and sampling a shape happens once at setup on a host array of candidate
# points. `tests/test_live_sph.py::test_the_host_sdfs_agree_with_the_lattice_kernels`
# holds the two sets equal so they cannot drift.


def _sdf(kind: int, p: np.ndarray, scale: Sequence[float]) -> np.ndarray:
    s = np.asarray(scale, dtype=np.float64)
    if kind == L.SHAPE_SPHERE:
        return np.linalg.norm(p, axis=1) - s[0]
    if kind == L.SHAPE_BOX:
        q = np.abs(p) - s[None, :3]
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(q.max(axis=1), 0.0)
        return outside + inside
    if kind == L.SHAPE_CAPSULE:
        dz = p[:, 2] - np.clip(p[:, 2], -s[1], s[1])
        return np.linalg.norm(np.stack([p[:, 0], p[:, 1], dz], axis=1), axis=1) - s[0]
    if kind == L.SHAPE_CYLINDER:
        dr = np.hypot(p[:, 0], p[:, 1]) - s[0]
        dz = np.abs(p[:, 2]) - s[1]
        outside = np.hypot(np.maximum(dr, 0.0), np.maximum(dz, 0.0))
        inside = np.minimum(np.maximum(dr, dz), 0.0)
        return outside + inside
    if kind == L.SHAPE_PLANE:
        return p[:, 2].copy()
    raise UnphysicalScaling(
        "the SPH backend samples boundary particles from a shape's signed distance, and "
        "it has no mesh sampler yet: a mesh shape would need either a wp.Mesh query over "
        "the candidate points or a surface-triangle sampler. Build the hull from "
        "spheres, boxes, capsules, cylinders or a plane for now, or add the mesh path to "
        "`sph._sdf`"
    )


def _quat_matrix(q: Sequence[float]) -> np.ndarray:
    """`(x, y, z, w)` to a rotation matrix. `world._quat_matrix`, same four lines.

    Copied rather than imported, which is the precedent `couple.py` set for the same
    four lines and for the same reason it gives: a shared private helper across three
    modules is a dependency for sixteen multiplications. The risk of a copy is that one
    of them changes; `tests/test_live_sph.py::test_the_quaternion_maths_matches_world`
    is what makes that an error instead of a drift.
    """
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class SphWorld:
    """A DFSPH fluid with a real free surface, coupled to rigid bodies.

        world = SphWorld(domain=Domain((0, 0, 0), (0.2, 0.2, 0.2), faces="wall"),
                         particle_radius=0.005, fluid="water")
        world.boundary_from_domain()
        world.fill_below(0.1)
        for _ in range(200):
            world.step()
        print(world.density_error(), world.report())

    `docs/sph-contract.md` is the interface. Everything is SI.
    """

    def __init__(
        self,
        domain: Any = None,
        *,
        particle_radius: float | None = None,
        fluid: str | Any = "water",
        gravity: Sequence[float] = (0.0, 0.0, -9.81),
        viscosity: float | None = None,
        surface_tension: float = 0.0,
        device: str | None = None,
        solver: str = "dfsph",
        dt: float | None = None,
        substeps: int | None = None,
        tolerance: float = 1.0e-3,
        max_iterations: int = 100,
        boundary_layers: int = 2,
        check_every: int = 1,
        density_diffusion: float = 0.0,
        wrench_filter: float = 0.0,
        cfl: float = CFL,
    ) -> None:
        """
        `particle_radius` is the sizing knob, the way `dx` is for the lattice: the
        spacing is `2 * particle_radius`, the kernel support is `4 * particle_radius`
        (`SUPPORT_FACTOR`), and a fluid particle's mass is `rho0 * spacing^3`. There is
        no default for the same reason `FluidWorld` has none for the resolution -- the
        whole cost of the run is in it.

        `dt` is the coupling step in seconds. Left out, it is one CFL substep. Given, the
        substep count is derived from the CFL limit unless `substeps` says otherwise, and
        a `substeps` that breaks the limit is refused rather than run.

        `viscosity` is the kinematic viscosity in m^2/s and defaults to the fluid's own.
        Pass a larger one for an eddy viscosity; `_check` holds the explicit scheme's
        stability limit over it.

        `tolerance` is the mean residual both solvers iterate down to, as a fraction of
        the rest density. `max_iterations` is the cap on each. `check_every` is how often
        the residual is read back to the host; see the module docstring on what that read
        costs and why it exists.

        `check_every=0` never reads it, which means **`max_iterations` becomes the actual
        iteration count, not a cap**, and the default of 100 is a cap and not a sensible
        count. A caller turning the check off has to set `max_iterations` to something
        it has measured -- 6 covers this tank's 5.0 mean and 8 worst case -- and a run
        that then needed more would silently return a less converged field. That is the
        trade the GPU path is offered, not a mode to drift into.

        `surface_tension` must be 0.0 -- see `_check`.

        `density_diffusion` is delta-SPH's `delta`, and **0.0 is off, which is the
        default and is the control every measurement of it is against.** See
        `diffuse_density` for the term and `_check` for the stability limit it is held
        to, which at a projection timestep is a real limit and not a formality.

        `wrench_filter` is a one-pole low-pass on the wrench `wrench()` **reports**, as
        an EMA coefficient in `(0, 1]`; 0.0 is off and is the default. It changes nothing
        the solver does -- see `wrench`.
        """
        self.wp = require()

        if domain is None:
            raise UnphysicalScaling("an SphWorld needs a Domain: the box it solves in")
        if particle_radius is None or float(particle_radius) <= 0.0:
            raise UnphysicalScaling(
                "an SphWorld needs `particle_radius=` in metres: particle spacing is "
                "twice it and the kernel support four times it. There is no default, "
                "because the whole cost of the run is in it"
            )
        self.solver = str(solver).strip().lower()
        if self.solver not in SOLVERS:
            raise UnphysicalScaling(
                f"no SPH solver called {self.solver!r}: {', '.join(SOLVERS)}. "
                f"A weakly compressible path is not shipped -- see `sph.SOLVERS`"
            )

        self.domain = domain
        self.fluid = resolve_fluid(fluid)
        self.device = device or str(wp.get_preferred_device())
        self.radius = float(particle_radius)
        self.spacing = 2.0 * self.radius
        self.support = SUPPORT_FACTOR * self.radius
        self.sigma = 8.0 / (math.pi * self.support ** 3)
        self.rho0 = float(self.fluid.rho)
        self.mass = self.rho0 * self.spacing ** 3
        self.gravity = tuple(float(v) for v in gravity)
        self.nu = float(self.fluid.nu if viscosity is None else viscosity)
        self.surface_tension = float(surface_tension)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.boundary_layers = max(1, int(boundary_layers))
        self.check_every = max(0, int(check_every))
        self.density_diffusion = float(density_diffusion)
        # **A knob because the floating cases need `CFL/4` and fluid-only cases do not.**
        # `CFL`'s own docstring carries the measurement: a quarter step takes the per-step
        # RMS force on a freely floating hull from 104.6% of its weight to 1.8-2.6%, and the
        # transition is a cliff between `CFL/2` and `CFL/3` rather than a slope -- which is
        # why halving it had been tried and had done nothing. It costs 1.9-2.5x wall clock,
        # not the 4x the step implies, because the solver's iteration count falls from about
        # nine to exactly two. The default is unchanged, because `CFL` is also the fluid's own
        # stability number and a fluid-only case has no wrench to be noisy.
        self.cfl = float(cfl)
        self.wrench_filter = float(wrench_filter)

        self.dt, self.substeps, self._peak_estimate = self._substep(dt, substeps)
        self.c0 = self._peak_estimate / math.sqrt(self.tolerance)
        """The weakly compressible sound speed this run is *equivalent* to, in m/s.

        **DFSPH has no speed of sound.** That is the point of it: the pressure is a
        Lagrange multiplier on the density constraint and not a function of the density,
        there is no equation of state, and nothing in this file reads this number. It is
        here because `docs/sph-contract.md` needs the graded density error and the graded
        pressure error to be connectable, and `dp = c0^2 drho` is the only thing that
        connects them.

        The equivalence, stated so it can be argued with: a weakly compressible scheme at
        peak speed `u` and sound speed `c0` carries a standing density error of order
        `(u/c0)^2` -- that is the whole reason WCSPH codes pick `c0 = 10 u`. Inverting it
        at the density error this solver is actually held to gives `c0 = u /
        sqrt(tolerance)`, so a DFSPH run at `tolerance=1e-3` is as incompressible as a
        WCSPH run at 31 times its peak speed. Both inputs are the run's own: `u` is
        `_substep`'s peak estimate and is reported as an estimate, and `tolerance` is the
        knob the caller set.

        What a case can do with it: measure `density_error()` and the pressure field, and
        check that `dp` and `c0^2 rho0 d(rho/rho0)` agree. What it must not do is treat
        this as a measured property of the solver -- it is a derived equivalence, and the
        measured quantity is the density error itself.
        """

        self._check()

        self.coupling_style = "particles"
        """How `couple.py` must treat this world's arrays. `docs/sph-contract.md`.

        The lattice backend solves in index space, so `couple.py` divides every shape
        scale by `dx` on the way in and multiplies the accumulated wrench by
        `rho dx^4/dt^2` on the way out. This backend solves in metres, seconds and
        kilograms, so all three conversions are the identity -- and an `SphWorld` has no
        `dx` at all, which is how the coupling found out: it raised
        `AttributeError: no attribute 'dx'` from `_static_shape_arrays` and took all four
        floating-body cases with it. A string rather than a `hasattr(world, "dx")` probe
        because a missing attribute is an accident and a declared style is a promise.
        """


        # -- the particles, on the host until the first step ------------------------
        #
        # Nothing is allocated here: `fill_box` and `add_shape` are called after
        # construction, so the array sizes are not known yet. `_commit` builds the
        # device arrays once, on the first `step()` or the first read.
        self._fluid_x: list[np.ndarray] = []
        self._fluid_v: list[np.ndarray] = []
        self._bdy_x: list[np.ndarray] = []
        self._bdy_body: list[np.ndarray] = []
        self._shapes: list[dict[str, Any]] = []
        self._sampled_volume: dict[int, float] = {}

        # The body table is `FluidWorld`'s, field for field, so that a case written
        # against one reads against the other and `couple.py` has one shape to talk to.
        self._poses: list[tuple[float, ...]] = []
        self._vels: list[tuple[float, ...]] = []
        self._coms: list[tuple[float, ...]] = []
        self._labels: list[str] = []
        self._device_bodies_dirty = True
        self.body_pose = None
        self.body_vel = None
        self.body_com = None

        self.n = 0
        self.n_fluid = 0
        self.n_boundary = 0
        self.steps = 0
        self.substeps_taken = 0
        self.seconds = 0.0
        self.time = 0.0
        """Simulated seconds advanced so far, which is not `seconds` (wall clock).

        `sph_cases.py::_elapsed` reads this if it is there and falls back to
        `steps * dt` if it is not, and the fallback is wrong as soon as a caller passes
        `step(substeps=...)` on some steps and not others -- not because the step length
        changes (it does not; `substeps` subdivides `dt`) but because nothing then forces
        the two to agree. Keeping the clock here means there is one of them.
        """
        self.iterations_density = 0
        self.iterations_divergence = 0
        self.solves = 0
        self.clamped = 0
        self.one_way = False
        """`couple.py` reads this. SPH has no one-way debug mode of its own; the flag is
        carried so the coupling does not have to know which backend it is holding."""
        self._committed = False
        self._grid = None
        self._wrench = None
        self._wrench_mean = None
        """The filtered wrench, `(n, 6)` in SI, or `None` until the first filtered step.

        Host-side and float64, because it is the number a caller reads and not a number
        the solver uses -- see `wrench`.
        """
        self._wall_faces: list[str] = []
        self.x = None
        self.v = None
        self.duplicates = 0

    # -- sizing ---------------------------------------------------------------------

    def _substep(self, dt: float | None, substeps: int | None) -> tuple[float, int, float]:
        """The substep from the CFL condition, and the coupling step around it.

        The CFL limit needs a peak velocity and measuring one needs a host read of the
        whole velocity field every step, so it is **estimated** and the estimate is
        reported rather than buried: the free-fall speed over the domain's extent along
        gravity, `sqrt(2 |g| H)`, which is the fastest anything in a gravity-driven case
        gets. For a 200 mm tank that is 1.98 m/s, and at a 10 mm spacing the substep is
        0.4 * 0.010 / 1.98 = 2.0 ms.

        An estimate and not a measurement, so it can be wrong: a case that injects fluid
        at 10 m/s into a shallow box has a peak this does not know about. The knob for
        that is `dt` with `substeps`, and the refusal below is what makes a wrong
        `substeps` an error message instead of a slowly diverging run.
        """
        g = math.sqrt(sum(v * v for v in self.gravity))
        if g > 0.0:
            # The drop available along gravity, which for the usual `(0, 0, -9.81)` is
            # the domain's height and for a tilted gravity is its projection.
            along = sum(abs(self.gravity[a]) / g * self.domain.size[a] for a in range(3))
            peak = math.sqrt(2.0 * g * along)
        else:
            # No gravity, so nothing falls and the estimate above is zero -- but a zero
            # peak makes the CFL limit infinite and `c0` zero, and a case that divides by
            # either gets a number rather than an error. A spacing a millisecond is the
            # speed at which this discretisation stops being able to follow its own
            # neighbourhood, which is the same thing the CFL limit says and is the honest
            # floor for a run whose speeds nothing has told us about.
            peak = self.cfl * self.spacing / 1.0e-3
        limit = self.cfl * self.spacing / peak
        if dt is None:
            return float(limit), 1, peak
        step = float(dt)
        if step <= 0.0:
            raise UnphysicalScaling(f"a coupling step is positive; got dt {step}")
        if substeps is None:
            return step, max(1, int(math.ceil(step / limit))), peak
        count = int(substeps)
        if count < 1:
            raise UnphysicalScaling(f"at least one substep; got substeps {count}")
        if step / count > limit * 1.001:
            raise UnphysicalScaling(
                f"{count} substeps of dt {step:g} s is {step / count:.4g} s each, past "
                f"the CFL limit {limit:.4g} s for a {self.spacing * 1e3:.3g} mm spacing "
                f"at an estimated peak of {peak:.3g} m/s. Raise `substeps` to at least "
                f"{int(math.ceil(step / limit))}, lower `dt`, or coarsen "
                f"`particle_radius`"
            )
        return step, count, peak

    @property
    def diffusivity(self) -> float:
        """delta-SPH's diffusivity `delta h' c0`, in m^2/s. Zero when the term is off.

        `h'` is the papers' smoothing length, half this file's support radius, and `c0`
        is the equivalent sound speed above -- the only speed a projection scheme has to
        offer a term whose published prefactor wants an acoustic one. Both conventions
        are a place to be wrong by a factor of two, so the product is computed once here,
        used by `_check`, `_prepare` and `report`, and printed in m^2/s where a reader can
        compare it against the physical `nu` in the line above it.

        **The largest usable delta is `0.429 sqrt(tolerance)` and nothing else**, which
        is worth having in closed form because it is the whole reason a published delta
        cannot be copied in here. At the default substep -- the CFL one, `dt = CFL
        spacing/u` with `spacing = h/2` -- with `c0 = u/sqrt(tolerance)` and the measured
        `lambda_max = 23.2963/h^2` (`DIFFUSION_EIGENVALUE`):

            delta h' c0 dt lambda_max <= 1
              =>  delta (h/2)(u/sqrt(tolerance))(0.2 h/u)(23.2963/h^2) <= 1
              =>  delta <= sqrt(tolerance) / 2.3296  =  0.429 sqrt(tolerance)

        The resolution cancels, the fluid cancels, the peak speed cancels. At the default
        `tolerance=1e-3` that is **0.01357**, so Molteni and Colagrossi's 0.1 and
        `toolbox/sph.py`'s 0.3 are both out of reach here by an order of magnitude -- and
        tightening the tolerance makes it *worse*: `tolerance=1e-4` allows only 0.00429,
        because a tighter tolerance is a higher equivalent sound speed and this term's
        prefactor is proportional to it. **So delta-SPH and a tighter DFSPH tolerance are
        not independent knobs; they trade against each other.** `_check` refuses past the
        limit and prints the number.

        What is NOT small about that delta: at the limit the sharpest density mode is
        annihilated in one substep, by construction. The ceiling is low because the
        operator is strong, not because it is weak.
        """
        return self.density_diffusion * (self.support / 2.0) * self.c0

    def _check(self) -> None:
        """The refusals, each naming the knob that moves it. `units.UnphysicalScaling`'s
        contract, and the same four knobs: resolution, timestep, substeps, viscosity."""
        if self.surface_tension != 0.0:
            raise UnphysicalScaling(
                f"surface_tension is {self.surface_tension:g} and this backend has no "
                f"surface tension model, so it would be silently ignored. Set "
                f"surface_tension=0.0. The model this would want is Akinci, Akinci and "
                f"Teschner (2013) -- a cohesion kernel plus a curvature term -- and it "
                f"is not written; `report()` says 'off' so no run claims it"
            )
        sub = self.dt / self.substeps
        limit = 0.125 * self.support ** 2 / self.nu if self.nu > 0.0 else float("inf")
        if sub > limit:
            raise UnphysicalScaling(
                f"the viscosity is explicit, so the substep is bounded by "
                f"0.125 h^2/nu = {limit:.4g} s, and {self.substeps} substeps of "
                f"dt {self.dt:g} s is {sub:.4g} s each. nu is {self.nu:g} m2/s, which is "
                f"{self.nu / self.fluid.nu:.0f}x {self.fluid.name}'s own. Raise "
                f"`substeps`, lower `viscosity`, or coarsen `particle_radius` (the limit "
                f"goes as h^2)"
            )
        if not 0.0 <= self.wrench_filter <= 1.0:
            raise UnphysicalScaling(
                f"`wrench_filter` is an EMA coefficient in [0, 1] and is "
                f"{self.wrench_filter:g}. 0.0 is off (the default), 1.0 is a filter that "
                f"passes the raw wrench through, and anything outside that is not a "
                f"low-pass -- above 1 it overshoots and amplifies, below 0 it alternates "
                f"sign. See `SphWorld.wrench`"
            )
        if self.density_diffusion < 0.0:
            raise UnphysicalScaling(
                f"`density_diffusion` is delta-SPH's delta and is {self.density_diffusion:g}. "
                f"Negative, the same term is an ANTI-diffusion: it sharpens the density "
                f"noise it exists to remove, and the run still looks plausible. 0.0 is off"
            )
        if self.density_diffusion > 0.0:
            # The explicit diffusion limit, from the operator's own largest eigenvalue
            # rather than by analogy with the `0.125 h^2/nu` viscous one above --
            # `DIFFUSION_EIGENVALUE` has the measurement and why the analogy is 3x wrong.
            diffusivity = self.diffusivity
            ceiling = self.support ** 2 / (DIFFUSION_EIGENVALUE * diffusivity)
            if sub > ceiling:
                raise UnphysicalScaling(
                    f"delta-SPH at delta {self.density_diffusion:g} is a diffusivity of "
                    f"delta h' c0 = {diffusivity:.4g} m2/s (h' {self.support / 2 * 1e3:.4g} mm, "
                    f"the papers' smoothing length; c0 {self.c0:.4g} m/s, this run's "
                    f"equivalent sound speed), so one substep multiplies the sharpest "
                    f"density mode this lattice carries by "
                    f"1 - {DIFFUSION_EIGENVALUE:g} D dt/h^2 = "
                    f"{1.0 - DIFFUSION_EIGENVALUE * diffusivity * sub / self.support ** 2:+.4g}. "
                    f"Below -1 that mode GROWS and above 0 it decays monotonically, "
                    f"which is the only behaviour a diffusion may have: the limit is a "
                    f"substep of {ceiling:.4g} s and this one is {sub:.4g} s. Lower "
                    f"`density_diffusion` to at most "
                    f"{self.density_diffusion * ceiling / sub:.4g}, raise `substeps`, or "
                    f"lower `dt`.\n"
                    f"This limit is why a published delta cannot be copied across: "
                    f"Molteni and Colagrossi's 0.1 is quoted against an ACOUSTIC "
                    f"timestep, `dt ~ 0.25 h'/c0` = {0.25 * self.support / 2 / self.c0:.4g} s "
                    f"here, which is {sub / (0.25 * self.support / 2 / self.c0):.3g}x "
                    f"smaller than this projection scheme's substep. The diffusion "
                    f"NUMBER is what transfers, not delta"
                )
        if self.spacing > min(self.domain.size) / 4.0:
            raise UnphysicalScaling(
                f"a particle spacing of {self.spacing * 1e3:.3g} mm in a domain "
                f"{min(self.domain.size) * 1e3:.3g} mm across its shortest axis is "
                f"fewer than four particles across it, and a kernel support is two "
                f"spacings, so no particle would have a full neighbourhood anywhere. "
                f"Lower `particle_radius`"
            )

    # -- bodies ---------------------------------------------------------------------

    def add_body(self, position: Sequence[float] = (0.0, 0.0, 0.0),
                 orientation: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
                 *, com: Sequence[float] | None = None,
                 velocity: Sequence[float] | None = None, label: str = "") -> int:
        """A body the world owns. `FluidWorld.add_body`, same signature, same meaning."""
        index = len(self._poses)
        self._poses.append(tuple(float(v) for v in position)
                           + tuple(float(v) for v in orientation))
        self._coms.append(tuple(float(v) for v in (com if com is not None else position)))
        self._vels.append(tuple(float(v) for v in (velocity or (0.0,) * 6)))
        self._labels.append(label or f"body_{index}")
        self._device_bodies_dirty = True
        return index

    def set_body_state(self, body: int, position: Sequence[float] | None = None,
                       orientation: Sequence[float] | None = None,
                       velocity: Sequence[float] | None = None,
                       com: Sequence[float] | None = None) -> None:
        """Move a world-owned body. `FluidWorld.set_body_state`, same signature."""
        pose = list(self._poses[body])
        if position is not None:
            pose[:3] = [float(v) for v in position]
        if orientation is not None:
            pose[3:] = [float(v) for v in orientation]
        self._poses[body] = tuple(pose)
        if velocity is not None:
            self._vels[body] = tuple(float(v) for v in velocity)
        if com is not None:
            self._coms[body] = tuple(float(v) for v in com)
        self._device_bodies_dirty = True

    # -- filling --------------------------------------------------------------------

    def _lattice(self, low: Sequence[float], high: Sequence[float]) -> np.ndarray:
        """Candidate sites on **one** global lattice, `origin + (i + 1/2) * spacing`.

        One lattice for the fluid and for every boundary shell, which is not tidiness:
        if the fluid were filled on its own grid and a hull sampled on the hull's grid,
        the nearest fluid-boundary pair would land anywhere between touching and a full
        spacing apart, and a pair at a tenth of a spacing is a stiffness spike -- the
        pressure solve sees a density of several rho0 at one particle and spends its
        whole iteration budget there. On one lattice the minimum separation is exactly
        one spacing, everywhere, by construction.

        The cost is that a shape's surface does not generally fall on a lattice plane, so
        the sampled body is up to half a spacing smaller or larger than the geometric
        one. That is a real error and it is why `volume(body)` reports the sampled volume
        rather than the analytic one -- the same reason `FluidWorld.solid_cells` exists.
        """
        origin = np.asarray(self.domain.origin, dtype=np.float64)
        axes = []
        for a in range(3):
            lo = (float(low[a]) - origin[a]) / self.spacing - 0.5
            hi = (float(high[a]) - origin[a]) / self.spacing - 0.5
            axes.append(np.arange(int(math.ceil(lo - 1e-9)),
                                  int(math.floor(hi + 1e-9)) + 1))
        if any(len(a) == 0 for a in axes):
            return np.zeros((0, 3))
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        return origin + (grid + 0.5) * self.spacing

    def fill_box(self, low: Sequence[float], high: Sequence[float], *,
                 velocity: Sequence[float] = (0.0, 0.0, 0.0)) -> int:
        """Fluid particles on the global lattice inside an axis-aligned box, in metres.

        Sites inside any shape added so far are dropped. Added *after* the shapes, then,
        and the order is not a convention this file can enforce -- a `fill_box` before an
        `add_shape` leaves fluid inside the hull, which shows up immediately as a
        six-figure density error. `report()` prints the counts so the mistake is visible
        in the first line of output rather than in the physics.
        """
        points = self._lattice(low, high)
        if len(points):
            keep = np.ones(len(points), dtype=bool)
            for shape in self._shapes:
                keep &= self._shape_sdf(shape, points) > 0.0
            points = points[keep]
        if len(points):
            self._fluid_x.append(points)
            self._fluid_v.append(np.tile(np.asarray(velocity, dtype=np.float64),
                                         (len(points), 1)))
        self._count()
        return len(points)

    def fill_below(self, level: float, *, axis: int = 2, amplitude: float = 0.0,
                   mode: int = 1, along: int | None = None) -> int:
        """Fluid up to a level along `axis`: the still-water start, in metres.

        With `amplitude`, the level is a standing cosine instead of a plane,

            level(s) = level + amplitude * cos(mode * pi * (s - s0) / L)

        where `s` is the coordinate along `along` (the first horizontal axis by default)
        and `L` the domain's extent on it. That is the first sloshing mode of a closed
        tank at `mode=1`: the free surface is horizontal at the two walls' midpoint and
        at its extremes against the walls, which is the shape with zero normal velocity
        at both ends and therefore the one that oscillates rather than breaking.

        It exists because the sloshing case was releasing the first mode as a staircase of
        abutting `fill_box` slabs, which depends on two things this file does not promise:
        that the slabs' lattices line up (they do -- `_lattice` is global -- but that is
        an implementation detail the contract does not state) and that a sampled staircase
        is a cosine (it is not; it is a cosine plus every harmonic the step edges carry).
        One expression here is cheaper than a promise about slabs.
        """
        axis = int(axis)
        if along is None:
            along = next(a for a in range(3) if a != axis)
        low = list(self.domain.origin)
        high = [low[a] + self.domain.size[a] for a in range(3)]
        if amplitude == 0.0:
            high[axis] = float(level)
            return self.fill_box(low, high)
        # Sample the whole box and keep what is under the surface, rather than filling
        # per column: `fill_box` already drops the sites inside a shape, and doing the
        # level test on the same candidate array keeps the single global lattice.
        high[axis] = float(level) + abs(float(amplitude))
        points = self._lattice(low, high)
        if len(points):
            span = self.domain.size[int(along)]
            phase = (points[:, int(along)] - self.domain.origin[int(along)]) / span
            surface = level + amplitude * np.cos(int(mode) * math.pi * phase)
            points = points[points[:, axis] < surface]
        if not len(points):
            return 0
        for shape in self._shapes:
            points = points[self._shape_sdf(shape, points) > 0.0]
        self._fluid_x.append(points)
        self._fluid_v.append(np.zeros((len(points), 3)))
        self._count()
        return len(points)

    def _shape_sdf(self, shape: dict[str, Any], points: np.ndarray) -> np.ndarray:
        """A shape's signed distance at world points, in metres, through its body pose."""
        local_p = np.asarray(shape["position"], dtype=np.float64)
        local_q = np.asarray(shape["orientation"], dtype=np.float64)
        rotation = _quat_matrix(local_q)
        if shape["body"] >= 0:
            pose = self._poses[shape["body"]]
            body_r = _quat_matrix(pose[3:])
            world_p = np.asarray(pose[:3]) + body_r @ local_p
            world_r = body_r @ rotation
        else:
            world_p, world_r = local_p, rotation
        local = (points - world_p) @ world_r
        return _sdf(shape["kind"], local, shape["scale"])

    def add_shape(self, kind: str | int, *, body: int = -1,
                  position: Sequence[float] = (0.0, 0.0, 0.0),
                  orientation: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
                  scale: Sequence[float] = (1.0, 1.0, 1.0),
                  mesh: Any = None) -> int:
        """A solid, in metres, sampled into Akinci boundary particles.

        `FluidWorld.add_shape`'s signature and `lattice.py`'s scale convention exactly:
        sphere `(radius, -, -)`, box `(hx, hy, hz)`, capsule and cylinder
        `(radius, half_height, -)` along local z, plane the local z = 0 half-space.

        The shell is `boundary_layers` lattice layers inside the surface, which at the
        default of 2 is one full kernel support thick -- so a fluid particle touching the
        surface sees a complete half-space of boundary and its Akinci density sum is not
        missing anything. Measured on the 280 mm still tank, averaged over 20 coupling
        steps after 100 of settling:

            layers   boundary particles   F_z/weight   max density error   lowest fluid z
               1             706            0.9960          0.691%            6.77 mm
               2            1708            0.9989          0.646%            4.51 mm
               3            3054            0.9986          0.605%            4.24 mm

        so one layer does **not** leak -- the Akinci volume correction is doing its job,
        and the 0.4% it loses on the load is real but small. What it costs is the last
        column: the first row of fluid starts at 10 mm (half a 20 mm spacing above the
        floor) and settles to 6.77 mm with one layer against 4.51 with two, so a single
        layer lets the fluid press 2.3 mm further into the wall before the boundary
        density sum stops it. Three layers buy 0.27 mm more and 79% more boundary
        particles, which at 7% of a substep each (`boundary_volumes`) is the most
        expensive 0.27 mm available. Two is the default on those numbers, and one is a
        legitimate choice for a case that is short of memory and does not grade a wall
        force.

        A `mesh` shape is refused with a message naming what is missing (`sph._sdf`).
        """
        code = kind if isinstance(kind, int) else {
            v: k for k, v in L.SHAPE_NAMES.items()
        }[str(kind).strip().lower()]
        shape = {
            "kind": int(code), "body": int(body),
            "position": tuple(float(v) for v in position),
            "orientation": tuple(float(v) for v in orientation),
            "scale": tuple(float(v) for v in scale),
            "mesh": mesh,
        }
        self._shapes.append(shape)
        self._sample(shape)
        return len(self._shapes) - 1

    def _sample(self, shape: dict[str, Any]) -> None:
        """Lattice sites inside the shape: the shell becomes particles, all of them count
        towards the body's sampled volume."""
        low = list(self.domain.origin)
        high = [low[a] + self.domain.size[a] for a in range(3)]
        points = self._lattice(low, high)
        if not len(points):
            return
        distance = self._shape_sdf(shape, points)
        inside = distance <= 0.0
        body = shape["body"]
        if body >= 0:
            volume = float(inside.sum()) * self.spacing ** 3
            self._sampled_volume[body] = self._sampled_volume.get(body, 0.0) + volume
        shell = inside & (distance >= -self.boundary_layers * self.spacing)
        if not shell.any():
            raise UnphysicalScaling(
                f"a {L.SHAPE_NAMES[shape['kind']]} of scale "
                f"{shape['scale']} sampled to no boundary particles at a "
                f"{self.spacing * 1e3:.3g} mm spacing: it is thinner than one particle, "
                f"or it is outside the domain. Lower `particle_radius`, or check the "
                f"shape's position against the domain"
            )
        self._bdy_x.append(points[shell])
        self._bdy_body.append(np.full(int(shell.sum()), body, dtype=np.int32))
        self._count()

    def boundary_from_domain(self, *, body: int = -1, faces: Sequence[str] | None = None
                             ) -> int:
        """Boundary particles on the domain's `wall` faces (or on the named faces).

        Six boxes rather than one hollow shell, because the shell would have to be
        sampled as a difference of SDFs and the faces are independently chosen: a flume
        is walls on four faces and open on two. A face that is not a wall gets nothing
        and the fluid leaves through it, which is what an open face means here --
        `clamp_to_domain` catches it on the way out so the hash grid never sees a
        runaway.

        `body=` makes the whole tank one body, which is how
        `tests/test_live_sph.py` weighs the water: the wrench on the tank is the weight
        of what is in it, and that is the sign check and the magnitude check in one
        number.
        """
        origin = np.asarray(self.domain.origin, dtype=np.float64)
        size = np.asarray(self.domain.size, dtype=np.float64)
        if faces is None:
            rules = self.domain.face_rules()
            names = [name for name, rule in zip(FACE_ORDER, rules)
                     if rule in (L.FACE_WALL, L.FACE_SLIP)]
        else:
            names = [str(f) for f in faces]
        if not names:
            raise UnphysicalScaling(
                "no face of this domain is a wall, so `boundary_from_domain()` would "
                "place no particles and the fluid would fall straight out. Set at least "
                "one face to 'wall' on the Domain, or name the faces explicitly"
            )
        thickness = self.boundary_layers * self.spacing
        before = len(self._bdy_x)
        for name in names:
            axis = "xyz".index(name[1])
            side = name[0]
            low = list(origin - thickness)
            high = list(origin + size + thickness)
            if side == "-":
                high[axis] = origin[axis]
                low[axis] = origin[axis] - thickness
            else:
                low[axis] = origin[axis] + size[axis]
                high[axis] = origin[axis] + size[axis] + thickness
            points = self._lattice(low, high)
            if not len(points):
                continue
            self._bdy_x.append(points)
            self._bdy_body.append(np.full(len(points), int(body), dtype=np.int32))
        self._count()
        self._wall_faces = names
        return sum(len(a) for a in self._bdy_x[before:])

    # -- commit to the device -------------------------------------------------------

    def _count(self) -> None:
        """Keep the particle counts true on the host, and mark the device stale.

        `n_fluid` and `n_boundary` are the numbers a case prints before it steps and the
        numbers a test weighs the water with, so they cannot wait for `_commit`. They
        read 0 until the first `step()` otherwise, which makes `n_fluid * mass * g` zero
        and a check against it pass for the wrong reason.
        """
        self.n_fluid = sum(len(a) for a in self._fluid_x)
        self.n_boundary = sum(len(a) for a in self._bdy_x)
        self.n = self.n_fluid + self.n_boundary
        self._committed = False

    @staticmethod
    def _distinct(points: np.ndarray) -> np.ndarray:
        """A mask keeping one particle per lattice site. Two at one site is not a near
        miss, it is a division.

        `boundary_from_domain` builds each wall as a slab that overhangs the other four
        axes by the shell thickness, so that the corners and edges of the tank are
        filled rather than notched -- and that means the site at a corner is produced by
        two or three of the six slabs. Two particles at *identical* positions have
        `r = 0` between them, so the Akinci volume `rho0 / sum W` halves, the fluid's
        density sum near that corner is served by two half-weight particles instead of
        one, and the gradient between them is the `r < eps` guard in `_spline_grad`
        returning zero. Nothing crashes and the forces come out roughly right, which is
        why this survived the first hydrostatic run: measured on a 0.12 x 0.12 x 0.26
        tank at a 20 mm spacing, 1,760 boundary particles were produced for 1,224
        distinct sites -- 536 duplicates, 30% of the wall, all of it on the edges, all of
        it paid for in every neighbour sweep.

        Deduplicated on the lattice *index* rather than on the float position, because
        two slabs compute the same site by two different routes and the two float32
        results need not be bit-identical. First occurrence wins, which for the normal
        order (walls, then shapes) means the wall owns a site a shape also claims.
        """
        if not len(points):
            return np.zeros(0, dtype=bool)
        _, first = np.unique(np.round(points, 9), axis=0, return_index=True)
        keep = np.zeros(len(points), dtype=bool)
        keep[first] = True
        return keep

    def _commit(self) -> None:
        """Build the device arrays. Once, from the host lists, outside the step loop."""
        if self._committed:
            return
        if not self._fluid_x:
            raise UnphysicalScaling(
                "this SphWorld has no fluid particles: call `fill_below(level)` or "
                "`fill_box(low, high)` before stepping"
            )
        fluid_x = np.concatenate(self._fluid_x)
        fluid_v = np.concatenate(self._fluid_v)
        bdy_x = (np.concatenate(self._bdy_x) if self._bdy_x
                 else np.zeros((0, 3), dtype=np.float64))
        bdy_body = (np.concatenate(self._bdy_body) if self._bdy_body
                    else np.zeros(0, dtype=np.int32))
        keep = self._distinct(bdy_x)
        bdy_x, bdy_body = bdy_x[keep], bdy_body[keep]
        keep = self._distinct(fluid_x)
        fluid_x, fluid_v = fluid_x[keep], fluid_v[keep]
        self.duplicates = (self.n_fluid + self.n_boundary) - (len(fluid_x) + len(bdy_x))
        self.n_fluid, self.n_boundary = len(fluid_x), len(bdy_x)
        self.n = self.n_fluid + self.n_boundary

        x = np.concatenate([fluid_x, bdy_x]).astype(np.float32)
        v = np.concatenate([fluid_v, np.zeros((self.n_boundary, 3))]).astype(np.float32)
        body = np.concatenate([np.full(self.n_fluid, -1, dtype=np.int32),
                               bdy_body]).astype(np.int32)

        # A boundary particle's position in its own body's frame, captured once. The
        # device poses from this every step; nothing reads the host positions again.
        local = x.astype(np.float64).copy()
        for index in range(self.n_fluid, self.n):
            b = int(body[index])
            if b < 0:
                continue
            pose = self._poses[b]
            local[index] = _quat_matrix(pose[3:]).T @ (local[index]
                                                       - np.asarray(pose[:3]))

        d = self.device
        self.x = wp.array(x, dtype=wp.vec3, device=d)
        self.v = wp.array(v, dtype=wp.vec3, device=d)
        self.local = wp.array(local.astype(np.float32), dtype=wp.vec3, device=d)
        self.body = wp.array(body, dtype=wp.int32, device=d)
        self.psi = wp.zeros(self.n, dtype=DTYPE, device=d)
        self.rho = wp.full(self.n_fluid, DTYPE(self.rho0), dtype=DTYPE, device=d)
        # The density the constant-density constraint reads. **The same array when
        # delta-SPH is off**, which is the usual case and costs neither the allocation
        # nor the launch; a second array only when there is a diffusion to put in it.
        # See `diffuse_density` on why the raw field stays the one every report reads.
        self.rho_c = (wp.zeros(self.n_fluid, dtype=DTYPE, device=d)
                      if self.density_diffusion > 0.0 else self.rho)
        self.alpha = wp.zeros(self.n_fluid, dtype=DTYPE, device=d)
        self.adv = wp.zeros(self.n_fluid, dtype=DTYPE, device=d)
        self.kappa = wp.zeros(self.n_fluid, dtype=DTYPE, device=d)
        self.kappa_sum = wp.zeros(self.n_fluid, dtype=DTYPE, device=d)
        # Where `nonpressure` leaves gravity and viscosity, so that its neighbour loop
        # is not reading the velocity field it is writing. Every fluid entry is written
        # by exactly one thread of every launch, so `zeros` is for tidiness and not for
        # correctness -- `wp.empty` would read the same, which is precisely the property
        # this file is not willing to rely on anywhere.
        self.accel = wp.zeros(self.n_fluid, dtype=wp.vec3, device=d)
        self._clamped = wp.zeros(1, dtype=wp.int32, device=d)
        # The reflection box is the domain box itself, computed once -- host arithmetic
        # with no business inside the substep loop.
        #
        # **Not inset.** It was inset by half a spacing first, and half a spacing is
        # exactly where `_lattice` puts the first row of fluid, so every particle on the
        # floor was clamped on every substep. The column was then being held up by a
        # position constraint rather than by pressure, and the measurable consequence
        # was that the tank read 63% of the weight of the water in it (-4.91 N against
        # 7.83 N) -- a missing third of the load, which is exactly the shape of a defect
        # that gets diagnosed as a boundary-force sign error. With the box on the domain
        # the same run reads the numbers in `tests/test_live_sph.py`.
        low = np.asarray(self.domain.origin)
        self._box_low = wp.vec3(*low)
        self._box_high = wp.vec3(*(low + np.asarray(self.domain.size)))

        # The hash grid's dimensions are the hash table's, not a bound on space: a
        # particle outside them is hashed, not lost. Sized at one cell per support radius
        # over the domain and capped, because the table is allocated and a 512^3 table
        # for 3,000 particles is 134 million buckets for nothing.
        cells = [max(4, min(128, int(math.ceil(s / self.support))))
                 for s in self.domain.size]
        self._grid = wp.HashGrid(cells[0], cells[1], cells[2], device=d)

        self._sync_bodies()
        self._committed = True

    def _sync_bodies(self) -> None:
        """Push the host body table to the device. SI both sides -- no scaling here."""
        if not self._device_bodies_dirty and self.body_pose is not None:
            return
        count = max(1, len(self._poses))
        poses = np.zeros((count, 7), dtype=np.float32)
        poses[:, 6] = 1.0
        vels = np.zeros((count, 6), dtype=np.float32)
        coms = np.zeros((count, 3), dtype=np.float32)
        for index, pose in enumerate(self._poses):
            poses[index] = pose
            coms[index] = self._coms[index]
            vels[index] = self._vels[index]
        d = self.device
        self.body_pose = wp.array(poses, dtype=wp.transform, device=d)
        self.body_vel = wp.array(vels, dtype=wp.spatial_vector, device=d)
        self.body_com = wp.array(coms, dtype=wp.vec3, device=d)
        if getattr(self, "_wrench", None) is None or self._wrench.shape[0] != count:
            self._wrench = wp.zeros((count, 6), dtype=DTYPE, device=d)
        self._device_bodies_dirty = False

    # -- the step -------------------------------------------------------------------

    def step(self, substeps: int | None = None) -> dict[str, Any]:
        """Advance the fluid by one coupling step. `docs/sph-contract.md`'s dictionary.

        `mlups` is particle updates per second over 1e6, named for symmetry with the
        lattice backend's lattice updates; `report()` says which it is. It is not
        comparable to an LBM MLUPS figure and nothing should compare them: a particle
        update here is a neighbour sweep with 26 neighbours and several solver
        iterations, and a lattice update is 19 reads and a collision.
        """
        self._commit()
        count = int(self.substeps if substeps is None else substeps)
        sub = self.dt / count
        started = time.perf_counter()

        self._sync_bodies()
        self._wrench.zero_()
        iterations_density = 0
        iterations_divergence = 0

        for _ in range(count):
            self._prepare(sub)
            iterations_divergence += self._solve(divergence=True, dt=sub)
            wp.launch(nonpressure, dim=self.n, device=self.device,
                      inputs=[self._grid.id, self.x, self.v, self.psi, self.rho,
                              self.body, self.body_com, self._wrench, self.accel,
                              DTYPE(self.mass), self.n_fluid, DTYPE(self.support),
                              DTYPE(self.sigma), wp.vec3(*self.gravity), DTYPE(self.nu),
                              DTYPE(self.rho0)])
            # `v += dt a`, which is `advect`'s `x += dt v` on the other two arrays. A
            # separate launch and not the tail of the kernel above, because that kernel's
            # neighbour loop reads `v[j]`: see `nonpressure`.
            wp.launch(advect, dim=self.n_fluid, device=self.device,
                      inputs=[self.v, self.accel, self.n_fluid, DTYPE(sub)])
            self.kappa_sum.zero_()
            iterations_density += self._solve(divergence=False, dt=sub)
            wp.launch(advect, dim=self.n_fluid, device=self.device,
                      inputs=[self.x, self.v, self.n_fluid, DTYPE(sub)])
            wp.launch(clamp_to_domain, dim=self.n_fluid, device=self.device,
                      inputs=[self.x, self.v, self._box_low, self._box_high,
                              self._clamped, self.n_fluid, DTYPE(0.5)])
            self.substeps_taken += 1

        wp.synchronize_device(self.device)
        # The wrench filter is driven here and nowhere else, because this is the one
        # thing that happens exactly once a coupling step. A filter updated inside
        # `wrench()` would advance by one sample every time a caller *looked*, and
        # `benchmarks/live/free_settle.py` looks twice on the steps it tabulates -- so
        # its time constant would be a function of the reporting code. See `wrench`.
        if self.wrench_filter > 0.0:
            raw = self._wrench.numpy().astype(np.float64) / max(float(count), 1.0)
            if self._wrench_mean is None:
                self._wrench_mean = raw
            else:
                a = self.wrench_filter
                self._wrench_mean = a * raw + (1.0 - a) * self._wrench_mean
        elapsed = time.perf_counter() - started
        self.seconds += elapsed
        self.time += self.dt
        self.steps += 1
        self.iterations_density += iterations_density
        self.iterations_divergence += iterations_divergence
        self.solves += count
        return {
            "step": self.steps,
            "substeps": count,
            "seconds": elapsed,
            # Wall clock and physical time, never the same key. A case measuring a
            # sloshing period, a dam-break front speed or a decay rate divides by one of
            # these, and dividing by the wrong one is wrong by a factor nobody can see:
            # the answer still has the right units and a plausible magnitude.
            "simulated": self.dt,
            "particles": self.n_fluid,
            "mlups": self.n_fluid * count / elapsed / 1e6 if elapsed > 0 else 0.0,
            "iterations": iterations_density / max(1, count),
            "divergence_iterations": iterations_divergence / max(1, count),
        }

    def _prepare(self, dt: float | None = None) -> None:
        """Pose the boundary, rebuild the grid, and recompute densities and factors.

        `dt` is the substep length, which only the delta-SPH term needs: it is a rate and
        has to be integrated over the interval `residual` is then asked about. Defaults
        to `self.dt/self.substeps`, which is what it is unless the caller overrode
        `substeps` on this `step()`.

        The grid is rebuilt every substep and not reused. It is the obvious thing to try
        to avoid and it is not avoidable here: the whole point of this backend is that
        bodies move, so the boundary particles move, so the cell assignment is stale after
        one substep.

        And it is not what a substep costs. Measured on the 2,394-particle tank, best of
        40 launches each:

            grid.build          0.35 ms    0.2% of a 143 ms substep
            boundary_volumes   10.23 ms    7.2%
            factors             4.98 ms    3.5%
            residual            4.64 ms    3.2%   (x4.7 density iterations)
            densities           4.30 ms    3.0%
            apply_pressure      3.40 ms    2.4%   (x4.7)

        The grid is a thousandth of the step and the queries over it are the whole of it.
        Those launch timings do not sum to the substep -- they are isolated best-of-40
        launches with a warm cache and the loop's marginal per-iteration cost is about
        three times the sum of `residual` and `apply_pressure` -- so the table is a
        ranking and not a decomposition, and it is the ranking that answers the question
        this docstring is about.
        """
        if self.n_boundary:
            wp.launch(pose_boundary, dim=self.n_boundary, device=self.device,
                      inputs=[self.x, self.v, self.local, self.body, self.body_pose,
                              self.body_vel, self.body_com, self.n_fluid])
        self._grid.build(self.x, self.support)
        common = [self._grid.id, self.x, self.psi]
        if self.n_boundary:
            wp.launch(boundary_volumes, dim=self.n, device=self.device,
                      inputs=[*common, self.n_fluid, DTYPE(self.support),
                              DTYPE(self.sigma), DTYPE(self.rho0)])
        wp.launch(densities, dim=self.n, device=self.device,
                  inputs=[*common, self.rho, DTYPE(self.mass), self.n_fluid,
                          DTYPE(self.support), DTYPE(self.sigma)])
        if self.density_diffusion > 0.0:
            # `dt delta h' c0`, the whole prefactor, formed on the host once a substep.
            # The substep and not `self.dt`, because this is a rate integrated over the
            # interval `residual` is about to be asked about.
            wp.launch(diffuse_density, dim=self.n, device=self.device,
                      inputs=[self._grid.id, self.x, self.rho, self.rho_c,
                              DTYPE(self.mass), self.n_fluid, DTYPE(self.support),
                              DTYPE(self.sigma),
                              DTYPE((self.dt / self.substeps if dt is None else dt)
                                    * self.diffusivity)])
        wp.launch(factors, dim=self.n, device=self.device,
                  inputs=[*common, self.rho, self.alpha, DTYPE(self.mass), self.n_fluid,
                          DTYPE(self.support), DTYPE(self.sigma)])

    def _solve(self, *, divergence: bool, dt: float) -> int:
        """One DFSPH solver, iterated to `tolerance`. Returns the iteration count.

        Two iterations minimum, which is the papers' own floor and not a safety margin:
        the first iteration's residual is computed from a velocity field nothing has
        corrected yet, so a solver that stopped after one would have applied a correction
        whose effect it never looked at.

        The convergence test is on the **mean** residual, which is the criterion in both
        papers (`eta` is stated for the average density error). The max is read in the
        same two floats and is in `report()`, because a mean that passes with one
        particle at 40% is worth knowing about and a max-based criterion on a free surface
        chases a single particle in a corner for the whole iteration budget.
        """
        # `rho_c` and not `rho`: the density the CONSTRAINT reads, which delta-SPH
        # diffuses and which is the same array when it is off. `apply_pressure` below
        # keeps the raw `rho`, where it only normalises a stiffness.
        args = [self._grid.id, self.x, self.v, self.psi, self.rho_c, self.adv,
                DTYPE(self.mass), self.n_fluid, DTYPE(self.support), DTYPE(self.sigma),
                DTYPE(self.rho0), DTYPE(dt), 0 if divergence else 1]
        iterations = 0
        while iterations < self.max_iterations:
            wp.launch(residual, dim=self.n, device=self.device, inputs=args)
            if self.check_every and iterations >= 2 and iterations % self.check_every == 0:
                # The residual field itself, summed on the host in numpy's own fixed
                # order and in float64. **Not a device reduction**, and that is the
                # second half of this file's reproducibility fix: the mean used to come
                # from a `wp.atomic_add` over `n_fluid` float32 values, which adds them
                # in whatever order the blocks happen to retire. float32 addition is not
                # associative, so the mean differed between runs in its sixth digit --
                # and this line is a *branch*, so a sixth-digit difference occasionally
                # bought one more solver iteration on one run than on the next, and the
                # two trajectories had no reason to agree about anything afterwards.
                # numpy's pairwise sum over a fixed-length array is the same arithmetic
                # every time. The cost is the array instead of eight bytes; see the
                # module docstring, which has it measured.
                if float(self.adv.numpy().mean(dtype=np.float64)) <= self.tolerance:
                    break
            wp.launch(stiffness, dim=self.n_fluid, device=self.device,
                      inputs=[self.adv, self.alpha, self.kappa, self.kappa_sum,
                              DTYPE(self.rho0), DTYPE(dt), 0 if divergence else 1])
            wp.launch(apply_pressure, dim=self.n, device=self.device,
                      inputs=[self._grid.id, self.x, self.v, self.psi, self.rho,
                              self.kappa, self.body, self.body_com, self._wrench,
                              DTYPE(self.mass), self.n_fluid, DTYPE(self.support),
                              DTYPE(self.sigma), DTYPE(dt)])
            iterations += 1
        return iterations

    # -- what the caller reads ------------------------------------------------------

    def positions(self) -> np.ndarray:
        """Fluid particle positions, `(n, 3)` in metres. A host read, for reports.

        Fluid only, and so are `velocities`, `densities` and `pressures`, so that the
        four line up row for row -- a boundary particle has no density of its own and no
        pressure, only an Akinci volume. `boundary_positions()` is the other half.
        """
        self._commit()
        return self.x.numpy()[:self.n_fluid].astype(np.float64)

    def velocities(self) -> np.ndarray:
        """Fluid particle velocities, `(n, 3)` in m/s."""
        self._commit()
        return self.v.numpy()[:self.n_fluid].astype(np.float64)

    def densities(self) -> np.ndarray:
        """Fluid particle densities, `(n,)` in kg/m^3."""
        self._commit()
        return self.rho.numpy().astype(np.float64)

    def pressures(self) -> np.ndarray:
        """Fluid particle pressures, `(n,)` in Pa, from the last substep's solve.

        `p_i = kappa_i rho_i`, where `kappa` is the constant-density solver's accumulated
        stiffness over its iterations. The identification is dimensional and exact: the
        pressure acceleration this file applies is `-sum_j m_j (k_i/rho_i + k_j/rho_j)
        grad W_ij`, standard SPH applies `-sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2) grad
        W_ij`, and matching the two term by term gives `p = k rho`. There is no equation
        of state anywhere in this backend, so this is the only pressure it has -- and it
        is a real one: `tests/test_live_sph.py` measures it against `rho g h` down a
        column.

        Zero at the free surface rather than negative, because `density_adv` clamps.
        """
        self._commit()
        return (self.kappa_sum.numpy().astype(np.float64)
                * self.rho.numpy().astype(np.float64))

    def boundary_positions(self, body: int | None = None) -> np.ndarray:
        """Boundary particle positions, `(n, 3)` in metres, optionally for one body."""
        self._commit()
        x = self.x.numpy()[self.n_fluid:].astype(np.float64)
        if body is None:
            return x
        return x[self.body.numpy()[self.n_fluid:] == int(body)]

    def volume(self, body: int) -> float:
        """The volume the solver sees for a body, m^3: lattice sites inside its shapes.

        Not the analytic volume, and the difference is the point. A shape's surface does
        not fall on the particle lattice (see `_lattice`), so the body the fluid actually
        pushes on is up to half a spacing larger or smaller than the one that was asked
        for. Grading a buoyancy force against the analytic volume grades the sampler;
        grading it against this grades the solver. `FluidWorld.solid_cells` is the same
        diagnostic for the same reason.
        """
        return float(self._sampled_volume.get(int(body), 0.0))

    def wrench(self, body: int | None = None, *, substeps: int | None = None):
        """The mean fluid wrench over the last coupling step, in SI.

        `(force, torque)` for one body as two 3-tuples in newtons and newton-metres, or
        an `(n, 6)` array for all of them. **The force on the body**, which is
        `lattice.stream`'s convention and the contract's, and the one thing in this file
        that must not be got backwards: water sitting in a tank pushes *down* on the
        tank's floor, and a submerged box feels `+rho V g` *up*.

        Divided by the substep count for the same reason the lattice backend divides by
        it: the accumulator holds one contribution per substep per solver iteration, and
        what the caller and Newton both want is the mean over the interval about to be
        integrated, not the instantaneous value at either end.

        Gravity on the body is **not** in here and must not be added twice. An SPH free
        surface has a real waterline and a real hydrostatic pressure field, so buoyancy
        comes out of the pressure solve -- which is exactly why
        `docs/sph-contract.md` makes `buoyancy=False` the default for this backend.

        **This is the one number in this file that is not reproducible bit for bit, and
        the bound on it is measured**: 1.4e-8 relative on the mean force over a run,
        from the float32 atomics `_accumulate` uses, whose docstring has the measurement
        and why it is not worth removing. Everything the fluid does is deterministic, so
        two runs of a case that *reads* this force differ by that much and no more.
        A case that *feeds* it back -- any `NewtonCoupling` run with a free body -- is
        integrating a body's motion from an input with 1.4e-8 of noise on it, and the
        bound above says nothing about what that grows to. **Measured, and it grows:**
        `sph_static_float` at a 4 mm radius and a 4,000-step settle, run twice on this
        code, read a draft of 11.7869 mm and 11.7648 mm -- 1.9e-3 relative, five orders
        of magnitude up from the input over about 4,600 steps -- with the wobble 5.8%
        apart. So the floating-body cases are **not** reproducible yet and must not be
        reported from a single run.
        Which of two sources that is has not been separated and both are real
        candidates: these atomics, and `newton`/`mujoco_warp`'s own solver, which runs
        its own reductions on the same device. The experiment that separates them is a
        body on a *prescribed* path -- `set_body_state` from the host, no Newton in the
        loop -- stepped twice: a difference there is this file's, and no difference there
        puts it in the rigid solver. Fixing this file's half means a fixed summation
        order over the fluid-boundary pairs, which is a sweep over boundary particles
        accumulating into its own slot with no atomic at all plus a fixed-order reduction
        into bodies -- a second neighbour sweep over a boundary that outnumbers the fluid
        here 9,600 to 3,840, so it is a measurable cost and a rewrite of the two kernels
        that carry this convention, not a dtype change.

        ## `wrench_filter`: the optional low-pass, which is OFF by default

        `SphWorld(wrench_filter=alpha)` makes this method return a one-pole EMA,
        `y_k = alpha x_k + (1 - alpha) y_(k-1)`, advanced once per `step()` and never
        per call. 0.0 is off, and off is the default: the raw per-step force is what
        this method has always returned and what every number in `PROGRESS_LIVE.md` was
        measured on.

        **It changes nothing the solver does, and that is the point and the limit.**
        `couple.py::_push_wrench` reads `world._wrench` on the device through
        `L.wrench_to_newton`, so a filtered world is coupled to Newton by exactly the
        same unfiltered force as an unfiltered one: the body's trajectory, the draft and
        the fluid are bit-for-bit what they were. The filter is on the number a
        **reader** gets -- a controller, a load report, a plot -- which is where the
        per-step noise is a problem and where a phase lag is a price somebody can decide
        to pay.

        **What it costs, which is a lag and not an approximation.** A one-pole EMA at
        `alpha` has a group delay of `(1 - alpha)/alpha` samples at low frequency and a
        -3 dB corner at `f_c ~ alpha/(2 pi (1 - alpha) dt)`. A controller closing a loop
        faster than that is being lied to about *when*, which is a different and worse
        failure than being lied to about *how much*. `report()` prints both numbers for
        the alpha in force, so a run says its own lag.

        **Measured on the signal, not argued.** The EMA run over the free hull's own
        per-step series at a 2 mm radius (`benchmarks/live/force_noise.py`, two
        independent settles, 1142.274 us step, heave period 253.75 ms):

            alpha   n     lag        corner     RMS          cut by    white   peak/mg
            raw     --    --         --         104.6/99.2%  --        --      3.27/4.08
            0.5     3.0   1.14 ms    139.3 Hz    40.6/38.8%  2.6x      1.73x   1.97/2.35
            0.3     5.7   2.67 ms     59.7 Hz    22.2/21.4%  4.7x      2.38x   1.54/1.75
            0.2     9.0   4.57 ms     34.8 Hz    14.1/13.7%  7.3x      3.00x   1.35/1.48
            0.1    19.0  10.28 ms     15.5 Hz     6.7/ 6.6%  15.3x     4.36x   1.16/1.23
            0.05   39.0  21.70 ms      7.3 Hz     3.3/ 3.2%  31.7x     6.24x   1.08/1.10

        `n = (2 - alpha)/alpha` is the equivalent averaging length and `white` is the
        `sqrt(n)` a white signal would give. **The noise is coherent, not white, and the
        factor is 2.4 to 5.1**: nine samples cut it 7.3x where white noise would give
        3.0x. That is why a filter is worth offering at all. The mean is untouched to
        four decimals at every alpha -- 1.0005 against the raw 1.0018 -- so this trades
        phase for amplitude and nothing else.

        **alpha = 0.1 is the setting that reaches the 10% target**, at a nine-step
        (10.28 ms) lag, which is 4.1% of this hull's heave period. alpha = 0.2 reaches
        14% at half that lag.

        And the peak, which is the number a load case wants: raw over-reports the peak
        vertical force by **2.4x to 2.8x** (3.27 `m g` against the filtered 1.35 at
        alpha 0.2). `openreynolds/live/carry.py` is the precedent at the same alpha and
        measured the same shape of error, 3.42 N raw against a true 2.18 N, 1.6x.

        **The filter is the fallback and not the fix.** A coupling step of `CFL/4` gets
        the *raw* per-step RMS to 1.3-5.4% with no lag at all, which is better than any
        alpha here and is what `CFL`'s docstring recommends. This exists for a caller who
        cannot afford 2.3x the wall clock, and for whom a known phase lag is the cheaper
        price.
        """
        self._commit()
        if self.wrench_filter > 0.0 and self._wrench_mean is not None:
            both = self._wrench_mean
        else:
            count = float(self.substeps if substeps is None else substeps)
            both = self._wrench.numpy().astype(np.float64) / max(count, 1.0)
        if body is None:
            return both
        return tuple(both[body, :3]), tuple(both[body, 3:])

    def density_error(self) -> dict[str, float]:
        """`{"max", "mean"}` as fractions of the rest density, plus what they leave out.

        `max` and `mean` are the **compression** error, `max(rho_i/rho0 - 1, 0)`, which
        is the quantity the constant-density solver drives to zero and therefore the only
        one a tolerance can be stated against. The signed and the deficit numbers are in
        the same dictionary under their own keys, because the compression error alone
        would be a flattering half of the story: a free surface particle has no fluid
        above it and its summation density reads 20 to 30% low *by construction*, in a
        converged, correct run. That deficit is not an error the solver can or should
        remove -- asking it to would mean a negative pressure at the surface -- and
        reporting a single `|rho - rho0|/rho0` would make every free-surface case look
        broken while hiding real compression inside the same number.

        A host read of the density field. The health check every case reports, outside
        the step loop.
        """
        self._commit()
        rho = self.densities()
        signed = rho / self.rho0 - 1.0
        over = np.maximum(signed, 0.0)
        under = np.maximum(-signed, 0.0)
        return {
            "max": float(over.max()) if len(over) else 0.0,
            "mean": float(over.mean()) if len(over) else 0.0,
            "max_signed": float(signed.max()) if len(signed) else 0.0,
            "min_signed": float(signed.min()) if len(signed) else 0.0,
            "mean_deficit": float(under.mean()) if len(under) else 0.0,
        }

    def surface_height(self, axis: int = 2, bins: int | tuple[int, int] | None = None
                       ) -> np.ndarray:
        """Free-surface depth per column, in metres, `nan` where the column is dry.

        Columns perpendicular to `axis`, binned over the other two. The height of the
        topmost particle in a column plus one particle radius (`DECISIONS_LIVE.md`
        D19 re-grades the floating cases on a corrected form of this) -- the
        particle's centre is
        half a spacing below the surface it represents, and leaving that out is a
        systematic `r` of missing depth, which on a 100 mm column at a 5 mm radius is 5%
        and would be read as a solver error.

        **The two horizontal axes are binned independently**, each from its own extent, so
        the bins are square in metres and not in index. This returned a square `(n, n)`
        grid sized from the first horizontal axis first, and on a long thin tank -- which
        is every quasi-2D case -- that put most of the columns outside the fluid, where
        they read `nan` correctly and then got zero-filled by a caller that did not expect
        them. A depth divided by the aspect ratio is a believable wrong answer, which is
        the worst kind available here.

        `nan` is dry and 0.0 is "a column of zero depth", which is a different statement.
        `bins` takes one count for both axes or a pair, and defaults to one bin per
        particle spacing on each.

        A host read. For figures, for the sloshing cases, and for checking that a column
        at rest stayed where it was put.
        """
        self._commit()
        axis = int(axis)
        others = [a for a in range(3) if a != axis]
        if bins is None:
            counts = [max(1, int(round(self.domain.size[a] / self.spacing)))
                      for a in others]
        elif isinstance(bins, (int, np.integer)):
            counts = [int(bins), int(bins)]
        else:
            counts = [int(v) for v in bins]
        out = np.full(tuple(counts), np.nan)
        x = self.positions()
        if not len(x):
            return out
        origin = np.asarray(self.domain.origin)
        size = np.asarray(self.domain.size)
        index = []
        for a, count in zip(others, counts):
            frac = (x[:, a] - origin[a]) / size[a]
            index.append(np.clip((frac * count).astype(int), 0, count - 1))
        flat = index[0] * counts[1] + index[1]
        top = np.full(counts[0] * counts[1], -np.inf)
        np.maximum.at(top, flat, x[:, axis])
        wet = np.isfinite(top)
        depth = np.full(counts[0] * counts[1], np.nan)
        depth[wet] = top[wet] + self.radius - origin[axis]
        return depth.reshape(tuple(counts))

    def stable(self) -> bool:
        """Whether every fluid particle still holds a finite position and density.

        The one way this solver fails: a stiffness spike puts a particle somewhere
        non-finite and the next neighbour query propagates it. A host read of two arrays,
        so it is for the long-run tests and not for every step, exactly like
        `FluidWorld.stable`.
        """
        self._commit()
        x = self.x.numpy()[:self.n_fluid]
        rho = self.rho.numpy()
        return bool(np.isfinite(x).all() and np.isfinite(rho).all()
                    and (rho > 0.0).all())

    def report(self) -> list[str]:
        """The world as a human reads it: the discretisation, the solvers, and the cost."""
        self._commit()
        error = self.density_error()
        lines = [
            f"  device:     {self.device}  (warp {self.wp.config.version})",
            f"  particles:  {self.n_fluid:,} fluid + {self.n_boundary:,} boundary "
            f"(Akinci 2012) = {self.n:,}"
            + (f"   ({self.duplicates:,} duplicate sites dropped)"
               if self.duplicates else ""),
            f"  sizing:     radius {self.radius * 1e3:.4g} mm   spacing "
            f"{self.spacing * 1e3:.4g} mm   support h {self.support * 1e3:.4g} mm "
            f"({SUPPORT_FACTOR:g}r, ~26 neighbours)",
            f"  kernel:     cubic spline, 3D, sigma 8/(pi h^3) = {self.sigma:.6g} 1/m3",
            f"  fluid:      {self.fluid.name} (rho {self.rho0:.5g} kg/m3, "
            f"particle mass {self.mass * 1e6:.5g} mg)",
            f"  viscosity:  nu {self.nu:.5g} m2/s"
            + ("" if self.nu == self.fluid.nu else
               f" ({self.nu / self.fluid.nu:.0f}x {self.fluid.name}'s own)")
            + "   Monaghan 1992, 2(d+2)=10, explicit",
            "  surface:    surface tension OFF (no model in this backend; the knob "
            "refuses a non-zero value rather than ignoring it)",
            f"  gravity:    {self.gravity} m/s2, on the fluid -- buoyancy comes out of "
            f"the pressure field, so do not add rho V g as well",
            "  diffusion:  density diffusion OFF (delta-SPH, Molteni & Colagrossi 2009 "
            "/ Antuono 2010); the constant-density constraint reads the summation "
            "density as summed" if self.density_diffusion <= 0.0 else
            f"  diffusion:  delta-SPH delta {self.density_diffusion:g} = a diffusivity "
            f"delta h' c0 of {self.diffusivity:.4g} m2/s "
            f"({self.diffusivity / self.nu:.3g}x nu), diffusion number "
            f"{self.diffusivity * (self.dt / self.substeps) / self.support ** 2:.4g} "
            f"of the 0.125 explicit limit. On the constraint's density only -- the "
            f"reported density error is still the raw summation density",
            f"  solver:     DFSPH (Bender & Koschier 2015/2017), constant-density and "
            f"divergence-free, mean residual <= {self.tolerance:g}, <= "
            f"{self.max_iterations} iterations each",
            "  wrench:     RAW per-step force (wrench_filter off, the default). "
            "`SphWorld.wrench` has the measured per-step noise and what the filter "
            "costs" if self.wrench_filter <= 0.0 else
            f"  wrench:     `wrench()` returns a one-pole EMA at alpha "
            f"{self.wrench_filter:g} -- a "
            f"{(1.0 - self.wrench_filter) / self.wrench_filter:.3g}-step group delay "
            f"({(1.0 - self.wrench_filter) / self.wrench_filter * self.dt * 1e3:.3g} ms) "
            f"and a -3 dB corner at "
            f"{self.wrench_filter / (2.0 * math.pi * (1.0 - self.wrench_filter) * self.dt):.3g} Hz. "
            f"The FLUID AND THE BODY ARE UNAFFECTED: couple.py reads the raw device "
            f"wrench, so this filters what a reader sees and not what Newton integrates",
            f"  coupling:   style {self.coupling_style!r} -- every array above is SI, "
            f"so couple.py's scale, pose and wrench conversions are the identity",
            f"  equivalent: c0 {self.c0:.4g} m/s, the weakly compressible sound speed "
            f"this run's tolerance matches at its peak estimate. DFSPH has none of its "
            f"own and nothing here reads it",
            f"  timestep:   dt {self.dt:.6g} s per step, {self.substeps} substeps of "
            f"{self.dt / self.substeps:.6g} s "
            f"(CFL {CFL:g} at an ESTIMATED peak of {self._peak_estimate:.4g} m/s)",
            f"  density:    max compression {error['max'] * 100:.3f}%, mean "
            f"{error['mean'] * 100:.3f}%; surface deficit (expected, not an error) mean "
            f"{error['mean_deficit'] * 100:.2f}%, worst {-error['min_signed'] * 100:.2f}%",
        ]
        if self._wall_faces:
            lines.append(f"  walls:      boundary particles on "
                         f"{', '.join(self._wall_faces)}, "
                         f"{self.boundary_layers} layers deep; no periodic axis "
                         f"(see this module's docstring for what to do instead)")
        lines.append(
            "  host reads: " + (
                f"the solver residual every {self.check_every} iteration(s) -- "
                f"{self.n_fluid * 4 / 1024:.1f} KiB and a device sync, summed on the "
                f"host so the convergence decision does not depend on a float32 atomic's "
                f"summation order; the only host round trip inside the step"
                if self.check_every else
                "none inside the step (check_every=0: both solvers run to the cap)"))
        outside = version_note()
        if outside is not None:
            lines.append(outside)
        if self.steps:
            rate = self.n_fluid * self.substeps_taken / self.seconds / 1e6
            lines.append(
                f"  cost:       {self.steps:,} coupling steps "
                f"({self.substeps_taken:,} substeps, {self.time:.4g} simulated "
                f"seconds) in {self.seconds:.1f} s wall clock at "
                f"{rate:.3f} M particle-updates/s, "
                f"{self.seconds / self.steps * 1e3:.2f} ms a step"
            )
            lines.append(
                f"  iterations: {self.iterations_density / max(1, self.solves):.1f} "
                f"density, {self.iterations_divergence / max(1, self.solves):.1f} "
                f"divergence, per substep"
            )
            lines.append(
                f"  realtime:   {self.dt * self.steps / self.seconds:.4f}x "
                f"(simulated seconds per wall-clock second)"
            )
        return lines
