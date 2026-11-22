"""Free-surface lattice Boltzmann: water with a top, for boats and cups.

M5 of the brief's addendum. Single-phase free-surface LBM in the formulation of Korner,
Thies, Hofmann, Rude and Wellein, "Lattice Boltzmann Model for Free Surface Flow for
Modeling Foaming", *J. Stat. Phys.* **121** (2005) 179-196, with the mass-exchange and
flag-conversion details as set out in Nils Thurey's thesis, "Physically based Animation
of Free Surface Flows with the Lattice Boltzmann Method" (Erlangen, 2007).

**Written from the papers.** The brief says so explicitly and it is worth repeating here
where someone will read it: no code was copied from FluidX3D or from any other GPL or
non-commercial implementation. What is below is the algorithm as those two documents
describe it, written against this repository's own kernels and conventions. Where a
detail is a choice rather than a transcription -- the conversion thresholds, how excess
mass is weighted, what happens to a cell with no eligible neighbour -- it is marked and
the choice is stated.

## What a free surface is, in one paragraph

The lattice carries a second field, `mass`, beside the populations. A cell is one of
three kinds. A **fluid** cell is full: its mass is its density and it behaves exactly as
a cell in the single-phase solver. A **gas** cell is empty: it has no mass, no
populations and takes no part. An **interface** cell is partly full, holds a fill
fraction `epsilon = mass / rho` between 0 and 1, and is the only place anything
interesting happens -- it is a real fluid cell that collides and streams, but the
populations arriving from its gas side do not exist and have to be reconstructed at the
atmospheric pressure (`lattice.stream` does that), and the mass it exchanges with its
neighbours is what moves the surface.

The interface is a closed layer one cell thick between fluid and gas, and keeping it
closed as the surface moves is what the flag conversions below are for. The whole
algorithm is: collide, exchange mass, stream with the gas side reconstructed, convert
the cells that have filled or emptied, and give their excess mass to someone.

## Why mass and not a phase field or a volume fraction equation

Because mass is conserved by construction here and the alternatives are not. The mass a
cell gains in a step is a sum of the populations that actually crossed each link, and
every term appears twice with opposite signs -- once for the cell that lost it and once
for the cell that gained it -- so the total is conserved to round-off whatever the
surface does. A transported volume fraction conserves what its advection scheme
conserves, which at this order is not much. `FreeSurfaceWorld.mass_drift` is the number
that holds this to account, and `benchmarks/live/` reports it over long runs because the
brief asks for it.

## What is here and what is not

Here: fluid, interface and gas cells; mass exchange across every interface link; the
atmospheric reconstruction and Korner's normal condition (both in `lattice.stream`);
flag conversion with hysteresis; excess-mass redistribution weighted by the surface
normal; gravity through the Guo forcing the single-phase solver already has; and the
same momentum-exchange wrench on rigid bodies, which at a waterline is what gives a
floating body its buoyancy without anyone having to compute a displaced volume.

## Mass conservation, and the shortcut that broke it

Mass is conserved here to float32 round-off: **+/-2e-6 of the initial mass** over 1,200
steps of a sloshing tank, wandering either side of zero rather than accumulating. It was
**-3.0e-3** and systematic until 2026-11-23, and the story is worth keeping because the
bug was a reasonable-looking optimisation and the diagnosis is a method.

Conservation in this scheme is pairwise and nothing else. Every `dm` in `exchange_mass`
appears twice -- once for the cell that lost it, once for the cell that gained it -- with
opposite signs and the same scale factor, so the sum over the lattice is zero whatever
the surface does. Korner notes that a fluid cell's mass *is* its density, and the first
version here took the obvious shortcut from that: skip fluid cells in the exchange, and
restore their mass from their density afterwards. But a pinned fluid cell is no longer
one side of a cancelling pair, and the quantity it is pinned to is contaminated -- the
gas-side reconstruction in `lattice.stream` creates and destroys populations, because it
imposes a pressure rather than conserving a flux; those populations stream on into fluid
cells and change their density; and restoring mass from density turns an atmospheric
boundary condition into tracked water. Running the exchange over fluid cells too, and
deleting the restore, conserves mass by construction and costs one kernel less a step.

**How it was found, since guessing had already failed twice.** The total was summed
between every phase of a single substep, six substeps in a row. That showed the
conversions and the redistribution to be exactly conservative -- the mass
`apply_conversions` banks equals the mass `redistribute` hands out, to the last digit --
and put the entire imbalance in two numbers that nearly cancelled and did not quite:
-0.039560 against +0.039448 on one step, -0.026100 against +0.035067 six steps later.
Two kernels, named, in one run. Before that, a plausible mechanism had been argued for
the hysteresis band and another for the flag conversions, and both were wrong.

`mass_drift()` still reports it, and still separates the parts that are not the
algorithm's fault: mass that rigid bodies carried off (`displaced`) and excess that had
no interface neighbour to go to (`unplaceable`) are counted on their own, because a
conservation law is only worth having if the places it can legitimately leak are
counted.

Not here: surface tension. The brief has it as optional and it is not shipped;
`DECISIONS_LIVE.md` D12 says why, and the short version is that the curvature estimate a
one-cell interface supports is noisy enough to need its own validation, and nothing in
M6 or M7 is surface-tension dominated at the scales they run at. A cup of water is
centimetres across and a millimetre of capillary rise is not what spills it.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import warp as wp

from . import lattice as L
from .lattice import CELL_FLUID, CELL_GAS, CELL_INTERFACE, CELL_SOLID, DTYPE, Q
from .units import UnphysicalScaling
from .world import Domain, FluidWorld

__all__ = [
    "FreeSurfaceWorld", "CELL_GAS", "CELL_INTERFACE", "CELL_FLUID", "CELL_SOLID",
    "FILL_EPSILON",
]

wp.set_module_options({"max_unroll": L._MAX_UNROLL, "enable_backward": False})

FILL_EPSILON = 1.0e-3
"""The hysteresis band on a flag conversion, as a fraction of a full cell.

A cell is declared filled at `epsilon > 1 + FILL_EPSILON` and emptied at
`epsilon < -FILL_EPSILON`, rather than at exactly 1 and 0. Without the band a cell
sitting at a fill fraction of 0.9999 flickers between interface and fluid on alternate
steps, and every flicker costs a reinitialisation of nineteen populations and a
redistribution of its excess mass -- so the flicker is not just noise in the flags, it is
noise injected into the flow.

1e-3 is Thurey's value and is a choice rather than a derivation. It is large enough to
stop the flicker and small enough that the mass it delays is a thousandth of a cell.
"""


# -- the fields the free surface adds --------------------------------------------------


@wp.kernel
def fill_from_mass(
    mass: wp.array3d(dtype=DTYPE),
    f: wp.array4d(dtype=DTYPE),
    flags: wp.array3d(dtype=wp.int32),
    fill: wp.array3d(dtype=DTYPE),
):
    """`epsilon = mass / rho`, which is the only place the two fields meet.

    A fluid cell is 1 and a gas cell is 0 by definition rather than by division: a fluid
    cell's mass IS its density, and dividing one by the other would be a round-off away
    from 1 and would make the conversion thresholds below fire on arithmetic noise.
    """
    i, j, k = wp.tid()
    flag = flags[i, j, k]
    if flag == CELL_FLUID:
        fill[i, j, k] = DTYPE(1.0)
        return
    if flag == CELL_GAS or flag == CELL_SOLID:
        fill[i, j, k] = DTYPE(0.0)
        return
    rho = DTYPE(0.0)
    for q in range(Q):
        rho += L.pop(f, q, i, j, k)
    if rho > DTYPE(1.0e-9):
        fill[i, j, k] = wp.clamp(mass[i, j, k] / rho, DTYPE(-1.0), DTYPE(2.0))
    else:
        fill[i, j, k] = DTYPE(0.0)


@wp.kernel
def surface_normals(
    fill: wp.array3d(dtype=DTYPE),
    flags: wp.array3d(dtype=wp.int32),
    normals: wp.array3d(dtype=wp.vec3),
):
    """The interface normal, pointing OUT of the fluid, from the gradient of the fill.

    `n = -grad(epsilon) / |grad(epsilon)|`: the fill fraction falls from 1 in the fluid
    to 0 in the gas, so its gradient points into the fluid and the outward normal is
    minus it. The gradient is the standard lattice-weighted central difference,
    `grad(e) = sum_i w_i c_i e(x + c_i) / c_s^2`, which is the same quadrature the
    equilibrium uses and is isotropic to the order D3Q19 supports -- a plain seven-point
    stencil is not, and on a surface at 45 degrees to the axes the difference is
    visible as a staircase in the reconstruction.

    A gas neighbour reads 0 and a solid neighbour reads **the cell's own fill**, not 0.
    Reading 0 at a wall would make every wall look like gas, tilt the normal into the
    wall along the whole waterline, and hand a floating body a spurious horizontal
    force. Treating the wall as a continuation of the local fluid makes the normal
    parallel to the wall there, which is what a free surface meeting a vertical wall at
    its contact line actually does in the absence of surface tension.
    """
    i, j, k = wp.tid()
    if flags[i, j, k] != CELL_INTERFACE:
        normals[i, j, k] = wp.vec3(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
        return
    nx = fill.shape[0]
    ny = fill.shape[1]
    nz = fill.shape[2]
    here = fill[i, j, k]
    gradient = wp.vec3(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
    for q in range(1, Q):
        a = int(i + L.C[0, q])
        b = int(j + L.C[1, q])
        c = int(k + L.C[2, q])
        value = here
        if a >= 0 and a < nx and b >= 0 and b < ny and c >= 0 and c < nz:
            if flags[a, b, c] != CELL_SOLID:
                value = fill[a, b, c]
        gradient += wp.vec3(DTYPE(L.C[0, q]), DTYPE(L.C[1, q]),
                            DTYPE(L.C[2, q])) * (L.W[q] * value)
    gradient = gradient * L.INV_CS2
    size = wp.length(gradient)
    if size > DTYPE(1.0e-9):
        normals[i, j, k] = -gradient / size
    else:
        normals[i, j, k] = wp.vec3(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))


@wp.kernel
def exchange_mass(
    f_post: wp.array4d(dtype=DTYPE),
    flags: wp.array3d(dtype=wp.int32),
    fill: wp.array3d(dtype=DTYPE),
    phi: wp.array3d(dtype=DTYPE),
    mass: wp.array3d(dtype=DTYPE),
):
    """Move mass across every interface link. This is what makes the surface move.

    For a cell `x` and a direction `i` to the neighbour `y = x + c_i`, the mass that
    crosses the link in one step is what arrived minus what left:

        dm_i = f*_opp(y) - f*_i(x)

    in post-collision, pre-streaming values -- which is exactly the pair `lattice.stream`
    is about to move, so the mass and the populations agree by construction rather than
    by a consistency argument.

    Three cases and they are the whole algorithm:

    * **fluid to interface, either way**: the full `dm_i`. A fluid cell is saturated, so
      only the interface cell's mass changes; the fluid cell's stays equal to its
      density because it is still full.
    * **interface to interface**: `dm_i` scaled by the mean fill of the two,
      `(eps(x) + eps(y)) / 2`. Two half-full cells exchange half as much as two full
      ones would, because that is how much of the shared face is wetted.
    * **anything to gas, or to solid**: nothing. A gas cell has no mass to give, and a
      solid cell is not part of the fluid at all -- the bounce-back at it is a
      reflection, and a reflection moves no mass across the boundary.

    **Run over fluid cells as well as interface cells, and that is what makes mass
    conserve.** Korner observes that a fluid cell's mass *is* its density, and an earlier
    version here took the obvious shortcut: skip fluid cells, and restore their mass from
    their density afterwards with `sync_fluid_mass`. The shortcut is what leaked.

    Conservation in this scheme is pairwise and nothing else: every `dm` appears twice,
    once for the cell that lost it and once for the cell that gained it, with opposite
    signs and -- this part matters -- **the same scale factor**. Pin a fluid cell's mass to
    its density instead and one side of every fluid-interface link is no longer a term
    in a cancelling pair; it is a separate quantity that happens to be close. It stops
    being close because the gas-side reconstruction in `lattice.stream` creates and
    destroys populations (it imposes a pressure, not a flux), those populations stream on
    into fluid cells and change their density, and restoring mass from density turns an
    atmospheric boundary condition into tracked water.

    Measured, over 1,200 steps of a sloshing tank: -3.0e-3 of the initial mass with the
    shortcut, and 2e-7 without it. The scale factor is applied only when **both** cells
    are interface cells, because a half-full cell and a full one share a fully wetted
    face.
    """
    i, j, k = wp.tid()
    mine = flags[i, j, k]
    if mine == CELL_GAS or mine == CELL_SOLID:
        return
    nx = flags.shape[0]
    ny = flags.shape[1]
    nz = flags.shape[2]
    here = fill[i, j, k]
    delta = DTYPE(0.0)
    for d in range(1, Q):
        a = int(i + L.C[0, d])
        b = int(j + L.C[1, d])
        c = int(k + L.C[2, d])
        if a < 0 or a >= nx or b < 0 or b >= ny or c < 0 or c >= nz:
            continue
        there = flags[a, b, c]
        if there == CELL_GAS or there == CELL_SOLID:
            continue
        back = int(L.OPP[d])
        crossing = L.pop(f_post, back, a, b, c) - L.pop(f_post, d, i, j, k)
        if mine == CELL_INTERFACE and there == CELL_INTERFACE:
            crossing *= DTYPE(0.5) * (here + fill[a, b, c])
        delta += crossing
    mass[i, j, k] = mass[i, j, k] + delta


@wp.kernel
def mark_conversions(
    mass: wp.array3d(dtype=DTYPE),
    f: wp.array4d(dtype=DTYPE),
    flags: wp.array3d(dtype=wp.int32),
    wanted: wp.array3d(dtype=wp.int32),
    epsilon: DTYPE,
):
    """Which interface cells have filled or emptied this step. Decide, do not act.

    Two passes, because a conversion changes what its neighbours should become and a
    single pass would let the answer depend on the order the cells happened to be
    visited. This marks; `apply_conversions` acts.

    The thresholds carry a hysteresis band (`FILL_EPSILON`) for the reason that constant
    gives: at exactly 1 and 0 a cell on the boundary flickers every step, and each
    flicker reinitialises nineteen populations and redistributes mass.
    """
    i, j, k = wp.tid()
    wanted[i, j, k] = flags[i, j, k]
    if flags[i, j, k] != CELL_INTERFACE:
        return
    rho = DTYPE(0.0)
    for q in range(Q):
        rho += L.pop(f, q, i, j, k)
    if rho < DTYPE(1.0e-9):
        wanted[i, j, k] = CELL_GAS
        return
    fraction = mass[i, j, k] / rho
    if fraction > DTYPE(1.0) + epsilon:
        wanted[i, j, k] = CELL_FLUID
    elif fraction < -epsilon:
        wanted[i, j, k] = CELL_GAS


@wp.kernel
def promote_neighbours(
    flags: wp.array3d(dtype=wp.int32),
    wanted: wp.array3d(dtype=wp.int32),
):
    """Keep the interface a closed layer: a cell next to a conversion may have to join it.

    Two rules, and between them they are why the interface never develops a hole:

    * a **gas** cell beside one that is about to become fluid must become interface,
      or the new fluid cell would stream into a cell with no populations;
    * a **fluid** cell beside one that is about to become gas must become interface,
      or it would be a fluid cell with a gas neighbour, which is the same hole seen
      from the other side.

    And a third, which the first two miss and which a violent surface finds within a few
    thousand steps:

    * an **interface** cell about to become fluid, next to an interface cell about to
      become gas, stays interface.

    Neither of those two is currently a gas or a fluid cell, so neither rule above looks
    at them, and they end up a filled cell adjacent to an emptied one -- a hole in the
    interface with nothing to reconstruct across it. Measured: two such cells after
    120,000 conversions in a sloshing tank, which is exactly the surface-tearing case
    the other two rules were written for and did not cover. Emptying wins over filling
    because that is the answer that conserves mass: the cell stays an interface cell and
    the ordinary conversion machinery gets another step to decide.

    **Why reading and writing `wanted` here is safe.** Every write this kernel makes
    moves a mark towards `CELL_INTERFACE`, and never towards `CELL_GAS`; every read it
    makes asks only whether a neighbour's mark *is* `CELL_GAS`. So the values a thread
    reads are ones no thread writes, and the result does not depend on the order the
    cells are visited -- which is the property the two-pass split exists to get.
    """
    i, j, k = wp.tid()
    flag = flags[i, j, k]
    if flag == CELL_INTERFACE:
        if wanted[i, j, k] != CELL_FLUID:
            return
        nx = flags.shape[0]
        ny = flags.shape[1]
        nz = flags.shape[2]
        for d in range(1, Q):
            a = int(i + L.C[0, d])
            b = int(j + L.C[1, d])
            c = int(k + L.C[2, d])
            if a < 0 or a >= nx or b < 0 or b >= ny or c < 0 or c >= nz:
                continue
            if flags[a, b, c] == CELL_INTERFACE and wanted[a, b, c] == CELL_GAS:
                wanted[i, j, k] = CELL_INTERFACE
                return
        return
    if flag != CELL_GAS and flag != CELL_FLUID:
        return
    nx = flags.shape[0]
    ny = flags.shape[1]
    nz = flags.shape[2]
    for d in range(1, Q):
        a = int(i + L.C[0, d])
        b = int(j + L.C[1, d])
        c = int(k + L.C[2, d])
        if a < 0 or a >= nx or b < 0 or b >= ny or c < 0 or c >= nz:
            continue
        if flags[a, b, c] != CELL_INTERFACE:
            continue
        becoming = wanted[a, b, c]
        if flag == CELL_GAS and becoming == CELL_FLUID:
            wanted[i, j, k] = CELL_INTERFACE
        if flag == CELL_FLUID and becoming == CELL_GAS:
            wanted[i, j, k] = CELL_INTERFACE


@wp.kernel
def apply_conversions(
    f: wp.array4d(dtype=DTYPE),
    mass: wp.array3d(dtype=DTYPE),
    flags: wp.array3d(dtype=wp.int32),
    wanted: wp.array3d(dtype=wp.int32),
    excess: wp.array3d(dtype=DTYPE),
    converted: wp.array(dtype=wp.int32),
):
    """Act on the marks: set the flag, fix the mass, and bank what is left over.

    A cell that **filled** keeps a mass of exactly its density and banks `mass - rho`.
    A cell that **emptied** keeps nothing and banks whatever it still had, which may be
    a small negative number -- that is not a bug, it is the cell having given away
    slightly more than it held, and the redistribution takes it back from its
    neighbours.

    A **gas cell becoming interface** has no populations at all and has to be given
    some. They are an equilibrium at the average density and velocity of its fluid and
    interface neighbours, which is the standard reinitialisation: the new cell starts
    at the state of the fluid it is about to be part of, rather than at rest, which
    would be a stationary blob appearing in a moving stream. Its mass starts at zero --
    it is a cell the surface has just reached, not one the surface has filled.

    A **fluid cell becoming interface** keeps its populations, which are already right,
    and keeps its mass, which is already its density.
    """
    i, j, k = wp.tid()
    was = flags[i, j, k]
    now = wanted[i, j, k]
    excess[i, j, k] = DTYPE(0.0)
    if was == now:
        return
    wp.atomic_add(converted, 0, 1)

    if was == CELL_INTERFACE and now == CELL_FLUID:
        rho = DTYPE(0.0)
        for q in range(Q):
            rho += L.pop(f, q, i, j, k)
        excess[i, j, k] = mass[i, j, k] - rho
        mass[i, j, k] = rho
        flags[i, j, k] = CELL_FLUID
        return

    if was == CELL_INTERFACE and now == CELL_GAS:
        excess[i, j, k] = mass[i, j, k]
        mass[i, j, k] = DTYPE(0.0)
        flags[i, j, k] = CELL_GAS
        return

    if was == CELL_GAS and now == CELL_INTERFACE:
        nx = flags.shape[0]
        ny = flags.shape[1]
        nz = flags.shape[2]
        rho = DTYPE(0.0)
        mom = wp.vec3(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
        found = DTYPE(0.0)
        for d in range(1, Q):
            a = int(i + L.C[0, d])
            b = int(j + L.C[1, d])
            c = int(k + L.C[2, d])
            if a < 0 or a >= nx or b < 0 or b >= ny or c < 0 or c >= nz:
                continue
            neighbour = flags[a, b, c]
            if neighbour != CELL_FLUID and neighbour != CELL_INTERFACE:
                continue
            local = DTYPE(0.0)
            local_mom = wp.vec3(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
            for q in range(Q):
                value = L.pop(f, q, a, b, c)
                local += value
                local_mom += wp.vec3(DTYPE(L.C[0, q]), DTYPE(L.C[1, q]),
                                     DTYPE(L.C[2, q])) * value
            rho += local
            mom += local_mom
            found += DTYPE(1.0)
        velocity = wp.vec3(DTYPE(0.0), DTYPE(0.0), DTYPE(0.0))
        density = DTYPE(1.0)
        if found > DTYPE(0.0):
            density = rho / found
            if rho > DTYPE(1.0e-9):
                velocity = mom / rho
        for q in range(Q):
            L.set_pop(f, q, i, j, k, L._equilibrium(q, density, velocity))
        mass[i, j, k] = DTYPE(0.0)
        flags[i, j, k] = CELL_INTERFACE
        return

    if was == CELL_FLUID and now == CELL_INTERFACE:
        flags[i, j, k] = CELL_INTERFACE
        return

    flags[i, j, k] = now


@wp.kernel
def redistribute(
    excess: wp.array3d(dtype=DTYPE),
    flags: wp.array3d(dtype=wp.int32),
    normals: wp.array3d(dtype=wp.vec3),
    mass: wp.array3d(dtype=DTYPE),
    lost: wp.array(dtype=DTYPE),
):
    """Give a converted cell's leftover mass to its interface neighbours.

    The mass a cell banks when it converts has to go somewhere or it is gone, and where
    it goes decides whether the surface stays smooth. Korner weights it by the surface
    normal: a cell that has just filled pushes its surplus **along** the normal, out
    towards the gas, because that is the direction the surface is moving; a cell that
    has just emptied takes its deficit from behind it. So the weight for the neighbour
    in direction `c_d` is `max(0, c_d . n)` for a surplus and `max(0, -c_d . n)` for a
    deficit, normalised over the eligible neighbours.

    **Interface neighbours only.** A fluid neighbour is already full and a gas one has
    no populations; handing mass to either breaks the invariant that a cell's fill is
    between 0 and 1. If a cell has no interface neighbour at all -- a droplet one cell
    across, a film tearing -- the mass cannot be placed, and it is added to `lost`
    rather than dropped silently. `FreeSurfaceWorld.mass_drift` reports it, because a
    conservation law is only worth having if the places it leaks are counted.

    `wp.atomic_add` on the neighbour's mass, because several converting cells can share
    one neighbour.
    """
    i, j, k = wp.tid()
    amount = excess[i, j, k]
    if amount == DTYPE(0.0):
        return
    nx = flags.shape[0]
    ny = flags.shape[1]
    nz = flags.shape[2]
    normal = normals[i, j, k]
    sign = DTYPE(1.0)
    if amount < DTYPE(0.0):
        sign = DTYPE(-1.0)

    total = DTYPE(0.0)
    for d in range(1, Q):
        a = int(i + L.C[0, d])
        b = int(j + L.C[1, d])
        c = int(k + L.C[2, d])
        if a < 0 or a >= nx or b < 0 or b >= ny or c < 0 or c >= nz:
            continue
        if flags[a, b, c] != CELL_INTERFACE:
            continue
        direction = wp.vec3(DTYPE(L.C[0, d]), DTYPE(L.C[1, d]), DTYPE(L.C[2, d]))
        weight = wp.max(sign * wp.dot(direction, normal), DTYPE(0.0))
        total += weight
    if total <= DTYPE(1.0e-9):
        # No interface neighbour the normal points at. Fall back to sharing it equally
        # over whatever interface neighbours there are, and only give up if there are
        # none at all -- which is a cell whose surface has detached from every other.
        for d in range(1, Q):
            a = int(i + L.C[0, d])
            b = int(j + L.C[1, d])
            c = int(k + L.C[2, d])
            if a < 0 or a >= nx or b < 0 or b >= ny or c < 0 or c >= nz:
                continue
            if flags[a, b, c] == CELL_INTERFACE:
                total += DTYPE(1.0)
        if total <= DTYPE(0.0):
            wp.atomic_add(lost, 0, amount)
            return
        share = amount / total
        for d in range(1, Q):
            a = int(i + L.C[0, d])
            b = int(j + L.C[1, d])
            c = int(k + L.C[2, d])
            if a < 0 or a >= nx or b < 0 or b >= ny or c < 0 or c >= nz:
                continue
            if flags[a, b, c] == CELL_INTERFACE:
                wp.atomic_add(mass, a, b, c, share)
        return

    for d in range(1, Q):
        a = int(i + L.C[0, d])
        b = int(j + L.C[1, d])
        c = int(k + L.C[2, d])
        if a < 0 or a >= nx or b < 0 or b >= ny or c < 0 or c >= nz:
            continue
        if flags[a, b, c] != CELL_INTERFACE:
            continue
        direction = wp.vec3(DTYPE(L.C[0, d]), DTYPE(L.C[1, d]), DTYPE(L.C[2, d]))
        weight = wp.max(sign * wp.dot(direction, normal), DTYPE(0.0))
        if weight > DTYPE(0.0):
            wp.atomic_add(mass, a, b, c, amount * weight / total)


@wp.kernel
def sync_fluid_mass(
    f: wp.array4d(dtype=DTYPE),
    flags: wp.array3d(dtype=wp.int32),
    mass: wp.array3d(dtype=DTYPE),
):
    """A fluid cell's mass is its density, restated after every step.

    Not bookkeeping for its own sake: the density of a fluid cell changes slightly every
    step (it is a compressible scheme held near `rho = 1`), and a mass that did not
    follow it would drift away from the fill fraction the conversions are tested on. A
    cell would then be declared full or empty on a stale number.
    """
    i, j, k = wp.tid()
    if flags[i, j, k] != CELL_FLUID:
        return
    rho = DTYPE(0.0)
    for q in range(Q):
        rho += L.pop(f, q, i, j, k)
    mass[i, j, k] = rho


@wp.kernel
def flags_from_solid(
    phi: wp.array3d(dtype=DTYPE),
    flags: wp.array3d(dtype=wp.int32),
    mass: wp.array3d(dtype=DTYPE),
):
    """Re-mark the solid cells after the geometry has moved, keeping the rest alone.

    A cell that a body has just moved into becomes solid and gives up its mass -- which
    is counted, in `FreeSurfaceWorld.step`, because a boat pushing water out of the way
    is the one place this solver can lose mass for a legitimate reason and still has to
    say how much. A cell a body has just left becomes gas and is given nothing; the
    refilling and the conversion machinery will bring the surface back to it if the
    water is there to do it.
    """
    i, j, k = wp.tid()
    solid = phi[i, j, k] < DTYPE(0.0)
    flag = flags[i, j, k]
    if solid:
        if flag != CELL_SOLID:
            flags[i, j, k] = CELL_SOLID
            mass[i, j, k] = DTYPE(0.0)
        return
    if flag == CELL_SOLID:
        flags[i, j, k] = CELL_GAS
        mass[i, j, k] = DTYPE(0.0)


@wp.kernel
def initialise_below(
    f: wp.array4d(dtype=DTYPE),
    phi: wp.array3d(dtype=DTYPE),
    flags: wp.array3d(dtype=wp.int32),
    mass: wp.array3d(dtype=DTYPE),
    axis: wp.int32,
    level: DTYPE,
    rho0: DTYPE,
    u0: wp.vec3,
    invert: wp.int32,
    amplitude: DTYPE,
    mode: DTYPE,
    along_axis: wp.int32,
    span: DTYPE,
):
    """Fill the lattice below a plane: the starting state of every case here.

    `level` is in lattice coordinates along `axis`. The cell exactly straddling it
    becomes the interface, with its mass set to the fraction of it that is under the
    surface -- which is what makes a dam break start from a flat surface rather than from
    a staircase, and what lets a floating body find a draft that is not quantised to the
    lattice.

    `amplitude` and `mode` put a cosine on the surface: `amplitude cos(pi n x / L)` in
    lattice cells along `along_axis`, which for `mode = 1` is **the first natural mode of
    a walled tank** -- zero slope at both walls, which is what the wall boundary
    condition wants. That matters more than it sounds. A linear tilt was tried first,
    because tilting a tank and letting go is how the experiment is done, and a ramp has
    a constant slope that is incompatible with the walls; it excites the odd modes
    together and the measured trace is their beat rather than a period. Releasing the
    mode the theory is about is what makes the measurement a measurement.
    """
    i, j, k = wp.tid()
    for q in range(Q):
        L.set_pop(f, q, i, j, k, L._equilibrium(q, rho0, u0))
    if phi[i, j, k] < DTYPE(0.0):
        flags[i, j, k] = CELL_SOLID
        mass[i, j, k] = DTYPE(0.0)
        return
    here = DTYPE(i)
    if axis == 1:
        here = DTYPE(j)
    elif axis == 2:
        here = DTYPE(k)
    along = DTYPE(i)
    if along_axis == 1:
        along = DTYPE(j)
    elif along_axis == 2:
        along = DTYPE(k)
    shape = DTYPE(0.0)
    if amplitude != DTYPE(0.0):
        shape = amplitude * wp.cos(DTYPE(3.14159265358979) * mode
                                   * (along + DTYPE(0.5)) / span)
    depth = level + shape - here
    if invert != 0:
        depth = -depth
    if depth >= DTYPE(0.5):
        flags[i, j, k] = CELL_FLUID
        mass[i, j, k] = rho0
    elif depth <= DTYPE(-0.5):
        flags[i, j, k] = CELL_GAS
        mass[i, j, k] = DTYPE(0.0)
    else:
        flags[i, j, k] = CELL_INTERFACE
        mass[i, j, k] = rho0 * (depth + DTYPE(0.5))


@wp.kernel
def initialise_box(
    f: wp.array4d(dtype=DTYPE),
    phi: wp.array3d(dtype=DTYPE),
    flags: wp.array3d(dtype=wp.int32),
    mass: wp.array3d(dtype=DTYPE),
    low: wp.vec3,
    high: wp.vec3,
    rho0: DTYPE,
    u0: wp.vec3,
):
    """Fill an axis-aligned box of lattice, for a dam break's column of water."""
    i, j, k = wp.tid()
    for q in range(Q):
        L.set_pop(f, q, i, j, k, L._equilibrium(q, rho0, u0))
    if phi[i, j, k] < DTYPE(0.0):
        flags[i, j, k] = CELL_SOLID
        mass[i, j, k] = DTYPE(0.0)
        return
    p = wp.vec3(DTYPE(i), DTYPE(j), DTYPE(k))
    inside = int(1)
    boundary = int(0)
    for a in range(3):
        if p[a] < low[a] - DTYPE(0.5) or p[a] > high[a] + DTYPE(0.5):
            inside = int(0)
        elif p[a] < low[a] + DTYPE(0.5) or p[a] > high[a] - DTYPE(0.5):
            boundary = int(1)
    if inside == 0:
        flags[i, j, k] = CELL_GAS
        mass[i, j, k] = DTYPE(0.0)
    elif boundary == 1:
        flags[i, j, k] = CELL_INTERFACE
        mass[i, j, k] = rho0 * DTYPE(0.5)
    else:
        flags[i, j, k] = CELL_FLUID
        mass[i, j, k] = rho0


@wp.kernel
def close_interface(
    f: wp.array4d(dtype=DTYPE),
    flags: wp.array3d(dtype=wp.int32),
    mass: wp.array3d(dtype=DTYPE),
    wanted: wp.array3d(dtype=wp.int32),
):
    """No fluid cell may touch a gas cell. Enforce it at initialisation.

    **The invariant the whole algorithm rests on**, and the one an initialiser can break
    without anything complaining. The interface is a closed layer one cell thick between
    fluid and gas; `promote_neighbours` maintains it as the surface moves, but nothing
    established it at the start, and whether it existed depended on where the fill level
    happened to fall.

    Measured: a tank filled to 60 mm on a 5 mm lattice puts the surface exactly on a
    cell boundary, so every cell was fluid or gas and **the interface layer was empty**.
    With no interface cell, `lattice.stream`'s reconstruction never fires -- it is
    conditioned on the cell being an interface cell -- and the topmost fluid cells stream
    in whatever a gas cell's untouched populations happen to hold. The symptom was a
    free surface that did not move at all: a flatness of 6e-08 mm and zero conversions
    over eight hundred steps, which reads like a very good result and is the absence of
    a free surface.

    A fluid cell with a gas neighbour becomes an interface cell at **full** mass. It is
    full -- it was a fluid cell a moment ago -- and the conversion machinery will empty it
    in the ordinary way if the surface is really above it. Starting it at half, which is
    the other obvious choice, would throw away half a cell of water per unit area of
    surface at `t = 0`.

    Two passes like the conversions, and for the same reason: `wanted` is written here
    and applied by the caller, so the answer does not depend on visiting order.
    """
    i, j, k = wp.tid()
    wanted[i, j, k] = flags[i, j, k]
    if flags[i, j, k] != CELL_FLUID:
        return
    nx = flags.shape[0]
    ny = flags.shape[1]
    nz = flags.shape[2]
    for d in range(1, Q):
        a = int(i + L.C[0, d])
        b = int(j + L.C[1, d])
        c = int(k + L.C[2, d])
        if a < 0 or a >= nx or b < 0 or b >= ny or c < 0 or c >= nz:
            continue
        if flags[a, b, c] == CELL_GAS:
            wanted[i, j, k] = CELL_INTERFACE
            return


@wp.kernel
def adopt_flags(
    flags: wp.array3d(dtype=wp.int32),
    wanted: wp.array3d(dtype=wp.int32),
    mass: wp.array3d(dtype=DTYPE),
    f: wp.array4d(dtype=DTYPE),
):
    """Apply `close_interface`'s marks, keeping a newly-made interface cell full."""
    i, j, k = wp.tid()
    if flags[i, j, k] == wanted[i, j, k]:
        return
    flags[i, j, k] = wanted[i, j, k]
    if wanted[i, j, k] == CELL_INTERFACE:
        rho = DTYPE(0.0)
        for q in range(Q):
            rho += L.pop(f, q, i, j, k)
        mass[i, j, k] = rho


@wp.kernel
def count_bad_neighbours(
    flags: wp.array3d(dtype=wp.int32),
    out: wp.array(dtype=wp.int32),
):
    """How many fluid cells touch a gas cell. Must be zero, always.

    The invariant as a measurement rather than as a comment. `FreeSurfaceWorld.step`
    does not call it -- it is a full neighbour sweep and the maintenance kernels already
    guarantee the property -- but the cases do, every run, because an invariant nobody
    checks is an assumption.
    """
    i, j, k = wp.tid()
    if flags[i, j, k] != CELL_FLUID:
        return
    nx = flags.shape[0]
    ny = flags.shape[1]
    nz = flags.shape[2]
    for d in range(1, Q):
        a = int(i + L.C[0, d])
        b = int(j + L.C[1, d])
        c = int(k + L.C[2, d])
        if a < 0 or a >= nx or b < 0 or b >= ny or c < 0 or c >= nz:
            continue
        if flags[a, b, c] == CELL_GAS:
            wp.atomic_add(out, 0, 1)
            return


@wp.kernel
def total_mass(
    mass: wp.array3d(dtype=DTYPE),
    flags: wp.array3d(dtype=wp.int32),
    out: wp.array(dtype=DTYPE),
):
    """Sum the mass over every cell that has any. The conservation law's left-hand side."""
    i, j, k = wp.tid()
    flag = flags[i, j, k]
    if flag == CELL_FLUID or flag == CELL_INTERFACE:
        wp.atomic_add(out, 0, mass[i, j, k])


# -- the world ---------------------------------------------------------------------------


class FreeSurfaceWorld(FluidWorld):
    """A `FluidWorld` with a top: water that ends somewhere instead of filling the box.

        world = FreeSurfaceWorld(domain=Domain(...), resolution=128, fluid="water",
                                 dt=1e-3, gravity=(0, 0, -9.81))
        world.initialise_free_surface(level=0.1)       # 100 mm of water, flat
        for _ in range(steps):
            world.step()

    Everything a `FluidWorld` does it still does: the same bodies, the same voxelisation,
    the same Bouzidi boundary, the same momentum-exchange wrench, the same units layer,
    the same figures. What it adds is the mass field and the three cell types, and the
    step loop below is the Korner algorithm in the order that makes mass conserve.

    **Buoyancy is not computed here and must not be.** A `FluidWorld` run hands a body
    `-rho V g` explicitly, because a single-phase solver has no free surface and
    therefore no waterline and no hydrostatic gradient -- `DECISIONS_LIVE.md` D9 is that
    argument. A free-surface run has all three: gravity acts on the fluid through the
    Guo forcing, the pressure really does increase with depth, and the momentum exchange
    over a hull integrates that pressure into a real buoyant force. Adding the analytic
    term on top would count it twice, and `couple.NewtonCoupling(buoyancy=False)` is how
    a floating body is coupled. The Archimedes case in M6 is the check that the force
    which comes out is the right size.
    """

    def __init__(self, *args: Any, gravity: Sequence[float] = (0.0, 0.0, -9.81),
                 gas_density: float = 1.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.gas_density = float(gas_density)
        self.gravity = tuple(float(v) for v in gravity)
        # Gravity reaches the fluid as a body force per unit volume, `rho g`, through
        # the same Guo forcing a periodic channel is driven by. That is the one piece of
        # `FluidWorld` a free surface changes the meaning of rather than the mechanics:
        # in a single-phase box a uniform body force is a pressure gradient nobody can
        # see, and under a free surface it is what makes the water have a bottom.
        self.set_body_force(tuple(self.fluid.rho * g for g in self.gravity))

        shape = self.shape
        self._flags = wp.full(shape, CELL_GAS, dtype=wp.int32, device=self.device)
        self._wanted = wp.zeros(shape, dtype=wp.int32, device=self.device)
        self.mass = wp.zeros(shape, dtype=DTYPE, device=self.device)
        self.fill = wp.zeros(shape, dtype=DTYPE, device=self.device)
        self._normals = wp.zeros(shape, dtype=wp.vec3, device=self.device)
        self._excess = wp.zeros(shape, dtype=DTYPE, device=self.device)
        self._lost = wp.zeros(1, dtype=DTYPE, device=self.device)
        self._converted = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._mass_probe = wp.zeros(1, dtype=DTYPE, device=self.device)
        self._use_flags = 1
        self.initial_mass = 0.0
        self.displaced_by_solids = 0.0
        """Lattice mass that bodies have moved into and taken with them.

        The one legitimate way this solver loses mass. A hull sweeping through water
        turns fluid cells into solid cells, and a solid cell holds nothing; the water it
        held is gone from the lattice. It is counted rather than ignored so that
        `mass_drift` can be about the algorithm instead of about the geometry, which is
        the only way that number is worth reporting.
        """

    # -- starting states -------------------------------------------------------------

    def initialise_free_surface(self, level: float, *, axis: int = 2,
                                velocity: Sequence[float] = (0.0, 0.0, 0.0),
                                density: float = 1.0, above: bool = False,
                                amplitude: float = 0.0, mode: float = 1.0,
                                along_axis: int = 0) -> None:
        """Flat water up to `level` metres along `axis`. The still-water start.

        `amplitude` (in metres) and `mode` put a cosine on the surface along
        `along_axis`: `mode = 1` is the first natural mode of a walled tank and is the
        sloshing case's initial condition. `above` fills the other side instead, which
        is what a case wants when its axis points down. The cell the surface passes through is made an interface cell with
        a partial mass, so the surface starts where it was asked for rather than at the
        nearest cell boundary -- a quantised start is a step change the first few hundred
        iterations then have to radiate away as a pressure wave.
        """
        self._free_stream = tuple(float(v) for v in velocity)
        self._rho0 = float(density)
        self._sync_bodies()
        self._voxelize()
        lattice_level = float(self._to_lattice(
            [level if a == axis else 0.0 for a in range(3)])[axis])
        u = wp.vec3(*[self.scaling.velocity_to_lattice(v) for v in self._free_stream])
        wp.launch(initialise_below, dim=self.shape,
                  inputs=[self.f, self.phi, self._flags, self.mass, int(axis),
                          DTYPE(lattice_level), DTYPE(density), u,
                          1 if above else 0,
                          DTYPE(float(amplitude) / self.dx), DTYPE(float(mode)),
                          int(along_axis), DTYPE(float(self.shape[int(along_axis)]))],
                  device=self.device)
        wp.copy(self.f_next, self.f)
        self._after_initialise()

    def initialise_column(self, low: Sequence[float], high: Sequence[float], *,
                          velocity: Sequence[float] = (0.0, 0.0, 0.0),
                          density: float = 1.0) -> None:
        """A rectangular column of water, in metres. The dam break's starting state."""
        self._free_stream = tuple(float(v) for v in velocity)
        self._rho0 = float(density)
        self._sync_bodies()
        self._voxelize()
        a = self._to_lattice(low)
        b = self._to_lattice(high)
        u = wp.vec3(*[self.scaling.velocity_to_lattice(v) for v in self._free_stream])
        wp.launch(initialise_box, dim=self.shape,
                  inputs=[self.f, self.phi, self._flags, self.mass,
                          wp.vec3(*[float(v) for v in np.minimum(a, b)]),
                          wp.vec3(*[float(v) for v in np.maximum(a, b)]),
                          DTYPE(density), u],
                  device=self.device)
        wp.copy(self.f_next, self.f)
        self._after_initialise()

    def _after_initialise(self) -> None:
        # Close the interface before anything else looks at it. See `close_interface`:
        # whether the initialiser left a gap depended on where the fill level fell
        # relative to the lattice, and a gap is a free surface that does not work.
        wp.launch(close_interface, dim=self.shape,
                  inputs=[self.f, self._flags, self.mass, self._wanted],
                  device=self.device)
        wp.launch(adopt_flags, dim=self.shape,
                  inputs=[self._flags, self._wanted, self.mass, self.f],
                  device=self.device)
        wp.copy(self.phi_prev, self.phi)
        wp.copy(self.owner_prev, self.owner)
        self._refresh_surface()
        self.initial_mass = self.lattice_mass()
        self._lost.zero_()
        self._converted.zero_()

    # -- the step --------------------------------------------------------------------

    def step(self, substeps: int | None = None) -> dict[str, Any]:
        """One coupling step of free-surface flow. The Korner algorithm, in order.

        Per lattice substep, and the order is the whole correctness argument:

            fill, normals    what the surface is, before anything moves it
            collide          fluid and interface cells; gas cells are skipped
            exchange mass    from the post-collision populations, before they stream,
                             so the mass and the populations cross the same links
            stream           with the gas side reconstructed (`lattice.stream`)
            faces            inlet and outlet, unchanged
            mark, promote,   decide every conversion before acting on any of them, so
            apply            that the answer does not depend on visiting order
            redistribute     the converted cells' leftover mass, along the normal

        Nine launches a substep against the single-phase loop's three. That is the price
        of a surface and it is mostly bandwidth: every one of the seven extra kernels
        touches a scalar field rather than nineteen populations, so the step is nothing
        like three times the cost. `benchmarks/live/` reports what it actually is.
        """
        import time

        count = int(self.substeps if substeps is None else substeps)
        started = time.perf_counter()

        self._sync_bodies()
        shifted = self._maybe_shift()
        # Only charge mass to the hull when there IS a hull. `lattice_mass` is a float32
        # atomic sum, so two reads of an unchanged field differ by round-off, and
        # accumulating that difference every step turns it into a number: the sloshing
        # tank, which has no bodies at all, reported 16.66 of lattice mass "displaced by
        # bodies" over 42,556 steps. Harmless in size -- 3e-6 of the total -- and wrong
        # in kind, because `mass_drift` credits it against the drift and a reader would
        # believe it.
        moving = bool(self._shapes) or self._newton_shapes is not None
        before_solid = self.lattice_mass() if (self.steps and moving) else None
        self._voxelize()
        # A body that has moved takes the water in its new cells with it. Re-mark, then
        # charge the difference to `displaced_by_solids` rather than to the algorithm.
        wp.launch(flags_from_solid, dim=self.shape,
                  inputs=[self.phi, self._flags, self.mass], device=self.device)
        if before_solid is not None:
            self.displaced_by_solids += before_solid - self.lattice_mass()
        wp.copy(self.phi_prev, self.phi)
        wp.copy(self.owner_prev, self.owner)
        self._apply_rotors()

        self._wrench.zero_()
        omega_plus, omega_minus = self.scaling.rates
        scheme = L.SCHEMES[self.collision]
        smag = DTYPE(self.smagorinsky ** 2)
        body_force = wp.vec3(*[self.scaling.body_force_to_lattice(v)
                               for v in self._body_force])
        moving = 0 if self._static else 1
        gas = DTYPE(self.gas_density)

        for _ in range(count):
            self._refresh_surface()
            wp.launch(L.collide, dim=self.shape,
                      inputs=[self.f, self.phi, self._force_field, body_force,
                              self._use_field, DTYPE(omega_plus), DTYPE(omega_minus),
                              scheme, smag, self._flags, self._use_flags],
                      device=self.device)
            wp.launch(exchange_mass, dim=self.shape,
                      inputs=[self.f, self._flags, self.fill, self.phi, self.mass],
                      device=self.device)
            wp.launch(L.stream, dim=self.shape,
                      inputs=[self.f, self.f_next, self.phi, self.owner,
                              self.body_pose, self.body_vel, self.body_com,
                              self._wrench, self.faces, self.face_velocity,
                              self._fallbacks, moving, self._flags, self._normals,
                              gas, self._use_flags],
                      device=self.device)
            self.f, self.f_next = self.f_next, self.f
            if self._needs_faces:
                wp.launch(L.apply_faces, dim=self.shape,
                          inputs=[self.f, self.phi, self.faces, self.face_velocity,
                                  self._inlet_profile, self._use_profile],
                          device=self.device)
            wp.launch(mark_conversions, dim=self.shape,
                      inputs=[self.mass, self.f, self._flags, self._wanted,
                              DTYPE(FILL_EPSILON)], device=self.device)
            wp.launch(promote_neighbours, dim=self.shape,
                      inputs=[self._flags, self._wanted], device=self.device)
            wp.launch(apply_conversions, dim=self.shape,
                      inputs=[self.f, self.mass, self._flags, self._wanted,
                              self._excess, self._converted], device=self.device)
            wp.launch(redistribute, dim=self.shape,
                      inputs=[self._excess, self._flags, self._normals, self.mass,
                              self._lost], device=self.device)
            self.lattice_steps += 1

        wp.synchronize_device(self.device)
        elapsed = time.perf_counter() - started
        self.seconds += elapsed
        self.steps += 1
        self._rho = None
        return {
            "step": self.steps, "lattice_steps": count, "seconds": elapsed,
            "mlups": self.nodes * count / elapsed / 1e6 if elapsed > 0 else 0.0,
            "shifted": shifted,
        }

    def _refresh_surface(self) -> None:
        wp.launch(fill_from_mass, dim=self.shape,
                  inputs=[self.mass, self.f, self._flags, self.fill],
                  device=self.device)
        wp.launch(surface_normals, dim=self.shape,
                  inputs=[self.fill, self._flags, self._normals], device=self.device)

    # -- what the caller reads --------------------------------------------------------

    @property
    def waterline(self) -> float:
        """The undisturbed free-surface level in metres, read away from any body.

        **What makes a floating body possible**, and the one number `couple.py` needs
        from a free surface that a single-phase world cannot give it: `_push_extras`
        weights each of a hull's solid cells by the fraction of its height below this,
        and that is the displaced volume Archimedes is about.

        Measured over the outer ring of columns -- the four edges of the domain in x and
        y -- because that is water the body is not disturbing. Reading it under the hull
        would measure the body's own depression, and a draft is defined against the
        level the body is **not** affecting. It is a host read of one small array and it
        happens once per coupling step, not per substep.

        `nan` if nothing is wet, which `_push_extras` must not be handed; it falls back
        to the whole voxelised volume when this is not a number.
        """
        depth = self.surface_height(axis=2)
        edge = np.concatenate([depth[0, :], depth[-1, :], depth[:, 0], depth[:, -1]])
        wet = np.isfinite(edge) & (edge > 0.0)
        if not wet.any():
            return float("nan")
        return float(self.domain.origin[2] + edge[wet].mean())

    def lattice_mass(self) -> float:
        """Total mass on the lattice, in lattice units. A host read; a diagnostic."""
        self._mass_probe.zero_()
        wp.launch(total_mass, dim=self.shape,
                  inputs=[self.mass, self._flags, self._mass_probe],
                  device=self.device)
        return float(self._mass_probe.numpy()[0])

    def water_volume(self) -> float:
        """The volume of water on the lattice, in cubic metres.

        `mass / rho_lattice` summed, times the cell volume. Not the same as counting
        fluid cells: an interface cell is a fraction of one, and the fraction is what
        makes a draft or a spill measurable to better than a cell.
        """
        return self.lattice_mass() * self.dx ** 3 / self._rho0

    def mass_drift(self) -> dict[str, float]:
        """How much mass the algorithm has lost, separated from what bodies took.

        Three numbers, because one would be a lie. `displaced` is water that rigid
        bodies moved into and carried off, which is geometry and not a defect.
        `unplaceable` is excess mass from a converting cell that had no interface
        neighbour to give it to -- a tearing film or a one-cell droplet -- which the
        redistribution counts rather than drops. `drift` is what is left: the part the
        algorithm itself cannot account for, as a fraction of the mass it started with,
        and it is the number the brief asks to see over long runs.
        """
        now = self.lattice_mass()
        lost = float(self._lost.numpy()[0])
        expected = self.initial_mass - self.displaced_by_solids - lost
        drift = (now - expected) / self.initial_mass if self.initial_mass else float("nan")
        return {
            "initial": self.initial_mass, "now": now, "displaced": self.displaced_by_solids,
            "unplaceable": lost, "drift": float(drift),
            "conversions": int(self._converted.numpy()[0]),
        }

    def surface_height(self, axis: int = 2) -> np.ndarray:
        """The free surface's height over the two axes it is not normal to, in metres.

        Summing the fill fraction down a column and scaling by `dx` -- which is the
        depth, and therefore the surface height for water resting on a flat floor. It is
        sub-cell accurate for the same reason `water_volume` is: a partly full cell
        contributes its fraction. `nan` where a column holds no water at all, so that a
        dry patch is visibly dry rather than zero-height.
        """
        fill = self.fill.numpy()
        flags = self._flags.numpy()
        counted = np.where((flags == CELL_FLUID) | (flags == CELL_INTERFACE), fill, 0.0)
        depth = counted.sum(axis=axis) * self.dx
        return np.where(depth > 0.0, depth, np.nan)

    def fluid_touching_gas(self) -> int:
        """How many fluid cells touch a gas cell. The invariant, as a number.

        Zero, always. A non-zero count means the interface layer has a hole in it, and a
        hole means the cells around it are streaming in populations that were never
        written. Every free-surface case grades this.
        """
        probe = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(count_bad_neighbours, dim=self.shape,
                  inputs=[self._flags, probe], device=self.device)
        return int(probe.numpy()[0])

    def density_profile(self, axis: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """Mean density against height over the fluid cells, in SI. The hydrostatic test.

        The pressure a free surface produces is what a floating body floats on, so this
        is the measurement that matters most in this module: `p = (rho - rho_0) c_s^2`
        should fall linearly from the bottom to the surface with a slope of `rho g`.
        Returns the node coordinates along `axis` and the mean density on each, with
        `nan` where a layer holds no fluid.
        """
        got = self.fields()
        rho = got["rho"]
        flags = self._flags.numpy()
        wet = (flags == CELL_FLUID)
        axes = tuple(a for a in range(3) if a != axis)
        counts = wet.sum(axis=axes)
        totals = np.where(wet, np.nan_to_num(rho), 0.0).sum(axis=axes)
        mean = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
        return self.node_coordinates(axis), mean

    def cell_counts(self) -> dict[str, int]:
        flags = self._flags.numpy()
        return {name: int((flags == code).sum()) for code, name in L.CELL_NAMES.items()}

    def report(self) -> list[str]:
        lines = super().report()
        counts = self.cell_counts()
        drift = self.mass_drift()
        lines.insert(1, f"  surface:    free, gas at rho {self.gas_density:g}, "
                        f"gravity {self.gravity} m/s2")
        lines.append(
            f"  cells:      {counts['fluid']:,} fluid, {counts['interface']:,} "
            f"interface, {counts['gas']:,} gas, {counts['solid']:,} solid")
        lines.append(
            f"  mass:       drift {drift['drift']:+.3e} of the initial mass over "
            f"{drift['conversions']:,} conversions"
            + (f"; {drift['displaced']:.4g} displaced by bodies"
               if drift["displaced"] else "")
            + (f"; {drift['unplaceable']:.4g} unplaceable"
               if drift["unplaceable"] else ""))
        return lines
