"""The free surface's own tests: the invariants, and that it did not break single phase.

**This backend is frozen** (`freesurface.FROZEN`, `DECISIONS_LIVE.md` D15) and every
construction here passes `experimental=True`. These tests are what keeps the frozen code
honest about what it does and does not do, so they stay and they stay green; free-surface
liquid work has moved to the SPH backend.

`tests/test_live_cases.py` grades the catalogue. This file holds what is specific to
`openreynolds/live/freesurface.py` -- the three cell types, the mass bookkeeping, and the
one test that `DECISIONS_LIVE.md` D11 rests on: a free-surface run with every cell
flagged fluid has to reproduce the single-phase solver exactly, because they are the
same two kernels.

Everything here runs on the CPU in seconds on a lattice of a few thousand nodes. The
graded free-surface physics is on the GPU in `benchmarks/live/`.
"""

from __future__ import annotations

import numpy as np
import pytest

from openreynolds.live.units import CS, Scaling

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

warp = pytest.importorskip("warp", reason="the live backend is an optional extra")


@pytest.fixture(scope="module")
def tank():
    """A small still tank, built once: 60 mm of water in a 100 mm box on 20 cells."""
    from openreynolds.live.freesurface import FreeSurfaceWorld
    from openreynolds.live.world import Domain

    height, depth, cells = 0.1, 0.06, 20
    dx = height / cells
    scaling = Scaling.for_froude(dx, depth, "water", tau=0.53)
    domain = Domain(origin=(0.0, 0.0, 0.0), size=(height * 0.5, height * 0.5, height),
                    faces="wall")
    world = FreeSurfaceWorld(domain=domain, dx=dx, fluid="water", scaling=scaling,
                             substeps=1, device="cpu", experimental=True)
    world.initialise_free_surface(level=depth)
    return world, scaling, depth


def test_the_interface_is_a_closed_layer_wherever_the_level_falls():
    """No fluid cell may touch a gas cell, at any fill level. The core invariant.

    Swept across levels on purpose. A level that lands exactly on a cell boundary used
    to produce **no interface cells at all** -- every cell fluid or gas, the
    reconstruction in `stream` never firing because it is conditioned on the cell being
    an interface cell, and a free surface that could not move. `close_interface` is the
    fix and this is what would catch its removal.
    """
    from openreynolds.live.freesurface import FreeSurfaceWorld
    from openreynolds.live.world import Domain

    height, cells = 0.1, 20
    dx = height / cells
    for fraction in (0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.625, 0.7, 0.75, 0.8):
        depth = fraction * height
        scaling = Scaling.for_froude(dx, depth, "water", tau=0.53)
        domain = Domain(origin=(0.0, 0.0, 0.0),
                        size=(height * 0.25, height * 0.25, height), faces="wall")
        world = FreeSurfaceWorld(domain=domain, dx=dx, fluid="water", scaling=scaling,
                                 substeps=1, device="cpu", experimental=True)
        world.initialise_free_surface(level=depth)
        counts = world.cell_counts()
        assert world.fluid_touching_gas() == 0, (
            f"fill level {depth * 1e3:g} mm leaves a hole in the interface: {counts}")
        assert counts["interface"] > 0, (
            f"fill level {depth * 1e3:g} mm produced no interface cells at all, which "
            f"is a free surface that cannot move: {counts}")
        world.step()
        assert world.fluid_touching_gas() == 0, (
            f"the interface opened after one step at {depth * 1e3:g} mm")


def test_the_interface_stays_closed_while_the_surface_moves(tank):
    """The invariant under motion, which is what `promote_neighbours` is for."""
    world, _, _ = tank
    for _ in range(40):
        world.step()
        assert world.fluid_touching_gas() == 0


def test_a_partly_full_cell_counts_as_its_fraction(tank):
    """The volume on the lattice is the volume it was filled with, to a part in 10^5.

    This is what the interface cell's partial mass buys: a surface that starts where it
    was asked for rather than at the nearest cell boundary. Counting an interface cell
    as a whole cell, or as none, is up to a whole cell of error on every unit area of
    surface -- 5% on a 20-cell tank.
    """
    world, _, depth = tank
    exact = world.domain.size[0] * world.domain.size[1] * depth
    assert world.water_volume() == pytest.approx(exact, rel=1e-5)


def test_mass_conserves_while_the_surface_moves(tank):
    """Mass drift stays under a part in 10^4 over a few hundred steps.

    Conservation is by construction here -- every term in the exchange appears twice with
    opposite signs -- so a drift that grows is a sign that mass is leaking through a
    conversion rather than that the scheme is approximate.
    """
    world, _, _ = tank
    start = world.steps
    while world.steps < start + 200:
        world.step()
    drift = world.mass_drift()
    assert abs(drift["drift"]) < 1e-4, drift
    assert drift["unplaceable"] == 0.0, (
        f"excess mass had nowhere to go: {drift}. On a still tank every converting cell "
        f"has interface neighbours, so this is zero or something is tearing")


def test_the_hydrostatic_pressure_gradient_is_right(tank):
    """`d(rho)/dz = -rho g / c_s^2` over the fluid cells, to 10%.

    The measurement buoyancy depends on, so it is worth a unit test and not only a
    graded case: a floating body floats on this gradient, and a solver that gets a draft
    right while getting this wrong is getting it right by accident.
    """
    world, scaling, _ = tank
    while world.steps < 1500:
        world.step()
    z, rho = world.density_profile()
    usable = np.isfinite(rho)
    slope = float(np.polyfit(z[usable], rho[usable], 1)[0])
    sound = CS * scaling.dx / scaling.dt
    exact = -world.fluid.rho * 9.81 / (sound * sound)
    assert slope / exact == pytest.approx(1.0, abs=0.10), (
        f"measured {slope:+.4f} against an exact {exact:+.4f} kg/m4")


def test_still_water_stays_still(tank):
    """The surface stays flat, to a thousandth of the depth."""
    world, _, depth = tank
    surface = world.surface_height()
    assert np.nanstd(surface) / depth < 1e-2
    assert np.isfinite(np.nanmean(surface))
    assert world.stable()


def test_a_fully_flooded_free_surface_run_matches_the_single_phase_one():
    """DECISIONS_LIVE.md D11: one collision operator, one Bouzidi boundary, two paths.

    `lattice.collide` and `lattice.stream` take a flag field and an integer that says
    whether to read it, so that a single-phase run and a free-surface run share them.
    The claim that makes that safe is that the free-surface path reduces exactly to the
    single-phase one when there is no gas, and this is the test of it: the same driven
    channel run twice, once as a `FluidWorld` and once as a `FreeSurfaceWorld` whose
    every cell is flagged fluid, and the velocity fields have to agree to single
    precision.

    If this fails, the two paths have diverged and every validated single-phase number
    in `benchmarks/live/` has stopped being evidence about the free-surface solver.
    """
    from openreynolds.live.freesurface import FreeSurfaceWorld, sync_fluid_mass
    from openreynolds.live.lattice import CELL_FLUID
    from openreynolds.live.world import Domain, FluidWorld

    def build(cls, **extra):
        dx = 0.01 / 12
        scaling = Scaling.for_reynolds(dx, 0.01, 40.0, "water", u_lattice=0.05)
        domain = Domain(origin=(0.0, 0.0, 0.0), size=(0.02, 0.01, 0.01),
                        faces=("periodic", "periodic", "wall", "wall", "wall", "wall"))
        world = cls(domain=domain, dx=dx, fluid="water", scaling=scaling, substeps=1,
                    device="cpu", **extra)
        return world

    plain = build(FluidWorld)
    plain.set_body_force((0.0, 0.0, 0.0))
    plain.initialise(velocity=(0.02, 0.0, 0.0))
    plain._body_force = (50.0, 0.0, 0.0)

    # Gravity off, so that the only difference left between the two runs is the flag
    # field itself -- a free-surface run with gravity on would be a different problem.
    flooded = build(FreeSurfaceWorld, gravity=(0.0, 0.0, 0.0), experimental=True)
    flooded.initialise(velocity=(0.02, 0.0, 0.0))
    flooded._body_force = (50.0, 0.0, 0.0)
    flooded._flags.fill_(CELL_FLUID)
    warp.launch(sync_fluid_mass, dim=flooded.shape,
                inputs=[flooded.f, flooded._flags, flooded.mass], device="cpu")
    flooded.initial_mass = flooded.lattice_mass()

    for _ in range(30):
        plain.step()
        flooded.step()

    assert flooded.fluid_touching_gas() == 0
    assert flooded.cell_counts()["gas"] == 0, "the flooded run grew a gas cell"
    a, b = plain.fields(), flooded.fields()
    for key in ("u", "rho"):
        left, right = np.nan_to_num(a[key]), np.nan_to_num(b[key])
        scale = max(float(np.abs(left).max()), 1e-12)
        worst = float(np.abs(left - right).max()) / scale
        assert worst < 1e-5, (
            f"the free-surface path and the single-phase path disagree on {key} by "
            f"{worst:.3e} of the field's scale; DECISIONS_LIVE.md D11 says they are the "
            f"same two kernels, so they should not")


def test_the_froude_scaling_says_what_it_gave_up():
    """A free-surface run's report has to carry the Reynolds number it did not match.

    `units.Scaling.nu_lattice_override` is the argument: a free surface's timestep is set
    by gravity, water's viscosity at that timestep is a tau of 0.50002, so the lattice
    viscosity is chosen rather than derived and the Reynolds number comes out orders of
    magnitude low. That is defensible and it is not allowed to be silent.
    """
    scaling = Scaling.for_froude(0.1 / 30, 0.06, "water", tau=0.53)
    assert scaling.nu_lattice_override > 0.0
    assert scaling.viscosity_ratio > 100.0
    assert scaling.froude(0.06) == pytest.approx(1.0, abs=1e-9)
    assert scaling.mach <= 0.1
    text = " ".join(scaling.report())
    assert "chosen, not derived" in text, (
        "the report has to say the lattice viscosity was chosen: " + text)


def test_refining_a_froude_run_raises_the_reynolds_number_and_holds_the_mach():
    """The thing that is backwards from every other case here, held by a test.

    `Ma = sqrt(L g_lattice / dx) / c_s`, so refining at a fixed lattice gravity **raises**
    the Mach number -- the opposite of a single-phase case. `for_froude` therefore
    derives `g_lattice` from a Mach target, and the compensation is that the effective
    viscosity falls as `dx` so the Reynolds number rises as `1/dx`. Both halves are
    checked.

    This docstring used to say `dx^1.5`, while the assertion below has always demanded a
    factor of **2.0** per doubling -- which is `dx^-1`, not the 2.83 that `dx^-1.5` would
    give. The assertion was right and the prose was wrong; `units.py::for_froude` has the
    measurement (`dx^1.000000`, residual 3.4e-15 over seven resolutions).
    """
    runs = [Scaling.for_froude(0.1 / cells, 0.06, "water", tau=0.53)
            for cells in (30, 60, 120)]
    for scaling in runs:
        assert scaling.mach == pytest.approx(0.07, rel=0.02)
    achieved = [s.reynolds_effective(0.06) for s in runs]
    for lower, higher in zip(achieved, achieved[1:]):
        assert higher / lower == pytest.approx(2.0, rel=0.1), achieved


def test_a_fill_level_on_a_boundary_is_not_silently_empty():
    """The specific regression: 60 mm on a 5 mm lattice. Named because it shipped.

    `initialise_below` put the surface exactly on a cell boundary, every cell came out
    fluid or gas, and the run reported a flatness of 6e-08 mm and zero conversions over
    eight hundred steps. That reads like an excellent result and was the absence of a
    free surface.
    """
    from openreynolds.live.freesurface import FreeSurfaceWorld
    from openreynolds.live.world import Domain

    dx = 0.1 / 20
    scaling = Scaling.for_froude(dx, 0.06, "water", tau=0.53)
    domain = Domain(origin=(0.0, 0.0, 0.0), size=(0.05, 0.05, 0.1), faces="wall")
    world = FreeSurfaceWorld(domain=domain, dx=dx, fluid="water", scaling=scaling,
                             substeps=1, device="cpu", experimental=True)
    world.initialise_free_surface(level=0.06)
    assert world.cell_counts()["interface"] == 100, world.cell_counts()
    before = world.fill.numpy().copy()
    for _ in range(250):
        world.step()
    assert world.mass_drift()["conversions"] > 0, (
        "nothing converted in 250 steps, which is the symptom the missing interface "
        "layer produced")
    moved = float(np.abs(world.fill.numpy() - before).max())
    assert moved > 1e-6, (
        f"no cell's fill fraction changed at all (worst {moved:.2e}): the surface is "
        f"frozen, which is what a missing interface layer looks like")
