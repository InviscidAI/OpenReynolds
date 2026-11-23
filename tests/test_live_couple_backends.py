"""Every world attribute `couple.py` reads must exist on both backends.

**Why this test and not three bug fixes.** `NewtonCoupling` was written against the
lattice and grew five `world.dx` divisions, a `world.scaling.dt`, a `world.rotors` test
and a guard that asked for a `world.shapes` no `SphWorld` has. Each one surfaced as a
separate `AttributeError` from a separate GPU validation run -- four H100 runs died on
the `shapes` guard alone, before a single particle moved -- and each was then fixed on
its own. Discovering a class of fault one paid run at a time is the expensive way.

So this enumerates the attribute reads straight out of the source and checks them against
a real world of each kind. A new lattice-only assumption fails here, in under a second,
instead of in a validation run an hour later.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

COUPLE = Path(__file__).resolve().parents[1] / "openreynolds" / "live" / "couple.py"

# Reads that are legitimately lattice-only. Each is either behind the `_style` guard or
# on a branch a particle world cannot reach, and each is named here so that adding to
# this list is a deliberate act with a reason, rather than a silent exemption.
LATTICE_ONLY = {
    "dx": "behind _to_solver's style guard",
    "scaling": "behind _step_seconds' style guard",
    "owner": "the volumes kernel, on the buoyancy path, which refuses for particles",
    "shape": "the volumes kernel, same path",
    "_to_lattice": "lattice index conversion, same path",
    "rotor_reaction": "rotors, which a particle world does not have",
    # Read once on the common path and once inside the branch it guards. The branch
    # CONDITION (`if self.buoyancy or ... rotors`) is evaluated for every world and so
    # goes through `getattr`; the comprehension INSIDE the branch is only reached when
    # rotors exist, which no particle world has. Same attribute, two reachabilities, and
    # only the first one needed fixing.
    "rotors": "inside the rotors branch; the branch condition itself uses getattr",
}


def _attributes_read() -> set[str]:
    text = COUPLE.read_text(encoding="utf-8")
    found = set(re.findall(r"self\.world\.([A-Za-z_][A-Za-z0-9_]*)", text))
    # writes are the coupling handing something to the world, not a requirement of it
    written = set(re.findall(r"self\.world\.([A-Za-z_][A-Za-z0-9_]*)\s*=", text))
    return found - written


def test_the_coupling_reads_nothing_an_sph_world_lacks() -> None:
    sph = pytest.importorskip("openreynolds.live.sph")
    from openreynolds.live.world import Domain

    domain = Domain(origin=(0.0, 0.0, 0.0), size=(0.06, 0.06, 0.06), faces="wall")
    world = sph.SphWorld(domain=domain, particle_radius=0.005, fluid="water",
                         gravity=(0.0, 0.0, -9.81), surface_tension=0.0, device="cpu")
    world.boundary_from_domain()
    world.fill_below(0.03)

    missing = sorted(a for a in _attributes_read()
                     if a not in LATTICE_ONLY and not hasattr(world, a))
    assert not missing, (
        f"couple.py reads {missing} off the world, and an SphWorld has none of them. "
        f"Either the SPH backend should provide them, or they belong behind the "
        f"`_style` guard and listed in LATTICE_ONLY with the reason"
    )


def test_the_boundary_particle_guard_does_not_fire_on_a_valid_world() -> None:
    """The guard must refuse only a world it has established is empty.

    Its first version asked for `world.shapes`; `getattr` returned `None`, `not None` was
    true, and four validation runs died on correctly built worlds. A guard that cannot
    find the state it checks must fail open.
    """
    sph = pytest.importorskip("openreynolds.live.sph")
    from openreynolds.live.world import Domain

    domain = Domain(origin=(0.0, 0.0, 0.0), size=(0.06, 0.06, 0.06), faces="wall")
    world = sph.SphWorld(domain=domain, particle_radius=0.005, fluid="water",
                         gravity=(0.0, 0.0, -9.81), surface_tension=0.0, device="cpu")
    world.boundary_from_domain()
    world.fill_below(0.03)

    counted = (len(getattr(world, "_shapes", ()) or ())
               + len(getattr(world, "_bdy_x", ()) or ())
               + int(getattr(world, "n_boundary", 0) or 0))
    assert counted > 0, (
        "boundary_from_domain() ran, so the guard in NewtonCoupling must see boundary "
        "particles here; if it sees none it will refuse a valid world"
    )


def test_a_particle_body_lands_where_it_was_put_not_half_a_metre_away() -> None:
    """The device pose must be the world pose on the particle path.

    `poses_from_newton` subtracts half a cell, because lattice node `i` sits at
    `origin + (i+0.5) dx`. That offset used to be the constant `0.5`, and a particle
    backend works in SI with `inv_dx = 1`, so it subtracted **half a metre**: a hull
    placed at `(0.06, 0.06, 0.04)` arrived on the device at `(-0.44, -0.44, -0.46)`,
    500 mm outside a 120 mm tank.

    Its boundary particles then touched no fluid and every body read a wrench of
    **exactly zero**, which reads as a coupling that runs and finds no force rather than
    as a unit error -- so four GPU validation runs reported it as a physics result. The
    tell was available for the price of printing one array.
    """
    sph = pytest.importorskip("openreynolds.live.sph")
    pytest.importorskip("newton")
    import numpy as np

    from openreynolds.live.world import Domain
    sph_cases = pytest.importorskip("openreynolds.live.sph_cases")

    placed = (0.06, 0.06, 0.04)
    domain = Domain(origin=(0.0, 0.0, 0.0), size=(0.12, 0.12, 0.12), faces="wall")
    world = sph.SphWorld(domain=domain, particle_radius=0.005, fluid="water",
                         gravity=(0.0, 0.0, -9.81), surface_tension=0.0, device="cpu")
    world.boundary_from_domain()
    world.fill_below(0.08)
    box = sph_cases._Box(beam=0.032, length=0.032, height=0.032, density_ratio=1.0)
    sph_cases._float_rig(box, world, start_z=placed[2], solver="mujoco")

    on_device = np.asarray(world.body_pose.numpy())[0][:3]
    assert np.allclose(on_device, placed, atol=1e-4), (
        f"the hull was placed at {placed} and reached the device at {tuple(on_device)}. "
        f"A particle backend takes no node offset: it works in metres, so subtracting "
        f"half a cell with inv_dx = 1 subtracts half a metre"
    )
