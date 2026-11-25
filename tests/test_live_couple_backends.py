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


def _one_free_body(gravity: float = 0.0):
    """One free body and a `SolverMuJoCo`, with the flags the floating cases use.

    No fluid: a pose write is a kinematics question and a `FluidWorld` would add a
    voxelisation and a second's compile to a test whose whole point is that it is cheap
    enough to run on every commit. Gravity is zero so that the expected position after a
    step is the position that was written, exactly, with no tolerance to argue about.
    """
    import newton
    import warp as wp

    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0),
                                               wp.quat_identity()), mass=1.0)
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 500.0
    cfg.has_shape_collision = False
    builder.add_shape_box(body, hx=0.02, hy=0.02, hz=0.02, cfg=cfg)
    model = builder.finalize(device="cpu")  # see the SIGSEGV note
    model.set_gravity((0.0, 0.0, -gravity))
    solver = newton.solvers.SolverMuJoCo(model, disable_contacts=True, solver="newton",
                                         integrator="implicitfast", iterations=10)
    return model, solver, body


def test_a_placed_body_is_where_it_was_put_after_the_solver_steps() -> None:
    """A pose write must survive a step. `place_body` does; `body_q` never did.

    **This is the test that was missing for weeks.** `sph_cases._float_rig` held a
    settling hull with `body_qd.zero_()` and `body_q.assign(held)` after each
    `link.step()`. `SolverMuJoCo` keeps its own `qpos` / `qvel`, integrates from those
    and does not re-read Newton's public arrays, so the hold was a lie told to an array:
    the engine's z drifted 0.060 -> 0.0055 m over fifty steps while the pinned array
    read 0.060000 throughout, the draft snapped to the engine's accumulated position on
    release, and all four floating-body cases reported drafts of 268,395 mm against a
    16 mm Archimedes target.

    Every part of that was visible in one step of one body with no fluid in it, which is
    what this is. The second assertion is the bug itself, kept as an assertion rather
    than a comment: if a future Newton starts honouring `body_q` as an input it will fail
    here, loudly, which is the right place to find that out.
    """
    pytest.importorskip("newton")
    import numpy as np

    from openreynolds.live.couple import place_body

    target = (0.05, 0.06, 0.07, 0.0, 0.0, 0.0, 1.0)
    model, solver, body = _one_free_body()
    state_0, state_1, control = model.state(), model.state(), model.control()

    place_body(model, state_0, body, target)
    before = np.asarray(state_0.body_q.numpy()[body])
    assert np.allclose(before[:3], target[:3], atol=1e-6), (
        f"place_body left body_q at {tuple(before[:3])} rather than {target[:3]}. "
        f"The eval_fk is what puts it there, and `NewtonCoupling._push_poses` drives "
        f"the fluid's body table and its boundary particles off body_q before the next "
        f"step has pulled it"
    )

    solver.step(state_0, state_1, control, None, 1e-3)
    after = np.asarray(state_1.body_q.numpy()[body])
    assert np.allclose(after[:3], target[:3], atol=1e-6), (
        f"the body was placed at {target[:3]} and the solver put it at "
        f"{tuple(after[:3])} one step later. With zero gravity and zero velocity it "
        f"must not have moved at all, so the pose did not reach the engine: a pose goes "
        f"through joint_q / joint_qd and forward kinematics, never through body_q"
    )

    # And the same write through `body_q`, which is what the bug did.
    model, solver, body = _one_free_body()
    state_0, state_1, control = model.state(), model.state(), model.control()
    poses = state_0.body_q.numpy()
    poses[body] = np.asarray(target, dtype=poses.dtype)
    state_0.body_q.assign(poses)
    state_0.body_qd.zero_()
    solver.step(state_0, state_1, control, None, 1e-3)
    ignored = np.asarray(state_1.body_q.numpy()[body])
    assert np.allclose(ignored[:3], (0.0, 0.0, 0.0), atol=1e-6), (
        f"a pose written to body_q alone reached the engine, putting the body at "
        f"{tuple(ignored[:3])} instead of leaving it at the origin it was built at. "
        f"That is a change in Newton's contract -- body_q has been a solver OUTPUT for "
        f"SolverMuJoCo -- so check `place_body` and the four sph_cases holds against it "
        f"before relaxing this"
    )


def test_a_hull_held_up_by_its_own_weight_in_body_f_does_not_move() -> None:
    """The whole Newton half of the floating coupling, in one assert and no fluid.

    `sph_static_float`'s draft was wrong while the hull sat still, and four things on the
    Newton side could do that: a `body_mass` Newton derived from the shape density rather
    than taking the case's figure, gravity arriving twice, a wrench applied at the body
    origin instead of the centre of mass, or `body_f` being angular-first so the force
    landed in a torque slot. All four were checked by hand; none of them was it. This is
    what checking them by hand should have left behind.

    The construction is the floating cases' own -- `_float_rig`'s builder calls, the same
    `SolverMuJoCo` flags -- with the fluid replaced by **exactly** the force Archimedes
    says it must supply at the equilibrium draft, `m g` upward. A hull in that force must
    not move and must not rotate, and each of the four faults breaks it visibly:

        body_mass wrong by a factor k   net a_z = (1/k - 1) g
        gravity applied twice           net a_z = -g
        body_f dropped by the solver    net a_z = -g
        body_f angular-first            net a_z = -g, plus a spin
        wrench at the wrong point       a torque, which for this hull is zero either way
                                        because body_com is (0, 0, 0) -- so this asserts
                                        the torque is zero rather than claiming to
                                        distinguish the two reference points

    Measured when this was written, 500 steps at the case's own dt: net a_z
    +0.0000 m/s2, dz 0.000 um, identity quaternion.

    **Water is 1000 kg/m3 here and that is deliberate**, so this test asserts against its
    own arithmetic and not against `units.FLUIDS`: a 64 x 64 x 32 mm box at 500 kg/m3
    weighs 642.9082 mN and `body_mass` gives 65.5360 g, six figures. `_float_rig` uses
    `FLUIDS["water"].rho`, which is **998.21**, so the case's own hull is 65.41869 g and
    641.7574 mN -- 0.18% lighter, and worth knowing because the prescribed-draft curve's
    measured 646.4 mN is divided by it. The Archimedes draft does not move with rho,
    since the body's density is defined as `ratio * rho_water`.
    """
    pytest.importorskip("newton")
    import newton
    import numpy as np
    import warp as wp

    gravity, rho, beam, height, ratio = 9.81, 1000.0, 0.064, 0.032, 0.5

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.096, 0.096, 0.044), wp.quat_identity()),
        label="hull")
    cfg = builder.default_shape_cfg.copy()
    cfg.density = ratio * rho
    cfg.has_shape_collision = False
    builder.add_shape_box(body, hx=beam / 2.0, hy=beam / 2.0, hz=height / 2.0, cfg=cfg)
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, -gravity))

    # (a) the mass is the case's figure and not something derived from a thickness
    intended = ratio * rho * beam * beam * height
    mass = float(model.body_mass.numpy()[body])
    assert mass == pytest.approx(intended, rel=1e-6), (
        f"Newton's body_mass is {mass:.9g} kg where the case's own arithmetic -- "
        f"{beam * 1e3:g} x {beam * 1e3:g} x {height * 1e3:g} mm at "
        f"{ratio * rho:g} kg/m3 -- says {intended:.9g}. A hull that weighs something "
        f"other than intended floats at a draft that is correct for the wrong hull"
    )
    com = np.asarray(model.body_com.numpy()[body], dtype=float)
    assert np.allclose(com, 0.0, atol=1e-9), (
        f"body_com is {tuple(com)} and not the origin, so the wrench's reference point "
        f"is no longer vacuous for this hull and the torque assert below is no longer "
        f"the whole of the (c) check"
    )

    solver = newton.solvers.SolverMuJoCo(model, disable_contacts=True, solver="newton",
                                         integrator="implicitfast", iterations=10)
    state_0, state_1, control = model.state(), model.state(), model.control()
    dt = 1.1423e-3  # the SPH timestep at the case's 2 mm particle radius
    steps = 200
    z0 = float(state_0.body_q.numpy()[body][2])
    for _ in range(steps):
        state_0.clear_forces()
        forces = state_0.body_f.numpy()
        forces[body] = np.array([0.0, 0.0, mass * gravity, 0.0, 0.0, 0.0],
                                dtype=forces.dtype)
        state_0.body_f.assign(forces)
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0

    pose = np.asarray(state_0.body_q.numpy()[body], dtype=float)
    drift = float(pose[2]) - z0
    elapsed = steps * dt
    accel = 2.0 * drift / elapsed ** 2
    assert abs(accel) < 1.0e-3, (
        f"a hull handed exactly its own weight upward in body_f accelerated at "
        f"{accel:+.4f} m/s2 over {steps} steps ({drift * 1e6:+.3f} um). It must float. "
        f"{-gravity:+.2f} means the wrench never reached the body or gravity arrived "
        f"twice; a clean fraction of {-gravity:+.2f} means body_mass is wrong by that "
        f"factor"
    )
    assert np.allclose(pose[3:7], (0.0, 0.0, 0.0, 1.0), atol=1e-6), (
        f"a purely linear body_f spun the hull to {tuple(pose[3:7])}. body_f is "
        f"linear-first with the torque in the last three components; a force in a "
        f"torque slot shows up here"
    )
