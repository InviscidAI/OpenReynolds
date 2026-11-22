#!/usr/bin/env python3
"""A sphere released in water, two-way coupled to Newton, approaching terminal velocity.

    python examples/live/coupled_sphere.py

What it shows: the fluid deciding what a body does. Newton integrates the sphere,
`NewtonCoupling.step()` runs the lattice, pushes the fluid wrench and the buoyancy into
`state.body_f` and then steps the solver, and the fall speed that comes out is compared
with the speed at which Schiller-Naumann's drag balances the sphere's weight less the
weight of the water it displaces. Nothing about that balance is in the solver.

Skipped with a message if `newton` is not installed: the fluid half needs Warp only,
the coupling needs the physics engine as well.

What it costs: 40 x 40 x 80 = 128,000 nodes and 400 coupling steps took **37 s** on one
CPU core of the machine this was written on. A GPU runs the same kernels much faster.

**This is a release transient and not a terminal velocity, and the reason is a
timescale rather than a budget.** The sphere's own response time is about 14 ms here,
but the fluid's is `diameter^2 / nu = 0.25 s`, eighteen times longer: the drag cannot
reach its steady value until the flow around the sphere has, and for *any* particle
settling in a liquid that ordering holds, because the ratio of the two is about
`18 / (rho_sphere/rho_fluid)`. So the run below reaches **50.8% of the
Schiller-Naumann terminal velocity after 37 ms and is still accelerating**, which the
printed history shows rather than hides. Terminal velocity costs thousands of coupling
steps on a lattice this size, which is why `cases.run_falling_sphere` -- the graded
version, at 30 cells per diameter -- is a GPU case. What this file demonstrates is the
coupling working in the right direction with the right magnitudes: the sphere falls, the
fluid pushes back, the force grows as the flow establishes.

Why the other numbers are these numbers, because none of them is free.

* **The terminal Reynolds number has to be low enough to resolve at 8 cells per
  diameter.** `nu_lattice = u_lattice * cells_per_diameter / Re`, so a faster sphere
  drives `tau_plus` towards the 1/2 at which the collision stops relaxing, and
  `units.Scaling` refuses the run rather than solving a stiff one. Re 17 here gives
  `tau_plus = 0.572`, comfortable.
* **The density ratio is 1.5 and not the graded case's 1.005**, because at half a per
  cent excess density the net driving force is the same size as the spurious fluid force
  a coarse boundary treatment produces. Measured on a 4 mm sphere at 8 cells per
  diameter, held still: a net weight-less-buoyancy of 1.6e-06 N against a fluid wrench
  fluctuating around 1e-06 N. At 50% excess density the driving force is clear of it.
  (The voxelised volume itself is not the problem at this resolution: it comes out
  6.543e-11 m3 against 6.545e-11 analytic, 0.03% low, and the run prints both.)
* **Re 17 at a 50% excess density means a 0.5 mm sphere**, because the terminal Reynolds
  number goes as `diameter^1.5`. The lattice does not care about the absolute size -- a
  sand grain settling in water is a real problem -- but the timestep does: `dt` comes
  out at 0.094 ms, below a robot's. A validation whose timestep is a microsecond proves
  nothing about a solver that has to run at a millisecond, and this one is nearer that
  end than `falling_sphere` is.

The 5-diameter box also adds drag that an unconfined sphere does not have. The graded
answer is `openreynolds live run falling_sphere`.
"""

import math

from openreynolds.live.units import FLUIDS
from openreynolds.toolbox.references import sphere_drag

DIAMETER = 0.0005
DENSITY_RATIO = 1.5
CELLS_PER_DIAMETER = 8
U_LATTICE = 0.05
BOX = (5.0, 5.0, 10.0)  # in diameters; the sphere starts at 70% of the height
STEPS = 400
GRAVITY = 9.81


def terminal_velocity(water, radius: float) -> float:
    """The speed at which Schiller-Naumann's drag balances weight less buoyancy.

    Cd falls with Re and Re rises with the speed, so the fixed point converges from
    below. Started from the Cd = 1 guess, as `cases.py` does.
    """
    volume = 4.0 / 3.0 * math.pi * radius ** 3
    area = math.pi * radius ** 2
    excess = (DENSITY_RATIO - 1.0) * water.rho * volume * GRAVITY
    speed = math.sqrt(2.0 * excess / (water.rho * area))
    for _ in range(80):
        cd = sphere_drag.schiller_naumann_cd(max(speed * 2.0 * radius / water.nu, 1e-6))
        speed = math.sqrt(2.0 * excess / (water.rho * area * cd))
    return speed


def main() -> int:
    try:
        import newton
    except ImportError as exc:
        print("this example needs the Newton physics engine, which the [live] extra "
              f"declares:\n    pip install 'openreynolds[live]'\n(the import said: {exc})")
        return 0
    import warp as wp

    from openreynolds.live import Domain, FluidWorld, NewtonCoupling, Scaling

    water = FLUIDS["water"]
    radius = DIAMETER / 2.0
    expected = terminal_velocity(water, radius)
    reynolds = expected * DIAMETER / water.nu

    dx = DIAMETER / CELLS_PER_DIAMETER
    scaling = Scaling.for_reynolds(dx, DIAMETER, reynolds, "water",
                                   u_lattice=U_LATTICE)
    size = tuple(side * DIAMETER for side in BOX)
    domain = Domain(origin=(0.0, 0.0, 0.0), size=size,
                    faces={"-x": "slip", "+x": "slip", "-y": "slip", "+y": "slip",
                           "-z": "outlet", "+z": "outlet"})
    world = FluidWorld(domain=domain, dx=dx, fluid="water", scaling=scaling,
                       substeps=1)

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    start = (size[0] / 2.0, size[1] / 2.0, size[2] * 0.7)
    # `add_body` already makes the free joint; `add_joint_free` as well gives it two.
    body = builder.add_body(xform=wp.transform(wp.vec3(*start), wp.quat_identity()),
                            label="sphere")
    shape = builder.default_shape_cfg.copy()
    shape.density = DENSITY_RATIO * water.rho
    shape.has_shape_collision = False
    builder.add_shape_sphere(body, radius=radius, cfg=shape)
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, -GRAVITY))

    world.initialise()
    engine = newton.solvers.SolverMuJoCo(model, disable_contacts=True, solver="newton",
                                         integrator="implicitfast", iterations=10)
    link = NewtonCoupling(model, engine, world, dt=scaling.dt, buoyancy=True)
    print("\n".join(link.report()))
    print(f"  expected:   {expected * 1e3:.4f} mm/s at Re {reynolds:.2f}, "
          f"Cd {sphere_drag.schiller_naumann_cd(reynolds):.4f}")

    speed = 0.0
    for taken in range(1, STEPS + 1):
        link.step()
        if taken % max(1, STEPS // 8) == 0:
            speed = -float(link.state.body_qd.numpy()[body][2])
            force, _torque = world.wrench(body)
            print(f"  {taken:>5,}/{STEPS:,} steps  t {taken * scaling.dt * 1e3:6.2f} ms"
                  f"  fall {speed * 1e3:7.4f} mm/s ({speed / expected:5.1%} of terminal)"
                  f"  fluid force {force[2]:+.3e} N")

    print(f"\n  fall speed {speed * 1e3:.4f} mm/s against a Schiller-Naumann terminal "
          f"{expected * 1e3:.4f} mm/s ({speed / expected - 1:+.1%})")
    print(f"  displaced volume, voxelised: {link.displaced_volume[body]:.4g} m3 against "
          f"{4.0 / 3.0 * math.pi * radius ** 3:.4g} m3 analytic")
    print(f"  lattice still finite and positive: {world.stable()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
