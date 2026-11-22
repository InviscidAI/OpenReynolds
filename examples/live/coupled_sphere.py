#!/usr/bin/env python3
"""A sphere released in water, two-way coupled to Newton, falling towards terminal velocity.

    python examples/live/coupled_sphere.py

What it shows: the fluid deciding what a body does. Newton integrates the sphere,
`NewtonCoupling.step()` runs the lattice, pushes the fluid wrench and the buoyancy into
`state.body_f` and then steps the solver, and the fall speed that comes out is compared
with the speed at which Schiller-Naumann's drag balances the sphere's weight less the
weight of the water it displaces. Nothing about that balance is in the solver.

Skipped with a message if `newton` is not installed: the fluid half needs Warp only,
the coupling needs the physics engine as well.

What it costs: 40 x 40 x 56 = 89,600 nodes and 400 coupling steps took **26 s** on one
CPU core of the machine this was written on. A GPU runs the same kernels much faster.

Why these numbers are these numbers. The density ratio is 1.005 -- a sphere half a per
cent heavier than water -- and that is not a decoration. The terminal Reynolds number
has to be low enough that the lattice resolves the viscosity at an affordable
resolution, and a denser sphere falls faster: twice the excess density here is a
terminal Re near 100, which at 8 cells per diameter puts `tau_plus` under
`units.STIFF_TAU` and is refused. `cases.run_falling_sphere` is the graded version of
this case and explains the same arithmetic at 30 cells per diameter.

What the number is worth. Buoyancy here is the **analytic** displaced volume, applied
as a reduced gravity with `buoyancy=False`, and that is a deliberate departure from the
graded case, which uses the volume the voxelisation encloses (`DECISIONS_LIVE.md` D9).
The reason is this resolution: at half a per cent excess density the net driving force
is the difference between two numbers that agree to 0.5%, so a voxelised volume 2% away
from the analytic one -- which 8 cells across a sphere is -- does not merely perturb the
answer, it reverses its sign. The graded case runs at 30 cells per diameter, where the
voxelised volume is the right thing to use and is used. At this size the honest thing is
to say which volume went in. 400 steps also reaches about 85% of terminal, not all of
it, and the printed history shows that rather than averaging it away.
"""

import math

from openreynolds.live.units import FLUIDS
from openreynolds.toolbox.references import sphere_drag

DIAMETER = 0.004
DENSITY_RATIO = 1.005
CELLS_PER_DIAMETER = 8
BOX = (5.0, 5.0, 7.0)
STEPS = 400
GRAVITY = 9.81


def terminal_velocity(water, radius: float) -> float:
    """The speed at which Schiller-Naumann's drag balances weight less buoyancy.

    Cd falls with Re and Re rises with the speed, so the fixed point converges from
    below in a handful of passes. Started from the Cd = 1 guess, as `cases.py` does.
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
    scaling = Scaling.for_reynolds(dx, DIAMETER, reynolds, "water", u_lattice=0.04)
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
    # Weight less buoyancy as a reduced gravity, with `buoyancy=False` below: see the
    # module docstring for why the analytic displaced volume and not the voxelised one.
    model.set_gravity((0.0, 0.0, -GRAVITY * (1.0 - 1.0 / DENSITY_RATIO)))

    world.initialise()
    link = NewtonCoupling(model, newton.solvers.SolverXPBD(model), world,
                          dt=scaling.dt, buoyancy=False)
    print("\n".join(link.report()))
    print(f"  expected:   {expected * 1e3:.4f} mm/s at Re {reynolds:.2f}, "
          f"Cd {sphere_drag.schiller_naumann_cd(reynolds):.4f}")

    speed = 0.0
    for taken in range(1, STEPS + 1):
        link.step()
        if taken % max(1, STEPS // 8) == 0:
            speed = -float(link.state.body_qd.numpy()[body][2])
            force, _torque = world.wrench(body)
            print(f"  {taken:>5,}/{STEPS:,} steps  t {taken * scaling.dt:6.3f} s  "
                  f"fall {speed * 1e3:7.4f} mm/s ({speed / expected:5.1%} of terminal)"
                  f"  fluid force {force[2]:+.3e} N")

    print(f"\n  fall speed {speed * 1e3:.4f} mm/s against a Schiller-Naumann terminal "
          f"{expected * 1e3:.4f} mm/s ({speed / expected - 1:+.1%})")
    print(f"  lattice still finite and positive: {world.stable()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
