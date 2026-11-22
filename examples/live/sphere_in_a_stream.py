#!/usr/bin/env python3
"""A sphere held in a uniform stream, with its drag coefficient against Schiller-Naumann.

    python examples/live/sphere_in_a_stream.py

What it shows: the whole `FluidWorld` API, used directly and not through `cases.py` --
a `Scaling` built from a Reynolds number, a `Domain` with an inlet and slip sides, a
body, a shape on it, `step()`, `wrench()` and `report()`. If you are reading one file
to find out how to drive this solver from your own code, it is this one.

What it costs: 8 cells per diameter in a 6 x 4 x 4 diameter box is 49,152 nodes, and
the 1,440 lattice steps below took **67 s** on one CPU core of the machine this was
written on. A GPU runs the same kernels much faster; `openreynolds live bench` measures
the rate on yours.

What the number is worth. Eight cells across a sphere is coarse and a four-diameter box
is narrow, and both errors push the same way: a confined sphere has a higher drag than a
free one. Measured here at Re 20 the drag comes out **Cd 2.95 against Schiller-Naumann's
2.6095, +13%**, and the graded case is `openreynolds live run sphere`, which uses 20
diameters of run, 12 diameters of cross-section and reports the blockage beside the
answer rather than correcting for it. What this file demonstrates is that the force the
momentum exchange produces is the right size and sign at a resolution that finishes
while you watch -- not that the solver agrees with a drag curve to a few per cent.

Re 20 and not Re 100, and that is the scheme rather than this example: at a fixed
lattice velocity `nu_lattice = u_lattice * cells_per_diameter / Re`, so a higher
Reynolds number at this resolution drives `tau_plus` towards 1/2, and
`units.Scaling.for_reynolds` refuses the run rather than solving a stiff one. Re 100 at
`u_lattice = 0.05` needs at least 17 cells per diameter; the refusal says so.
"""

import math

from openreynolds.live import Domain, FluidWorld, Scaling
from openreynolds.toolbox.references import sphere_drag

DIAMETER = 0.01
REYNOLDS = 20.0
CELLS_PER_DIAMETER = 8
SPAN, CROSS = 6.0, 4.0
"""The box, in diameters: along the flow, and across it. Two diameters of upstream run
below, four of wake."""
U_LATTICE = 0.05
"""The reference speed in lattice units, which is what `dt` is chosen from. 0.05 keeps
the lattice Mach number at 0.0866, inside `units.MACH_MAX`'s 0.1 ceiling with room for
a peak above the free stream."""
CROSSINGS = 1.5
"""How many times the free stream crosses the box before the drag is read. The wake has
to arrive and settle, and this is the cheapest statement of it."""


def main() -> int:
    dx = DIAMETER / CELLS_PER_DIAMETER
    scaling = Scaling.for_reynolds(dx, DIAMETER, REYNOLDS, "water",
                                   u_lattice=U_LATTICE)
    u_inf = scaling.reference_velocity
    size = (SPAN * DIAMETER, CROSS * DIAMETER, CROSS * DIAMETER)
    domain = Domain(
        origin=(0.0, 0.0, 0.0), size=size,
        faces={"-x": "inlet", "+x": "outlet", "-y": "slip", "+y": "slip",
               "-z": "slip", "+z": "slip"},
        face_velocity={"-x": (u_inf, 0.0, 0.0)},
    )
    world = FluidWorld(domain=domain, dx=dx, fluid="water", scaling=scaling,
                       substeps=1)

    centre = (2.0 * DIAMETER, size[1] / 2.0, size[2] / 2.0)
    sphere = world.add_body(position=centre, label="sphere")
    # No `position=` on the shape: it is an offset in the body's own frame, and the body
    # is already at the sphere's centre.
    world.add_shape("sphere", body=sphere, scale=(DIAMETER / 2.0, 0.0, 0.0))
    world.initialise(velocity=(u_inf, 0.0, 0.0))
    print("\n".join(world.report()))

    area = math.pi * (DIAMETER / 2.0) ** 2
    expected = sphere_drag.schiller_naumann_cd(REYNOLDS)
    steps = int(CROSSINGS * size[0] / dx / U_LATTICE)
    drag = 0.0
    for taken in range(1, steps + 1):
        world.step()
        if taken % max(1, steps // 5) == 0 or taken == steps:
            force, _torque = world.wrench(sphere)
            drag = 2.0 * force[0] / (scaling.fluid.rho * u_inf * u_inf * area)
            print(f"  {taken:>6,}/{steps:,} steps   drag {force[0]:+.4e} N   "
                  f"Cd {drag:.4f}   against {expected:.4f} "
                  f"({drag / expected - 1:+.1%})")

    print(f"\n  Cd {drag:.4f} at Re {REYNOLDS:g}, {CELLS_PER_DIAMETER} cells per "
          f"diameter, {1.0 / CROSS:.0%} blockage")
    print(f"  Schiller-Naumann {expected:.4f}: {drag / expected - 1:+.1%}, and the "
          f"blockage and the resolution both raise it")
    print(f"  lattice still finite and positive: {world.stable()}")
    print("\n".join(world.report()[-2:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
