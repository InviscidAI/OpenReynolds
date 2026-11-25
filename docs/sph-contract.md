# The SPH backend's interface, fixed up front

Written before the code so that the solver and its validation cases can be built at the
same time without guessing at each other. Anything here is a promise; anything not here
is the implementer's choice.

## Construction

```python
from openreynolds.live import FluidWorld

world = FluidWorld(domain=Domain(...), free_surface=True, ...)   # picks SPH
world = FluidWorld(domain=Domain(...), backend="sph", ...)       # or say so
world = FluidWorld(domain=Domain(...), backend="lbm", ...)       # single phase
```

`free_surface=True` selects SPH. `backend=` overrides. `backend="lbm"` with
`free_surface=True` raises, naming `freesurface.FROZEN`.

`SphWorld(domain, *, particle_radius, fluid="water", gravity=(0,0,-9.81),
viscosity=..., surface_tension=0.0, device=None, solver="dfsph")`.

`particle_radius` is the sizing knob, the way `dx` is for the lattice. Particle spacing
is `2 * particle_radius`; the smoothing length follows the implementer's choice and is
reported.

## Filling and bodies

| call | meaning |
| --- | --- |
| `fill_box(low, high, *, velocity=(0,0,0))` | fluid particles on a lattice inside an axis-aligned box, in metres |
| `fill_below(level, *, axis=2)` | fluid up to a level, the still-water start |
| `add_shape(kind, *, body=-1, position, orientation, scale, mesh=None)` | same signature as `FluidWorld.add_shape`; becomes Akinci boundary particles |
| `boundary_from_domain()` | boundary particles on the domain's wall faces |

## Stepping and reading

`step(substeps=None)` returns `{"step", "substeps", "seconds", "simulated", "particles",
"mlups"}`. `seconds` is wall clock; **`simulated` is the physical time the step advanced**,
and it is not optional: every period, front speed and decay rate a case measures is wrong
by one unknown factor without it, and all of them still look plausible. `world.dt` is the
same number per step and is part of the interface too. (`mlups` is particle updates per
second / 1e6, named for symmetry with the lattice backend; the report says so.)

`world.coupling_style` must be the string `"particles"`. `couple.py` branches on it: the
lattice solves in index space and converts shape scales, poses and the accumulated wrench
by the spacing, while SPH solves in metres and all three conversions are the identity.
Without it the coupling divides by a `dx` an `SphWorld` does not have.

`world.c0` is the numerical speed of sound, in m/s. A density error and a pressure error
cannot be connected without it, and both are graded.

| read | returns |
| --- | --- |
| `positions()` | `(n, 3)` metres |
| `velocities()` | `(n, 3)` m/s |
| `densities()` | `(n,)` kg/m3 |
| `pressures()` | `(n,)` Pa |
| `wrench(body)` | `(force, torque)` in SI, same convention as the lattice backend |
| `surface_height(axis=2, bins=None)` | depth per column in metres, `nan` where dry. **Bin the two horizontal axes independently** -- a square grid sized from one of them leaves most columns of a long thin tank dry, and a caller that zero-fills instead of skipping them then divides the depth by the aspect ratio and gets a believable wrong answer. `nan` means dry, never 0. |
| `volume(body)` | the sampled volume of a coupled body, m3 |
| `density_error()` | `{"max", "mean"}` as fractions of the rest density |
| `report()` | the lines the cases print |

`wrench` **must** have the lattice backend's sign convention: the force **on the body**.
`tests/test_live_cases.py` has the standing check that a submerged body feels `+rho V g`.

## Coupling

`NewtonCoupling(model, solver, world, dt=..., buoyancy=False)` must accept an `SphWorld`
unchanged. Boundary particles follow `body_q` on the device; the wrench goes back through
`body_f`. **No host round trips in the step loop** and no copies that Warp can alias:
the rule the lattice backend follows, for the same reason.

`buoyancy=False` is the default for SPH and should stay that way -- an SPH free surface
has a real waterline and real hydrostatic pressure, so buoyancy comes out of the
pressure field and adding `rho V g` on top would count it twice.

## What the cases may assume

- CPU (Warp's CPU backend) runs a few thousand particles in seconds, for the suite.
- `density_error()` is the health check every case reports.
- Nothing is graded with a widened tolerance, every number is reproducible from a
  script, and every recorded run carries its commit. Same rules as the lattice backend.
