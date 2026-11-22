# Live sync: a GPU lattice Boltzmann fluid in lockstep with Newton

A robot simulator integrates rigid bodies. It does not know what the air or the water
around them is doing, so a quadrotor in most simulators does not feel its own downwash,
a drone near a wall does not feel the wall, and an underwater vehicle's drag is a
coefficient somebody typed in. The `[live]` extra puts a real fluid in that loop: a
D3Q19 lattice Boltzmann solver written in NVIDIA Warp, stepped N times inside every
Newton step, handing each body the force and the torque the flow actually produced.

    pip install 'openreynolds[live]'

    from openreynolds.live import Domain, FluidWorld, NewtonCoupling

Everything below is in `openreynolds/live/`. `ARCHITECTURE_LIVE.md` is where it plugs
into the rest of OpenReynolds and what it reuses; `DECISIONS_LIVE.md` is every choice
taken while building it, with the evidence.

## What this is not

**It is not a replacement for the OpenFOAM path, and that path is unchanged.** Nothing
in `backend/`, `tools.py`'s `bash` and `job_start`, the templates or the toolbox moved
for this. An OpenReynolds study still provisions a workspace, still meshes with
`snappyHexMesh` or `cfMesh`, still runs `simpleFoam` or `pimpleFoam` on the workspace's
cores, and is still the thing to use when the answer has to be defensible.

| use | reach for |
|---|---|
| a steady external aerodynamics answer you will put in a report | the OpenFOAM path: body-fitted mesh, a turbulence model, mesh convergence, `benchmarks/` |
| a transonic aerofoil, a 20-million-cell hull, heat transfer, combustion, a multiphase free surface | the OpenFOAM path. None of them is in this extension and none is planned |
| a robot that has to feel the fluid **while it is being controlled**, at a 1 to 5 ms timestep, in the same process as the physics engine | this extension |
| a flow in an awkward passage at Re in the hundreds, on a machine with nothing installed | `openreynolds/toolbox/lbm.py`, the D2Q9 numpy solver that reaches every workspace with no image and no GPU |
| **a 10^9-cell static aerodynamics run** | **Autodesk XLB**, by name. It has a multi-resolution grid, out-of-core arrays, a KBC collision and a distributed multi-GPU path; this extension has none of those and will not have them soon. `DECISIONS_LIVE.md` D1 is the full evaluation, including the three reasons it is not a dependency here: JAX is a hard requirement of it and Warp is optional, its force operator copies to the host and returns one aggregate vector with no torque and no split by body, and its boundary data is organised around a mask built from a fixed mesh rather than around N bodies whose poses change every step |

The dividing line is not accuracy, it is what the solver is for. The OpenFOAM path
answers one question carefully. This one answers a cheaper question 500 times a second
while a controller is in the loop.

## Install

    pip install 'openreynolds[live]'

**Core OpenReynolds installs and imports without it, and that is an acceptance
criterion rather than a nicety.** The extra pulls Warp, Newton and a MuJoCo backend,
which is a CUDA toolchain and a robotics simulator; most people who install
OpenReynolds will never want them. So `import openreynolds.live` imports nothing heavy
(every public name resolves through a lazy `__getattr__`), `openreynolds --help` works
on a machine with no GPU, and `openreynolds live` without the extra prints the one
command that fixes it and exits 1. `tests/test_live_optional.py` checks all of that by
hiding Warp from the import system.

The tested dependency range, which `pyproject.toml` and
`openreynolds/live/units.py::WARP_TESTED` both carry and `tests/test_live_versions.py`
asserts cannot drift apart:

| dependency | declared | measured on |
|---|---|---|
| `warp-lang` | `>=1.17,<1.20` | 1.19.0 and 1.17.0 |
| `newton` | `>=1.6,<1.9` | 1.8.0 and 1.6.1 |
| `mujoco-warp` | `>=3.12` | 4.0.1, and only if you choose `SolverMuJoCo` |

The lower ends are not arbitrary: **Isaac Lab 3.0.0 pins `warp-lang==1.17.0` and
`newton[sim]==1.6.1`**, read out of its own metadata, so a user running this inside
Isaac Lab gets those whatever this package asks for. It works on them, with no shim:
the Re 100 cavity comes out bit-identical on 1.19.0 and 1.17.0 (same SHA-1 over the
saved velocity plane), every Newton field the adapter reads exists on 1.6.1 with the
same dtype, and `shape_scale`'s component packing is the same. The one real difference
is speed: the older Warp is several times slower on the CPU backend, from its own code
generation rather than from anything here.
[`live-version-matrix.md`](live-version-matrix.md) is that measurement in full,
including the sixteen Warp API probes and what was **not** tested (CUDA on either, warp
1.18, newton 1.7).

A run outside the tested range is not refused. There is nothing to refuse, so
`FluidWorld.report()` appends one line saying so instead, and a number copied out of a
report carries the fact that its dependencies were untested.

## Quickstart

A lid-driven cavity at Re 100 on a 24-cell lattice. It needs no files, no arguments and
no GPU, and it runs as written:

```python
from openreynolds.live import Domain, FluidWorld, Scaling

dx = 1.0 / 24
scaling = Scaling.for_reynolds(dx, 1.0, 100.0, "water", u_lattice=0.055)
lid = scaling.reference_velocity

# Three lattice planes, periodic in z: a z-invariant field stays z-invariant exactly
# under D3Q19, so this is the two-dimensional answer computed by the 3D kernels.
domain = Domain(origin=(0.0, 0.0, 0.0), size=(1.0, 1.0, 3 * dx),
                faces={"-x": "wall", "+x": "wall", "-y": "wall", "+y": "wall",
                       "-z": "periodic", "+z": "periodic"},
                face_velocity={"+y": (lid, 0.0, 0.0)})

world = FluidWorld(domain=domain, dx=dx, fluid="water", scaling=scaling, substeps=1)
world.initialise()
for _ in range(8000):
    world.step()

print("\n".join(world.report()))

# x = 0.5 falls between two nodes for an even cell count, so the centreline is the mean
# of the two lines straddling it rather than the nearest one.
u = world.fields()["u"][:, :, 1, 0] / lid
centreline = u[11:13, :].mean(axis=0)
print(f"u minimum on x = 0.5: {centreline.min():.5f}  "
      f"(Ghia, Ghia and Shin 1982: -0.21090)")
```

It prints `-0.20848` against Ghia's `-0.21090`, 1.1% low, which is 24 cells and not the
scheme: the same extraction at 32 cells is `-0.21135`, 0.21% out, and that number and
the RMS behind it are what `openreynolds live run cavity` grades.

**It takes about 8 s on one CPU core** -- 1,728 nodes and 8,000 steps, plus a second or
two for Warp to load its cached kernels -- and a GPU runs the same kernels much faster.
`openreynolds live bench` measures the rate on yours and names the device.

The three runnable examples in [`examples/live/`](../examples/live/) are the next step
up, and each says in its own docstring what it costs on a CPU: the graded cavity
against Ghia, a sphere in a uniform stream with its drag coefficient against
Schiller-Naumann, and a sphere released in water two-way coupled to Newton.

## The Python API

Everything outside `FluidWorld` is SI: metres, seconds, newtons, kilograms, radians.
Everything inside the kernels is lattice units. `units.Scaling` is the whole conversion
and `world.py` is the only boundary between the two.

### `Domain`

The box and what its six faces do.

| field | default | what it is |
|---|---|---|
| `origin` | required | the low corner of the box in metres, which is the plane a `wall` face puts its no-slip surface on. **Not** the position of node (0, 0, 0): a halfway bounce-back puts the wall half a cell outside the last fluid node, so node `i` sits at `origin + (i + 0.5) * dx`. Getting that half cell wrong is a `1/(2n)` error in every length in the problem and it looks exactly like a solver defect |
| `size` | required | wall-to-wall extent in metres, or the period on a periodic axis |
| `faces` | `"outlet"` | one rule for all six, or a dict keyed `-x`, `+x`, `-y`, `+y`, `-z`, `+z` |
| `face_velocity` | `{}` | wall velocities in m/s by face name. A moving wall is how the cavity's lid is expressed: it is a boundary condition, not a body, so it needs no shape, no voxelisation and no force accumulation |

The face rules:

| rule | what it does |
|---|---|
| `periodic` | wraps. A property of the **axis**, not of a face: periodic on one side and walled on the other is refused, because streaming would wrap one way and reflect the other and the symptom is a slow density drift that reads as a solver bug |
| `wall` | halfway bounce-back, with that face's `face_velocity` through the Ladd term |
| `slip` | specular reflection, which is a frictionless wall |
| `inlet` | writes the equilibrium at density 1 and the face velocity, or a per-node profile from `set_inlet_profile` |
| `outlet`, `open` | first-order extrapolation of the populations from one cell inside: the "do nothing" outflow, first-order accurate and unconditionally stable. Two names for one rule, because "open" is what people call it |

**A box of six outlets is refused.** An outflow imposes no density, so a weakly
compressible lattice with a momentum source in it and no reference anywhere has a free
mean density, and a free mean density is a divergence: measured at six outflows round a
hovering rotor, the whole lattice went non-finite inside fifteen coupling steps. At
least one face has to conserve what crosses it. For an open box the usual answer is
slip sides and an outflow on the axis the flow leaves by.

Also on `Domain`: `resolution(dx)` gives the node counts (at least 3 a side, so D3Q19
has a neighbour on every axis), and `spacing(cells)` the cell size that puts `cells`
cells along the longest axis.

### `FluidWorld`

Every constructor argument. `domain` and one of `resolution` / `dx` / `scaling` are the
only ones a caller must supply.

| argument | default | what it is |
|---|---|---|
| `model` | `None` | a `newton.Model`, or None. With one, its bodies and shapes become the world's and `NewtonCoupling` refreshes them every step; without one, bodies are added by hand, which is what the validation cases do |
| `domain` | required | the `Domain` above. There is no default box |
| `resolution` | `None` | cells along the **longest** domain axis |
| `fluid` | `"air"` | `"air"`, `"water"`, or a `units.Fluid` of your own. Both built-ins are 20 C values and carry their provenance in an `about` field, so a report can say which water it meant |
| `dx` | `None` | metres per cell. Wins if both this and `resolution` are given. One of the two is required: a lattice has no default resolution, because the whole trade-off between cost and accuracy lives in it |
| `dt` | `None` | the **coupling** step in seconds: the robot's step, the interval one `step()` advances the fluid by |
| `substeps` | `None` | lattice steps per coupling step. Without it, `Scaling.for_coupling` chooses the smallest count that keeps the peak lattice velocity at 0.05, and explains itself if it cannot |
| `reference_velocity` | `0.0` | m/s, the speed the Reynolds number and the Mach check are taken on |
| `peak_velocity` | `None` | m/s, the fastest motion expected anywhere (a rotor tip, a terminal velocity). Defaults to the reference speed; it is what the `u_lattice < 0.4` bound is checked against |
| `collision` | `"trt"` | `trt`, `bgk` or `regularized`. See the table below |
| `smagorinsky` | `0.0` | the Smagorinsky constant `C_s`; 0 disables the subgrid model, 0.1 to 0.17 is the usual range, and it is squared on the way into the kernel |
| `scaling` | `None` | a ready-made `units.Scaling`, which bypasses `dt` and the automatic substep choice. This is what the graded cases pass |
| `device` | `None` | a Warp device string, `cuda:0` or `cpu`. Defaults to `wp.get_preferred_device()` |
| `one_way` | `False` | the fluid feels the bodies and the bodies do not feel the fluid. The wrench is still computed, because a one-way run whose forces you cannot see is not a debugging tool; it only stops `couple.py` pushing it into Newton |
| `follow` | `None` | a body index for the moving window: the lattice shifts to keep that body near the box centre |
| `follow_slack` | `0.25` | how far, as a fraction of the box, the followed body may drift before the window shifts |
| `mesh_search_cells` | `4.0` | how far, in cells, `wp.mesh_query_point_sign_normal` searches for a mesh shape. A node further than this from the mesh is reported as far fluid, which is true and is also the only way to keep the query cheap |
| `far_cells` | `3.0` | the signed distance written where nothing is near, in cells. It only has to exceed one lattice spacing for the `q` arithmetic at a boundary to be exact; clamping stops a float32 subtraction carrying enormous numbers |

The three collision operators, which exist because the two halves of this extension
want different things (`DECISIONS_LIVE.md` D3):

| `collision` | what it is | where it is the default |
|---|---|---|
| `trt` | two relaxation times with `(tau_plus - 1/2)(tau_minus - 1/2) = 3/16`, the value that puts a straight no-slip wall exactly on the lattice link midpoint **at any viscosity**. `benchmarks/lbm/` grades that property at three viscosities a factor of twenty apart | every graded case, because the wall position is what a drag coefficient is most sensitive to |
| `regularized` | recomputes every population's non-equilibrium part from the deviatoric second moment alone before relaxing it, which annihilates the ghost modes that make an under-resolved high-Reynolds run diverge. It **loses** the magic-parameter wall property, and that is exactly why it is not the default for the cases with published answers | the robot demos, and anything with the subgrid model on |
| `bgk` | one relaxation time: the baseline, so that the difference the other two make stays measurable rather than asserted | nothing. It is there to be compared with |

#### Geometry in

| call | what it does |
|---|---|
| `add_body(position, orientation, *, com, velocity, label) -> int` | a body the world owns: pose and velocity in SI, and nobody refreshes them, which is what "static" means here. `orientation` is a quaternion `(x, y, z, w)`, Warp's and Newton's order; `velocity` is a six-vector, linear then angular, matching `body_qd`; `com` defaults to `position` |
| `set_body_state(body, position, orientation, velocity, com)` | move a world-owned body. SI in, lattice units on the device at the next step |
| `add_shape(kind, *, body=-1, position, orientation, scale, mesh) -> int` | a solid, in metres, in the frame of `body`, or in the world if `body < 0`. `position` is an offset **within the body's frame**, so a single-shape body leaves it at its default. A shape with `body < 0` is scenery: it is a wall and its force is not accumulated anywhere, which is what a ground plane wants |
| `add_rotor(body, centre, axis, radius, thickness=0.0) -> int` | an actuator disk on `body`, in the body's frame, in metres. `axis` points the way the rotor pushes the **vehicle**; the fluid gets the opposite. `thickness` defaults to two cells rather than to a physical number, because the disk is a numerical device: thinner than two cells and the momentum source falls between nodes and the delivered thrust depends on where the rotor happens to sit |
| `set_rotor_thrust(index, thrust)` | total thrust in newtons, applied from the next `step()`. Newton's third law in one call: the fluid gets `-thrust * axis` as a momentum source and the body is owed `+thrust * axis`, which `rotor_reaction` returns and `NewtonCoupling` adds. Neither half is hidden in a kernel |
| `set_inlet_profile(velocity)` | a per-node inflow over the `-x` face, shape `(ny, nz, 3)`, in m/s. The Schafer-Turek channel needs this and nothing else does: its inflow is a parabola, and a uniform inlet at the same mean speed is a different benchmark with a different answer |
| `set_body_force(force_per_volume)` | a uniform body force in N/m^3, which is how a periodic channel is driven. **Not gravity.** In a weakly compressible solver a hydrostatic gradient is a real density gradient and at these Mach numbers a large one, so gravity is left out of the fluid and a body gets its weight and its buoyancy from `NewtonCoupling` instead. `couple.py`'s header docstring is the argument in full |

The `scale` convention is the one `lattice.py`'s signed distance functions use and the
one `couple.py` maps Newton's `shape_scale` onto. It is the identity for every supported
type, which is the point: the SDFs were written to Newton's **measured** packing rather
than to a convention of their own, so there is nothing to convert and nothing to get
wrong.

| shape | `scale` | notes |
|---|---|---|
| `sphere` | `(radius, -, -)` | Newton stores `(radius, 0, 0)` and **not** `(r, r, r)`; its own comment in `builder.py` says otherwise and is wrong, on 1.6.1 and 1.8.0 alike |
| `box` | `(half_x, half_y, half_z)` | |
| `capsule` | `(radius, half_height, -)` | about the local z axis, which is Newton's convention for both capsules and cylinders |
| `cylinder` | `(radius, half_height, -)` | Newton's third component is a barrel radius for a rounded cylinder and is dropped here rather than silently misread |
| `plane` | ignored | the local `z = 0` half-space with `+z` outward: an infinite ground plane. A finite slab is a box and should be built as one |
| `mesh` | `(sx, sy, sz)` | a scaling applied to the mesh. `mesh=` takes a `wp.Mesh`, a `newton.Mesh`, or a `(vertices, indices)` pair |

#### Stepping, reading, reporting

| call | what it does |
|---|---|
| `initialise(velocity=(0,0,0), density=1.0)` | every node at equilibrium with a uniform state, `velocity` in m/s. It also voxelises once and copies the result into the previous-step fields, so the first `step()` finds no fresh nodes rather than refilling every node inside every body |
| `step(substeps=None)` | one coupling step, and returns what it cost: `{"step", "lattice_steps", "seconds", "mlups", "shifted"}`. In order: poses in, window shift if the followed body drifted, voxelise, refill fresh nodes, rotor force field, zero the wrench, then `substeps` times collide / stream / apply faces |
| `wrench(body=None, *, substeps=None)` | the **mean** fluid wrench over the last coupling step, in SI: `(force, torque)` as two 3-tuples for one body, or an `(n, 6)` array for all of them. Mean and not instantaneous, because that is the interval Newton is about to integrate over. A host read, and deliberately not in the step loop: `NewtonCoupling` pushes the wrench into Newton with a kernel and never copies it |
| `rotor_reaction(body)` | the six numbers the rotors on `body` owe it, force then torque. The arm is meant to be to the centre of mass, because `body_f` is documented as a wrench referenced there, and with a Newton model behind the world it is not: see the last entry under *Limitations* |
| `fields()` | `{"rho", "u", "solid"}` as numpy arrays in SI: `(nx, ny, nz)` in kg/m^3, `(nx, ny, nz, 3)` in m/s, and the solid mask. Solid nodes are `nan` in the first two, so a figure does not draw the inside of a body as stationary fluid. A host read and not cheap, 32 MB of velocity at two million nodes; cached until the next `step()` |
| `momentum()` | total fluid momentum in kg m/s, summed over the fluid nodes. The quantity the conservation check is made on: with every face periodic and no body force a closed lattice must hold it constant to round-off, and any drift is the boundary treatment leaking. `shifts` has to be zero for it to mean anything |
| `stable()` | whether the lattice still holds finite, positive densities at its fluid nodes, which is the only way this solver fails. A full host read of the density field, so it is for the long runs rather than every step |
| `node_coordinates(axis=None)` | the world coordinates of the lattice nodes in metres, per axis or all three. The half cell from `Domain.origin` lives here and nowhere else, so this is what a profile should be sampled against |
| `report()` | the world as a human reads it: device and Warp version, lattice size and memory, domain and faces, the scaling, the coupling, the measured MLUPS and real-time factor, the fresh-node count, the Bouzidi fallback count, and the shift count. It also appends the untested-dependency line, and the under-resolved line when the run is one |

Two counters in that report are worth reading rather than skipping.
**`fallbacks`** counts links whose second fluid node was also solid, where the Bouzidi
interpolation degrades to halfway bounce-back at `q = 1/2`: a body thinner than a cell
or a sharp edge, which is a resolution problem the caller needs to be told about rather
than a quiet loss of order. **`refilled`** counts lattice nodes that have left a body
and been given an equilibrium state at the extrapolated density.

### `NewtonCoupling`

    link = NewtonCoupling(model, solver, world)
    for _ in range(steps):
        link.step()

and after each step `link.state.body_q` is the new pose and `link.world.wrench(body)` is
the fluid wrench that produced it. One `step()` does, in order: poses from
`state.body_q` / `body_qd` into the lattice body table, `world.step()`,
`state.clear_forces()`, the fluid's surface wrench into `state.body_f`, the buoyancy,
the rotors' reaction, `solver.step(state_0, state_1, control, contacts, dt)`, swap.

| argument | default | what it is |
|---|---|---|
| `model` | required | a finalised `newton.Model`. A model with no bodies is refused by name |
| `solver` | required | any Newton solver. `SolverMuJoCo` is what the demos use and `SolverXPBD` works too; the loop only ever calls `solver.step` with five positionals, which is why it works on newton 1.6.1 and 1.8.0 alike |
| `world` | required | the `FluidWorld`. Its body table is resized to Newton's and written by a kernel from here on |
| `state_0`, `state_1` | `model.state()` | the two states to ping-pong between |
| `control` | `model.control()` | Newton's control object, which is where `joint_f` lives |
| `contacts` | `None` | passed straight through, positionally |
| `one_way` | the world's | the fluid wrench is computed and not applied |
| `buoyancy` | `True` | add `-rho_fluid * V * g` to each body, with `V` the volume the **voxelisation** encloses rather than the volume the geometry has. The two differ (measured on a sphere at 8 cells per diameter, the voxelised volume is 0.03% below the analytic one), and a near-neutrally-buoyant body's terminal velocity is set by the difference between two numbers of nearly the same size, so using the analytic volume for the buoyancy and the voxelised surface for the drag does not balance and the error lands squarely on the quantity being graded. `False` for a body in air, where it is a part in a thousand, or for a caller applying it themselves |
| `dt` | the world's | the Newton step |

**Nothing in that loop touches the host** except one `body_count`-sized pose read, and
only when there are rotors or buoyancy to compose on the host. The poses in, the wrench
out and the voxelisation are single Warp kernel launches on arrays that already live on
the device; the obvious implementation, `numpy()` out and `assign()` back, is four PCIe
transfers per coupling step, which at a two-millisecond robot step is most of the step.

`link.report()` prints the model, the solver, the gravity and then the world's own
report. `link.displaced_volume` is the per-body voxelised volume in m^3, written each
step when buoyancy is on.

### `units.Scaling`

A lattice Boltzmann scheme has three free numbers and one of them is not free:

    dx      metres per lattice cell       chosen (it is the resolution)
    dt      seconds per lattice step      chosen
    nu_l    viscosity in lattice units    NOT chosen: nu_l = nu * dt / dx^2

and the whole art of setting up a run is that the third lands somewhere the scheme can
work, because `tau_plus = nu_l/c_s^2 + 1/2` and **both** ends of tau's range are
failures. `Scaling` is that conversion and its own verdict, built through one of four
classmethods rather than directly, because the useful question is never "what is dt":

| constructor | the question it answers | what falls out |
|---|---|---|
| `Scaling.for_reynolds(dx, length, reynolds, fluid="water", *, u_lattice=0.05, peak_factor=1.0, ...)` | "put Re on this length with this cell size" -- which is what a benchmark is specified by, and nothing else | `dt` and the reference speed. The fluid sets `dt` and nothing else here, because the Reynolds number has already absorbed it |
| `Scaling.from_velocity(dx, reference_velocity, fluid="air", *, u_lattice=0.05, ...)` | "I have a resolution and a flow speed" | `dt`, from the lattice velocity asked for |
| `Scaling.for_coupling(dx, dt_robot, fluid="air", *, substeps=None, reference_velocity=0.0, peak_velocity=None, scheme="regularized", les=True, u_lattice=0.05, max_substeps=4000, ...) -> (Scaling, int)` | "I have a resolution and the robot's timestep" | the **substep count**, which is the coupled case's one free number. Returns it beside the scaling, because deriving one from the other at three call sites is how they come to disagree |
| `Scaling.from_dt(dx, dt, fluid="air", *, reference_velocity=0.0, ...)` | "I have both and I know what I am doing" | nothing. The honest primitive; the other three call it |

`for_coupling` chooses N from the **velocity** ceiling and not from tau, and the
difference is physical rather than a matter of taste. `tau_plus` *decreases* with N, so
there is no N that makes a stiff tau comfortable: the largest `dt` available is
`dt_robot` itself, because the fluid has to arrive at the same instant the robot does,
and a finer `dx` makes tau worse. What N does control is the lattice velocity, so
`N = ceil(u_peak * dt_robot / (u_lattice * dx))`, the smallest N that puts the peak at
`u_lattice`, and the smallest N is the cheapest run.

Every constructor **refuses at construction** rather than advising at run time, and
every refusal names the knob that moves it (the convention `benchmarks/*/grade.py`
follows, for the same reason: a message that says what is wrong and not what to do
about it costs a round of guessing). `UnphysicalScaling` is the exception, and
`complaints()` returns a list rather than one message because a scaling that is both
stiff and supersonic should say both.

| refusal | the bound | why |
|---|---|---|
| lattice Mach number | `u_l / c_s < 0.1` | the scheme solves the compressible equations and is used for the incompressible ones; the error in that substitution is `O(Ma^2)`, so 0.1 is a 1% ceiling on it. Every published value this is graded against is an incompressible answer. Ma falls with `dt` at fixed `dx`, so the knobs are more substeps or a finer lattice |
| peak lattice velocity | `u_l_peak < 0.4` | `Ma < 0.1` is the stricter bound for a uniform flow, so this one is for a field whose peak is far above its reference speed, where the second-order equilibrium's own truncation stops being a small correction and starts being the answer. Checked against whatever peak the caller declares |
| stiffness | `tau_plus > 0.52` | at `tau -> 1/2` the collision stops relaxing anything. The number is `toolbox/lbm.py`'s own floor and its own wording, and it is where the trouble starts rather than a cliff: that script records a graded cavity converging at 0.538 and a 16-cell cavity at the same Reynolds number sitting at 0.5048 and not converging at all. `tau_plus` rises with `dt/dx^2`, so coarsen `dx` or take longer steps; **a finer lattice at a fixed Reynolds number makes this worse, not better** |

A fourth check, that `dx` and `dt` are positive and finite, runs first and short-circuits
the rest, because nothing below it is meaningful without them.

**The stiffness refusal does not apply to a large-eddy run, and that is the whole reason
the subgrid model is here.** A robot in air is the case: at `dx = 20 mm` and
`dt = 2 ms`, air's molecular viscosity gives `nu dt/dx^2 = 7.6e-05` and
`tau_plus = 0.50002`, and no knob fixes it. The resolved-viscosity run simply does not
exist at any resolution a real-time loop can afford, which is what it means to say a
flow is under-resolved. So a stiff tau with `les=True` is **reported** as an
under-resolved large-eddy run, in those words, in every report the run produces, and a
stiff tau without it is refused.

The rest of the module: `FLUIDS` is two fluids at one temperature each, and two rather
than twenty because a table of fluids is a table of numbers somebody has to be right
about; `Fluid(name, nu, rho, about)` is public so a caller with a third one passes it in
with its own provenance. `CS2`, `LAMBDA` and `STIFF_TAU` are `toolbox/lbm.py`'s
constants, copied rather than imported (the toolbox may not import this package) and
held equal by `tests/test_live_units.py`, because copied constants with no check between
them is how two files come to hold two different opinions about the speed of sound.

## The command line

    openreynolds live cases
    openreynolds live run <case|case.yaml>
    openreynolds live bench
    openreynolds live crosscheck

Nothing in the group imports anything under `openreynolds.live` at module scope, so
`openreynolds --help` and every other command still work without the extra. Each
subcommand exits **0 if every graded check passed and 1 if any did not**, so a script in
CI needs to parse nothing, and `--json` on any of them prints one object and nothing
else.

### `openreynolds live cases`

Lists what can be run, one line each: the case name, its milestone, what it measures,
and `[needs a GPU to be quick]` where the case is heavy. No options but `--json`.

| case | what it measures, and against what |
|---|---|
| `smoke` | every kernel once on a 12-cube. Not a validation; it is the cheapest thing that fails if the solver is broken, and it is in the main test suite |
| `poiseuille` | a force-driven periodic channel against the **exact** parabola, which the two-relaxation-time wall makes exact rather than second-order |
| `cavity` | the lid-driven cavity at Re 100 and Re 1000 against Ghia, Ghia and Shin (1982) Tables I and II, on both centrelines |
| `cylinder` | Schafer and Turek's 2D-2 cylinder at Re 100, at **three** resolutions, with the observed order of convergence on the drag error. One resolution that agrees with a published value is evidence that one grid agreed |
| `sphere` | drag on a fixed sphere at four Reynolds numbers against Schiller-Naumann |
| `crosscheck` | the same cylinder on both backends. See below |
| `translating_sphere` | a sphere moving through still fluid: the moving-boundary and fresh-node refill check, graded against Galilean invariance, which is exact |
| `falling_sphere` | a sphere released in water, two-way coupled to Newton, against the Schiller-Naumann terminal velocity. Needs Newton |
| `momentum` | a closed periodic box with one moving body: the momentum the fluid gained against the impulse the solver reported. A bounce-back that reflects the wrong population shows up here and not in a drag coefficient that is within 5% of a published fit |
| `ground_effect` | one actuator disk at five heights over a ground plane against Cheeseman and Bennett (1955). One disk and not four, because Sanchez-Cuevas, Heredia and Ollero (2017) measured that a complete multirotor does not follow that model |
| `quadrotor` | a quadrotor hovering on four actuator disks, Newton-coupled, with ground and wall effects. Graded for bookkeeping and stability, **not** against a published number. Needs Newton |
| `underwater` | a towed capsule in water, graded at zero length against the sphere it reduces to, and reported rather than graded for the elongated hull because no published drag coefficient for a capsule of arbitrary fineness ratio has been read for this repository |

### `openreynolds live run CASE`

`CASE` is a name from `live cases`, or a path to a case file ending `.yaml` or `.yml`.

| option | default | what it does |
|---|---|---|
| `--resolution N` | the case's own | cells across the domain's longest side. The cost of a run is this cubed times the steps |
| `--steps N` | the case's own | lattice steps. The default is what the case's reference measurement needs; fewer answers sooner and measures a flow that has not settled |
| `--collision trt\|bgk\|regularized` | the case's own | the collision operator |
| `--fluid air\|water` | the case's own | the fluid, at 20 C |
| `--out DIR` | `.` | where the figures are meant to go. **It is not honoured today**: see below |
| `--device STR` | the best one here | a Warp device, `cuda:0` or `cpu` |
| `--json` | off | one JSON object instead of the report |

**`--out` is dropped on the way through, as of 2026-11-23.** `cli.py` passes it as
`out=` and `cases.run_case` takes `out_dir=`, so it lands in the `**overrides` that are
forwarded to the runner, where every runner's trailing `**_` swallows it; `out_dir`
keeps its default and the figures are written to the working directory. Measured with
`openreynolds live run cavity --resolution 16 --steps 200 --out <dir>`: both PNGs
appeared in the current directory and the named one stayed empty. `live crosscheck
--out` and the agent tool's study-renders directory go the same way. Calling
`cases.run_case(..., out_dir=...)` from Python does work, which is what the examples do.

**Only the flags you give are passed on.** A case's defaults are part of what it
validates against (the resolution Ghia's cavity needs in order to agree with Ghia is not
a taste), so an unset flag leaves the case's own number alone rather than substituting
one of the CLI's choosing. A laminar case with the subgrid model switched on is
**refused**, not warned about: an eddy viscosity in a laminar flow lowers the effective
Reynolds number, and a drag coefficient measured at a lower Reynolds number than the one
the case claims is closer to the published value for reasons that have nothing to do
with the solver being right.

What it prints is the case's own report and then one line per check, with the measured
value, the target and the verdict, in the layout `benchmarks/*/grade.py` already uses:

    # cavity: Re 100, 32 cells a side
      fluid:      water (nu 1.003e-06 m2/s, rho 998.2 kg/m3)
      lattice:    dx 0.03125 m   dt 17.1293 s   dx/dt 0.00182436 m/s
      viscosity:  nu_lattice 0.0176   tau_plus 0.55280   tau_minus 4.05114
      ...
      PASS  u on x = 0.5, RMS against Ghia Table I (15 interior points)   0.00217988   against <= 2% of the lid speed
      PASS  v on y = 0.5, RMS against Ghia Table II (15 interior points)  0.00262542   against <= 2% of the lid speed
            u minimum on x = 0.5                                          -0.211351   against -0.21090 (Ghia)

      2/2 graded checks passed
      figure:     live-cavity-re100-n32.png

### `openreynolds live bench`

The lattice update rate on this machine's device.

| option | default | what it does |
|---|---|---|
| `--resolution N` | 64 and 128 | cells across the longest side. **Repeatable**, and two of them in one go is the only way the rate's scaling is a measurement rather than an extrapolation |
| `--steps N` | 200 | lattice steps per resolution |
| `--collision trt\|bgk\|regularized` | `trt` | the operator to measure |
| `--json` | off | one JSON object |

It prints MLUPS and milliseconds a step per resolution, and names the device and the
Warp version, because a rate with no device beside it is not a number anybody can use.
The rate is deliberately not one number: a small lattice is launch-bound and a large one
is bandwidth-bound.

### `openreynolds live crosscheck`

The same cylinder on two backends, which is the most useful single number in the suite.
`tests/data/moving_mesh/viv_fixed/` is a finished OpenFOAM `pimpleFoam` run committed to
this repository (a fixed cylinder at Re 100 in water, D = 10 mm, U = 10 mm/s, 2.5%
blockage) with its force coefficients intact, and `benchmarks/moving_mesh/grade.py`
reads the Strouhal number and the mean drag straight out of those files. This runs the
same case here and reports the difference.

So it needs **no OpenFOAM install, no cluster and no network**, and if the OpenFOAM
numbers ever move the comparison moves with them: they are read through the existing
grader, not copied into a constant. And it is a comparison between two completely
independent discretisations of the same physics -- a finite-volume pressure-correction
solver on a body-fitted mesh, and a lattice Boltzmann scheme on a uniform grid with an
interpolated boundary, sharing no code, no mesh and no boundary treatment. Agreement
between those two is evidence of a kind that agreement with a published table is not.
It runs from a source checkout only: the wheel does not carry `benchmarks/` or the test
data.

Options: `--resolution`, `--steps`, `--out`, `--json`.

## The case file, and the backend selector

A small YAML file, read by `cases.run_case_file` and by `openreynolds live run
whatever.yaml`:

```yaml
case: cylinder
backend: live
resolution: 30
collision: trt
smagorinsky: 0
```

| `backend` | what happens |
|---|---|
| `live` | the case runs here, on this machine's Warp device |
| `openfoam` | **nothing runs.** It reports where the same case is on the OpenFOAM path and stops |

That asymmetry is deliberate and is logged as `DECISIONS_LIVE.md` D2. The OpenFOAM
backend runs in a workspace with its own image, its own mesher and its own
dictionaries, and a case file read on this side cannot conjure one; `openreynolds` with
no subcommand starts a study that can. So the selector picks the backend and `openfoam`
answers with the case rather than running it, and `openreynolds live crosscheck` is the
one place both backends' answers exist at once, which is what makes the selector useful
rather than decorative.

The parser is a tiny YAML subset, `key: value` and one level of `key:` block, and not
PyYAML: PyYAML is not a dependency of this package and this extra should not add one for
a dozen scalars. `benchmarks/felix/grade.py` reads Felix's own `case.yaml` the same way
and for the same reason. A case file that needs more than this is a Python script, which
is a perfectly good way to drive `FluidWorld`.

The backend selector is in the case file and **not** in `openreynolds/solvers.py`, which
keeps its two solvers. A study's `solver` decides which kind of workspace the hosted
service provisions, which tool list the model is given, which system prompt the loop
sends, and the routing request that picks a solver from your first message. Three of
those four live wholly or partly in the closed-source hosted service, and none of them
can serve a backend that runs on your own GPU in your own process. `DECISIONS_LIVE.md`
D2 has the full argument.

## The agent tool

The Reynolds agent can launch a demo from a plain-language request. Ask for one in a
session:

> "Show me what the ground effect does to a hovering rotor -- run the live demo and
> show me the figure."

and the model calls `live_sim` with `case="ground_effect"`, which runs the case, draws
its figures and hands back the same report `openreynolds live run` would have printed:
every check's measured value, its target, whether it passed, and the figure paths.
`case="list"` returns the catalogue and runs nothing. The other arguments are
`resolution`, `steps`, `collision`, `fluid`, and a free text `note` that is recorded
with the result and changes nothing, so a figure found later says what question it was
drawn for. The figures are meant to land in the study's own renders directory, beside
every other picture the study made, and today they do not: `live_tool.handler` passes
that directory as `out=`, which is the same drop described under `live run --out` above,
so they are written to the directory the session was started in.

**It runs on your machine, in the session's own process, and not in the study's
workspace.** Every other tool except `fetch` and `cad` runs a command in the workspace;
this one cannot. The workspace image has no Warp, no CUDA toolchain and no GPU, and
putting them there would charge the 1.2 GB-compressed image pull to every user who never
asks for a live run, which is the argument `toolbox/lbm.py` already makes about OpenLB
and Palabos with a CUDA runtime being bigger than either. `DECISIONS_LIVE.md` D6 is the
record.

It is **withheld** from the tool list rather than offered-and-refusing when the extra is
not importable, the way `cad` is withheld when there is no desk behind it: a tool that
can only answer "not installed" costs the model a call to find that out, every session,
on every machine without a GPU. The refusal text still exists and is still returned if a
call arrives anyway, because a replayed transcript or an MCP client can dispatch a name
the list never offered, and the answer to that should be the sentence naming the install
command and not a traceback.

The figures come back as paths and never as attachments. They are already on your
machine, and a tool result carrying a picture the model cannot act on is a megabyte in
every request after it.

## Isaac Lab

`openreynolds.live.isaac.HoverFluidEnv` is a quadrotor hover task with the fluid on:

    python -m openreynolds.live.isaac --steps 250

    from openreynolds.live.isaac import HoverFluidEnv, HoverFluidEnvCfg

    env = HoverFluidEnv(HoverFluidEnvCfg(fluid_on=True))
    obs, _ = env.reset(seed=0)
    for _ in range(250):
        obs, reward, terminated, truncated, extras = env.step(policy(obs["policy"]))

**It is not a `DirectRLEnv` subclass, and it cannot be.** `DirectRLEnvCfg` has five
fields with no default and one of them is `scene: InteractiveSceneCfg`, and an
`InteractiveScene` cannot be constructed without Isaac Sim's USD stage: it clones prims,
resolves USD paths and asks a `SimulationContext` for a physics view.
`DirectRLEnv.__init__` reaches the stage before it reaches anything this file could
substitute, so a class that inherited from it here would import, fail on construction,
and be a claim this repository cannot stand behind.

**What it is instead is an environment that conforms to that interface.** Every method
signature is the one `isaaclab.envs.direct_rl_env.DirectRLEnv` declares, checked against
the real package with `inspect.signature` whenever `isaaclab` is importable rather than
remembered. `step` returns the gymnasium >= 1.0 five-tuple with SAME_STEP autoreset,
`is_vector_env` is True, observations are a dict with a `"policy"` key, and every tensor
is a `torch.Tensor` on `self.device` with `terminated` and `truncated` as `torch.bool` of
shape `(num_envs,)`. So it drops into any gymnasium >= 1.0 training loop and into Isaac
Lab's own RL wrappers (`isaaclab_rl.rsl_rl`, `.skrl`, `.sb3`), which read
`unwrapped.num_envs`, `unwrapped.device`, `is_vector_env` and the five-tuple and never
touch the scene.

What that costs, from `isaac.py`'s own docstrings, unsoftened:

- **no USD scene**, so nothing here can be viewed in Omniverse, articulated from a URDF
  through `InteractiveScene`, or sensed with Isaac Sim's cameras, contact sensors or
  IMUs;
- **no `InteractiveSceneCfg` cloning**, so there is no single batched physics view across
  environments and therefore no multi-GPU scale-out;
- **no `cfg.events`, no `observation_noise_model`, no `action_noise_model`, no XR, no
  video recorders.** Those are `DirectRLEnvCfg` fields this file does not read, and
  `HoverFluidEnvCfg` does not carry them rather than carrying them and ignoring them.

The vehicle is a Crazyflie-class quadrotor built from primitives: one box and four
actuator disks. `newton/examples/assets/crazyflie.usd` was read and carries no
`RigidBodyAPI`, no mass and no joints, so `add_usd` yields 0 bodies and 12 visual-only
meshes, and voxelising 12 triangle meshes per coupling step would cost more than the
lattice they were voxelised onto. The mass, arm and rotor radius are the Crazyflie 2.x's
(30 g, 46 mm, 23 mm); the inertia is **derived** from them as four point masses at the
arm tips rather than quoted, because the measured Crazyflie values are a factor of two
below that and a constant nobody in this repository measured is the failure
`toolbox/references/sphere_drag.py` exists to prevent. The idealisation is stated and
the factor of two is stated.

`fluid_on=False` builds the same vehicle, the same reward, the same termination and the
same observation with no lattice at all, and it is not a mock: it is the
point-mass-plus-rigid-body task every quadrotor RL paper actually trains. Having both
behind one flag is what makes "the fluid changes the task" a number somebody can produce
rather than a claim.

The default resolution is 16 cells across a 0.3 m box, so `dx = 18.75 mm` and the 23 mm
rotor is 1.2 cells in radius. That resolves **nothing** about the rotor. What it does
resolve is the momentum the disks put into the box and the download that momentum puts
back on the airframe, which is the term that makes a quadrotor in ground effect behave
differently from a point mass, and it is chosen so that `main()` finishes in about half
a minute on a CPU. A real training run wants `resolution=96` or more (2 mm cells, 11
cells per rotor radius, the airframe's boundary layer actually on the lattice), which is
884,736 nodes and about 100 substeps a coupling step: a GPU's work and not a laptop's.

## Limitations

Blunt, because every one of these is a thing somebody will otherwise discover by
measuring it.

- **D3Q19 only.** D3Q27 is not shipped. It is a second instantiation of every kernel for
  42% more memory and no gain at the order the second-order equilibrium is accurate to
  (`lattice.py`'s `Q` says so where the constant is set).
- **Single precision throughout.** A population sits near its weight and the interesting
  part of it is five or six orders of magnitude smaller, which is the classic argument
  for double precision; it is not taken, because double precision doubles the memory and
  halves a bandwidth-bound step rate, which is the whole product. One consequence to
  know about: the residual of a converged run settles at a few times `1e-5` of the
  reference speed and stays there, so a convergence tolerance below that never trips and
  the run goes to `max_steps` looking like a failure to converge.
- **One lattice per `FluidWorld`.** There is no batched fluid, so `num_envs > 1` is N
  lattices and N times the work, not one wider kernel. That is the single largest thing
  standing between this and an RL training run at scale.
- **The voxeliser handles spheres, boxes, capsules, cylinders, planes and meshes, and
  nothing else.** No height fields, no ellipsoids, no cones, no Gaussian splats. A height
  field is a terrain and wants its own sampling kernel rather than a generic SDF; an
  ellipsoid has **no closed-form exact signed distance** and the usual formula is a bound
  rather than a distance, which matters because `q` is a distance *ratio*; a cone and a
  Gaussian splat are not shapes a robot's collision geometry usually contains. Each of
  them raises by name with the suggestion that fits, which is to convert it to a mesh --
  supported, and something `newton.ModelBuilder` can do.
- **The coupling is first-order explicit in time.** The body is held fixed across the
  substeps and the wrench Newton gets is the mean over the interval the fluid has just
  crossed. The alternative, iterating the pair to convergence within a step, is what a
  tightly coupled fluid-structure solver does, and `benchmarks/moving_mesh/` in this
  repository measured what it is worth there: a loosely coupled vortex-induced-vibration
  run reported a net fluid work per cycle of about **-10%** of the body's energy while
  that energy held constant, which is impossible, and the tightly coupled continuation
  measured **+0.1%**. It does not matter at the mass ratios a robot in air has (a
  quadrotor is a thousand times denser than the air it displaces, so the added-mass term
  the iteration exists to capture is a thousandth of its inertia). For a **neutrally
  buoyant body in water it matters a great deal**, and `one_way` is not the answer; the
  answer is a smaller `dt`.
- **The subgrid model is carrying the rotor cases, and their answers away from a wall
  are the model's.** A 120 mm rotor pushing 5 m/s of air is a Reynolds number near 80,000
  on the diameter, and there is no resolution at which a millisecond timestep resolves
  air's molecular viscosity. That is legitimate and it is stated in every report such a
  run produces, in those words. It also means `ground_effect` is graded on a *ratio* to
  the same disk in free air, which cancels everything about the actuator disk that is not
  the ground, and that `quadrotor` is graded for bookkeeping and stability rather than
  against a published number.
- **The moving window does not conserve momentum.** A shift is an integer-cell gather,
  so fluid leaves at one face and uniform free stream arrives at the other. The shift
  count is in `report()` for that reason: a momentum balance measured across a shift is
  a measurement of the window and not of the solver.
- **Gravity is not in the fluid**, deliberately, and a body in a heavy fluid gets its
  weight and its buoyancy from `NewtonCoupling` instead. A hydrostatic gradient in a
  weakly compressible solver is a real density gradient and at these Mach numbers a
  large one.
- **The Bouzidi interpolation has a fallback and it is counted.** Where the node on the
  far side of a wall along a link is also solid (a body thinner than one cell, a sharp
  edge) the scheme degrades to halfway bounce-back at `q = 1/2`. `report()` prints how
  often that happened, because a body that is mostly sub-cell is a resolution problem
  rather than a quiet loss of accuracy.
- **No immersed-boundary option.** Bouzidi is the shipped treatment because the forces
  this extension exists to produce are surface integrals and a staircased wall puts a
  `1/(2 dx)` error into every one of them. The immersed-boundary alternative is better
  for thin membranes and worse for drag on a bluff body; a swimmer with a thin fin is the
  case that would want it (`DECISIONS_LIVE.md` D4).
- **A rotor's reaction moment is wrong when a Newton model drives the world, as of
  2026-11-23.** `FluidWorld.rotor_reaction` takes the moment arm from the host body table
  that `add_body` fills, and `NewtonCoupling` writes the body *poses* into that table
  every step but leaves the centres of mass at zero, so the arm is measured from the
  domain's world origin rather than from the body. Measured on one body at
  `(1, 0, 1)` m carrying four disks on 0.18 m arms with 0.3 N of thrust across them: a
  pitch moment of 0.309 N m where the moment about the centre of mass is 0.009 N m. The
  error scales with how far the vehicle is from the origin and vanishes at the origin.
  `openreynolds.live.isaac` then adds the correct moment a second time through
  `control.joint_f`, on the belief that `rotor_reaction` carries force only, which it
  did until 2026-11-22. Until both are fixed, treat the attitude moment on a rotor
  vehicle as unusable with the fluid on. Nothing else in this list is a defect; this one
  is.

## Performance

The figure of merit is **MLUPS**: million lattice updates a second, nodes times steps
divided by seconds. It is the only portable number an LBM has, and it is not one number
for a solver -- a small lattice is launch-bound and a large one bandwidth-bound, so
`openreynolds live bench` reports it per resolution and names the device and the Warp
version beside it.

Two things in this extension are measured rather than assumed, and both are in
`report()`'s output so that no rate is ambiguous about what produced it. The **array
layout** (`soa` direction-major, `aos` node-major) is chosen from the device and can be
overridden with `OPENREYNOLDS_LIVE_LAYOUT`; `openreynolds live bench --resolution a
--resolution b` run with both layouts is the measurement that picks the default, and it
has to be made on the device in question, because the node-major layout is several times
faster on one CPU core and the two came out within a couple of per cent of each other on
both GPUs tested -- which is **not** what the coalescing argument predicts, and is the
kind of thing this repository would rather measure than argue. The **Warp version** is
there because 1.17.0 and 1.19.0 produce identical answers at several times the
difference in speed on the CPU backend.

**The measured rates live in [`PROGRESS_LIVE.md`](../PROGRESS_LIVE.md) and are not
copied here.** A number in two places is a number that will disagree with itself, and a
performance figure is exactly the kind that goes stale silently. Every rate in that file
names the device that produced it, which matters more than usual here: the extension was
written on a machine with no CUDA device at all (`nvidia-smi` absent, Warp reporting one
device, `cpu`), and the GPU numbers came later, from one-shot Modal jobs with pinned
dependency versions, through `benchmarks/live/modal_run.py`. That file is also where the
brief's 1000+ MLUPS target is settled, on the device it is a claim about.

What a device does not change is the physics. The same Warp kernels compile for CUDA
from the same source, and the validations are resolution-limited rather than
device-limited: a cavity that agrees with Ghia on the CPU agrees with Ghia on an H100.
The one caveat worth carrying is that the layout choice makes the GPU code path a
*different* generated kernel from the one measured on the CPU, so the bit-identical
result across Warp versions does not transfer to it.
[`benchmarks/live/README.md`](../benchmarks/live/README.md) is the case family in this
repository's benchmark format, and its grader **solves** rather than reading a finished
run.
