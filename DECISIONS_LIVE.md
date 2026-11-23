# Decisions taken while building `openreynolds/live/`

Every choice made without being able to ask, what the alternatives were, and what the
evidence was. Dated, newest last. The brief for this work said: when a decision is
ambiguous, pick the most reasonable option, log it here, and keep going.

---

## D1. Write the LBM rather than depend on Autodesk XLB

**2026-11-22.** The brief asked for XLB to be evaluated before writing anything, and
for the decision to be logged. It was evaluated by downloading the wheel
(`xlb-0.3.2-py3-none-any.whl`, 200 kB) and reading it. The answer is: write our own,
and the reasons are specific rather than preferential.

**What XLB has, in its favour.** It is Apache-2.0, which is compatible with this
repository's MIT licence and with the brief's "only permissive dependencies". It is not
a toy: `xlb/velocity_set/` carries D2Q9, D3Q19 and D3Q27; `xlb/operator/collision/`
carries BGK, KBC and `smagorinsky_les_bgk.py`; `xlb/operator/boundary_condition/`
carries halfway and fullway bounce-back, Zou-He, regularised, extrapolation outflow and
a hybrid; there is a `momentum_transfer.py`, an `ibm_stepper.py`, a
`mesh_boundary_masker.py` with a voxelisation method enum, and a multi-resolution grid.
Its Warp backend is real, not a stub -- `_construct_warp` is implemented across the
operator tree. Had the three things below not been true, depending on it would have
been the right answer and this file would say so.

**Reason 1: JAX is not optional in XLB, and Warp is.** `xlb-0.3.2`'s core
`Requires-Dist` list is `matplotlib`, `numpy`, `pyvista`, `trimesh`, `numpy-stl`,
`pydantic`, **`ruff`**, **`jax>=0.8.2`** and `nvtx`; `warp-lang` appears only under
`Provides-Extra: warp`. 45 of its 101 Python modules import jax, and they do it at
module scope -- `xlb/operator/force/momentum_transfer.py` line 2 is
`import jax.numpy as jnp` -- so the Warp path imports JAX anyway. The brief's whole
framing of this extension is "an optional extra, so users who don't need it don't pull
in Warp/Newton". Adding JAX, a second array framework with its own GPU runtime, plus a
linter (`ruff`) as a *runtime* dependency, to an extra whose purpose is to stay
skippable, is the opposite of that.

**Reason 2: its force operator round-trips to the host, and returns one vector.**
`MomentumTransfer.warp_implementation` ends `return self.force.numpy()[0]`. That is a
device-to-host copy per call and a single aggregate 3-vector: no torque, and no split
by body. The brief's two hard requirements on the step loop are "no host round trips
inside the step loop" and "computing per-body force and torque on the GPU". Meeting
them through XLB means forking that operator, and then the dependency is carrying the
parts we would have to replace.

**Reason 3: the geometry is bound the wrong way round for a robot.** XLB's boundary
machinery is organised around a `bc_mask` built by a masker from a mesh, with
`boundary_condition_registry` ids -- a good fit for a fixed geometry and an awkward one
for N rigid bodies whose poses change every coupling step and whose identity has to
survive into the force reduction. The design this extension needs (one signed-distance
field plus one owner index, both rebuilt per coupling step, with the Bouzidi fraction
and the wall velocity derived in the kernel that needs them) is not a configuration of
XLB's masker; it is a different data layout.

**What was given up by deciding this.** XLB's multi-resolution grid, its KBC collision,
its out-of-core arrays and its distributed (multi-GPU) path are all real work that this
extension does not have and will not have soon. If a user needs a 10^9-cell static
aerodynamics run, XLB is the better tool and `docs/live-sync.md` says so by name.

**Reused instead of written:** the numerics come from this repository's own
`openreynolds/toolbox/lbm.py` -- two relaxation times, `LAMBDA = 3/16`, the Ladd
moving-wall term, the `STIFF_TAU` floor -- which is already validated in
`benchmarks/lbm/` at three viscosities. See D3.

---

## D2. `openreynolds/solvers.py` keeps two solvers; the backend selector lives in the case file

**2026-11-22.** The brief asks for "the same case/config format, with a backend
selector (e.g. backend: openfoam | live)". The obvious reading is to add `live` to
`solvers.CHOICES` beside `openfoam` and `felix`. That was rejected after reading what
that enum actually drives.

A study's `solver` decides: which **kind of workspace** the hosted service provisions
and binds the study to (`solvers.py`'s own docstring: "a workspace has one kind and a
study binds only to a workspace of its own kind"); which **tool list** the model is
given (`tools.py::tools_for`); which **system prompt** the loop sends; and the
**routing request** that picks a solver from the user's first message before the first
turn (`desk.route`). Three of those four live wholly or partly in the closed-source
hosted service. A third value would have to be answerable by a service that cannot
provision a GPU workspace for it, and `tools_for`'s docstring records what happens when
the tool list or the prompt moves under a running session: a 400 from Bedrock, and a
turn that ends with nothing said.

The live backend is also not that kind of thing. It runs on the machine the user is
sitting at, next to their GPU, in the same process as their Newton simulation. There is
no workspace to provision.

**So the selector is `backend:` in the live case file**, read by
`openreynolds/live/cases.py`, with `live` running here and `openfoam` answering with
the OpenFOAM case the same description corresponds to. That makes the selector decide
something real -- it is what `openreynolds live crosscheck` uses to put one case
through both backends -- rather than adding an unreachable third state to a study
record.

---

## D3. Two collision operators, with the default chosen per use rather than globally

**2026-11-22.** The brief asks for "a collision operator that stays stable at high
Reynolds number (regularized, MRT, or cumulant) plus Smagorinsky LES". Both halves of
that are shipped, and they are not the same operator, because they are good at
different things and this repository's style is to keep the contrast measurable.

- **`trt`** is `toolbox/lbm.py`'s two-relaxation-time scheme lifted to D3Q19, with the
  same `LAMBDA = 3/16`. That value puts a straight no-slip wall exactly on the lattice
  link midpoint at any viscosity, which `benchmarks/lbm/` already grades at three
  viscosities a factor of twenty apart. It is the default for the graded cases because
  the wall location is what a drag coefficient is most sensitive to.
- **`regularized`** recomputes the non-equilibrium part of every population from the
  deviatoric second moment alone before relaxing it, which annihilates the ghost modes
  that make an under-resolved high-Reynolds run diverge. It is the default for the
  robot demos. It loses the magic-parameter wall property -- the regularised
  non-equilibrium is even in `c`, so there is no odd part left for a second relaxation
  rate to act on -- and that is precisely why it is not the default for the cases with
  published answers.

Smagorinsky is orthogonal and available on both: the eddy viscosity is computed from
the same deviatoric second moment the regularisation already forms, so it adds one
square root per node and no extra pass.

A `--collision` flag selects it, and `benchmarks/live/grade.py` runs the cavity under
both, so "the regularised operator is worse at the wall" is a measurement in this
repository rather than a claim in this file.

---

## D4. Bouzidi interpolated bounce-back, with halfway bounce-back as the fallback

**2026-11-22.** The brief offered "interpolated bounce-back (Bouzidi) or immersed
boundary". Bouzidi, because the forces this extension exists to produce are surface
integrals, and a staircased wall puts a 1/(2*dx) error into every one of them -- which
at the resolutions a real-time run can afford is larger than every other error in the
calculation. The immersed-boundary alternative spreads the body over a kernel width and
is better for thin membranes and worse for drag on a bluff body; a swimmer with a thin
fin is the case that would want it, and it is not shipped.

The fallback matters and is deliberate: when the node on the far side of the wall along
a link is *also* solid (a body thinner than one lattice cell, or a sharp edge), the
interpolation has no second fluid node to read and the scheme degrades to halfway
bounce-back at `q = 1/2` rather than reading a garbage population. `FluidWorld.step`
counts how often that happened and the report prints it, because a body that is mostly
sub-cell is a resolution problem the user needs to be told about rather than a silent
loss of accuracy.

---

## D5. Fresh nodes are refilled with equilibrium at the extrapolated density

**2026-11-22.** When a body moves, lattice nodes that were inside it become fluid and
hold populations from no physical state at all. Three schemes are usual: equilibrium
refill, equilibrium plus a neighbour-averaged non-equilibrium part, and normal
extrapolation. Equilibrium refill is shipped: density averaged over the fluid
neighbours that existed last step, velocity taken as the body's own surface velocity at
that point, which is continuous with the boundary condition the node had a moment
earlier.

This is the cheapest of the three and the brief asked specifically for "correct fresh
node refilling", so the check is explicit rather than assumed: `benchmarks/live/` runs
a sphere translating at constant velocity through a quiescent lattice and reports the
spurious force the refilling produces, which is the quantity a bad refill ruins. A
scheme is only worth more than this if that number is too big.

---

## D6. The agent tool runs on the host, not in the workspace

**2026-11-22.** Every tool in `tools.py` except `fetch` and `cad` runs a command in the
study's workspace. `live_sim` cannot: the workspace image has no Warp, no CUDA
toolchain and no GPU, and putting them there would cost every user who pulls the image
(the argument `toolbox/lbm.py`'s docstring already makes about OpenLB and Palabos,
which applies more strongly to a CUDA runtime).

So `live_sim` is a host-side tool, in the shape `cad` already established: it is
offered only when it can actually do something -- `tools_for` withholds `cad` without a
desk behind it, and withholds `live_sim` without the `[live]` extra importable -- and
it writes its figures into the study's own directory so the existing mirror and capture
machinery picks them up with no change.

---

## D7. No GPU on the machine this was built on, so the GPU runs go to Modal

**2026-11-22, revised 2026-11-23.** `nvidia-smi` is not present on the development
machine and Warp reports exactly one device, `"cpu"`. Every number measured there is one
core of Warp's single-threaded CPU backend.

**What this decision said on 2026-11-22, and why it was wrong to leave it there.** It
said the brief's "1000+ MLUPS on an H100-class GPU" would be recorded as *not measured*
rather than estimated, on the grounds that a performance claim nobody ran is the one kind
of number this repository's `benchmarks/README.md` is written against. The first half of
that is right and stands. The second half was a decision not to try, and it was the wrong
one: this project has a Modal workspace, an H100 is a few lines of `modal.App` away, and
"we could not measure it" was never true -- only "we had not".

**What it says now.** `benchmarks/live/modal_run.py` runs the cases on a real GPU, and
every GPU number in `PROGRESS_LIVE.md` comes from one. The runner is built so that it
cannot become an expense nobody is watching: one-shot `@app.function`s with explicit
timeouts, no `@app.cls`, no warm pool, no `min_containers`, no volume. A job starts, runs
one case, returns its result and its figures as bytes, and the container exits.

The target is met: **1478.7 MLUPS** on an H100 80GB at 384 cells, and 1095.2 at 128 --
which is the useful half, because the target is reached at a lattice a real-time loop can
afford rather than only at one that fills the card.

**What the GPU found that the CPU could not.** Three things, and they justify the runs on
their own:

- `NewtonCoupling` called `np.asarray(model.gravity)`, which Warp permits on a CPU array
  and refuses on a CUDA one. It is a `TypeError` on every GPU and on no CPU.
- The layout argument in `lattice._preferred_layout` predicted a large direction-major
  win on a GPU from a coalescing argument. Measured: 1.5% on an A10 and **0.07%** on an
  H100. The reasoning was sound and the magnitude was wrong by an order of magnitude, and
  only a GPU could say so.
- The cost of being wrong about a timescale. The Re 1000 cavity burned 42 minutes of
  H100 running to a step cap it should never have reached, which is how the diffusive
  spin-up time got written down. On a CPU that run would simply never have been attempted.

**What is still not measured, and is still not claimed:** the drone demo's real-time
factor against the brief's 0.5x, which is reported by every `quadrotor` run rather than
asserted here; and warp 1.18 and newton 1.7, which are inside the declared range and were
never installed. `PROGRESS_LIVE.md`'s closing section is the list.

---

## D8. D3Q19 only; D3Q27 is not shipped

**2026-11-22.** The brief says "D3Q19 (D3Q27 optional)", and the option is not taken.

D3Q27 adds the eight body-diagonal directions, which buys two things: better rotational
isotropy of the fourth-order moments, and a little more stability at high Reynolds number
for the same collision operator. It costs 42% more memory and 42% more bandwidth, and
this kernel is bandwidth-bound on every device measured -- so a D3Q27 run is roughly 0.7
times the step rate of a D3Q19 one at the same resolution, or the same rate at 0.89 of
the resolution in each direction.

Three reasons it is not here.

**The accuracy it buys is not the accuracy these cases are limited by.** D3Q19's
quadrature is exact to the order the second-order equilibrium uses -- the moment checks in
`tests/test_live_cases.py::test_the_velocity_set_satisfies_the_moments_it_is_a_quadrature_for`
are what that means -- and the errors measured in `benchmarks/live/` are resolution and
boundary errors, not isotropy errors. Spending 42% of the bandwidth on a fourth moment
while the cylinder's drag is limited by how many cells are across its diameter is
spending it in the wrong place: the same 42% buys 12% more cells in each direction.

**The stability it buys is bought more cheaply by the collision operator.** The
regularised operator annihilates every ghost mode every step, which is a stronger
intervention than adding directions, and it costs nothing in memory.
`DECISIONS_LIVE.md` D3 is that choice.

**It would be a second instantiation of every kernel.** The velocity set is baked in as a
`wp.constant` and `Q` appears in every loop bound and every local vector type, so
supporting both means two compiled variants of the whole module, two layouts each, and
four combinations to test and to report numbers for. That is a real maintenance surface
for a gain the cases cannot see.

**What would change this:** a rotating flow at a high Reynolds number, where D3Q19's
anisotropy is documented to matter, or a measurement in `benchmarks/live/` that an error
is isotropy-limited rather than resolution-limited. Neither exists here. If it is added,
the honest way is a `Q` module option beside `LAYOUT`, measured on both devices like the
layout was, and `lattice.py` is written so that `Q`, `C`, `W`, `OPP` and `MIRROR_*` are
the only things that would have to change.

---

## D9. The fluid has no gravity in it; the body gets weight minus buoyancy

**2026-11-22.** Gravity is applied to the rigid bodies and **not** to the lattice. That is
not an omission and it is not an approximation: it is what every incompressible solver
does, and in a weakly compressible one it is close to mandatory.

A lattice Boltzmann fluid is weakly compressible by construction: pressure is
`rho c_s^2`, so a pressure gradient **is** a density gradient. A metre of water is 9.8 kPa,
and against water's `rho c_s^2` that is a density variation of order 1e-5 -- which sounds
small until it is compared with the density variations the scheme is resolving, which at
`Ma < 0.1` are of the same order. A hydrostatic column would therefore put a real,
solver-visible density stratification into a problem that has none, and it would grow with
the domain height. The usual lattice Boltzmann answer is to subtract the hydrostatic part
analytically, which for a problem with no free surface means leaving it out entirely.

So `FluidWorld` has no gravity term, `set_body_force` is documented as "not gravity", and
`couple.NewtonCoupling` gives each body `-rho_fluid * V * g` -- its buoyancy -- beside the
surface wrench, with Newton applying the weight itself.

**`V` is the voxelised volume and not the analytic one, and that is the part worth
arguing about.** `lattice.volumes` counts the nodes `owner` assigns to each body, which
at twenty cells across a sphere differs from `pi d^3/6` by a few per cent. The analytic
volume is the more accurate number for the real sphere; the voxelised one is the volume
the **boundary condition actually encloses**, and it is the right one here because the
drag is computed on that same voxelised surface. Using the analytic volume for the
buoyancy and the voxelised surface for the drag means the two do not act on the same
body, and a terminal velocity -- which is the difference between two numbers of similar
size -- lands the mismatch squarely on the quantity being graded.

`run_falling_sphere` grades the two volumes against each other for that reason, and the
first design of that case is the evidence: at a density ratio of 1.005 the net driving
force is 0.5% of the weight, the volume error is a few per cent, and the sphere sat
motionless at -0.003 mm/s of an expected 13.2 while every individual part of the
calculation was correct. The case now runs at a ratio of 2, where a 2% volume error moves
the terminal velocity by 1%.

**What this rules out:** a free surface, a stratified flow, thermal convection, and
anything else in which gravity acts on the fluid rather than only on bodies in it. None
of those is in scope -- the brief is aerodynamic and hydrodynamic loads on robots -- and
all of them would want the OpenFOAM path, which has `interFoam` and a real buoyant solver.

---

## D10. The quadrotor is built from primitives, not from Newton's Crazyflie asset

**2026-11-22.** `newton/examples/assets/crazyflie.usd` exists and is not used. The demo
builds a box and four actuator disks instead.

The asset was loaded to find out. It carries **no `RigidBodyAPI`, no `MassAPI`, no joints
and no applied schemas at all**: `ModelBuilder.add_usd` on it yields **zero bodies, twelve
mesh shapes all with `shape_body == -1`, `shape_flags == 0x1` (visible only) and a total
mass of 0.0**. It is a visual prop. No Newton example loads it either -- the drone example's
own line reads `# TODO: Use drone path to load USD asset and draw it`, and
`self.drone_path` is never read again; that example's drone is two crossed boxes. Reading
it also needs `usd-core` and `newton-usd-schemas`, two packages the `[live]` extra does
not declare and would have to.

A flyable body can be built from it by re-adding the twelve meshes with a density, which
was also tried and works: one body, twelve shapes, one free joint, 14.8 g at 1750 kg/m^3.
But then every coupling step voxelises twelve triangle meshes through
`wp.mesh_query_point_sign_normal`, which for an airframe whose aerodynamic effect is
"there is a flat thing under the rotors" is the most expensive possible way to say so --
and `FluidWorld.solid_cells` is the reminder that at any affordable resolution the whole
airframe is a handful of cells thick anyway. A mesh resolved to a tenth of a cell is a
mesh query spent on nothing.

So the demo uses a box sized from the asset's measured geometry -- its rotors sit at
`(+/-0.029565, +/-0.029772, 0.026824)` m in body-local, Z up, which is where the real
Crazyflie's are -- scaled up to a 1.2 kg, 180 mm-arm class vehicle so that a 120 mm rotor
is resolvable at a lattice a real-time loop can afford. A 65 mm Crazyflie would need
sub-millimetre cells to resolve its rotor at all.

**What is given up:** the demo does not look like a Crazyflie, and a user who wants their
own airframe's geometry in the flow has to give `FluidWorld` a mesh -- which it supports,
through Newton's `shape_source_ptr`, and which `couple.py` documents. What is kept: the
physics the demo exists to show, which is a vehicle feeling the downwash its own rotors
made, off a geometry the lattice can actually see.

---

## D11. The free surface reuses `collide` and `stream` rather than copying them

**2026-11-23.** `freesurface.py` needs a collision that skips gas cells and a streaming
step that reconstructs what arrives from them. The obvious implementation is a second
copy of each kernel with those two changes, and the obvious objection is that there
would then be two collisions to keep correct.

So instead `lattice.collide` and `lattice.stream` each took two extra parameters -- a
flag field and an integer that says whether to read it -- and a single-phase run passes
`use_flags = 0` and a one-element dummy array. The cost is two parameters and one
comparison per node in a kernel that already does nineteen of everything. The gain is
that there is exactly one collision operator in this extension and exactly one Bouzidi
boundary, so a fix to either reaches both paths, and the validated single-phase results
in `benchmarks/live/` are evidence about the free-surface path as well.

**Held by a test, not by intention.**
`tests/test_live_freesurface.py::test_a_fully_flooded_free_surface_run_matches_the_single_phase_one`
runs the same case both ways -- once as a `FluidWorld` and once as a `FreeSurfaceWorld`
whose every cell is flagged fluid -- and requires the fields to agree to single
precision. If the two paths ever diverge, that is what says so.

What this rules out: a free-surface-specific collision operator, which some
implementations use (a different `tau` at the interface, for instance). If one is ever
wanted, it belongs as a `scheme` value in `units.rates` rather than as a second kernel.

---

## D12. Surface tension is not shipped

**2026-11-23.** The addendum has it as "optional via curvature" and the option is not
taken. Three reasons, in the order they decided it.

**Nothing in M6 or M7 is surface-tension dominated.** The number that decides is the
Bond number, `Bo = rho g L^2 / sigma`: the ratio of gravity to surface tension on a
length `L`. For water, `sigma = 0.072 N/m`, so `Bo = 1` at `L = 2.7 mm` -- the capillary
length. A cup of water is 70 mm across (`Bo` about 650), a boat hull is hundreds of
millimetres (`Bo` in the tens of thousands), a dam break is a tank. Surface tension
changes the shape of a meniscus a few millimetres wide at the wall and changes nothing
about whether the cup spills or the boat floats.

**The curvature a one-cell interface supports is noisy.** The standard estimate takes
the divergence of the normal, and the normal is already a finite difference of a fill
fraction that is piecewise constant across a cell. Differentiating it twice on a lattice
where the interface is one cell thick gives a curvature whose noise is comparable to its
signal unless it is smoothed, and the smoothing is itself a parameter nobody can
validate against anything in this repository. A term that needs its own validation
before it can be trusted is a term that needs its own milestone.

**The cases that would need it are the ones not asked for.** Droplets, bubbles, foams,
jets breaking up, and thin films are where it matters -- and the Korner paper this
implementation follows is a *foaming* paper for exactly that reason. None of those is in
M5 to M7.

**What is lost, stated so nobody has to discover it:** a free surface here will not form
a meniscus at a wall, will not hold a droplet together below the capillary length, and
will let a thin film tear where a real one would not. `FreeSurfaceWorld`'s unplaceable
mass counter (`mass_drift()["unplaceable"]`) is where a tearing film shows up as a
number, which is the closest this can come to warning about its own absence.

**What it would take:** a curvature field from the smoothed fill fraction, and a
modified gas density at the interface, `rho_gas -> rho_gas - 2 sigma kappa / c_s^2`,
which is a change to one argument of the reconstruction in `lattice.stream` and a new
kernel for the curvature. The validation it would need is a static droplet obeying the
Young-Laplace law, `dp = 2 sigma / r`, which is exact and cheap -- so the work is the
curvature estimate and its smoothing, not the plumbing.

---

## D13. A thin direction is periodic, decided once, in the world builder

**2026-11-23.** Several cases here are quasi-two-dimensional: a few lattice planes deep,
with no dependence on the thin direction, so that the three-dimensional kernels compute
the two-dimensional answer. The cavity has done this from the start and `run_cavity` has
the argument for why it is exact under D3Q19.

The free-surface cases want the same thing, and the first two asked for it by passing
`faces="wall"` -- "a closed tank", which is what a tank is. That sets **all six** faces to
no-slip, including the two ends of a three-cell-wide slot, and a three-cell slot with
walls on both sides has a Stokes layer thicker than the slot. There is no throughflow at
all.

**What it looked like.** The sloshing tank did not slosh. Released in its first natural
mode at 2.00 mm, the surface decayed monotonically to 0.03 mm without ever crossing zero
-- a damped oscillator never does that, so the thing being measured was not an oscillator
-- and the solver converted 102,455 cells trying. Hours went into the free-surface
algorithm looking for the dissipation: the mass exchange, the flag conversions, the
hysteresis band, the gas reconstruction. The same tank with those two faces periodic
oscillates, converts 13,026, and brackets linear theory.

So `_free_surface_world` takes a zero in `size` to mean "thin", gives it `thin` cells,
and **sets those two faces periodic itself**, overriding whatever the case asked for. The
alternative is each case spelling out a six-entry face dictionary and one of them getting
it wrong, which is what happened.

What this rules out: a genuinely wall-bounded narrow slot, which no case here wants and
which would now need to state its faces without a zero in `size`.

---

## D14. The dam break is graded on Ritter's profile, never on the front position

**2026-11-23.** The addendum asks for "dam break vs Martin & Moyce (1952)".
`benchmarks/README.md` already records, for the SPH dam break, that **the paper could not
be read** -- the publisher refuses the request -- and that the two non-dimensionalisations
in circulation, `t sqrt(g/a)` and `t sqrt(2g/a)`, differ by 1.414 at every point.
Quoting a number through a secondary source is the error this repository argues hardest
against. So the comparison is computed and printed in both conventions and **graded in
neither**, and the grading goes to Ritter's (1892) shallow-water solution, which is three
lines of characteristics and needs no paper.

Then two further things had to change, and both were measurements.

**Not the front position.** Ritter's front is a tongue that thins to zero depth, so
"where the front is" is entirely a question of how little water still counts as water.
Working it out on paper: the place where Ritter's own profile reaches half a lattice cell
of depth sits at 0.168 m when the true front is at 0.220 m, a 24% difference that is the
reference's and not the solver's. The front is reported at a quarter, a half and one full
cell so the spread is visible, and graded nowhere. What is graded is the depth at the dam
-- which Ritter fixes at exactly `4 h0 / 9`, independent of time, needing no threshold and
no fit -- and the RMS of the profile where Ritter exceeds two cells.

**Not a square column.** The first version released a column as tall as it was wide, and
that geometry cannot be graded against Ritter at all. Shallow-water theory assumes
hydrostatic pressure, which a collapsing square column violates while it is still
accelerating vertically: measured, the depth at the dam ran 2.07 times Ritter's at
`T = 0.5` and settled to 1.08 by `T = 2`. But the depression wave reaches the back wall
at `T = a/h0` and returns to the dam at `T = 2 a/h0`, so for a square column Ritter
expires at `T = 2` -- exactly when the agreement arrives, which makes the agreement
worthless. The reservoir is now four depths wide, which buys a valid window of `T` in
[2, 8], and the case grades over [2, 4] inside it.

---

## D15. The lattice-Boltzmann free surface is frozen; free-surface liquids move to SPH

**2026-11-23, on instruction.** `freesurface.py` stays, behind an `experimental=True`
gate, with its tests green. It is not deleted and it is not supported.

**What it does.** Still water, correctly. `run_hydrostatic` passes every check: the
volume it was filled with to a part in 1e6, a surface flat to 0.0011 of the depth, mass
drift 3.1e-7 over 5,000 steps, and -- the one that matters -- `d(rho)/dz` within 0.5% of
`-rho g / c_s^2`, which is the pressure gradient buoyancy *is*. A fully submerged body
in it feels `+1.000 x rho V g`. The gravity-wave frequency is right: a standing wave in
a periodic box brackets linear theory, and the sloshing tank converges to +5.1% of it.

**What it does not do: a moving rigid body.** The mass exchange requires the interface to
be a layer exactly one cell thick, and measured in interface cells per wet column on a
released floating box:

| | | |
| --- | --- | --- |
| hull pinned | **0.957** | one layer, as required |
| released + 10 steps | **1.391** | |
| released + 100 steps | 1.584 | and the run diverges later |

Ten steps is far too few for the hull to have travelled anywhere, and the trigger is
measured rather than guessed: the same run with Ladd's moving-wall term disabled holds
the layer at 0.957. Switching that term off is not an option -- a moving wall does hand
momentum to the fluid, and without it a hull feels no drag. Scaling it by the interface
cell's fill fraction is right on its face and changes nothing, which puts the damage at
the fluid cells beside the hull. Korner's formulation has no moving solids in it at all;
this is **missing rather than broken**, and supplying it is a research task.

**So the direction changed.** Free-surface liquids -- boats, cups, pouring -- move to an
SPH backend, where rigid coupling through boundary particles (Akinci et al. 2012) is a
solved problem rather than an open one. LBM remains the supported backend for
single-phase flow: air, and fully submerged water, which is what M1 to M4 validate and
where its advantages are real.

What this rules out: spending more on the LBM moving-contact-line problem. The findings
above are kept so that nobody has to rediscover them, and `freesurface.FROZEN` says the
same thing to anyone who constructs the class without the flag.
