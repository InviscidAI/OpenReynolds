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

## D7. No GPU on the machine this was built on, and what that means for the numbers

**2026-11-22.** `nvidia-smi` is not present here and Warp reports exactly one device,
`"cpu"`. Every number in `PROGRESS_LIVE.md` was therefore measured on Warp's CPU
backend. That is stated at the top of that file and beside every rate in it, and the
brief's "1000+ MLUPS on an H100-class GPU" target is recorded as **not measured**
rather than estimated from a scaling argument. A performance claim nobody ran is the
one kind of number this repository's `benchmarks/README.md` is written against.

What this does *not* compromise: the physics. The same Warp kernels compile for CUDA
from the same source, and the validations are resolution-limited rather than
device-limited -- a cavity that agrees with Ghia on the CPU agrees with Ghia on an
H100. What it does compromise: the two end-to-end rate targets (the drone demo at 0.5x
real time, and the 1000 MLUPS floor), which are device claims and are left open.
`benchmarks/live/grade.py` prints MLUPS on whatever device it is run on, so the moment
this is run on a GPU the number exists.
