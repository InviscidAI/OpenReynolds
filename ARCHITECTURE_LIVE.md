# `openreynolds/live/` -- where the live-sync backend plugs in, and what it reuses

A map of the extension written *after* reading the repository and *before* writing any
of it, so that the places it touches are chosen rather than discovered. Every claim
about existing code here was read out of the file named, and the line numbers are
deliberately absent: the file names are stable, the line numbers are not.

## What the repository already is

OpenReynolds is an Anthropic tool-use loop over a real workspace. The pieces that
matter to a new solver are these five, and the extension plugs into each of them:

| piece | file | what it is |
|---|---|---|
| the solver selector | `openreynolds/solvers.py` | `openfoam` or `felix`, fixed per study, chosen before the first request |
| the agent's tools | `openreynolds/tools.py` | `TOOLS` (Anthropic tool dicts) and `_HANDLERS` (name -> callable), picked per solver by `tools_for` |
| the CLI | `openreynolds/cli.py` | one `click.Group` called `main`, flat commands, `--json` on everything a script would read |
| the solvers that are not OpenFOAM | `openreynolds/toolbox/` | standalone scripts copied into the workspace; `lbm.py` is a D2Q9 TRT lattice Boltzmann solver in numpy, `sph.py` a particle solver |
| validation | `benchmarks/<family>/grade.py` + `README.md` | one grader and one index per family; `tests/conftest.py::families()` and `tests/test_benchmarks_index.py` make the index's numbers the grader's numbers |

Two facts about that list decided most of the design below.

**`openreynolds/toolbox/` cannot hold this.** `tests/test_toolbox.py` asserts that every
script in the toolbox imports only the standard library, a closed set of eight
third-party modules, and its siblings -- because the toolbox is copied verbatim into a
workspace whose image is fixed, and an import the image cannot serve is a script nobody
can run. Warp is not in that set and must not be: the workspace image is 1.2 GB
compressed already, and the whole argument in `toolbox/lbm.py`'s docstring for why that
solver is numpy applies with more force to a CUDA toolchain. So the live backend is a
real subpackage, `openreynolds/live/`, installed on the user's own machine next to the
GPU, and it is reached from the host side rather than from inside a workspace.

**There is already a lattice Boltzmann solver here, and its conventions are the
house's.** `toolbox/lbm.py` is D2Q9, two relaxation times, `LAMBDA = 3/16`,
`CS2 = 1/3`, halfway bounce-back with the Ladd term for a moving wall, and a
`STIFF_TAU = 0.52` floor below which a run is *reported* as too stiff rather than
refused. `openreynolds/live/units.py` carries the same four constants, for the same
reasons, and `tests/test_live_units.py` asserts the two files agree digit for digit --
it cannot import the toolbox script (see above), so the agreement is a test instead of
a shared module. That is the same device `tests/test_benchmarks_index.py` uses to keep
two indexes from drifting.

## The shape of the extension

```
openreynolds/live/
  __init__.py     the public API, and the one place that reports the extra is missing
  units.py        physical <-> lattice, the fluid table, and the three refusals
  lattice.py      D3Q19 and D3Q27, and every Warp kernel
  world.py        FluidWorld: allocation, voxelisation, the substep loop, the wrenches
  couple.py       the Newton adapter: body_q/body_qd in, body_f out
  cases.py        the validation and demo cases, their references, and their figures
  isaac.py        the Isaac Lab compatible env wrapper
openreynolds/live_tool.py   the agent tool (`live_sim`), beside tools.py rather than in it
benchmarks/live/  grade.py + README.md: the heavy validations, in the house's format
tests/test_live_*.py        small-grid CPU tests in the main suite; GPU behind a marker
docs/live-sync.md           the quickstart and the API, in the existing docs directory
```

### `units.py` -- the units layer, and three refusals

Nothing in the repository converted physical units to lattice units before this: the
OpenFOAM path is dimensional throughout, Felix's `case.yaml` is dimensional, and
`toolbox/lbm.py` is written entirely in lattice units and leaves the conversion to
whoever set `--re`. A lattice solver driven from a robot simulator cannot do that,
because the robot's timestep is given in seconds and its geometry in metres.

`Scaling` is the whole conversion, built from four numbers a caller actually has
(`dx` in metres, `dt` in seconds, the fluid, and a reference speed) and it exposes the
three checks the goal asks for as *refusals at construction*, not as advice:

- `Ma = u_lattice / c_s < 0.1`, because the scheme's error is `O(Ma^2)` and a run above
  that is not solving the incompressible equations it is being graded against;
- `tau_plus > STIFF_TAU`, the same floor and the same wording `toolbox/lbm.py` reports;
- `u_lattice < 0.4`, the point at which the second-order equilibrium's own truncation
  starts to dominate.

Each refusal names the knob that fixes it, in the style of `benchmarks/*/grade.py`'s
`GradeError`: "the case cannot be run as it stands; the message says what to do."

### `lattice.py` -- D3Q19, and why the collision is a choice

Two collision operators, because the two halves of this extension want different
things and the repository's own style is to keep a contrast measurable rather than
assert one is better:

- `trt` (default for validation) is `toolbox/lbm.py`'s scheme in three dimensions:
  `LAMBDA = 3/16` puts a straight no-slip wall on the link midpoint at any viscosity,
  which is the property `benchmarks/lbm/` grades, and it is what makes the cavity and
  the cylinder answers trustworthy at the resolutions a laptop can afford.
- `regularized` (default for robots) projects the non-equilibrium part onto the
  second-order Hermite basis before relaxing it, which removes the ghost modes that
  blow up a high-Reynolds run. It loses the magic-parameter wall property, which is
  exactly why it is not the default for the graded cases.

Smagorinsky applies to both: the eddy viscosity is computed from the same deviatoric
second moment the regularisation already needs, so the LES costs one square root.

Streaming is a two-kernel step -- collide in place, then stream-and-reflect pulling
from the post-collision lattice -- because Bouzidi needs post-collision values at the
boundary node and at its next neighbour away from the wall, which a single fused
pull-stream cannot see.

### `world.py` -- the solid is a signed distance, not a mesh

The lattice carries one float per node (`phi`, the signed distance to the nearest
solid, negative inside) and one int (`owner`, which body that solid belongs to).
Everything else about the boundary is derived from those two in the kernel that needs
it:

- the solid mask is `phi < 0`;
- the Bouzidi fraction is `q = phi[x] / (phi[x] - phi[x+c_i])`, a linear interpolation
  of the same field along the link, clamped away from 0 and 1;
- the wall velocity at the cut point is `v_body + omega x (x_w - com)`, read from the
  body arrays with `owner` as the index;
- a node whose `phi` changed sign since the last coupling step is a fresh node and is
  refilled.

That is why there is no `q` array and no per-link boundary list: at 128^3 a float per
link is another 160 MB and a list has to be rebuilt every time the robot moves. `phi`
is rebuilt instead, by one kernel over the lattice that takes the minimum over the
body shapes -- analytically for Newton's `SPHERE`, `BOX`, `CAPSULE`, `CYLINDER` and
`PLANE`, and through `wp.mesh_query_point_sign_normal` for `MESH`, so a robot made of
primitives and a robot made of triangles voxelise through the same kernel.

**Reused rather than rewritten:** the geometry is Newton's own `Model.shape_type`,
`shape_transform`, `shape_scale` and `shape_source` arrays, already on the device the
kernels run on. Nothing in this extension loads an STL, parses a URDF or triangulates
anything: whatever Newton's `ModelBuilder` can build -- URDF, MJCF, USD -- is what the
fluid sees, and `openreynolds/live/couple.py` is the only file that knows Newton's
field names.

### `couple.py` -- the Newton side, read from Newton's own source

The wrench API was read out of the installed package rather than guessed:
`newton._src.sim.state.State.body_f` is documented there as "an external wrench in
world frame with the body's center of mass (COM) as reference point", shape
`(body_count,)`, dtype `wp.spatial_vector`, linear force in the first three components
and torque in the last three. The loop is Newton's own, from
`newton/examples/basic/example_basic_dzhanibekov.py`: `state.clear_forces()`, then
accumulate into `body_f`, then `solver.step(state_0, state_1, control, contacts, dt)`,
then swap. `SolverMuJoCo` is the default and was confirmed to run here on the Warp CPU
device with `mujoco_warp` installed.

The fluid's wrench array is `wp.array(dtype=wp.spatial_vector)` with the same layout
and the same length, so handing it to Newton is `wp.launch` of a six-line add kernel --
no host round trip, which is the one hard constraint on the step loop.

### How a study reaches it

Three doors, matching the three the goal asks for:

1. **`openreynolds live ...`** -- a `click` group on the existing `main`, with
   `cases`, `run`, `bench` and `crosscheck` subcommands, `--json` on each, and the
   import of anything from `openreynolds.live` deferred into the command body so that
   `openreynolds --help` on a machine without the extra still works. This is the same
   deferred-import shape `cli.py` already uses for `site`, `video` and `compare`.
2. **the `live_sim` agent tool** -- a host-side tool, like `fetch` and `cad` and
   unlike `bash`: it does not run in the workspace, because the workspace has no GPU
   and no Warp. It takes a case name or a plain description, runs it, writes the
   figures into the study's own directory, and returns the report text the CLI would
   have printed. Offered only when the extra is importable, by the same mechanism
   `tools_for` already uses to withhold `cad` when there is no desk behind it.
3. **the case file, with a backend selector** -- `backend: live | openfoam` in a
   small YAML the live backend reads. `live` runs it here; `openfoam` is accepted and
   answers with the OpenFOAM case the same description corresponds to, which is what
   makes `openreynolds live crosscheck` a comparison of two backends on one case
   rather than two separately written cases.

### The cross-check, and why it needs no OpenFOAM install

`tests/data/moving_mesh/viv_fixed/` is a finished OpenFOAM run, committed: a fixed
cylinder at Re = 100, D = 10 mm, U = 10 mm/s, water, 2.5% blockage, with its
`postProcessing/forceCoeffs/*/coefficient.dat` intact.
`benchmarks/moving_mesh/grade.py` reads it and reports St = 0.166141 and mean
Cd = 1.3555, and `tests/test_benchmark_moving_mesh.py` asserts both to five digits.

So the cross-check runs the *same* case on the live backend and reports the difference
against those two numbers, read out of the OpenFOAM files by the existing grader
rather than copied into a constant. It is a real two-backend comparison on one case,
it needs no cluster and no OpenFOAM, and if the OpenFOAM numbers ever move the
cross-check moves with them.

### What is deliberately not touched

- **`openreynolds/solvers.py` keeps two solvers.** A third entry in `CHOICES` would
  reach the hosted service's workspace kinds, `desk.route`'s routing request, the
  system prompt variants and `tools_for`'s tool lists, none of which can serve a
  backend that runs on the user's own GPU. The selector the goal asks for is in the
  case file and in `openreynolds live`, where it decides something this repository can
  actually honour. Logged in `DECISIONS_LIVE.md`.
- **The OpenFOAM path.** Nothing in `backend/`, `tools.py`'s `_bash`/`_job_start`, the
  toolbox or the templates changes. The only edits outside `openreynolds/live/` are
  additive: one `click` group in `cli.py`, one tool dict and one handler entry in
  `tools.py`, one extra in `pyproject.toml`, one row in `benchmarks/README.md`, and
  the docs.
- **`toolbox/lbm.py`.** It stays the solver that reaches every workspace with no image
  and no GPU, and `advise.py` keeps pointing at it. The live backend is not its
  replacement; it is the one that can carry a moving robot.
