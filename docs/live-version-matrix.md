# `openreynolds/live/` across warp-lang 1.17-1.19 and newton 1.6-1.8

The `[live]` extra was built against warp-lang 1.19.0 and newton 1.8.0. **Isaac Lab
3.0.0 pins `warp-lang==1.17.0` and `newton[sim]==1.6.1`** (read out of its own
`METADATA`, not a changelog), so a user running this extension inside Isaac Lab gets
those regardless of what `pyproject.toml` asks for. This file is what happened when the
extension was run on both.

**The answer is that it works on both, and that no source change was needed to make it.**
Every Warp symbol the kernels use behaves identically on 1.17.0 and 1.19.0; every Newton
field, `GeoType` name, `GeoType` value and `shape_scale` component the adapter reads is
unchanged between 1.6.1 and 1.8.0. The Re 100 lid-driven cavity comes out **bit-identical**
on the two. The one real difference is speed, and it is a factor of 3.2 against the older
Warp on the CPU backend.

The declared range is therefore `warp-lang>=1.17,<1.20` and `newton>=1.6,<1.9`, carried
in `pyproject.toml`'s `live` extra and in `openreynolds/live/units.py::WARP_TESTED`, with
`tests/test_live_versions.py` asserting those two agree so they cannot drift apart.

## The two environments

| | A: development | B: Isaac Lab |
|---|---|---|
| interpreter | `C:\Users\reach\AppData\Local\Programs\Python\Python312\python.exe` | `C:\iv\v\Scripts\python.exe` |
| python | 3.12.10 | 3.12.10 |
| warp-lang | **1.19.0** | **1.17.0** |
| newton | **1.8.0** | **1.6.1** |
| isaaclab | not installed | 3.0.0 |
| torch | not installed | 2.11.0+cpu |
| mujoco-warp | 4.0.1 | not installed |
| device | `cpu` (no CUDA driver on this machine) | `cpu` (same) |

Environment B was made importable without disturbing its pins:

```
C:\iv\v\Scripts\python.exe -m pip install "newton==1.6.1" --no-deps
C:\iv\v\Scripts\python.exe -m pip install -e C:\Users\reach\Documents\Reynoldsv2\OpenReynolds --no-deps
C:\iv\v\Scripts\python.exe -m pip list | grep -iE "warp-lang|^newton |isaaclab|torch "
  isaaclab             3.0.0
  newton               1.6.1
  torch                2.11.0+cpu
  warp-lang            1.17.0
```

`newton` was absent from that venv when this started (`isaaclab` declares it but it had
not been installed), so it was installed at Isaac Lab's own pin. The `--no-deps` on both
commands is the whole point: without it pip resolves and moves `warp-lang`. The pins were
re-read after every install and after the full test run, and are unchanged.

## The matrix

| | A: warp 1.19.0 / newton 1.8.0 | B: warp 1.17.0 / newton 1.6.1 |
|---|---|---|
| `import openreynolds.live` | yes | yes |
| `import openreynolds.live.lattice` (every kernel declared) | yes | yes |
| kernels compile | yes, all | yes, all |
| small CPU lattice runs to the right answer | yes | yes, **bit-identical** |
| Newton coupling steps | yes | yes, same wrench to the last float |
| zero-copy warp <-> newton | yes, by pointer | yes, by pointer |
| zero-copy warp <-> torch | torch not installed here | yes, by pointer, both directions |
| `tests/test_live_versions.py` | 16 passed, 1 skipped (torch) | **17 passed** |
| step rate, Re 100 cavity | **3.51 MLUPS** | **1.11 MLUPS** |

## The correctness check, run on both

```
dx = 1.0/32; nz = 3
s = Scaling.for_reynolds(dx, 1.0, 100.0, "water", u_lattice=0.055)
d = Domain(origin=(0,0,0), size=(1.0,1.0,nz*dx),
           faces={"-x":"wall","+x":"wall","-y":"wall","+y":"wall",
                  "-z":"periodic","+z":"periodic"},
           face_velocity={"+y": (s.reference_velocity,0,0)})
w = FluidWorld(domain=d, dx=dx, fluid="water", scaling=s, substeps=1, device="cpu")
w.initialise()
for _ in range(20000): w.step()
u = w.fields()["u"][:,:,nz//2,:] / s.reference_velocity
```

Output, with the plane extrema the snippet prints and the Ghia centreline profiles the
plane extrema are often mistaken for (see the note below):

```
=== MAIN (warp 1.19) ===
warp 1.19.0
PLANE-EXTREMA -0.2407 -0.5246 0.2603
CENTRELINE    -0.21662 -0.26414 0.18441
GHIA 1982     -0.21090 -0.24533 +0.17527
MLUPS 3.51  wall 17.5s for 20000 steps
=== ISAAC (warp 1.17) ===
warp 1.17.0
PLANE-EXTREMA -0.2407 -0.5246 0.2603
CENTRELINE    -0.21662 -0.26414 0.18441
GHIA 1982     -0.21090 -0.24533 +0.17527
MLUPS 1.11  wall 55.2s for 20000 steps
```

Not just equal to four digits -- equal byte for byte. The velocity plane was saved from
each run and hashed:

```
shape (32, 32, 3) bitwise identical: True
max abs diff: 0.0
sha1 1.19: 604460532ea4f095e2b9a78da16482de693b00d0
sha1 1.17: 604460532ea4f095e2b9a78da16482de693b00d0
```

That is the strongest form the answer could take. The two Warps generate different
machine code at different speeds and produce the same float32 lattice, which means no
kernel in `lattice.py` is relying on anything that moved between them.

### A correction to the reference numbers this check is usually quoted with

The brief for this work gave the expected output of that snippet as approximately
`-0.2112`, `-0.2480`, `+0.1765`, "which match Ghia, Ghia & Shin (1982) Table I/II at
Re 100 (`-0.21090`, `-0.24533`, `+0.17527`) to about 1%". Neither half of that is what
this code does, and both halves are recorded here because the same claim would otherwise
be re-measured by the next person:

* The snippet's three numbers are **extrema over the whole mid-plane**, not centreline
  values. Ghia's Table I and II are profiles along the vertical centreline (`u` against
  `y` at `x = 0.5`) and the horizontal centreline (`v` against `x` at `y = 0.5`). The
  plane minimum of `v` is near a corner and is twice the centreline minimum, which is
  the `-0.5246` against `-0.24533` above. The two sets of numbers are not comparable and
  the agreement figure for the plane extrema is not 1%, it is 114%.
* There is then a **second** correction, and it is the one that decides the number.
  `u[nx//2, :, 0]` is not the centreline either, for an even cell count. The lattice
  nodes sit at `(i + 0.5)/n`, so for `n = 32` there is no node at `x = 0.5`: index 16 is
  at `0.515625`, half a cell past it. The `v` profile through the middle of a cavity is
  steep there -- it crosses zero and reaches its extremes within a few cells -- so half
  a cell of offset is several per cent. `openreynolds/live/cases.py::run_cavity` and
  `openreynolds/toolbox/lbm.py::cavity` both take the **mean of the two lines
  straddling the centre** for exactly this reason, and say so in a comment.

  All three extractions, from one 20,000-step run on this machine:

  | extraction | u min | v min | v max | against Ghia |
  |---|---|---|---|---|
  | extrema over the whole mid-plane | -0.24074 | -0.52455 | +0.26025 | meaningless: includes the lid and the corners |
  | single nearest column, `u[n//2]` | -0.21661 | -0.26414 | +0.18441 | 2.7%, 7.7%, 5.2% |
  | **mean of the two straddling lines** | **-0.21135** | **-0.24814** | **+0.17663** | **0.21%, 1.15%, 0.78%** |
  | Ghia, Ghia & Shin (1982), Re 100 | -0.21090 | -0.24533 | +0.17527 | |

  and the RMS over Ghia's 15 interior points, which is what is actually graded, is
  **0.22% on u and 0.26% on v**. So "to about 1%" is a true sentence after all, and the
  2.7%/7.7%/5.2% figures above it are an artefact of where the profile was sampled
  rather than a property of the solver. A 32x32 lattice agreeing with a 129x129
  second-order solution to a quarter of a per cent is better than it sounds and partly
  luck: `benchmarks/README.md` already records, about the Felix run of this same case,
  that Ghia's own Re 1000 extrema sit 1.46% to 2.20% from Botella and Peyret's spectral
  benchmark, so agreement much tighter than Ghia's own error is agreement with his
  error.

Neither correction has anything to do with the version question, and in particular
neither environment is better or worse than the other at it: they agree bit for bit.
The graded version of this lives in `openreynolds/live/cases.py::run_cavity`, which does
the straddling-mean extraction, drops Ghia's two boundary rows (they are the boundary
condition, not a solution, and a lattice has no node on the wall to sample them at), and
is run at 128 and 256 cells on a GPU by `benchmarks/live/modal_run.py`.

## The Newton coupling, run on both

A one-sphere model, a 20x20x20 lattice, three coupling steps. Environment A used
`SolverMuJoCo` and also `SolverXPBD`; environment B has no MuJoCo backend installed
(`newton[sim]` pins `mujoco-warp ~=3.12.0` at newton 1.6.1 and `~=3.17.0` at 1.8.0, and
nothing in `openreynolds/live/` imports mujoco), so the comparison below is `SolverXPBD`
on both. Gravity was zeroed so the body stays inside the box.

```
A: warp 1.19.0 / newton 1.8.0          B: warp 1.17.0 / newton 1.6.1
  solver: SolverXPBD                     solver: SolverXPBD
  step 0: solid nodes=552                step 0: solid nodes=552
          wrench=(3.6129134e-12, ...)            wrench=(3.6129134e-12, ...)
  step 1: solid nodes=552                step 1: solid nodes=552
          wrench=(-5.3912903e-12, ...)           wrench=(-5.3912903e-12, ...)
  step 2: solid nodes=552                step 2: solid nodes=552
          wrench=(0.0, 0.0, 0.0)                 wrench=(0.0, 0.0, 0.0)
  body_q: [0,0,1, 0,0,0,1]               body_q: [0,0,1, 0,0,0,1]
```

Same voxelisation, same wrench, same pose. On A, `SolverMuJoCo` also steps the same
model for three steps without error.

## Zero-copy sharing, in environment B specifically

This is the property that matters most for an Isaac Lab user, because their observations
are torch tensors and this extension's lattice is `wp.array`s. Run in `C:\iv\v`, warp
1.17.0 / newton 1.6.1 / torch 2.11.0+cpu, actual output:

```
=== wp.from_torch: torch tensor -> wp.array ===
  t.data_ptr() = 0x434aa0100c0
  a.ptr        = 0x434aa0100c0
  SAME POINTER : True
  mutated through wp, read through torch: [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
  mutated through torch, read through wp: [-99.0, 10.0, 20.0]

=== wp.to_torch: wp.array -> torch tensor ===
  b.ptr         = 0x1f23acfabc0
  tb.data_ptr() = 0x1f23acfabc0
  SAME POINTER  : True
  mutated through wp, read through torch: [3.5, 3.5, 3.5, 3.5, 3.5]
  mutated through torch, read through wp: [3.5, 3.5, -7.0, 3.5, 3.5]

=== round trip wp -> torch -> wp ===
  c.ptr == c2.ptr: True 0x1f23ad49ac0 0x1f23ad49ac0

=== newton handover: couple.py's five arrays ===
  state.body_f             ptr=0x1f23acfd000 device=cpu dtype=spatial_vectorf
  state.body_q             ptr=0x1f23acfd400 device=cpu dtype=transformf
  state.body_qd            ptr=0x1f23acfe040 device=cpu dtype=spatial_vectorf
  model.body_com           ptr=0x1f23ad4afc0 device=cpu dtype=vec3f
  model.shape_source_ptr   ptr=0x1f23ad48e40 device=cpu dtype=uint64
  after a mixed launch -- any pointer moved? {'state.body_f': False,
      'state.body_q': False, 'state.body_qd': False, 'model.body_com': False,
      'model.shape_source_ptr': False}
  body_f written by the kernel: [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
  our array written by the same kernel: [1.0]
```

Three things are established there, each by a pointer comparison rather than by an
absence of an error, because a copy would also work:

1. **`wp.from_torch` and `wp.to_torch` wrap, they do not copy.** The pointers are equal
   in both directions and so is a round trip, and a write through either side is visible
   through the other -- checked by mutating through Warp and reading through torch, and
   then the reverse.
2. **An array Newton created and an array this extension created go into the same kernel
   with no copy.** The kernel below reads `state.body_q` and `model.body_com`, writes
   `state.body_f`, and writes an array `wp.zeros` made here, all in one launch.
3. **Nothing in `couple.py`'s handover silently copies.** The five arrays' `.ptr` values
   were taken before the launch and re-read afterwards off freshly-accessed attributes,
   in case any of them is a property that rebuilds. None moved, and the floats the kernel
   wrote are readable from both sides. `tests/test_live_versions.py::test_newton_and_our_arrays_share_a_kernel_without_a_copy`
   is that check, in the suite.

Environment A gives the same pointer-stability result for the Newton arrays; its torch
rows are blank only because torch is not installed there, and installing it was out of
scope (the main environment's pins were not to be touched).

## Every API difference found, warp 1.17.0 against 1.19.0

Sixteen probes, each one a thing `lattice.py` or `world.py` actually does, run in both.
**All sixteen pass identically.** The full list, because a negative result is only useful
if it says what was looked at:

| probe | 1.19.0 | 1.17.0 |
|---|---|---|
| `wp.set_module_options({"max_unroll", "enable_backward"})` | accepted | accepted |
| `wp.set_module_options` with an invented key | accepted silently | accepted silently |
| `wp.constant(wp.types.matrix(shape=(3,19), dtype=int32))` | `mat_t` | `mat_t` |
| `wp.constant(wp.types.vector(length=19, ...))`, f32 and i32 | ok | ok |
| `wp.where` exists / `wp.select` still exists | both | both |
| `wp.where(cond, if_true, if_false)` argument order | `[111, 222]` | `[111, 222]` |
| `wp.where` with a module-level Python bool and `array.shape[i]` | 4 | 4 |
| `wp.mesh_query_point_sign_normal` + `.result/.sign/.face/.u/.v` + `wp.mesh_eval_position` | `(-1.0, -1.0)` | `(-1.0, -1.0)` |
| `transform_get_translation/rotation`, `transform_multiply`, `transform_inverse`, `transform_point`, `quat_identity` | same floats | same floats |
| `wp.spatial_vector(...)` built in a kernel, `v[0]`..`v[5]`, `+` | same | same |
| `wp.atomic_add(array2d(f32), i, j, x)` and `wp.atomic_add(array(int32), i, 1)` | `[10]*6`, 60 | `[10]*6`, 60 |
| `wp.full`, `wp.copy`, `array.zero_()`, `array.fill_()`, `array.assign()` | `(2.5, 0.0, 7.0, 3.0, -1)` | `(2.5, 0.0, 7.0, 3.0, -1)` |
| `wp.vec3i`, `wp.clamp`, `wp.synchronize_device`, `wp.get_preferred_device`, `wp.get_cuda_device_count` | `(6, 1.0)` | `(6, 1.0)` |
| `wp.mat33` host constructor | ok | ok |
| `int(-1)` / `float32(0.0)` locals mutated inside a dynamic `range(19)` loop | `(90, 9.5)` | `(90, 9.5)` |
| a local `vector(length=19)` written element by element in a loop | `9.0000010` | `9.0000010` |

Two of those deserve the detail, being the two the house comments single out:

**`max_unroll` is honoured by both, and is worth less on 1.17.0.** Warp unrolls a
constant-bound `range` only to `max_unroll`, whose default is 16, and `Q` is 19, so
`lattice.py` raises it to 24. On 1.17.0 the option is read, applied and reported back by
`get_module_options()` exactly as on 1.19.0 -- there was no shim to write. But its effect
reverses. An isolated Q-loop kernel, 48^3 nodes, best of three:

```
                  max_unroll 16   max_unroll 24
warp 1.19.0          49.71 MLUPS     55.29 MLUPS
warp 1.17.0          43.95 MLUPS     38.84 MLUPS
```

The value stays 24: the answers are bit-identical either way, and 1.19.0 and every GPU
are what the step rate is for.

**`wp.from_torch` needs the Warp runtime to exist first, on both.** Called as the first
Warp operation in a process it raises
`AttributeError: 'NoneType' object has no attribute 'cpu_device'` -- Warp's runtime is
created lazily and `device_from_torch` does not force it. The source of
`warp/_src/torch.py::device_from_torch` is byte-identical in 1.17.0 and 1.19.0, so this
is a Warp-wide ordering trap and **not** a version difference. It is listed here because
it is the first thing an Isaac Lab user hits and because the extension happens to fix it:
`lattice._preferred_layout` calls `wp.get_cuda_device_count()` at module scope, which
creates the runtime, so torch interop works from the moment anything in this package is
imported. Verified:

```
runtime before: None
runtime after importing lattice: Runtime
from_torch works straight after our import, ptr match: True
```

`tests/test_live_versions.py::test_torch_interop_needs_the_runtime_and_importing_the_extension_creates_it`
turns that accident into a property.

## Every API difference found, newton 1.6.1 against 1.8.0

A model with one body carrying a sphere, a box, a capsule, a cylinder and a mesh, plus a
static plane, was built and finalised in both and printed.

**`shape_scale`'s packing is identical.** This was the one that had to be measured rather
than trusted: `couple.py::_SCALE` is the identity for every supported type, so a changed
packing could not raise -- it would put the radius where the half-height belongs and the
body would voxelise as a point or a slab, with forces that are wrong and plausible.
Identical output from both:

```
  shape 0 SPHERE       scale=(0.3, 0.0, 0.0)
  shape 1 BOX          scale=(0.1, 0.2, 0.3)
  shape 2 CAPSULE      scale=(0.11, 0.42, 0.0)
  shape 3 CYLINDER     scale=(0.13, 0.44, 0.0)
  shape 4 PLANE        scale=(0.0, 0.0, 0.0)
  shape 5 MESH         scale=(0.5, 0.6, 0.7)
```

which is `couple.py`'s header table, confirmed on 1.6.1. The sphere is still
`(radius, 0, 0)` and not `(r, r, r)`; Newton's own comment in `builder.py` is wrong in
the same way in both releases.

**Every field exists with the same dtype.** `Model.shape_source_ptr` (`uint64`),
`shape_type` (`int32`), `shape_body` (`int32`), `shape_transform` (`transformf`),
`shape_scale` (`vec3f`), `body_com` (`vec3f`), `body_mass` (`float32`), `gravity`
(`vec3f`), and `shape_label` / `body_label` as plain lists. `State.body_q`
(`transformf`), `body_qd` (`spatial_vectorf`), `body_f` (`spatial_vectorf`) and
`clear_forces()`. `shape_source` is still a plain Python list in both, which is why
`shape_source_ptr` is the one passed to a kernel.

**`newton.GeoType`'s member names *and values* are unchanged**, all twelve: `NONE` 0,
`PLANE` 1, `HFIELD` 2, `SPHERE` 3, `CAPSULE` 4, `ELLIPSOID` 5, `CYLINDER` 6, `BOX` 7,
`MESH` 8, `CONE` 9, `CONVEX_MESH` 10, `GAUSSIAN` 11. `GEO_TO_SHAPE` maps by name on
purpose and so would have survived a renumbering; it did not have to.

The four differences that do exist, none of which this extension touches except the
first:

1. **`solver.step()` gained a keyword-only `observables=None` in 1.8.0.** On 1.6.1 the
   signature is `step(state_in, state_out, control, contacts, dt)` and nothing else.
   `NewtonCoupling.step` passes exactly those five positionally, which is why it works on
   both; `contacts=None` stays an explicit positional because on 1.6.1 it has no default.
   A comment at the call site now says so.
2. **`newton.solvers.SolverSteel` is new in 1.8.0.** Not used here. `SolverMuJoCo`,
   `SolverXPBD`, `SolverSemiImplicit`, `SolverFeatherstone`, `SolverVBD`,
   `SolverStyle3D`, `SolverKamino`, `SolverImplicitMPM` and `SolverBase` exist on both.
3. **`State.body_q_prev` exists on 1.6.1 and is gone in 1.8.0.** Not used here.
4. **`newton[sim]`'s MuJoCo pin moved**: `mujoco-warp ~=3.12.0` and `mujoco ~=3.12.0` at
   1.6.1, `~=3.17.0` at 1.8.0, against the 4.0.1 this was developed on (Newton warns and
   runs). This is packaging rather than API -- nothing in `openreynolds/live/` imports
   mujoco, since `SolverMuJoCo` is an object the caller constructs -- so the `live`
   extra's floor is `mujoco-warp>=3.12`, low enough to install beside newton 1.6.1 at
   all, with the band left to whoever chooses that solver.

Also worth recording because it bears on the range: **newton 1.6.1's own requirement is
`warp-lang>=1.17.0` with no upper bound**, so Newton does not itself prevent the other
three corners of the version square.

## What broke, and what fixed it

Nothing in the extension broke. The three source edits are not shims; they are the
evidence written down where someone changing those lines will read it:

| file | change |
|---|---|
| `openreynolds/live/lattice.py` | the version evidence at the `wp.set_module_options` call (both keys accepted on both, neither version validates key names, so there is nothing to catch and no try/except), and in `_preferred_layout`'s docstring the re-measured unroll table, the 3.2x CPU gap with its cause, and the runtime-creation side effect that makes torch interop work |
| `openreynolds/live/couple.py` | a section in the header docstring recording the 1.6.1 re-read of every field, the `GeoType` names and values, and `shape_scale`'s packing; and a comment at the `solver.step` call saying why it is five positionals and no keywords |
| `openreynolds/live/units.py` | `WARP_TESTED`, `NEWTON_TESTED` and `version_note()`: the band in code, with the evidence, plus one report line when a run is outside it |
| `openreynolds/live/world.py` | `FluidWorld.report` appends that line, so a number copied out of a report carries the fact that its dependencies were untested |
| `pyproject.toml` | `warp-lang>=1.17,<1.20`, `newton>=1.6,<1.9`, `mujoco-warp>=3.12`, with the evidence above the table |
| `tests/test_live_versions.py` | new: sixteen assertions, each one a claim this file rests on, checked against whatever is installed |

Three things were found that are not version problems but were found by looking:

* the brief's expected cavity triple and its "1%" agreement claim are both wrong, for the
  reason given above. The solver is fine; the comparison was between plane extrema and
  centreline profiles.
* the `C:\iv\v` venv had `isaaclab` but no `newton` at all, despite `isaaclab` declaring
  it. Installed at the pin with `--no-deps`.
* that venv has no `anthropic`, so the repository's `tests/conftest.py` cannot import
  there. `tests/test_live_versions.py` is run in it with `--noconftest`, which it does
  not need. Worth knowing before anyone tries to run more of the suite in an Isaac Lab
  environment.

## What was not tested, and why

* **CUDA, on either.** This machine has no NVIDIA driver (`nvidia-smi` absent, Warp
  reports one device, `cpu`), which `DECISIONS_LIVE.md` D7 already records for the
  performance targets. Everything above is the CPU backend. The layout choice in
  `lattice._preferred_layout` flips to `soa` on a GPU, so the GPU code path is a
  *different* generated kernel from the one measured here and the bit-identical result
  does not transfer to it. That is the largest open hole in this matrix.
* **warp-lang 1.18.** Not installed. It sits between two releases that agree bit for bit,
  which is weaker evidence than a run and stronger than nothing. The upper bound is
  `<1.20` rather than open because an open end is a claim about releases that do not
  exist.
* **newton 1.7.** Same, between 1.6.1 and 1.8.0.
* **`SolverMuJoCo` on newton 1.6.1.** The class exists and its signature was checked, but
  the MuJoCo backend is not installed in environment B and installing `mujoco-warp
  ~=3.12.0` there would have pulled a resolver run across the pinned `warp-lang`. The
  coupling was stepped with `SolverXPBD` instead, which exercises the same handover --
  `couple.py` only ever calls `solver.step(...)` with five positionals.
* **matplotlib in environment B**, so `openreynolds/live/figures.py` and the figures every
  case draws were not exercised there. Nothing in the fluid or the coupling needs it.

## Reproducing

The probes above are not loose scripts: every one of them is a test in
`tests/test_live_versions.py`, which is the durable form and is what to run in a new
environment. The whole matrix is reproduced by running that file in each interpreter.

```
# environment A, the development env (torch is not installed, so one test skips)
C:\Users\reach\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_live_versions.py -q
  16 passed, 1 skipped in 15.91s

# environment B, Isaac Lab. --noconftest because that venv has no `anthropic`, which
# the repository's tests/conftest.py imports; this file does not need the conftest.
C:\iv\v\Scripts\python.exe -m pytest tests/test_live_versions.py -q --noconftest
  17 passed in 26.45s

# the pins, re-read after the run
C:\iv\v\Scripts\python.exe -m pip list | grep -iE "warp-lang|^newton |isaaclab|torch "
  isaaclab 3.0.0   newton 1.6.1   torch 2.11.0+cpu   warp-lang 1.17.0
```

The cavity is the snippet quoted under "The correctness check" above, run with
`PYTHONPATH` set to the repository root in environment B (it is installed `-e` there, so
that is only belt and braces). It takes about 18 s on warp 1.19.0 and 55 s on 1.17.0 for
20000 steps. The three-digit plane extrema and the five-digit centreline triple are
printed from the same `w.fields()["u"]` plane; the SHA-1 comparison above was made by
`np.save`-ing that plane from each run and hashing the two files.
