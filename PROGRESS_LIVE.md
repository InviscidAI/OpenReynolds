# `openreynolds[live]` -- what was built, what it measures, and on what

The record of the live-sync extension: a GPU lattice Boltzmann fluid in lockstep with
the Newton physics engine, inside this repository, on the branch `feature/live-sync`.

**Every number in this file names the device that produced it.** A rate without a device
is not a measurement, and an agreement with a published value is not improved by the
hardware behind it. Where something was not measured, it says "not measured" rather than
an estimate.

## Where everything is

| document | what it is |
|---|---|
| [ARCHITECTURE_LIVE.md](ARCHITECTURE_LIVE.md) | where the extension plugs into OpenReynolds and which existing pieces it reuses, written after reading the repository and before writing any of it |
| [DECISIONS_LIVE.md](DECISIONS_LIVE.md) | every choice taken without being able to ask, the alternatives, and the evidence. D1 is the XLB evaluation |
| [docs/live-sync.md](docs/live-sync.md) | the quickstart, the Python API, the CLI, and the limitations |
| [docs/live-version-matrix.md](docs/live-version-matrix.md) | the tested dependency range, warp 1.17 to 1.19 and newton 1.6 to 1.8, with the zero-copy pointer checks |
| [benchmarks/live/README.md](benchmarks/live/README.md) | the case family, in this repository's benchmark format |
| `benchmarks/live/grade.py` | the grader. It solves, like `benchmarks/lbm/grade.py` |
| `benchmarks/live/modal_run.py` | the GPU runner. Not part of the package |

## The hardware

Two machines, and the split matters for reading every number below.

**The development machine** has no CUDA device at all: `nvidia-smi` is absent and Warp
reports one device, `"cpu"` (Intel Core Ultra, 8 threads). Warp's CPU backend is
single-threaded, so every rate from it is one core. It is where the kernels were written,
where the test suite runs, and where four of the five bugs recorded below were found.

**The GPU runs are on Modal**, `inviscid` workspace, through
`benchmarks/live/modal_run.py`: one-shot functions with explicit timeouts, no warm pool,
no volume, nothing left running. Each job pins warp-lang 1.19.0, newton 1.8.0 and
mujoco-warp 4.0.1 exactly, so the published numbers are comparable with each other.

| job | shape | what it was for | app |
|---|---|---|---|
| probe | H100 80GB HBM3, driver 610.57.04 | confirm Warp sees a CUDA device | `ap-d8IWqzlGQzhqdbpVAoEHFc` |
| smoke | NVIDIA A10 22 GiB, sm_86 | validate the image and the whole path end to end | `ap-2CxxrOfPQxw1nigoSJ2Ic1` |
| layouts | H100 80GB | both array layouts, 128/256/384 cells | see `benchmarks/live/results/layouts.json` |
| layouts | A10 | both array layouts, 128/192 cells | " |
| M1, M2, M3 | H100 80GB | the validations and the demos | `benchmarks/live/results/<case>/result.json` |

## Performance

D3Q19, million lattice updates per second, nodes times steps over seconds, best of three
after a warm-up launch. The collision is `trt` with no subgrid model, which is the
configuration the graded cases run in.

| device | layout | 128^3-equivalent | 256^3-equivalent | largest measured |
|---|---|---|---|---|
| H100 80GB | soa | 1095.2 | 1453.6 | **1478.7** at 384 cells (14,155,776 nodes) |
| H100 80GB | aos | 1069.8 | 1452.1 | 1477.6 at 384 cells |
| A10 22 GiB | soa | 454.5 | -- | 510.8 at 192 cells (1,769,472 nodes) |
| A10 22 GiB | aos | 452.2 | -- | 503.1 at 192 cells |
| one CPU core | aos | 4.4 | -- | 4.4 at 128 cells (262,144 nodes) |
| one CPU core | soa | 1.3 | -- | 1.3 at 128 cells |

**The brief's 1000+ MLUPS on an H100-class GPU: met, 1478.7 MLUPS.** It is met at 128
cells too (1095.2), which is the useful part -- the target is reached at a lattice a
real-time loop can afford and not only at one that fills the card.

Two things in that table are worth more than the headline.

**The direction loop was not unrolling at all.** Warp unrolls a constant-bound `range`
up to `max_unroll`, which defaults to 16, and `Q` is 19. Raising the limit and switching
to the node-major layout took the CPU solver from 1.3 to 4.4 MLUPS, a factor of 3.4, and
on an isolated streaming kernel from 7.9 to 29.8. It is the single largest factor in the
CPU step rate and it was a default nobody had reason to look at.

**The layout is worth a factor of three on a CPU core and nothing on a GPU.** The
coalescing argument -- a warp's reads of one direction are one transaction under `soa`
and thirty-two under `aos` -- predicts a large `soa` win. Measured, it is 1.5% on an A10
and **0.07%** on an H100, because the `aos` reads a warp issues are a contiguous
2,432-byte run rather than a scatter, and an H100's 50 MB L2 absorbs the rest. The
per-device default stands, but it is now chosen from numbers on both devices rather than
from the argument, and the honest summary is that only the CPU default matters.
`lattice._preferred_layout` carries both tables.

### Coupled step rate, and real time

The end-to-end figure a robot cares about is simulated seconds per wall-clock second at a
usable resolution. `FluidWorld.report` prints it for every run and the demo rows below
carry it.

## The bugs, and what found each one

Recorded because each says something about what the validation was for. Four of five were
found by a check built to find them rather than by reading the code.

| bug | found by | what it was |
|---|---|---|
| a stream source leaving the domain on two axes at once got only the first axis's rule, then read out of bounds | **the exact case.** Poiseuille, where a two-relaxation-time wall at `LAMBDA = 3/16` reproduces the analytic parabola to the residual, came out **2.4%** off and would not converge | `if/elif` over the three axes in `lattice.stream`. Direction `(-1,-1,0)` from node `(0,0,k)` applied the x rule, skipped the y wall entirely, and read `phi[nx-1,-1,k]` -- which Warp does not bounds-check in release mode. Fixed: **0.041%**, converging in 14,600 steps instead of hitting the 200,000 cap |
| `rotor_reaction` returned a force and no moment | writing the Isaac Lab hover task against it: the quadrotor had one controllable degree of freedom | four rotors on four arms, all their thrust summed at the centre of mass, no attitude moment whatever the differential. `r x F` was missing |
| `FluidWorld.stable()` could not detect a divergence | an under-constrained box diverging within fifteen coupling steps while `stable()` returned True | it read the figure-masked density field, where solid nodes are `nan`, and treated `nan` as "solid" |
| every body in `cases.py` was at twice its intended coordinate | **a voxel-count test.** 114 solid nodes against an exact 905, which is one octant of a sphere in a corner of the box; the symmetry test measured a transverse force 220 times the drag | `add_shape(position=...)` is an offset in the **body's** frame and the world position was passed to both `add_body` and `add_shape`. Seven call sites |
| a domain of six zero-gradient outflows has no density reference and diverges | building the rotor demos | not a code bug but a usable-API one. `Domain` now refuses it and names the fix; the rotor cases use slip sides |

## Milestones

### M0 -- repo study, skeleton, packaging, existing suite green

| item | state |
|---|---|
| `ARCHITECTURE_LIVE.md` written before any code | done |
| extension at `openreynolds/live/`, not in the toolbox (`tests/test_toolbox.py` forbids a toolbox script importing this package) | done |
| `pip install 'openreynolds[live]'`, Apache-2.0 only, licences read off the installed wheels | done |
| core installs and imports **without** the extra | done, checked in a subprocess with Warp hidden (`tests/test_live_optional.py`) |
| CLI: `openreynolds live {cases,run,bench,crosscheck}`, every import deferred into the command body | done |
| agent tool `live_sim`, host-side, withheld when the extra is absent | done |
| existing suite unchanged | **19 failed, 5984 passed, 110 skipped** on `main` before this branch, and the same 19 after. All nineteen are an OCP that no longer exports `TopTools_IndexedMapOfShape` (15 in `test_buildup_cad_export.py`, 2 in `test_toolbox_export_patches.py`), one `cad_brief` index test and one SOCKS proxy test. None is reachable from this branch |

### M1 -- validation against published answers

Status: **running on H100 at the time of writing.** The numbers land in
`benchmarks/live/results/<case>/result.json` and the summary table at the end of this
file is filled from them.

What is already measured, on one CPU core, with the subgrid model off:

| case | target | measured | pass |
|---|---|---|---|
| Poiseuille, 32 cells, exact parabola | max error <= 2e-3 of u_max | **4.07e-4**, converged at residual 3.4e-7 | PASS |
| Poiseuille, u_max against the closed form | 0.05 +/- 1% | **0.0499309**, -0.14% | PASS |
| cavity Re 100, 32 cells, u on x=0.5 | RMS <= 2% of the lid speed, 15 interior points of Ghia Table I | **0.223%** | PASS |
| cavity Re 100, 32 cells, v on y=0.5 | RMS <= 2%, 15 interior points of Ghia Table II | **0.265%** | PASS |
| cavity Re 100 extrema | Ghia: -0.21090, -0.24533, +0.17527 | **-0.21135, -0.24814, +0.17663** (0.21%, 1.15%, 0.78%) | -- |

### M2 -- moving boundaries and two-way coupling

Status: **running on H100 at the time of writing.**

### M3 -- the robot demos

Status: queued behind M2.

### M4 -- performance and docs

| item | state |
|---|---|
| D3Q19 MLUPS at two resolutions on an H100 | done, above: 1095.2 at 128 cells and 1478.7 at 384 |
| both layouts measured on the GPU, GPU default chosen from GPU numbers | done, above |
| docs page in the existing docs structure | `docs/live-sync.md` |
| example scripts | `examples/live/` |
| a rendered figure per case, and a GIF for the unsteady ones | every case draws at least one; the cylinder, the falling sphere and the quadrotor animate |
| CPU-runnable small-grid tests in the main suite | `tests/test_live_cases.py`, 49 passed / 3 GPU skipped |
| GPU tests behind a marker | `needs_a_gpu` skipif plus a registered `gpu` marker |

## What is not measured, and will not be claimed

- **The drone demo at 0.5x real time or better.** It is a device claim. The rate is
  reported by every `quadrotor` run and is in the M3 table; whether it clears 0.5x is
  read from there rather than asserted here.
- **CUDA against CPU bit-identity.** They are not bit-identical and are not expected to
  be: the generated code differs and the layout flips between devices.
  `tests/test_live_cases.py::test_the_gpu_and_the_cpu_agree_on_the_same_small_case`
  asserts agreement to single precision instead.
- **warp 1.18 and newton 1.7.** Neither was installed. The declared range
  `warp-lang>=1.17,<1.20` and `newton>=1.6,<1.9` is tested at its two ends and
  interpolated across the middle, which `docs/live-version-matrix.md` says in those
  words.
- **`SolverMuJoCo` on newton 1.6.1.** Its signature was checked statically; the
  mujoco-warp backend was not installed in that environment, so the coupling was stepped
  there with `SolverXPBD`.
- **Anything about a cylinder's drag from Williamson (1996).** There is no canonical
  published Cd at Re 100; the literature clusters 1.32 to 1.37, and the review that gives
  St = 0.164 gives no drag value at all. The cross-check grades drag against this
  repository's own OpenFOAM run instead, which is what an honest cross-check can do.

## The summary table

Filled from `benchmarks/live/results/*/result.json` once the GPU runs complete. Each row:
the test, its target, the measured value, pass or fail, the device, and the commit the
code was at.

<!-- SUMMARY TABLE -->
