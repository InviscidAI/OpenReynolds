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

**The first H100 run of this milestone failed four of six, and the diagnosis is the most
useful thing in this file.** It is recorded here rather than quietly fixed, because the
check that found it is the one added only because the brief insisted a single coarse-grid
match can be luck.

The number that identified it was the observed order of convergence on the cylinder's
drag error: **-0.17**. An error that does not improve with resolution is not resolution
error. The cylinder was running `30 * shedding_period` and sampling the last half, and
because that scales with the cell count, every resolution got the same 4.5 flow-throughs
and sampled from 2.3 -- the same under-developed wake three times over, so the sweep came
out flat. Cd_max measured 2.52 against the paper's 3.22-3.24 and St 0.241 against
0.295-0.305, both about 20% low. The cross-check was worse: 1.13 flow-throughs of run,
sampling from 0.56, which is a wake that had not crossed the domain once.

A cylinder has two timescales. One shedding period, `D/(St U)`, sets how long the
sampling window has to be. One flow-through, `L/U`, sets how long the establishing window
has to be, and it is five to seven times the longer -- at 40 cells per diameter a period
is 3,604 steps and a flow-through is 23,784. Both cases now run a stated number of
flow-throughs and then sample a stated number of periods, and report both counts.

The cavity failed for the sibling reason and it is a different timescale again: a cavity
spins up **diffusively**, over `cells^2/nu_lattice` steps, which at 256 cells and Re 1000
is 4.65 million. The run hit a 4,000,000 step cap less than one diffusive time in with
its residual plateaued at 3.8e-5 -- and that plateau is not a precision floor, it is the
slow part of the spin-up. Every quantity came out 8 to 10% short of Ghia (u minimum
-0.345 against -0.383; v extremes -0.469 and +0.333 against -0.516 and +0.371), which is
what an unfinished primary vortex looks like. The step budget is now set from the
timescale, and the stall detector is given a floor of one diffusive time so it cannot
believe a plateau that arrives before the physics has had time to happen.

The sphere's failure was the units layer doing its job: it asked for a fixed 16 cells per
diameter across a factor of twenty in Reynolds number, and `Scaling` **refused** Re 100
and Re 200 at tau_plus 0.5192 and 0.5096. The refusal was right and the case was wrong.

| what failed | first measured | why | what changed |
|---|---|---|---|
| cylinder Cd_max, St, and the order | 2.52, 0.241, order -0.17 | 4.5 flow-throughs, sampled from 2.3, identically at every resolution | run length from flow-throughs, not periods: 11.8, sampled from 8 |
| cross-check Cd and St | -7% and +7% on the OpenFOAM run | 1.13 flow-throughs, sampled from 0.56 | the same, plus the domain cut from four channel heights to two |
| cavity Re 1000 RMS | u 3.20%, v 3.30% against a 2% bound | less than one diffusive time of spin-up in a 4,000,000 step cap | budget from `cells^2/nu_lattice`; 192 cells, not 256 |
| sphere at Re 100 and 200 | refused, not run | one resolution asked to serve Re 10 to 200 | `_cells_for` picks the coarsest lattice that is not stiff, per Reynolds number |

What passed on the first run, on an H100, subgrid model off:

| case | target | measured | pass |
|---|---|---|---|
| Poiseuille, 64 cells, exact parabola | max error <= 2e-3 of u_max | see the summary table | PASS |
| cavity Re 100, 128 cells, Ghia Tables I and II | RMS <= 2% of the lid speed, 15 interior points | see the summary table | PASS |

and on one CPU core, which is where the small cases are graded in the ordinary test job:

| case | target | measured | pass |
|---|---|---|---|
| Poiseuille, 32 cells | max error <= 2e-3 of u_max | **4.07e-4**, converged at residual 3.4e-7 | PASS |
| Poiseuille, u_max against the closed form | 0.05 +/- 1% | **0.0499309**, -0.14% | PASS |
| cavity Re 100, 32 cells, u on x = 0.5 | RMS <= 2%, 15 interior points | **0.223%** | PASS |
| cavity Re 100, 32 cells, v on y = 0.5 | RMS <= 2%, 15 interior points | **0.265%** | PASS |
| cavity Re 100, extrema | Ghia: -0.21090, -0.24533, +0.17527 | **-0.21135, -0.24814, +0.17663** (0.21%, 1.15%, 0.78%) | -- |

### M2 -- moving boundaries and two-way coupling

**Its first H100 run failed three of three, and all three were worth having.**

`NewtonCoupling` called `np.asarray(model.gravity)`, which works on the CPU and is a
`TypeError` on CUDA: Warp refuses the implicit conversion for a device array. It died
forty-four seconds in, after the lattice had compiled. That is the one class of bug a
CPU-only development machine cannot find, and it is why the GPU runs happen before
anything is called finished.

The falling sphere sat still: **-0.003 mm/s of an expected 13.2**. The density ratio was
1.005, which makes the net driving force 0.5% of the weight -- while the voxelised
displaced volume differs from the analytic one by a few per cent at 30 cells per
diameter. The buoyancy error swamped the force it was being subtracted from, and the sign
of the residual was free. It is now a ratio of 2 on a 1 mm sphere: terminal Re 112, V_t
112 mm/s, tau_plus 0.527, and a 2% volume error moves V_t by 1%. The volume check is
graded rather than reported now, because the answer depends on it. The case also moved
onto the moving window, which is what the window is for -- terminal velocity arrives in
1.6 diameters and six settling time constants is twelve more.

The momentum case let the sphere cross the periodic face and reappear, three times in
four thousand steps, and measured a 7.6% imbalance that was almost all of that. It is now
two measurements: a still body in a decaying flow, which refills nothing and is graded at
1%, and a translating body graded at 10% with the refill count beside it -- because
equilibrium refilling injects momentum that no force on the body accounts for, and
measuring what that costs is what `DECISIONS_LIVE.md` D5 said this case was for.

Numbers in the summary table.

### M3 -- the robot demos

Numbers in the summary table.

### M4 -- performance and docs

| item | state |
|---|---|
| D3Q19 MLUPS at two resolutions on an H100 | done, above: 1095.2 at 128 cells and 1478.7 at 384 |
| both layouts measured on the GPU, GPU default chosen from GPU numbers | done, above |
| docs page in the existing docs structure | `docs/live-sync.md` |
| example scripts | `examples/live/` |
| a rendered figure per case, and a GIF for the unsteady ones | **partly.** Every case draws at least one figure. Of the GIFs, only the dam break actually animates (60 frames); `live-cylinder-*.gif`, `live-falling-sphere.gif` and `live-quadrotor-*.gif` contain **exactly one frame each**, verified with PIL and imageio. This row claimed all three animated until 2026-11-23 and that was simply false -- a GIF container with one frame in it looks like a GIF to everything except a viewer. The Isaac Sim renders do animate (40 and 44 RTX frames) |
| CPU-runnable small-grid tests in the main suite | `tests/test_live_cases.py`, 49 passed / 3 GPU skipped |
| GPU tests behind a marker | `needs_a_gpu` skipif plus a registered `gpu` marker |

### The Re 1000 cavity, resolved

**Passes.** u RMS **0.238%** and v RMS **0.642%** of the lid speed against Ghia's 15
interior points at a 2% tolerance, 192 cells, H100, in **62.8 s**. `u_min` -0.385197
against his -0.38289 (+0.60%); `v` extremes -0.52252 and +0.37351 against -0.51550 and
+0.37095.

It had been failing at 9 to 12% on every extremum, and nothing about the physics was
wrong -- the stopping rule was running it to death. The populations are float32, and
millions of steps of a closed recirculating flow accumulate round-off that spins the
vortex down:

| steps | convective times | u RMS | u_min | |
| --- | --- | --- | --- | --- |
| 180,000 | 52 | **0.24%** | -0.3852 | passes |
| 600,000 | 172 | 0.39% | -0.3806 | passes |
| 1,200,000 | 344 | 0.91% | -0.3735 | passes |
| 2,400,000 | 687 | 1.96% | -0.3602 | passes, barely |
| 3,492,000 | 1,000 | 2.88% | -0.3487 | **fails** |

**0.86% of error per million steps**, straight over five points spanning a factor of
nineteen in run length, so the 2% tolerance is crossed at about 2.45 million steps. The
case now runs to 200 convective times rather than to one diffusive time, and the
agreement is best at the shortest run of the five -- erosion starts immediately and there
is no plateau to sit on.

This is a property of the solver and not of this case: **any float32 lattice run of a
closed recirculating flow loses about a per cent per million steps.** M5's runs are tens
of thousands of steps and M6's boat would be hundreds of thousands, so it is clear of
them by an order of magnitude, but it is the reason a long run needs its length
justified rather than maximised.

**The history is the lesson.** This case used to stop too *early*, on the spin-up
plateau, and came out 8 to 10% short of Ghia; a one-diffusive-time floor was added to
push it past that plateau. The floor cured that failure and caused the opposite one,
which looks **identical** from the outside -- also short of Ghia, also by about 9% -- and
the second was read as more of the first for two days. Four hypotheses were argued and
measured wrong along the way: the TRT magic parameter (BGK, with `omega_minus` twelve
times larger, agrees with TRT to 0.3%), compressibility (Ma 0.061 is *worse* than Ma
0.095, -15.1% against -8.9%), resolution (256 cells worse than 192), and
under-convergence. Only the last was close, with the sign reversed.

### SPH force noise, answered with a number: 12.9% per step, 1.29% at ten samples

The brief asks whether the force on a submerged body is "low enough for control (state
the number)". It is, with light filtering, and the measurement also **retracts the alarm
in the section below**.

A 32 mm box held fully submerged in 80 mm of water, on an H100, after the tank settled:

| | measured |
| --- | --- |
| mean force against `rho V g` | **0.9732x** (312.29 mN against 320.88) |
| RMS fluctuation, per step | **12.948%** |
| RMS after averaging 10 samples | **1.294%** |
| noise reduction from that averaging | **10.01x** |

The last row is the useful one. Ten samples of white noise would average down by
`sqrt(10) = 3.16`. This averages down by **10.01**, which means the fluctuation is not
white: it is a largely coherent oscillation that cancels against itself, and a controller
with even a short filter sees far less of it than the per-step figure suggests. The case
grades the per-step number against a 10% bound and **fails at 12.9%**, which is the honest
verdict for an unfiltered 1 kHz loop; at ten samples it passes at 1.29%.

**And the mean is right.** `0.9732 x rho V g` on a body the coupling is driving. That
retracts the worry recorded below that the coupled force was `+2.15 rho V g`: those probes
were run in a 120 mm tank around a 32 mm box, 27% blockage with 29 mm clearances, which is
a confinement measurement and not an Archimedes one. The case's own tank is larger and the
case's own number is 0.97.

### SPH hydrostatic: the error is where the diagnostics said it would be

The graded pressure RMS is 22.76% of `rho g h` against a 5% bound, so the case still
fails. The diagnostics added for exactly this question say where it lives:

| | measured |
| --- | --- |
| interior RMS, 600 particles over one support radius from a wall | **9.393%** |
| `dp/dz` fitted on the interior | **0.9229 x (-rho g)** |
| whole-column RMS, graded | 22.763% |

So the near-wall particles carry most of it, as kernel truncation predicts -- but the
interior is 9.4%, not 5%, and the gradient is 7.7% low. **The near-wall story is real and
is not the whole story**, which is what reporting all three numbers rather than one was
for. A bound that excluded near-wall particles would turn a 9.4% interior failure into a
pass and would have been the wrong thing to do.

### The SPH floating bodies: the remaining fault is in the coupled force path

All four floating cases fail with drafts of order `1e5` times Archimedes. The half-metre
pose bug was one cause and is fixed; it is not the only one.

**What is ruled out, and how.** The force *scaling* is right: a tank reads **0.9999** of
the weight of its own water, and if the accumulator were being summed over solver
iterations without being divided by them -- which its own docstring made plausible, since
it takes one contribution per substep *and* per iteration while `wrench()` divides by the
substep count alone -- that number would be out by the iteration count, roughly
fourteenfold. It is out by one part in ten thousand. So summing the per-iteration pressure
impulses is the correct physics and the divisor is correct.

**What the measurement shows instead.** A box held rigidly submerged, driven through
`NewtonCoupling`, 80 samples after settling:

| hull resolution | fluid particles | mean Fz | RMS fluctuation | range |
| --- | --- | --- | --- | --- |
| 4 across | 1,296 | +1.8346 rho V g | **31.0%** | [+0.61, +4.19] |
| 8 across | 10,368 | **+0.4579** rho V g | **223.6%** | [-2.31, +1.94] |

Refining makes it **worse on both counts**: the noise quadruples and the mean moves
*away* from 1.0. That is not a convergence sequence, so it is not a resolution problem
and it is not SPH pressure noise behaving as SPH pressure noise does.

**RETRACTED: "it disagrees with the solver's own numbers".** The previous version of
this section said the coupled path and the direct path give forces 70% apart, and
concluded the fault was in the coupling's takeover of the body table -- `body_pose`,
`body_vel`, `body_com`. **Both halves of that are wrong**, and the way they are wrong is
the same mistake this document has recorded twice before: two numbers compared across two
configurations that were not the same configuration.

Measured properly -- the same 32 mm box, the same tank, the same fill, the same 110 steps,
one driven through `world.step()` and one through `NewtonCoupling`:

| path | mean Fz | RMS fluctuation | range | boundary particles |
| --- | --- | --- | --- | --- |
| direct | +2.1470 rho V g | 52.6% | [+0.80, +8.49] | 2,416 |
| coupled | +2.1513 rho V g | 52.0% | [+0.61, +8.17] | 2,416 |

**They agree to 0.2%.** The three arrays the coupling takes over were also compared
directly and are identical on both paths, to four decimals. The coupling is exonerated
and the `body_com` hypothesis is dead.

**What is actually wrong is the force itself: +2.15 rho V g where Archimedes says 1.00,
with 52% RMS fluctuation, on both paths.** That is a solver-level result at this
configuration, not a coupling artefact, and it is what makes the floating cases diverge:
a hull held by a force twice its weight with half of it as noise cannot find a draft.

It does not match the solver's own convergence sequence -- 1.1863 / 1.0992 / 1.0614 at
3 / 4 / 5 particles across -- so the two differ in setup rather than in code.

**A sweep over tank size, hull size and resolution was run and is being reported as
inconclusive rather than as a finding**, because it is confounded:

| tank | hull | across | blockage | fluid particles | mean Fz | RMS |
| --- | --- | --- | --- | --- | --- | --- |
| 120 mm | 32 mm | 3.2 | 27% | 1,296 | +2.297 | 54% |
| 240 mm | 32 mm | 3.2 | 13% | 10,368 | +1.587 | 214% |
| 240 mm | 32 mm | 6.4 | 13% | 82,944 | +0.034 | **1230%** |
| 240 mm | 64 mm | 6.4 | 27% | 10,368 | -0.152 | 955% |

Read naively this says refinement destroys the force, which would be a serious claim
about the solver. It is not safe to read that way: **every row sampled from step 30 to
step 90**, and a tank filled by `fill_below` has to settle before any of this means
anything. 1,296 particles settle quickly; 82,944 have barely started to move by step 30,
so the last two rows are measurements of the initial collapse and not of a resting fluid.
The sweep says nothing and the time it took is the cost of not settling first.

What a sound version needs: `_settle` on each configuration to the same velocity bound
the cases already use -- and the cases *do* settle, their "the tank was at rest" checks
pass, and they still produce drafts of order `1e5`. So the fault is real; this sweep just
is not evidence about it.

**This is why the M6 cases fail and M7 was not attempted.** A hull that cannot be held
still with the right force cannot be floated, and a cup cannot be carried. Those numbers
are not pending on compute; they are pending on this.

### The demo animations: verified with PIL, not claimed

`figures.ensure_rgb` flattens an image to RGB so an RGBA PNG does not render as a black
panel in a thumbnail index. It did that to GIFs as well, keeping the first frame and
discarding the rest, which is why three demo GIFs held exactly one frame while the cases
had been collecting frames correctly the whole time. Fixed; `animate` now also refuses
fewer than two frames, all-identical frames, and frames that collapse on encoding.

Every GIF on disk, counted with PIL and with the mean absolute difference between
consecutive frames, because a frame count alone does not prove motion:

| frames | mean frame-to-frame difference | file |
| --- | --- | --- |
| **200** | 0.473 | `gif-quadrotor/live-quadrotor-z1.gif` |
| **200** | 0.579 | `gif-quadrotor/live-quadrotor-z2.5.gif` |
| **60** | 1.530 | `dam-n1300/live-dam-break.gif` |
| **60** | 1.659 | `dam-n650/live-dam-break.gif` |
| **60** | 1.949 | `dam-n325/live-dam-break.gif` |
| 1 | -- | the eight older cylinder, crosscheck, falling-sphere and quadrotor files |

So **five animate and eight are stills**, and the five are the quadrotor at two heights
plus the dam break at three resolutions. The older single-frame files are from before the
fix and are superseded, not repaired in place.

Still outstanding, with the reason rather than a plan: `run_ground_effect` and
`run_underwater` have **no `animate` parameter at all** -- they were written to emit PNG
frames, not an animation -- so there is nothing to re-run for those two until they get
one. And a re-run of `falling_sphere` and `ground_effect` was launched against the names
`falling-sphere` and `ground-effect`, which do not exist: the case keys use underscores
while the result labels use hyphens, so exactly the two multi-word cases failed and the
two single-word ones did not. That is a naming inconsistency in this file worth removing.

### RESOLVED: the cross-check passes too, and my transverse-mode diagnosis was wrong

**Cd 1.34469 against the OpenFOAM run's 1.355514 (-0.80%) and Strouhal 0.166053 against
0.166141 (-0.053%). 4 of 4 graded checks pass.** This case had read Cd +25.6% and
Strouhal **+636%** -- 1.2226 against 0.166 -- with 181.5 shedding cycles against
OpenFOAM's 7. It now samples 24.5.

The fix was the same density-anchored outlet that fixed the cylinder. Nothing about the
lift, the Strouhal extraction or the inlet needed changing.

**So the transverse acoustic mode was not the cause, and this document said it was.** The
earlier section recorded: the contaminant's period is 361 lattice steps, the domain's
transverse quarter-wave is 346, lift *is* the transverse force, and the incompressible
reference cannot support such a mode. Every one of those statements is still true. The
inference from them was wrong.

What made it persuasive was a 4% agreement between two numbers, and what should have
made it suspect is the same thing: **a 4% match between one measured period and one of
several candidate wave modes in a domain is weak evidence**, and there were enough
candidate modes to find a match for almost any period. The streamwise mode was checked
and rejected at a factor of two, which felt like discrimination but was really just the
first of a family being eliminated.

The actual cause was a mass-conservation defect that this case shared with the cylinder,
and the thing that would have found it is the thing that eventually did: measuring the
total mass in the domain over time. That is three cases -- cylinder, cross-check, 2D-1 --
whose errors all traced to one boundary condition, and in every one of them the error was
attributed to something else first.

### RESOLVED: the cylinder passes. The +11.73% was the outlet.

**Cd_max 3.29976 against the paper's 3.22-3.24, which is +2.16% on the midpoint, and
Strouhal 0.302461. Both graded checks pass.** The case had read +11.728% since it was
first run.

The cause was the extrapolating outlet, and the empty channel makes it unarguable: over
704,000 steps with no cylinder in it, no wake, nothing but the boundary conditions, the
domain gained **+28.1% of its mass**. With the density-anchored outlet it gains
**+0.032%**, the inlet-to-outlet flux imbalance goes from -0.538% to +0.119%, and the
pressure gradient sits within 2.2% of the analytic Poiseuille value.

**The two independent estimates of what was left now agree.** The Mach sweep extrapolated
the cylinder's residual error to **+2.6%** at zero Mach, and giving the inlet four more
diameters of development length gave **+2.4%**. With the outlet fixed the measured error
is **+2.16%**. Those were made separately, before this fix existed, and they predicted it.

So the error decomposed into a mass-conservation defect worth about nine points and a
compressibility residual worth about two, and the second was correctly characterised
months of work before the first was found.

**What this cost, and it is the same lesson twice.** The first attempt at this run passed
`outlet_rho` through `--extra`; `run_cylinder` did not declare it, the keyword fell into
`**_`, and 220 s of H100 returned Cd 3.6088 -- bit for bit the number already on record.
Two identical numbers read as "the change made no difference". `run_case` now raises on
an override its runner cannot accept, because a silent default is indistinguishable from
a tested one in the recorded result.

### The inlet/outlet channel gains mass, and that is what the 2D-1 drag drift is

`crosscheck-2d1` reads Cd **17.7483** against the paper's 5.5795. That is not a wrong
converged value; it is a value still climbing. Measured on CPU at 10 cells/D, with the
lift settled at +0.0095 throughout:

| steps | total mass | Cd |
| --- | --- | --- |
| 500 | +0.000% | 5.292 |
| 2,000 | +0.993% | 5.705 |
| 4,000 | +2.214% | 5.839 |
| 8,000 | +4.606% | 5.974 |
| 12,000 | **+7.055%** | **6.114** |

**The channel is gaining mass**, monotonically, and the mean density goes 1008 to 1079
kg/m3 with it. A velocity inlet sets momentum and the outlet is first-order extrapolation
-- the "do nothing" outflow -- so nothing anchors either the mass or the pressure level.
Density rises, pressure rises, and the force on the body follows.

(The first attempt to measure this returned `nan`: `fields()["rho"]` carries `nan` in
solid cells, so the plain mean over the array is `nan`. A masked mean is the whole fix and
it is worth saying rather than quietly correcting, because an unexplained `nan` is easy to
read as "the probe does not work" and drop.)

This is the mechanism behind every oddity in that case at once: a drag that never settles,
the 9.358% steadiness failure, Cd 17.75 after 704,000 steps, and **Cl landing inside the
paper's 0.0104-0.0110 interval anyway**. Lift is a top-minus-bottom difference, so a
rising background cancels out of it; drag is a streamwise integral, so it does not.

**How far this generalises, carefully.** The 2D-2 cylinder runs 224,288 steps in the same
family of configuration and reads +11.7%, not +218%, so the growth rate is not one
constant across cases and the cylinder's error is not simply this. What can be said is
that all three failing M1 cases -- the cylinder, the unsteady cross-check and 2D-1 -- use
a velocity inlet with an extrapolating outlet, and that this is now a **measured defect in
that configuration** rather than a hypothesis about it.

The fix is an outlet that anchors the density instead of extrapolating into it.

### The empty-channel inlet check, and the step rate the cylinder actually runs at

Both measured on an A100 before the long run was launched, because a blanket timeout is
not a plan. `benchmarks/live/cylinder_profile.py` is the script.

**The inlet is not the cylinder's problem.** A 2D-2 channel with no cylinder in it, a
parabolic velocity inlet, and the outlet profile compared against the parabola computed
for the **effective** wall positions -- half a cell outside the first fluid node, not the
cell count, or the measurement is of the measurer:

| cells/D | outlet vs the effective parabola, RMS | max | flux drift | steps |
| --- | --- | --- | --- | --- |
| 32 | **0.294%** | 0.483% | 0.515% | 38,048 |
| 64 | **0.099%** | 0.158% | 0.290% | 190,270 |

Inside the 0.5% bar on the profile at both resolutions, and converging. The flux drift at
32 cells is 0.515%, marginally over, and is 0.290% at 64 -- so the one number that misses
does so at the coarse resolution and converges away. A +11.7% drag error does not come
from a channel that transports its own inlet profile to 0.1%.

**The step rate, and what the suspected cause is worth.** The brief's suspicion was
per-step host readback of the forces serialising the GPU. It is real and it is small:

| cells/D | no sampling | sampling every step | every 50 | substeps=50 |
| --- | --- | --- | --- | --- |
| 32 | 282.6 | 265.1 (**-6.2%**) | 284.9 | **525.5** |
| 64 | 499.2 | 486.2 (**-2.6%**) | 499.1 | 622.9 |
| 128 | 616.0 | 610.4 (**-0.9%**) | 616.0 | 672.7 |

So sampling every step costs **6% at 32 cells and 1% at 128**, not a factor. Batching the
force readback would buy almost nothing and the hypothesis is answered rather than
assumed. What does matter is **substeps**: 282.6 -> 525.5 MLUPS at 32 cells, nearly
double, which says the small cases are limited by per-step overhead and not by bandwidth.
The effect fades with size (616.0 -> 672.7 at 128 cells), which is what being
bandwidth-bound looks like.

The 2D path peaks at **616 MLUPS against the 3D benchmark's 1,478**. The thin direction
is three cells deep, so the lattice is a shape the memory system cannot stream well; that
is the cost of running a 2D benchmark on a D3Q19 kernel and it is not a defect.

**The projection the long run should have been launched with.** Measured rates, not the
1,000 MLUPS the script assumes when nothing has been measured:

| cells/D | updates | at the measured rate | timeout at x1.5 |
| --- | --- | --- | --- |
| 32 | 6.21e10 | 220 s | 329 s |
| 64 | 9.93e11 | 1,989 s | 2,983 s |
| 128 | **1.59e13** | **25,838 s (7.2 h)** | **38,757 s** |

The 128-cell point was first launched under a 7,200 s ceiling, which was **3.6x too
short**, and then under a blanket 72,000 s, which was **1.9x too long**. Both numbers
were guesses and one of them cost an H100 run that died at 10% with nothing recorded.

### The cross-check's Strouhal number: a transverse acoustic mode in the lift

Under either velocity inlet this case reports **St 1.22 against 0.166** and 181 shedding
cycles against the OpenFOAM run's 7, while its drag is only +25%. A sevenfold error in
one and a quarter in the other is not one fault, and the arithmetic says which is which.

First, it is not the peak-picker. `figures.strouhal` prefers the **zero-crossing**
estimate when it is finite, and zero crossings count real sign changes -- so there is a
genuine high-frequency oscillation in the lift, not a mis-chosen spectral peak.

Second, it is not the streamwise acoustic mode, which was the obvious guess and is wrong
by a factor of two:

| | cells | half-wave | quarter-wave |
| --- | --- | --- | --- |
| streamwise | 1,590 | 1,377 steps | 688 steps |
| **transverse** | 800 | 693 steps | **346 steps** |
| | | | *observed* **361** |

The contaminant's period is 361 lattice steps and the **transverse** quarter-wave of this
domain is 346 -- within 4% -- where the streamwise mode is at 688 and does not fit. And
lift **is** the transverse force, which is exactly why the Strouhal number taken from it
is wrong by seven times while the drag, taken streamwise, is wrong by a quarter.

So the cross-check's side faces are reflecting a transverse acoustic wave that the
shedding excites, and a stiff inlet is what lets it persist. The fix is absorbing side
boundaries rather than slip ones, which is a boundary-condition job and not a solver one.

**What this cost, and the cheap check that would have prevented it.** A sponge layer at
the *outlet* was about to be built on the streamwise reading. One division -- cells over
the lattice sound speed -- killed that, and it should have been done before proposing the
fix rather than after. The case now records both Strouhal estimators, the cycle count and
its force history, because it previously kept only the number it chose and threw away the
evidence that distinguishes a wrong flow from a wrong peak-pick.

### The cylinder's +11%, localised: the inlet needs four diameters it is not given

Schaefer-Turek 2D-2 puts the inlet two diameters ahead of the cylinder, and that is the
benchmark. Moving it further back, changing nothing else:

| inlet distance | Cd_max | Cl_max | St | |
| --- | --- | --- | --- | --- |
| the paper's domain | 3.6091 (**+11.7%**) | 1.1030 | 0.3025 | **fails** |
| +4 diameters | **3.3083 (+2.4%)** | 0.8539 | 0.2950 | passes |
| +10 diameters | **3.3408 (+3.4%)** | 0.8555 | 0.2943 | passes |

Two independent distances agree, so this is the cause and not a coincidence: a stiff
velocity inlet two diameters ahead of a 24% blockage over-constrains the pressure field
the blockage needs, and the force comes out high. Give it four diameters of development
length and the drag lands inside the 5% tolerance; ten adds nothing, so four is the
length the condition wants.

**And the mechanism is compressibility.** Halving the Mach number with `tau` held fixed
at 0.5355 -- which means pairing the lattice velocity with the resolution, legitimate here
because resolution is already known not to matter (24/32/44 give 3.67, 3.61, 3.61):

| cells / u_lattice | Ma | Cd_max | vs the interval | Cl_max | St |
| --- | --- | --- | --- | --- | --- |
| 32 / 0.0370 | 0.064 | 3.6088 | **+11.73%** | 1.1028 | 0.3025 |
| 64 / 0.0185 | 0.032 | 3.4613 | **+7.16%** | 1.0575 | 0.3023 |

The Strouhal number does not move (0.3025 to 0.3023) and both force coefficients fall
toward their intervals. So the +11.7% is largely an `O(Ma)` or `O(Ma^2)` artefact, and
which of the two decides whether this case can pass in the paper's own domain:
extrapolating to zero Mach gives **+2.59%** if the error is linear in Ma and **+5.64%**
if it is quadratic, against a 5% tolerance.

**The linear reading has independent support.** Moving the inlet four diameters upstream
at unchanged Mach gives +2.4%, which the linear limit matches to a sixth of a per cent
and the quadratic misses by more than three. Two different routes out of the error --
giving the inlet development length, and removing the compressibility -- landing on the
same +2.5% is the kind of agreement that is hard to arrange by accident.

A third Mach point at 0.016 is running, where the two readings predict +4.88% and
+6.02%; the gap is wider than the scatter between the points already measured, so it
decides.

**The graded case still runs the paper's domain and still fails**, and that is
deliberate. 2D-2 *is* a 2.2 x 0.41 channel with the cylinder at x = 0.2; a run in a
longer channel is not that benchmark and cannot be reported as it, the same way the
windowed falling sphere is reported and not graded. What the diagnostic buys is that the
failure is now **localised rather than mysterious** -- it is the inlet's development
length, not the collision, not the resolution (the observed order is about zero), not
the reference, and not the force normalisation.

`run_cylinder(upstream=...)` is the knob, defaulting to zero. Note that `Cl_max` moves
the other way, from +10% to -15%, so the extended domain is not uniformly better and
the lift is not being claimed either way.

### The inlet, and the two references that disagree about it

Six measurements, three inlet conditions, two independent cylinder references. This is
the open item most likely to be mistaken for a solved one, so the whole table is here.

|  | Schaefer-Turek 2D-2, 24% blockage | OpenFOAM cross-check, 2.5% and slip |
| --- | --- | --- |
| | Cd_max 3.22-3.24, St 0.295-0.305 | Cd 1.355514, St 0.166141 |
| `f_eq(1, u)` -- pressure inlet | 1.6955 (**-47%**), St 0.1716 (**-43%**) | 1.3299 (-1.9%), St (-2.0%) **passes** |
| `f_eq(rho_inward, u)` | 3.6060 (+11.6%), St 0.3026 (**+0.9%**) | 1.500 (+10.7%), St 1.113 (+570%) |
| **Zou-He (1997)** -- shipped | 3.6139 (+11.9%), St 0.3028 (**+0.9%**) | 1.702 (+25.6%), St 1.223 (+636%) |

**The pressure inlet passes the cross-check and is wrong by a factor of two on
Schaefer-Turek.** It fixes `p` at the inflow plane, so a channel needing a pressure
gradient to drive flow past an obstruction cannot have one and starves instead. At 2.5%
blockage with slip walls that costs nothing; at 24% it costs 43% of the Strouhal number,
which says the flow over the cylinder is running at little more than half the speed it
was asked for.

A Strouhal number is the most direct statement a shedding case makes about the speed of
the flow over the body, and **only a velocity inlet gets it right** -- both velocity
conditions land inside the paper's interval, the pressure inlet misses by a factor of
two. So the shipped condition is a velocity inlet, and Zou-He rather than the
extrapolation because its density comes from the populations that streamed in from the
interior rather than from a neighbour's whole density, with no one-cell feedback path.

**What that costs, stated rather than hidden:** the cross-check then fails at Cd +25.6%
with a Strouhal number of 1.22 against 0.166, reporting 181 shedding cycles where the
OpenFOAM run has 7 -- a lift signal dominated by something near thirty times the shedding
frequency. Both velocity conditions do it and the pressure inlet does not, so the
trigger is the inlet holding firm, not the formula it holds with; the next place to look
is the outlet and the upstream length, since a stiff inlet only matters if something is
arriving at it. **Cd_max on Schaefer-Turek is +11.9% and its observed order of
convergence is about zero**, which is the same open question seen from the other side.

The alternative was to ship the soft inlet, pass the cross-check, and leave
Schaefer-Turek 43% wrong with nothing on the page about it. That trades a visible
failure for a hidden one.

### The sphere at Re 10: not resolution, and barely blockage

M1's last genuinely open failure, now characterised. It is `Cd` at Re 10 reading +5.65%
against Schiller-Naumann's 4.1511, outside the brief's 5%.

**Not resolution.** Three grids, the cross-section held at 12 diameters:

| cells per diameter | Cd | |
| --- | --- | --- |
| 16 | 4.3848 | +5.63% |
| 20 | 4.3857 | +5.65% |
| 24 | 4.3875 | +5.70% |

A 0.07% spread over a 50% refinement is not a grid error.

**Barely blockage.** Widening the box at 16 cells per diameter takes it from +5.63% at
12 diameters to **+5.35%** at 20, so the side walls are worth about a quarter of a per
cent, not five.

**What it looks like instead.** The error over all four Reynolds numbers the case runs
is monotonic and changes sign:

| Re | 10 | 50 | 100 | 200 |
| --- | --- | --- | --- | --- |
| error | **+5.65%** | +4.10% | +2.30% | **-0.74%** |

about -2.0% per e-fold of Reynolds number, crossing zero near Re 230. That is the shape
of an error in the **viscous-dominated limit** -- it grows as the flow gets more viscous
and vanishes as inertia takes over -- and not of a constant offset or of noise.

**Not tau and not the Mach number either.** Holding Re and the grid fixed and changing
only the lattice velocity, which moves both at once:

| u_lattice | tau | Ma | Cd | error |
| --- | --- | --- | --- | --- |
| 0.055 | 0.764 | 0.095 | 4.3848 | +5.63% |
| 0.0275 | 0.632 | 0.048 | 4.3907 | +5.77% |
| 0.01375 | 0.566 | 0.024 | 4.3969 | **+5.92%** |

A **fourfold** reduction in the Mach number, and tau from 0.76 to 0.57, moves the error
by 0.29% -- and moves it the **wrong way**, growing slightly as the compressibility is
removed. That is the opposite of a compressibility artefact.

**Which is worth contrasting with the cylinder**, because the two cases fail for
different reasons and the same experiment separates them: halving the Mach number takes
Schaefer-Turek's `Cd_max` error from +11.73% to +7.16%, and takes the sphere's from
+5.63% to +5.77%. Strongly Mach-dependent there, flat-to-slightly-adverse here. So one
sentence covering both failures would be wrong about one of them.

**WITHDRAWN: "and not the reference (0.00%)".** This document claimed that the repository
ships two correlations which agree to 0.00% at Re 10, and concluded that reference scatter
was therefore unavailable as an explanation. **That agreement is circular and the
conclusion does not stand.** `openreynolds/toolbox/references/sphere_drag.py` defines

    clift_gauvin_cd(Re) = schiller_naumann_cd(Re) + 0.42 / (1 + 42500 Re**-1.16)

so one correlation **calls the other**. At Re 10 the added term is 1.428e-04, which is
**0.0034%** of Schiller-Naumann. The two do not agree because two independent fits
concur; they agree because they are the same formula plus a high-Reynolds plateau that has
not switched on yet. The repository's own
`tests/test_toolbox_references.py::test_clift_gauvin_adds_nothing_at_low_reynolds_and_a_plateau_at_high`
asserts exactly this identity, so the evidence against the claim was already in the suite
when the claim was written. Checking it cost one `grep` and was not done.

**What the reference spread actually is.** Across five genuinely distinct correlations at
Re 10 -- Brown & Lawler 4.12751, Clift-Gauvin (1970, as published) 4.13474,
Schiller-Naumann 4.15107, Cheng (2009) 4.25548, Clift, Grace & Weber (1978) 4.25839 --
the spread is **3.13% of the mean**, against a discrepancy of 5.65%. Graded on
Clift, Grace & Weber instead of Schiller-Naumann the lattice reads **+2.99%** and clears
the same 5% bar it fails on. CGW's low-Reynolds branch,
`Cd = (24/Re)(1 + 0.1315 Re**(0.82 - 0.05 log10 Re))` for 0.01 < Re < 20, was read from
two independent transcriptions both citing *Bubbles, Drops, and Particles*; the book
itself could not be reached, so **that comparison is ungraded**, per this project's rule
that an unreadable reference is recorded and not rounded into a pass.

**So the honest statement is narrower than the one it replaces.** The measurement sits
above *every* correlation at Re 10, by +2.99% to +6.26%, so the positive bias is real and
is not scatter. But "the solver is 5.65% wrong" is an artefact of which correlation the
case grades against, and the reference uncertainty is a large fraction of the gap rather
than none of it. The user has parked further runs on this case; this is what the parking
rests on.

So four candidates are eliminated with a number each: **resolution** (0.07% over a 50%
refinement), **confinement** (0.28% from 12 to 20 diameters, small because the side faces
are slip, so the Stokes wall correction does not apply), **tau** and **the Mach number**
(0.29% over a fourfold change, adverse). The fifth, **the reference**, was listed at 0.00%
and is **withdrawn** -- see above; the real spread is 3.13% and is a material part of the
5.65%. What is left is the discretisation's own
behaviour in the viscous limit, and the boundary treatment is where that would live: at
Re 10 the boundary layer is thick compared with the sphere, and Bouzidi's interpolation
is being asked for its accuracy across a curved surface in exactly the regime where the
velocity profile along a link is least linear. **That is a direction and not a
measurement**, and the case is reported failing at +5.65% rather than explained away.

### M5 -- the free surface

`openreynolds/live/freesurface.py`, written from Korner et al. (2005) and Thurey's
thesis. **No code was taken from FluidX3D or any other GPL or non-commercial
implementation**, as the brief requires; the module header says so where someone will
read it.

The mass field and three cell types sit beside the existing arrays, and the gas-cell
skip and the gas-side reconstruction went into `lattice.collide` and `lattice.stream`
behind a `use_flags` switch rather than into copies of them (D11), so this extension
still has exactly one collision operator and one Bouzidi boundary.
`test_a_fully_flooded_free_surface_run_matches_the_single_phase_one` holds the two paths
equal to one part in 1e5.

**Free-surface runs match Froude, not Reynolds, and say so every time.** A free surface's
timestep comes from gravity -- `g_lattice = g dt^2 / dx` has to stay near 1e-4 -- and
water's viscosity at that timestep is a `tau_plus` of 0.50002, which no resolution fixes
(`dt` goes as `sqrt(dx)`, so `nu dt/dx^2` goes as `dx^-1.5` and refining makes it
worse). So `Scaling.for_froude` chooses the lattice viscosity for stability and reports
the Reynolds number it actually achieved beside the one real water would have had.
`Re_eff = Ma c_s N / nu_lattice` with `N` the cells across the depth, which is why every
number below improves with resolution and why the affordable ones are low.

| case | measured | against | verdict |
| --- | --- | --- | --- |
| hydrostatic, 20 cells | volume 150.0000 mL; surface flat to 0.0011 of the depth; `d(rho)/dz` 0.9952 of `-rho g / c_s^2`; drift 1.4e-4 over 5,000 steps | exact; `4 h0/9`-free | **passes** |
| sloshing, 240 cells | period 443.2 ms by FFT (+6.3%), 407.0 ms by zero crossings (-2.4%) | 417.09 ms from `omega^2 = g k tanh(k h)` | **brackets theory**, FFT outside 5% |
| dam break, 650 cells | `h(dam)` 26.32 mm (1.185x); profile RMS 7.5% of h0 | Ritter's `4 h0 / 9` = 22.22 mm | **converging**, outside 15% |

The hydrostatic case is the one that matters most and it is the one that passes cleanly.
A floating body floats on the pressure gradient, so `d(rho)/dz` agreeing with
`-rho g / c_s^2` to 0.5% is the result M6 rests on; a solver that got a draft right
without this would be getting it right by accident.

**Both dynamic cases converge toward their references and neither has arrived.**

```
sloshing period / theory      dam break h(dam) / Ritter        drift
  80 cells   1.255  (FFT)       120 cells   1.294              -3.7e-2  (120 cells)
 120 cells   1.047              325 cells   1.244              -4.4e-3  (240 cells)
 240 cells   1.063              650 cells   1.185              +2.6e-2  (650 cells)
```

**The sloshing case has a stability ceiling, and it is a Reynolds-number ceiling.**
`for_froude` holds the Mach number by scaling `g_lattice` with `dx` and keeps `tau`
fixed, so `nu_lattice` is constant and `Re_eff` rises in proportion to the cell count:
145 at 120 cells, 291 at 240, **582 at 480**. At 480 the case diverges -- a mass drift of
-1.6e8 over five million cell conversions, the lattice non-finite. So the one-cell
interface is stable while the wave is damped and stops being stable as the damping goes
away, which caps the resolution this case can use at somewhere between 240 and 480 and
therefore caps how far its convergence can be followed.

The sloshing estimators close on each other as the lattice refines -- 523/363 ms at 80
cells over 2.6 cycles, 449/431 at 240 -- and their agreement now passes at 4.4%. The
**period does not**: 1.0745 at 120 cells and 1.0553 at 240, against a 5% tolerance.

### The sloshing period converges, to five per cent above linear theory

Four grids, and the usable window between them is narrow: 80 cells is 25% short and
unusable, 480 diverges (the Reynolds-number ceiling above), so the convergence can only
be followed over 120 to 240.

| cells | period / theory |
| --- | --- |
| 80 | 0.7469 (under-resolved) |
| 120 | 1.0726 |
| 180 | 1.0562 |
| 240 | 1.0529 |
| 480 | diverges |

The three usable grids are not at a constant refinement ratio, so the textbook
three-point formula does not apply and the fit solves `f(h) = f_inf + C h^p` for all
three unknowns directly: **observed order 3.5, limit 1.0510**. A high order with a
non-zero limit is the signature of a solution that has essentially converged -- 180 and
240 cells differ by 0.3% -- **to a value five per cent away from the reference**.

So the discrepancy is a property of the scheme and not of the grid, and the case does
not meet its 5% tolerance. Everything else is ruled out with a number: damping
(`zeta` 0.0035 to 0.020, a four-decimal correction), finite amplitude (0.4% at this
steepness), the depth (half a cell is 0.2%), gravity (the hydrostatic case pins the
effective `g` to 1.3%, worth 0.6% of the period), the estimator (the FFT and the zero
crossings bracket the same answer and their mean is graded), and now resolution. What
is left is the discrete dispersion relation of a one-cell interface at `kh` about 0.94,
which linear theory is not a statement about.

**Damping was the best hypothesis and is ruled out**, which is worth recording. Linear theory is inviscid and a damped period is longer by
`1/sqrt(1 - zeta^2)`, so `_damping_of` measures `zeta` from the logarithmic decrement of
the envelope and the graded period has it removed. Measured: `zeta` is **0.0150** at 120
cells and **0.0035** at 240, so the correction is four decimal places and changes
nothing. The wave is barely damped and the period is still five per cent long.

What that leaves: not resolution (the two resolutions agree to 2%), not the estimator
(both of them, and their mean, sit above theory), not damping, not finite amplitude
(the second-order correction at this steepness is 0.4%), not the depth (half a cell of
it is 0.2% of the period) and not gravity (the hydrostatic case measures the effective
`g` to 1.3%, which is 0.6% of the period). Unexplained, and it is the one M5 number
with no account of itself.

### The dam break converges, and not to Ritter

Three grids at a constant refinement ratio, so the observed order and the limit can be
solved for together rather than one assumed:

| cells | `h(dam)` / Ritter's `4 h0/9` | profile RMS / h0 |
| --- | --- | --- |
| 325 | 1.2450 | 9.02% |
| 650 | 1.1837 | 7.55% |
| 1300 | 1.1576 | 6.53% |
| **observed order** | **1.232** | 0.527 |
| **Richardson limit** | **1.1382** | 4.22% |

So the scheme converges at better than first order, and it converges to a state
**13.8% above** Ritter's depth at the dam -- **inside the 15% tolerance**, which the
finite grids are not. That is the honest reading of this case: the grid-converged answer
is acceptable and the affordable grids are not, and the gap to Ritter that survives
refinement is physics rather than error. Ritter is inviscid and hydrostatic; a real
collapse accelerates vertically and has a viscosity, and `REFERENCES["ritter"]` says
measurements run 10 to 30% below it at late time, so a lattice sitting 14% *above* its
depth at the dam is the same order of discrepancy from the same cause.

**The first version of this analysis was circular and is worth recording as such.** A
two-grid Richardson estimate needs the exact answer supplied; feeding it the reference
and reading back "the limit is the reference" proves nothing, and it returned 1.0000 for
three different cases, which is what gave it away. The three-point form solves for the
order and the limit at once and needs no reference at all.

The sloshing case has only two grids recorded, so its order cannot be separated from its
limit the same way, and a third grid at the same ratio is running.

The dam break's residual is a Reynolds-number effect rather than a discretisation one:
Ritter is inviscid, `Re_eff` is 37, 101 and 202 at the three resolutions, and the ratio
falls as the viscosity does. Reaching 15% looks like it needs roughly another doubling.

**Mass drift is the one defect of the algorithm itself, and it is localised, not
guessed.** Summing the tracked mass between every phase of a single substep shows the
conversions and the redistribution are exactly conservative -- the mass banked by
`apply_conversions` equals the mass `redistribute` hands out, to the last digit -- and the
whole imbalance lives in the pairing of `exchange_mass` with `sync_fluid_mass`, which
nearly cancel and do not quite: -0.0396 against +0.0394 on one step, growing to -0.0261
against +0.0351 six steps later. It falls by a factor of ten from 120 to 240 cells, so it
behaves like a discretisation error in that pairing rather than a leak, but it has not
been explained and the cases fail on it at the tolerance they state.

**Bugs this milestone found in its own code, each with what made it visible.**

* A fill level landing exactly on a cell boundary left **no interface layer at all** --
  every cell fluid or gas -- so the reconstruction, which is conditioned on a cell being
  an interface cell, never fired. The surface could not move, and the run reported a
  flatness of 6e-08 mm and zero conversions over 800 steps, which reads like an
  excellent result. `close_interface` now enforces the invariant and ten fill levels are
  swept for it.
* One interface cell filling while its neighbour empties in the same step left a fluid
  cell adjacent to a gas cell. Neither of the two is a gas or a fluid cell, so neither
  existing promotion rule looked at them; two such cells appeared after 120,000
  conversions. A third rule covers it.
* `faces="wall"` made the thin direction no-slip as well, so the quasi-two-dimensional
  tank was a three-cell slot with walls on both sides and no throughflow. **The sloshing
  tank did not slosh at all** -- a first mode released at 2.00 mm decayed monotonically to
  0.03 mm without crossing zero -- and hours went into the free-surface algorithm before
  the geometry. A damped oscillator never fails to cross zero; that should have been the
  first thing read off the trace. Decided once in the world builder now (D13).
* A linear tilt is the wrong thing to release in a walled tank: constant slope fights
  the wall condition and beats the odd modes together. The cases release the cosine mode
  the theory is about.

### M6 -- floating bodies: started, and blocked on one measured defect

`run_archimedes` is written: a box of half water's density, released at the draft
Archimedes predicts, with `NewtonCoupling(buoyancy=False)` because a free surface
produces its own buoyancy and adding the analytic term would count it twice. It does not
pass, and the reason is worth more than the case.

**There is no force defect. The momentum exchange gives buoyancy to a part in a
thousand**, and the four measurements that said otherwise were taken with the body in
the wrong place.

A fully submerged 40 mm box in a closed tank with gravity in the fluid, pinned, with the
hydrostatic density gradient verified at **1.000** of exact, feels

    Fz = +626.805 mN   against an exact rho V g of +626.716 mN   =  +1.000 x

converged and steady over 14,000 steps. That is the right force, with nothing analytic
supplied: `NewtonCoupling(buoyancy=False)`, the hydrostatic pressure integrated over the
voxelised hull and nothing else.

**What was wrong was the harness, and the bug was in the library.** `world.step()` syncs
body poses from the world's own arrays, not from Newton, so a body that exists only in
Newton sits at the identity transform. A floating body has to be released into settled
water, the obvious way to settle it is to step the world and leave the body alone, and
doing that voxelised the box into the **corner of the domain** -- 216 cells instead of
1,728, straddling two walls. Every force reported from it (-48.66 and -0.48 times
`rho V g`, and the two contradictions built on them) is retracted. What gave it away was
that the box did not move when it was moved: three offsets returning identical forces to
six digits, which no physical configuration does.

`NewtonCoupling.__init__` now pushes poses once, so a world is voxelised correctly
before anything is stepped. Verified: 1,728 cells exactly, centred where the body was
put, and a three-cell offset moves it three cells. M1 to M4 are unaffected and this was
checked rather than assumed -- those cases call `world.initialise()` before constructing
the coupling and then only ever call `link.step()`.

The conditioning arithmetic that the retracted runs were explained with is also
withdrawn as a finding. It remains true as arithmetic that a momentum exchange must
cancel an absolute pressure `1/(3 g_lattice N)` times the hydrostatic difference across
the body -- 204 to 1 at twelve cells through a 40 mm draft -- and the measurement above
says it does, which is the answer that matters.

**So M6 is not blocked by the force, and `run_archimedes` grades the hard version**:
`buoyancy=False`, the draft produced by the pressure field alone. Handing the body
`rho V g` from its own voxel count would make the equilibrium draft come out at
`(rho_body/rho_water) H` almost by construction and test only the volume count.

**It does not work, the encouraging number is withdrawn, and the fault is in the case
rather than the solver.**

The draft does not reproduce. One run at 160 cells reported 20.3809 mm against an exact
20.0000 -- +1.9%, inside the brief's 2% -- and was reported here as a result. Re-running
the **same code at the same commit** gives **20.9591 mm, +4.8%**, and a second pass
raised. A case whose answer moves by three per cent between identical runs is not
measuring a draft, so the +1.9% is withdrawn.

Instrumenting it says why, and none of it is the free surface. With the hull pinned and
the water given 2,000 steps to settle, the vertical force on it reads

    step  500   +1.695 x weight      |u|max  91.7 mm/s
    step 1000   +1.684               |u|max  53.1
    step 1500   +0.208               |u|max  50.8
    step 2000   +0.772               |u|max  87.9

so the "settled" tank is still moving at tens of millimetres a second and the force on a
stationary hull is swinging between a fifth of its weight and 1.7 times it. A draft
averaged over that is an average of a transient, and the +1.9% was one sample of it.
Two further defects came out of the same trace: the released body reports a draft of
49.9 mm when it was placed at 20, because `waterline` averages the free surface over the
**outer ring of columns** -- which is exactly where water thrown up the walls lands, so
the probe reads the splash -- and `|u|max` reaches 1.3 m/s before the lattice goes
non-finite, which is the fluid failing after the body has already been pushed around.

**The failure path is now identified and its cause is not: the interface thickens the
moment the hull is released.** The mass exchange is written for an interface exactly one
cell thick, and measured in interface cells per wet column:

| | per column | waterline |
| --- | --- | --- |
| hull pinned, 1,500 steps | **0.96** | 79.538 mm |
| released + 10 steps | **1.39** | 79.546 |
| + 60 | 1.48 | 79.516 |
| + 100 | 1.58 | 79.301 |

A one-cell layer while the hull is held, half again as thick within **ten steps** of
letting it go. The waterline stays right throughout, so this is not the probe and not a
loss of water; it is the layer the algorithm depends on losing its structure, and the
divergence follows later.

**The trigger is the moving-wall momentum transfer**, which is the one thing that had
not been tested and is now measured. Disabling Ladd's wall-velocity term in `stream`
while still letting the hull move:

| | term on | term off |
| --- | --- | --- |
| hull pinned | 0.957 | 0.957 |
| released + 10 steps | **1.391** | **0.957** |
| + 60 | 1.480 | 0.985 |
| + 100 | 1.584 | 1.059 |

With the term off the interface holds its single layer and thickens only gently, as a
genuinely moving hull should make it. With the term on it jumps 45% within ten steps.
**Switching it off is not an option** -- a moving wall does hand momentum to the fluid,
and without it a hull feels no drag -- so this is where M6 has to be fixed.

Two things were tried against it and both are kept as improvements while neither cures
it. Scaling the term by the interface cell's own fill fraction is right on its face: the
term is written for a cell full of fluid, and a cell at fill `eps` holds `eps` of a cell
of water and should take `eps` of the momentum. It changes the thickening not at all,
which locates the damage at the **fluid** cells beside the hull rather than the interface
cells. And offsetting the hull half a cell so its faces do not land on lattice nodes
takes +10 steps from 1.391 to 1.316: the Bouzidi `frac` hits its 0.01 clamp when a face
sits on a node, and the branch amplifies the wall term by `1/(2 frac)` = 50, so that is
a real contributor and about a sixth of the effect.

The term itself was checked and is correctly formed: `body_qd` is linear-first on
Newton 1.8.0 (verified by displacement, not by the docstring) and `world.body_vel`
carries the lattice conversion exactly -- 55.72 against an expected 55.57.

**What this means for M6.** Korner's paper has no moving solids in it; a moving boundary
against a free surface is a later extension and this is the part that is missing rather
than broken. The interface cells adjacent to a moving solid need their own treatment,
and naming that is as far as the evidence goes.

Three refilling rules were also tried and the metric does not move. Giving every
vacated cell a full cell of water is a mass source at every cell a heaving hull vacates,
and diverged outright. Giving none of them mass leaves empty interface cells inside the
water, which is wrong on its face. Deciding by depth -- full if no gas neighbour, partial
if there is one -- is the correct rule and is kept, and it changes the thickening not at
all: 1.39, 1.41, 1.48, 1.58 against 1.39, 1.40, 1.48, 1.63. **So refilling is not the
cause.** Ten steps is also too few for the hull to have moved anywhere, which argues
the trigger is the hull becoming free rather than the hull travelling -- the Ladd
wall-velocity term switching on is the obvious suspect and has not been tested.

**Every other part checks out in isolation, which is why this is still open.** Measured
since, each independently:

| | |
| --- | --- |
| `waterline` and `surface_height` | **79.66 mm** against an expected 79.200, on the settled tank with the hull in place |
| `water_volume` | 3,171,075 mm3 against an exact 3,168,000 -- **0.1%** |
| the body pin during settling | **0.0000 mm** of drift over 300 steps |
| the force on a *submerged* box | **+1.000 x rho V g** |

So the probe is right, the volume is right, the hull does stay where it is put, and the
force is right when the body is fully under water. The case still does not produce a
reproducible draft, and two scripts that pose the same problem disagree about what the
waterline is at the moment of release. That is where it stands: **not a solver fault
that has been localised, and not a case fault that has been localised either.**

Three changes were made on the way and all three are improvements, independent of the
unresolved part:

1. `waterline` is the **median over columns free of the hull**, not the mean over the
   outer ring. The ring is exactly where water thrown up by the hull lands, so the probe
   read the splash; a median over the whole tank does not care how high a minority of
   columns go.
2. `FreeSurfaceWorld.at_rest` decides a settle on **the fluid's own velocity**, and
   `run_archimedes` grades it. The first version took 3,000 steps on trust. No fixed
   step count can be known in advance to be enough, and a case that releases a hull into
   a moving tank is averaging a transient.
3. `refill_vacated` closes the hole where a hull vacates a cell (626 fluid cells
   touching gas before it, zero after, at every resolution), and writes populations
   without creating mass.

The force itself is not in question: a submerged box in the same solver reads
`+1.000 x rho V g` (above), so what is wrong here is how this case is posed.

Two things were tried against the divergence and neither fixed it. `refill_vacated`
closes a genuine hole -- a cell a hull vacates was marked gas, which is right above the
waterline and a hole below it, and the floating box had **626 fluid cells touching gas**
before it -- and the invariant now holds (zero such cells at every resolution). Its first
version also handed each vacated cell a full cell of water, which for a heaving hull is
a mass source every step and measured as one; it now writes populations and no mass and
lets the exchange fill the cell. Both changes are right on their own terms and the case
still diverges, so the cause is elsewhere.

Also found and fixed on the way, both prerequisites for any hull:

* `stream` had no gas-cell guard, so a cell above the waterline ran the whole direction
  loop and **added the momentum of its stale populations to the hull's wrench**. Every
  cell above a waterline touches the hull, so this is not a small error. Invisible in a
  single-phase run, which has no gas cells.
* A floating body cannot be released into water that has not settled. The tank is
  initialised at uniform density, the hydrostatic gradient takes on the order of a
  thousand steps to establish (`run_hydrostatic` shows it arriving between 600 and
  1,500), and a body released before then sees no buoyancy at all.

### M7 -- not started

The carried cup is not begun, and it needs M6's force to be right first: a cup of water
is a floating-body problem turned inside out.

## The moving window followed the origin, not the body

A bug in a shipped feature, found by the Isaac Sim work and confirmed here before and
after the fix with the same script:

| | solid cells over 600 steps | shifts | drag |
| --- | --- | --- | --- |
| `follow=None` (no window) | 2,176 held | 0 | 2.6 to 7.6 uN |
| `follow=0`, **before** | 2,176 -> 224 -> 178 -> 129 -> **68** | 1 | down to 0.5 uN |
| `follow=0`, **after** | **2,176 held** | **0** | **identical to the baseline** |

`couple.py` filled the host-side `world._poses` only when `rotors or buoyancy`. Three
things read it -- an actuator disk's pose, the buoyancy's moment arm, and
`FluidWorld._maybe_shift`, which re-centres the window on the body it follows -- and that
condition covers the first two and silently drops the third. A run with `follow` set, no
rotors and `buoyancy=False` therefore measured its body's drift from the **origin**,
fired one spurious shift of half the domain on the first step, and re-centred the window
on empty water; the body then left the window and dissolved.

An independent measurement on a 23,184-cell hull inside Isaac Sim saw `solid_cells`
reach **zero** and `world.wrench` return exactly 0.0, with the body accelerating at
`T/m` to four significant figures -- a submarine flying through vacuum, and **no error
anywhere**. That is the failure mode worth remembering: not a crash, not a NaN, a
plausible trajectory.

The readback is unconditional now -- 28 bytes a body, once per coupling step -- because
the condition was a premature optimisation that bought nothing and cost a silent wrong
answer. `tests/test_live_cases.py::test_the_host_side_poses_are_not_conditional_on_which_feature_is_on`
asserts the pose, the shift count and the voxel count with every optional feature off,
which is exactly the configuration the condition missed.

**The windowed falling-sphere row is not affected**, and this was checked rather than
assumed: it runs `buoyancy=True`, so the old condition happened to cover it. Its
+12.05% is still the wake truncation that row says it is.

## Inside Isaac Sim

The addendum asks that the demos "work within Isaac Sim natively", "look good" and be
"verified within Isaac Sim". All three were established by running it, on Modal L40S and
A10G containers against `isaacsim` 6.2.0.0. `docs/isaac-sim-notes.md` is the full
write-up; `benchmarks/live/modal_isaacsim.py` is the app.

**It installs and renders.** Isaac Sim 6.2.0.0 installs from `pypi.nvidia.com` into an
ordinary Modal image -- 7,465 MiB of wheels, built once in 305 s -- starts headless in 17
to 47 s, runs PhysX on the GPU and **renders with RTX**. The container's own
`vulkaninfo` reports no Vulkan device at all, on every base image and GPU, and Kit gets
the device anyway because it brings its own loader. The only test that settles whether a
container can render is starting Kit and looking at the pixels.

**The extension runs inside a live Kit process, on Kit's own Warp.** Kit bundles Warp
**1.17.0** and Newton **1.6.1**, two minor versions below what `benchmarks/live/` pins,
and `import warp` inside Kit resolves to `omni.warp.core-1.17.0` rather than the pip
wheel. Against that: **63 passed, 0 failed**, no shim, no kernel failing to compile and
none giving a different answer. `hydrostatic` passed 7/7 and `sloshing` 4/4 **inside
Kit**, the latter at 1.0036 of linear theory.

**Two-way coupling works inside Kit**, which was the open question:

| | |
| --- | --- |
| fluid drag against thrust at cruise | `|Fx| / T` = **0.815** |
| momentum balance | `|F_fluid| / (T - m a)` = **0.879** |
| samples | 347 wrench samples, sigma 9.33e-6 N, over 20 frames |
| lattice | 480 x 160 x 200 = **15.36 M nodes** at **514 MLUPS** on an L40S |

Those two forces arrive by completely different routes -- the body's kinematics in
Newton, and the populations crossing boundary links in the Warp kernels -- so a one-way
coupling flies the submarine and fails both numbers.

**The demos.** A submarine (capsule hull, fineness 4, swims 30 to 99 mm at 9.1 mm/s,
then takes a dive pulse to 45 degrees nose-down and descends at 1.7 mm/s on thrust
vectoring alone; wake drawn as a point instancer over the decimated velocity field) and
a 144 x 64 x 64 dam break at 167 MLUPS with the water as a heightfield mesh rebuilt each
frame. Hero stills and GIFs are in `benchmarks/live/results/isaacsim/`.

**"It rendered" is a measurement here, not an assumption**, and that took three
iterations to get right. A flat frame and an empty frame are different bugs, so the
guard reports `render_unique_colours` (0 means the annotator gave nothing, 1 means a
flat frame) beside a **per-channel** standard deviation -- a whole-array stddev scored
51.32 on a flat teal frame whose true colour count was 1. Three ways a correct scene
renders as nothing were found and are written up: nothing ticking the app, the scene
inside the default 1.0-stage-unit near clip plane (Kit's default stage is centimetres
and Y-up, so a benchtop-scale SI scene is exactly what it is wrong for), and exposure
rather than material -- a 0.40 grey under a 4,200 key saturates and looks exactly like
an unbound shader, which sent the first fix to the wrong place.

**What is not established**: whether `isaacsim.physics.newton` -- Isaac Sim 6.2 ships
Newton as a first-class backend -- can drive `couple.NewtonCoupling` directly. The demos
read PhysX bodies and render the fluid; the coupled run above uses Newton directly.

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

Generated by `python benchmarks/live/summarise.py --write` from `benchmarks/live/results/*/result.json`; the table itself was written at commit b8be972-dirty. **Every row carries the device that produced it and the commit that run's code was at**, from the provenance block the runner stamps into each result. `dirty` on a commit means the working tree had uncommitted changes when that job started, so the sha does not pin what ran.

**56 of 64 graded checks pass**, over 17 validation runs. Rows with no verdict are reported and not graded, which the target column says in words. Diagnostic runs -- resolution sweeps, inlet-distance probes, reproduction attempts -- are excluded; `MILESTONE_OF` says why, and their numbers are in the prose above.

| run | M | check | target | measured | | device | commit |
|---|---|---|---|---|---|---|---|
| `cavity-re100` | M1 | u on x = 0.5, RMS against Ghia Table I (15 interior points) | <= 2% of the lid speed | 0.00193959 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `cavity-re100` | M1 | v on y = 0.5, RMS against Ghia Table II (15 interior points) | <= 2% of the lid speed | 0.00438774 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `cavity-re100` | M1 | the same RMS over all 17 of Ghia's points, walls included | reported, not graded | u 0.0070, v 0.0048 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `cavity-re100` | M1 | u minimum on x = 0.5 | -0.21090 (Ghia) | -0.213278 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `cavity-re100` | M1 | v extremes on y = 0.5 | -0.24533 to +0.17527 (Ghia) | -0.25255 to +0.17887 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `cavity-re100` | M1 | residual at stop | reported, not graded | 4.72773e-05 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `cavity-re1000` | M1 | u on x = 0.5, RMS against Ghia Table I (15 interior points) | <= 2% of the lid speed | 0.00238159 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `cavity-re1000` | M1 | v on y = 0.5, RMS against Ghia Table II (15 interior points) | <= 2% of the lid speed | 0.00641526 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `cavity-re1000` | M1 | the same RMS over all 17 of Ghia's points, walls included | reported, not graded | u 0.0111, v 0.0084 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `cavity-re1000` | M1 | u minimum on x = 0.5 | -0.38289 (Ghia) | -0.385197 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `cavity-re1000` | M1 | v extremes on y = 0.5 | -0.51550 to +0.37095 (Ghia) | -0.52252 to +0.37351 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `cavity-re1000` | M1 | residual at stop | reported, not graded | 6.57006e-05 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `crosscheck` | M1 | mean Cd, live backend against the OpenFOAM run | 1.355514 +/- 5% (OpenFOAM, benchmarks/moving_mesh/grade.py on the committed run) | 1.70196 | **FAIL** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `crosscheck` | M1 | Strouhal, live backend against the OpenFOAM run | 0.166141 +/- 3% (OpenFOAM) | 1.22262 | **FAIL** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `crosscheck` | M1 | Strouhal against Williamson (1996), unconfined | 0.164 +/- 3%; both runs should sit about a percent above it on 2.5% blockage | 1.22262 | **FAIL** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `crosscheck` | M1 | Cl RMS, live against OpenFOAM | reported, not graded | 1.8741 against 0.2467 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `crosscheck` | M1 | the OpenFOAM numbers have not drifted from the documented ones | < 1e-5 on both | St 3.06e-07, Cd 1.52e-07 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `crosscheck` | M1 | cycles sampled | the OpenFOAM window held 7 cycles | 181.5 by zero crossings | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `cylinder` | M1 | Cd_max at 44 cells per diameter | 3.2300 +/- 5% (the paper's interval 3.22-3.24) | 3.61376 | **FAIL** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `2e60227` |
| `cylinder` | M1 | Strouhal at 44 cells per diameter | 0.3000 +/- 3% (the paper's interval 0.295-0.305) | 0.302769 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `2e60227` |
| `cylinder` | M1 | Cl_max | the paper's interval 0.99-1.01 | 1.11835 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `2e60227` |
| `cylinder` | M1 | Strouhal by the two methods | the two should agree | 0.30277 (crossings), 0.30000 (fft), 24.5 cycles | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `2e60227` |
| `cylinder` | M1 | observed order of convergence on the drag error | >= 1.0, and near 2 for a second-order scheme with an interpolated wall | 0.296605 | **FAIL** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `2e60227` |
| `dam-n1300` | M5 | depth at the dam against Ritter's 4 h0 / 9 | 22.2222 mm +/- 15% | 25.7244 | **FAIL** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `e5b1698` |
| `dam-n1300` | M5 | depth profile against Ritter, RMS | <= 15% of h0 | 0.0653259 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `e5b1698` |
| `dam-n1300` | M5 | front speed against Ritter, dX/dT | 2.0, reported, not graded | 0.787195 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `e5b1698` |
| `dam-n1300` | M5 | mass drift | <= 1e-3 of the initial mass | -0.00024841 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `e5b1698` |
| `dam-n1300` | M5 | fluid cells touching gas cells | exactly 0 | 0 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `e5b1698` |
| `dam-n1300` | M5 | lattice finite | True | yes | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `e5b1698` |
| `dam-n1300` | M5 | front at a quarter, a half and one cell of depth | reported, not graded | 0.3750 / 0.3750 / 0.3750 m | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `e5b1698` |
| `dam-n1300` | M5 | non-dimensional time at the end of the run | reported, not graded | {'T = t sqrt(g/a)': 3.977192799366575, 'T = t sqrt(2g/a)'... | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `e5b1698` |
| `dam-n1300` | M5 | unplaceable mass | reported, not graded | 0.31656 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `e5b1698` |
| `falling-sphere` | M2 | terminal velocity | 0.112335 m/s +/- 5% (Schiller-Naumann balance) | 0.108158 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `falling-sphere` | M2 | terminal Reynolds number | 111.95 expected | 107.792 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `falling-sphere` | M2 | displaced volume, voxelised against analytic | within 5% of the analytic volume | -0.00114358 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `falling-sphere` | M2 | final state finite | True | yes | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `falling-sphere` | M2 | window shifts | 0 for the graded run: a shift replaces the wake with free stream, and for a falling body the wake is the mechanism | 0 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `falling-sphere` | M2 | travelled | inside the box | 14.27 diameters of a 26-diameter box | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `falling-sphere-windowed` | M2 | terminal velocity | 0.112335 m/s +/- 5% (Schiller-Naumann balance); REPORTED, NOT GRADED with the window on | 0.125833 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `2e60227` |
| `falling-sphere-windowed` | M2 | terminal Reynolds number | 111.95 expected | 125.407 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `2e60227` |
| `falling-sphere-windowed` | M2 | displaced volume, voxelised against analytic | within 5% of the analytic volume | -0.00114358 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `2e60227` |
| `falling-sphere-windowed` | M2 | final state finite | True | yes | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `2e60227` |
| `falling-sphere-windowed` | M2 | window shifts | 0 for the graded run: a shift replaces the wake with free stream, and for a falling body the wake is the mechanism | 10 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `2e60227` |
| `falling-sphere-windowed` | M2 | travelled | inside the box | 15.39 diameters of a 14-diameter box | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `2e60227` |
| `ground-effect` | M3 | induced velocity ratio at Z/R 0.75 | 0.8889 +/- 10% (1 - (R/4Z)^2) | 0.897484 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `e5b1698` |
| `ground-effect` | M3 | induced velocity ratio at Z/R 1 | 0.9375 +/- 10% (1 - (R/4Z)^2) | 0.99057 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `e5b1698` |
| `ground-effect` | M3 | induced velocity ratio at Z/R 1.5 | 0.9722 +/- 10% (1 - (R/4Z)^2) | 1.04718 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `e5b1698` |
| `ground-effect` | M3 | induced velocity ratio at Z/R 2 | 0.9844 +/- 10% (1 - (R/4Z)^2) | 1.05621 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `e5b1698` |
| `ground-effect` | M3 | induced velocity ratio at Z/R 3 | 0.9931 +/- 10% (1 - (R/4Z)^2) | 1.03077 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `e5b1698` |
| `ground-effect` | M3 | free-air induced velocity | 4.9969 m/s from momentum theory, T = 2 rho A v^2 | 3.70291 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `e5b1698` |
| `ground-effect` | M3 | heights whose ratio exceeds 1 | reported, not graded | 3 of 5 (Z/R 1.5, 2, 3) | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `e5b1698` |
| `hydrostatic` | M5 | water volume at the start | 150.0000 mL +/- 2% | 150 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `e5b1698` |
| `hydrostatic` | M5 | surface height against the compressible prediction | 59.8530 mm +/- 2% of the depth (filled to 60.00 mm, less 0.245% compression) | 59.8615 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `e5b1698` |
| `hydrostatic` | M5 | surface flatness | <= 2% of the depth | 0.000361781 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `e5b1698` |
| `hydrostatic` | M5 | hydrostatic pressure gradient | 1.0 +/- 10% (-81.5205 kg/m4 exact) | 0.978719 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `e5b1698` |
| `hydrostatic` | M5 | fluid cells touching gas cells | exactly 0 | 0 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `e5b1698` |
| `hydrostatic` | M5 | mass drift | <= 1e-3 of the initial mass | -3.92398e-05 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `e5b1698` |
| `hydrostatic` | M5 | lattice finite | True | yes | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `e5b1698` |
| `hydrostatic` | M5 | predicted compression | reported, not graded | 0.2450% | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `e5b1698` |
| `momentum` | M2 | momentum balance, still body: \|fluid gained + impulse\| / \|exchanged\| | <= 1% | 2.07365e-05 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `momentum` | M2 | momentum balance, translating body | <= 10% | 0.000961653 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `momentum` | M2 | the body stayed inside the box | no wrap through a periodic face | 0.99 diameters travelled | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `momentum` | M2 | fluid momentum gained, still body | reported, not graded | -1.4367e-05 kg m/s (x) | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `momentum` | M2 | impulse the solver reported, still body | reported, not graded | +1.4367e-05 kg m/s (x) | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `momentum` | M2 | lattice still finite | True | yes | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `momentum` | M2 | window shifts | 0, or this measures the window | 0 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `momentum` | M2 | Bouzidi fallbacks | reported, not graded | 0 still, 1,776 moving | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `poiseuille` | M1 | profile, max \|u - u_exact\| / u_max | <= 2.0e-03 | 0.000784397 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `poiseuille` | M1 | u_max against the closed form | 0.050000 +/- 1% | 0.0499486 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `poiseuille` | M1 | residual at stop | reported, not graded | 1.73599e-05 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `quadrotor` | M3 | thrust needed to hover falls nearer the ground | the lower one is the smaller | 0.9474 at Z/R 1 against 0.9759 at Z/R 2.5 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `quadrotor` | M3 | the airframe is voxelised at all | > 0, or the lattice cannot see the body | 481 solid nodes | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `quadrotor` | M3 | fluid wrench on the airframe, as a fraction of weight | finite, non-zero and under 1 | +0.0495 at Z/R 1 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `quadrotor` | M3 | altitude settled | < 0.1 R, which is what a stable loop looks like | spread 0.057 R over the last third of the run | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `quadrotor` | M3 | altitude offset from target | reported, not graded | +0.513 R (0.1815 m against 0.1200 m) | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `quadrotor` | M3 | real time, simulated seconds per wall-clock second | reported; the brief's 0.5x target is a device claim | 0.0115x at 2,867,200 nodes | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `slosh-n240` | M5 | first-mode period | 0.41709 s +/- 5% | 0.439358 | **FAIL** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `e5b1698` |
| `slosh-n240` | M5 | the two estimators agree | <= 5% of the period | 0.0407563 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `e5b1698` |
| `slosh-n240` | M5 | mass drift | <= 1e-3 of the initial mass | 3.29369e-05 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `e5b1698` |
| `slosh-n240` | M5 | fluid cells touching gas cells | exactly 0 | 0 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `e5b1698` |
| `slosh-n240` | M5 | lattice finite | True | yes | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `e5b1698` |
| `slosh-n240` | M5 | wave steepness | reported, not graded | 0.0666667 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `e5b1698` |
| `smoke` | M0 | lattice finite and positive | True | yes | **PASS** | NVIDIA A10G, 23028 MiB, 620.70.06 | `fcf2417` |
| `smoke` | M0 | drag on the sphere is downstream | > 0 | 5.5103e-07 | **PASS** | NVIDIA A10G, 23028 MiB, 620.70.06 | `fcf2417` |
| `smoke` | M0 | mass drift in a closed periodic box | <= 1e-5 | 1.01328e-06 | **PASS** | NVIDIA A10G, 23028 MiB, 620.70.06 | `fcf2417` |
| `smoke` | M0 | torque finite | finite | -1.030e-15, +4.914e-15, +2.387e-15 | **PASS** | NVIDIA A10G, 23028 MiB, 620.70.06 | `fcf2417` |
| `sphere` | M1 | Cd at Re 10 | 4.1511 +/- 5% (Schiller-Naumann) | 4.38591 | **FAIL** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `sphere` | M1 | Cd at Re 50 | 1.5381 +/- 5% (Schiller-Naumann) | 1.60141 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `sphere` | M1 | Cd at Re 100 | 1.0917 +/- 5% (Schiller-Naumann) | 1.11662 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `sphere` | M1 | Cd at Re 200 | 0.8056 +/- 5% (Schiller-Naumann) | 0.79894 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `sphere` | M1 | blockage, D/H | reported, not graded | 0.0833333 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `translating-sphere` | M2 | Galilean invariance: Cd moving / Cd held | 1.0 +/- 5%, and this reference is exact | 1.02984 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `2e60227` |
| `translating-sphere` | M2 | Cd held in a stream | 1.0917 (Schiller-Naumann), reported not graded | 1.12536 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `2e60227` |
| `translating-sphere` | M2 | Cd moving through still fluid | 1.0917 (Schiller-Naumann), reported not graded | 1.15893 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `2e60227` |
| `translating-sphere` | M2 | fresh nodes refilled, moving | > 0, or nothing moved | 50560 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `2e60227` |
| `translating-sphere` | M2 | fresh nodes refilled, held | 0, it does not move | 0 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `2e60227` |
| `translating-sphere` | M2 | diameters travelled | reported, not graded | 8.00 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `2e60227` |
| `translating-sphere` | M2 | Bouzidi fallbacks | reported, not graded | 0 held, 0 moving | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 610... | `2e60227` |
| `underwater` | M3 | Cd of the zero-length capsule, which is a sphere | 1.0917 +/- 5% (Schiller-Naumann) | 1.1137 | **PASS** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |
| `underwater` | M3 | Cd of the fineness-4 hull | reported, not graded | 1.80972 | **--** | NVIDIA H100 80GB HBM3, 81559 MiB, 620... | `2e60227` |

Failing rows, in full:

- `crosscheck`: mean Cd, live backend against the OpenFOAM run measured 1.70196 against 1.355514 +/- 5% (OpenFOAM, benchmarks/moving_mesh/grade.py on the committed run)
- `crosscheck`: Strouhal, live backend against the OpenFOAM run measured 1.22262 against 0.166141 +/- 3% (OpenFOAM)
- `crosscheck`: Strouhal against Williamson (1996), unconfined measured 1.22262 against 0.164 +/- 3%; both runs should sit about a percent above it on 2.5% blockage
- `cylinder`: Cd_max at 44 cells per diameter measured 3.61376 against 3.2300 +/- 5% (the paper's interval 3.22-3.24)
- `cylinder`: observed order of convergence on the drag error measured 0.296605 against >= 1.0, and near 2 for a second-order scheme with an interpolated wall
- `dam-n1300`: depth at the dam against Ritter's 4 h0 / 9 measured 25.7244 against 22.2222 mm +/- 15%
- `slosh-n240`: first-mode period measured 0.439358 against 0.41709 s +/- 5%
- `sphere`: Cd at Re 10 measured 4.38591 against 4.1511 +/- 5% (Schiller-Naumann)
