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
| a rendered figure per case, and a GIF for the unsteady ones | every case draws at least one; the cylinder, the falling sphere and the quadrotor animate |
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

The sloshing estimators also close on each other as the lattice refines -- 523/363 ms at
80 cells, 437/396 at 120, 443/407 at 240 -- which is the shape of a converging
measurement rather than a wrong one. The measured period sits **above** linear theory at
every resolution, and that is the direction a finite amplitude pushes it: the default
release is 6.7% of the depth and linear theory is an expansion in that. An amplitude
sweep at fixed resolution is running to separate the two, and until it reports, the 5%
tolerance is **not met and not widened**.

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

**It does not work yet, and the one encouraging number from it is unconfirmed.** One run
at 160 cells -- sixteen through the draft -- reported a draft of **20.3809 mm against an
exact 20.0000, +1.9%**, inside the brief's 2%, with wobble 1.3% of the draft and mass
drift 3.1e-4, and then went non-finite after the measurement window. Every run since
has diverged outright, at 80, 120 and 160 cells, across three different commits. Git
confirms the +1.9% came from code that had both the pose fix and the pinned settle, so
it is not obviously an artefact of either -- but **a single run of a case that later
diverges is not a result**, and it is being re-run twice at its own commit before
anything is concluded from it. If it does not reproduce it will be withdrawn like the
four force measurements above.

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
