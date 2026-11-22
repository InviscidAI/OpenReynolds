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
| 3,492,000 | 1,000 | 2.88% | -0.3487 | **fails** |

Roughly 0.8% of error per million steps, growing without bound, so the case now runs to
200 convective times rather than to one diffusive time.

**The history is the lesson.** This case used to stop too *early*, on the spin-up
plateau, and came out 8 to 10% short of Ghia; a one-diffusive-time floor was added to
push it past that plateau. The floor cured that failure and caused the opposite one,
which looks **identical** from the outside -- also short of Ghia, also by about 9% -- and
the second was read as more of the first for two days. Four hypotheses were argued and
measured wrong along the way: the TRT magic parameter (BGK, with `omega_minus` twelve
times larger, agrees with TRT to 0.3%), compressibility (Ma 0.061 is *worse* than Ma
0.095, -15.1% against -8.9%), resolution (256 cells worse than 192), and
under-convergence. Only the last was close, with the sign reversed.

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

**The force on a body in a fluid that carries a body force is wrong, and M1 to M4 could
not have caught it.** Measured on a fully submerged 40 mm box in a closed tank with
gravity applied to the fluid, where the answer is exactly `+rho V g` = +626.7 mN upward:

| momentum exchange | measured | |
| --- | --- | --- |
| as M1 to M4 ran it | **-48.66 x rho V g** | the absolute pressure, uncancelled |
| with the reference pressure subtracted | **-0.48 x rho V g** | right order, wrong sign |

The first number is the one that matters for reading the rest of this document: it is
not a regression, it is what the code has always done, and **no case before M5 could see
it**. `DECISIONS_LIVE.md` D9 is why. A single-phase run applies gravity to the *body*,
through Newton, and never to the fluid -- that is exactly why `buoyancy=True` adds
`-rho V g` analytically -- so the fluid has no hydrostatic gradient, the absolute-pressure
contribution cancels over a closed wetted surface, and the drag and momentum-balance
results in M1 to M4 stand. Put gravity in the fluid, as a free surface must, and the
cancellation goes.

Subtracting the reference pressure per link (`2 W[d] rho_ref c`) is kept, and it is
verified not to disturb anything already measured: the OpenFOAM cross-check reads
**Cd 1.329871** with it and **Cd 1.3299** without, the same number to five digits. So it
is a no-op on the validated cases and removes a large artefact in the still-tank case.

**What it removes, and the contradiction that goes with it.** A submerged box in a
closed tank of still fluid with *no* body force must feel exactly zero force, and
measured:

| | force on the box |
| --- | --- |
| no subtraction, as M1 to M4 ran it | **(-30199, -30199, -30199) mN** = 83 x rho V g |
| with the subtraction | (-0.12, -0.12, -0.12) mN = 2e-4 x rho V g |

Equal in all three components, and **identical to four digits when the box is shifted
half a cell**, so it is not a voxelisation asymmetry. For a cube in uniform fluid the
momentum-exchange sum is zero by symmetry -- the formula is right, and was checked line
by line: `cj` is the direction toward the solid, `f_in` is the population heading into
it, and `impulse = cj (f_in + reflected)` is the momentum in minus the momentum out.
A bias equal along all three axes is the signature of one face per axis not
contributing.

**And the cross-check does not show it at all**, which those two facts cannot both
explain. An 83 x rho V g constant on every body would destroy any drag measurement, and
the cross-check's drag is unmoved to five digits. So either the artefact is specific to
a flat-faced body in still fluid, or the isolated harness has a quirk that four separate
checks have not found. **That contradiction is unresolved and is the reason no force fix
is claimed here.**

Separately, and smaller: with the subtraction in place a submerged box in a tank with
gravity reads **-0.48 x rho V g** where it should read `+1`. The residual needed to
reach `+1` is 3.1% more than the subtracted term and the hydrostatic density variation
across that tank is about 2.5%, which hints that the reference should be the **local**
density rather than a global constant. A hint is not a derivation. The interaction of
Guo forcing with the momentum-exchange method needs working out properly: four
mechanisms have been argued and measured wrong in one sitting (the hysteresis band, the
flag conversions, the collision operator, compressibility), and the pattern says stop
guessing.

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
