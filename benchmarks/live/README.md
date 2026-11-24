# The live backend: a D3Q19 lattice Boltzmann fluid, coupled to Newton

The cases `openreynolds/live/` is validated on, what each one is worth, and where the
remaining difference comes from. `grade.py` runs them:

```
python benchmarks/live/grade.py list
python benchmarks/live/grade.py poiseuille --resolution 32 --device cpu
python benchmarks/live/grade.py cavity --resolution 128 --reynolds 1000
python benchmarks/live/grade.py crosscheck
python benchmarks/live/grade.py all --milestone M1
```

**This grader solves**, like `../lbm/grade.py` and unlike `../felix/grade.py` and
`../moving_mesh/grade.py`. Those two read a finished run, because the cases they grade
cost a cluster an hour. This solver is in this repository, so committing its output would
replace a check that re-derives the number with a number nobody can re-derive -- the
argument `../lbm/README.md` makes for the same reason.

The cross-check is the one exception and it is the interesting one: it **solves one side
and reads the other.**

## Two devices, and which numbers came from which

The physics here is device-independent and the rates are not.
`tests/test_live_cases.py::test_the_gpu_and_the_cpu_agree_on_the_same_small_case` holds
the CPU and the GPU to single precision on the same case, so an agreement with a
published value means the same thing from either. What differs is what is affordable:
Warp's CPU backend is single-threaded, so a lid-driven cavity at the resolution Ghia used
is of order 10^11 lattice updates -- seconds on an H100, hours on one core.

So the small cases are graded on a CPU in the ordinary test job and the rest on an H100
through `modal_run.py`. **Every number below names its device**, and `PROGRESS_LIVE.md`
carries the job identifiers.

## Why these cases and not others

Each one is here because it fails differently from the others.

**`poiseuille` is the only exact one, and it has already earned its place twice.** With
two relaxation times and `(tau_+ - 1/2)(tau_- - 1/2) = 3/16` the bounce-back wall sits on
the lattice link midpoint at any viscosity, so this scheme has no discretisation error on
this flow and the answer is the analytic parabola to the convergence residual.
`../lbm/README.md` grades the two-dimensional version of exactly that claim, at three
viscosities, with single-relaxation-time beside it -- because a validation whose reference
is the thing being validated is no validation.

What it earned: the first version of `lattice.stream` applied only the first
out-of-range axis's rule to a diagonal link and then read out of bounds, and this case
came out **2.4% off and would not converge**. Nothing else in the suite would have
noticed. 2.4% on a drag coefficient reads as a resolution problem and gets tuned away.

**`cavity` is graded against Ghia at both Reynolds numbers, on his interior points.** His
tables start and end on the boundaries -- the first row for `u` is `y = 1.0, u = 1.0` --
and those are the boundary condition, not a solution. A lattice has no node on the wall
to sample them at: the nodes sit half a spacing inside, so interpolating to `y = 1.0`
clamps to the last node and that one point contributes 0.039 to the RMS, twice the whole
tolerance. The two boundary rows come out, the count that went in is printed, and the RMS
over all seventeen is reported beside it so the choice is visible rather than hidden.
`../felix/grade.py` keeps all seventeen and is right to: Felix is a mesh solver with
nodes **on** the wall.

**`cylinder` runs at three resolutions, because one that agrees is not evidence.** The
observed order of convergence on the drag error,
`p = log(e_coarse/e_fine) / log(h_coarse/h_fine)`, is the only number in this directory
that distinguishes "the scheme is second-order and this grid is fine enough" from "this
grid happened to agree". A `p` near zero means the error is not grid error and something
else -- the domain, the inflow, the sampling window -- is dominating.

**`crosscheck` is the most useful single number here.** The same cylinder at Re 100 on
two completely independent discretisations: a finite-volume pressure-correction solver on
a body-fitted morphing mesh, and a lattice Boltzmann scheme on a uniform grid with an
interpolated boundary. They share no code, no mesh and no boundary treatment. The
OpenFOAM side is `tests/data/moving_mesh/viv_fixed/`, a finished `pimpleFoam` run
committed in September, and it is read through `../moving_mesh/grade.py` rather than
copied into a constant -- so if that run is ever re-trimmed the comparison moves with it,
and a drift between the grader's answer and the documented one is itself a graded check.

Agreement between two such solvers is evidence of a kind that agreement with a published
table is not: there is no shared way for both to be wrong.

**`momentum` is the one that catches what a drag coefficient cannot.** With every face
periodic and no body force, `d/dt(fluid momentum) = -(force on the body)` exactly, so the
field integral the solver holds must equal the time integral of the force it reported. A
bounce-back that reflects the wrong population, a Bouzidi `q` with the wrong sign, a
refill that invents mass -- all of them show up here and none of them shows up in a drag
within 5% of a published fit. It is measured twice, and the split is the point: a still
body in a decaying flow refills nothing and is graded at 1%; a translating body is graded
at 10%, because equilibrium refilling injects momentum that no force on the body accounts
for, and measuring what that costs is what `DECISIONS_LIVE.md` D5 said this case was for.

**`ground_effect` is the only graded case with the subgrid model on, and it is one disk
and not four.** A 120 mm rotor pushing 5 m/s of air is Re ~ 80,000 on the diameter, and
`units.STIFF_TAU`'s arithmetic shows there is no resolution at which a millisecond
timestep resolves air's molecular viscosity -- the eddy viscosity carries the run, and the
case says so in its own report every time. It is legitimate for this quantity, because
the ground effect is an inviscid image-source argument and the model it is compared
against is itself inviscid. One disk, because Sanchez-Cuevas, Heredia and Ollero (2017)
measured that a complete multirotor does **not** follow Cheeseman and Bennett: the effect
is larger and reaches about five radii rather than two, because of a fountain under the
airframe. A quadrotor graded against that model would be graded against a model the later
literature shows does not apply to it.

**Everything laminar runs with the subgrid model off, and `cases._official` refuses to
grade it otherwise.** A Smagorinsky term adds a positive eddy viscosity wherever the
strain rate is non-zero, which in a laminar flow is everywhere; the effective Reynolds
number is then below the one the case claims, and the drag comes out closer to the
published value for a reason that has nothing to do with the solver being right. A
flattering number that nothing refuses is the worst failure mode a suite like this has.

## The references, and which of them were read

| reference | read | the caveat that changes what the check is worth |
|---|---|---|
| Ghia, Ghia & Shin (1982), *J. Comput. Phys.* **48**, 387-411, Tables I and II | the scanned paper; Table I also against an independent transcription | Ghia solved to second order on 129x129, and his Re 1000 centreline extrema sit **1.46% to 2.20%** from Botella & Peyret's (1998) spectral benchmark. An agreement much tighter than that is agreement with his error. The Re 100 **v** column rests on a single OCR of the scan, because no second transcription of Table II could be found; it is in `openreynolds/toolbox/references/ghia_re100.json` with that stated in the file |
| Schafer & Turek (1996), *Notes on Numerical Fluid Mechanics* **52**, 547-566, case 2D-2, Table 4 | the full paper text | **there is no single reference value**: the paper gives intervals, which it calls "estimated intervals for the exact results as suggested by the authors" -- Cd_max 3.2200-3.2400, Cl_max 0.9900-1.0100, St 0.2950-0.3050. The Strouhal number is `f D / U` with `U` the **mean** inflow of 1.0 m/s, not the 1.5 m/s peak, and reading it as the peak is a 50% error. The paper's experimental St = 0.287 +/- 0.003 is an extended-inlet variant and is **not** this case |
| Schiller & Naumann (1933), through `openreynolds/toolbox/references/sphere_drag.py` | this repository's own reference module | a fit, not a derivation, so it carries the scatter of the data behind it; the module is explicit that its accuracy should be quoted from a source and not from the formula. Holds below Re ~ 1000. Stokes is the exact check and only in its own limit: at Re 1 the inertial correction is already ~15% |
| Cheeseman & Bennett (1955), ARC R&M 3021, equation (2) | the original report | three things the usual quotation drops: the constant inside the bracket is **4**, the ratio is derived **at constant power**, and `Z` is the rotor's height above the ground. The report states **no** numeric lower bound on `Z/R` -- only that small `Z/R` gives "large errors" and is "of no practical interest" -- and its one number is in the conclusions, that flight results agree for `Z/R > 0.6`. The `Z/R >= 0.25` figure that circulates could not be traced to it |
| Williamson (1996), *Annu. Rev. Fluid Mech.* **28**, 477-539 | a full text of the review | St = 0.164 at Re 100 is **not tabulated** there: page 494, figure 7b prints the fit `S0 = -3.3265/Re + 0.1816 + 1.6e-4 Re`, which at Re 100 evaluates to 0.16433. The value and the attribution are both right, with that provenance. **The review gives no drag coefficient at Re 100 and none is attributed to it** |
| this repository's OpenFOAM cylinder, `tests/data/moving_mesh/viv_fixed/` | read live by `../moving_mesh/grade.py` on every cross-check | a cylinder at Re 100 in water with 2.5% blockage, `pimpleFoam` on a morphing mesh. St 0.166141 and mean Cd 1.355514, asserted to five digits by `tests/test_benchmark_moving_mesh.py` so they cannot drift without a test failing. **There is no canonical published Cd at Re 100** -- the literature clusters 1.32 to 1.37 -- which is why drag is graded against this run and not against a published number |

## The force noise on a floating hull: one settle, every candidate

`free_settle.py` settles a tank to its own velocity bound and refuses to report a draft
until it is met. `force_noise.py` is that settle **once**, with every candidate remedy
released from the identical settled state:

```
MODAL_PROFILE=inviscid python -m modal run --detach benchmarks/live/force_noise.py --radius 0.002
MODAL_PROFILE=inviscid python -m modal run benchmarks/live/force_noise.py --fetch
python benchmarks/live/force_noise.py --demo        # the EMA arithmetic, no GPU
```

The settle is 19 of `free_settle.py`'s 22 minutes, so twelve variants its way is four
hours of H100 for forty minutes of release window. Settling once and restoring the fluid
positions, the velocities and the hull's pose between windows is 4.7x cheaper **and a
better experiment**: every variant starts from literally the same water, which separate
runs cannot promise. `base-again` is the last variant and is the control on exactly that
-- `base` run twice, first and last, which bounds what the restore leaks.

What it found, with the hull's weight as the unit and two independent settles of every
configuration (H100, 64 x 64 x 32 mm hull at 0.5 of water's density, 60 mm of water):

| per-step RMS of `Fz`, free hull | 2.5 mm | 2.0 mm | 1.5 mm |
|---|---|---|---|
| the shipped coupling step (`CFL` 0.4) | 92.9 / 94.0% | 104.6 / 99.2% | 114.9 / 115.0% |
| `CFL/2` | 92.1 / 89.8% | 97.3 / 97.2% | 90.8 / 89.4% |
| **`CFL/4`** | **5.1 / 5.4%** | **1.8 / 2.6%** | **1.3 / 1.3%** |
| `CFL/8` | 5.8 / 5.4% | -- | 1.4 / 1.6% |
| DFSPH tolerance 1e-5 instead of 1e-3 | 88.7 / 91.2% | 89.2 / 89.6% | -- |
| delta-SPH at 88% of its stability ceiling | 100.6 / 102.7% | 102.3 / 105.8% | -- |

A hull **held** at the Archimedes draft, where Newton is never stepped and there is no
coupling lag to shorten, goes **22.0% to 1.3%** under the same change -- so the cause is
the pressure field converging in time and not the explicit coupling. The coarse row has
the honest caveat: a third settle at 2.5 mm read **8.3%** against those 5.1% and 5.4%, so
the spread at 3.2 spacings through the draft is 3.2 points and the margin to a 10% target
is thin there. It widens with refinement.

`openreynolds/live/sph.py`'s `CFL` docstring is the write-up: it is a cliff between
`CFL/2` and `CFL/3` rather than a slope, it is a *ratio* and not an absolute step (476 us
is quiet at 2.5 mm and 428 us is noisy at 1.5 mm), it costs 1.9x to 2.5x the wall clock
rather than the 3x to 4x the step count suggests, and the two things everyone tries first
-- a tighter pressure tolerance and density diffusion -- buy 13% and nothing
respectively. `SphWorld.wrench` carries the optional EMA for a caller who cannot afford
the shorter step, with its lag measured.

## What the recorded runs measured

See `../README.md` for the index rows and `../../PROGRESS_LIVE.md` for the full table with
devices, step counts and commit hashes. `grade.py --json` reproduces any of them.

## Running them

The small ones need nothing:

```
python benchmarks/live/grade.py smoke --device cpu
python benchmarks/live/grade.py poiseuille --resolution 32 --device cpu
```

The rest want a GPU. `modal_run.py` is how they are actually run here:

```
MODAL_PROFILE=inviscid python -m modal run benchmarks/live/modal_run.py::m1
MODAL_PROFILE=inviscid python -m modal run benchmarks/live/modal_run.py::layouts
```

which starts one-shot containers with explicit timeouts, no warm pool and no volume, and
writes each result and its figures into `benchmarks/live/results/<case>/`. That directory
is not committed (`.gitignore`): the numbers that matter are in `../../PROGRESS_LIVE.md`
and in the index rows, and the figures are reproducible from the grader.
