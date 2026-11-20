# Moving-mesh benchmarks

Two prompts and a grader for the cases every moving-mesh claim rests on: a sliding
interface, and a body that moves because the flow moves it. Both were run end to end
through the agent on 2026-09-12, and both taught something a residual plot would not
have shown, so they are kept as cases anyone can rerun and grade the same way.

| prompt | what it tests | the reference |
|---|---|---|
| `couette_ami.txt` | a `cyclicAMI` sliding interface cut through a 2D circular-Couette annulus, inner wall turning | exact: the Couette profile and the torque on the inner cylinder are closed form |
| `viv.txt` | a cylinder on a spring, free to move across the stream, on a morphing mesh at Re = 100, m\* = 10, zero damping, U\* = 5 | Williamson (1996) St = 0.164 for the fixed cylinder; no amplitude is given, on purpose |

## Running one

```sh
openreynolds -p "$(cat benchmarks/moving_mesh/couette_ami.txt)"
openreynolds pull --study <study-id>          # the case files, home

python benchmarks/moving_mesh/grade.py couette <case> [--control <case without the interface>]
python benchmarks/moving_mesh/grade.py viv <fixed case> <released case> [<continuation> ...]
```

`openreynolds/toolbox/viv.py` is the same measurement as an instrument rather than a
grader: it reads one finished case and prints A/D, the frequency against both the
structure's own and the fixed shedding, the energy balance per cycle and a figure, and
`openreynolds/toolbox/templates/viv_cylinder.py` writes the two cases to begin with.
`tests/test_toolbox_viv.py` holds it to this grader's numbers on the same committed runs,
digit for digit, which is what makes the second implementation safe to keep.

Add `--json` for a machine-readable result. The exit status is 0 when every graded check
passes, 1 when one fails, and 2 when the case could not be read.

## What the grader reads, and why only that

Only what the solver wrote: `postProcessing/*/*.dat`, `log.*`, `constant/` and
`system/`. Never the run's own analysis. The first time these two cases were graded by
hand, four of the runs' own summary numbers disagreed with their raw files, and one
analysis script had read a torque sample from t = 13 and labelled it t = 20, because
the solver had restarted and the script never opened the restart's directory.

So the grader stitches every restart directory, and cuts each one at the start time of
the next, since a run that died past its last write left rows for a solution nobody
kept. Columns are found by the names in the `# Time` header, never by position. Every
physical constant comes from a case file (viscosity from `transportProperties`, density
from the function object, rotation rate and spring from `dynamicMeshDict`, radii off the
mesh points, diameter and free stream from the `forceCoeffs` header). If one is missing,
the grader stops and names the flag that supplies it rather than assuming a value.

## Graded in CI

The `benchmarks` job of `.github/workflows/tests.yml` runs both graders on every pull
request, against the 2026-09-12 runs themselves: `tests/data/moving_mesh/` is those five
cases cut down to the files the grader opens, and they reproduce every number below
digit for digit. Both benchmarks exit 1 there, because both of these runs failed a check
and the failure is the finding; the job holds each to its recorded verdict, so a grader
that stopped reading real output the way it does now is a red tick on the pull request
rather than a surprise months later.

## What passing means

The thresholds are in `THRESHOLDS` at the top of `grade.py`, each with where it came
from. Two of them are worth understanding before reading a verdict.

**Couette: the torque imbalance.** In steady circular Couette flow the torques on the two
walls are equal and opposite exactly, because angular momentum is conserved. That makes
the imbalance a free test of the interface that needs no reference at all. The check
the prompt asks for, the solver's `sum(weights)`, measures whether the interface kept
its area. It does not measure whether it kept the torque, and on the 2026-09-12 run the
two disagreed: the weights stayed within 43 ppm of 1 while the walls disagreed by 0.91%,
against 0.011% on the same annulus meshed without an interface. The threshold is 0.1%,
and that run fails it. The failure is the finding.

**VIV: the energy audit.** With zero structural damping, nothing can dissipate energy,
so on a real limit cycle the fluid does no net work on the body per cycle. The grader
integrates lift times velocity over each whole cycle. A loosely coupled run can settle
into an amplitude that is stationary, sinusoidal and spectrally clean, and is still held
there by the coupling scheme rather than the physics. The energy audit is the only
check here that catches that.

**The amplitude is reported, not graded.** No published amplitude for exactly this case
has been found, so the grader prints A/D without a verdict. The two nearest:

*Menon & Mittal*, arXiv:2006.11649, published as *J. Fluid Mech.* **907**, A37 (2021),
doi 10.1017/jfm.2020.854. Re-read at source 2026-11-21, and two things changed. First,
**the paper states no amplitude number at all**: it reports "the maximum stationary state
oscillation amplitude" as a curve, and the value at U\* = 5 is readable only off its
Figure 5(a), so the "about 0.56" here is a figure read and is written as one from now on.
Second, its mass-ratio definition is in the text verbatim as `m* = 2m/(rho D^2)`, so its
m\* = 10 is **6.37** on displaced mass against this case's 10 -- a 57% difference in body
mass between two papers both saying "m\* = 10". Confirmed verbatim: Re = 100, "two-
dimensional flow simulations", "there is zero structural damping in the model used here",
domain 25D x 20D. The body is a superellipse of aspect ratio 1.0 to 1.25, of which 1.0 is
the circle.

*Placzek, Sigrist and Hamdouni* (Computers & Fluids 38, 2009) peak at 0.58 at an
effective stiffness k\*eff = 2.32, where this case sits at about 1.48, mostly at a
lighter mass ratio of 3.3. **This one could not be re-read**: the HAL record
(hal-00534010) and the publisher both refuse an automated request, so those figures are
carried over from whoever first read them here and are not verified at source. They are
not what anything is graded against.
Lock-in is graded (the response frequency lies nearer the spring's natural frequency
than the fixed cylinder's shedding frequency), as are the Strouhal gate and the energy
audit.

## The known result, 2026-09-12

The grader reproduces these from the raw output of those runs.

**Couette, one 50 x 240 mesh.** Taylor number 64, well below the 1,708 onset. Inner-wall
torque 1.21154e-3 N m/m against an exact 1.20637e-3, +0.43% (pass). Inner/outer
imbalance -0.911% of the inner torque (fail), against +0.011% on the conformal control,
85 times smaller. AMI weights between 1.0 and 1.0000428 (pass).

**Grid convergence, added 2026-09-15.** Three meshes per case at 2,880, 11,520 and 46,080
cells with the time step scaled with the cell size (Courant 0.255 throughout, and four
pressure correctors per step: at two, halving the cells doubles the cell diffusion number
`nu*dt/dr^2` and the run diverges, which is what killed the first attempt). The control
converges at observed order 1.93 and extrapolates to within 0.0004% of the exact torque;
the sliding interface converges at 0.79. Holding the time step and multiplying the cells by
four moves the interface's torque error from 0.2174% to 0.2205%, so what it is first order
in is the angular slide per step `omega*dt`, not the cell size. Grade any rerun of this case
at a stated `omega*dt`, not at a stated cell count.

**VIV.** Fixed cylinder St = 0.166141 over 7 cycles, +1.31% on 0.164 (pass), mean Cd
1.3555. Released with loose coupling: A/D 0.640, f/f_n 0.976, locked in, and net fluid
work between -10.1% and -10.4% of the body's energy per cycle (fail). Continued with the
body inside the PIMPLE loop and no acceleration relaxation: A/D 0.567, f = 0.19394 Hz,
f/f_n 0.970, locked in, work +0.04% to +0.18% per cycle (pass). The 13% drop in
amplitude is the coupling error the energy audit caught. The tight result is a
continuation from the loose one, not an independent start from rest, and there was
one mesh with no grid study.
