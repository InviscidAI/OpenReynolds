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
has been verified from the paper itself, so the grader prints A/D without a verdict.
Lock-in is graded (the response frequency lies nearer the spring's natural frequency
than the fixed cylinder's shedding frequency), as are the Strouhal gate and the energy
audit.

## The known result, 2026-09-12

The grader reproduces these from the raw output of those runs.

**Couette, one 50 x 240 mesh.** Taylor number 64, well below the 1,708 onset. Inner-wall
torque 1.21154e-3 N m/m against an exact 1.20637e-3, +0.43% (pass). Inner/outer
imbalance -0.911% of the inner torque (fail), against +0.011% on the conformal control,
85 times smaller. AMI weights between 1.0 and 1.0000428 (pass). No grid convergence: the
refinement runs that were launched diverged, so +0.43% is a single-mesh measurement.

**VIV.** Fixed cylinder St = 0.166141 over 7 cycles, +1.31% on 0.164 (pass), mean Cd
1.3555. Released with loose coupling: A/D 0.640, f/f_n 0.976, locked in, and net fluid
work between -10.1% and -10.4% of the body's energy per cycle (fail). Continued with the
body inside the PIMPLE loop and no acceleration relaxation: A/D 0.567, f = 0.19394 Hz,
f/f_n 0.970, locked in, work +0.04% to +0.18% per cycle (pass). The 13% drop in
amplitude is the coupling error the energy audit caught. The tight result is a
continuation from the loose one, not an independent start from rest, and there was
one mesh with no grid study.
