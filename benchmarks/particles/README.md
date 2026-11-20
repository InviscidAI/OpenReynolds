# Particle tracking: two exact closed forms, and one published threshold

The three cases `openreynolds/toolbox/tracks.py` is run on.

```
python benchmarks/particles/grade.py drag
python benchmarks/particles/grade.py tracer
python benchmarks/particles/grade.py capture
```

Like `../lbm/grade.py` and `../sph/grade.py`, this grader **solves** rather than reading
a finished run: the integrator is numpy in this repository and the whole file finishes in
about ninety seconds, so committing its output would replace a check that re-derives the
number with a number nobody can re-derive.

| case | measures | the answer, and where it comes from | the recorded run |
|---|---|---|---|
| `drag` | the velocity of a particle released into a uniform stream | `v(t) = U (1 - exp(-t/tau_p))`, **exact** for linear drag | max error **1.9e-12** of U over five relaxation times |
| " | the terminal velocity of a 20 um quartz grain in air | `w_t = tau_p g (1 - rho_f/rho_p)`, **exact**; 0.0320799 m/s, Re_p 0.043 | **2.2e-14** as a fraction |
| " | the same for a 200 um grain, where Stokes drag is wrong | the root of `w = tau_p g' / phi(Re_p(w))`, solved independently of the integrator | SN terminal **0.4619** of the Stokes value at Re_p 42.8; matches the implicit root to **5.1e-15** |
| `tracer` | a massless tracer's orbit in solid-body rotation | it closes **exactly** after one period | **5.1e-8** of the radius at 200 steps per revolution, observed order **4.00** |
| `capture` | the critical Stokes number for inertial capture on a circular cylinder in inviscid flow | **1/8** in the `tau_p U_inf / R` convention -- see below | **0.1262909, +1.03%**, converging to 1/8 from above |

## The published threshold, and the factor of two in front of it

A particle approaching the forward stagnation point of a bluff body either reaches the
surface or does not, and which one it does depends on a single number. Below a critical
Stokes number nothing hits the body at all however long you wait; above it, particles on
the axis arrive in finite time.

The source read for the value is **Robinson, Warren, Turner & Sear, *Critical inertia for
particle capture is determined by surface geometry at forward stagnation point*,
arXiv:2310.03474v2 (25 Feb 2025), Section III.3**, which states "So here the critical
Stokes number St_c = 1/4 for a cylinder", and attributes the result to Taylor and to
Langmuir and Blodgett. Its Stokes number is `St = m U / (l_y xi)`, with `l_y` the
transverse semi-axis of the body (the **radius**, for a circle) and `U` the local flow
speed approaching the stagnation point, which for inviscid flow past a circle is
`2 U_inf`. The same paragraph notes of Langmuir and Blodgett that they "actually quote
St_c = 1/8" "because of their alternative choice to set U = U_inf".

So **1/4 and 1/8 are the same result in two conventions a factor of two apart**, and in
the convention used everywhere in this directory -- `St = tau_p U_inf / R` -- the value
is **1/8**. `../README.md` is a long argument about exactly that class of mistake: of its
eleven reference values, re-checking against the sources changed two, one of them
Gartling's reattachment length read as step heights where the paper states channel
heights, a factor of two that reached a customer-facing page. Both errors flattered the
result, which is why neither was noticed. The convention is therefore written into the
threshold's key in `grade.py`, into the printed line, and into the test that guards it.

**What was not read.** Taylor's 1940 note (*Reports and Memoranda of the Aeronautical
Research Committee* 2024) and Langmuir & Blodgett's 1946 AAF technical report, the two
primary attributions, were not reached. The value graded here comes from a paper that
derives it and states both conventions, which is better than a secondary citation of a
bare number but is not the original, and this file does not pretend otherwise.

## And it is derivable, which is better than citable

Three lines, so a reader does not have to take it on trust either. On the upstream axis
of potential flow past a cylinder of radius `R`, `u = U_inf (1 - R^2/x^2)`, so with
`s = |x| - R` the distance to the wall,

```
u(s) = U_inf (1 - R^2/(R+s)^2) ~ (2 U_inf / R) s      as s -> 0
```

and the strain rate at the stagnation point is `A = 2 U_inf / R`. A particle with linear
drag on that axis obeys `tau_p s'' + s' + A s = 0`, whose roots are real when
`1 - 4 tau_p A >= 0` and complex otherwise. Real roots mean `s` decays to zero
monotonically and reaches it only as `t -> infinity`; complex roots mean `s` crosses zero
in finite time, which is the collision. So capture begins at `tau_p A = 1/4`, that is at

```
tau_p U_inf / R = 1/8
```

which is `St_c = 1/4` in the `l_y = R, U = 2 U_inf` convention above. The test
`test_the_cylinder_field_is_tangent_to_the_wall_and_uniform_far_away` checks the strain
rate `2U/R` out of the field itself rather than assuming it, because that one number is
all the threshold depends on.

## Why the measurement lands 1% high, and why it has to

**The threshold is an infinite-time statement and the measurement is not.** Below `1/8`
the particle approaches the wall for the rest of time, so a tracker with any finite
stopping rule has to decide when to stop waiting, and whatever rule it uses will call
some sub-critical particles captured. The bias is therefore **one-sided: the measured
threshold can only be above 1/8**, and a measurement below it would be a bug rather than
a better answer. The grader asserts that sign as a separate check.

`tracks.py` stops a particle when its approach speed toward the capture surface falls
below `--stop-speed` times the free stream. Tightening that rule by eight decades:

| stop speed / U | measured `tau_p U/R` | against 1/8 |
|---|---|---|
| 1e-6 | 0.1320892 | +5.671% |
| 1e-8 | 0.1289398 | +3.152% |
| 1e-10 | 0.1274994 | +2.000% |
| 1e-12 | 0.1267181 | +1.375% |
| 1e-14 | 0.1262909 | **+1.033%** |
| 1e-16 | 0.1262909 | +1.033% |

It converges monotonically toward 1/8 and then stops, at `+1.033%`, which is where double
precision runs out: the approach speed near the wall decays like `exp(-4 U t / R)`, and
below about 1e-15 of `U` the position increment per step is smaller than the ULP of the
radius, so the particle stops moving rather than arriving. 1e-16 measures the same
0.1262909 as 1e-14, which is the floor saying so.

**The remaining 1% is the stopping rule and not the integrator**, and that is a
measurement rather than an assertion. Time step, at `stop_speed` 1e-14:

| `dt` [R/U] | measured `tau_p U/R` | against 1/8 |
|---|---|---|
| 0.01 | 0.1262177 | +0.974% |
| 0.005 | 0.1262909 | +1.033% |
| 0.002 | 0.1263641 | +1.091% |

A factor of five in the time step moves the threshold by 1.5e-4 in `St`, **0.12%**,
against a 1.03% gap, and it moves it the wrong way: a finer step makes the measurement
very slightly worse, not better, which is what a discretisation error does not do. At
this stopping speed the stop is decided by floating-point granularity, and that granularity
is per step, so a smaller step reaches the floor a little sooner. The time step is two
orders below the gap either way and is not what the gap is. Release point, at `dt` 0.005
and `stop_speed` 1e-14: -3, -5, -10 and -20
radii upstream give 0.1262848, 0.1262909, 0.1262787 and 0.1262848, a spread of 1.2e-5 in
`St`, **0.01%**. That independence is the property the inviscid threshold is supposed to
have, since it depends only on the flow near the stagnation point, so measuring it is a
check on the tracker and not only on the number.

The grader also checks that the fate is **monotone** in the Stokes number across the
threshold -- 91 values from 0.118 to 0.136, no holes -- because a bisection on a fate
that is not monotone returns a number that is not a threshold.

## What this does not measure

**Not a parcel cloud.** This is post-processing over a solved Eulerian field, and the
particles do not change the flow. For a dilute cloud that is the same physics as one-way
coupled parcels; for a dense one it is not. The workspace image does carry Lagrangian
solvers -- `DPMFoam`, `DPMDyMFoam`, `MPPICFoam` and `MPPICDyMFoam` are in `$FOAM_APPBIN`
on the instance, measured in `studies/20260823-213712-babc/messages.jsonl` seq 14, and
`tutorials/lagrangian/` carries `kinematicParcelFoam`,
`icoUncoupledKinematicParcelDyMFoam`, `uncoupledKinematicParcelDyMFoam` and
`MPPICDyMFoam`. What could not be verified from outside the image is the exact spelling
of the uncoupled kinematic solver in v2512 or whether a `kinematicCloudProperties`
written against an older release still parses, and a case generator naming a binary
nobody has run is a generator that fails on the instance.

**No added mass, no Basset history, no lift, no collisions, no feedback on the fluid.**
For a heavy particle in a gas those are small. For a bubble or a neutrally buoyant
particle they are the physics, and `tracks.py` is the wrong instrument.

**No published collection-efficiency curve.** `tracks.py cylinder --stokes` prints the
single-cylinder collection efficiency, and it is **not graded**: the efficiency data for
a cylinder in a cross-flow lives in the aerosol literature behind publishers that refuse
an automated request, and quoting a curve from a secondary source is the error
`../README.md` argues hardest against. The efficiency is printed with the seed spacing
beside it so it is not read as tighter than it is.
