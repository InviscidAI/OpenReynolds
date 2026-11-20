# Tutorial cases

Three cases cheap enough to run as a first look at the solver, each with an answer that
comes from somewhere other than a solver, and each with a result below from a run that
really happened. They are three different kinds of starting point: an exact analytic
answer, a published numerical benchmark, and a published benchmark whose units are a trap
that this repository has already fallen into.

| prompt | what it tests | the reference |
|---|---|---|
| `pipe_nu.txt` | an axisymmetric wedge, a scalar transported on a converged flow, and a Nusselt number taken against a mixed-mean reference temperature | exact: the Graetz problem gives Nu = 3.6568 for a circular pipe at constant wall temperature |
| `cavity_ra1e5.txt` | buoyancy-driven flow, a graded mesh, and an energy balance across two walls | de Vahl Davis (1983), Table V, Ra 1e5 |
| `bfs_re800.txt` | a separating and reattaching flow, a reattachment length off the wall shear, and which length a benchmark is normalised on | Gartling (1990), the 2D backward-facing step test problem |

Each is one prompt in a user's own words, the same form as the cases in `felix/` and
`moving_mesh/`. The reference is named inside the prompt, so the comparison is part of the
answer rather than something done to it afterwards.

## Running one

```sh
openreynolds -p "$(cat benchmarks/tutorials/pipe_nu.txt)"
openreynolds pull --study <study-id>          # the case directory, home
```

## What measures the answer

All three turn on a quantity the toolbox already measures, which is why there is no grader
in this directory:

| the quantity | what reads it | the part that is easy to get wrong |
|---|---|---|
| wall Nusselt number | `heat.py --reynolds --prandtl` | which reference temperature. The same wall gives three different Nusselt numbers against a free-stream, an inlet and a mixed-mean reference, and only the mixed mean is what an internal flow's Nu is defined on. |
| whether the heat budget closes | `heat.py` per patch | a hot wall and a cold wall that disagree have not converged, whatever the residuals say. It costs one extra integral. |
| reattachment length | `reattach.py` | which sign change, and in what unit. The first is the corner eddy; a later one may be a secondary bubble. |
| whether a transient has settled | `spectra.py` | a record that is stationary and spectrally clean can still be held there by the numerics rather than by the physics. |

## The known result

The numbers below were recomputed for this file from each run's own final field and mesh,
not copied from the run's summary, and the reference values were re-checked against the
sources rather than against the repository. That exercise changed three of them, so the
recomputation is what is quoted and the disagreement is noted where it exists.

### The heated pipe, against an exact answer

A 150 x 40 axisymmetric wedge, half-angle 2.5 degrees, radius 0.01 m and length 0.3 m,
graded 0.4 toward the wall, the collapsed axis edge typed `empty` and the two wedge faces
`wedge`. `simpleFoam` to converge the flow, then a scalar transport for the energy
equation, which is legitimate here because the properties are constant and the temperature
does not feed back into the momentum. The case ran with nu = 1.5e-5 and DT = 2.152e-5, so
Pr = 0.697, with the wall at 350 K against a 300 K inlet.

**The run reported Nu = 3.677 over x/D > 10, which is +0.55% on the exact 3.6568.**

The flow is genuinely developed where that average was taken. The hydrodynamic entrance
length is about 0.05 Re D = 0.1 m and the thermal entrance length about
0.05 Re Pr D = 0.07 m against a 0.3 m pipe, and the centreline velocity in the final field
is 1.991 times the mean against a parabolic 2.

**What the case actually teaches is that the third significant figure is not established.**
Recomputing the same quantity from the run's own final `T` and `U`:

| wall gradient | radial cell position taken as | Nu over x/D > 10 |
|---|---|---|
| secant to the wall | annular centroid | 3.654 |
| secant to the wall | midpoint of the radial levels | 3.650 |
| quadratic through the wall and two cells | annular centroid | 3.635 |
| quadratic through the wall and two cells | midpoint of the radial levels | 3.628 |

Every one of those is a defensible way to take the number, they span 0.7%, and the
discrepancy being reported is 0.55%. **The post-processing convention moves the answer by
more than the error does.** So the honest claim is sub-percent agreement with an exact
result, which is a real and useful statement, and not a figure in the third place. There
is a second trap in the reference itself: the textbook rounding 3.66 is 0.09% above
3.6568, so comparing against 3.66 manufactures a bias of its own at this accuracy, and
4.364 (= 48/11) is the constant-heat-flux answer rather than this one.

Reference: Nu = lambda_1^2 / 2 with the first Graetz eigenvalue lambda_1^2 = 7.3136, so
Nu = 3.6568; solving that eigenvalue problem directly gives 3.65679. The problem is
Graetz (1883, 1885) and Nusselt (1910), properly the Graetz-Nusselt problem; the accurate
eigenvalues are Brown, *AIChE J.* **6**, 179-183 (1960). Incropera & DeWitt Table 8.4
carries it as Nu = 3.66 (Eq. 8.55) against 4.36 for constant flux (Eq. 8.53).

### Natural convection at Ra 1e5, and a reference that is widely misquoted

A 40 x 40 mesh symmetrically graded toward all four walls, differentially heated vertical
walls, adiabatic top and bottom, solved steady with a Boussinesq buoyant solver.
`checkMesh` clean: maximum non-orthogonality 0, skewness about 7e-14.

The setup is worth copying because the Rayleigh number was reached by choosing the one
free quantity rather than by bending a property. With g = 9.81, beta = 3.3333e-3,
nu = 1.5e-5, Pr = 0.71 and L = 0.1 m, the walls were set to 300.485 K and 299.515 K, a
difference of 0.970 K, giving Ra = g beta dT L^3 / (nu alpha) = 9.999e4 and Pr = 0.710.

**Mean Nu on the hot wall 4.5335, against de Vahl Davis's hot-wall 4.509: +0.54%.**

The reference needed correcting and this is the part worth reading. de Vahl Davis reports
*several* Nusselt numbers at Ra 1e5 in his Table V, and they are not interchangeable:

| his symbol | what it is | Ra 1e5 |
|---|---|---|
| `Nu_0` | the average on the vertical boundary at x = 0, the **hot wall** | **4.509** |
| `Nu` | the average throughout the cavity | 4.519 |
| `Nu_1/2` | the average on the vertical mid-plane | 4.519 |

`Nu` and `Nu_1/2` coincide to four figures here, and secondary sources routinely cite
**"de Vahl Davis hot-wall Nu = 4.519"**, which is his cavity-average and not his wall
value. The two are 0.22% apart, and his own estimated accuracy at Ra 1e5 is 0.3%, so the
misattribution is the same size as the reference's uncertainty and larger than it is
comfortable with. A wall-integrated Nusselt number, which is what this run computes and
what `heat.py` computes, belongs against **4.509**. Measured against the commonly quoted
4.519 the same result reads +0.32%, which is how a validation can be half a percent
wrong in the flattering direction without anyone touching the solver. He also recommends
`Nu_1/2` as the number to describe heat flow across a cavity with adiabatic ends, on the
grounds that it contains both wall values.

The check that makes the result believable is not the agreement with the benchmark, it is
the agreement with itself: the mean Nusselt number on the **cold** wall recomputes to
4.5335 as well, matching the hot wall to seven significant figures. With adiabatic top and
bottom in a steady state every watt entering one vertical wall leaves through the other
exactly, so the two walls agreeing is a closed energy balance, and it is free. The local
Nusselt number along the hot wall runs from 0.720 at the top to 7.791 at the bottom, which
is the textbook shape, and de Vahl Davis's own extremes are Nu_max 7.717 at height 0.081
and Nu_min 0.729 at the top.

Convergence as a measurement rather than a tolerance: at the final iteration the initial
residuals were 5.2e-8 on Ux, 3.9e-8 on Uy, 3.3e-8 on T and 6.7e-8 on p_rgh, with the local
continuity error at 8.3e-11. Ra 1e5 is far below the transition where this geometry needs
transient treatment.

Where the remaining 0.54% comes from: a 40 x 40 mesh, a one-sided wall gradient at the
first cell centre, which is the same stencil question the pipe case is dominated by, and
0.3% of reference uncertainty. Two more free checks are in the prompt: his mid-plane
velocity maxima are 34.73 and 68.59, non-dimensionalised on **alpha/L** and not nu/L, and
mixing those two scales up is a factor of Pr and the usual reason a correct solve appears
to fail this benchmark.

Reference: de Vahl Davis, *Int. J. Numer. Methods Fluids* **3**, 249-264 (1983), Table V
on p. 256, with the definitions on p. 252 and the section "Which Nusselt number?" on
pp. 262-263. The error estimates are his Table VII, p. 262.

### The backward-facing step, where the units are the whole case

This one is not shipped as a passing validation. It is shipped because working out what
it should be compared against found an error that this repository asserts in three places,
and the error is a single mistake that produces two factors of two.

The run built, in its own `blockMeshDict` comment, "step height h=1, expansion ratio 2":
an inlet duct over y in [1, 2], a step face at x = 0 over y in [0, 1], a downstream channel
over y in [0, 2] and 30 length units long, with an inlet profile u = 6 y'(1 - y') whose
mean over the inlet is exactly 1, and nu = 0.00125. Its stated Reynolds number is
`Re = Umean*h/nu = 1*1/nu = 800` on the **inlet** channel height.

Gartling's case is the same shape with the unit length chosen differently: his downstream
channel is height 1 (y from -0.5 to 0.5), his inlet is the upper half, his step is
therefore 0.5, his inlet mean velocity is 1 and his viscosity is 0.00125, so his
Re = u_mean H / nu = 800 is on the **downstream channel height**.

Set those side by side and the run's case is Gartling's geometry scaled up by a factor of
two with the same viscosity and the same velocity. That is one mistake, taking Gartling's
unit length to be the step height when it is the channel height, and it has two
consequences:

**The Reynolds number is twice the benchmark's.** Same nu, same u_mean, every length
doubled, so Re is doubled on whichever length you measure it: 1600 on the downstream
channel height where Gartling's is 800. This part needs no reading of Gartling's text at
all, only his geometry, and it is a geometric fact rather than a naming dispute.

**The reattachment length is in a different unit.** Recomputed from the final
`wallShearStress` field and the mesh, the streamwise wall shear changes sign at 0.21,
6.42, 7.57, 9.70 and further downstream; 0.21 is the counter-rotating corner eddy and
6.42 closes the primary bubble. Those are in the run's length unit, which is the step
height, so 6.42 step heights, or 3.21 channel heights. Gartling's 6.1 is in his unit
length, the channel height, which is 12.2 step heights.

So the comparison the run made, and that `reattach.py`'s docstring, `claims.py` and the
internal record all repeat, is **6.42 step heights against 6.1 channel heights**. Taken
in one unit it is not "+5%, good agreement": it is 3.21 against 6.10, about 47% short.
And because the run is also at twice the Reynolds number, where the bubble is longer
rather than shorter in this range, a correct solve of the run's own case should give more
than 12.2 step heights, not less. The two factors of two do not cancel. The unit error
manufactured an agreement and the Reynolds error made the real shortfall worse.

**What is confirmed and what is not.** Gartling (1990), *Int. J. Numer. Methods Fluids*
**11**, 953-967, is behind a paywall with no open copy, so his own table was not read.
The geometry, the inlet profile, the Re definition and the value 6.10 are each stated
identically by three independent benchmark suites that reproduce the case (MFIX-Exa, SU2,
and the GALES benchmark set) and by the Abaqus verification manual, which also carries his
upper-wall bubble from 4.85 to 10.48 in channel heights. The factor of two is closed
independently by a published table that lists Gartling as x_r/h = 12.20 with h the step
height, which is exactly 2 x 6.10, alongside 11.26 = 2 x 5.63 for the upper bubble.
Armaly's experiment supports the same reading: at the same flow rate it measures about 14
step heights, and a 2D steady computation is expected to underpredict a flow that is
three-dimensional above Re about 400, which puts 12.2 in the right place and 6.1 nowhere
near it. **Until somebody reads the paper, the normalisation rests on that corroboration
and is flagged here rather than asserted.**

The other two things the run did are worth keeping regardless of the units. It reported
that the pressure residual plateaued around 0.25 to 0.29 out to 15,000 iterations,
oscillating rather than decaying, and that at this Reynolds number the 2D laminar step is
at or past the threshold where a steady solution stops existing, so the "steady, laminar"
premise was itself questionable. Recomputing the reattachment at each of its writes gives
5.92, 6.18, 6.32 and 6.42 at t = 2500, 3000, 14500 and 15000: still climbing when it
stopped, so the number was never converged in any unit.

And the figure it delivered is annotated 7.5667 while its text said 6.4. The 7.5667 is not
a second estimate that disagreed; it is the **third** sign change in the same list at the
same time, so the script that drew the picture took `crossings[2]` where the answer took
`crossings[1]`. `reattach.py` exists because of that: one implementation, which says which
crossing it chose and why, and which carries its own number into the figure it draws.

## What these deliberately leave out

**No grader, and no fixtures.** All three graded quantities are recoverable from solver
output alone, and were recovered that way for this file: the pipe's Nusselt number from
`T` and `U` on the wedge, the cavity's from `T` and the mesh grading, the step's from the
`wallShearStress` field's `boundaryField`. What stops them being graded here the way
`felix/` and `moving_mesh/` are graded is that none of that output is in this repository,
because `studies/` is not committed. Trimmed copies in `tests/data/`, on the pattern
`tests/data/moving_mesh/PROVENANCE.md` sets out, are what would close the gap; until they
exist these numbers are quoted rather than re-derivable here.

**No grid study on any of the three.** Each is one mesh, so none has a claim about how much
of its error is discretisation, and on two of them the post-processing stencil is already
shown to matter as much as the mesh might.

**No turbulent case.** All three are laminar or buoyancy-driven at a Rayleigh number where
steady laminar is correct. Nothing here exercises a turbulence model.

**The step case is not a validation.** It is a worked example of a units failure with an
unresolved reference, and it is labelled that way above rather than counted as a pass.
