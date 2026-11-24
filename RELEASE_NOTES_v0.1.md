# openreynolds[live] v0.1

A GPU fluid that runs in lockstep with a robot simulation, so a policy can be trained
against water and air that push back.

This is an optional extra. Nothing in the core package changes, and the core imports
without it.

    pip install -e ".[live]"

There are **two solvers** in here and they are not at the same level of trust. That split
is the most important thing on this page.

---

## What is stable: the lattice backend, for air and submerged water

Use it for flow with no free surface in the picture: air around a drone, water around a
fully submerged hull, flow through a passage.

It is two-way coupled to Newton's MuJoCo solver with no host round trips in the step loop.
It runs **inside Isaac Sim**, in a live Kit process, on Kit's own Warp build rather than a
pinned one.

**What was measured, on real GPUs:**

| | |
| --- | --- |
| two-way coupling, 4 cases | **16 of 16 checks** |
| robot demos -- drone in ground effect, quadrotor, towed hull | **10 of 10 checks** |
| throughput | **1,478 MLUPS** on an H100, against a 1,000 target |
| laminar benchmarks | **5 of 7 cases pass without qualification** |
| inside Isaac Sim | **105 passed, 0 failed** on an L40S |

The two benchmark cases that do not fully pass are in *Known issues* below, with numbers.

**The one engineering result worth knowing**, because it fixed three failures at once: the
outlet boundary condition used to extrapolate, which with a velocity inlet meant nothing
held the domain's mass. An **empty** channel -- no obstacle, nothing but the boundary
conditions -- gained **28.1% of its mass** over 704,000 steps, losing 0.6% of its inflow
every step forever. An outlet that anchors the density holds it to **0.03%**. That one
change took the cylinder benchmark from +11.7% to **+2.2%**, and the cross-check against
this project's OpenFOAM run from +25.6% drag and +636% Strouhal to **-0.8% and -0.05%**.

---

## What is beta: the SPH backend, for cups, pouring and floating bodies

Use it for liquid with a surface: a cup of water on a robot arm, pouring, a hull floating
rather than submerged. This is the part the extension exists for, and it is the part that
is newer.

It is DFSPH with Akinci boundary particles, written from the papers. No third-party solver
code is in it, so nothing is owed in a licence file.

**What works, measured:**

| | |
| --- | --- |
| a tank's weight on its own floor | **0.9999x** the weight of the water in it |
| force on a submerged body | **1.0010x** rho V g |
| **per-step force noise on a floating body** | **1.3%** of its weight, from 12.9% |
| sloshing period | **5 of 5 checks** |
| dam break against Ritter | **4 of 5 checks** |
| a cup carried by a robot arm | runs **inside Isaac Sim**, with spill measured in grams |
| the angle a cup starts to pour at | **57.3 +/- 0.2 deg** against 57.4 predicted |

**The force noise is the headline.** A floating body's per-step force used to fluctuate by
**more than its own weight** -- useless for a controller. The cause turned out to be the
timestep, and it is a cliff rather than a slope: a quarter step takes it to **1.3%**, and
halving had been tried and done nothing. The backend now takes the shorter step
**automatically whenever a body is coupled**, and leaves fluid-only scenes alone, because
they have no force to be noisy. It costs about twice the wall clock, not four times,
because the solver needs fewer iterations per step.

**Why it is beta and not stable**, concretely:

* **Fluid-only runs are bit-for-bit repeatable. Runs with a floating body are not.** A
  read-write race in the viscous term is fixed, but float32 rounding in the force sum
  grows over a long settle, so **no floating-body number should be read from a single
  run.** Repeat it.
* The draft a floating hull settles at **does not pass its 2% check** -- see below.
* A body thinner than about four particle spacings now **warns at construction**, because
  the force on it is not reliable.

---

## Known issues, with numbers

**The floating draft fails its check, and two ways of measuring it disagree.** A hull
should float with 16.00 mm submerged. The case reports **14.51 mm**. A separate
implementation of the same mass-conserving calculation, run by hand on the same settled
state, gives **15.90 mm** -- which would pass. Two implementations of one quantity
differing by 1.4 mm means at least one is wrong and **we have not found out which.** The
original cause of a much larger error *was* found: the surface probe added a particle
radius to the topmost particle, which is right in bulk water and **0.9 mm short at a
settled surface**.

**Sphere drag at Reynolds 10 reads +3.0%** against Clift, Grace & Weber. It passes its
bound. Worth knowing that the five standard drag correlations for this case **disagree
with each other by 3.1%**, so which one you grade against decides the verdict -- and the
measurement sits above all five, so the bias is real even though its size depends on the
reference. The textbook itself could not be obtained; the formula came from two
independent transcriptions that both cite it.

**The 2D-1 lift reads about a third of the published range** (0.0035 against 0.0104-0.0110)
while its drag passes at +1.8%. Lift here is 0.01 against a drag of 5.6 -- a small
difference between two large numbers -- and this case's lift was already recorded as
reported-rather-than-graded for that reason.

**Sloshing on the lattice converges to +5.1%** of the linear-theory period, outside its 5%
bound. It is a scheme error, not a resolution one: a finite-amplitude correction was
suggested as the explanation and is **190 times too small and points the wrong way**.

**The lattice free surface is frozen by decision, not broken-and-unfixed.** It is correct
for still water and loses its one-cell interface within ten steps of a hull moving in it.
That is why free-surface liquid moved to SPH.

**Demo animations:** nine of the lattice cases animate, plus four rendered inside Isaac
Sim. A few older files on disk hold a single frame from before a colour-scaling bug was
fixed -- that bug also meant three clips were **saturated blocks of one colour** while
appearing to work.

---

## A short v0.2 list

1. **Settle the waterline question.** Run the two mass-conserving implementations side by
   side on one settled state with their intermediates printed. Until then no floating
   draft is trustworthy to better than a millimetre.
2. **Make a floating run repeatable.** Fixed-order summation over the fluid-boundary
   pairs, so a draft does not depend on thread scheduling.
3. **Sloshing's +5.1%.** The one experiment that separates a scheme error from a
   Mach-number artefact is a Mach sweep at a fixed grid, which is now possible and has
   not been run.
4. **The 2D-1 lift.**
5. **A round cup.** The cup is five boxes, because there is no hollow-vessel primitive.
6. **Train a policy.** The carry task exists and is verified; the spill weight in its
   reward is a guess, not a tuned number.
7. **`pip install -e ".[cad]"`** on the development box, which clears 18 of the 19
   unrelated test failures the pre-commit hook now lists explicitly.
