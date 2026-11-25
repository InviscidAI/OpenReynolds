# Two tutorials for the live backend

[`../../docs/live-sync.md`](../../docs/live-sync.md) is the reference page for
`openreynolds[live]`: every constructor argument, every face rule, every known issue with
its measurement. This page is the other thing -- two cases walked from the install to a
number, in the order a reader actually meets them, with the cost of each step beside it.

The two are chosen because they are the two halves of the extra and they are **not at the
same level of validation**:

| | tutorial 1, the drone | tutorial 2, the cup |
|---|---|---|
| backend | lattice Boltzmann, `backend="lbm"` | SPH, `free_surface=True` |
| state | **supported** | **beta**, and [what that means](#what-beta-means-here-in-numbers) is a section of its own |
| what it needs | a Warp device. **No Newton model and no Isaac Sim** | Newton, and `SolverMuJoCo`. Isaac Sim only for the cross-check rows |
| the graded number | an induced-velocity ratio against a 1955 model, 5 of 5 | an angle against a quadrature, and a load in newtons |
| where its numbers below come from | [`../../PROGRESS_LIVE.md`](../../PROGRESS_LIVE.md)'s table, commit `e5b1698`, an H100 80GB HBM3 | `openreynolds/live/carry.py`'s own docstring tables: Warp CPU, and an L40S inside a live Isaac Sim Kit process |

Read tutorial 1 first even if the cup is what you came for. It is the clearer
demonstration of why a fluid in the loop is worth its cost, it is the supported backend,
and it is the one whose answer has a published model behind it.

## Before either one

    pip install 'openreynolds[live]'

That pulls Warp, Newton and a MuJoCo backend. **Core OpenReynolds installs and imports
without it**, which is an acceptance criterion and not a nicety, so nothing here changes
anything about the OpenFOAM path the rest of this directory is about.

Then find out what device you have, because every cost on this page depends on it:

    openreynolds live bench --resolution 64 --resolution 128
    openreynolds live cases

`bench` names the device and the Warp version beside the rate, and it reports two
resolutions because a small lattice is launch-bound and a large one bandwidth-bound, so
one number is not a measurement. `cases` lists what can be run and marks the heavy ones
`[needs a GPU to be quick]`.

**If `openreynolds live bench` names `cpu`, stop here and run the cavity instead.** The
quickstart block in [`../../docs/live-sync.md`](../../docs/live-sync.md) is a lid-driven
cavity at Re 100 on a 24-cell lattice that needs no files and no arguments; it was run
verbatim out of that page on 2026-11-23 on warp 1.19.0 with no CUDA device and took
**9.7 s at 1.4 MLUPS**, printing `-0.20848` against Ghia, Ghia and Shin's `-0.21090`. That
is 1.1% at a resolution chosen to finish in ten seconds and it is **not** a validation
figure. It is, however, proof that the install works, and it is the cheapest such proof
there is.

Both tutorials below say where a GPU stops being a convenience, at the step where it
stops.

---

# Tutorial 1: a drone in ground effect

## The claim, and why a policy needs it

A rotor hovering near the floor does more with the same power than the same rotor in free
air. The mechanism is not subtle: the ground will not let the jet keep accelerating
downward, so the induced velocity at the disk falls, and at constant power a lower induced
velocity is more thrust. A quadrotor landing therefore floats, and one lifting off a
confined pad feels a push that is not in its thrust model.

A robot simulator that integrates rigid bodies has none of this. The rotor is a force, the
force is whatever the controller asked for, and the floor is a collision surface. A policy
trained there learns a plant whose thrust does not depend on altitude, and the first thing
it meets on real hardware is an altitude-dependent plant. That is what the `[live]` extra
is for, and `ground_effect` is the smallest case that shows it: one actuator disk, a
floor, and the same disk again with the floor taken away.

**The no-fluid comparison already exists** -- `openreynolds.live.isaac.HoverFluidEnv` with
`fluid_on=False` builds the same vehicle, reward, termination and observation with no
lattice at all, and that is the point-mass-plus-rigid-body task every quadrotor RL paper
actually trains. Read [what does not
work](#what-does-not-work-in-this-case-with-the-numbers) below before you believe a
difference that task reports, because it has a live defect in exactly that comparison.

## What the case is

`openreynolds/live/cases.py::run_ground_effect`, and every default below is read off its
signature rather than described:

| | |
|---|---|
| the rotor | one actuator disk, `radius` 120 mm, `thrust` 2.72 N, thickness two cells |
| heights | `Z/R` 0.75, 1, 1.5, 2 and 3, plus a free-air reference at `Z/R` 6 |
| resolution | `cells_per_r` 12, so `dx = R/12 = 10 mm` |
| the box | 16`R` square in plan and `max(Z + 8R, 12R)` tall, so 1.92 x 1.92 x 1.44 m at the lowest height |
| the faces | **slip** on the four sides, **outflow** on top, **wall** on the floor -- and an outflow floor instead for the free-air run, which is what "free air" means here |
| the operator | `regularized` collision, with the subgrid model on at `C_s = 0.12` |
| the coupling step | 2 ms, with the substep count chosen by `Scaling.for_coupling` from the peak velocity |

**A box of six outflows is refused, and that refusal is worth knowing before you build a
rotor case of your own.** An outflow imposes no density, so a weakly compressible lattice
with a momentum source in it and no density reference anywhere has a free mean density;
measured at six outflows round a hovering rotor, the whole lattice went non-finite inside
fifteen coupling steps. This case was first written that way. Slip sides conserve what
crosses them and pin it.

## Step 1: run it

    openreynolds live run ground_effect --json > ground-effect.json

One command runs all six configurations -- the five graded heights and the free-air
reference -- because the graded quantity is a ratio and a ratio needs both halves from the
same code at the same settings. **This is the step that wants a GPU.** The recorded H100
run of the six cost about **1,087 s** of H100 time; `cases.py`'s own comment carries that
figure, because the run in question was taken down by a bug in a report line after all
1,087 s with every number already computed.

**`--out DIR` is dropped on the way through, as of 2026-11-23**, so the figures land in
the working directory whatever you pass: `cli.py` passes it as `out=`, `cases.run_case`
takes `out_dir=`, and it ends up in the `**overrides` that every runner's trailing `**_`
swallows. Calling the runner from Python does honour it, which is also how to run one
height instead of six while you are finding your feet:

```python
from openreynolds.live.cases import run_ground_effect

result = run_ground_effect(heights=(0.75,), cells_per_r=8, out_dir="renders")
print("\n".join(result["report"]))
for check in result["checks"]:
    print(check)
```

`cells_per_r=8` is a coarser lattice than the graded 12 and is **not** the graded
configuration; it is there to make the shape of the output visible in minutes. Every
number in the rest of this tutorial is from the 12-cell run.

## Step 2: read the ratio, not the velocity

What the case samples is the mean axial velocity through the disk's own plane, averaged
over the last third of the sampling window. What it **grades** is that velocity divided by
the same disk's velocity in free air, and the division is the whole design: the disk's
finite thickness, how its momentum is spread over two cells, the lattice resolution and
the subgrid constant all appear in the numerator and in the denominator, and cancel.

The model is Cheeseman and Bennett (1955), ARC R&M 3021, equation (2), and **the original
report was read**. Three things the usual quotation drops:

- the constant inside the bracket is **4**, so 16 when squared:
  `T_g/T_inf = 1/(1 - R^2/(16 Z^2))`;
- the ratio is derived **at constant power**;
- `Z` is the **rotor's** height above the ground, not the undercarriage's.

The report states **no** numeric lower bound on `Z/R` -- only that small `Z/R` gives "large
errors" and is "of no practical interest" -- and its one number is in the conclusions, that
flight results agree for `Z/R > 0.6`. The `Z/R >= 0.25` figure that circulates could not
be traced to it.

A thrust ratio is not a quantity this case can measure, because an actuator disk is a
momentum source of a set strength: the thrust is the input. The induced velocity is
measurable, and the two are the same statement -- the ground's image source reduces the
induced velocity at the disk by `(R/4Z)^2`, so

    v_g / v_inf = 1 - (R / 4Z)^2

and that is what the five graded checks compare against, at a 10% bound. `Z/R` is the
rotor radius as the length scale and the disk's own free-air induced velocity as the
velocity scale; nothing else enters.

## Step 3: the five numbers

H100 80GB HBM3, commit `e5b1698`, read off
[`../../PROGRESS_LIVE.md`](../../PROGRESS_LIVE.md)'s summary table, which
`benchmarks/live/summarise.py --write` generates from the recorded `result.json` files
rather than being typed:

| `Z/R` | model `1 - (R/4Z)^2` | measured `v_g/v_inf` | error | verdict |
|---|---|---|---|---|
| 0.75 | 0.8889 | **0.897484** | **+0.97%** | PASS |
| 1 | 0.9375 | 0.99057 | +5.66% | PASS |
| 1.5 | 0.9722 | 1.04718 | +7.71% | PASS |
| 2 | 0.9844 | 1.05621 | +7.30% | PASS |
| 3 | 0.9931 | 1.03077 | +3.80% | PASS |

**The number to take away is the first row**, and it is the only row whose agreement is
better than the suite's own scatter: at three quarters of a radius off the floor the
induced velocity through the disk is 10.3% below the same disk's in free air, against a
model that says 11.1%, and the two agree to one per cent. That is the ground producing a
measurable change in what the rotor does, from a fluid rather than from a lookup table.

What it is worth in thrust is a one-line derivation and is labelled as one rather than
quoted as a measurement. Under Cheeseman and Bennett's own constant-power argument the
thrust ratio is the reciprocal of the induced-velocity ratio, so the model's
`1/0.8889 = 1.1250` is **12.5% more thrust for the same power** at `Z/R` 0.75, and
inverting the measured 0.897484 the same way gives **1.1142, +11.4%**. The case prints the
model's figure as `thrust_ratio_model` and grades neither.

## Step 4: the pair of animations, which is the measurement

    renders/ground-effect-disk-z0.75.mp4
    renders/ground-effect-free-air.mp4

Those two files **are** the comparison, and the case writes two rather than six for that
reason: the same disk with the floor under it and without it. In the first the jet forms,
reaches the ground and turns outward into a wall jet, with the fountain rising at both
sides. In the second there is no floor and the jet simply leaves. 200 frames each, from
the `gif-ground-effect` H100 run, 5 of 5 graded checks.
[`../../renders/README.md`](../../renders/README.md) is the provenance, including the mean
frame difference measured with PIL: **0.140** and **0.276** on 0-255 RGB.

**0.140 is not a dead file.** The ground-effect box is sixteen radii wide and most of it
is still air for the whole run, so a small mean frame difference is a true measure of a
picture in which a small, real thing moves. Compare a clip against its own still, not
against another clip. That number earned its place: files in this project were once
claimed to animate while holding exactly one frame, and `figures.animate` now refuses
fewer than two frames, all-identical frames, and frames that collapse on encoding.

## What does not work in this case, with the numbers

A tutorial that implies everything works is worse than none, so:

- **The free-air denominator reads low.** Measured 3.70291 m/s against momentum theory's
  `sqrt(T / (2 rho A))` = 4.9969 m/s, which is **-25.9%**. It is reported and not graded,
  and the reason it is tolerable is the reason the graded quantity is a ratio: an actuator
  disk of finite thickness on a lattice does not have to reproduce an inviscid
  momentum-theory value. The reason it is *listed* is the next item.
- **Three of the five ratios exceed 1, and the model cannot.** `1 - (R/4Z)^2` approaches 1
  from below, so a measured ratio above 1 is the denominator being low rather than the
  ground pushing harder -- and `Z/R` 1.5, 2 and 3 all come out above 1. The case reports
  "3 of 5" as a check of its own, ungraded. Read the other way round: **only `Z/R` 0.75
  and 1 reproduce the direction of the effect at all.** Inverting the measured ratios the
  way step 3 does gives thrust ratios of 0.955, 0.947 and 0.970 at the three outer
  heights, which is *less* lift than free air where the model says slightly more. The 10%
  bound is passed and the trend beyond two radii is not reproduced. The free-air run's
  floor is an outflow a few radii down rather than infinity and its -25.9% is roughly the
  size of the overshoot; a larger free-air box would tighten it and nobody has run one.
- **The subgrid model is carrying the run, and the run says so every time.** A 120 mm
  rotor pushing 5 m/s of air is a Reynolds number near 80,000 on the diameter, and there
  is no resolution at which a millisecond timestep resolves air's molecular viscosity. The
  answer away from the wall is the model's. It is legitimate *for this quantity*, because
  the ground effect is an inviscid image-source argument and the model it is compared
  against is itself inviscid, and `ground_effect` is the only graded case in the suite
  with the subgrid model on. `cases._official` refuses to grade any laminar case that way,
  because an eddy viscosity in a laminar flow lowers the effective Reynolds number and
  moves a drag coefficient *toward* the published value for reasons that have nothing to
  do with the solver being right.
- **One disk and not four, on purpose.** Sanchez-Cuevas, Heredia and Ollero (2017)
  measured that a complete multirotor does **not** follow Cheeseman and Bennett: the effect
  is larger and reaches about five radii rather than two, because of a fountain under the
  airframe. A quadrotor graded against this model would be graded against a model the
  later literature shows does not apply to it. The quadrotor case exists
  (`openreynolds live run quadrotor`) and is graded for bookkeeping and stability, not
  against a published number.
- **The Isaac Lab hover task applies its rotors' attitude moment twice when the fluid is
  on**, as of 2026-11-23. `FluidWorld.rotor_reaction` returns force *and* torque,
  `NewtonCoupling` adds all six components to `state.body_f`, and `isaac._Rig.apply` adds
  the same moment again through `control.joint_f` on the belief -- stated in its own
  docstring and at the call site -- that `rotor_reaction` carries force only, which was
  true until the torque was added on 2026-11-22. The force half is guarded by an `if`; the
  moment half is not. So a `HoverFluidEnv` with `fluid_on=True` rolls and pitches at twice
  the rate the same differential thrust gives it with `fluid_on=False`, **which is exactly
  the comparison that task exists to make.** It does not touch `ground_effect`, which
  builds no Newton model at all. It does mean that if you go from this tutorial to the
  hover task to measure what the fluid is worth to a policy, the attitude half of that
  answer is wrong today.
- **Two counters in `report()` are worth reading rather than skipping.** `fallbacks` counts
  links where the Bouzidi interpolation degraded to halfway bounce-back because the second
  fluid node was also solid, which for a disk two cells thick is a resolution statement
  about the disk. `shifted` has to be zero for any momentum balance to mean anything,
  because the moving window is an integer-cell gather and does not conserve momentum.

---

# Tutorial 2: a cup of water carried by a robot arm

## What this is

`openreynolds.live.carry.CarryFluidEnv`. A robot end effector holds an open cup with water
in it and has to move it to a goal pose. The water is **not** a payload mass: it is a
DFSPH free surface, it sloshes, it pushes back on the cup through Akinci boundary
particles, and the particles that leave the cup are the spill the reward is graded on.

    python -m openreynolds.live.carry --steps 120

`fluid_on=False` builds the same cup, controller, goal, reward and terminations with the
water replaced by a **rigid payload of the same mass**, which is exactly the pick-and-place
task every arm RL paper actually trains. Spill is identically zero there by construction
and not by luck. `main()` prints both, and that pair is what makes "the fluid changes the
task" a number rather than a claim.

**This is the beta backend. The next section is not optional reading.**

## What beta means here, in numbers

`DECISIONS_LIVE.md` D21 is the decision, dated 2026-11-23, and it spells the word out
rather than letting it mean nothing:

- **Fluid-only cases are bit-for-bit reproducible after a read-write race in the viscous
  term was fixed. Cases with a floating body are not.** The wrench's float32 atomics are
  bounded at 1.4e-8 relative on a *held* body and that grows about five orders of magnitude
  over a 4,000-step settle, so **no floating-body number should be reported from a single
  run.** Repeat it.
- **The per-step force on a freely floating hull carries 101.5% RMS against 23.0% held**,
  with the mean correct in both. The cause was found and it is the timestep, and it is a
  cliff rather than a slope: a quarter step takes the free hull to 1.3 to 2.6%, while
  halving it does essentially nothing. [`../live/README.md`](../live/README.md) has the
  full table at three particle radii with two independent settles of every configuration.
  The backend now takes the shorter step automatically whenever a body is coupled.
- **`sph_static_float` passes at 0.993x Archimedes on the corrected probe** (D19), and the
  other floating cases are re-graded on it with every verdict carrying which probe produced
  it.
- Why it shipped rather than being held, in D21's own words: it is the only backend in this
  project that does what the extension exists for -- a cup of water carried by a robot arm
  -- and that path is verified inside a live Isaac Sim Kit process. Holding it back would
  ship the benchmarks and not the product.

And the floating-draft question is **open**, with both numbers on the record: a hull that
should float with 16.00 mm submerged is reported at **14.51 mm** by the case and at
**15.90 mm** by a standalone implementation of the same mass-conserving calculation,
hand-run on the same settled state. **1.4 mm apart, at least one is wrong, and which one
has not been established.** What *was* established is the original and larger cause -- the
surface probe added one particle radius to the topmost particle in a column, which is right
for bulk water and **0.9047 mm short at a settled free surface**, repeatable to 0.01 mm
across four runs at two resolutions. The cup task grades no draft, so none of that reaches
a number below. It is the reason the word on this backend is "unverified" rather than
"inaccurate", and it is why every number in this tutorial names the machine it came from.

## Step 1: run it on a CPU

The defaults are sized for a laptop: `particle_radius` 5 mm so a 10 mm spacing,
8 x 8 x 4 = **256 fluid particles** in an 80 mm square cup filled 40 mm deep, about 900
boundary particles for the cup's five boxes, `device="cpu"`, `dt` 5 ms, `num_envs` 1.

    python -m openreynolds.live.carry --steps 120

The one number whose answer is known before the run is the floor load: a still cup's floor
carries the weight of the water in it. A prototype at a 12 mm spacing and 2,352 particles
ran **45 ms a coupling step** and read a floor load of **12.0 N against the 12.18 N weight
of its water, 98.5%**, after 50 steps of settling.
`tests/test_live_isaac.py::test_the_load_the_cup_feels_is_the_weight_of_the_water` is the
standing check on it, and it is the first thing to run on any machine, because a sign
convention and a magnitude are both in that one number and both have gone wrong in this
repository before.

**There is no arm.** No links, no joints, no IK, no reachability, no joint limits, no
self-collision. The cup is one free rigid body in Newton and the action is a Cartesian
velocity command, integrated into a pose setpoint that a Cartesian impedance controller
tracks through `control.joint_f`. That is the right v1 and the reason is worth repeating:
a Cartesian impedance controller taking a twist command is what an arm's top layer
actually is, so a policy trained here is giving the commands a real arm takes. What a real
arm adds underneath -- reachability, singularities, joint limits -- changes which commands
are achievable, not what the water does.

Three settings stop that controller being a teleport, and if you change them, change them
knowingly. `max_force` and `max_torque` clamp the PD output: without them the cup follows
the setpoint exactly and the water is a decoration. `gravity_feedforward` adds `m g` for
the **cup's own** mass only, because a real controller does not know how much water is
left and fighting that load is the task. And `control_hz`, 12 Hz by default, sets both
gains at once, critically damped; the arithmetic is in `carry.py`'s own docstring.

## Step 2: the observation is a wrench, and it is filtered

`(num_envs, 20)`: goal error 3, cup quaternion 4, linear velocity 3, angular velocity 3,
**filtered fluid force 3**, **filtered fluid torque 3**, spilled fraction 1.

The fluid term is the wrench on the cup rather than the free-surface height, for one reason
that outranks the others: **a wrist force/torque sensor is hardware that exists**, so a
policy whose only fluid input is the six numbers a real robot already measures can be put
on a real arm with no fluid estimator in the loop. `world.surface_height()` is a 2D grid
with no hardware analogue.

**The filtering is not optional, and the measurement says why.** The SPH force on a held
body carries a per-step RMS of **12.9% of the mean**, falling to **1.29% over ten
samples** -- a 10.01x reduction where white noise would give `sqrt(10) = 3.16`, so the
fluctuation is coherent and filters well. `wrench_filter` is a one-pole EMA at
`alpha = 0.2`, whose ten-sample equivalence is the whole reason for that number, and
`wrench_filter=1.0` turns it off so the difference can be measured rather than argued. On
the convergence sweep below the **unfiltered** per-step peak horizontal force reads 4.478,
3.285, 3.589 and 3.487 N against the filtered 2.573, 2.209, 2.245 and 2.192 N: a raw
wrench in the observation reports a peak load about **1.6x the real one**, which a policy
would read as signal.

## Step 3: refine, and read the right row

    python -m openreynolds.live.carry --mode convergence

The arm shuttles the cup at the cup's own first sloshing mode (334 ms, `lamb_period`) for
four cycles and then holds still for one more: 1.67 s simulated, 167 environment steps,
identical at every radius. Warp CPU backend, `--settle 100`, seed 0:

| radius | spacing | fluid | depth | `F_xy` peak | `F_xy` rms | `F_z`/weight | spill | of cup | ms/step |
|---|---|---|---|---|---|---|---|---|---|
| 10 mm | 20 mm | 32 | 2 p | 2.573 N | 1.087 N | 88.8% | 63.89 g | 25.0% | 17.4 |
| 5 mm | 10 mm | 256 | 4 p | 2.209 N | 1.086 N | 91.0% | 50.91 g | 19.9% | 65.2 |
| 2.5 mm | 5 mm | 2,048 | 8 p | 2.245 N | 1.147 N | 89.8% | 70.12 g | 27.4% | 677.9 |
| 2 mm | 4 mm | 4,000 | 10 p | **2.192 N** | **1.154 N** | 88.6% | 74.87 g | 29.3% | 1,283.1 |

**The force converges and the spill does not, and the second half is the honest one.**
Across a factor of 125 in particle count the peak horizontal load on the end effector
moves 2.573 -> 2.192 N and the last refinement moves it 2.4%; the RMS lands on 1.15 N and
moves 0.6%. Those are converged to the precision anything here is measured at. The spill
goes 25.0 -> 19.9 -> 27.4 -> 29.3% of the cup, is **not monotone**, and the last
refinement is still worth 1.9 points -- because it is a counting measurement in units of
one particle, and one particle is 3.1% of the cup at the top of that table against 0.025%
at the bottom. The coarse rows do not have the resolution to disagree with the fine ones.
**Read the fine row and treat the trend as a bound, not a limit.**

`F_z`/weight sits at 89 +/- 1% at every radius because it is the mean vertical load *while
the cup is being shaken*: a resonant shuttle unloads the floor on half of every cycle and
the EMA lags it. It is not the still cup, whose answer is step 1's.

**This is where the GPU stops being a convenience.** The finest row is 1,283 ms a coupling
step on one CPU core and the whole four-radius sweep is about 70 minutes. The same sweep
inside a live Isaac Sim Kit process on an **L40S**, on Kit's own Warp 1.17.0 out of
`isaacsim/extscache/omni.warp.core` rather than the pip 1.19, and its Newton 1.6.1,
`device="cuda:0"`, takes **86 s**:

| radius | fluid | `F_xy` peak | `F_xy` rms | `F_z`/weight | spill | of cup | ms/step |
|---|---|---|---|---|---|---|---|
| 10 mm | 32 | 2.529 N | 1.075 N | 87.3% | 63.89 g | 25.0% | 69.3 |
| 5 mm | 256 | 2.090 N | 1.072 N | 91.2% | 53.90 g | 21.1% | 16.4 |
| 2.5 mm | 2,048 | 2.159 N | 1.132 N | 89.0% | 68.75 g | 26.9% | 23.5 |
| 2 mm | 4,000 | 2.184 N | 1.136 N | 88.3% | 78.52 g | 30.7% | 28.6 |

Every force agrees with the CPU run to within 6% and the two finest rows to 2%, and the
spill to 5%. The coarsest row is *slower* than the next one because 32 particles do not
fill a GPU and the launch overhead is the whole cost -- which is also why `ms/step` stops
falling after 256 particles and the sweep is nearly free to refine.

Four particles through the depth resolves the sloshing mode and nothing else: no breaking
crest, no droplet, no film on the wall. A run that wants those wants
`particle_radius = 1.5 mm` or finer, which is about 16,000 fluid particles. **The depth in
particles is the number to quote**, and `report()` prints it.

Spill is measured, not modelled. `world.positions()` gives every fluid particle in metres;
a particle is out when its position in the cup's own frame leaves the cavity, and the flag
is **latched** -- once out, counted out -- because the domain box reflects escaped
particles and a droplet bouncing off the workspace floor back into the cup is an artefact
of that clamp rather than water that was never spilled. The quantity reported is
**kilograms**: `escaped_count * world.mass`, with `world.mass = rho0 * spacing^3` the
particle mass. Nothing in it is in arbitrary units.

## Step 4: the quantity that actually converges

    python -m openreynolds.live.carry --mode onset

`--mode pour` weighs mass at fixed angles; `--mode onset` measures the **angle** at a
fixed event, and that is the better conditioned of the two by a long way. Mass out of this
cup is a count quantised at `1/n_fluid`; a threshold angle is quantised only by how fast
the tilt sweeps through it -- `rate/100` radians at a 100 Hz step -- and refining the fluid
does not coarsen it.

Two closed forms get quoted for a tipping cup and **they are different formulas**:

    tan(theta) = D^2 / (4 a h) = 1.5625  ->  57.381 deg   the dry-floor wedge
    tan(theta) = (H - h) / R   = 1.5000  ->  56.310 deg   the rim form, floor wet across

The rim form carries a precondition -- the floor has to still be wet under the tilted
plane, `tan(theta) <= h/R` -- and this cup fails it at **45 degrees**, twelve below the
angle the formula itself returns (`h` is 40 mm against `H/2` = 50 mm, short by 10 mm). So
this was never a tie for a measurement to break: one of the two is being evaluated outside
its own domain. And the square plan is not the explanation, which is the thing to check
before believing that: `tests/test_live_isaac.py` integrates the water volume under the
onset plane assuming **neither** branch and gets 57.381 deg for the square cup to 0.01 deg,
56.80 on a cylinder of `R = a`, and 53.56 on the equal-area one. So the plan shape is worth
0.58 deg, the branch 1.07, and the `R` convention 3.3.

Measured on an L40S (`benchmarks/live/modal_run.py::onset`, Warp 1.19.0, pip Newton 1.8.0),
tilting about world y at 0.1 rad/s from a settled cup, the angle read off the cup's own
quaternion on a 0.057 deg/step grid, first escaped particle:

| radius | fluid | depth | one particle | onset | vs 57.38 | vs 56.31 | 5% out |
|---|---|---|---|---|---|---|---|
| 10 mm | 32 | 2 p | 7.986 g, 3.12% | 56.93 deg | -0.45 | +0.62 | 57.36 |
| 5 mm | 256 | 4 p | 0.998 g, 0.39% | 51.76 deg | -5.62 | -4.55 | 55.63 |
| 2.5 mm | 2,048 | 8 p | 0.125 g, 0.05% | 57.15 deg | -0.23 | +0.84 | 61.02 |
| 2 mm | 4,000 | 10 p | 0.064 g, 0.03% | **57.23 deg** | **-0.15** | +0.92 | 61.61 |

**The error converges to the wedge form and away from the rim form**: over the two finest
rows it falls 0.23 -> 0.15 deg while the error against `(H - h)/a` grows 0.84 -> 0.92. The
whole sweep is 290 s.

The error bar is **measured, not assumed**, and measuring it is D21's beta rule in
practice: SPH neighbour reductions are not bit-reproducible on the GPU, so a repeat is a
sample and not a copy. Three runs at 2 mm and 0.1 rad/s on the same seed gave 57.23, 57.18
and 57.53 deg; two at 5 mm gave 51.76 and 51.50. Both scatter **0.19 deg**, and the angle
between the first and the second particle out is 0.43, 0.00, 0.38 and 0.24 deg across the
four radii -- the same order, which is what says the bound is the one-particle event itself
rather than the integration. So the finest radius measures **57.31 +/- 0.19 deg**:

    D^2 / (4 a h)    57.381 deg    measured - predicted = -0.07 +/- 0.19    indistinguishable
    (H - h) / a      56.310 deg    measured - predicted = +1.00 +/- 0.19    5.3 sigma away
    (H - h) / R_eq   53.048 deg    measured - predicted = +4.26 +/- 0.19    22 sigma away

The wedge form is the one the measurement supports, the rim form is ruled out at five
sigma, **and the rim form was already ruled out by its own precondition before a particle
moved.** Those two agreeing is the result.

## What does not work in this case, with the numbers

- **The 5 mm row is not evidence for anything**, and its own crest says so. Its highest
  particle stands **22.2 mm above its bulk surface**, against 7.5, 13.4 and 10.3 mm at the
  other three radii. A first-particle onset is by construction an event in the crest and
  not in the bulk surface, so at 5 mm the crest clears the rim while the surface is still
  five degrees short of it -- which is why that row's first and second particle leave on the
  *same step*, 3.9 deg before the 5% threshold.
- **The coarse rows are not monotone, and the two O(spacing) errors pull in opposite
  directions**, so there is no reason they should shrink in step. The settled column stands
  2 to 4 mm proud of the still-water line at every radius and does not shrink with
  refinement, because it is not a discretisation error: the surface particles carry a
  density deficit, `density_error()["mean_deficit"]` is 0.078 to 0.095 on these runs, and
  an 8% deficit over a 40 mm column is 3 mm of dilation -- worth about **-2.5 deg**.
  Meanwhile `_Cup.outside` allows half a spacing past the rim before a particle counts as
  spilled, 10 mm at the coarsest radius against 2 mm at the finest -- worth about
  **+1.0 deg** at 2 mm and more at every coarser radius. They partly cancel, to a net
  -1.5 deg of artefact against a measured -0.07, so about a degree and a half is left over
  that this does not account for. **It is a budget, not a model.**
- **The tilt rate is a real term and it runs the other way from the obvious guess.** At
  5 mm, same seed: 55.90, 53.51, 51.50 and 51.21 deg at 0.40, 0.20, 0.10 and 0.05 rad/s. A
  *faster* tilt pours *later*, because the water lags the cup: at the moment the cup passes
  a given angle the surface has not redistributed to the plane that angle implies, so it is
  still low at the pouring rim. The quasi-static limit is about 51.1 deg and 0.1 rad/s is
  within 0.3 of it, which is what makes the radius sweep a near-quasi-static measurement
  rather than a rate measurement. At 2 mm the dependence has collapsed with the bulge that
  caused it -- 57.87, 57.59, 57.53 and 56.95 deg over the same factor of eight in rate, a
  0.92 deg spread against 5 mm's 4.69.
- **`--mode pour` is above its prediction everywhere**, by 4% at 80 degrees and **125% at
  60**. That ordering is the physical one and it is why a ratio is reported instead of a
  tolerance: `retained_fraction` is quasi-static and inviscid, so it misses the water that
  carries over the rim with momentum, and that miss is a roughly fixed quantity of water
  against a *predicted* transfer that shrinks to 31 g at 60 degrees -- three degrees past a
  57.4 deg threshold. At 80 degrees the prediction is 187 g and the same overshoot is 4%.
  The row that is a **test** rather than a comparison is the 45 deg one: below the spill
  angle it transfers **exactly 0.00 g** over three seconds, which says the walls hold and
  the spill counter is not counting the cup's own motion.
- **The cup is square, because there is no hollow-vessel primitive.** `sph._sdf` has
  sphere, box, capsule, cylinder and plane, and a *solid* cylinder sampled by
  `SphWorld._sample` becomes a two-layer shell -- a closed can, lid included, not a cup. So
  the cup is a floor box and four wall boxes. What that costs is the shape of the sloshing
  modes, at a slightly different frequency; a round cup wants a difference-of-SDFs sampler
  in `sph.py`.
- **`w_spill` is a guess and is labelled one.** At the default weights, spilling the entire
  cup over an episode costs 100 against a perfect 100-step episode's roughly +120 of
  shaping. **No policy has been trained on this task**, so that balance is a starting point
  and not a measurement.
- **`NewtonCoupling` will not turn a Newton model's shapes into boundary particles for you**
  on this path, and there is no periodic axis, because `wp.HashGrid` has no periodic mode.
  Call `world.add_shape(...)` once per coupled body and `world.boundary_from_domain()` for
  the walls *before* constructing the coupling. A world with neither is refused at
  construction, by name, because the alternative is bodies with no wetted surface and a
  wrench of exactly zero -- which reads as a coupling that is running and finding no force.
  `mesh=` shapes are refused by name on this path rather than sampled wrongly.
- **`surface_tension` must be 0.0**, and the knob refuses a non-zero value rather than
  accepting it and ignoring it. For a cup at these scales that is the right physics to
  leave out; for a droplet it *is* the physics, and this backend is not for droplets.
- **There is no `Scaling` on this backend to refuse a bad setup at construction**, because
  SPH has none of the three numbers `Scaling` exists to reconcile. `world.report()` is the
  setup review instead, and `density_error()` is the number to watch: a case whose
  compression is growing is a case whose timestep or tolerance is wrong.

---

# What both of these leave out

- **No trained policy.** Both tutorials end at a measured number about the *plant*. The
  claim that a policy trained without the fluid learns the wrong thing is argued here from
  the rotor's altitude dependence and from the cup's 2.2 N of sloshing load, and it is
  **not** demonstrated by a training curve, because nothing in this repository has trained
  one.
- **One lattice per `FluidWorld`.** There is no batched fluid, so `num_envs > 1` is N
  lattices and N times the work rather than one wider kernel. That is the single largest
  thing standing between either of these and an RL training run at scale.
- **No turbulent case with a reference.** `ground_effect` runs with the subgrid model on
  and is graded on a ratio in which that model cancels; nothing on this page grades a
  turbulence model against a measured flow. It is the same gap [`README.md`](README.md)
  calls the largest single one in its own index.
- **No grid study on the ground effect.** It is one resolution, 12 cells per rotor radius,
  so it has no claim about how much of its error is discretisation, and the case that does
  carry an observed order of convergence in this family is the cylinder rather than this
  one (`openreynolds live run cylinder`, three resolutions). The cup has a four-point
  sweep, which is why its force is quoted from the fine row and its spill mass is quoted
  as a bound.
- **No test count anywhere on this page**, deliberately.
  `tests/test_wiring.py::test_the_documents_do_not_carry_a_test_count` forbids one, because
  a number written into a document is true for one commit and silently wrong afterwards.
  Run the suite and read today's figure.
