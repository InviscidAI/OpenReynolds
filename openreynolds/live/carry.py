"""Carry a cup of water without spilling it, with the water simulated live.

    python -m openreynolds.live.carry --steps 120
    python -m openreynolds.live.carry --mode convergence   # the sloshing load and the
                                                           # spill, at three resolutions
    python -m openreynolds.live.carry --mode pour          # transferred mass against
                                                           # the geometric prediction

    from openreynolds.live.carry import CarryFluidEnv, CarryFluidEnvCfg

    env = CarryFluidEnv(CarryFluidEnvCfg(fluid_on=True))
    obs, _ = env.reset(seed=0)
    for _ in range(120):
        obs, reward, terminated, truncated, extras = env.step(policy(obs["policy"]))

A robot end effector holds an open cup with water in it and has to move the cup to a
goal pose. The water is **not** a payload mass: it is a DFSPH free surface
(`openreynolds/live/sph.py`), it sloshes, it pushes back on the cup through the
boundary particles, and the particles that leave the cup are the spill the reward is
graded on. The task's whole content is the trade between getting there and not
spilling, and both halves are measured rather than modelled.

## What this is, and the one thing it is not

**It is not a `DirectRLEnv` subclass, and it cannot be, for the reason
`openreynolds/live/isaac.py` records at length and this file does not repeat.**
`isaaclab.envs.DirectRLEnvCfg` has five fields with no default and one of them is
`scene: InteractiveSceneCfg`; an `InteractiveScene` cannot be built without Isaac Sim's
USD stage, so a class that inherited `DirectRLEnv` here would import, fail on
construction, and be a claim this repository cannot stand behind.

What it is instead is a sibling of `HoverFluidEnv`: the same conformance to
`isaaclab.envs.direct_rl_env.DirectRLEnv`, checked method for method in
`tests/test_live_isaac.py` against both literal signatures and -- where `isaaclab` is
importable -- the installed package. Five-tuple `step`, SAME_STEP autoreset,
`is_vector_env`, `{"policy": ...}` observations, torch tensors on `self.device`.

## Why SPH and not the lattice

Free-surface liquid is SPH's job. The lattice backend's free surface is frozen
(`DECISIONS_LIVE.md` D15) and `FluidWorld` refuses `free_surface=True` by name. There
is no cup of water on the lattice to be had, so this task is an `SphWorld` directly,
with the cup's walls as Akinci boundary particles bound to the moving body
(`docs/sph-contract.md`).

## The cup is five boxes, and it is square

`sph._sdf` has sphere, box, capsule, cylinder and plane, and none of them is a hollow
vessel. A **solid** cylinder sampled by `SphWorld._sample` becomes a two-layer shell --
which is a closed can, lid included, not a cup. So the cup is a floor box and four wall
boxes and is therefore square in plan. That is a real limitation and the only thing it
costs is the shape of the sloshing modes: a square tank's first mode is the same
physics as a round one's, at a slightly different frequency. A round cup wants a
difference-of-SDFs sampler in `sph.py`, which is a change to a file this task does not
own.

Every dimension is a whole number of particle spacings, because `SphWorld._lattice`
puts candidate sites at `origin + (i + 1/2) * spacing` and a wall whose thickness is
not a multiple of the spacing samples to one layer in some places and three in others.
`_check_cup` refuses a configuration that samples the walls unevenly rather than
running one.

## The arm: what is here and what is honestly not

**There is no arm.** No links, no joints, no IK, no reachability, no joint limits and
no self-collision. The cup is one free rigid body in Newton, and the action is a
**Cartesian velocity command** which is integrated into a pose setpoint that a
Cartesian PD (impedance) controller tracks through `control.joint_f` -- the same
channel `isaac.py::_Rig.apply` writes the quadrotor's wrench into, and the only one
that survives `State.clear_forces()` between the fluid and the solver.

That is deliberately a v1 and it is the right v1: a Cartesian impedance controller
taking a twist command is what an arm's top layer actually is, so a policy trained
here is giving the commands a real arm takes. What a real arm adds underneath --
reachability, singularities, joint limits -- changes which commands are achievable,
not what the water does. Bolting this onto an `ArticulationCfg` under an
`InteractiveSceneCfg` is the Isaac Sim version and is three fields and no new physics;
see `isaac.py`'s "How it would be adapted with Isaac Sim present".

Three numbers make the controller a controller and not a teleport:

- `max_force` / `max_torque` clamp the PD output. Without them the cup follows the
  setpoint exactly and the water is a decoration.
- `gravity_feedforward` adds `m g` for the **cup's own** mass, which is what every
  real arm's controller does from its own link model. The water's weight is **not**
  fed forward: the controller does not know how much water is left, and fighting that
  load is the task.
- the gains come from one knob, `control_hz`: `kp = m (2 pi f)^2`, `kd = 2 m (2 pi f)`,
  critically damped. At the default 12 Hz and a 250 g cup that is 1,421 N/m and
  37.7 N.s/m, and the droop under the water's 2.5 N is 1.8 mm.

## The observation, and why the fluid part of it is a wrench

`(num_envs, 20)`: goal error 3, cup quaternion 4, linear velocity 3, angular velocity
3, **filtered fluid force 3**, **filtered fluid torque 3**, spilled fraction 1.

The fluid term is the wrench on the cup and not the free-surface height, for one
reason that outranks the others: **a wrist force/torque sensor is hardware that
exists**, and a policy whose only fluid input is the six numbers a real robot already
measures is one that can be put on a real arm without a fluid estimator in the loop.
`world.surface_height()` is a 2D grid with no hardware analogue, it needs a
representation choice this task would then be stuck with, and it is a host read of the
whole particle array.

**It is filtered, and that is not optional.** The SPH force on a held body carries a
per-step RMS of 12.9% of the mean, falling to 1.29% over ten samples -- a 10.01x
reduction where white noise would give sqrt(10) = 3.16, so the fluctuation is coherent
and filters well. A raw per-step wrench in the observation is mostly noise and in the
reward is worse. `wrench_filter` is a one-pole EMA at `alpha = 0.2`, whose ten-sample
equivalence is the whole reason for that number; `wrench_filter=1.0` turns it off and
is there so the difference can be measured rather than argued.

Normalised by the **initial** water weight (and by weight x cup half-width for the
torque) so the numbers are O(1) and so a half-empty cup reads as a half-weight load
rather than being renormalised back to 1.

## The reward, and the spill in it

Per environment, per step:

    + w_progress * (d_previous - d)           metres closed on the goal, telescoping
    + exp(-(d / pos_scale)^2)                 1 at the goal, falls off over pos_scale
    - w_spill * spilled_fraction_this_step    the term the task is named after
    - w_tilt * (1 - up_z)                     up_z is the cup's z axis in the world
    - w_action * mean(a^2)
    - crash_penalty                           on the step that terminates

A terminating step scores **exactly** `-crash_penalty` and none of the shaping terms,
for `isaac.py`'s reason: one of the ways to terminate is a non-finite state, and a
shaping term on a NaN pose returns NaN into the replay buffer and poisons a training
run hours later with nothing in the logs pointing here.

**Spill is measured, not modelled.** `world.positions()` gives every fluid particle in
metres; a particle is out when its position in the cup's own frame is outside the
cavity. The flag is **latched** -- once out, counted out -- because the domain box
reflects escaped particles (`sph.clamp_to_domain`) and a droplet bouncing off the floor
of the workspace back into the cup is an artefact of that clamp and not water that was
never spilled. The quantity reported is **kilograms**: `escaped_count * world.mass`,
where `world.mass = rho0 * spacing^3` is the particle mass, and the fraction is of the
water the cup started with. Nothing here is in arbitrary units.

`w_spill` is the knob the whole task balances on and it is a **guess**: at the default
weights, spilling the entire cup over an episode costs 100 against a perfect 100-step
episode's roughly +120 of shaping. No policy has been trained on this, so that balance
is a starting point and not a measurement, and it is said here rather than implied by
a number that looks settled.

## Termination

Terminated is: the cup emptied past `empty_fraction`; **dropped**, meaning the
controller lost it -- the tracking error `|setpoint - cup|` past `drop_distance`, which
is what force saturation looks like from the outside; tipped past `min_upright`; the
cup outside the workspace, because outside it there is no domain for the water to be
in; or a non-finite number anywhere. Truncated is `episode_length_s`, kept apart from
termination so a value function bootstraps through one and not the other.

## `fluid_on`, which is the measurement

`fluid_on=False` is the same cup, the same controller, the same goal, the same reward
and the same terminations, with the water replaced by a **rigid payload of the same
mass** bolted to the cup -- which is exactly the pick-and-place task every arm RL paper
actually trains. Spill is identically zero there, by construction and not by luck, and
that is the point: it is the baseline against which "the fluid changes the task" is a
number. `main()` prints both.

## Cost, and what a real run wants

The default is `particle_radius = 5 mm`, so a 10 mm spacing, 8 x 8 x 4 = 256 fluid
particles in an 80 mm square cup filled 40 mm deep, and about 900 boundary particles
for the cup's five boxes. Measured on Warp's CPU backend, a prototype at a 12 mm
spacing and 2,352 particles ran 45 ms a coupling step, and the still cup read a floor
load of 12.0 N against the 12.18 N weight of the water in it (98.5%) after 50 steps of
settling.

Four particles through the depth resolves the sloshing mode and nothing else: no
breaking crest, no droplet, no film on the wall. A run that wants those wants
`particle_radius = 1.5 mm` or finer, which is about 16,000 fluid particles and a GPU's
work. The depth in particles is the number to quote and `report()` prints it.

## The sloshing load and the spill, at four particle counts

`--mode convergence`, which is `convergence` below: the arm shuttles the cup at the
cup's own first sloshing mode (334 ms, `lamb_period`) for four cycles and then holds
still for one more, 1.67 s simulated and 167 environment steps, identical at every
radius. Measured on Warp's CPU backend, `--settle 100`, seed 0:

    radius  spacing  fluid  depth   F_xy peak  F_xy rms  F_z/weight    spill   of cup
    10 mm    20 mm      32   2 p     2.573 N   1.087 N      88.8%     63.89 g   25.0%
     5 mm    10 mm     256   4 p     2.209 N   1.086 N      91.0%     50.91 g   19.9%
     2.5 mm   5 mm   2,048   8 p     2.245 N   1.147 N      89.8%     70.12 g   27.4%
     2 mm     4 mm   4,000  10 p     2.192 N   1.154 N      88.6%     74.87 g   29.3%

**The force converges and the spill does not, and the second is the honest half.**
Across a factor of 125 in particle count the peak horizontal load on the end effector
moves 2.573 -> 2.192 N and the last refinement moves it 2.4%; the RMS lands on 1.15 N
and moves 0.6%. Those are converged to the precision anything here is measured at. The
spill goes 25.0 -> 19.9 -> 27.4 -> 29.3% of the cup, is not monotone, and the last
refinement is still worth 1.9 points: **it is a counting measurement in units of one
particle**, and one particle is 3.1% of the cup at the top of this table against 0.025%
at the bottom. The coarse rows do not have the resolution to disagree with the fine
ones. Read the fine row and treat the trend as a bound, not a limit.

`F_z/weight` is the mean vertical load **while the cup is being shaken**, not the still
cup: it sits at 89 +- 1% at every radius because a resonant shuttle unloads the floor on
half of every cycle and the EMA lags that. The still cup is the one whose answer is known
in advance, and `tests/test_live_isaac.py::test_the_load_the_cup_feels_is_the_weight_of_
the_water` is the standing check on it.

The filter, on the same runs: the **unfiltered** per-step peak horizontal force is
4.478, 3.285, 3.589, 3.487 N against the 2.573, 2.209, 2.245, 2.192 N above. So a raw
wrench in the observation reports a peak load 1.6x the real one, which is the measured
12.9%-per-step fluctuation showing up where a policy would read it as signal.

Cost, same runs, same machine: 17.4, 65.2, 677.9 and 1,283.1 ms a coupling step, so the
finest row is a GPU's job and `--device cuda:0` is the same command.

**The same sweep inside a live Isaac Sim Kit process on an L40S**, on Kit's own Warp
(1.17.0 out of `isaacsim/extscache/omni.warp.core`, not the pip 1.19) and its Newton
1.6.1, `device="cuda:0"`:

    radius  fluid  F_xy peak  F_xy rms  F_z/weight    spill   of cup   ms/step
    10 mm      32   2.529 N   1.075 N      87.3%     63.89 g   25.0%     69.3
     5 mm     256   2.090 N   1.072 N      91.2%     53.90 g   21.1%     16.4
     2.5 mm 2,048   2.159 N   1.132 N      89.0%     68.75 g   26.9%     23.5
     2 mm   4,000   2.184 N   1.136 N      88.3%     78.52 g   30.7%     28.6

Every force agrees with the CPU run to within 6% and the two finest rows to 2%, the
spill to 5%, and the whole four-radius sweep takes **86 s** against about 70 minutes on
the CPU. The coarsest row is slower than the next one because 32 particles do not fill a
GPU and the launch overhead is the whole cost -- which is also why `ms/step` stops
falling after 256 particles and the sweep is nearly free to refine.

## The pour, against the geometry

`--mode pour`, which is `pour` below; `spill_angle` and `retained_fraction` are the
closed form it is measured against, derived in their own docstrings. The cup's spill
angle at the defaults is **57.4 degrees**, `tan = D^2 / 4ah = 1.5625`. Tilted at
0.4 rad/s about the world y axis and held 3 s, at `particle_radius = 5 mm` (256
particles, 255.5 g of water):

    target   reached   transferred   predicted   ratio   left in cup
      45 deg  45.5 deg     0.00 g      0.00 g      --      255.54 g
      60 deg  60.6 deg    68.88 g     30.55 g     2.25     186.67 g
      70 deg  70.3 deg   128.77 g    112.72 g     1.14     126.77 g
      80 deg  80.2 deg   193.65 g    186.53 g     1.04      61.89 g

**Below the spill angle it transfers exactly nothing**, which is the one row that is a
test rather than a comparison: a cup tilted 45 degrees for three seconds and losing no
mass at all says the walls hold and the spill counter is not counting the cup's own
motion.

Above it the measured transfer is above the prediction everywhere, by 4% at 80 degrees
and by 125% at 60. That ordering is the physical one and it is why the ratio is reported
instead of a tolerance: `retained_fraction` is quasi-static and inviscid, so it misses
the water that carries over the rim with momentum, and that miss is a roughly fixed
quantity of water against a *predicted* transfer that shrinks to 31 g at 60 degrees --
three degrees past a 57.4 degree threshold. At 80 degrees the prediction is 187 g and
the same overshoot is 4%. The angle reached is the cup's own, read back from its
quaternion, and the controller tracks the setpoint to within 0.6 degrees.

Cross-checked on an L40S inside a live Kit process at the same settings, on Kit's own
Warp and Newton: 0.00 g, 67.88 g, 126.77 g and 195.65 g, ratios --, 2.21, 1.12 and 1.05,
which is within 1.5% of the CPU run at every angle. Four angles in 79 s.

**Every mass in that table is a count of particles**, and the table now prints the
quantum beside it: at 5 mm the cup holds 256 of them, so one particle is 0.998 g =
0.391% of the cup and the four transfers are 0, 69, 129 and 194 particles. Those are
hundreds of quanta, so the pour masses are not quantum-limited and the ratios above are
real. The spill in `convergence` is the one that is, and `--mode onset` below is the
measurement that does not have the problem at all.

## The spill angle, which is the quantity that converges

`--mode onset`, which is `spill_onset` and `onset_convergence` below. `pour` weighs mass
at fixed angles; this measures the **angle** at a fixed event, and that is the better
conditioned of the two by a long way. Mass out of this cup is a count, quantised at
1/`n_fluid`; a threshold angle is quantised only by how fast the tilt sweeps through it,
`rate / 100` radians at a 100 Hz step, and refining the fluid does not coarsen it.

**Two closed forms get quoted for a tipping cup and they are different formulas.**

    tan(theta) = D^2 / (4 a h) = 1.5625 -> 57.381 deg   `spill_angle`, dry-floor wedge
    tan(theta) = (H - h) / R   = 1.5000 -> 56.310 deg   `rim_angle`, floor wet across

The second is the trapezoid branch `spill_angle` already derives, written for a round
cup; it is the right formula for a cup filled past half. It carries a precondition --
the floor has to still be wet under the tilted plane, `tan(theta) <= h / R` -- and this
cup fails it at **45 degrees**, twelve below the angle the formula itself returns. So
this was never a tie for a measurement to break: one of the two is being evaluated
outside its own domain. `h = 40 mm` against `H / 2 = 50 mm`, short by 10 mm.

**The square plan is not the explanation, which is the thing to check before believing
the above.** `tests/test_live_isaac.py::test_the_spill_angle_is_the_one_quadrature_
agrees_with_and_the_rim_form_is_not` integrates the water volume under the onset plane
assuming **neither** branch and gets 57.381 deg for the square cup, to 0.01 deg. The
same integral on a cylinder of `R = a` gives 56.80, and on one of the equal-area
`R = 45.14 mm` gives 53.56. So the plan shape is worth 0.58 deg, the branch 1.07, and
the `R` convention 3.3.

Measured on an **L40S** (`benchmarks/live/modal_run.py::onset`, Warp 1.19.0, pip Newton
1.8.0), tilting about world y at 0.1 rad/s from a settled cup, the angle read off the
cup's own quaternion on a 0.057 deg/step grid, first escaped particle:

    radius  fluid  depth   1 particle      onset    err vs 57.38  err vs 56.31   5% out
    10 mm      32   2 p   7.986 g 3.12%   56.93 deg    -0.45         +0.62       57.36
     5 mm     256   4 p   0.998 g 0.39%   51.76 deg    -5.62         -4.55       55.63
     2.5 mm  2,048  8 p   0.125 g 0.05%   57.15 deg    -0.23         +0.84       61.02
     2 mm    4,000 10 p   0.064 g 0.03%   57.23 deg    -0.15         +0.92       61.61

**The error converges to the wedge form and away from the rim form.** Over the two
finest rows it falls 0.23 -> 0.15 deg while the error against `(H - h) / a` grows
0.84 -> 0.92; a linear extrapolation in the spacing puts the converged onset at
57.5 deg, 0.2 above the wedge and 1.2 above the rim form. The whole sweep is 290 s.

The coarse rows are not monotone, and the two things that put an O(spacing) error on
this measurement pull in **opposite directions**, so there is no reason they should
shrink in step. Both are measured on the settled cup rather than asserted -- 100
coupling steps, Warp CPU, particle heights in the cup's own frame, the free surface
taken as the median of the per-column topmost particle plus half a spacing:

    radius  bulk surface   above the 40 mm line   highest particle   crest above bulk
    10 mm      41.6 mm            +4.0%                39.1 mm            7.5 mm
     5 mm      43.7 mm            +9.3%                60.9 mm           22.2 mm
     2.5 mm    42.8 mm            +7.0%                53.7 mm           13.4 mm
     2 mm      44.0 mm            +9.9%                52.3 mm           10.3 mm

* **The column stands 2 to 4 mm proud of the still-water line, at every radius**, and
  it does not shrink with refinement, because it is not a discretisation error: the
  surface particles carry a density deficit -- `density_error()["mean_deficit"]` is
  0.078 to 0.095 on these four runs -- and an 8% deficit over a 40 mm column is 3 mm of
  dilation. At the 2 mm row's measured 44.0 mm the wedge form gives 54.9 rather than
  57.4, so this is worth about **-2.5 deg**.
* **`_Cup.outside` allows half a spacing past the rim** before a particle counts as
  spilled, 10 mm at the coarsest radius against 2 mm at the finest. Raising the rim
  from 100 to 102 mm on that same dilated column gives 55.9 rather than 54.9, so this
  is worth about **+1.0 deg** at 2 mm and more at every coarser radius.

They partly cancel, to a net -1.5 deg of artefact at 2 mm against a measured -0.07, so
about a degree and a half is left over that this does not account for -- the two terms
above are a budget, not a model, and a cup tipping is not a cup standing still. It does
not need to be accounted for to pick a formula: the two forms are 1.07 deg apart, the
error against the wedge form shrinks with refinement (-0.45, -0.23, -0.07) while the
error against the rim form grows (+0.62, +0.84, +1.00), and a bias that pushed the
measurement toward the wedge form would have to be a coincidence that also reversed
sign between the two candidates.

**What is wrong with the 5 mm row is the crest, and the table above shows it.** Its
highest particle stands 22.2 mm above its bulk surface, against 7.5, 13.4 and 10.3 mm
at the others. A first-particle onset is by construction an event in the crest and not
in the bulk surface, and at 5 mm the crest is tall enough to clear the rim while the
surface is still 5 degrees short of it -- which is why that row's first and second
particle leave on the **same step**, 3.9 deg before the 5% threshold. The coarse rows
are not evidence for `(H - h) / R`; the 5 mm row is not evidence for anything.

**The tilt rate is a real term and it runs the other way from the obvious guess.** At
5 mm, same seed:

    rate        onset     err vs 57.38    5% out
    0.40 rad/s  55.90 deg    -1.48        61.21
    0.20 rad/s  53.51 deg    -3.87        57.86
    0.10 rad/s  51.50 deg    -5.89        55.71
    0.05 rad/s  51.21 deg    -6.17        54.08

A **faster** tilt pours **later**, because the water lags the cup: at the moment the cup
passes a given angle the surface has not yet redistributed to the plane that angle
implies, so it is still low at the pouring rim. The quasi-static limit is about 51.1 deg
and 0.1 rad/s is within 0.3 of it, which is what makes the radius sweep above a
near-quasi-static measurement rather than a rate measurement. It is also not in
contradiction with `pour`'s ratios being above 1: a fast tilt delays the onset and then
throws more water over the rim once it is past it.

At 2 mm the rate dependence has collapsed with the bulge that caused it -- 57.87, 57.59,
57.53 and 56.95 deg over the same factor of eight in rate, a 0.92 deg spread against
5 mm's 4.69 -- which is the same evidence read a second way: what the tilt rate acts on
is the time the surface has to redistribute, and a resolved surface has little to
redistribute.

**The error bar on a first-particle event is about a fifth of a degree and it is
measured, not assumed.** The SPH neighbour reductions are not bit-reproducible on the
GPU, so a repeat is a sample and not a copy. Three runs at 2 mm and 0.1 rad/s, same
seed, gave 57.23, 57.18 and 57.53 deg; two at 5 mm gave 51.76 and 51.50. Both scatter
0.19 deg, and the angle between the first and the second particle out is 0.43, 0.00,
0.38 and 0.24 deg across the four radii -- the same order, which is what says the bound
is the one-particle event itself and not the integration.

So the finest radius measures **57.31 +- 0.19 deg** against a 1.07 deg gap between the
two closed forms:

    D^2 / (4 a h)   57.381 deg   measured - predicted = -0.07 +- 0.19   indistinguishable
    (H - h) / a     56.310 deg   measured - predicted = +1.00 +- 0.19   5.3 sigma away
    (H - h) / R_eq  53.048 deg   measured - predicted = +4.26 +- 0.19   22 sigma away

The wedge form is the one the measurement supports, the rim form is ruled out at five
sigma, and the rim form was already ruled out by its own precondition before a particle
moved. Those two agreeing is the result.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from .isaac import AUTORESET_SAME_STEP, G, _torch

__all__ = ["CarryFluidEnvCfg", "CarryFluidEnv", "lamb_period", "spill_angle",
           "rim_angle", "retained_fraction", "shake", "convergence", "pour",
           "spill_onset", "onset_convergence", "main"]


@dataclass
class CarryFluidEnvCfg:
    """Everything `CarryFluidEnv` reads, in SI, with Isaac Lab's five required fields.

    Same discipline as `HoverFluidEnvCfg`: `DirectRLEnvCfg`'s five fields with no
    default are all present with values, and its other 19 are absent rather than
    present-and-ignored.
    """

    # -- the five Isaac Lab requires ----------------------------------------------------

    decimation: int = 2
    """Physics (and coupling) steps per environment step. The env's step is
    `dt * decimation`, so the default is a 100 Hz policy over a 200 Hz controller."""
    episode_length_s: float = 1.0
    """Seconds before truncation. One second is 100 env steps at the defaults, which is
    long enough to carry the cup the default 120 mm and short enough to demo."""
    scene: Any = None
    """`None`, where Isaac Lab wants an `InteractiveSceneCfg`. Not a stub for one: an
    `InteractiveScene` needs Isaac Sim's USD stage. `_Cup` does the scene's job -- the
    body, its shapes and `num_envs` copies of them -- from `newton.ModelBuilder` and
    `SphWorld`. The field exists so a `DirectRLEnvCfg` and this can be fed to the same
    code, and so the day Isaac Sim is present this is the one field that changes."""
    observation_space: int = 20
    """Observation width: goal error 3, quaternion 4, linear velocity 3, angular
    velocity 3, filtered fluid force 3, filtered fluid torque 3, spilled fraction 1.
    An `int` rather than a `gymnasium.spaces.Box`, which `DirectRLEnvCfg` also accepts,
    so this config can be built without gymnasium installed."""
    action_space: int = 6
    """A Cartesian twist command in [-1, 1]: three linear, three angular. Scaled by
    `max_speed` and `max_rate` and integrated into the pose setpoint the controller
    tracks. Not joint torques -- there are no joints; see the module docstring."""

    # -- the fluid ----------------------------------------------------------------------

    fluid_on: bool = True
    """Whether the cup holds simulated water or a rigid payload of the same mass.
    `False` is the baseline the fluid's effect is measured against, and its spill is
    identically zero."""
    particle_radius: float = 0.005
    """SPH sizing knob in metres; the spacing is twice it. The whole cost of the run is
    here. The default puts four particles through the water's depth, which resolves the
    sloshing mode and nothing finer; the module docstring says what a real run wants."""
    fluid: str = "water"
    """A name in `units.FLUIDS`, or a `units.Fluid`."""
    dt: float = 0.005
    """Physics step in seconds: Newton's step and the interval one coupling step
    advances the fluid by. `SphWorld` derives its substep count from it and the CFL
    limit, and refuses a combination that breaks it."""
    device: str = "cpu"
    """The torch device every returned tensor lives on, and the Warp device the
    particles are allocated on. One string for both, because two would let them
    disagree and the symptom is a silent host copy every step."""
    num_envs: int = 1
    """Independent environments. Each gets its own `newton.Model`, its own solver and
    its own `SphWorld`, stepped in a Python loop: wall clock is linear in this and there
    is no batching, for the reason `HoverFluidEnv`'s docstring gives."""

    # -- the cup ------------------------------------------------------------------------

    cup_half_width: float = 0.04
    """Inner half-width of the square cup in metres. 80 mm across at the default."""
    cup_depth: float = 0.10
    """Inner depth in metres, floor to rim. Deep enough that the settled surface -- which
    at four particles through the depth sits about 40% above the fill line, the coarse
    free surface's own bulge -- has room under the rim that is not already a spill."""
    wall_thickness: float | None = None
    """Cup wall and floor thickness in metres. `None` is **two particle spacings**, which
    is the thing to leave it at: it tracks `particle_radius`, so refining the fluid thins
    the wall instead of leaving a 20 mm slab on an 80 mm cup.

    Two and not one, measured on this cup at a 10 mm spacing over 120 settling steps:

        wall        boundary particles   F_z/weight   lowest water below the floor
        1 spacing          388              75%              -145 mm
        2 spacings         928             100.2%              +1.8 mm

    One layer does not hold this fluid in: the water goes **through the floor** and the
    load the cup feels is three quarters of the water's weight, which is exactly the
    shape of a defect that gets diagnosed as a coupling sign error. `sph.add_shape`'s
    own table measures one layer as adequate on a 280 mm tank at a 20 mm spacing, which
    is four times the wall-to-support ratio this cup has; the difference is the cup, not
    a contradiction. `_check_cup` refuses fewer than two rather than running one."""
    fill_depth: float = 0.04
    """Still-water depth in the cup at reset, in metres. Half full at the defaults,
    which is what leaves the sloshing room to matter."""
    cup_mass: float = 0.5
    """Mass of the cup and the gripper holding it, in kg, water excluded. It sets the
    controller gains through `control_hz` and is the mass `gravity_feedforward`
    compensates.

    **It is also what keeps the partitioned coupling quiet, and that is why it is 500 g
    and not a mug's 250.** The fluid and the body are advanced in sequence -- the water
    sees last step's cup pose, the cup sees last step's water wrench -- and a staggered
    scheme like that rings when the fluid's added mass approaches the body's. Measured
    on this cup over 40 steps with the arm holding still, reading the raw per-coupling-
    step force on the cup:

        cup_mass   cup/water mass   mean F_z         ripple RMS
         0.25 kg       0.98         100% of weight   99% of the mean
         0.50 kg       1.96         100%              2%
         1.00 kg       3.91         100%              2%
         0.25 kg at half the timestep                89%

    The **mean is right at every ratio** -- the coupling conserves what it should -- and
    at a ratio near 1 the per-step force alternates between about twice the weight and
    zero. Halving `dt` barely touches it, which is the signature of added mass rather
    than of a timestep: the instability is in the ratio, not the discretisation.

    So 0.5 kg, where the ripple is 2%, and the knob is here rather than hidden. A run
    that wants a light cup wants the fix this does not have: a strongly coupled
    iteration (or a predicted added-mass term) inside `NewtonCoupling.step`, which is a
    change to `couple.py` and not to this file. `_apply_action` says what the ringing
    does to the observation even at ratios where it is small."""

    # -- the arm's controller -----------------------------------------------------------

    control_hz: float = 12.0
    """Closed-loop bandwidth of the Cartesian PD, in Hz. One knob for all four gains:
    `kp = m (2 pi f)^2` and `kd = 2 m (2 pi f)`, critically damped, and the same with
    the cup's inertia for the angular pair. The calibration knob a real arm needs:
    12 Hz is a stiff-but-not-rigid wrist, and `f * dt` must stay well under 1/(2 pi)
    for the explicit integration to hold -- at the defaults it is 0.06."""
    max_force: float = 20.0
    """Newtons the controller can apply to the cup, each axis clamped. Must exceed the
    weight of the cup and a full load of water (5.0 N at the defaults) or the cup cannot
    be held at all; the headroom above that is the acceleration available, and it is
    what makes moving fast a choice rather than free."""
    max_torque: float = 2.0
    """Newton-metres, each axis clamped. The same trade about the cup's attitude."""
    gravity_feedforward: bool = True
    """Whether the controller adds `cup_mass * g` to hold its own link up, which is what
    a real arm's controller does from its own model. The **water's** weight is never fed
    forward -- the controller does not know how much is left, and fighting that load is
    the task. `False` makes the cup sag by the whole weight and is there so the term can
    be measured rather than believed."""
    max_speed: float = 0.5
    """Metres a second at `action = +-1` on a linear axis: the commanded twist's scale."""
    max_rate: float = 2.0
    """Radians a second at `action = +-1` on an angular axis."""

    # -- the workspace and the goal -----------------------------------------------------

    workspace: tuple[float, float, float] = (0.12, 0.12, 0.08)
    """Half-extents in metres of the box the cup's centre may be in, about the start
    position. Leaving it terminates: outside it there is no fluid domain, because the
    SPH domain is sized from it."""
    goal_distance: float = 0.12
    """How far the goal is from the start, in metres. The direction is random per
    episode, in the horizontal plane, so a policy cannot learn one trajectory."""
    pos_scale: float = 0.02
    """Metres at which the Gaussian goal reward has fallen to `1/e`."""

    # -- the reward ---------------------------------------------------------------------

    w_progress: float = 20.0
    """Reward per metre closed on the goal. Telescoping, so the whole episode's progress
    term is `w_progress * (d_start - d_end)` however it got there -- which is why the
    policy cannot farm it by oscillating, the way a `-|d|` term invites."""
    w_spill: float = 100.0
    """Cost of spilling the cup's entire contents, spread over the steps it spills on.
    The knob the task balances on, and a **guess**: no policy has been trained on this.
    The module docstring says what it is a guess against."""
    w_tilt: float = 1.0
    """Weight on `1 - up_z`, the cup's z axis against the world's. A cost and not a
    bonus here, unlike the hover task's upright term: a cup that is level is the whole
    precondition for not spilling, and there is nothing to farm by standing still
    because the progress term pays nothing for it."""
    w_action: float = 0.01
    """Weight on `mean(action^2)`. Zero action is "hold still", which is the trim."""
    crash_penalty: float = 10.0
    """Subtracted on the step that terminates, and nothing else is scored on it."""
    empty_fraction: float = 0.5
    """Spilled fraction at which the episode is over: the cup is empty enough that
    what is left cannot teach anything about carrying it."""
    drop_distance: float = 0.06
    """Metres of tracking error `|setpoint - cup|` that count as dropping it. What
    force saturation looks like from outside the controller."""
    min_upright: float = 0.3
    """Smallest `up_z` that is still a cup and not a spilled one. 0.3 is a 73 degree
    tilt, well past anything a carry needs."""

    # -- the observation ----------------------------------------------------------------

    wrench_filter: float = 0.2
    """One-pole EMA coefficient on the fluid wrench before it reaches the observation.
    0.2 is a ten-sample time constant, which is the averaging the measured noise
    (12.9% per step, 1.29% over ten) was characterised at. 1.0 is unfiltered and is
    there so the difference can be measured."""

    # -- derived ------------------------------------------------------------------------

    scene_pad: float = 0.06
    """Metres of SPH domain beyond the workspace and the cup, on every side. It is where
    spilled water goes, and `sph.clamp_to_domain` reflects at its walls, so it has to be
    big enough that a reflection is not immediately back inside the cup."""

    settle_steps: int = 100
    """Coupling steps the water is settled for, once, before the first episode.

    A cold `fill_box` lattice is **not** a still cup: every particle starts at the rest
    density with no hydrostatic gradient, and the first pressure solve pays the whole
    column's compression back at once. Measured on this cup, that first step applies
    19.3 N to the floor -- 770% of the water's 2.51 N weight -- and throws the surface
    upward at 0.88 m/s, so an unsettled episode spills most of the cup while the arm
    holds perfectly still. Over 120 steps the load relaxes to 100.2% of the weight and
    the velocity RMS falls to 0.10 m/s; 100 is where that is essentially done.

    Paid once at construction and never again: `_Cup.place` restores the settled
    positions, so a reset costs two array writes. It is the SPH analogue of
    `FluidWorld.initialise()`, which `SphWorld` does not have -- see `_build_fluid`.
    """

    def __post_init__(self) -> None:
        if self.wall_thickness is None:
            self.wall_thickness = 2.0 * self.spacing
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be at least 1, got {self.num_envs}")
        if self.decimation < 1:
            raise ValueError(f"decimation must be at least 1, got {self.decimation}")
        if self.fill_depth > self.cup_depth:
            raise ValueError(
                f"fill_depth {self.fill_depth} m is deeper than the cup's "
                f"{self.cup_depth} m: it would start outside the cup and read as spilled"
            )
        if not 0.0 < self.wrench_filter <= 1.0:
            raise ValueError(
                f"wrench_filter is an EMA coefficient in (0, 1], got {self.wrench_filter}"
            )
        full = (self.cup_mass + self.water_mass) * G
        if self.max_force <= full:
            raise ValueError(
                f"max_force {self.max_force} N cannot hold a {self.cup_mass} kg cup with "
                f"{self.water_mass:.3f} kg of water in it ({full:.2f} N). Raise it, or "
                f"lower cup_mass or fill_depth"
            )
        self._check_cup()

    @property
    def spacing(self) -> float:
        """Particle spacing in metres, `2 * particle_radius`. `docs/sph-contract.md`."""
        return 2.0 * float(self.particle_radius)

    @property
    def water_mass(self) -> float:
        """Kilograms of water the cup starts with, from the filled volume.

        The **nominal** figure, from `(2a)^2 * fill`, which is what the baseline payload
        weighs and what the refusals above are checked against. The number the spill is
        graded against is the one the sampler actually produced,
        `_Cup.water_mass`, and it differs by the half-spacing the lattice rounds by.
        """
        from .units import fluid as resolve_fluid

        rho = resolve_fluid(self.fluid).rho
        return rho * (2.0 * self.cup_half_width) ** 2 * self.fill_depth

    def _check_cup(self) -> None:
        """Refuse a cup whose dimensions do not land on the particle lattice.

        `SphWorld._lattice` puts candidate sites at `origin + (i + 1/2) * spacing`, and
        `_sample` keeps the sites inside a shape. A wall whose thickness is not a whole
        number of spacings therefore samples to one layer in some columns and two in
        others, and an uneven boundary shell is the one defect that looks like a solver
        problem: the fluid presses through the thin places, the measured wall load comes
        out low, and nothing in the output says the wall was the cause. Checked here,
        where it is four comparisons and an error message naming the knob.
        """
        spacing = self.spacing
        for name, value in (("wall_thickness", self.wall_thickness),
                            ("cup_half_width", self.cup_half_width),
                            ("cup_depth", self.cup_depth),
                            ("fill_depth", self.fill_depth)):
            cells = value / spacing
            if abs(cells - round(cells)) > 1e-6:
                raise ValueError(
                    f"{name} is {value * 1e3:g} mm, which is {cells:.3f} particle "
                    f"spacings of {spacing * 1e3:g} mm. Every cup dimension must be a "
                    f"whole number of spacings or the walls sample unevenly. Change "
                    f"{name}, or particle_radius"
                )
        if round(self.wall_thickness / spacing) < 2:
            raise ValueError(
                f"wall_thickness {self.wall_thickness * 1e3:g} mm is "
                f"{self.wall_thickness / spacing:.0f} particle spacing(s) of "
                f"{spacing * 1e3:g} mm, and a one-layer wall does not hold this fluid "
                f"in: measured, the water goes through the floor and the cup feels 75% "
                f"of its weight. Leave wall_thickness at None for two spacings, or "
                f"lower particle_radius"
            )


class _Cup:
    """One environment's cup: a Newton body, a solver, and the water in it.

    What `InteractiveScene` would have been for one clone of the robot, and the sibling
    of `isaac.py::_Rig`. It holds the two things that have to be decided once from
    `fluid_on` and then done the same way every step: where the payload's mass is (in
    the Newton body when the fluid is off, in the particles when it is on) and whether
    there is a coupling to advance.
    """

    def __init__(self, cfg: CarryFluidEnvCfg, snapshot: np.ndarray | None = None
                 ) -> None:
        """`snapshot` is a settled water configuration in the cup's own frame.

        Every cup in a `CarryFluidEnv` has identical geometry, so settling is done once
        and handed to the others: `cfg.settle_steps` coupling steps is the single
        largest cost in constructing an environment with many of them, and the result is
        the same array every time.
        """
        import warp as wp
        import newton

        self.cfg = cfg
        half = np.asarray(cfg.workspace, dtype=np.float64)
        pad = float(cfg.scene_pad)
        outer = float(cfg.cup_half_width + cfg.wall_thickness)

        # The domain holds the whole workspace the cup's centre may visit, plus the cup
        # around that centre, plus the pad. It is sized in whole spacings so the cup's
        # start pose lands on the particle lattice -- see `_check_cup`.
        extent = half + outer + pad
        span = np.ceil(2.0 * extent / cfg.spacing) * cfg.spacing

        from .world import Domain

        # The cup's reset position: the middle of the domain in x and y, and low enough
        # in z that the rim and the pad above it are still inside. Rounded onto the
        # particle lattice so the walls and `fill_box` share the grid `_check_cup` is
        # about -- the cup's dimensions being whole spacings buys nothing if its origin
        # sits half a spacing off.
        self.start = np.round((span / 2.0) / cfg.spacing) * cfg.spacing
        self.start[2] = round((span[2] / 2.0 - cfg.cup_depth / 2.0) / cfg.spacing) \
            * cfg.spacing
        self.domain = Domain(origin=(0.0, 0.0, 0.0), size=tuple(span), faces="wall")
        self.low = self.start - half
        self.high = self.start + half

        inertia = self._inertia()
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        self.body = builder.add_body(
            xform=wp.transform(wp.vec3(*self.start), wp.quat_identity()),
            mass=float(cfg.cup_mass) + (0.0 if cfg.fluid_on else cfg.water_mass),
            inertia=wp.mat33(inertia[0], 0.0, 0.0, 0.0, inertia[1], 0.0,
                             0.0, 0.0, inertia[2]),
            label="cup",
        )
        # `density=0`: `add_shape_box` otherwise ADDS the box's own mass to the body's,
        # and `isaac.py` records what that costs -- a 30 g quadrotor flying at 161 g and
        # a free-fall acceleration of 1.28 m/s^2 where 0 was expected. The mass and the
        # inertia are both stated above, so no shape may contribute either.
        shape = newton.ModelBuilder.ShapeConfig(density=0.0)
        shape.has_shape_collision = False
        for position, scale in self._boxes():
            builder.add_shape_box(self.body, hx=scale[0], hy=scale[1], hz=scale[2],
                                  xform=wp.transform(wp.vec3(*position),
                                                     wp.quat_identity()),
                                  cfg=shape, label="cup")
        self.model = builder.finalize(device=cfg.device)
        self.model.set_gravity((0.0, 0.0, -G))
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, disable_contacts=True, solver="newton",
            integrator="implicitfast", iterations=10)
        self.control = self.model.control()
        self._joint_f = np.zeros(6, dtype=np.float32)

        self.world = None
        self.coupling = None
        self._rest_local = np.zeros((0, 3))
        self.water_mass = float(cfg.water_mass)
        self.particle_mass = 0.0
        if cfg.fluid_on:
            self._build_fluid(snapshot)
            self.state_0 = self.coupling.state_0
            self.state_1 = self.coupling.state_1
        else:
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()

    # -- geometry -----------------------------------------------------------------------

    def _boxes(self) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
        """The cup's five boxes as `(body-local centre, half-extents)`, in metres.

        The body's origin is the **inner floor's centre**, which is also where the water
        starts, so one number -- the start position -- places the cup, the water and the
        goal. A floor slab and four walls; the walls overlap the floor's footprint at
        the corners and `SphWorld._commit` drops the duplicate sites (`report()` counts
        them), so the overlap costs nothing but is why the corner columns are two layers
        and not four.
        """
        cfg = self.cfg
        a, t, d = cfg.cup_half_width, cfg.wall_thickness, cfg.cup_depth
        return [
            ((0.0, 0.0, -t / 2.0), (a + t, a + t, t / 2.0)),
            ((-(a + t / 2.0), 0.0, d / 2.0), (t / 2.0, a + t, d / 2.0)),
            ((+(a + t / 2.0), 0.0, d / 2.0), (t / 2.0, a + t, d / 2.0)),
            ((0.0, -(a + t / 2.0), d / 2.0), (a + t, t / 2.0, d / 2.0)),
            ((0.0, +(a + t / 2.0), d / 2.0), (a + t, t / 2.0, d / 2.0)),
        ]

    def _inertia(self) -> tuple[float, float, float]:
        """The cup's principal inertias, derived from its mass and its outer box.

        A solid box of the cup's outer dimensions, `m (b^2 + c^2) / 12`. **Derived and
        stated, not quoted**: a real mug is a thin shell and its inertia about a
        horizontal axis is larger than a solid block's for the same mass, by roughly a
        factor of 1.5. The idealisation is here, it moves with `cup_mass` and
        `cup_half_width`, and the factor it is wrong by is written down -- which is the
        discipline `isaac.py` applies to the quadrotor's inertia for the same reason.

        It matters because it is the only thing setting the angular gains, and a wrong
        inertia there is a controller that rings or that will not hold the cup level.
        """
        cfg = self.cfg
        mass = float(cfg.cup_mass)
        width = 2.0 * (cfg.cup_half_width + cfg.wall_thickness)
        height = cfg.cup_depth + cfg.wall_thickness
        return (mass * (width * width + height * height) / 12.0,
                mass * (width * width + height * height) / 12.0,
                mass * (width * width + width * width) / 12.0)

    def _build_fluid(self, snapshot: np.ndarray | None = None) -> None:
        """The `SphWorld`, the cup's boundary particles, the water, and the coupling.

        The order is not a convention: `SphWorld.fill_box` drops candidate sites that
        fall inside any shape **added so far**, so the cup has to exist before the water
        does. Filling first leaves water inside the walls, which shows up as a six-figure
        density error on the first step.

        The fluid world's body index has to equal Newton's, because `NewtonCoupling`
        resizes the world's body table to Newton's count and writes it from the model
        thereafter -- a hull bound to the wrong index is driven by nothing and sits at
        the identity transform. `sph_cases.py::_float_rig` checks the same thing for the
        same reason and `PROGRESS_LIVE.md` records what it cost there.
        """
        from . import sph
        from .couple import NewtonCoupling

        cfg = self.cfg
        self.world = sph.SphWorld(
            domain=self.domain, particle_radius=float(cfg.particle_radius),
            fluid=cfg.fluid, gravity=(0.0, 0.0, -G), dt=float(cfg.dt),
            device=cfg.device,
        )
        owned = self.world.add_body(position=tuple(self.start),
                                    orientation=(0.0, 0.0, 0.0, 1.0), label="cup")
        if owned != self.body:
            raise RuntimeError(
                f"the fluid world put this cup at body {owned} and Newton put it at "
                f"{self.body}; NewtonCoupling drives the world's table by Newton's index"
            )
        for position, scale in self._boxes():
            self.world.add_shape("box", body=owned, position=position,
                                 orientation=(0.0, 0.0, 0.0, 1.0), scale=scale)
        a, fill = cfg.cup_half_width, cfg.fill_depth
        self.world.fill_box((self.start[0] - a, self.start[1] - a, self.start[2]),
                            (self.start[0] + a, self.start[1] + a, self.start[2] + fill))
        self.world._commit()
        self.particle_mass = float(self.world.mass)
        self.water_mass = self.particle_mass * int(self.world.n_fluid)
        if self.world.n_fluid == 0:
            raise RuntimeError(
                "the cup sampled to no water particles: fill_depth is under one "
                "particle spacing, or the cup is outside the domain"
            )

        self.coupling = NewtonCoupling(
            model=self.model, solver=self.solver, world=self.world,
            state_0=None, state_1=None, control=self.control, contacts=None,
            dt=float(cfg.dt), buoyancy=False,
        )
        """`buoyancy=False`, which `docs/sph-contract.md` requires for this backend: an
        SPH free surface has a real waterline and real hydrostatic pressure, so the load
        the water puts on the cup comes out of the pressure solve and adding `rho V g`
        on top would count it twice."""

        # The reset snapshot, in the cup's own frame, **after settling**: a cold
        # `fill_box` lattice is not a still cup and `cfg.settle_steps` says what it costs
        # to pretend otherwise. Body-local, which is what lets a reset put the cup at a
        # different position and take its water with it.
        #
        # This reaches into `world.x` and `world.v`, which `docs/sph-contract.md` does
        # not promise. The call it wants is an `SphWorld.refill()` -- the counterpart of
        # `FluidWorld.initialise()`, which this backend has no equivalent of -- and
        # `sph.py` is not this task's to change, so the request is written here concretely
        # rather than worked around silently.
        if snapshot is None:
            for _ in range(int(cfg.settle_steps)):
                self.world.step()
            snapshot = self.world.positions() - self.start
        self._rest_local = np.asarray(snapshot, dtype=np.float64).copy()
        # No `place` here: the water is in the state it just settled to and the body has
        # not moved, so the two already agree. `CarryFluidEnv.reset` places them both.

    # -- what the env reads and writes --------------------------------------------------

    @property
    def state(self) -> Any:
        return self.coupling.state if self.coupling is not None else self.state_0

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        """`(position, quaternion xyzw)` of the cup, in SI, world frame."""
        row = self.state.body_q.numpy()[self.body]
        return (np.asarray(row[:3], dtype=np.float64),
                np.asarray(row[3:], dtype=np.float64))

    def velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """`(linear, angular)` of the cup, m/s and rad/s. Newton's `body_qd` is linear
        first, which `isaac.py` checked by dropping a body and reading which three
        components moved."""
        row = self.state.body_qd.numpy()[self.body]
        return (np.asarray(row[:3], dtype=np.float64),
                np.asarray(row[3:], dtype=np.float64))

    def fluid_wrench(self) -> tuple[np.ndarray, np.ndarray]:
        """The mean fluid `(force, torque)` on the cup over the last coupling step, SI.

        Zero with the fluid off, and zero is the truthful answer there: a rigid payload
        exerts no wrench of its own, it is part of the body's mass.
        """
        if self.world is None:
            return np.zeros(3), np.zeros(3)
        force, torque = self.world.wrench(self.body)
        return np.asarray(force, dtype=np.float64), np.asarray(torque, dtype=np.float64)

    def apply(self, force: np.ndarray, torque: np.ndarray) -> None:
        """The controller's wrench into the body, through the free joint's `joint_f`.

        `control.joint_f` and not `state.body_f`, for the reason `isaac.py` records:
        `NewtonCoupling.step()` calls `State.clear_forces()` between the fluid and the
        solver, so a wrench written to `body_f` out here is deleted before the solver
        sees it. `joint_f` is the same channel in the world frame -- linear first, then
        angular -- and it survives.
        """
        self._joint_f[:3] = force
        self._joint_f[3:] = torque
        self.control.joint_f.assign(self._joint_f)

    def advance(self) -> None:
        """One physics step: the water and Newton together, or Newton alone."""
        if self.coupling is not None:
            self.coupling.step()
            self.state_0, self.state_1 = self.coupling.state_0, self.coupling.state_1
            return
        self.state_0.clear_forces()
        self.solver.step(self.state_0, self.state_1, self.control, None,
                         float(self.cfg.dt))
        self.state_0, self.state_1 = self.state_1, self.state_0

    def place(self, position: np.ndarray) -> None:
        """Put the cup upright and at rest at a position, with the water still in it.

        **`joint_q` and `joint_qd` are what actually move it; `body_q` alone does
        nothing.** `SolverMuJoCo` works in generalised coordinates: its `step` pushes
        `joint_q`/`joint_qd` into MuJoCo, integrates, and *pulls* `body_q` back out. A
        reset written to `body_q` reads back correctly and is overwritten on the next
        step by a cup still doing whatever it was doing -- measured elsewhere in this
        repository as a body still spinning at 25.6 rad/s one step after being placed at
        rest, and as a held hull whose engine z drifted 0.060 to 0.0055 m while the
        pinned array read 0.060000. So the write goes through `couple.place_body`, which
        is the one implementation of it.

        The water goes back to the snapshot taken at construction, mapped through the new
        position, with zero velocity -- the particles' own state is the one thing that
        would otherwise leak a whole episode's sloshing into the next one.
        """
        from .couple import place_body

        pose = np.concatenate([position, [0.0, 0.0, 0.0, 1.0]])
        for state in (self.state_0, self.state_1):
            # `couple.place_body` and not a hand-written `joint_q` write: it finds the
            # free joint's own `joint_q_start` rather than assuming slot 0, and it runs
            # the `eval_fk` that makes `body_q` agree before the next step pulls it --
            # which is what `NewtonCoupling._push_poses` and `pose()` read. One shared
            # implementation, because the drifting-hull failure its docstring records
            # came from four callers each doing this by hand.
            place_body(self.model, state, self.body, pose)
            if state.body_f is not None:
                state.body_f.zero_()
        self._joint_f[:] = 0.0
        self.control.joint_f.assign(self._joint_f)
        if self.world is None:
            return
        self.world.set_body_state(self.body, position=tuple(position),
                                  orientation=(0.0, 0.0, 0.0, 1.0),
                                  velocity=(0.0,) * 6)
        x = self.world.x.numpy()
        x[:self.world.n_fluid] = (self._rest_local + position).astype(x.dtype)
        self.world.x.assign(x)
        v = self.world.v.numpy()
        v[:self.world.n_fluid] = 0.0
        self.world.v.assign(v)
        self.world._wrench.zero_()
        if self.coupling is not None:
            self.coupling._push_poses()

    def outside(self, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Which fluid particles are out of the cup, as a boolean over `positions()`.

        The test is in the cup's own frame, against the cavity the five boxes enclose
        with half a spacing of slack on each side: inside the walls in x and y, above
        the floor, and below the rim. A particle sitting on the rim is not yet spilled
        and a particle a spacing above it is, which is the only place this is a judgement
        call -- the slack is half a spacing, so the call is made at the resolution the
        simulation has rather than at a tighter one it does not.

        All-false with the fluid off: a rigid payload cannot leave.
        """
        if self.world is None:
            return np.zeros(0, dtype=bool)
        cfg = self.cfg
        from .world import _quat_matrix

        rotation = _quat_matrix(orientation)
        local = (self.world.positions() - position) @ rotation
        slack = cfg.spacing / 2.0
        a = cfg.cup_half_width + slack
        return ~((np.abs(local[:, 0]) <= a) & (np.abs(local[:, 1]) <= a)
                 & (local[:, 2] >= -slack) & (local[:, 2] <= cfg.cup_depth + slack))


class CarryFluidEnv:
    """Carry a cup of water to a goal without spilling it, water simulated live.

    The interface is `isaaclab.envs.direct_rl_env.DirectRLEnv`'s, method for method; the
    module docstring says why this is not a subclass of it and what that gives up.

    `num_envs > 1` works and is a Python loop over independent `_Cup`s, for the reason
    `HoverFluidEnv` gives: one `SphWorld` is one particle set with one body table, and
    sharing it between cups would make them share water. Wall clock is linear in
    `num_envs` and model construction is too, because `SolverMuJoCo` compiles its Warp
    kernels per model.
    """

    is_vector_env: ClassVar[bool] = True
    """What `DirectRLEnv` declares, and what every gymnasium >= 1.0 wrapper branches
    on: `step` takes a batch of actions and returns a batch of everything."""

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "autoreset_mode": AUTORESET_SAME_STEP,
    }
    """`DirectRLEnv.metadata`, with the autoreset mode as a string at class level and as
    the real `gymnasium.vector.AutoresetMode` member on the instance -- see
    `isaac.AUTORESET_SAME_STEP` for why it cannot be the member here."""

    def __init__(self, cfg: CarryFluidEnvCfg, render_mode: str | None = None,
                 **kwargs: Any) -> None:
        """`DirectRLEnv.__init__`'s signature, including the `**kwargs` it ignores."""
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode {render_mode!r} is not one of "
                f"{self.metadata['render_modes']}"
            )
        self.torch = _torch()
        self.cfg = cfg
        self.render_mode = render_mode
        self.num_envs = int(cfg.num_envs)
        self.device = str(cfg.device)
        self.num_actions = int(cfg.action_space)
        self.num_observations = int(cfg.observation_space)
        self.physics_dt = float(cfg.dt)
        self.step_dt = float(cfg.dt) * int(cfg.decimation)
        self.max_episode_length = max(1, int(round(cfg.episode_length_s / self.step_dt)))
        self.common_step_counter = 0
        self._rng = np.random.default_rng()

        try:
            import gymnasium as gym

            self.metadata = dict(self.metadata)
            self.metadata["autoreset_mode"] = gym.vector.AutoresetMode.SAME_STEP
            box = gym.spaces.Box(-np.inf, np.inf, (self.num_observations,), np.float32)
            self.single_observation_space = gym.spaces.Dict({"policy": box})
            self.single_action_space = gym.spaces.Box(-1.0, 1.0, (self.num_actions,),
                                                      np.float32)
            self.observation_space = gym.vector.utils.batch_space(
                self.single_observation_space, self.num_envs)
            self.action_space = gym.vector.utils.batch_space(
                self.single_action_space, self.num_envs)
        except Exception:
            # No gymnasium: keep the config's integers, which `DirectRLEnvCfg` also
            # accepts. Nothing in this file reads the spaces.
            self.single_observation_space = self.observation_space = cfg.observation_space
            self.single_action_space = self.action_space = cfg.action_space

        del kwargs  # `DirectRLEnv` takes and ignores them too
        # The first cup settles its water; the rest are handed the result. Identical
        # geometry, so it is the same array, and `cfg.settle_steps` coupling steps is
        # otherwise the largest single cost of building a many-environment run.
        first = _Cup(cfg)
        self.cups = [first] + [_Cup(cfg, snapshot=first._rest_local)
                               for _ in range(self.num_envs - 1)]

        # The controller's gains, from one bandwidth. Critically damped on the cup's own
        # mass and inertia: the water is the load, not part of the plant.
        omega = 2.0 * math.pi * float(cfg.control_hz)
        inertia = np.asarray(self.cups[0]._inertia())
        self.kp = float(cfg.cup_mass) * omega * omega
        self.kd = 2.0 * float(cfg.cup_mass) * omega
        self.kp_ang = inertia * omega * omega
        self.kd_ang = 2.0 * inertia * omega

        torch = self.torch
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long,
                                              device=self.device)
        self._actions = torch.zeros((self.num_envs, self.num_actions),
                                    dtype=torch.float32, device=self.device)
        self._command = np.zeros((self.num_envs, 6))
        self._setpoint = np.zeros((self.num_envs, 3))
        self._setpoint_q = np.zeros((self.num_envs, 4))
        self._goal = np.zeros((self.num_envs, 3))
        self._distance = np.zeros(self.num_envs)
        self._escaped = [np.zeros(0, dtype=bool) for _ in range(self.num_envs)]
        self._spilled = np.zeros(self.num_envs)
        """Spilled **kilograms** per environment, cumulative within the episode. The
        quantity the user named; the fraction below is derived from it."""
        self._spilled_step = np.zeros(self.num_envs)
        self._wrench = np.zeros((self.num_envs, 6))
        """The EMA-filtered fluid wrench, which is what the observation carries."""
        self._sampled = np.zeros((self.num_envs, 6))
        self._samples = 0
        """The sum of the raw wrench over the coupling steps of the current env step,
        and how many went into it. See `_apply_action` for why one sample is not
        enough."""
        self._terminated = torch.zeros(self.num_envs, dtype=torch.bool,
                                       device=self.device)
        self._truncated = torch.zeros(self.num_envs, dtype=torch.bool,
                                      device=self.device)
        self._crashed = np.zeros(self.num_envs, dtype=bool)
        self.water_mass = float(self.cups[0].water_mass)
        self.episode_spill: list[float] = []
        """Kilograms spilled in each episode that has ended, for `main()` to report."""

    # -- the gymnasium / Isaac Lab surface ---------------------------------------------

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None
              ) -> tuple[dict[str, Any], dict]:
        """Reset every environment. `DirectRLEnv.reset`'s signature and return shape.

        `options` is accepted and unused, as it is in `DirectRLEnv`: the signature is
        gymnasium's and dropping the argument would stop a wrapper that passes it.
        """
        del options
        if seed is not None:
            self.seed(seed)
            self._rng = np.random.default_rng(seed)
        self._reset_idx(np.arange(self.num_envs))
        self.episode_length_buf.zero_()
        return self._get_observations(), {}

    def step(self, action: Any) -> tuple[dict[str, Any], Any, Any, Any, dict]:
        """One environment step: `decimation` coupled steps, then the five-tuple.

        **SAME_STEP autoreset**: the environments that ended on this step are reset
        *before* the observation is taken, so `obs` is the new episode's state while
        `reward`, `terminated` and `truncated` describe the one that just ended. That is
        what `DirectRLEnv.metadata` declares; returning the terminal observation instead
        is the bug that teaches a value function the wrong bootstrap.
        """
        self._actions = action.to(self.device).clamp_(-1.0, 1.0).reshape(
            self.num_envs, self.num_actions)
        self._pre_physics_step(self._actions)
        for _ in range(self.cfg.decimation):
            self._apply_action()
        self._measure_spill()
        self.episode_length_buf += 1
        self.common_step_counter += 1

        terminated, truncated = self._get_dones()
        reward = self._get_rewards()

        done = (terminated | truncated).nonzero(as_tuple=False).flatten()
        if done.numel():
            indices = done.detach().cpu().numpy()
            self.episode_spill += [float(self._spilled[i]) for i in indices]
            self._reset_idx(indices)
            self.episode_length_buf[done] = 0
        return self._get_observations(), reward, terminated, truncated, {}

    def close(self):
        """Drop the models, the solvers and the particles.

        No annotations, because `DirectRLEnv.close` has none and the conformance test
        compares the signatures. Warp arrays are freed with their Python objects, so
        this drops the references and makes a second call harmless.
        """
        self.cups = []

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """A top-down scatter of the first environment's water, or `None`.

        `rgb_array` returns a `(n, n, 3)` uint8 image: the water in white over the cup's
        footprint in grey, looking down the z axis at the whole domain, built in numpy.
        No matplotlib -- this is called inside a training loop and importing a plotting
        stack there is how a recorded run becomes slower than an unrecorded one. `human`
        and `None` return `None`; there is no window to open without Isaac Sim.

        `recompute` is accepted for signature compatibility; the particle read is the
        expensive part and there is nothing a `recompute=False` could skip.
        """
        del recompute
        if self.render_mode != "rgb_array" or not self.cups:
            return None
        cup = self.cups[0]
        if cup.world is None:
            return None
        size = 128
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        origin = np.asarray(cup.domain.origin)
        extent = np.asarray(cup.domain.size)
        position, _ = cup.pose()
        half = self.cfg.cup_half_width + self.cfg.wall_thickness
        corner = np.clip(((position[:2] - half - origin[:2]) / extent[:2] * size)
                         .astype(int), 0, size - 1)
        far = np.clip(((position[:2] + half - origin[:2]) / extent[:2] * size)
                      .astype(int), 0, size - 1)
        frame[corner[0]:far[0] + 1, corner[1]:far[1] + 1] = 60
        x = cup.world.positions()
        index = np.clip(((x[:, :2] - origin[:2]) / extent[:2] * size).astype(int),
                        0, size - 1)
        frame[index[:, 0], index[:, 1]] = 255
        return np.transpose(frame, (1, 0, 2))[::-1]

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Seed numpy and torch globally and return the seed, as `DirectRLEnv.seed` does.

        A static method: it seeds the process, not the instance, so an instance's own
        episode randomisation is `self._rng` and is seeded by `reset(seed=...)`. Both are
        done, because a caller who seeds one and gets a reproducible run is going to be
        surprised later.
        """
        np.random.seed(seed if seed >= 0 else None)
        try:
            import torch

            torch.manual_seed(seed)
        except Exception:  # pragma: no cover - no torch, no torch seed
            pass
        return seed

    # -- the five hooks a `DirectRLEnv` subclass implements -----------------------------

    def _pre_physics_step(self, actions: Any) -> None:
        """Turn the action into a Cartesian twist command, once per environment step.

        Here rather than in `_apply_action` because that runs `decimation` times and the
        command does not change across them -- which is what `decimation` means.
        """
        command = actions.detach().to("cpu").numpy()
        self._command[:, :3] = command[:, :3] * float(self.cfg.max_speed)
        self._command[:, 3:] = command[:, 3:] * float(self.cfg.max_rate)
        self._sampled[:] = 0.0
        self._samples = 0

    def _apply_action(self) -> None:
        """Integrate the setpoint, run the Cartesian PD, step the water and the body.

        The setpoint moves at the commanded twist and the controller chases it; the cup
        follows only as well as `max_force` and `max_torque` let it, and the gap between
        the two is what `drop_distance` terminates on.

        The orientation error is the **vector part of the quaternion error**, `2 * q_e`
        for a small rotation, which is the standard Cartesian attitude error and is
        singularity-free -- an Euler-angle error has a discontinuity that a tipping cup
        reaches. The sign is forced onto the short way round (`w < 0` flips it), without
        which a cup more than 180 degrees from its setpoint would be driven the long way.
        """
        from .world import _quat_multiply

        cfg = self.cfg
        dt = float(cfg.dt)
        for index, cup in enumerate(self.cups):
            position, orientation = cup.pose()
            linear, angular = cup.velocity()
            command = self._command[index]

            self._setpoint[index] += command[:3] * dt
            rate = command[3:]
            angle = float(np.linalg.norm(rate)) * dt
            if angle > 1e-12:
                axis = rate / np.linalg.norm(rate)
                delta = np.concatenate([axis * math.sin(angle / 2.0),
                                        [math.cos(angle / 2.0)]])
                self._setpoint_q[index] = _quat_multiply(delta,
                                                         self._setpoint_q[index])

            # q_error = q_setpoint * conj(q_cup); its vector part is half the rotation
            # vector for a small error, hence the 2 below.
            conjugate = np.asarray(orientation) * np.array([-1.0, -1.0, -1.0, 1.0])
            error = _quat_multiply(self._setpoint_q[index], conjugate)
            if error[3] < 0.0:
                error = -error
            force = self.kp * (self._setpoint[index] - position) - self.kd * linear
            if cfg.gravity_feedforward:
                force = force + np.array([0.0, 0.0, float(cfg.cup_mass) * G])
            torque = self.kp_ang * (2.0 * error[:3]) - self.kd_ang * angular
            cup.apply(np.clip(force, -cfg.max_force, cfg.max_force),
                      np.clip(torque, -cfg.max_torque, cfg.max_torque))
            cup.advance()
            # The wrench is accumulated over every coupling step in the env step, not
            # read once at the end of them. **Reading once aliases it.** The partitioned
            # coupling rings at exactly one coupling step (see `cup_mass`), so a sample
            # taken at the end of each env step lands on the same phase of that ringing
            # every time: measured at `decimation=2`, the two steps read -4.95 N and
            # -0.02 N, their mean is the water's 2.51 N weight to within 1%, and the
            # end-of-step sample reports -0.02 N forever. A filter cannot recover a
            # signal that was sampled at its own period; the sum here is what makes the
            # EMA in `_measure_spill` a filter rather than a decimator.
            force_in, torque_in = cup.fluid_wrench()
            self._sampled[index] += np.concatenate([force_in, torque_in])
        self._samples += 1

    def _get_observations(self) -> dict[str, Any]:
        """`{"policy": (num_envs, 20)}`, and `"critic"` only if a state space is set.

        Goal **error** rather than goal and position separately, because the goal is a
        per-episode value and a policy that has to learn the subtraction is learning the
        task's bookkeeping. Orientation as the raw `(x, y, z, w)` quaternion rather than
        Euler angles: the gimbal singularity in any Euler convention is a discontinuity a
        tipping cup reaches.

        The fluid wrench is the filtered one, normalised by the water's initial weight
        (and by that times the cup's half-width for the torque), so both are O(1) and a
        half-empty cup reads as half a load.
        """
        cfg = self.cfg
        weight = max(self.water_mass * G, 1e-9)
        arm = max(float(cfg.cup_half_width), 1e-9)
        rows = np.zeros((self.num_envs, self.num_observations))
        for index, cup in enumerate(self.cups):
            position, orientation = cup.pose()
            linear, angular = cup.velocity()
            fraction = self._spilled[index] / max(self.water_mass, 1e-12)
            rows[index] = np.concatenate([
                position - self._goal[index], orientation, linear, angular,
                self._wrench[index, :3] / weight,
                self._wrench[index, 3:] / (weight * arm),
                [fraction],
            ])
        tensor = self.torch.as_tensor(rows, dtype=self.torch.float32,
                                      device=self.device)
        observations: dict[str, Any] = {"policy": tensor}
        if getattr(self.cfg, "state_space", None):
            observations["critic"] = tensor
        return observations

    def _get_rewards(self) -> Any:
        """`(num_envs,)` float. The module docstring carries the terms and the argument.

        Called after `_get_dones`, which is what lets a terminated environment score one
        finite number instead of a shaping term evaluated on a NaN pose.
        """
        from .world import _quat_matrix

        cfg = self.cfg
        reward = np.zeros(self.num_envs)
        actions = self._actions.detach().to("cpu").numpy()
        for index, cup in enumerate(self.cups):
            if self._crashed[index]:
                # One finite number and none of the shaping terms: see the docstring.
                reward[index] = -float(cfg.crash_penalty)
                continue
            position, orientation = cup.pose()
            distance = float(np.linalg.norm(position - self._goal[index]))
            up_z = float(_quat_matrix(orientation)[2, 2])
            fraction = self._spilled_step[index] / max(self.water_mass, 1e-12)
            reward[index] = (
                cfg.w_progress * (self._distance[index] - distance)
                + math.exp(-(distance / cfg.pos_scale) ** 2)
                - cfg.w_spill * fraction
                - cfg.w_tilt * (1.0 - up_z)
                - cfg.w_action * float(np.mean(actions[index] ** 2))
            )
            self._distance[index] = distance
        return self.torch.as_tensor(reward, dtype=self.torch.float32,
                                    device=self.device)

    def _get_dones(self) -> tuple[Any, Any]:
        """`(terminated, truncated)`, both `torch.bool` of shape `(num_envs,)`.

        Terminated is emptied, dropped, tipped, outside the workspace, or holding a
        non-finite number anywhere. Truncated is the episode length, and the two are kept
        apart because a value function has to bootstrap through a truncation and must not
        bootstrap through a termination.

        The non-finite test is the one that is not about carrying: a DFSPH stiffness
        spike can put a particle somewhere non-finite, the wrench goes with it and the
        body follows. Ending the episode is the only honest response from here and it
        keeps the NaN out of the policy. `SphWorld.stable()` is the diagnostic for how
        often it happens and is a whole-array host read, so it is not called per step.
        """
        from .world import _quat_matrix

        torch = self.torch
        cfg = self.cfg
        crashed = np.zeros(self.num_envs, dtype=bool)
        for index, cup in enumerate(self.cups):
            position, orientation = cup.pose()
            linear, angular = cup.velocity()
            if not all(np.all(np.isfinite(value)) for value in
                       (position, orientation, linear, angular,
                        self._wrench[index], self._spilled[index:index + 1])):
                crashed[index] = True
                continue
            emptied = (self._spilled[index]
                       >= float(cfg.empty_fraction) * self.water_mass)
            dropped = bool(np.linalg.norm(self._setpoint[index] - position)
                           > float(cfg.drop_distance))
            tipped = float(_quat_matrix(orientation)[2, 2]) < float(cfg.min_upright)
            outside = bool(np.any(position < cup.low) or np.any(position > cup.high))
            crashed[index] = emptied or dropped or tipped or outside
        self._crashed = crashed
        # `torch.tensor` and not `as_tensor`: the latter SHARES the numpy buffer on CPU,
        # so `_reset_idx` writing `self._crashed` would reach back into the `terminated`
        # tensor this call has already returned.
        self._terminated = torch.tensor(crashed, dtype=torch.bool, device=self.device)
        self._truncated = self.episode_length_buf >= self.max_episode_length
        return self._terminated, self._truncated

    # -- the rest ----------------------------------------------------------------------

    def _measure_spill(self) -> None:
        """Latch which particles have left each cup, and convert the count to kilograms.

        Latched (`|=`) rather than instantaneous: `sph.clamp_to_domain` reflects a
        particle that reaches the domain wall, so a droplet can bounce off the floor of
        the workspace and land back in the cup, and un-counting it would be crediting the
        policy for an artefact of the clamp. Once out, out.

        Also runs the wrench EMA, here rather than in `_get_observations` so it is one
        update per environment step whatever a caller does with the observation.
        """
        alpha = float(self.cfg.wrench_filter)
        mean = self._sampled / max(self._samples, 1)
        for index, cup in enumerate(self.cups):
            self._wrench[index] += alpha * (mean[index] - self._wrench[index])
            if cup.world is None:
                continue
            position, orientation = cup.pose()
            if not np.all(np.isfinite(position)) or not np.all(np.isfinite(orientation)):
                continue
            before = float(self._escaped[index].sum())
            self._escaped[index] |= cup.outside(position, orientation)
            after = float(self._escaped[index].sum())
            self._spilled_step[index] = (after - before) * cup.particle_mass
            self._spilled[index] = after * cup.particle_mass

    def _reset_idx(self, env_ids: Any) -> None:
        """Respawn these environments: cup upright at rest, full, with a fresh goal.

        The start position is jittered inside the inner half of the workspace and the
        goal is `goal_distance` away in a random horizontal direction, clamped back into
        the workspace. Horizontal because a cup of water is carried across a table, and
        because a vertical goal is a different task -- the acceleration that spills a cup
        going up is the one term that does not need the fluid to predict.

        The setpoint is reset **onto the cup**, not left where the last episode's action
        integrated it to: a setpoint 60 mm away at reset is a cup that terminates as
        dropped on its first step, forever, which is the failure `isaac.py` records the
        solver-side version of.
        """
        for index in np.atleast_1d(np.asarray(env_ids)).astype(int):
            cup = self.cups[index]
            span = (cup.high - cup.low) / 2.0
            start = cup.start + self._rng.uniform(-0.25, 0.25, 3) * span
            direction = self._rng.normal(size=2)
            direction /= max(float(np.linalg.norm(direction)), 1e-12)
            goal = start + np.concatenate([direction * float(self.cfg.goal_distance),
                                           [0.0]])
            self._goal[index] = np.clip(goal, cup.low, cup.high)
            cup.place(start)
            self._setpoint[index] = start
            self._setpoint_q[index] = np.array([0.0, 0.0, 0.0, 1.0])
            self._distance[index] = float(np.linalg.norm(start - self._goal[index]))
            self._escaped[index] = np.zeros(
                0 if cup.world is None else cup.world.n_fluid, dtype=bool)
            self._spilled[index] = 0.0
            self._spilled_step[index] = 0.0
            self._wrench[index] = 0.0

    def report(self) -> list[str]:
        """The task as a human reads it, in the shape `SphWorld.report` returns."""
        cfg = self.cfg
        cup = self.cups[0] if self.cups else None
        depth = int(round(cfg.fill_depth / cfg.spacing))
        lines = [
            f"  task:       carry a cup {cfg.goal_distance * 1e3:g} mm to a goal, "
            f"{self.num_envs} env(s), {self.num_observations} obs, "
            f"{self.num_actions} actions",
            f"  cup:        {2 * cfg.cup_half_width * 1e3:g} mm square x "
            f"{cfg.cup_depth * 1e3:g} mm deep, {cfg.wall_thickness * 1e3:g} mm walls, "
            f"{cfg.cup_mass * 1e3:g} g empty",
            f"  water:      {cfg.fill_depth * 1e3:g} mm deep = "
            f"{self.water_mass * 1e3:.1f} g"
            + (f" as {cup.world.n_fluid:,} particles, {depth} through the depth"
               if cup is not None and cup.world is not None
               else " as a rigid payload (fluid off)"),
            f"  controller: Cartesian PD at {cfg.control_hz:g} Hz -- kp {self.kp:.4g} "
            f"N/m, kd {self.kd:.4g} N.s/m, clamped at {cfg.max_force:g} N / "
            f"{cfg.max_torque:g} N.m"
            + (", cup weight fed forward" if cfg.gravity_feedforward else ""),
            f"  episode:    {cfg.episode_length_s:g} s = {self.max_episode_length} steps "
            f"of {self.step_dt * 1e3:g} ms ({cfg.decimation} x {cfg.dt * 1e3:g} ms)",
        ]
        if cup is not None and cup.world is not None:
            lines.append(
                f"  spill:      measured as particles out of the cup x "
                f"{cup.particle_mass * 1e6:.4g} mg each, latched; "
                f"empty at {cfg.empty_fraction:g} of "
                f"{self.water_mass * 1e3:.1f} g"
            )
            lines += cup.world.report()
        else:
            lines.append("  fluid:      off -- a rigid payload, and spill is exactly 0")
        return lines


# -- the two measurements the task is graded on ----------------------------------------
#
# Both are scripted open-loop drives of the same `CarryFluidEnv`, not a policy: the
# numbers have to be reproducible at three particle counts and a policy is the one thing
# in this file that is not.


def lamb_period(cfg: CarryFluidEnvCfg) -> float:
    """First-mode sloshing period of the still cup, seconds, from Lamb's formula.

    `omega^2 = (pi g / L) tanh(pi h / L)` for the lowest antisymmetric mode of a
    rectangular tank of width `L` filled to depth `h` -- Lamb, *Hydrodynamics* (1932),
    art. 257, the standard shallow-to-deep interpolation. Used for two things and
    graded for neither: to pick the shuttle frequency in `shake` (driving a tank at its
    own first mode is what makes sloshing the dominant term rather than a small
    correction), and as the period the measured force oscillation can be read against.

    The cup is square in plan, so `L = 2a` is exact for a mode running along one axis;
    the measured period on this geometry inside Isaac Sim came out 0.3162 s against
    0.306 s predicted for a twelve-gon mug of a similar width
    (`benchmarks/live/results/isaacsim/carry-l40s.json`), which is the accuracy to
    expect of it.

    The depth used is `fill_depth`, the still-water line, and **not** the settled
    surface the coarse free surface actually sits at -- which is up to 40% higher at
    four particles through the depth. That is a real error in the prediction and it is
    the reason this is a drive frequency rather than a graded number.
    """
    length = 2.0 * float(cfg.cup_half_width)
    omega_sq = (math.pi * G / length) * math.tanh(math.pi * cfg.fill_depth / length)
    return 2.0 * math.pi / math.sqrt(omega_sq)


def spill_angle(cfg: CarryFluidEnvCfg) -> float:
    """The tilt at which a quasi-statically tipped cup starts to pour, radians.

    The free surface stays horizontal, so in the cup's own frame it is a plane at the
    tilt angle; the cup is square, so the whole problem is the 2D cross-section
    `x in [-a, a]`, `z in [0, D]` carrying an area `2 a h`, and the volume is that area
    times `2 a`. Tilting about the y axis by `theta` makes `+x` the downhill side, and
    the surface in the cup frame is the line `z = x tan(theta) + k`.

    Two things can happen first as `theta` grows, and which one does depends on how full
    the cup is:

    * the surface reaches the **rim** at `(a, D)` while the floor is still wet all the
      way across -- the area is the trapezoid `2 a k`, so `k = h` and the rim is reached
      at `tan(theta_c) = (D - h) / a`. Needs `h - a tan(theta_c) >= 0`, i.e. `h >= D/2`.
    * the **floor** on the `-x` side goes dry first and the water becomes a triangle of
      base `b` and height `b tan(theta)`, so `b = sqrt(4 a h / tan(theta))`; that
      triangle's deep corner reaches the rim when `b tan(theta) = D`, which gives
      `tan(theta_c) = D^2 / (4 a h)`. This is the branch for `h < D/2`.

    The two agree at `h = D/2`, where both give `D / (2 a)`, which is the check that the
    algebra above is not two unrelated formulas with a seam in the middle.

    At the defaults (a = 40 mm, D = 100 mm, h = 40 mm) it is `0.01 / 0.0064 = 1.5625`,
    so **57.4 degrees**. `retained_fraction` continues past it.
    """
    a = float(cfg.cup_half_width)
    depth, fill = float(cfg.cup_depth), float(cfg.fill_depth)
    tangent = ((depth - fill) / a if fill >= depth / 2.0
               else depth * depth / (4.0 * a * fill))
    return math.atan(tangent)


def rim_angle(cfg: CarryFluidEnvCfg, radius: str = "half_width") -> dict:
    """The other spill-angle formula, `tan(theta) = (H - h) / R`, and whether it holds.

    This is the **trapezoid branch** of `spill_angle` written for a round cup: a vessel
    of inner height `H` filled to depth `h`, tilted until the free surface -- still a
    plane covering the whole floor -- first touches the rim. For a cylinder the volume
    under the plane `z = x tan(theta) + k` is `pi R^2 k` by symmetry, so `k = h`
    exactly, and the plane reaches `(R, H)` at `tan(theta) = (H - h) / R`. For a square
    cup the same symmetry gives `k = h` with `R = a`, so the formula is shape-blind in
    this branch; `(H - h) / a` is the trapezoid branch `spill_angle` already derives.

    **It carries a precondition that this cup fails.** The floor stays wet only while
    the plane's uphill end is still above it, `h - R tan(theta) >= 0`, i.e.
    `tan(theta) <= h / R`. Past that the water pulls away from the uphill wall and
    becomes a wedge, which is deeper at the pouring rim for the same volume, and the
    governing angle is `spill_angle`'s triangle branch instead. The precondition is
    equivalent to `h >= H / 2` at onset, which is why `spill_angle` branches on exactly
    that. At the defaults `h = 40 mm` against `H / 2 = 50 mm`: short by 10 mm, so this
    formula is being read 12 degrees outside its own domain and the number it returns
    is not a prediction of anything.

    `radius` says what to use for `R` on a cup that has no radius, since it is five
    boxes and square in plan (see the module docstring):

    * `"half_width"` -- `R = a`, matching the cross-section the tilt runs through. This
      is the one that makes the two formulas comparable, because it is the same `a` the
      triangle branch uses.
    * `"equal_area"` -- `R = sqrt((2a)^2 / pi)`, the radius of a cylinder with the same
      plan area and therefore the same volume at the same fill depth. Wider than `a` by
      13%, so it predicts a smaller angle.

    Returns the tangent, the angle in radians and degrees, the floor-dry limit, and
    `valid`. **The plan shape is not where the two formulas differ**: a cylinder of
    `R = a`, integrated without assuming a branch, spills at 56.80 degrees against the
    square cup's 57.38 -- 0.58 degrees. The branch is worth 1.07 degrees and the
    `R` convention 3.3. `tests/test_live_isaac.py` pins all of it against quadrature.
    """
    a = float(cfg.cup_half_width)
    height, fill = float(cfg.cup_depth), float(cfg.fill_depth)
    if radius == "half_width":
        r = a
    elif radius == "equal_area":
        r = math.sqrt((2.0 * a) ** 2 / math.pi)
    else:
        raise ValueError(f"radius must be 'half_width' or 'equal_area', got {radius!r}")
    tangent = (height - fill) / r
    return {
        "convention": radius,
        "R": r,
        "tangent": tangent,
        "angle": math.atan(tangent),
        "angle_deg": math.degrees(math.atan(tangent)),
        "floor_dry_tangent": fill / r,
        "floor_dry_deg": math.degrees(math.atan(fill / r)),
        "valid": tangent <= fill / r,
    }


def retained_fraction(cfg: CarryFluidEnvCfg, angle: float) -> float:
    """What fraction of the water a cup tilted quasi-statically to `angle` keeps.

    One above `spill_angle`; below it the surface is pinned at the pouring rim `(a, D)`
    with slope `tan(theta)`, and the water is whatever is under that line:

        the surface meets the floor at x0 = a - D / tan(theta)
        x0 >= -a  (tan(theta) >= D / 2a)   a triangle,   area = D^2 / (2 tan(theta))
        x0 <  -a                           a trapezoid,  area = 2 a D - 2 a^2 tan(theta)

    and the fraction is that over `2 a h`. The two branches agree at
    `tan(theta) = D / 2a`, where both are `a D`. At the defaults the triangle branch is
    the only reachable one, because `tan(theta_c) = 1.5625` is already past
    `D / 2a = 1.25`; the trapezoid branch is for a cup filled past half, and it is here
    because a formula that is right on one set of defaults and silently wrong on another
    is worse than no formula.

    **It is quasi-static and it is inviscid.** It knows nothing about how fast the cup
    was tipped, about the sheet of water that carries over the rim with momentum it
    picked up on the way, or about the film that stays on the wall. So the measured
    transfer should come out **above** this at any real tilt rate, and by how much is
    the number `pour` reports rather than a tolerance it hides inside.
    """
    a = float(cfg.cup_half_width)
    depth, fill = float(cfg.cup_depth), float(cfg.fill_depth)
    tangent = math.tan(angle)
    if tangent <= math.tan(spill_angle(cfg)):
        return 1.0
    area = (depth * depth / (2.0 * tangent) if tangent >= depth / (2.0 * a)
            else 2.0 * a * depth - 2.0 * a * a * tangent)
    return max(0.0, area / (2.0 * a * fill))


def _carry_env(radius: float, device: str, **overrides: Any) -> CarryFluidEnv:
    """A one-environment `CarryFluidEnv` at a particle radius, for a scripted drive.

    `episode_length_s` is long and the three nuisance terminations are off, because
    these two measurements script the arm and a truncation or a reset in the middle of
    one is a measurement of two half runs averaged together. `min_upright` has to go for
    `pour` -- an 80 degree tilt is exactly what it exists to stop -- and `empty_fraction`
    with it, because pouring the cup out is the point rather than a failure. They are
    passed here, in one place, so that no caller can quietly disable a termination the
    carry task is graded with.
    """
    settings: dict[str, Any] = dict(
        particle_radius=radius, device=device, num_envs=1,
        episode_length_s=1e4, min_upright=-1.1, empty_fraction=1e3,
        drop_distance=0.12,
    )
    settings.update(overrides)
    return CarryFluidEnv(CarryFluidEnvCfg(**settings))


def shake(radius: float = 0.005, *, cycles: float = 4.0, device: str = "cpu",
          seed: int = 0, settle: int = 100, amplitude: float = 1.0) -> dict:
    """Shuttle the end effector at the cup's own sloshing frequency, and measure.

    The drive is `action_x = amplitude * sin(2 pi t / T)` with `T` from `lamb_period`,
    for `cycles` periods, then zero command for one more period while the water rings
    down. **At the cup's own first mode on purpose**: a carry at an arbitrary frequency
    measures mostly the controller, and the convergence question is about the water. The
    setpoint amplitude that comes out is `max_speed * amplitude * T / (2 pi)`, 26 mm at
    the defaults, so the cup stays well inside the workspace and nothing terminates.

    Identical at every particle radius -- same seed, same reset, same command sequence,
    same wall clock in simulated seconds -- which is what makes the numbers it returns
    comparable across the three counts `convergence` runs it at.

    What it returns, all in SI:

    * `force_peak`, `force_rms`: the **horizontal** fluid force on the end effector, the
      sloshing load, from the filtered wrench the observation carries. Horizontal
      because the vertical component is dominated by the water's weight, which is not
      sloshing and is known in advance.
    * `force_raw_peak`: the same peak off the unfiltered per-step wrench, so the filter's
      effect is a number here rather than a claim made elsewhere.
    * `weight_fraction`: the mean vertical load over the drive, against the weight of
      the water in the cup. A coupling that has lost its sign or a wall that leaks shows
      up here and nowhere else.
    * `spill`, `spill_fraction`: kilograms out of the cup, latched, and that over the
      water the cup started with.
    """
    torch = _torch()
    env = _carry_env(radius, device, settle_steps=settle)
    try:
        period = lamb_period(env.cfg)
        steps = int(round((cycles + 1.0) * period / env.step_dt))
        drive = int(round(cycles * period / env.step_dt))
        env.reset(seed=seed)
        action = torch.zeros((1, env.num_actions), device=device)
        lateral, vertical, raw_lateral = [], [], []
        started = time.perf_counter()
        for step in range(steps):
            phase = 2.0 * math.pi * step * env.step_dt / period
            action[0, 0] = amplitude * math.sin(phase) if step < drive else 0.0
            env.step(action)
            lateral.append(float(np.linalg.norm(env._wrench[0, :2])))
            vertical.append(float(env._wrench[0, 2]))
            raw = env._sampled[0] / max(env._samples, 1)
            raw_lateral.append(float(np.linalg.norm(raw[:2])))
        elapsed = time.perf_counter() - started
        cup = env.cups[0]
        weight = env.water_mass * G
        return {
            "radius": float(radius),
            "wrench_filter": float(env.cfg.wrench_filter),
            "spacing_mm": env.cfg.spacing * 1e3,
            "fluid_particles": int(cup.world.n_fluid),
            "boundary_particles": int(cup.world.n_boundary),
            "depth_in_particles": int(round(env.cfg.fill_depth / env.cfg.spacing)),
            "water_mass": float(env.water_mass),
            "slosh_period_s": period,
            "env_steps": steps,
            "simulated_s": steps * env.step_dt,
            "force_peak": max(lateral),
            "force_rms": float(np.sqrt(np.mean(np.square(lateral)))),
            "force_raw_peak": max(raw_lateral),
            "force_peak_over_weight": max(lateral) / weight,
            "weight_fraction": -float(np.mean(vertical[:drive])) / weight,
            "spill": float(env._spilled[0]),
            "spill_fraction": float(env._spilled[0]) / max(env.water_mass, 1e-12),
            "density_error": {k: float(v) for k, v in
                              cup.world.density_error().items()},
            "seconds": elapsed,
            "ms_per_coupling_step": elapsed / max(steps * env.cfg.decimation, 1) * 1e3,
        }
    finally:
        env.close()


def convergence(radii: tuple[float, ...] = (0.01, 0.005, 0.0025), *,
                device: str = "cpu", cycles: float = 4.0, seed: int = 0,
                settle: int = 100) -> list[dict]:
    """`shake` at several particle counts, printed as a table. The deliverable.

    Three counts is the floor and the default is three, each a factor of two in radius
    and therefore eight in particle count -- a factor of 64 across the sweep, which is
    enough for a trend to be a trend. A single number at one resolution says nothing
    about an under-resolved free surface, and four particles through the depth is
    under-resolved by any standard.

    **Spill is the quantity expected to converge worst and it is reported anyway.** A
    particle leaves the cup when its centre crosses the rim, so the spill is quantised
    in units of one particle mass, and one particle is 1/32 of the cup at the coarsest
    radius here against 1/2048 at the finest. A spill of "one particle" therefore means
    3% of the cup at the top of the table and 0.05% at the bottom, and no amount of
    averaging fixes that: it is the resolution of the measurement.
    """
    rows = [shake(radius, cycles=cycles, device=device, seed=seed, settle=settle)
            for radius in radii]
    print(f"\n  shuttle at the cup's first mode "
          f"({rows[0]['slosh_period_s'] * 1e3:.0f} ms), {cycles:g} cycles then ring-down"
          f" -- {rows[0]['simulated_s']:.2f} s simulated, {rows[0]['env_steps']} env "
          f"steps, identical at every radius\n")
    print("  radius  spacing  fluid  depth    F_xy peak   F_xy rms   F_z/weight  "
          "    spill        of cup    particles out   1 p = quantum   ms/step")
    for row in rows:
        # Every spill figure carries its quantum, because the spill is a count of
        # escaped particles and nothing finer than one of them exists: two rows that
        # differ by less than a quantum are the same measurement reported twice.
        quantum = 1.0 / max(row["fluid_particles"], 1)
        print(f"  {row['radius'] * 1e3:4.1f} mm  {row['spacing_mm']:4.1f} mm  "
              f"{row['fluid_particles']:5d}  {row['depth_in_particles']:2d} p  "
              f"{row['force_peak']:7.3f} N    {row['force_rms']:6.3f} N    "
              f"{row['weight_fraction'] * 100:6.1f}%    "
              f"{row['spill'] * 1e3:7.2f} g   {row['spill_fraction'] * 100:5.1f}%   "
              f"{row['spill_fraction'] / quantum:9.0f} p      "
              f"{quantum * row['water_mass'] * 1e3:6.3f} g = {quantum * 100:5.2f}%  "
              f"{row['ms_per_coupling_step']:6.1f}")
    raw = ", ".join("%.3f" % row["force_raw_peak"] for row in rows)
    filtered = ", ".join("%.3f" % row["force_peak"] for row in rows)
    print(f"\n  the filter: peak horizontal force {raw} N unfiltered against "
          f"{filtered} N through the EMA (alpha {rows[0]['wrench_filter']:g})")
    return rows


def pour(angles_deg: tuple[float, ...] = (45.0, 60.0, 70.0, 80.0), *,
         radius: float = 0.005, device: str = "cpu", rate: float = 0.4,
         hold_s: float = 3.0, seed: int = 0, settle: int = 100) -> list[dict]:
    """Tilt the cup past the spill angle and weigh what came out. The deliverable.

    One run per target angle, each from a fresh reset of the same environment: the arm
    is commanded to rotate about the world y axis at `rate` rad/s until the **setpoint**
    reaches the target, then to hold for `hold_s` while the pour finishes. The angle
    reported and compared against is the one the **cup** reached, read back from its
    quaternion as `acos(R[2][2])`, not the one that was commanded -- the controller is a
    finite-stiffness PD with the water's moment on it and the two are not the same
    number.

    `retained_fraction` is the prediction and it is quasi-static, so the honest
    expectation is that the measured transfer runs **above** it: the water carries
    momentum over the rim that a static surface does not have. The ratio is printed as
    `measured / predicted` rather than as a pass or a fail, because this is a
    measurement against a closed-form idealisation and not a test with a tolerance.

    `rate` is a real term in the answer and is therefore a parameter: 0.4 rad/s takes
    3.5 s to reach 80 degrees, which is eleven sloshing periods, and a faster tilt
    throws more water out. Slower is more quasi-static and costs proportionally more.
    """
    torch = _torch()
    from .world import _quat_matrix

    env = _carry_env(radius, device, settle_steps=settle)
    rows = []
    try:
        critical = spill_angle(env.cfg)
        print(f"\n  cup {2 * env.cfg.cup_half_width * 1e3:g} mm square x "
              f"{env.cfg.cup_depth * 1e3:g} mm deep, filled "
              f"{env.cfg.fill_depth * 1e3:g} mm = {env.water_mass * 1e3:.1f} g in "
              f"{env.cups[0].world.n_fluid} particles")
        print(f"  spill angle from the geometry: {math.degrees(critical):.1f} deg "
              f"(tan = {math.tan(critical):.4f} = D^2 / 4ah), tilted at {rate:g} rad/s "
              f"then held {hold_s:g} s\n")
        print(f"  one particle = {env.cups[0].particle_mass * 1e3:.3f} g = "
              f"{100.0 / max(env.cups[0].world.n_fluid, 1):.3f}% of the cup, which is "
              f"the quantum every mass below is counted in\n")
        print("  target   reached   transferred  particles   predicted   ratio"
              "    left in cup")
        for target in angles_deg:
            env.reset(seed=seed)
            action = torch.zeros((1, env.num_actions), device=device)
            action[0, 4] = min(1.0, rate / float(env.cfg.max_rate))
            tilt_steps = int(round(math.radians(target) / rate / env.step_dt))
            for _ in range(tilt_steps):
                env.step(action)
            action[0, 4] = 0.0
            for _ in range(int(round(hold_s / env.step_dt))):
                env.step(action)
            _, orientation = env.cups[0].pose()
            reached = math.acos(min(1.0, max(-1.0,
                                             float(_quat_matrix(orientation)[2, 2]))))
            transferred = float(env._spilled[0])
            predicted = (1.0 - retained_fraction(env.cfg, reached)) * env.water_mass
            rows.append({
                "target_deg": float(target),
                "reached_deg": math.degrees(reached),
                "spill_angle_deg": math.degrees(critical),
                "transferred": transferred,
                "transferred_fraction": transferred / max(env.water_mass, 1e-12),
                "predicted": predicted,
                "predicted_fraction": 1.0 - retained_fraction(env.cfg, reached),
                "ratio": transferred / predicted if predicted > 0 else float("nan"),
                "water_mass": float(env.water_mass),
                "fluid_particles": int(env.cups[0].world.n_fluid),
                "particle_mass": float(env.cups[0].particle_mass),
                "quantum_fraction": 1.0 / max(int(env.cups[0].world.n_fluid), 1),
                "transferred_particles": transferred
                                         / max(env.cups[0].particle_mass, 1e-18),
                "tilt_rate": float(rate),
                "hold_s": float(hold_s),
            })
            row = rows[-1]
            print(f"  {target:5.0f}d   {row['reached_deg']:5.1f}d   "
                  f"{transferred * 1e3:7.2f} g   "
                  f"{transferred / max(env.cups[0].particle_mass, 1e-18):6.0f} p  "
                  f"  {predicted * 1e3:7.2f} g   "
                  f"{row['ratio']:6.2f}   "
                  f"{(env.water_mass - transferred) * 1e3:7.2f} g")
        return rows
    finally:
        env.close()


def spill_onset(radius: float = 0.005, *, device: str = "cpu", rate: float = 0.1,
                cap_deg: float = 70.0, seed: int = 0, settle: int = 100,
                stop_fraction: float = 0.05) -> dict:
    """Tilt slowly and record the cup's own angle at which water first leaves. Measured.

    `pour` weighs mass at four fixed angles; this measures the **angle** at a fixed
    event, and the angle is the better conditioned of the two by a long way. Mass out
    of this cup is a count of escaped particles, so it is quantised in units of
    `1 / n_fluid` of the cup -- 3.1% at a 10 mm radius against 0.025% at 2 mm -- and
    that quantum is why the spill masses in `convergence` do not converge. A threshold
    angle is quantised only by how fast the tilt sweeps through it: at `rate` rad/s and
    a 100 Hz environment step the angle grid is `rate / 100` radians, 0.057 degrees at
    the default, and refining the fluid does not coarsen it.

    The drive is a single continuous rotation about the world y axis at `rate` rad/s
    from a settled upright cup, stepped until `stop_fraction` of the water is out or the
    cup reaches `cap_deg`. The angle recorded is the **cup's**, `acos(R[2][2])` off its
    quaternion, not the setpoint's -- at 0.1 rad/s the finite-stiffness PD lags by about
    0.08 degrees, which matters at the 1 degree resolution the two closed forms differ
    at.

    **`rate` is a real term in the answer.** The closed forms are quasi-static and the
    water sloshes, so a faster tilt spills earlier; `onset_convergence` sweeps the rate
    at one radius so the extrapolation to zero is a measurement rather than an
    assumption. 0.1 rad/s is the default here against `pour`'s 0.4 because this is a
    threshold measurement and 0.4 rad/s crosses the whole interesting interval in
    0.3 sloshing periods.

    Two resolution-dependent biases sit on the number and they have opposite signs, so
    the convergence is not expected to be one-sided:

    * the coarse free surface settles **above** the still-water line -- up to 40% of the
      fill depth at four particles through it -- which reaches the rim early;
    * `_Cup.outside` allows half a spacing of slack past the rim before it calls a
      particle spilled, 10 mm at the coarsest radius against 2 mm at the finest, which
      calls it late.

    Both are O(spacing), so the error should fall roughly linearly in the spacing, and
    the reported `onset_deg` is an onset for the *discrete* free surface either way.

    Returns the angle at the first escaped particle, at the second (their difference is
    the quantum expressed in degrees, which is the honest error bar on a one-particle
    event), at `stop_fraction`, and both closed forms with the error against each.
    """
    torch = _torch()
    from .world import _quat_matrix

    env = _carry_env(radius, device, settle_steps=settle)
    try:
        env.reset(seed=seed)
        action = torch.zeros((1, env.num_actions), device=device)
        action[0, 4] = min(1.0, rate / float(env.cfg.max_rate))
        cup = env.cups[0]
        quantum = float(cup.particle_mass)
        water = float(cup.water_mass)
        particles = int(cup.world.n_fluid)
        cap = math.radians(cap_deg)
        angles: list[float] = []
        counts: list[int] = []
        started, ended = time.perf_counter(), ""
        while True:
            _, _, terminated, truncated, _ = env.step(action)
            if bool(terminated.any()) or bool(truncated.any()):
                ended = "the episode ended mid-measurement"
                break
            _, orientation = cup.pose()
            angles.append(math.acos(min(1.0, max(-1.0, float(
                _quat_matrix(orientation)[2, 2])))))
            counts.append(int(round(float(env._spilled[0]) / max(quantum, 1e-18))))
            if angles[-1] >= cap:
                ended = f"reached the {cap_deg:g} degree cap"
                break
            if counts[-1] * quantum >= stop_fraction * water:
                ended = f"{stop_fraction * 100:g}% of the cup is out"
                break
        elapsed = time.perf_counter() - started

        def angle_at(n: int) -> float | None:
            """Cup angle, degrees, at the step the nth particle was first counted out."""
            for angle, count in zip(angles, counts):
                if count >= n:
                    return math.degrees(angle)
            return None

        onset = angle_at(1)
        second = angle_at(2)
        bulk = angle_at(max(1, int(math.ceil(stop_fraction * particles))))
        triangle = math.degrees(spill_angle(env.cfg))
        trapezoid = rim_angle(env.cfg)["angle_deg"]
        return {
            "radius": float(radius),
            "spacing_mm": env.cfg.spacing * 1e3,
            "fluid_particles": particles,
            "depth_in_particles": int(round(env.cfg.fill_depth / env.cfg.spacing)),
            "particle_mass": quantum,
            "quantum_fraction": 1.0 / max(particles, 1),
            "water_mass_sampled": water,
            "water_mass_nominal": float(env.water_mass),
            "rate": float(rate),
            "angle_grid_deg": math.degrees(rate * env.step_dt),
            "onset_deg": onset,
            "second_particle_deg": second,
            "quantum_deg": None if (onset is None or second is None)
                           else second - onset,
            "bulk_deg": bulk,
            "bulk_fraction": float(stop_fraction),
            "max_angle_deg": math.degrees(max(angles)) if angles else 0.0,
            "escaped_particles": counts[-1] if counts else 0,
            "escaped_fraction": (counts[-1] / max(particles, 1)) if counts else 0.0,
            "triangle_deg": triangle,
            "trapezoid_deg": trapezoid,
            "error_vs_triangle": None if onset is None else onset - triangle,
            "error_vs_trapezoid": None if onset is None else onset - trapezoid,
            "env_steps": len(angles),
            "device": device,
            "stopped_because": ended,
            "seconds": elapsed,
            "ms_per_coupling_step": elapsed / max(len(angles) * env.cfg.decimation, 1)
                                    * 1e3,
        }
    finally:
        env.close()


def onset_convergence(radii: tuple[float, ...] = (0.01, 0.005, 0.0025, 0.002), *,
                      device: str = "cpu", rate: float = 0.1,
                      rates: tuple[float, ...] = (), seed: int = 0,
                      settle: int = 100, stop_fraction: float = 0.05) -> dict:
    """`spill_onset` at several particle counts, and optionally at several tilt rates.

    The deliverable for "does the spill angle converge, and to which closed form".
    Prints the quantum beside every mass fraction, because a difference smaller than one
    particle is not a difference: at a 10 mm radius the cup holds 32 particles, so the
    finest spill fraction that exists is 3.1% and any two numbers inside that band are
    the same measurement.
    """
    cfg = CarryFluidEnvCfg(particle_radius=radii[0])
    triangle = math.degrees(spill_angle(cfg))
    half = rim_angle(cfg, "half_width")
    area = rim_angle(cfg, "equal_area")
    print(f"\n  cup {2 * cfg.cup_half_width * 1e3:g} mm square (five boxes, not round) "
          f"x H = {cfg.cup_depth * 1e3:g} mm deep, filled h = {cfg.fill_depth * 1e3:g} mm"
          f" -- h/H = {cfg.fill_depth / cfg.cup_depth:.2f}, so h < H/2")
    print(f"  prediction A  tan = D^2/4ah  = {math.tan(spill_angle(cfg)):.4f}  -> "
          f"{triangle:6.3f} deg   dry-floor wedge, equal volume   VALID (needs tan >= h/a"
          f" = {cfg.fill_depth / cfg.cup_half_width:.2f})")
    for form in (half, area):
        print(f"  prediction B  tan = (H-h)/R  = {form['tangent']:.4f}  -> "
              f"{form['angle_deg']:6.3f} deg   R = {form['R'] * 1e3:.2f} mm "
              f"({form['convention']}), floor wet across   "
              f"{'VALID' if form['valid'] else 'INVALID past'} "
              f"{form['floor_dry_deg']:.2f} deg")
    rows = [spill_onset(radius, device=device, rate=rate, seed=seed, settle=settle,
                        stop_fraction=stop_fraction) for radius in radii]
    print(f"\n  tilted about world y at {rate:g} rad/s from settled, angle read off the "
          f"cup's quaternion (grid {rows[0]['angle_grid_deg']:.3f} deg/step)\n")
    print("  radius  spacing  fluid  depth    1 particle = quantum    onset    err vs A"
          "   err vs B    5% out   ms/step")
    for row in rows:
        quantum = (f"{row['particle_mass'] * 1e3:6.3f} g = "
                   f"{row['quantum_fraction'] * 100:5.2f}%")
        onset = ("  --  " if row["onset_deg"] is None
                 else f"{row['onset_deg']:6.2f}d")
        err_a = ("   --  " if row["error_vs_triangle"] is None
                 else f"{row['error_vs_triangle']:+6.2f}d")
        err_b = ("   --  " if row["error_vs_trapezoid"] is None
                 else f"{row['error_vs_trapezoid']:+6.2f}d")
        bulk = "  --  " if row["bulk_deg"] is None else f"{row['bulk_deg']:6.2f}d"
        print(f"  {row['radius'] * 1e3:4.1f} mm  {row['spacing_mm']:4.1f} mm  "
              f"{row['fluid_particles']:5d}  {row['depth_in_particles']:2d} p   "
              f"{quantum}   {onset}    {err_a}     {err_b}    {bulk}  "
              f"{row['ms_per_coupling_step']:7.1f}")
    print(f"\n  the quantum in degrees -- first to second particle out: "
          + ", ".join("%s" % ("--" if row["quantum_deg"] is None
                              else "%.2f deg at %.1f mm"
                                   % (row["quantum_deg"], row["radius"] * 1e3))
                      for row in rows))
    sweep = [spill_onset(radii[-1] if len(radii) == 1 else 0.005, device=device,
                         rate=r, seed=seed, settle=settle,
                         stop_fraction=stop_fraction) for r in rates]
    if sweep:
        print(f"\n  tilt rate at {sweep[0]['radius'] * 1e3:g} mm -- the closed forms are "
              f"quasi-static, so slower should read later\n")
        print("    rate      onset    err vs A    5% out")
        for row in sweep:
            onset = "  --  " if row["onset_deg"] is None else f"{row['onset_deg']:6.2f}d"
            err = ("   --  " if row["error_vs_triangle"] is None
                   else f"{row['error_vs_triangle']:+6.2f}d")
            bulk = "  --  " if row["bulk_deg"] is None else f"{row['bulk_deg']:6.2f}d"
            print(f"    {row['rate']:4.2f} rad/s  {onset}    {err}    {bulk}")
    return {"radii": rows, "rates": sweep, "triangle_deg": triangle,
            "trapezoid_half_width_deg": half["angle_deg"],
            "trapezoid_equal_area_deg": area["angle_deg"]}


def main(argv: list[str] | None = None) -> int:
    """`--mode rollout` (the default), `convergence` or `pour`.

    `rollout` is a random-policy rollout with and without the water, and what each one
    cost. The point of printing both is the comparison: the same cup, the same
    controller, the same reward and the same terminations, so the difference in spill,
    in mean reward and in step rate is the water's. A random policy spills; that is the
    baseline a trained one has to beat, and it is a number rather than a claim.

    `convergence` and `pour` are the two measurements, documented on their own
    functions. They are here rather than in `benchmarks/` so that the environment and
    the numbers quoted about it cannot drift apart.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", default="rollout",
                        choices=["rollout", "convergence", "pour", "onset"])
    parser.add_argument("--radii", type=float, nargs="+",
                        default=[0.01, 0.005, 0.0025],
                        help="particle radii for --mode convergence or onset")
    parser.add_argument("--rate", type=float, default=0.1,
                        help="tilt rate in rad/s for --mode onset")
    parser.add_argument("--rates", type=float, nargs="*", default=[],
                        help="extra tilt rates to sweep at 5 mm for --mode onset")
    parser.add_argument("--angles", type=float, nargs="+",
                        default=[45.0, 60.0, 70.0, 80.0],
                        help="tilt targets in degrees for --mode pour")
    parser.add_argument("--settle", type=int, default=100,
                        help="coupling steps the water is settled for")
    parser.add_argument("--steps", type=int, default=120,
                        help="environment steps per rollout")
    parser.add_argument("--radius", type=float, default=0.005,
                        help="SPH particle radius in metres (the cost knob)")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-fluid", action="store_true",
                        help="run only the rigid-payload baseline")
    parser.add_argument("--hold", action="store_true",
                        help="zero action instead of random: the cup should not spill")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.mode == "convergence":
        convergence(tuple(args.radii), device=args.device, seed=args.seed,
                    settle=args.settle)
        return 0
    if args.mode == "pour":
        pour(tuple(args.angles), radius=args.radius, device=args.device,
             seed=args.seed, settle=args.settle)
        return 0
    if args.mode == "onset":
        onset_convergence(tuple(args.radii), device=args.device, rate=args.rate,
                          rates=tuple(args.rates), seed=args.seed,
                          settle=args.settle)
        return 0

    torch = _torch()
    modes = [False] if args.no_fluid else [False, True]
    for fluid_on in modes:
        cfg = CarryFluidEnvCfg(particle_radius=args.radius, num_envs=args.num_envs,
                               device=args.device, fluid_on=fluid_on)
        env = CarryFluidEnv(cfg)
        print(f"\n--- water {'on' if fluid_on else 'off'} "
              f"-----------------------------------------")
        print("\n".join(env.report()))
        observations, _ = env.reset(seed=args.seed)
        assert set(observations) >= {"policy"}
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        total, episodes, started = 0.0, 0, time.perf_counter()
        peak = 0.0
        for _ in range(args.steps):
            if args.hold:
                action = torch.zeros((env.num_envs, env.num_actions))
            else:
                action = (torch.rand((env.num_envs, env.num_actions),
                                     generator=generator) * 2.0 - 1.0)
            _, reward, terminated, truncated, _ = env.step(action.to(args.device))
            total += float(reward.sum())
            episodes += int((terminated | truncated).sum())
            peak = max(peak, float(env._spilled.max()))
        elapsed = time.perf_counter() - started
        steps = args.steps * env.num_envs
        ended = env.episode_spill
        print(f"  rollout:    {args.steps} steps x {env.num_envs} env(s), "
              f"{episodes} episode end(s), "
              f"{'zero' if args.hold else 'random'} policy")
        print(f"  spill:      {peak * 1e3:.2f} g worst in flight, "
              f"{env.water_mass * 1e3:.1f} g carried "
              f"({peak / max(env.water_mass, 1e-12) * 100:.1f}% of the cup)")
        if ended:
            print(f"              {np.mean(ended) * 1e3:.2f} g a finished episode "
                  f"over {len(ended)}, worst {max(ended) * 1e3:.2f} g")
        print(f"  reward:     {total / steps:+.4f} a step, {total:+.2f} total")
        print(f"  rate:       {steps / elapsed:.1f} env steps/s, {elapsed:.1f} s wall, "
              f"{env.step_dt * args.steps / elapsed:.3f}x real time")
        env.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - the demo in the docs
    raise SystemExit(main())
