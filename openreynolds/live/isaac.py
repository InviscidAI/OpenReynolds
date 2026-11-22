"""A quadrotor hover task with the fluid on, in Isaac Lab's `DirectRLEnv` interface.

    python -m openreynolds.live.isaac --steps 250

    from openreynolds.live.isaac import HoverFluidEnv, HoverFluidEnvCfg

    env = HoverFluidEnv(HoverFluidEnvCfg(fluid_on=True))
    obs, _ = env.reset(seed=0)
    for _ in range(250):
        obs, reward, terminated, truncated, extras = env.step(policy(obs["policy"]))

## What this is, and the one thing it is not

**It is not a `DirectRLEnv` subclass, and it cannot be.** `isaaclab.envs.DirectRLEnvCfg`
has five fields with no default and one of them is `scene: InteractiveSceneCfg`, and an
`InteractiveScene` cannot be constructed without Isaac Sim's USD stage: it clones
prims, resolves USD paths and asks a `SimulationContext` for a physics view. Isaac Sim
is a multi-gigabyte Omniverse install, it does not run on this machine, and no amount of
wrapping makes it optional -- `DirectRLEnv.__init__` reaches the stage before it reaches
anything this file could substitute. A class that inherited `DirectRLEnv` here would
import, would fail on construction, and would be a claim this repository cannot stand
behind. `benchmarks/README.md` is written against exactly that kind of claim.

**What it is instead: an environment that conforms to that interface.** Every method
signature below is the one `isaaclab.envs.direct_rl_env.DirectRLEnv` declares, checked
against the real package rather than remembered (`tests/test_live_isaac.py`, and the
conformance test there compares the two with `inspect.signature` whenever `isaaclab` is
importable). `step` returns the gymnasium >= 1.0 five-tuple
`(obs, reward, terminated, truncated, extras)` with **SAME_STEP autoreset**, which is
what `DirectRLEnv.metadata` declares and what makes the difference below worth stating
twice. `is_vector_env` is True, observations are a dict with a `"policy"` key, and every
tensor is a `torch.Tensor` on `self.device` with `terminated` and `truncated` as
`torch.bool` of shape `(num_envs,)`. So it drops into any gymnasium >= 1.0 training
loop, and into Isaac Lab's own RL wrappers (`isaaclab_rl.rsl_rl`, `.skrl`, `.sb3`),
which read `unwrapped.num_envs`, `unwrapped.device`, `is_vector_env` and the five-tuple
and never touch the scene.

The physics underneath is Newton plus `FluidWorld` directly, through
`couple.NewtonCoupling`: a rigid body in `newton.solvers.SolverMuJoCo`, a lattice
Boltzmann fluid in Warp, and four actuator disks.

**What that costs, said plainly.** No USD scene, so nothing here can be viewed in
Omniverse, articulated from a URDF through `InteractiveScene`, or sensed with Isaac
Sim's cameras, contact sensors or IMUs. No `InteractiveSceneCfg` cloning, so there is no
single batched physics view across environments and therefore no multi-GPU scale-out
(see `num_envs` below for what is there instead). No `cfg.events`, no
`observation_noise_model`, no `action_noise_model`, no XR, no video recorders: those are
`DirectRLEnvCfg` fields this file does not read, and a `HoverFluidEnvCfg` does not carry
them rather than carrying them and ignoring them.

**How it would be adapted with Isaac Sim present.** Make `HoverFluidEnv` a real
`DirectRLEnv` subclass; move the airframe and the four rotors into an
`ArticulationCfg` under an `InteractiveSceneCfg` and let the scene clone it `num_envs`
times; delete `_Rig` and read poses from `self._robot.data.root_state_w` instead of
`newton.State.body_q`; keep `_pre_physics_step`, `_apply_action`, `_get_observations`,
`_get_rewards` and `_get_dones` as they are written here, because those five are the
subclass hooks `DirectRLEnv` calls and their bodies do not depend on where the pose came
from. The fluid half is unchanged: `FluidWorld` reads `newton.Model`'s own shape arrays,
and Isaac Lab's Newton-backed scene has the same arrays.

## The vehicle, and every number in it

A Crazyflie-class quadrotor, built from primitives rather than from
`newton/examples/assets/crazyflie.usd`. That file was read: it carries no
`RigidBodyAPI`, no mass and no joints, so `newton.ModelBuilder.add_usd` yields 0 bodies
and 12 visual-only meshes, and voxelising 12 triangle meshes per coupling step would
cost more than the lattice it was voxelised onto. So the airframe is one box and the
rotors are four `FluidWorld.add_rotor` actuator disks, which is also the only shape the
fluid can resolve at a resolution a laptop can afford.

Mass, arm and rotor radius are the Crazyflie 2.x's (30 g, 46 mm, 23 mm). The inertia is
**derived** from them -- four point masses of `mass/4` at the arm tips, giving
`Ixx = Iyy = m a^2 / 2` and `Izz = m a^2` -- rather than quoted, because the measured
Crazyflie values (about 1.4e-5 and 2.2e-5 kg m^2) are a factor of two below that and a
constant nobody in this repository measured is the failure
`toolbox/references/sphere_drag.py` exists to prevent. The idealisation is stated, the
factor of two is stated, and `cfg.mass` and `cfg.arm` move both together.

The shapes are added with `density=0`, which is not tidiness: `ModelBuilder.add_shape_*`
defaults to a real density and **adds** the shape's mass to the body's. The default
airframe box at the default density is 131 g, so a quadrotor asked for 30 g flies at
161 g, and the symptom is a vehicle that will not leave the ground at any thrust the
rotors can produce -- measured as a free-fall acceleration of 1.28 m/s^2 against an
expected 0 when the weight was supposedly being cancelled exactly. Passing an explicit
`inertia` is the same trap one field over: with `density=0` no shape contributes any,
and `add_body`'s fallback is an isotropic 1e-6 kg m^2, which is thirty times too small
and spins the vehicle at hundreds of rad/s on the first differential-thrust command.

`max_thrust` defaults to `mass * g / 2` per rotor, so four rotors at half throttle hover
exactly. That makes `action = 0` the hover trim, which is why the action cost below can
be `mean(a^2)` -- a cost on control effort away from trim, rather than a tax on flying
at all, which is what `mean(thrust^2)` would be.

## The rotor wrench: where it enters, and the one asymmetry

`FluidWorld.set_rotor_thrust(i, T)` is both halves of Newton's third law in one call:
the fluid gets `-T * axis` as a momentum source over the disk
(`lattice.disks_to_force`) and the body is owed `+T * axis`, which
`couple.NewtonCoupling._push_extras` adds to `state.body_f` through
`FluidWorld.rotor_reaction`.

`rotor_reaction` returns a **force and no torque** -- it sums `thrust * axis` over the
rotors on a body and writes it at the centre of mass. That is correct for what it is,
and it means differential thrust across the four arms produces no attitude moment
through that path, which would leave this vehicle with one controllable degree of
freedom. The moment is therefore applied here, as `control.joint_f` on the free joint
`ModelBuilder.add_body` already created (do **not** also call `add_joint_free`: the body
would get two joints and the solver would fight itself). `joint_f` was checked against
`state.body_f` on the same model and the two agree to the last digit, so it is the same
channel in the world frame, linear first then angular -- and it is the one that survives
`State.clear_forces()`, which `NewtonCoupling.step()` calls between the fluid and the
solver and which is why the wrench cannot simply be added to `body_f` from out here.

So: the moment goes in through `joint_f` always, and the thrust **force** goes in
through `joint_f` only when the fluid is off. With the fluid on it arrives through
`rotor_reaction`, and adding it here as well would double the thrust. The asymmetry is
one `if` and it is the only place the two paths differ.

## The reward, and what it is shaped to do

Per environment, per step, with the weights in `HoverFluidEnvCfg`:

    + exp(-(|p - target| / pos_scale)^2)      1 at the target, falls off over 5 cm
    + w_upright * max(0, up_z)                up_z is the body z axis in the world
    - w_velocity * |v|                        m/s
    - w_spin * |omega|                        rad/s
    - w_action * mean(a^2)                    a in [-1, 1], 0 is the hover trim
    - crash_penalty                           on the step that terminates

A terminating step scores **exactly** `-crash_penalty` and none of the shaping terms.
That is not tidiness: one of the ways to terminate is a non-finite state, and the
shaping terms evaluated on a NaN pose return NaN, which goes into the replay buffer,
into the advantage estimate and into the gradient, and a training run poisoned that way
fails hours later with nothing in the logs pointing back here. A crash is worth one
number and that number is finite.

A Gaussian position term rather than `-|p - target|`, because the linear term's gradient
is constant and the policy learns to arrive fast and overshoot; the Gaussian's gradient
vanishes at the target, which is where hovering is supposed to be the easy thing to do.
The upright term is a bonus rather than a tilt penalty so that it cannot be farmed by
sitting still in the wrong place.

Termination is a crash (inverted, `up_z < 0`) or leaving the lattice box, and the second
is the honest reason: outside the box there is no fluid to compute, so the episode is
over whether or not the vehicle is in trouble. A training run that wants the vehicle to
fly away instead passes `FluidWorld(follow=body)` and moves the window with it;
`world.py::_maybe_shift` is that, and it is deliberately not on here, because a shifted
window does not conserve momentum and this task's whole point is the comparison below.
Truncation is `episode_length_s`.

## Resolution, cost, and what a real run wants

The default is 16 cells across a 0.3 m box, so `dx = 18.75 mm` and the 23 mm rotor is
1.2 cells in radius. That resolves **nothing** about the rotor; what it does resolve is
the momentum the disks put into the box and the download that momentum puts back on the
airframe, which is the term that makes a quadrotor in ground effect or in its own wake
behave differently from a point mass. It is chosen so `main()` finishes in about half a
minute on a CPU: measured on Warp's CPU backend at 4,096 nodes and 19 substeps a
coupling step, `main()` runs at 10.3 environment steps a second and a 250-step episode
takes 24 seconds, against 63 steps a second and 4 seconds with the fluid off.

A real training run wants `resolution=96` or more -- 2 mm cells, 11 cells per rotor
radius, the disk thickness above two cells, and the airframe's boundary layer actually
on the lattice. That is 884,736 nodes and, because the substep count rises with the
resolution too, about 100 substeps a coupling step: a GPU's work and not a laptop's.
`DECISIONS_LIVE.md` D7 records that no GPU was available when this was written, so no
rate on one is claimed here.

## `fluid_on`, which is the measurement

`fluid_on=False` builds the same vehicle, the same reward, the same termination and the
same observation, and no lattice at all. It is not a mock: it is the point-mass-plus-
rigid-body task every quadrotor RL paper actually trains, and it runs six times faster.
Having both behind one flag is what makes "the fluid changes the task" a number somebody
can produce rather than a claim, and at the default settings it already is one. The same
250-step random-policy rollout, same seed, same vehicle:

    fluid off    +0.887 reward a step     2 episodes ended     63.2 env steps/s
    fluid on     +0.466 reward a step    10 episodes ended     10.3 env steps/s

Five times as many episodes ended, because the rotors' own wash in a 0.3 m box pushes
the vehicle out of it. Whether that is a *correct* 0.3 m box is the next question and it
is a resolution question, not a coupling one -- the paragraph above is the honest answer.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

__all__ = ["HoverFluidEnvCfg", "HoverFluidEnv", "main"]


AUTORESET_SAME_STEP = "SameStep"
"""The value of `gymnasium.vector.AutoresetMode.SAME_STEP`, as a string.

`metadata` is a class attribute and gymnasium must not be imported at module scope (see
`__init__.py`: importing this package decides whether the agent is offered a tool, and
that question cannot cost a dependency). `AutoresetMode` is a plain `Enum` and not a
`str` subclass, so the string is not equal to the member and cannot pretend to be;
`HoverFluidEnv.__init__` replaces it on the instance with the real member when
gymnasium is importable, which is every environment an Isaac Lab wrapper runs in.
"""

G = 9.81
"""Standard gravity, m/s^2. Newton's own default, and what the hover thrust comes from."""

ROTOR_PEAK_FACTOR = 2.0
"""How far above momentum theory the lattice's own peak velocity runs, measured here.

`Scaling.for_coupling` sizes the lattice substep count from the fastest speed in the
problem, and getting that number too low is not a slow run, it is a run above
`units.MACH_MAX` that is no longer solving the incompressible equations. The obvious
number is momentum theory's induced velocity, `sqrt(T / (2 rho A))`, which at the hover
thrust is 4.29 m/s. **It is wrong by about a factor of two**, because momentum theory
gives the velocity through a *resolved* disk and at the default resolution the disk is
1.2 cells in radius: the thrust lands in a handful of nodes and the local velocity is
much higher than the far-field value.

Measured over a 250-step random-thrust rollout at `resolution=16`, reading the peak
lattice velocity back off the field every ten steps:

    factor 1   10 substeps     max u_lattice 0.0608   Ma 0.105   -- over the ceiling
    factor 2   19 substeps     max u_lattice 0.0336   Ma 0.058
    factor 3   28 substeps     max u_lattice 0.0227   Ma 0.039
    factor 5   46 substeps     max u_lattice 0.0141   Ma 0.024

So 2 is the smallest factor that keeps the run inside `MACH_MAX = 0.1`, and the smallest
is the cheapest -- the substep count, and therefore the whole cost of the run, is linear
in it. A finer lattice resolves the disk and should need less of this; the factor is not
re-measured per resolution, which is the honest limitation of a constant derived from one
measurement.
"""

FACES = {"-x": "slip", "+x": "slip", "-y": "slip", "+y": "slip",
         "-z": "outlet", "+z": "outlet"}
"""The lattice box's six faces: frictionless sides, outflow above and below.

**`faces="outlet"` on all six was tried first and diverges.** A zero-gradient outflow
sets no density, so a box with six of them and a momentum source inside it has no
pressure reference at all: the measured run went non-finite across the entire lattice
within 15 coupling steps, with the wrench the quadrotor is flown by going NaN a step
before anything else did. Air comes in over the rotors and goes out below them, which is
what a hovering vehicle in a column of air actually does, so the outflow belongs on the
two z faces and the sides only have to not add drag. Measured over 250 random-thrust
steps: these faces hold the density at 1.147 kg/m3 against air's 1.204 (a 5% drop, the
mass the rotors are pumping out of the bottom), six outlets go NaN, and periodic sides
with z outflow drop to 0.85, which is a 29% compressibility error and not usable.
"""


def _torch() -> Any:
    """Torch, or an ImportError naming the one command that fixes it.

    Imported here and not at module scope for the same reason Warp is not: `tools.py`
    imports `openreynolds.live` to decide whether to offer the agent a tool, and a
    laptop with no torch must still get `openreynolds --help`. Torch is also not in the
    `[live]` extra -- the fluid does not need it, only this file does.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised on a machine without it
        raise ImportError(
            "the Isaac Lab compatible env needs PyTorch, because its interface is "
            "torch tensors on a device and nothing else can stand in for that:\n"
            "    pip install torch\n"
            f"(the import said: {exc})"
        ) from exc
    return torch


@dataclass
class HoverFluidEnvCfg:
    """Everything `HoverFluidEnv` reads, in SI, with Isaac Lab's five required fields.

    `DirectRLEnvCfg` has 24 fields and five of them have no default (`decimation`,
    `episode_length_s`, `scene`, `observation_space`, `action_space`). All five are
    present here with values rather than with `MISSING`, because this config is
    constructed by hand in a test and in `main()` and a `MISSING` sentinel there buys
    nothing. The other 19 are absent rather than present-and-ignored.
    """

    # -- the five Isaac Lab requires ----------------------------------------------------

    decimation: int = 2
    """Physics steps per environment step. The env's step is `dt * decimation`."""
    episode_length_s: float = 1.0
    """Seconds before truncation. One second is 250 env steps at the defaults, which is
    long enough for a hover to drift out of the box and short enough to demo."""
    scene: Any = None
    """`None`, where Isaac Lab wants an `InteractiveSceneCfg`.

    Not a stub for one. An `InteractiveScene` needs Isaac Sim's USD stage to clone prims
    onto, so there is nothing to put here that would work; the scene's job -- the bodies,
    their shapes and `num_envs` copies of them -- is done by `_Rig` from
    `newton.ModelBuilder` instead. The field exists so that a `DirectRLEnvCfg` and a
    `HoverFluidEnvCfg` can be fed to the same code, and so that the day Isaac Sim is
    present this is the one field that has to change.
    """
    observation_space: int = 13
    """Observation width: position error 3, linear velocity 3, quaternion 4, angular
    velocity 3. An `int` rather than a `gymnasium.spaces.Box`, which `DirectRLEnvCfg`
    also accepts, so that this config can be built without gymnasium installed."""
    action_space: int = 4
    """Four rotor thrust commands in [-1, 1]."""

    # -- the fluid ----------------------------------------------------------------------

    resolution: int = 16
    """Lattice cells along the longest domain axis. The whole cost of the run is here;
    the module docstring says what 16 does and does not resolve, and what a training run
    wants instead."""
    fluid: str = "air"
    """A name in `units.FLUIDS`, or a `units.Fluid`."""
    dt: float = 0.002
    """The physics step in seconds: Newton's step, and the interval one coupling step
    advances the fluid by. `units.Scaling.for_coupling` picks the lattice substep count
    from it and from the rotor's induced velocity."""
    domain_size: float = 0.3
    """Edge of the cubic lattice box in metres, centred on the target. It is the arena:
    leaving it terminates the episode."""
    collision: str = "regularized"
    """`regularized`, `trt` or `bgk`. Regularised is the default for a robot run and
    `DECISIONS_LIVE.md` D3 is why: a quadrotor's own wake is under-resolved at any
    affordable resolution and the ghost modes a TRT collision leaves are what diverge."""
    smagorinsky: float = 0.16
    """The Smagorinsky constant. Non-zero is not optional here -- at `dx = 19 mm` and a
    2 ms step, air's molecular viscosity gives `tau_plus = 0.50001`, so the subgrid
    viscosity is carrying the entire run (`units.STIFF_TAU` says so at length)."""
    fluid_on: bool = True
    """Whether there is a lattice at all. `False` is the same task with no fluid, which
    is the baseline the fluid's effect is measured against."""
    num_envs: int = 1
    """Independent environments. Each gets its own `newton.Model`, its own solver and its
    own `FluidWorld`; they are stepped in a Python loop, so wall-clock is linear in this
    and there is no batching. See `HoverFluidEnv`'s docstring for why, and for what the
    alternative would have cost."""
    device: str = "cpu"
    """The torch device every returned tensor lives on, and the Warp device the lattice
    is allocated on. One string for both, because two would let them disagree and the
    symptom is a silent host copy every step."""

    # -- the vehicle --------------------------------------------------------------------

    mass: float = 0.03
    """Airframe mass in kg. The Crazyflie 2.x's 30 g. Drives the inertia and, through
    `max_thrust`, the hover trim."""
    arm: float = 0.046
    """Rotor offset from the centre of mass in metres, in a plus configuration: rotors at
    `(+a, 0)`, `(0, +a)`, `(-a, 0)`, `(0, -a)`."""
    rotor_radius: float = 0.023
    """Actuator disk radius in metres. Also what sets the induced velocity, which is what
    `Scaling.for_coupling` sizes the lattice substep count from."""
    max_thrust: float | None = None
    """Newtons per rotor at `action = +1`. `None` fills in `mass * g / 2`, so that four
    rotors at `action = 0` hover exactly and the action cost is a cost on effort away
    from trim rather than on flying."""
    target_height: float = 0.5
    """Hover altitude in metres. The lattice box is centred on it, so it is also the
    middle of the arena rather than a target the vehicle has to climb to."""

    # -- the reward ---------------------------------------------------------------------

    pos_scale: float = 0.05
    """Metres at which the Gaussian position reward has fallen to `1/e`. Five
    centimetres: about a third of the box, and about one lattice cell at the resolution a
    real run uses."""
    w_upright: float = 0.5
    """Weight on the body z axis's world z component. A bonus and not a tilt penalty, so
    it cannot be farmed by sitting still in the wrong place."""
    w_velocity: float = 0.05
    """Weight on linear speed, per m/s."""
    w_spin: float = 0.01
    """Weight on angular speed, per rad/s. Small: a quadrotor has to rotate to translate,
    and penalising that hard teaches it not to correct."""
    w_action: float = 0.01
    """Weight on `mean(action^2)`, which is zero at the hover trim."""
    max_spin: float = 20.0
    """Angular speed in rad/s above which the episode is over. Twenty is three
    revolutions a second: a tumble and not a correction, well outside anything a hover
    needs, and inside what the lattice can carry -- the airframe tip at 20 rad/s is
    0.005 in lattice units against the 0.4 ceiling. It is here because without it a
    random policy spins this vehicle to hundreds of rad/s on its tiny inertia and every
    episode is spent tumbling, which is a reward signal with no hover in it."""
    crash_penalty: float = 10.0
    """Subtracted on the step that terminates, inverted or out of the box. Ten is about
    four seconds of perfect hovering, which is the trade it is meant to express."""

    def __post_init__(self) -> None:
        if self.max_thrust is None:
            self.max_thrust = self.mass * G / 2.0
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be at least 1, got {self.num_envs}")
        if self.decimation < 1:
            raise ValueError(f"decimation must be at least 1, got {self.decimation}")


class _Rig:
    """One environment's own vehicle: a Newton model, a solver, and maybe a lattice.

    This is what `InteractiveScene` would have been, for one clone of the robot. It is a
    class rather than a tuple because the two halves of the rotor wrench (the moment,
    which goes in through `joint_f`, and the thrust, which goes in through the
    coupling's `rotor_reaction` when there is a coupling) have to be decided once, from
    `fluid_on`, and then applied the same way every step.
    """

    def __init__(self, cfg: HoverFluidEnvCfg) -> None:
        import warp as wp
        import newton

        from .world import Domain, FluidWorld

        self.cfg = cfg
        self.target = np.array([0.0, 0.0, float(cfg.target_height)])
        arm = float(cfg.arm)
        self.airframe = (arm * 0.7, arm * 0.7, arm * 0.35)
        """Airframe half-extents in metres, derived from `arm` so that one knob moves the
        whole vehicle. At the default 46 mm arms that is a 64 x 64 x 32 mm box, which
        reaches most of the way to the rotors and is 1.7 cells thick at the default
        resolution -- thin, and the module docstring says what that does not resolve."""
        deck = self.airframe[2] + 0.01
        self.rotor_offsets = np.array([
            [arm, 0.0, deck], [0.0, arm, deck],
            [-arm, 0.0, deck], [0.0, -arm, deck],
        ])
        """Rotor positions in the body frame, plus configuration, a centimetre above the
        airframe's top face so the disk is clear of the box it is bolted to -- a disk
        inside the solid has its momentum source deleted by the voxelisation, and the
        symptom is a vehicle that produces a fraction of the thrust it was asked for."""

        # Four point masses of mass/4 at the arm tips. Derived, not quoted; the module
        # docstring says what the measured Crazyflie values are and by how much they
        # differ.
        inertia_xy = cfg.mass * arm * arm / 2.0
        inertia_z = cfg.mass * arm * arm
        builder = newton.ModelBuilder()
        self.body = builder.add_body(
            xform=wp.transform(wp.vec3(*self.target), wp.quat_identity()),
            mass=float(cfg.mass),
            inertia=wp.mat33(inertia_xy, 0.0, 0.0, 0.0, inertia_xy, 0.0,
                             0.0, 0.0, inertia_z),
            label="airframe",
        )
        # density=0: `add_shape_box` otherwise ADDS the box's own mass to the body's, and
        # a 30 g quadrotor becomes a 230 g one that cannot take off. See the docstring.
        builder.add_shape_box(
            self.body, hx=self.airframe[0], hy=self.airframe[1], hz=self.airframe[2],
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0), label="airframe",
        )
        self.model = builder.finalize(device=cfg.device)
        self.solver = newton.solvers.SolverMuJoCo(self.model)
        self.control = self.model.control()
        self._joint_f = np.zeros(6, dtype=np.float32)

        self.world = None
        self.coupling = None
        if cfg.fluid_on:
            half = float(cfg.domain_size) / 2.0
            domain = Domain(
                origin=tuple(self.target - half), size=(cfg.domain_size,) * 3,
                faces=FACES,
            )
            self.world = FluidWorld(
                domain=domain, resolution=int(cfg.resolution), fluid=cfg.fluid,
                dt=float(cfg.dt), peak_velocity=self.induced_velocity(),
                collision=cfg.collision, smagorinsky=float(cfg.smagorinsky),
                device=cfg.device,
            )
            # The airframe is NOT added with `world.add_shape`. `NewtonCoupling` hands
            # the world Newton's own shape arrays (`world._newton_shapes`) and
            # `world._voxelize` then uses those *instead of* the world's own host list,
            # so a hand-added box here would be silently ignored -- and with the fluid
            # off there is no voxelisation to add it to. One source of geometry.
            for offset in self.rotor_offsets:
                self.world.add_rotor(self.body, tuple(offset), (0.0, 0.0, 1.0),
                                     float(cfg.rotor_radius))
            self.world.initialise()
            from .couple import NewtonCoupling

            self.coupling = NewtonCoupling(
                model=self.model, solver=self.solver, world=self.world,
                state_0=None, state_1=None, control=self.control, contacts=None,
                dt=float(cfg.dt),
            )
            self.state_0 = self.coupling.state_0
            self.state_1 = self.coupling.state_1
        else:
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()

    def induced_velocity(self) -> float:
        """Momentum-theory induced velocity at the hover thrust, m/s.

        `sqrt(T / (2 rho A))`, the fastest thing in the problem by an order of magnitude,
        and therefore the speed `Scaling.for_coupling` has to keep under the lattice Mach
        ceiling. Passing the vehicle's own translation speed instead would choose a
        substep count that the rotor wash then violates, and a lattice Boltzmann run
        above `MACH_MAX` is not solving the incompressible equations it is graded on.

        Times `ROTOR_PEAK_FACTOR`, which is where the measurement is: momentum theory
        describes a resolved disk and this one is a cell across.
        """
        from .units import fluid as resolve_fluid

        area = math.pi * self.cfg.rotor_radius ** 2
        rho = resolve_fluid(self.cfg.fluid).rho
        momentum_theory = math.sqrt(self.cfg.mass * G / 4.0 / (2.0 * rho * area))
        return ROTOR_PEAK_FACTOR * momentum_theory

    @property
    def state(self) -> Any:
        return self.coupling.state if self.coupling is not None else self.state_0

    # -- what the env reads and writes --------------------------------------------------

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        """`(position, quaternion xyzw)` of the airframe, in SI, world frame."""
        row = self.state.body_q.numpy()[self.body]
        return (np.asarray(row[:3], dtype=np.float64),
                np.asarray(row[3:], dtype=np.float64))

    def velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """`(linear, angular)` of the airframe, m/s and rad/s. Newton's `body_qd` is
        linear first, which was checked by dropping a body and reading which three
        components changed."""
        row = self.state.body_qd.numpy()[self.body]
        return (np.asarray(row[:3], dtype=np.float64),
                np.asarray(row[3:], dtype=np.float64))

    def apply(self, thrusts: np.ndarray) -> None:
        """Four rotor thrusts in newtons, into the fluid and into the rigid body."""
        from .world import _quat_matrix

        rotation = _quat_matrix(self.pose()[1])
        if self.world is not None:
            # **Both halves arrive through `rotor_reaction`, so neither is added here.**
            #
            # This used to zero the force and still write the moment, on the stated
            # grounds that "`rotor_reaction` does not carry" one. It did not, when this
            # was written: it returned three numbers. It returns six now -- a rotor off
            # the centre of mass applies `r x F` and a quadrotor with no attitude moment
            # has one controllable degree of freedom -- and `couple._push_extras` adds
            # all six into `body_f`.
            #
            # Leaving the moment here was therefore applying it twice, and only with
            # the fluid on. That is the worst possible place for it: `fluid_on` is the
            # one comparison this task exists to make, and a doubled roll authority on
            # one side of it would have read as the fluid doing something.
            for index, thrust in enumerate(thrusts):
                self.world.set_rotor_thrust(index, float(thrust))
            self._joint_f[:] = 0.0
        else:
            # No fluid, so nothing else is applying the rotors: both halves here.
            moment = np.zeros(3)
            for offset, thrust in zip(self.rotor_offsets, thrusts):
                moment += np.cross(offset, np.array([0.0, 0.0, float(thrust)]))
            self._joint_f[3:] = rotation @ moment
            self._joint_f[:3] = rotation @ np.array([0.0, 0.0, float(np.sum(thrusts))])
        self.control.joint_f.assign(self._joint_f)

    def advance(self) -> None:
        """One physics step: the fluid and Newton together, or Newton alone."""
        if self.coupling is not None:
            self.coupling.step()
            self.state_0, self.state_1 = self.coupling.state_0, self.coupling.state_1
            return
        self.state_0.clear_forces()
        self.solver.step(self.state_0, self.state_1, self.control, None,
                         float(self.cfg.dt))
        self.state_0, self.state_1 = self.state_1, self.state_0

    def place(self, position: np.ndarray, orientation: np.ndarray) -> None:
        """Put the vehicle at rest at a pose, and give the fluid a clean lattice.

        **`joint_q` and `joint_qd` are what actually move it, and `body_q` alone does
        nothing.** `SolverMuJoCo` works in generalised coordinates: its `step` pushes
        `state_in.joint_q` and `joint_qd` into MuJoCo's own `mjData`, integrates, and
        *pulls* `body_q` and `body_qd` back out. So `body_q` is an output. Writing a
        reset pose into it reads back correctly -- and is then overwritten on the next
        step by a vehicle still doing whatever it was doing, which was measured here as
        a body still spinning at 25.6 rad/s one step after being placed at rest. The
        symptom in a training run is an environment that terminates on every single step
        after its first crash and a mean reward pinned at `-crash_penalty`, with the
        observation showing a perfectly healthy vehicle, because the observation is the
        reset state that `body_q` faithfully holds and the solver is ignoring.

        Both are written: `joint_q` because it is what the solver reads, and `body_q`
        because `NewtonCoupling._push_poses` and `_Rig.pose` read it before the next step
        has had a chance to pull it.

        The fluid gets `initialise()` rather than a partial reset: the populations in the
        old vehicle's wake are the one piece of state that would leak an episode's
        history into the next one, and it is two kernel launches plus a voxelisation.
        """
        pose = np.concatenate([position, orientation]).astype(np.float32)
        for state in (self.state_0, self.state_1):
            joint = state.joint_q.numpy()
            joint[:7] = pose
            state.joint_q.assign(joint)
            state.joint_qd.zero_()
            body = state.body_q.numpy()
            body[self.body] = pose
            state.body_q.assign(body)
            state.body_qd.zero_()
            if state.body_f is not None:
                state.body_f.zero_()
        self._joint_f[:] = 0.0
        self.control.joint_f.assign(self._joint_f)
        if self.world is not None:
            self.world._poses[self.body] = tuple(position) + tuple(orientation)
            self.world._device_bodies_dirty = True
            self.world.initialise()


class HoverFluidEnv:
    """A quadrotor stabilisation task with a lattice Boltzmann fluid in the loop.

    The interface is `isaaclab.envs.direct_rl_env.DirectRLEnv`'s, method for method;
    the module docstring says why this is not a subclass of it and what that gives up.

    ## `num_envs > 1` works, and it is a Python loop

    Each environment gets its own `newton.Model`, its own `SolverMuJoCo` and its own
    `FluidWorld`, and `step()` loops over them. That is the honest implementation and it
    is also the only one available: a `FluidWorld` is one lattice with one body table,
    and the two ways to make N of them cheap are both out of reach here. Sharing one
    lattice between N vehicles would make them share a wake -- they would collide
    through the fluid, which is a different task and a wrong one. Batching N lattices
    into one set of Warp arrays is a real option and is what a GPU run should do, but it
    is a change to `world.py` and `lattice.py` (every kernel gains a leading
    environment index), not to this file.

    **The cost, stated so nobody measures it by surprise:** wall-clock is linear in
    `num_envs`, and the constant is the fluid's. At the default resolution one
    environment is about 54 ms a physics step on a CPU, so 16 environments is 0.9
    seconds a step and not 54 ms. `num_envs=1` with `fluid_on=False` is the fast path
    and is what the shapes should be developed against; the fluid is what the run is
    for, and it is not free.

    Model construction is also linear and not cheap -- `SolverMuJoCo` compiles its Warp
    kernels per model -- so a large `num_envs` spends real time in `__init__` before the
    first step.
    """

    is_vector_env: ClassVar[bool] = True
    """What `DirectRLEnv` declares, and what every gymnasium >= 1.0 wrapper branches on:
    `step` takes a batch of actions and returns a batch of everything."""

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "autoreset_mode": AUTORESET_SAME_STEP,
    }
    """`DirectRLEnv.metadata`, with the autoreset mode as a string at class level and as
    the real `gymnasium.vector.AutoresetMode` member on the instance -- see
    `AUTORESET_SAME_STEP` for why it cannot be the member here."""

    def __init__(self, cfg: HoverFluidEnvCfg, render_mode: str | None = None,
                 **kwargs: Any) -> None:
        """`DirectRLEnv.__init__`'s signature, including the `**kwargs` it ignores.

        The construction cost is here and not in `reset`: each rig compiles its own
        `SolverMuJoCo` kernels, so a large `num_envs` spends seconds before the first
        step rather than inside the loop, where it would look like a slow environment.
        """
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

        # The real autoreset member, on the instance, where gymnasium may be imported.
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
            # No gymnasium: keep the config's integers, which is what `DirectRLEnvCfg`
            # accepts too. Nothing in this file reads the spaces.
            self.single_observation_space = self.observation_space = cfg.observation_space
            self.single_action_space = self.action_space = cfg.action_space

        del kwargs  # `DirectRLEnv` takes and ignores them too; nothing here needs one
        self.rigs = [_Rig(cfg) for _ in range(self.num_envs)]

        torch = self.torch
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long,
                                              device=self.device)
        self._actions = torch.zeros((self.num_envs, self.num_actions),
                                    dtype=torch.float32, device=self.device)
        self._thrusts = np.zeros((self.num_envs, self.num_actions))
        self._terminated = torch.zeros(self.num_envs, dtype=torch.bool,
                                       device=self.device)
        self._truncated = torch.zeros(self.num_envs, dtype=torch.bool,
                                      device=self.device)
        self._crashed = np.zeros(self.num_envs, dtype=bool)
        self.target = self.rigs[0].target

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
        """One environment step: `decimation` physics steps, then the five-tuple.

        **SAME_STEP autoreset**, which is the one thing about this return that is easy to
        get wrong: the environments that terminated or truncated on this step are reset
        *before* the observation is taken, so `obs` is the state of the **new** episode
        while `reward`, `terminated` and `truncated` describe the one that just ended.
        That is what `DirectRLEnv.metadata` declares and what every gymnasium >= 1.0
        vector wrapper assumes. Returning the terminal observation instead is the bug
        that makes a value function learn the wrong bootstrap and shows up as a policy
        that is slightly worse than it should be, with nothing in the logs about it.

        `extras` is `{}`. `DirectRLEnv` writes exactly one key, `extras["final_obs"]`,
        and only when `cfg.compute_final_obs` -- which this config does not carry -- so
        an empty dict is conformant rather than a shortcut.
        """
        self._actions = action.to(self.device).clamp_(-1.0, 1.0).reshape(
            self.num_envs, self.num_actions)
        self._pre_physics_step(self._actions)
        for _ in range(self.cfg.decimation):
            self._apply_action()
        self.episode_length_buf += 1
        self.common_step_counter += 1

        terminated, truncated = self._get_dones()
        reward = self._get_rewards()

        done = (terminated | truncated).nonzero(as_tuple=False).flatten()
        if done.numel():
            self._reset_idx(done.detach().cpu().numpy())
            self.episode_length_buf[done] = 0
        return self._get_observations(), reward, terminated, truncated, {}

    def close(self):
        """Drop the models, the solvers and the lattices.

        No annotations, because `DirectRLEnv.close` has none and the conformance test
        compares the signatures. There is nothing to shut down -- Warp arrays are freed
        with their Python objects -- so this drops the references and makes a second
        call harmless, which is what gymnasium's wrappers rely on.
        """
        self.rigs = []

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """An image of the first environment's fluid, or `None`.

        `rgb_array` returns a `(n, n, 3)` uint8 image of the velocity magnitude on the
        mid-plane through the vehicle, built in numpy: no matplotlib, because this is
        called inside a training loop and importing a plotting stack there is how a
        recorded run becomes slower than an unrecorded one. `human` and `None` return
        `None`; there is no window to open without Isaac Sim.

        `recompute` is accepted for signature compatibility. The field read is a host
        copy of the whole lattice and is the expensive part of this call, and
        `FluidWorld.fields` already caches it until the next step, so there is nothing a
        `recompute=False` could skip that is worth skipping.
        """
        del recompute
        if self.render_mode != "rgb_array" or not self.rigs:
            return None
        world = self.rigs[0].world
        if world is None:
            return None
        u = world.fields()["u"]
        plane = np.linalg.norm(u[:, :, u.shape[2] // 2, :], axis=-1)
        top = float(np.nanmax(plane)) or 1.0
        grey = (np.nan_to_num(plane / top, nan=0.0) * 255.0).astype(np.uint8)
        return np.repeat(grey.T[::-1, :, None], 3, axis=2)

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Seed numpy and torch globally and return the seed, as `DirectRLEnv.seed` does.

        A static method, which is what makes it the odd one in this class: it seeds the
        process, not the instance, so an instance's own episode randomisation is
        `self._rng` and is seeded by `reset(seed=...)` instead. Both are done, because a
        caller who seeds only one of them and gets a reproducible run is going to be
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
        """Turn the action into four thrusts in newtons, once per environment step.

        Here rather than in `_apply_action` because that runs `decimation` times and the
        action does not change across them -- which is what `decimation` means.
        """
        fraction = (actions.detach().to("cpu").numpy() + 1.0) * 0.5
        self._thrusts = np.clip(fraction, 0.0, 1.0) * float(self.cfg.max_thrust)

    def _apply_action(self) -> None:
        """One physics step on every rig: the rotor wrench in, the pair advanced."""
        for index, rig in enumerate(self.rigs):
            rig.apply(self._thrusts[index])
            rig.advance()

    def _get_observations(self) -> dict[str, Any]:
        """`{"policy": (num_envs, 13)}`, and `"critic"` only if a state space is set.

        Position **error** rather than position, because the target is a config value
        and a policy that has to learn it from a constant offset is learning the config.
        Orientation as the raw `(x, y, z, w)` quaternion rather than as Euler angles: the
        gimbal singularity in any Euler convention is a discontinuity in the observation
        that a quadrotor reaches by rolling 90 degrees, which is well inside the
        envelope this task visits.
        """
        rows = np.zeros((self.num_envs, self.num_observations))
        for index, rig in enumerate(self.rigs):
            position, orientation = rig.pose()
            linear, angular = rig.velocity()
            rows[index] = np.concatenate([position - rig.target, linear, orientation,
                                          angular])
        tensor = self.torch.as_tensor(rows, dtype=self.torch.float32,
                                      device=self.device)
        observations: dict[str, Any] = {"policy": tensor}
        if getattr(self.cfg, "state_space", None):
            observations["critic"] = tensor
        return observations

    def _get_rewards(self) -> Any:
        """`(num_envs,)` float. The module docstring carries the terms and the argument.

        Called after `_get_dones`, which is what lets a crashed environment score one
        finite number instead of a shaping term evaluated on a NaN pose.
        """
        from .world import _quat_matrix

        cfg = self.cfg
        reward = np.zeros(self.num_envs)
        actions = self._actions.detach().to("cpu").numpy()
        for index, rig in enumerate(self.rigs):
            if self._crashed[index]:
                # One finite number, and none of the shaping terms: see the docstring.
                reward[index] = -float(cfg.crash_penalty)
                continue
            position, orientation = rig.pose()
            linear, angular = rig.velocity()
            up_z = _quat_matrix(orientation)[2, 2]
            distance = float(np.linalg.norm(position - rig.target))
            reward[index] = (
                math.exp(-(distance / cfg.pos_scale) ** 2)
                + cfg.w_upright * max(0.0, float(up_z))
                - cfg.w_velocity * float(np.linalg.norm(linear))
                - cfg.w_spin * float(np.linalg.norm(angular))
                - cfg.w_action * float(np.mean(actions[index] ** 2))
            )
        return self.torch.as_tensor(reward, dtype=self.torch.float32,
                                    device=self.device)

    def _get_dones(self) -> tuple[Any, Any]:
        """`(terminated, truncated)`, both `torch.bool` of shape `(num_envs,)`.

        Terminated is inverted (`up_z < 0`), outside the lattice box, spinning past
        `max_spin`, or holding a non-finite number anywhere. Truncated is the episode
        length, and the two are kept apart because a value function has to bootstrap
        through a truncation and must not bootstrap through a termination -- collapsing
        them is the standard way to teach a hovering policy that running out of time is
        fatal.

        The non-finite test is the one that is not about flying. A lattice Boltzmann run
        at `tau_plus = 0.50001` can diverge, and when it does the fluid wrench goes NaN
        a step before anything else and the rigid body follows. Ending the episode is the
        only honest response from here -- the lattice cannot be un-diverged -- and it
        keeps the NaN out of the policy. If it happens often the resolution or the
        substep count is wrong, and `FACES` records the configuration that made it happen
        every time. `FluidWorld.stable()` cannot be used for this: it marks solid nodes
        `nan` and then accepts `nan` as solid, so a diverged lattice reads as stable.
        """
        from .world import _quat_matrix

        torch = self.torch
        half = float(self.cfg.domain_size) / 2.0
        crashed = np.zeros(self.num_envs, dtype=bool)
        for index, rig in enumerate(self.rigs):
            position, orientation = rig.pose()
            linear, angular = rig.velocity()
            if not all(np.all(np.isfinite(value))
                       for value in (position, orientation, linear, angular)):
                crashed[index] = True
                continue
            outside = bool(np.any(np.abs(position - rig.target) > half - self.margin))
            inverted = float(_quat_matrix(orientation)[2, 2]) < 0.0
            spinning = float(np.linalg.norm(angular)) > float(self.cfg.max_spin)
            crashed[index] = outside or inverted or spinning
        self._crashed = crashed
        # `torch.tensor` and not `as_tensor`: the latter SHARES the numpy buffer on CPU,
        # so anything that later writes `self._crashed` would reach back into the
        # `terminated` tensor this call has already returned.
        self._terminated = torch.tensor(crashed, dtype=torch.bool, device=self.device)
        self._truncated = self.episode_length_buf >= self.max_episode_length
        return self._terminated, self._truncated

    margin: float = 0.01
    """Metres of the box kept clear of the termination test. The lattice's outermost node
    sits half a cell inside the wall (`Domain.origin` says why), and a vehicle whose
    centre is exactly on the boundary has its airframe voxelised partly outside the
    lattice, where the signed distance field is not defined."""

    # -- the rest ----------------------------------------------------------------------

    def _reset_idx(self, env_ids: Any) -> None:
        """Respawn these environments: near the target, at rest, slightly tilted.

        A small random tilt and offset rather than an exact hover, because a task whose
        initial state is already the goal has nothing to learn from its first steps and
        a policy trained on it falls over the first time anything disturbs it. The angle
        is up to 0.15 rad about a random axis, which is inside what full differential
        thrust recovers from in about a tenth of a second.
        """
        for index in np.atleast_1d(np.asarray(env_ids)).astype(int):
            rig = self.rigs[index]
            offset = self._rng.uniform(-0.02, 0.02, 3)
            axis = self._rng.normal(size=3)
            axis /= max(float(np.linalg.norm(axis)), 1e-12)
            angle = float(self._rng.uniform(0.0, 0.15))
            orientation = np.concatenate([axis * math.sin(angle / 2.0),
                                          [math.cos(angle / 2.0)]])
            rig.place(rig.target + offset, orientation)

    def report(self) -> list[str]:
        """The task as a human reads it, in the shape `FluidWorld.report` returns."""
        cfg = self.cfg
        lines = [
            f"  task:       hover at {cfg.target_height:g} m, {self.num_envs} env(s), "
            f"{self.num_observations} obs, {self.num_actions} actions",
            f"  vehicle:    {cfg.mass * 1e3:g} g, {cfg.arm * 1e3:g} mm arms, "
            f"{cfg.rotor_radius * 1e3:g} mm rotors, {cfg.max_thrust:.4g} N each "
            f"(hover at action 0)",
            f"  episode:    {cfg.episode_length_s:g} s = {self.max_episode_length} steps "
            f"of {self.step_dt * 1e3:g} ms ({cfg.decimation} x {cfg.dt * 1e3:g} ms)",
        ]
        if self.rigs and self.rigs[0].world is not None:
            lines.append(
                f"  induced:    {self.rigs[0].induced_velocity():.2f} m/s peak "
                f"({ROTOR_PEAK_FACTOR:g} x momentum theory at the hover thrust), which "
                f"is what the substep count is sized from"
            )
            lines += self.rigs[0].world.report()
        else:
            lines.append("  fluid:      off -- this is the rigid-body task alone")
        return lines


def main(argv: list[str] | None = None) -> int:
    """A random-policy rollout, with and without the fluid, and what each one cost.

    The point of printing both is the comparison: the same vehicle, the same reward and
    the same terminations, so the difference in mean reward and in step rate is the
    fluid's. At the default resolution the fluid run is about two hundred times slower
    and finishes in well under a minute on a CPU.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=250,
                        help="environment steps per rollout (default: one episode)")
    parser.add_argument("--resolution", type=int, default=16,
                        help="lattice cells across the box")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-fluid", action="store_true",
                        help="run only the rigid-body baseline")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch = _torch()
    modes = [False] if args.no_fluid else [False, True]
    for fluid_on in modes:
        cfg = HoverFluidEnvCfg(resolution=args.resolution, num_envs=args.num_envs,
                              device=args.device, fluid_on=fluid_on)
        env = HoverFluidEnv(cfg)
        print(f"\n--- fluid {'on' if fluid_on else 'off'} "
              f"-----------------------------------------")
        print("\n".join(env.report()))
        observations, _ = env.reset(seed=args.seed)
        assert set(observations) >= {"policy"}
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        total, episodes, started = 0.0, 0, time.perf_counter()
        for _ in range(args.steps):
            action = (torch.rand((env.num_envs, env.num_actions), generator=generator)
                      * 2.0 - 1.0).to(args.device)
            _, reward, terminated, truncated, _ = env.step(action)
            total += float(reward.sum())
            episodes += int((terminated | truncated).sum())
        elapsed = time.perf_counter() - started
        steps = args.steps * env.num_envs
        print(f"  rollout:    {args.steps} steps x {env.num_envs} env(s), "
              f"{episodes} episode end(s)")
        print(f"  reward:     {total / steps:+.4f} a step, {total:+.2f} total "
              f"(a random policy; hovering perfectly is about +1.5)")
        print(f"  rate:       {steps / elapsed:.1f} env steps/s, {elapsed:.1f} s wall, "
              f"{env.step_dt * args.steps / elapsed:.3f}x real time")
        env.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - the demo in the docs
    raise SystemExit(main())
