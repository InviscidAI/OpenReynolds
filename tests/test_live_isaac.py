"""`openreynolds/live/isaac.py` and `carry.py` still match the interface they claim.

Two environments live behind this file: `isaac.HoverFluidEnv`, a quadrotor hover with a
lattice Boltzmann fluid, and `carry.CarryFluidEnv`, a robot end effector carrying a cup
of water with an SPH free surface in it. They are siblings -- same conformance claim,
same five hooks, same autoreset contract -- so every interface test here is
parametrised over both and a new sibling is one entry in `ENVS`.

Their value is a claim about somebody else's package: that
`step`, `reset`, `close`, `render` and `seed` are
`isaaclab.envs.direct_rl_env.DirectRLEnv`'s, so the env drops into Isaac Lab's RL
wrappers and into any gymnasium >= 1.0 training loop without Isaac Sim installed. A
claim like that has two ways to go stale and they need two different tests:

- **our file drifts.** Somebody adds a keyword to `step`, or drops `options` from
  `reset` because nothing in this repository passes it, and the env stops being a
  drop-in while every test that only calls it still passes. Caught by
  `test_the_signatures_are_the_ones_isaac_lab_declares`, which compares against
  **literal strings** rather than against a live import, so it fails in the main
  environment -- where `isaaclab` is deliberately absent -- rather than skipping.
- **Isaac Lab drifts.** Caught by `test_our_signatures_match_the_real_direct_rl_env`,
  which is the same comparison against the installed `DirectRLEnv`. It skips here and
  is the only test in this file that would notice a change in `isaaclab` 3.1.

`isaaclab` must never be installed into this environment: it pins `warp-lang==1.17.0`
and `newton==1.6.1`, and the live backend needs the Warp 1.19 and Newton 1.8 that are
here. So the second test is run in a throwaway virtual environment by hand and its
result is recorded in the commit message, not in CI. The first test is what CI has.

Everything else in this file is cheap: a config's fields, the class attributes a vector
wrapper branches on, the five hooks, and one short rollout on a lattice small enough to
be a unit test.

The carry task's own tests are at the bottom and are **not** all cheap. Three of them
are physics: that a cup held still does not leak, that the load the cup feels is the
weight of the water in it, and that shaking it spills more than holding it. Those are
the three that would catch a coupling sign error, a leaking boundary or a spill counter
measuring the wrong thing, and no amount of shape checking substitutes for them. They
share one settled environment through a module fixture, because settling the water is
the expensive part and `reset()` is exactly the call that undoes an episode.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

from openreynolds.live import carry, isaac


ROOT = Path(__file__).resolve().parents[1]

HAS_ISAACLAB = importlib.util.find_spec("isaaclab") is not None
HAS_WARP = importlib.util.find_spec("warp") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_NEWTON = importlib.util.find_spec("newton") is not None

needs_isaaclab = pytest.mark.skipif(
    not HAS_ISAACLAB,
    reason="isaaclab is not installed here, and must not be: it pins warp-lang 1.17 "
           "and newton 1.6, which would downgrade the live backend's 1.19 and 1.8",
)
needs_the_stack = pytest.mark.skipif(
    not (HAS_WARP and HAS_TORCH and HAS_NEWTON),
    reason="a rollout needs warp, newton and torch on this machine",
)

ENVS = {"hover": (isaac.HoverFluidEnv, isaac.HoverFluidEnvCfg),
        "carry": (carry.CarryFluidEnv, carry.CarryFluidEnvCfg)}
"""Every environment in this extension that claims to be a `DirectRLEnv`.

Keyed by name so a failure says which one. Adding a third task means adding it here and
nothing else: the conformance tests, the five hooks, the class attributes and the
required-field checks are all parametrised over this.
"""

ISAAC_LAB_SIGNATURES = {
    "__init__": "(self, cfg: '{cfg}', render_mode: 'str | None' = None, "
                "**kwargs: 'Any') -> 'None'",
    "reset": "(self, seed: 'int | None' = None, "
             "options: 'dict[str, Any] | None' = None) "
             "-> 'tuple[dict[str, Any], dict]'",
    "step": "(self, action: 'Any') "
            "-> 'tuple[dict[str, Any], Any, Any, Any, dict]'",
    "close": "(self)",
    "render": "(self, recompute: 'bool' = False) -> 'np.ndarray | None'",
    "seed": "(seed: 'int' = -1) -> 'int'",
}
"""What `inspect.signature` must print for each method, as a literal.

The *shape* is `DirectRLEnv`'s, checked against the installed package on 2026-11-22:
same parameter names, same defaults, same kinds, `close` bare with no annotations at
all, `seed` a `staticmethod` with no `self`. The annotation *text* differs where it has
to -- `cfg` is a `HoverFluidEnvCfg` and not a `DirectRLEnvCfg`, and the torch types are
spelled `Any` because this module must import without torch -- so
`test_our_signatures_match_the_real_direct_rl_env` compares the parts that have to
agree (names, kinds, defaults) and this one pins the whole string so that an accidental
edit to our file fails in CI, where `isaaclab` is not available to compare against.
"""

THE_FIVE_HOOKS = ("_pre_physics_step", "_apply_action", "_get_observations",
                  "_get_rewards", "_get_dones")
"""The methods `DirectRLEnv` declares abstract and calls from its own `step`. A
subclass that omits one raises `TypeError` on construction; this env is not a subclass,
so nothing but a test says they are all here, and `_apply_action` is the one that would
be quietly dropped -- `_pre_physics_step` could do its work and the env would still
run, at one physics step per environment step."""


def tiny(**overrides) -> isaac.HoverFluidEnvCfg:
    """A config small enough for a test: 8 cells, two env steps of episode.

    8 cells across a 0.3 m box resolves nothing at all, which is the point -- these
    tests are about shapes, dtypes and the autoreset contract, and the physics is graded
    in `benchmarks/live/`. It keeps the rollout test under a couple of seconds, which is
    what decides whether it stays in the main suite.
    """
    settings = dict(resolution=8, num_envs=1, episode_length_s=0.008, device="cpu")
    settings.update(overrides)
    return isaac.HoverFluidEnvCfg(**settings)


TINY_CARRY = dict(particle_radius=0.01, wall_thickness=0.04, settle_steps=5,
                  workspace=(0.06, 0.06, 0.04), scene_pad=0.04, num_envs=1,
                  episode_length_s=0.02, device="cpu")
"""The carry task's equivalent of `tiny`, as the kwargs rather than as a config.

A 20 mm particle spacing puts **two** particles through the water's depth and 32 in the
cup, which resolves nothing at all -- the point, exactly as with `tiny`'s 8 cells.
`settle_steps=5` is the other half of the cost: settling is `_Cup`'s dominant expense
and these tests are about shapes, dtypes and the autoreset contract, not about a still
cup. The physics tests at the bottom of this file use the real defaults instead, and
say why.

A dict and not a function because `CarryFluidEnvCfg.__post_init__` refuses dimensions
that do not land on the particle lattice, so a caller that overrides `particle_radius`
has to override the cup with it and wants to see both in one place.
"""


def tiny_carry(**overrides) -> carry.CarryFluidEnvCfg:
    """`TINY_CARRY` as a config, with overrides."""
    settings = dict(TINY_CARRY)
    settings.update(overrides)
    return carry.CarryFluidEnvCfg(**settings)


# -- the config ------------------------------------------------------------------------


@pytest.mark.parametrize("env_name", sorted(ENVS))
def test_the_config_carries_isaac_labs_five_required_fields(env_name):
    """`DirectRLEnvCfg`'s five `MISSING` fields are all present, and all filled in.

    They are the five a `DirectRLEnvCfg` cannot be constructed without: `decimation`,
    `episode_length_s`, `scene`, `observation_space`, `action_space`. `scene` is `None`
    here on purpose (an `InteractiveScene` needs Isaac Sim's USD stage), so the test
    asserts it *exists* rather than that it is truthy -- and that is the one field whose
    absence would be easiest to rationalise away.
    """
    config = ENVS[env_name][1]
    fields = config.__dataclass_fields__
    for name in ("decimation", "episode_length_s", "scene", "observation_space",
                 "action_space"):
        assert name in fields, f"{name} is one of DirectRLEnvCfg's required five"
    cfg = config()
    assert cfg.decimation >= 1
    assert cfg.episode_length_s > 0
    assert int(cfg.observation_space) > 0 and int(cfg.action_space) > 0
    assert cfg.scene is None


@pytest.mark.parametrize("env_name", sorted(ENVS))
def test_every_config_field_is_documented(env_name):
    """A field with no docstring is a knob nobody can use.

    `ARCHITECTURE_LIVE.md`'s whole argument for this extension is written in prose
    beside the code, and a dataclass is the one place a silent addition is invisible:
    the class still works, the report still prints, and the field is a number somebody
    has to guess the meaning of. Checked by parsing, because `__doc__` on a dataclass
    field does not exist.
    """
    import ast

    config = ENVS[env_name][1]
    module = {"hover": "isaac.py", "carry": "carry.py"}[env_name]
    source = (ROOT / "openreynolds" / "live" / module).read_text(encoding="utf-8")
    tree = ast.parse(source)
    body = next(node.body for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name == config.__name__)
    undocumented = []
    for index, node in enumerate(body):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        following = body[index + 1] if index + 1 < len(body) else None
        documented = (isinstance(following, ast.Expr)
                      and isinstance(following.value, ast.Constant)
                      and isinstance(following.value.value, str))
        if not documented:
            undocumented.append(node.target.id)
    assert not undocumented, (
        f"{config.__name__} fields with no docstring: {undocumented}")


def test_max_thrust_defaults_to_the_hover_trim():
    """Four rotors at `action = 0` hover, which is what makes the action cost a cost on
    effort away from trim rather than a tax on flying at all. If this drifts, the reward
    quietly starts paying the policy to fall."""
    cfg = isaac.HoverFluidEnvCfg()
    assert cfg.max_thrust == pytest.approx(cfg.mass * isaac.G / 2.0)
    four_at_half = 4 * cfg.max_thrust * 0.5
    assert four_at_half == pytest.approx(cfg.mass * isaac.G)


def test_a_bad_config_is_refused_at_construction():
    for bad in (dict(num_envs=0), dict(decimation=0)):
        with pytest.raises(ValueError):
            isaac.HoverFluidEnvCfg(**bad)


# -- the interface ---------------------------------------------------------------------


@pytest.mark.parametrize("module", ["isaac.py", "carry.py"])
def test_the_module_does_not_import_torch_gymnasium_or_isaaclab(module):
    """Importing this module costs none of the three, which is the acceptance criterion.

    `openreynolds/tools.py` imports `openreynolds.live` to decide whether to offer the
    agent its `live_sim` tool, so a module-scope `import torch` here would make a plain
    `openreynolds --help` on a laptop pay for a 200 MB wheel -- or fail, which is what
    it did before the lazy discipline in `live/__init__.py` existed. Checked by parsing
    rather than by watching `sys.modules`, because another test in the same session may
    legitimately have imported torch already.
    """
    import ast

    source = (ROOT / "openreynolds" / "live" / module).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"torch", "gymnasium", "isaaclab"}
    offenders = []
    for node in tree.body:  # module scope only: a function-level import is the point
        if isinstance(node, ast.Import):
            offenders += [alias.name for alias in node.names
                          if alias.name.split(".")[0] in forbidden]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in forbidden:
                offenders.append(node.module)
    assert not offenders, f"{module} imports at module scope: {offenders}"


@pytest.mark.parametrize("env_name", sorted(ENVS))
def test_it_is_a_vector_env_with_same_step_autoreset(env_name):
    """The two class attributes every gymnasium >= 1.0 vector wrapper branches on.

    SAME_STEP autoreset is the one that changes what `step` must return: an environment
    that terminates hands back the observation of the **reset** state in the same call.
    `AutoresetMode` is a plain `Enum` and not a `str` subclass, so the class attribute
    carries the member's value as a string and the instance upgrades it -- which is why
    this test accepts either form rather than asserting one.
    """
    env = ENVS[env_name][0]
    assert env.is_vector_env is True
    assert env.metadata["render_modes"] == [None, "human", "rgb_array"]
    mode = env.metadata["autoreset_mode"]
    assert _autoreset_name(mode) == "SAME_STEP", mode


def _autoreset_name(mode) -> str:
    """`SAME_STEP`, from either the gymnasium member or the string standing in for it."""
    if hasattr(mode, "name"):
        return str(mode.name)
    return {"SameStep": "SAME_STEP", "NextStep": "NEXT_STEP",
            "Disabled": "DISABLED"}.get(str(mode), str(mode))


@pytest.mark.parametrize("env_name", sorted(ENVS))
@pytest.mark.parametrize("name", sorted(ISAAC_LAB_SIGNATURES))
def test_the_signatures_are_the_ones_isaac_lab_declares(name, env_name):
    """Each method's `inspect.signature`, against a literal, for every environment.

    A literal and not a live comparison, so that this fails in the environment CI runs
    in, where `isaaclab` is absent and must stay absent. The companion test below is the
    live comparison and skips here. `{cfg}` is filled in per environment because the
    only part of the string that is allowed to differ between siblings is the config
    type `cfg` is annotated with.
    """
    env, cfg = ENVS[env_name]
    want = ISAAC_LAB_SIGNATURES[name].replace("{cfg}", cfg.__name__)
    got = str(inspect.signature(getattr(env, name)))
    assert got == want, f"{env_name}.{name}{got}"


@pytest.mark.parametrize("env_name", sorted(ENVS))
def test_seed_is_a_static_method_as_direct_rl_env_declares_it(env_name):
    """`DirectRLEnv.seed` is a `@staticmethod`, so it seeds the process and not an
    instance.

    Asserted separately from the signature, because `inspect.signature` on a plain
    method and on a static one differ only by `self` and a rename would hide it. What
    this protects is `HoverFluidEnv.seed(0)` on the class, before anything is
    constructed, which is how a training script seeds a run.
    """
    env = ENVS[env_name][0]
    assert isinstance(inspect.getattr_static(env, "seed"), staticmethod)
    assert env.seed(7) == 7


@pytest.mark.parametrize("env_name", sorted(ENVS))
@pytest.mark.parametrize("hook", THE_FIVE_HOOKS)
def test_the_five_direct_rl_env_hooks_exist(hook, env_name):
    assert callable(getattr(ENVS[env_name][0], hook, None)), f"{env_name}.{hook}"


@pytest.mark.parametrize("env_name", sorted(ENVS))
def test_render_modes_are_enforced(env_name):
    """An unknown `render_mode` is refused before anything expensive is built, which is
    the difference between a typo found in a second and one found after a lattice, a
    particle set and a MuJoCo model have been compiled."""
    env, config = ENVS[env_name]
    settings = dict(TINY_CARRY) if env_name == "carry" else dict(resolution=8)
    with pytest.raises(ValueError, match="render_mode"):
        env(config(**settings), render_mode="pretty")


# -- the rollout -----------------------------------------------------------------------


@needs_the_stack
@pytest.mark.parametrize("fluid_on", [False, True])
def test_a_short_rollout_returns_a_conformant_five_tuple(fluid_on):
    """The contract `step` actually has to keep, on both sides of `fluid_on`.

    `(obs, reward, terminated, truncated, extras)` with `obs` a dict carrying `"policy"`
    of shape `(num_envs, 13)`, `reward` float `(num_envs,)`, the two done flags
    `torch.bool` `(num_envs,)`, and `extras` a dict -- all on `self.device`. Both sides
    of the flag, because the whole point of `fluid_on` is that the task is the same one,
    and a change that altered the observation width on only one path would be invisible
    to a test that ran one of them.
    """
    import torch

    env = isaac.HoverFluidEnv(tiny(fluid_on=fluid_on, num_envs=2))
    try:
        observations, extras = env.reset(seed=3)
        assert isinstance(extras, dict)
        assert set(observations) == {"policy"}
        assert observations["policy"].shape == (2, env.num_observations)

        for _ in range(4):
            action = torch.zeros((2, env.num_actions), device=env.device)
            returned = env.step(action)
            assert len(returned) == 5, "the gymnasium >= 1.0 five-tuple"
            observations, reward, terminated, truncated, extras = returned
            assert observations["policy"].shape == (2, env.num_observations)
            assert observations["policy"].dtype == torch.float32
            assert reward.shape == (2,) and reward.dtype.is_floating_point
            assert terminated.shape == (2,) and terminated.dtype == torch.bool
            assert truncated.shape == (2,) and truncated.dtype == torch.bool
            assert isinstance(extras, dict)
            assert torch.isfinite(observations["policy"]).all()
            assert torch.isfinite(reward).all()
    finally:
        env.close()


@needs_the_stack
def test_it_truncates_on_the_episode_length_and_resets_in_the_same_step():
    """SAME_STEP autoreset, measured rather than asserted from the metadata.

    The episode is two env steps long, so step 2 must report `truncated` and hand back
    an observation of a *fresh* episode. The check is that the episode counter is back
    to zero and that the position error is small again, which it would not be if the
    terminal observation were returned instead -- that is the bug SAME_STEP exists to
    name, and it is invisible to every test that only reads shapes.
    """
    import torch

    env = isaac.HoverFluidEnv(tiny(fluid_on=False, episode_length_s=0.008))
    try:
        assert env.max_episode_length == 2, env.max_episode_length
        env.reset(seed=1)
        action = torch.zeros((1, env.num_actions), device=env.device)
        _, _, terminated, truncated, _ = env.step(action)
        assert not bool(truncated[0]) and not bool(terminated[0])
        observations, _, terminated, truncated, _ = env.step(action)
        assert bool(truncated[0]), "the second step is the end of a two-step episode"
        assert int(env.episode_length_buf[0]) == 0, "the counter resets in that step"
        error = float(torch.linalg.norm(observations["policy"][0, :3]))
        assert error < 0.05, "the observation is the reset state, not the terminal one"
    finally:
        env.close()


@needs_the_stack
def test_full_throttle_climbs_and_zero_throttle_falls():
    """The sign of the actuator path, which no shape check can see.

    This is the test that would have caught `add_shape_box`'s default density adding
    200 g to a 30 g airframe: with the mass wrong by a factor of eight the vehicle does
    not rise at full throttle, and everything else in this file still passes. Run with
    the fluid off, because the rigid-body path is the one where full throttle has a
    closed-form answer.
    """
    import torch

    def climb(level: float) -> float:
        """The change in altitude over 10 steps, not the altitude.

        `_reset_idx` spawns with a random offset of up to 20 mm, which is larger than
        the 8 mm a full-thrust climb covers in 40 ms, so the absolute height says
        nothing. The difference is what the thrust did.
        """
        env = isaac.HoverFluidEnv(tiny(fluid_on=False, episode_length_s=1.0))
        try:
            observations, _ = env.reset(seed=0)
            start = float(observations["policy"][0, 2])
            action = torch.full((1, env.num_actions), level, device=env.device)
            for _ in range(10):
                observations, *_ = env.step(action)
            return float(observations["policy"][0, 2]) - start
        finally:
            env.close()

    assert climb(1.0) > 0.0, "four rotors at full thrust lift a vehicle that hovers at 0"
    assert climb(-1.0) < 0.0, "no thrust falls"
    assert abs(climb(0.0)) < abs(climb(-1.0)), "action 0 is the hover trim"


@needs_the_stack
def test_differential_thrust_produces_an_attitude_moment():
    """The rotor moment reaches the body, which `FluidWorld.rotor_reaction` cannot do.

    `rotor_reaction` returns a force at the centre of mass and no torque, so if
    `_Rig.apply` stopped writing `control.joint_f` the quadrotor would silently lose
    every degree of freedom but altitude -- and a reward curve is the only place that
    would show, months later. One step of a one-sided thrust split has to spin it.
    """
    import torch

    env = isaac.HoverFluidEnv(tiny(fluid_on=False, episode_length_s=1.0))
    try:
        env.reset(seed=0)
        action = torch.tensor([[1.0, 0.0, -1.0, 0.0]], device=env.device)
        for _ in range(3):
            observations, *_ = env.step(action)
        spin = float(torch.linalg.norm(observations["policy"][0, 10:13]))
        assert spin > 0.1, f"differential thrust produced no rotation ({spin} rad/s)"
    finally:
        env.close()


@needs_the_stack
def test_a_termination_is_reported_and_the_reset_after_it_sticks():
    """Two bugs that both show up as "the environment never ends", for opposite reasons.

    **The flag has to survive the reset.** `step` resets the done environments before it
    takes the observation (SAME_STEP), so if the `terminated` tensor shares memory with
    the numpy array `_reset_idx` clears -- which `torch.as_tensor` does on CPU -- the
    caller is handed `False` on the one step that was `True`. Measured as a 250-step
    rollout with a mean reward of `-crash_penalty` and zero reported episode ends.

    **The reset has to reach the solver.** `SolverMuJoCo` integrates generalised
    coordinates and only *pulls* `body_q` out, so a reset written to `body_q` alone
    reads back correctly and is overwritten on the next step by a vehicle still spinning
    at the rate it crashed with. The vehicle then terminates again immediately, every
    step, forever.

    Driven with one rotor at full thrust and the rest off, which tumbles it past
    `max_spin` in well under a second.
    """
    import torch

    env = isaac.HoverFluidEnv(tiny(fluid_on=False, episode_length_s=10.0))
    try:
        env.reset(seed=0)
        action = torch.tensor([[1.0, -1.0, -1.0, -1.0]], device=env.device)
        for step in range(200):
            _, _, terminated, truncated, _ = env.step(action)
            if bool(terminated[0]):
                break
        else:
            pytest.fail("a one-sided thrust never terminated in 200 steps")
        assert not bool(truncated[0]), "a tumble is a termination, not a truncation"
        assert int(env.episode_length_buf[0]) == 0

        # The step after the reset has to be a healthy vehicle, not the tumbling one.
        observations, _, terminated, _, _ = env.step(
            torch.zeros((1, env.num_actions), device=env.device))
        spin = float(torch.linalg.norm(observations["policy"][0, 10:13]))
        assert spin < 1.0, f"the reset did not reach the solver: {spin} rad/s"
        assert not bool(terminated[0])
    finally:
        env.close()


@needs_the_stack
def test_render_gives_an_image_only_with_a_fluid_and_an_rgb_mode():
    import numpy as np

    off = isaac.HoverFluidEnv(tiny(fluid_on=False), render_mode="rgb_array")
    try:
        assert off.render() is None, "no lattice, no picture"
    finally:
        off.close()
    quiet = isaac.HoverFluidEnv(tiny(fluid_on=True))
    try:
        assert quiet.render() is None, "render_mode None draws nothing"
    finally:
        quiet.close()
    env = isaac.HoverFluidEnv(tiny(fluid_on=True), render_mode="rgb_array")
    try:
        env.reset(seed=0)
        frame = env.render()
        assert isinstance(frame, np.ndarray) and frame.dtype == np.uint8
        assert frame.ndim == 3 and frame.shape[2] == 3
    finally:
        env.close()


# -- the one test that would catch Isaac Lab changing ----------------------------------


@needs_isaaclab
@pytest.mark.parametrize("env_name", sorted(ENVS))
@pytest.mark.parametrize("name", sorted(ISAAC_LAB_SIGNATURES))
def test_our_signatures_match_the_real_direct_rl_env(name, env_name):
    """The live comparison, which skips everywhere `isaaclab` is rightly absent.

    Parameter names, kinds and defaults rather than the whole signature string: the
    annotation text cannot match (our `cfg` is a `HoverFluidEnvCfg`, and the torch types
    are spelled `Any` so that this module imports without torch), and pinning text that
    is allowed to differ is how a conformance test turns into a test of nothing but
    itself. Names, kinds and defaults are exactly what a caller or a wrapper can
    observe.
    """
    from isaaclab.envs.direct_rl_env import DirectRLEnv

    def shape(method):
        return [(parameter.name, parameter.kind, parameter.default)
                for parameter in inspect.signature(method).parameters.values()]

    theirs = shape(getattr(DirectRLEnv, name))
    ours = shape(getattr(ENVS[env_name][0], name))
    assert ours == theirs, f"{env_name}.{name}: ours {ours} vs DirectRLEnv's {theirs}"


@needs_isaaclab
@pytest.mark.parametrize("env_name", sorted(ENVS))
def test_the_class_attributes_match_the_real_direct_rl_env(env_name):
    from isaaclab.envs.direct_rl_env import DirectRLEnv

    env = ENVS[env_name][0]
    assert env.is_vector_env == DirectRLEnv.is_vector_env
    assert env.metadata["render_modes"] == DirectRLEnv.metadata["render_modes"]
    assert (_autoreset_name(env.metadata["autoreset_mode"])
            == _autoreset_name(DirectRLEnv.metadata["autoreset_mode"]))


@needs_isaaclab
@pytest.mark.parametrize("env_name", sorted(ENVS))
def test_our_config_covers_isaac_labs_required_fields(env_name):
    """Isaac Lab's own `MISSING` set, read from the installed package.

    If `isaaclab` 3.1 makes a sixth field required, this is the test that says so --
    `test_the_config_carries_isaac_labs_five_required_fields` only knows the five that
    were required on 2026-11-22.

    The resolved value is what has to be inspected, and the sentinel has to be matched
    by **type**. `isaaclab.utils.configclass` rewrites every `= MISSING` into a
    `default_factory`, and the factory returns a deep *copy* of the sentinel, so
    `field.default is dataclasses.MISSING` is false for all 24 fields and
    `DirectRLEnvCfg()` constructs with no arguments at all -- the five required fields
    are enforced inside `DirectRLEnv.__init__`, not by the dataclass. An `is` comparison
    here reports zero required fields and passes for the wrong reason, which is the one
    outcome a conformance test must not have.
    """
    import dataclasses

    from isaaclab.envs.direct_rl_env_cfg import DirectRLEnvCfg

    required = set()
    for name, field in DirectRLEnvCfg.__dataclass_fields__.items():
        resolved = (field.default if field.default_factory is dataclasses.MISSING
                    else field.default_factory())
        if isinstance(resolved, dataclasses._MISSING_TYPE):
            required.add(name)
    assert required, "no field read as MISSING: configclass's layout has changed"
    ours = set(ENVS[env_name][1].__dataclass_fields__)
    assert required <= ours, (
        f"{env_name}: DirectRLEnvCfg requires {sorted(required - ours)} too")


# -- the carry task's own tests --------------------------------------------------------


def test_the_pour_geometry_agrees_at_the_seam_between_its_two_branches():
    """`carry.spill_angle` and `carry.retained_fraction` are each two formulas.

    The spill angle is `(D - h) / a` for a cup filled past half and `D^2 / (4 a h)` for
    one filled below it, and the retained fraction past that angle is a triangle in one
    regime and a trapezoid in the other. Four closed-form branches with two seams
    between them is exactly the shape of thing that is right on the defaults and
    silently wrong one knob away, and it costs nothing to check -- no warp, no newton,
    no particles, just the algebra in the docstrings.

    The seams: both spill-angle formulas must give `D / 2a` at `h = D / 2`, and both
    retained-area formulas must give `a D` at `tan(theta) = D / 2a`. Plus the two things
    that are true of any cup: nothing pours below the spill angle, and more tilt never
    retains more.
    """
    import math

    half = carry.CarryFluidEnvCfg(cup_depth=0.10, fill_depth=0.05)
    assert math.isclose(math.tan(carry.spill_angle(half)),
                        half.cup_depth / (2.0 * half.cup_half_width), rel_tol=1e-9), (
        "the two spill-angle branches disagree at h = D/2, where they are the same cup")

    # The retained-fraction seam, approached from both sides of tan = D / 2a. A cup
    # filled to 0.09 m of 0.10 reaches its spill angle below the seam, so both branches
    # are reachable on the same cup and the seam is a real crossing rather than a
    # formula nothing evaluates.
    deep = carry.CarryFluidEnvCfg(cup_depth=0.10, fill_depth=0.09)
    seam = math.atan(deep.cup_depth / (2.0 * deep.cup_half_width))
    assert math.isclose(carry.retained_fraction(deep, seam - 1e-6),
                        carry.retained_fraction(deep, seam + 1e-6), rel_tol=1e-4), (
        "the triangle and the trapezoid disagree at the seam between them")

    for cfg in (half, deep, carry.CarryFluidEnvCfg()):
        critical = carry.spill_angle(cfg)
        assert carry.retained_fraction(cfg, critical * 0.99) == 1.0, "poured too early"
        assert math.isclose(carry.retained_fraction(cfg, critical), 1.0, abs_tol=1e-6)
        kept = [carry.retained_fraction(cfg, math.radians(d)) for d in range(1, 89)]
        assert all(b <= a + 1e-12 for a, b in zip(kept, kept[1:])), (
            "tilting further retained more water")
    assert carry.retained_fraction(carry.CarryFluidEnvCfg(), math.radians(80.0)) < 0.3, (
        "a cup at 80 degrees keeps almost nothing, and this one claims it does")


def test_the_spill_angle_is_the_one_quadrature_agrees_with_and_the_rim_form_is_not():
    """`carry.spill_angle` against a numeric volume integral, and `carry.rim_angle` not.

    Two closed forms get quoted for a tipping cup and they are different formulas:

        tan(theta) = D^2 / (4 a h)    `spill_angle`, the dry-floor wedge
        tan(theta) = (H - h) / R      `rim_angle`, the floor-wet plane

    The second is the trapezoid branch and it carries a precondition -- the floor has to
    still be wet, `tan(theta) <= h / R` -- that the default cup fails at 45 degrees,
    twelve degrees below the angle the formula itself returns. So this is not a tie to
    be broken by measurement: one of them is being evaluated outside its own domain, and
    the test that says which is a volume integral that assumes **neither** branch.

    The integral: the free surface is the plane `z = x tan(theta) + k`, at onset it
    passes through the pouring rim `(R, H)` so `k = H - R tan(theta)`, and the volume
    under it falls monotonically with `theta`. Bisect for the `theta` that holds the
    cup's water and compare. No warp, no newton, no particles.

    It also prices the shape, because the cup is square and the task's hint is that the
    square may be the whole explanation: it is not. A cylinder of `R = a` integrates to
    0.58 degrees off the square cup, against 1.07 degrees between the two branches.
    """
    import math

    cfg = carry.CarryFluidEnvCfg()
    a, height, fill = cfg.cup_half_width, cfg.cup_depth, cfg.fill_depth

    def under(theta, radius, square):
        """Water volume under the onset plane, by the trapezoidal rule in x."""
        tangent, n = math.tan(theta), 2001
        total, k = 0.0, height - radius * math.tan(theta)
        xs = [-radius + 2 * radius * i / (n - 1) for i in range(n)]

        def column(x):
            depth = min(max(x * tangent + k, 0.0), height)
            width = (2 * a if square else
                     2 * math.sqrt(max(radius * radius - x * x, 0.0)))
            return depth * width

        for lo, hi in zip(xs, xs[1:]):
            total += (column(lo) + column(hi)) / 2 * (hi - lo)
        return total

    def onset(radius, square, volume):
        lo, hi = math.radians(0.1), math.radians(89.0)
        for _ in range(80):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if under(mid, radius, square) > volume else (lo, mid)
        return math.degrees((lo + hi) / 2)

    water = (2 * a) ** 2 * fill
    square = onset(a, True, water)
    assert math.isclose(square, math.degrees(carry.spill_angle(cfg)), abs_tol=0.01), (
        f"quadrature says this cup spills at {square:.3f} deg and spill_angle says "
        f"{math.degrees(carry.spill_angle(cfg)):.3f}")

    for convention in ("half_width", "equal_area"):
        form = carry.rim_angle(cfg, convention)
        assert not form["valid"], (
            f"rim_angle({convention}) claims its floor-wet branch holds on a cup filled "
            f"{fill * 1e3:g} mm of {height * 1e3:g}, which is below H/2")
        assert abs(form["angle_deg"] - square) > 0.5, (
            "the invalid branch landed on the quadrature answer, so this test proves "
            "nothing and the numbers below are coincidence")

    # The same cup made round, which is the other candidate explanation for the two
    # formulas disagreeing. It is worth half the gap the branch is worth.
    round_cup = onset(a, False, math.pi * a * a * fill)
    assert abs(round_cup - square) < 1.0, (
        f"a cylinder of the same half-width spills at {round_cup:.2f} deg against the "
        f"square cup's {square:.2f}; the plan shape was supposed to be the small term")

    # And the seam, from the other side: at h = H/2 the rim form becomes the valid one
    # and quadrature has to agree with it.
    seam_cfg = carry.CarryFluidEnvCfg(cup_depth=0.10, fill_depth=0.05)
    seam = carry.rim_angle(seam_cfg, "half_width")
    assert seam["valid"], "at h = H/2 the floor-wet branch is exactly on its boundary"
    assert math.isclose(onset(a, True, (2 * a) ** 2 * 0.05), seam["angle_deg"],
                        abs_tol=0.01), "the rim form fails quadrature where it is valid"


@needs_the_stack
@pytest.mark.parametrize("fluid_on", [False, True])
def test_a_carry_rollout_returns_a_conformant_five_tuple(fluid_on):
    """The contract `CarryFluidEnv.step` has to keep, on both sides of `fluid_on`.

    The same test as the hover task's, at the carry task's 20-wide observation, and for
    the same reason it is run on both sides of the flag: `fluid_on=False` is meant to be
    the *same task* with the water replaced by a rigid payload, and a change that altered
    the observation width or the dtype on only one path would be invisible to a test that
    ran one of them.
    """
    import torch

    env = carry.CarryFluidEnv(tiny_carry(fluid_on=fluid_on, num_envs=2))
    try:
        observations, extras = env.reset(seed=0)
        assert isinstance(extras, dict)
        assert observations["policy"].shape == (2, env.num_observations) == (2, 20)
        for _ in range(3):
            result = env.step(torch.zeros((2, env.num_actions), device=env.device))
            assert len(result) == 5
            observations, reward, terminated, truncated, extras = result
            assert observations["policy"].shape == (2, 20)
            assert observations["policy"].dtype == torch.float32
            assert reward.shape == (2,) and reward.dtype == torch.float32
            assert terminated.shape == truncated.shape == (2,)
            assert terminated.dtype == truncated.dtype == torch.bool
            assert isinstance(extras, dict)
            assert torch.isfinite(reward).all(), "a non-finite reward poisons training"
    finally:
        env.close()


@pytest.fixture(scope="module")
def settled_cup():
    """One carry environment at the **real** defaults, settled once and shared.

    The three tests below are physics and the tiny config cannot carry them: two
    particles through the depth is a cup whose water is two slabs, and a 5-step settle
    is the unsettled first step that `settle_steps` documents as applying 770% of the
    water's weight to the floor. So these run at `particle_radius = 5 mm` -- 256
    particles, four through the depth -- with the default 100-step settle, and they
    share the result because settling is the whole cost and `reset()` is exactly the
    call that undoes an episode.

    `episode_length_s` is raised so that no test here is cut short by a truncation it
    did not ask for; everything else is the shipped default, because a physics test on
    a configuration nobody runs is a test of nothing.
    """
    if not (HAS_WARP and HAS_TORCH and HAS_NEWTON):
        pytest.skip("a settled cup needs warp, newton and torch on this machine")
    env = carry.CarryFluidEnv(carry.CarryFluidEnvCfg(episode_length_s=10.0))
    yield env
    env.close()


def _hold(env, steps: int) -> tuple[float, float]:
    """Hold the cup still for `steps` env steps; return `(peak spill kg, mean F_z)`.

    The mean is over the second half of the hold, because `place()` restores the settled
    particles with their velocities zeroed and that is itself a small transient -- a
    couple of sloshing periods of it. The peak spill is tracked per step rather than
    read at the end because a termination inside the loop resets the counter.
    """
    import numpy as np
    import torch

    action = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    peak, vertical = 0.0, []
    for step in range(steps):
        env.step(action)
        peak = max(peak, float(env._spilled[0]))
        vertical.append(float(env._wrench[0, 2]))
    return peak, float(np.mean(vertical[steps // 2:]))


def test_a_cup_held_still_does_not_leak(settled_cup):
    """The one failure that would make every other number here meaningless.

    A one-layer Akinci wall does not hold this fluid in -- measured, the water goes
    through the floor and the cup feels 75% of its weight -- and `wall_thickness`
    records it at length. The symptom from outside is a cup that spills while the arm
    holds perfectly still, which is a reward signal with the task's whole content
    inverted: the policy is punished for the solver. Zero spill, exactly, and the
    quantity is kilograms.
    """
    settled_cup.reset(seed=0)
    spill, _ = _hold(settled_cup, 30)
    assert spill == 0.0, f"a still cup leaked {spill * 1e3:.3f} g through its own walls"


def test_the_load_the_cup_feels_is_the_weight_of_the_water(settled_cup):
    """The coupling's sign and its magnitude, in one number.

    `docs/sph-contract.md` fixes `wrench(body)` as the force **on the body**, and the
    sign of it has been wrong twice in this repository. A still cup of water is the one
    configuration whose answer is known in advance and is not zero: the floor carries
    the weight of what is in it, `m g` downward, and nothing else. So this catches a
    flipped sign, a wall that leaks half the load, a doubled buoyancy term
    (`buoyancy=False` is required for this backend precisely because the pressure solve
    already has it) and a wrench read on the wrong body.

    The tolerance is 15%, which is loose on purpose and is not slack: at four particles
    through the depth the free surface sits above the fill line by a fraction of a
    spacing, so the sampled water mass and the load it puts down are both resolution
    quantities. Measured on this configuration -- 256 particles, 255.5 g of water, the
    default 100-step settle, averaged over steps 20 to 40 of a still hold -- it reads
    **-2.5037 N against a 2.5069 N weight, 99.9%**, and spills exactly nothing. A
    tolerance tight enough to assert 99.9% would be asserting that resolution rather
    than the coupling, which is what this test is for.
    """
    settled_cup.reset(seed=0)
    _, vertical = _hold(settled_cup, 30)
    weight = settled_cup.water_mass * carry.G
    assert vertical < 0.0, (
        f"the water pushes the cup UP by {-vertical:.3f} N: the wrench's sign is "
        f"inverted, which is the error docs/sph-contract.md fixes the convention to stop"
    )
    assert abs(-vertical - weight) / weight < 0.15, (
        f"a still cup feels {-vertical:.3f} N from {weight:.3f} N of water "
        f"({-vertical / weight * 100:.1f}%)")


def test_shaking_the_cup_spills_and_the_wrench_says_so(settled_cup):
    """The task has content: the same cup spills when the arm moves it badly.

    Driven at the cup's own first sloshing mode (`carry.lamb_period`), which is the
    drive `carry.shake` uses and the worst case for a given speed. Three things have to
    be true together, and each one of them is a different bug if it is not:

    * spill is **greater than zero**, where holding still gave exactly zero. If this
      fails the spill counter is measuring the cup's frame rather than the world's, or
      the latch is never set.
    * the horizontal fluid force is greater than it was while holding still. A cup of
      water that is being shaken pushes sideways; one that does not is a payload bolted
      in, which is the `fluid_on=False` baseline and not this.
    * the **filtered** peak is below the unfiltered one. The measured per-step noise is
      12.9% RMS falling to 1.29% over ten samples, which is what the EMA's `alpha=0.2`
      is sized for, and an observation carrying the raw wrench is carrying mostly that.
    """
    import math

    import numpy as np
    import torch

    settled_cup.reset(seed=0)
    period = carry.lamb_period(settled_cup.cfg)
    steps = int(round(2.0 * period / settled_cup.step_dt))
    action = torch.zeros((settled_cup.num_envs, settled_cup.num_actions),
                         device=settled_cup.device)
    peak, filtered, raw = 0.0, 0.0, 0.0
    for step in range(steps):
        action[0, 0] = math.sin(2.0 * math.pi * step * settled_cup.step_dt / period)
        settled_cup.step(action)
        peak = max(peak, float(settled_cup._spilled[0]))
        filtered = max(filtered, float(np.linalg.norm(settled_cup._wrench[0, :2])))
        sample = settled_cup._sampled[0] / max(settled_cup._samples, 1)
        raw = max(raw, float(np.linalg.norm(sample[:2])))

    assert peak > 0.0, "shaking the cup at its own sloshing mode spilled nothing at all"
    assert filtered > 0.05 * settled_cup.water_mass * carry.G, (
        f"the water pushed sideways with only {filtered:.3f} N while being shaken")
    assert filtered < raw, (
        f"the EMA's peak ({filtered:.3f} N) is not below the raw one ({raw:.3f} N), so "
        f"it is not filtering anything")
