"""`openreynolds/live/isaac.py` still matches the interface it claims to match.

The file's whole value is a claim about somebody else's package: that
`HoverFluidEnv.step`, `.reset`, `.close`, `.render` and `.seed` are
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
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

from openreynolds.live import isaac


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

ISAAC_LAB_SIGNATURES = {
    "__init__": "(self, cfg: 'HoverFluidEnvCfg', render_mode: 'str | None' = None, "
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


# -- the config ------------------------------------------------------------------------


def test_the_config_carries_isaac_labs_five_required_fields():
    """`DirectRLEnvCfg`'s five `MISSING` fields are all present, and all filled in.

    They are the five a `DirectRLEnvCfg` cannot be constructed without: `decimation`,
    `episode_length_s`, `scene`, `observation_space`, `action_space`. `scene` is `None`
    here on purpose (an `InteractiveScene` needs Isaac Sim's USD stage), so the test
    asserts it *exists* rather than that it is truthy -- and that is the one field whose
    absence would be easiest to rationalise away.
    """
    fields = isaac.HoverFluidEnvCfg.__dataclass_fields__
    for name in ("decimation", "episode_length_s", "scene", "observation_space",
                 "action_space"):
        assert name in fields, f"{name} is one of DirectRLEnvCfg's required five"
    cfg = isaac.HoverFluidEnvCfg()
    assert cfg.decimation >= 1
    assert cfg.episode_length_s > 0
    assert cfg.observation_space == 13 and cfg.action_space == 4
    assert cfg.scene is None


def test_every_config_field_is_documented():
    """A field with no docstring is a knob nobody can use.

    `ARCHITECTURE_LIVE.md`'s whole argument for this extension is written in prose
    beside the code, and a dataclass is the one place a silent addition is invisible:
    the class still works, the report still prints, and the field is a number somebody
    has to guess the meaning of. Checked by parsing, because `__doc__` on a dataclass
    field does not exist.
    """
    import ast

    source = (ROOT / "openreynolds" / "live" / "isaac.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    body = next(node.body for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name == "HoverFluidEnvCfg")
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
    assert not undocumented, f"HoverFluidEnvCfg fields with no docstring: {undocumented}"


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


def test_the_module_does_not_import_torch_gymnasium_or_isaaclab():
    """Importing this module costs none of the three, which is the acceptance criterion.

    `openreynolds/tools.py` imports `openreynolds.live` to decide whether to offer the
    agent its `live_sim` tool, so a module-scope `import torch` here would make a plain
    `openreynolds --help` on a laptop pay for a 200 MB wheel -- or fail, which is what
    it did before the lazy discipline in `live/__init__.py` existed. Checked by parsing
    rather than by watching `sys.modules`, because another test in the same session may
    legitimately have imported torch already.
    """
    import ast

    source = (ROOT / "openreynolds" / "live" / "isaac.py").read_text(encoding="utf-8")
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
    assert not offenders, f"imported at module scope: {offenders}"


def test_it_is_a_vector_env_with_same_step_autoreset():
    """The two class attributes every gymnasium >= 1.0 vector wrapper branches on.

    SAME_STEP autoreset is the one that changes what `step` must return: an environment
    that terminates hands back the observation of the **reset** state in the same call.
    `AutoresetMode` is a plain `Enum` and not a `str` subclass, so the class attribute
    carries the member's value as a string and the instance upgrades it -- which is why
    this test accepts either form rather than asserting one.
    """
    assert isaac.HoverFluidEnv.is_vector_env is True
    assert isaac.HoverFluidEnv.metadata["render_modes"] == [None, "human", "rgb_array"]
    mode = isaac.HoverFluidEnv.metadata["autoreset_mode"]
    assert _autoreset_name(mode) == "SAME_STEP", mode


def _autoreset_name(mode) -> str:
    """`SAME_STEP`, from either the gymnasium member or the string standing in for it."""
    if hasattr(mode, "name"):
        return str(mode.name)
    return {"SameStep": "SAME_STEP", "NextStep": "NEXT_STEP",
            "Disabled": "DISABLED"}.get(str(mode), str(mode))


@pytest.mark.parametrize("name", sorted(ISAAC_LAB_SIGNATURES))
def test_the_signatures_are_the_ones_isaac_lab_declares(name):
    """Each method's `inspect.signature`, against a literal.

    A literal and not a live comparison, so that this fails in the environment CI runs
    in, where `isaaclab` is absent and must stay absent. The companion test below is the
    live comparison and skips here.
    """
    got = str(inspect.signature(getattr(isaac.HoverFluidEnv, name)))
    assert got == ISAAC_LAB_SIGNATURES[name], f"{name}{got}"


def test_seed_is_a_static_method_as_direct_rl_env_declares_it():
    """`DirectRLEnv.seed` is a `@staticmethod`, so it seeds the process and not an
    instance.

    Asserted separately from the signature, because `inspect.signature` on a plain
    method and on a static one differ only by `self` and a rename would hide it. What
    this protects is `HoverFluidEnv.seed(0)` on the class, before anything is
    constructed, which is how a training script seeds a run.
    """
    assert isinstance(inspect.getattr_static(isaac.HoverFluidEnv, "seed"), staticmethod)
    assert isaac.HoverFluidEnv.seed(7) == 7


@pytest.mark.parametrize("hook", THE_FIVE_HOOKS)
def test_the_five_direct_rl_env_hooks_exist(hook):
    assert callable(getattr(isaac.HoverFluidEnv, hook, None)), hook


def test_render_modes_are_enforced():
    """An unknown `render_mode` is refused before anything expensive is built, which is
    the difference between a typo found in a second and one found after a lattice and a
    MuJoCo model have been compiled."""
    with pytest.raises(ValueError, match="render_mode"):
        isaac.HoverFluidEnv(tiny(), render_mode="pretty")


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
@pytest.mark.parametrize("name", sorted(ISAAC_LAB_SIGNATURES))
def test_our_signatures_match_the_real_direct_rl_env(name):
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
    ours = shape(getattr(isaac.HoverFluidEnv, name))
    assert ours == theirs, f"{name}: ours {ours} vs DirectRLEnv's {theirs}"


@needs_isaaclab
def test_the_class_attributes_match_the_real_direct_rl_env():
    from isaaclab.envs.direct_rl_env import DirectRLEnv

    assert isaac.HoverFluidEnv.is_vector_env == DirectRLEnv.is_vector_env
    assert (isaac.HoverFluidEnv.metadata["render_modes"]
            == DirectRLEnv.metadata["render_modes"])
    assert (_autoreset_name(isaac.HoverFluidEnv.metadata["autoreset_mode"])
            == _autoreset_name(DirectRLEnv.metadata["autoreset_mode"]))


@needs_isaaclab
def test_our_config_covers_isaac_labs_required_fields():
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
    ours = set(isaac.HoverFluidEnvCfg.__dataclass_fields__)
    assert required <= ours, f"DirectRLEnvCfg requires {sorted(required - ours)} too"
