"""The live-sync backend: a GPU lattice Boltzmann fluid in lockstep with Newton.

    pip install 'openreynolds[live]'

    from openreynolds.live import Domain, FluidWorld

    world = FluidWorld(domain=Domain((0, 0, 0), (0.4, 0.4, 0.02), faces={
                           "-x": "inlet", "+x": "outlet",
                           "-y": "slip", "+y": "slip",
                           "-z": "periodic", "+z": "periodic"}),
                       resolution=200, fluid="water", dt=1e-3,
                       reference_velocity=0.01)
    world.add_shape("cylinder", position=(0.1, 0.2, 0.0), scale=(0.005, 0.02, 0.0))
    world.initialise(velocity=(0.01, 0, 0))
    for _ in range(2000):
        world.step()

and with a robot on the other side,

    from openreynolds.live import FluidWorld, NewtonCoupling

    link = NewtonCoupling(model, solver, world)
    for _ in range(steps):
        link.step()          # one Newton step, N lattice substeps, wrenches both ways

`docs/live-sync.md` is the quickstart, `ARCHITECTURE_LIVE.md` is where this plugs into
the rest of OpenReynolds, and `DECISIONS_LIVE.md` is every choice taken while building
it, including why this is not XLB.

## Importing this does not import Warp

Every name below is resolved lazily. That is not tidiness: the acceptance criterion for
this extension is that core OpenReynolds installs and imports without the `[live]`
extra, and `openreynolds/tools.py` imports this module to find out whether to offer the
agent its `live_sim` tool. If that import pulled Warp, a plain `openreynolds --help` on
a laptop with no GPU would fail, which is the whole product.

So `available()` answers the question without importing anything heavy, `require()`
raises `LiveUnavailable` with the one command that fixes it, and the real classes arrive
through `__getattr__` on first use. `tests/test_live_optional.py` checks all three by
hiding Warp from the import system.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "available", "require", "LiveUnavailable",
    "Domain", "FluidWorld", "Rotor",
    "NewtonCoupling", "Scaling", "Fluid", "FLUIDS", "UnphysicalScaling",
    "CASES", "run_case", "describe_cases",
]

_LAZY = {
    "Domain": ("world", "Domain"),
    "FluidWorld": ("world", "FluidWorld"),
    "Rotor": ("world", "Rotor"),
    "LiveUnavailable": ("units", "LiveUnavailable"),
    "NewtonCoupling": ("couple", "NewtonCoupling"),
    "Scaling": ("units", "Scaling"),
    "Fluid": ("units", "Fluid"),
    "FLUIDS": ("units", "FLUIDS"),
    "UnphysicalScaling": ("units", "UnphysicalScaling"),
    "CASES": ("cases", "CASES"),
    "run_case": ("cases", "run_case"),
    "describe_cases": ("cases", "describe_cases"),
}
"""Name -> (submodule, attribute). `units.py` imports nothing but the standard library
and so is safe to reach eagerly; everything else here touches Warp."""


def available() -> bool:
    """Whether the `[live]` extra is installed, cheaply and without importing it.

    `importlib.util.find_spec` rather than a try/except around the import: finding the
    spec does not execute Warp's module-level `wp.init()`, which probes for a CUDA
    driver and prints a line about it. A question asked once per session to decide
    whether to offer a tool should not have a side effect.
    """
    import importlib.util

    try:
        return importlib.util.find_spec("warp") is not None
    except (ImportError, ValueError):
        return False


def require() -> Any:
    """Warp, or `LiveUnavailable` naming the install command. See `units.require`.

    From `units`, not from `world`: reading `world.py` imports Warp (through
    `lattice.py`), so going through it meant that on a machine without the extra this
    function raised `lattice.py`'s own bare import error and the sentence naming the
    install command never ran. `units.py` imports nothing but the standard library.
    """
    from .units import require as _require

    return _require()


def __getattr__(name: str) -> Any:
    try:
        module, attribute = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(f".{module}", __package__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
