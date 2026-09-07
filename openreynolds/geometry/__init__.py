"""The geometry segment: a kernel (sketch, compile, measure, lint, claims, preview,
runner, fitness, case, library) that runs anywhere gmsh does, and a desk (desk.py) that
runs where the model is. The kernel imports nothing from `openreynolds`; only `desk`
needs the agent, and it is imported on first use, never on package import."""
from __future__ import annotations

_DESK = {"GeometryAgent", "GeometryResult", "BRIEF", "GEOMETRY_SYSTEM", "unavailable",
         "extract_reply", "extract_spec", "grammar", "MAX_LAPS", "MAX_SECONDS",
         "MAX_REPLY_TOKENS", "FINISH_TIMEOUT_S", "RUN_TIMEOUT_S", "COMMIT"}


def __getattr__(name):
    if name in _DESK:
        from . import desk
        return getattr(desk, name)
    raise AttributeError(f"module 'openreynolds.geometry' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _DESK)
