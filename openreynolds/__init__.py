"""OpenReynolds — a CFD agent with a real OpenFOAM workspace."""

__version__ = "0.4.2"

__all__ = [
    "Study",
    "__version__",
    "fetch",
    "results",
    "shutdown",
    "start",
    "studies",
    "upload",
]
"""The library surface: what `import openreynolds` is for (`sdk.py`). A curated export
list, as `openreynolds.backend` has one -- everything else in the package is wiring
that may be rearranged, and this is the part that will not move under a caller."""

_LAZY = frozenset(__all__) - {"__version__"}


def __getattr__(name: str) -> object:
    """Fetch the surface from `sdk` on first use.

    Importing this package has to stay a cheap import: a session needs the TUI, a model
    client and the workspace plumbing, and `import openreynolds` to read `__version__`
    or to hold a `Study` must not pay for any of it. So the names are declared here and
    fetched when one is touched (PEP 562).
    """
    if name in _LAZY:
        from . import sdk

        return getattr(sdk, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
