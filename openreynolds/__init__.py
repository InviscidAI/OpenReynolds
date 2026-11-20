"""OpenReynolds — a CFD agent with a real OpenFOAM workspace."""

__version__ = "0.4.2"

__all__ = [
    "Study",
    "__version__",
    "fetch",
    "follow",
    "results",
    "show",
    "shutdown",
    "start",
    "studies",
    "upload",
]
"""The library surface: what `import openreynolds` is for (`sdk.py`). A curated export
list, as `openreynolds.backend` has one -- everything else in the package is wiring
that may be rearranged, and this is the part that will not move under a caller."""

_LAZY = {name: "sdk" for name in frozenset(__all__) - {"__version__"}}
_LAZY.update(follow="notebook", show="notebook")
"""Which module each exported name is fetched from. `show` and `follow` are the
notebook's two (`notebook.py`); everything else is the SDK's. They are on the same
surface because a notebook user imports the package once and should not have to learn
which submodule a convenience lives in."""


def __getattr__(name: str) -> object:
    """Fetch the surface from the module that owns it, on first use.

    Importing this package has to stay a cheap import: a session needs the TUI, a model
    client and the workspace plumbing, and `import openreynolds` to read `__version__`
    or to hold a `Study` must not pay for any of it. So the names are declared here and
    fetched when one is touched (PEP 562).
    """
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
