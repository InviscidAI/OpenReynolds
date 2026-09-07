"""The toolbox scripts, loaded by path.

`openreynolds/toolbox/` is a directory of scripts that run on the instance, not a
package: `mesh2d.py` is copied there and rebuilds a case from `geometry.json` with no
kernel present. The kernel calls the same file by path (D2), under one module name per
script, cached so every caller shares the one module object (dataclass identity holds).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

TOOLBOX: Path = Path(__file__).resolve().parents[1] / "toolbox"
"""openreynolds/toolbox beside this package."""

_LOADED: dict[str, ModuleType] = {}


def load(name: str) -> ModuleType:
    """toolbox/<name>.py loaded by path under the module name `toolbox_<name>`, cached.
    The toolbox is a directory of scripts, not a package (geometry.py loads mesh_digest
    this way today). `name` is one of mesh2d, case_gen, mesh_digest."""
    if name in _LOADED:
        return _LOADED[name]
    path = TOOLBOX / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"no toolbox script {name}.py under {TOOLBOX}; the toolbox "
                                "scripts are mesh2d, case_gen and mesh_digest")
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} cannot be loaded as a module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _LOADED[name] = module
    return module
