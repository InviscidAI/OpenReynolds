"""The seam around `mesh2d.generate`.

In Phase 1 the case is written by `mesh2d.main` (case.py), which calls `generate`
itself; the seam exists so Phase 2 can route `cli mesh --strategy` (bl-quads,
transfinite) without moving code. Skeleton (U0): the types; `QuadStrategy.mesh` raises
NotImplementedError until the seam is wired (Phase 2 owns the routing; Phase 1's test
pins only that "quads" is `mesh2d.generate`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from . import _toolbox  # noqa: F401  (mesh2d.generate / mesh_sizes2d by path)

_TODO = ("not built in the U0 skeleton: the quads strategy routes to mesh2d.generate by path "
         "(DESIGN.md section 3.12)")


@dataclass
class Sizes:
    """mesh2d.mesh_sizes2d as a type."""

    cell: float
    wall_cell: float
    thickness: float

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Sizes":
        return Sizes(cell=d["cell"], wall_cell=d["wall_cell"], thickness=d["thickness"])


class Strategy:
    name: str

    def mesh(self, gmsh, built, axis: int, rules, curve_patch, sizes: Sizes, threads: int | None):
        """-> mesh2d.MeshResult2D."""
        raise NotImplementedError(_TODO)


class QuadStrategy(Strategy):
    """name "quads": mesh2d.generate by path (Algorithm 8 + Recombination 3 + one-cell
    extrude), untouched."""

    name = "quads"

    def mesh(self, gmsh, built, axis: int, rules, curve_patch, sizes: Sizes, threads: int | None):
        raise NotImplementedError(_TODO)


STRATEGIES = {"quads": QuadStrategy()}
"""Phase 2 adds "bl-quads", "transfinite"."""
