"""The annotated picture: what the model sees first each lap and what a person sees
in the tool result.

matplotlib is imported inside the functions, never at module level: the child
interpreter imports it after the script has run (the import guard is gone by then) and
`child_env` gives it an MPLCONFIGDIR. Skeleton (U0): signatures and docstrings; U4
implements them.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from . import _toolbox  # noqa: F401  (mesh2d.preview by path for draw_plain)
from .claims import ComplianceTable  # noqa: F401
from .lint import Finding  # noqa: F401
from .measure import Measurements  # noqa: F401

if TYPE_CHECKING:
    from .compile import Plan
    from .library import ReferenceMatch

_U4 = ("not built in the U0 skeleton: U4 (preview + runner + cli) implements it against "
       "DESIGN.md section 3.9")


def draw(gmsh, face: int, plan: "Plan", m: Measurements, findings: list[Finding],
         table: ComplianceTable | None, out: Path, reference: "ReferenceMatch | None" = None,
         width_px: int = 1024) -> Path:
    """The annotated picture (matplotlib Agg, 1024 px wide, dpi 110). Main axes: the face
    filled by a coarse triangulation (Mesh.Algorithm 6, thrown away), every boundary curve
    coloured by patch (mesh2d.COLOURS plus one colour per named patch), equal aspect, light
    grid, two dimension lines for the extent. Per feature: a heading arrow at each leg's
    midpoint labelled in degrees ('260 deg (-x)'); an arc's centre dot and radius line
    labelled 'r 4.5 (outer 6)' and its sweep; a Row's pitch dimension between the first two
    instances labelled 'pitch 14.66 = 13.02 + 1.64 (solved)'; a Disk's diameter line; a Row
    refusal's footprint as a dashed box (the `partial` path). Findings: red crosses (error)
    / orange (warn) at `where` with the code in a label box; `draw` marks. Insets: an
    inset_axes window three widths across at every junction (a Bypass's leave and landing,
    every `lands_on` leg, every `start=(wall, u)` branch), every lip and every fluid cusp,
    tiled along the top edge, titled 'loops[1] landing (17.68, 1.5)', cap 8 then '...'.
    Reference panel: when given, a second axes to the right at the same scale with the
    golden PNG, titled 'library: tesla_valve (t01) -- approved <date>', and the golden
    outline polyline overlaid in grey on the main axes translated so the inlet centres
    coincide; the Hausdorff distance in the caption. Caption: the LINT block and the
    CLAIMS FAIL rows, monospace."""
    raise NotImplementedError(_U4)


def draw_plain(gmsh, built, curve_patch, out: Path, title: str, caption, marks) -> Path:
    """mesh2d.preview by path, unchanged; `cli preview --plain`."""
    raise NotImplementedError(_U4)


def outline_polyline(gmsh, face: int, spacing: float) -> list[list[tuple[float, float]]]:
    """The boundary as ordered loops of points at `spacing` arc length (the golden artefact)."""
    raise NotImplementedError(_U4)
