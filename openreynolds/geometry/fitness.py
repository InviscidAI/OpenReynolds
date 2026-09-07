"""The mesh fitness table: measured, never asserted.

A fitness table has no placeholder: a missing number is `Unmeasured(reason)`, a stated
absence, and the constructor refuses None or NaN in any required field (D14). `meshed`
on a result can only become True through `mark_meshed`, which takes one of these.
Stdlib only at import; `from_digest` loads the toolbox's `case_gen` by path for the y+
correlation when the study has a flow, so the number comes from the same function the
case writer uses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .strategy import Sizes

NO_FLOW = "no flow speed in a mesh-only study"
NO_ASPECT = "checkMesh's log has no aspect ratio line"
NO_2D_PASSAGE = "no 2D passage measure for a 3D body in Phase 1"

SCHEMES_LIMITED_FROM = 65.0
SCHEMES_REFUSED_ABOVE = 70.0
"""Non-orthogonality bands: standard below 65, limited 65-70, refused above 70 (the one
Phase 2 gate that is cheap now)."""

Y_PLUS_RESOLVED_BELOW = 5.0
Y_PLUS_WALL_FUNCTION_FROM = 30.0


@dataclass(frozen=True)
class Unmeasured:
    reason: str
    """"no flow speed in a mesh-only study": a stated absence, not a placeholder."""

    def as_dict(self) -> dict:
        return {"unmeasured": self.reason}

    @staticmethod
    def from_dict(d: dict) -> "Unmeasured":
        return Unmeasured(reason=d["unmeasured"])

    def __str__(self) -> str:
        return f"not measured ({self.reason})"


Verdict_ = Literal["resolved", "wall function", "buffer layer"]

_REQUIRED = ("cells", "hex_fraction", "cell_m", "wall_cell_m", "first_cell_height_m", "non_orth_max",
             "non_orth_mean", "skew_max")
_ABSENT_OK = ("smallest_passage_m", "cells_across_smallest_passage", "y_plus", "y_plus_verdict", "aspect_max")


def _is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _mm(metres: float) -> str:
    return f"{float(metres) * 1000:.3g} mm"


@dataclass
class FitnessTable:
    cells: int
    hex_fraction: float
    """hexahedra / cells from checkMesh."""
    smallest_passage_m: float | Unmeasured
    """measure.passage.min * scale (a measurement, never a declared width);
    Unmeasured("no 2D passage measure for a 3D body in Phase 1") on the 3D branch (D34)."""
    cell_m: float
    """The target cell (mesh_sizes2d)."""
    cells_across_smallest_passage: float | Unmeasured
    """Unmeasured whenever smallest_passage_m is."""
    wall_cell_m: float
    first_cell_height_m: float
    """wall_cell / 2."""
    y_plus: float | Unmeasured
    y_plus_verdict: Verdict_ | Unmeasured
    non_orth_max: float
    non_orth_mean: float
    skew_max: float
    aspect_max: float | Unmeasured
    """checkMesh prints it in some versions; Unmeasured("checkMesh's log has no aspect
    ratio line") otherwise."""
    schemes: Literal["standard", "limited", "refused"]
    quality_gate: Literal["pass", "limited schemes", "refused", "not applied"] = "not applied"
    measured_from: str = ""
    """"log.checkMesh + mesh_sizes2d cell 4.75e-4 m + passage.min 2.98 mm"."""

    def __post_init__(self):
        """Refuses None or NaN in any required field (cells, hex_fraction, cell_m,
        wall_cell_m, first_cell_height_m, non_orth_max, non_orth_mean, skew_max) and None in
        smallest_passage_m / cells_across_smallest_passage / y_plus / y_plus_verdict /
        aspect_max (Unmeasured is the way to say absent, and a float smallest_passage_m
        with an Unmeasured cells_across, or the reverse, is refused too): ValueError
        ('FitnessTable.<f> is required; a fitness table is measured, not asserted -- there
        is no placeholder'). measured_from must be non-empty."""
        for name in _REQUIRED:
            value = getattr(self, name)
            if value is None or _is_nan(value):
                raise ValueError(f"FitnessTable.{name} is required; a fitness table is measured, "
                                 "not asserted -- there is no placeholder")
        for name in _ABSENT_OK:
            value = getattr(self, name)
            if value is None or _is_nan(value):
                raise ValueError(f"FitnessTable.{name} is required; a fitness table is measured, "
                                 "not asserted -- there is no placeholder (Unmeasured(reason) says "
                                 "why a number is absent)")
        passage_absent = isinstance(self.smallest_passage_m, Unmeasured)
        across_absent = isinstance(self.cells_across_smallest_passage, Unmeasured)
        if passage_absent != across_absent:
            raise ValueError("FitnessTable.cells_across_smallest_passage is Unmeasured exactly when "
                             "smallest_passage_m is; one measured and the other absent is not a "
                             "measurement")
        if not (self.measured_from or "").strip():
            raise ValueError("FitnessTable.measured_from is required; a fitness table says what it was "
                             "measured from (the checkMesh log, the sizing, the passage measure)")
        if self.schemes not in ("standard", "limited", "refused"):
            raise ValueError(f"FitnessTable.schemes is {self.schemes!r}; one of standard, limited, refused "
                             "(schemes_for(non_orth_max) chooses it)")

    def lines(self) -> list[str]:
        """The FITNESS block: "cells 2611 (hex 100 %)", "across the smallest passage 6.3 cells
        (2.98 mm / 0.475 mm)", "y+ not estimated: no flow speed in a mesh-only study", ..."""
        out = [f"cells {self.cells} (hex {self.hex_fraction * 100:.0f} %)"]
        if isinstance(self.smallest_passage_m, Unmeasured):
            out.append(f"across the smallest passage: not measured ({self.smallest_passage_m.reason})")
        else:
            out.append(f"across the smallest passage {float(self.cells_across_smallest_passage):.1f} cells "
                       f"({_mm(self.smallest_passage_m)} / {_mm(self.cell_m)})")
        out.append(f"target cell {_mm(self.cell_m)}, wall cell {_mm(self.wall_cell_m)}, "
                   f"first cell height {_mm(self.first_cell_height_m)}")
        if isinstance(self.y_plus, Unmeasured):
            out.append(f"y+ not estimated: {self.y_plus.reason}")
        else:
            verdict = self.y_plus_verdict if not isinstance(self.y_plus_verdict, Unmeasured) else "unjudged"
            out.append(f"y+ {float(self.y_plus):.3g} ({verdict})")
        quality = f"non-orthogonality max {self.non_orth_max:.4g}, mean {self.non_orth_mean:.3g}   skewness max {self.skew_max:.3g}"
        if isinstance(self.aspect_max, Unmeasured):
            quality += f"   aspect ratio not in the log ({self.aspect_max.reason})"
        else:
            quality += f"   aspect ratio max {float(self.aspect_max):.3g}"
        out.append(quality)
        out.append(f"schemes {self.schemes}")
        out.append(f"quality gate {self.quality_gate}")
        return out

    def as_dict(self) -> dict:
        """`Unmeasured` serialised as {"unmeasured": "<reason>"} (DESIGN.md 4.4)."""
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = value.as_dict() if isinstance(value, Unmeasured) else value
        return out

    @staticmethod
    def from_dict(d: dict) -> "FitnessTable":
        values = {}
        for f in fields(FitnessTable):
            if f.name not in d:
                continue
            value = d[f.name]
            values[f.name] = Unmeasured.from_dict(value) if isinstance(value, dict) and "unmeasured" in value else value
        return FitnessTable(**values)


def schemes_for(non_orth_max: float) -> str:
    """'standard' below 65, 'limited' 65-70 (case_gen writes the limited set then), 'refused'
    above 70 (the one Phase 2 gate that is cheap now)."""
    value = float(non_orth_max)
    if math.isnan(value):
        raise ValueError("schemes_for: non_orth_max is NaN; checkMesh's non-orthogonality line is the number")
    if value < SCHEMES_LIMITED_FROM:
        return "standard"
    if value <= SCHEMES_REFUSED_ABOVE:
        return "limited"
    return "refused"


def y_plus_verdict(y_plus: float) -> str:
    """Below 5 the first cell is in the viscous sublayer (resolved); 5..30 is the buffer
    layer no wall model covers; from 30 the wall function's range."""
    if y_plus < Y_PLUS_RESOLVED_BELOW:
        return "resolved"
    if y_plus < Y_PLUS_WALL_FUNCTION_FROM:
        return "buffer layer"
    return "wall function"


def _number(value, what: str) -> float:
    if value is None:
        raise ValueError(f"the checkMesh log has no {what}; a fitness table is measured, not asserted")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"the checkMesh log's {what} is not a number: {value!r}") from None


def from_digest(digest: dict, sizes: "Sizes", smallest_passage_m: float | Unmeasured,
                study: str, flow, scale: float) -> FitnessTable:
    """digest = mesh_digest.parse(log.checkMesh): counts['cells'], counts['hexahedra'],
    non_ortho (max, avg), skewness, aspect_ratio when present. `sizes` is the Sizes
    write_case returns (3.13). y+ from case_gen.estimate_y_plus(first_cell, flow) when the
    study has a flow, else Unmeasured. An Unmeasured smallest passage gives an Unmeasured
    cells_across with the same reason (the 3D branch, D34)."""
    counts = digest.get("counts") or {}
    cells = int(_number(counts.get("cells"), "cells count"))
    if cells <= 0:
        raise ValueError(f"the checkMesh log counts {cells} cells; nothing was meshed")
    hexahedra = int(counts.get("hexahedra") or 0)
    hex_fraction = hexahedra / cells
    non_ortho = digest.get("non_ortho")
    if not isinstance(non_ortho, (tuple, list)) or len(non_ortho) < 2:
        raise ValueError("the checkMesh log has no non-orthogonality line (Max: ... average: ...); a fitness "
                         "table is measured, not asserted")
    non_orth_max = _number(non_ortho[0], "non-orthogonality max")
    non_orth_mean = _number(non_ortho[1], "non-orthogonality average")
    skew_max = _number(digest.get("skewness"), "max skewness line")
    aspect = digest.get("aspect_ratio")
    aspect_max: float | Unmeasured = Unmeasured(NO_ASPECT) if aspect is None else _number(aspect, "aspect ratio")

    cell_m = _number(getattr(sizes, "cell", None), "target cell (Sizes.cell)")
    wall_cell_m = _number(getattr(sizes, "wall_cell", None), "wall cell (Sizes.wall_cell)")
    first_cell = wall_cell_m / 2.0

    if isinstance(smallest_passage_m, Unmeasured):
        across: float | Unmeasured = Unmeasured(smallest_passage_m.reason)
        passage_text = f"passage not measured ({smallest_passage_m.reason})"
    else:
        passage = _number(smallest_passage_m, "smallest passage")
        across = passage / cell_m
        passage_text = f"passage.min {_mm(passage)}"

    speed = getattr(flow, "speed", None) if flow is not None else None
    if flow is None or not speed or float(speed) <= 0:
        reason = NO_FLOW if (study in ("mesh", "", None) or flow is None) else f"no flow speed in a {study} study"
        y_plus: float | Unmeasured = Unmeasured(reason)
        y_verdict: str | Unmeasured = Unmeasured(reason)
    else:
        from . import _toolbox
        estimate = _toolbox.load("case_gen").estimate_y_plus
        y_plus = float(estimate(first_cell, flow))
        y_verdict = y_plus_verdict(y_plus)

    return FitnessTable(
        cells=cells, hex_fraction=hex_fraction, smallest_passage_m=smallest_passage_m, cell_m=cell_m,
        cells_across_smallest_passage=across, wall_cell_m=wall_cell_m, first_cell_height_m=first_cell,
        y_plus=y_plus, y_plus_verdict=y_verdict, non_orth_max=non_orth_max, non_orth_mean=non_orth_mean,
        skew_max=skew_max, aspect_max=aspect_max, schemes=schemes_for(non_orth_max), quality_gate="not applied",
        measured_from=f"log.checkMesh + mesh_sizes2d cell {cell_m:.3g} m + {passage_text}",
    )
