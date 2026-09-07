"""The mesh fitness table: measured, never asserted.

A fitness table has no placeholder: a missing number is `Unmeasured(reason)`, a stated
absence, and the constructor refuses None or NaN in any required field (D14). `meshed`
on a result can only become True through `mark_meshed`, which takes one of these.
Stdlib only. Skeleton (U0): the types, their guards and round trips; `from_digest` and
`schemes_for` raise NotImplementedError until U3 lands.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .strategy import Sizes

_U3 = ("not built in the U0 skeleton: U3 (claims + report + fitness) implements it "
       "against DESIGN.md section 3.11")


@dataclass(frozen=True)
class Unmeasured:
    reason: str
    """"no flow speed in a mesh-only study": a stated absence, not a placeholder."""

    def as_dict(self) -> dict:
        return {"unmeasured": self.reason}

    @staticmethod
    def from_dict(d: dict) -> "Unmeasured":
        return Unmeasured(reason=d["unmeasured"])


Verdict_ = Literal["resolved", "wall function", "buffer layer"]

_REQUIRED = ("cells", "hex_fraction", "cell_m", "wall_cell_m", "first_cell_height_m", "non_orth_max",
             "non_orth_mean", "skew_max")
_ABSENT_OK = ("smallest_passage_m", "cells_across_smallest_passage", "y_plus", "y_plus_verdict", "aspect_max")


def _is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


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

    def lines(self) -> list[str]:
        """The FITNESS block: "cells 2611 (hex 100 %)", "across the smallest passage 6.3 cells
        (2.98 mm / 0.475 mm)", "y+ not estimated: no flow speed in a mesh-only study", ..."""
        raise NotImplementedError(_U3)

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
    raise NotImplementedError(_U3)


def from_digest(digest: dict, sizes: "Sizes", smallest_passage_m: float | Unmeasured,
                study: str, flow, scale: float) -> FitnessTable:
    """digest = mesh_digest.parse(log.checkMesh): counts['cells'], counts['hexahedra'],
    non_ortho (max, avg), skewness, aspect_ratio when present. `sizes` is the Sizes
    write_case returns (3.13). y+ from case_gen.estimate_y_plus(first_cell, flow) when the
    study has a flow, else Unmeasured. An Unmeasured smallest passage gives an Unmeasured
    cells_across with the same reason (the 3D branch, D34)."""
    raise NotImplementedError(_U3)
