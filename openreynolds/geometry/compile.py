"""The compiler: a Sketch to mesh2d ops, patch rules and port intents.

The constraint features (Bypass, Row, Serpentine, the mitred corner, `line_to`) are
solved here in closed form, a Row's fit is checked here rather than at construction
(a Passage leg can grow after the Row is made, D4/D7), and the ops are handed to
`mesh2d.build_face` by path at scale 1 (D26: the classifier is wrong on a transformed
face, so the kernel never scales or mirrors the fluid; the dilate to metres happens once,
when the case is written). Skeleton (U0): signatures and docstrings from DESIGN.md
section 3.4; U1 implements them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from . import _toolbox  # noqa: F401  (mesh2d.parse_spec / build_face by path)
from .sketch import Feature, PortIntent, Sketch  # noqa: F401

if TYPE_CHECKING:
    from .claims import ClaimSet, ComplianceTable
    from .lint import Finding
    from .measure import Measurements

_U1 = ("not built in the U0 skeleton: U1 (sketch + compile) implements it against "
       "DESIGN.md section 3.4")


@dataclass
class Solved:
    kind: str
    """"Bypass" | "Row" | "Serpentine" | "Passage" | "Rect" | ..."""
    params: dict
    """The typed parameters, as the script gave them (after every transform of the tree)."""
    solved: dict
    """What the closed form produced; the key set per kind is listed below."""
    ops: list[str]
    """The op names this feature owns."""

    # Solved.solved keys, per kind (a test pins every key present; lint and measure read
    # these names, never positions):
    #   Bypass:     R, sweep, P1, C, P2, landing, return_len, lip_u, lip_deg, footprint (lo, hi),
    #               height, wall_line (axis, value, flow, outward, span), theta_leave, theta_return,
    #               lands_on (the WallRef as a string), leave_length, leave_length_default (bool)
    #   Passage:    legs (mesh2d's leg records: kind line|arc|to|corner, from, to, heading,
    #               heading_out, length, centre, radius, sweep, lands, line), length, end,
    #               end_heading, leaves (WallRef string | None), lands_on (WallRef string | None)
    #   Row:        pitch, gap, gap_declared, footprint, anchors, span (lo, hi), margin, along,
    #               free_parameter, table (the five-value table)
    #   Serpentine: pass_pitch, wall_between, bends, end, end_heading, ends_on_start_side,
    #               pass_centrelines
    #   Rect/Disk/Polygon/Band/Outline: the primitive's parameters in the sketch frame
    #               (mirrored/moved/rotated applied)

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Solved":
        return Solved(kind=d["kind"], params=dict(d.get("params", {})), solved=dict(d.get("solved", {})),
                     ops=list(d.get("ops", [])))


SOLVED_KEYS: dict[str, frozenset[str]] = {
    "Bypass": frozenset({"R", "sweep", "P1", "C", "P2", "landing", "return_len", "lip_u", "lip_deg",
                         "footprint", "height", "wall_line", "theta_leave", "theta_return", "lands_on",
                         "leave_length", "leave_length_default"}),
    "Passage": frozenset({"legs", "length", "end", "end_heading", "leaves", "lands_on"}),
    "Row": frozenset({"pitch", "gap", "gap_declared", "footprint", "anchors", "span", "margin", "along",
                      "free_parameter", "table"}),
    "Serpentine": frozenset({"pass_pitch", "wall_between", "bends", "end", "end_heading",
                             "ends_on_start_side", "pass_centrelines"}),
}
"""The listed `Solved.solved` key set per constraint kind (`test_solved_keys_are_the_listed_set`)."""


@dataclass
class Plan:
    units: str
    scale: float
    ops: list[dict]
    """mesh2d grammar, the fluid op named "fluid"; never a mirror/dilate/affine on the fluid (D30)."""
    rules: list[dict]
    """mesh2d patch rules compiled from the port intents (resolved after the build)."""
    ports: list[PortIntent]
    """The intents, verbatim."""
    features: dict[str, Solved]
    """name -> solved."""
    instances: dict[str, dict]
    """row name -> {"count", "step", "gap", "gap_declared": bool, "footprint": (w, h)}."""
    outlines: dict[str, list[list[tuple[float, float]]]]
    """Every primitive an EdgeRef can name -> its closed-form outline in the sketch frame
    (64 points per arc, a polygon's vertices, an Outline file's points), every transform of
    the feature tree applied; Row instances under "<row>[k]". This is what resolve_ports
    matches against: the OCC operands do not survive the fuse (build_face removes every
    stray entity)."""
    edge_targets: dict[str, dict]
    """"<feature>.<which>[leg]" -> {"kind": "cap" | "side" | "loop", "points": [...], "width": w};
    one entry per EdgeRef in `ports` (a test pins the cover); resolve_ports reads ONLY this."""
    apart: list[tuple[str, str, float | None]]
    """Pairs declared or defaulted apart (D35), with the declared gap."""
    declared_widths: list[float]
    expected: tuple[int, int]
    notes: list[str]
    clip_boxes: list[tuple[str, tuple]] = field(default_factory=list)
    """(feature, (x0, y0, x1, y1)) for the preview's insets."""

    def as_dict(self) -> dict:
        return {
            "units": self.units, "scale": self.scale, "ops": self.ops, "rules": self.rules,
            "ports": [p.as_dict() for p in self.ports],
            "features": {k: v.as_dict() for k, v in self.features.items()},
            "instances": self.instances, "outlines": self.outlines, "edge_targets": self.edge_targets,
            "apart": [list(a) for a in self.apart], "declared_widths": self.declared_widths,
            "expected": list(self.expected), "notes": self.notes,
            "clip_boxes": [[name, list(box)] for name, box in self.clip_boxes],
        }

    @staticmethod
    def from_dict(d: dict) -> "Plan":
        return Plan(
            units=d["units"], scale=d["scale"], ops=list(d.get("ops", [])), rules=list(d.get("rules", [])),
            ports=[PortIntent.from_dict(p) for p in d.get("ports", [])],
            features={k: Solved.from_dict(v) for k, v in d.get("features", {}).items()},
            instances=dict(d.get("instances", {})),
            outlines={k: [[tuple(p) for p in loop] for loop in v] for k, v in d.get("outlines", {}).items()},
            edge_targets=dict(d.get("edge_targets", {})),
            apart=[(a[0], a[1], a[2]) for a in d.get("apart", [])],
            declared_widths=list(d.get("declared_widths", [])),
            expected=tuple(d.get("expected", (1, 1))), notes=list(d.get("notes", [])),
            clip_boxes=[(name, tuple(box)) for name, box in d.get("clip_boxes", [])],
        )


def plan(sketch: Sketch, claims: "ClaimSet | None" = None, gmsh=None) -> Plan:
    """Walk the feature tree from `sketch.fluid`, solve every constraint feature (closed
    form; a Row on a non-Bypass item needs `gmsh` for one scratch build of the item), check
    every Row's fit (here, not at construction, D7), assign op names (the feature's `name`
    or `<kind><n>`; Row instances `<name>[k]`, 0-based, D29), and compile to the ops
    grammar. `claims` chooses a Row refusal's free parameter (3.3). Raises SketchError
    with the codes of section 5; a Row refusal carries `partial=item`."""
    raise NotImplementedError(_U1)


def plan_feature(feature: Feature, gmsh=None) -> Plan:
    """One feature alone as its own fluid (a Row's item after E-ROW-FIT): the cli builds,
    measures and draws it so an rc-5 lap still shows the single instance with its
    footprint (3.15)."""
    raise NotImplementedError(_U1)


def build(gmsh, plan: Plan) -> tuple[int, dict, list[dict]]:
    """mesh2d.build_face(gmsh, plan.ops, scale=1.0, legs=, checks=) -- in the sketch's
    units, unscaled (D26); returns (face, legs, checks) where checks are mesh2d's legacy
    dicts (lint.from_legacy turns them into Findings). Before returning: the canary --
    `isInside(2, face, p)` for p one span past the sampled bounds along +x must be 0, and
    the repaired walk's outer signed area must be positive -- else SketchError
    E-KERNEL-STATE ("the classifier answers inside for a point 60 outside the fluid; the
    face carries a transform the kernel cannot trust", rc 3, an internal error the model
    never has to fix). The dilate to metres is done by case.write_case through mesh2d.main,
    never here."""
    raise NotImplementedError(_U1)


def record(sketch: Sketch, plan: Plan, script: str, m: "Measurements", findings: list["Finding"],
           table: "ComplianceTable | None", resolved_rules: list[dict]) -> dict:
    """The record of section 4.2: a superset of the spec mesh2d.parse_spec reads."""
    raise NotImplementedError(_U1)


def recompile(record: dict) -> list[dict]:
    """Run the recorded script through plan() and return its ops; a test asserts they equal
    the recorded ops (idempotence)."""
    raise NotImplementedError(_U1)
