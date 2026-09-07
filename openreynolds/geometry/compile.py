"""The compiler: a Sketch to mesh2d ops, patch rules and port intents.

The constraint features (Bypass, Row, Serpentine, the mitred corner, `line_to`) are
solved here in closed form, a Row's fit is checked here rather than at construction
(a Passage leg can grow after the Row is made, D4/D7), and the ops are handed to
`mesh2d.build_face` by path at scale 1 (D26: the classifier is wrong on a transformed
face, so the kernel never scales or mirrors the fluid; the dilate to metres happens once,
when the case is written).

Every feature is compiled in its own declared frame; the tree's transforms apply on top:
`moved` and `rotated` become `translate` / `rotate` ops (both correct for the classifier,
measured), `mirrored` is pushed down to the leaves as a reflection of their parameters
(D30) -- a rect stays a rect, a turn changes hand, a Bypass follows its mirrored wall --
so the fluid never carries a `mirror` op. Beside the ops the plan carries closed-form
outlines of every primitive and one `edge_targets` entry per port intent, because the
OCC operands do not survive the fuse and the port resolution after the build reads only
those (section 3.4).

A `from` or `to` cut in the ops grammar is an axis-aligned line, so a branch or a loop on
an inclined wall is compiled in the frame where its wall is horizontal and turned back
with a `rotate` op; a Passage with mitred corners is one `channel` per straight run (a
turtle path cannot step back to start the next run early), fused under the passage's
name, which is exactly the two rects of section 7.5.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from . import _toolbox
from . import sketch as sk
from .sketch import (Band, BodyInBox, Bypass, Cut, Disk, EdgeRef, Feature, Fuse, Intersect, Mirrored, Moved,
                     Outline, Passage, Polygon, PortIntent, Rect, Rotated, Row, RowInstance, Serpentine, Sketch,
                     SketchError, WallRef, fmt, label, wall_line)

if TYPE_CHECKING:
    from .claims import ClaimSet, ComplianceTable
    from .lint import Finding
    from .measure import Measurements

Point = tuple[float, float]
Affine = tuple[float, float, float, float, float, float]
"""x' = a x + b y + tx ; y' = c x + d y + ty, as (a, b, c, d, tx, ty)."""

IDENTITY: Affine = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
ARC_POINTS = 64
"""Points per arc in a closed-form outline (section 3.4)."""
_EPS = 1e-9


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
        return {"kind": self.kind, "params": _jsonable(self.params), "solved": _jsonable(self.solved),
                "ops": list(self.ops)}

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
    external: dict | None = None
    """A BodyInBox's flow box in body lengths ({"ahead", "behind", "above", "below", "far",
    "body"}): `build` puts the box round the body through `mesh2d.external_face`; None for
    an internal passage."""

    def as_dict(self) -> dict:
        return {
            "units": self.units, "scale": self.scale, "ops": _jsonable(self.ops), "rules": _jsonable(self.rules),
            "ports": [p.as_dict() for p in self.ports],
            "features": {k: v.as_dict() for k, v in self.features.items()},
            "instances": _jsonable(self.instances), "outlines": _jsonable(self.outlines),
            "edge_targets": _jsonable(self.edge_targets),
            "apart": [list(a) for a in self.apart], "declared_widths": self.declared_widths,
            "expected": list(self.expected), "notes": self.notes,
            "clip_boxes": [[name, list(box)] for name, box in self.clip_boxes],
            "external": _jsonable(self.external),
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
            external=d.get("external"),
        )


# -- small geometry: affines, points, outlines ------------------------------------------------


SNAP_DECIMALS = 10
"""Every number in an emitted op and in a leg record is rounded to this many decimals: a
turtle walk through 180 degrees leaves sin(pi) = 1.2e-16 on a coordinate that is 0, and
an op or a leg record that reads (-1.225e-16, 18) is noise the model would see (7.2
prints (0, 18)). Ten decimals is 1e-10 of a sketch unit, far below anything the kernel
measures at."""


def _snap(v: float) -> float:
    r = round(float(v), SNAP_DECIMALS)
    return 0.0 if r == 0 else r


def _jsonable(value):
    """Tuples to lists, recursively, so the plan and the record survive a JSON round trip
    unchanged and `recompile` compares like with like."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _snapped(value):
    """`_jsonable` with every float snapped: what an op carries when it is emitted, so
    the ops grammar never sees the walk's noise (the plan's other values are kept as
    computed and printed to four figures)."""
    if isinstance(value, dict):
        return {str(k): _snapped(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapped(v) for v in value]
    if isinstance(value, float):
        return _snap(value)
    return value


def _apply(T: Affine, p) -> Point:
    a, b, c, d, tx, ty = T
    x, y = float(p[0]), float(p[1])
    return (a * x + b * y + tx, c * x + d * y + ty)


def _apply_vec(T: Affine, v) -> Point:
    a, b, c, d, _, _ = T
    return (a * v[0] + b * v[1], c * v[0] + d * v[1])


def _compose(outer: Affine, inner: Affine) -> Affine:
    """`inner` first, then `outer`."""
    a2, b2, c2, d2, tx2, ty2 = outer
    a1, b1, c1, d1, tx1, ty1 = inner
    return (a2 * a1 + b2 * c1, a2 * b1 + b2 * d1, c2 * a1 + d2 * c1, c2 * b1 + d2 * d1,
            a2 * tx1 + b2 * ty1 + tx2, c2 * tx1 + d2 * ty1 + ty2)


def _det(T: Affine) -> float:
    return T[0] * T[3] - T[1] * T[2]


def _mirror_affine(axis: str, at: float) -> Affine:
    return (-1.0, 0.0, 0.0, 1.0, 2 * at, 0.0) if axis == "x" else (1.0, 0.0, 0.0, -1.0, 0.0, 2 * at)


def _rotate_affine(deg: float, about: Point) -> Affine:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    ax, ay = about
    return (c, -s, s, c, ax - c * ax + s * ay, ay - s * ax - c * ay)


def _translate_affine(dx: float, dy: float) -> Affine:
    return (1.0, 0.0, 0.0, 1.0, dx, dy)


def _angle(v) -> float:
    return math.degrees(math.atan2(v[1], v[0]))


def _dir(deg: float) -> Point:
    return (math.cos(math.radians(deg)), math.sin(math.radians(deg)))


def _left(deg: float) -> Point:
    return (-math.sin(math.radians(deg)), math.cos(math.radians(deg)))


def _heading_under(T: Affine, deg: float) -> float:
    return _angle(_apply_vec(T, _dir(deg)))


def _norm360(deg: float) -> float:
    out = deg % 360.0
    return 0.0 if abs(out - 360.0) < 1e-9 else out


def _cross(a, b) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _signed_area(pts: list[Point]) -> float:
    return 0.5 * sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
                     for i in range(len(pts)))


def _ccw(pts: list[Point]) -> list[Point]:
    return list(reversed(pts)) if _signed_area(pts) < 0 else list(pts)


def _dedupe(pts: list[Point]) -> list[Point]:
    out: list[Point] = []
    for p in pts:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > _EPS:
            out.append(p)
    if len(out) > 1 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= _EPS:
        out.pop()
    return out


def _bbox(pts) -> tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _arc_points(centre: Point, radius: float, a0: float, a1: float, n: int = ARC_POINTS) -> list[Point]:
    return [(centre[0] + radius * math.cos(math.radians(a0 + (a1 - a0) * j / n)),
             centre[1] + radius * math.sin(math.radians(a0 + (a1 - a0) * j / n))) for j in range(n + 1)]


def _line_hit(p: Point, d: Point, q: Point, n: Point) -> Point:
    """The point of the line through p along d that lies on the line through q with normal n."""
    denom = d[0] * n[0] + d[1] * n[1]
    t = ((q[0] - p[0]) * n[0] + (q[1] - p[1]) * n[1]) / denom
    return (p[0] + t * d[0], p[1] + t * d[1])


def _clip_half_plane(loop: list[Point], q: Point, n: Point) -> list[Point]:
    """Sutherland-Hodgman: the part of the polygon on the +n side of the line through q."""
    def inside(p):
        return (p[0] - q[0]) * n[0] + (p[1] - q[1]) * n[1] >= -_EPS

    def cross_point(a, b):
        da, db = _dot((a[0] - q[0], a[1] - q[1]), n), _dot((b[0] - q[0], b[1] - q[1]), n)
        t = da / (da - db)
        return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))

    out: list[Point] = []
    for i, b in enumerate(loop):
        a = loop[i - 1]
        if inside(b):
            if not inside(a):
                out.append(cross_point(a, b))
            out.append(b)
        elif inside(a):
            out.append(cross_point(a, b))
    return _dedupe(out)


def _transform_records(records: list[dict], T: Affine) -> list[dict]:
    """Leg records under an affine: points moved, headings turned, turns changing hand
    under a reflection."""
    flip = -1.0 if _det(T) < 0 else 1.0
    out = []
    for rec in records:
        r = dict(rec)
        for key in ("from", "to", "centre", "lands", "at", "outer_vertex", "inner_vertex"):
            if r.get(key) is not None:
                x, y = _apply(T, r[key])
                r[key] = (_snap(x), _snap(y))
        for key in ("heading", "heading_out"):
            if r.get(key) is not None:
                r[key] = _snap(_heading_under(T, r[key]))
        for key in ("sweep", "deg"):
            if r.get(key) is not None:
                r[key] = _snap(flip * r[key])
        out.append(r)
    return out


def strip_walls(width: float, records: list[dict], from_line: tuple[Point, Point] | None = None,
                to_line: tuple[Point, Point] | None = None) -> tuple[list[Point], list[Point]]:
    """The two walls of a constant-width channel along its leg records (kind line | arc |
    corner | to), as polylines in walk order: (left wall, right wall). `from_line` /
    `to_line` are (point, normal) of the flush cuts: the first leg's walls are extended back
    to the from line and the last leg's (or a `to` leg's) forward to the to line, so the
    caps lie ON the wall the channel leaves or lands on. A corner leg replaces the two
    wall points at the centreline corner by the mitre's outer and inner vertices (D36)."""
    hw = width / 2
    left: list[Point] = []
    right: list[Point] = []
    after_corner = False
    last = len(records) - 1
    for i, rec in enumerate(records):
        kind = rec["kind"]
        if kind in ("line", "to"):
            d, nl = _dir(rec["heading"]), _left(rec["heading"])
            a, b = rec["from"], rec["to"]
            la, lb = (a[0] + nl[0] * hw, a[1] + nl[1] * hw), (b[0] + nl[0] * hw, b[1] + nl[1] * hw)
            ra, rb = (a[0] - nl[0] * hw, a[1] - nl[1] * hw), (b[0] - nl[0] * hw, b[1] - nl[1] * hw)
            if i == 0 and from_line is not None:
                la, ra = _line_hit(la, d, *from_line), _line_hit(ra, d, *from_line)
            if kind == "to" or (i == last and to_line is not None):
                cut = to_line if to_line is not None else (rec["lands"], d)
                lb, rb = _line_hit(lb, d, *cut), _line_hit(rb, d, *cut)
            if not after_corner:
                left.append(la)
                right.append(ra)
            left.append(lb)
            right.append(rb)
            after_corner = False
        elif kind == "arc":
            centre, radius, sweep = rec["centre"], rec["radius"], rec["sweep"]
            a0 = _angle((rec["from"][0] - centre[0], rec["from"][1] - centre[1]))
            r_left = radius - hw if sweep > 0 else radius + hw
            r_right = radius + hw if sweep > 0 else radius - hw
            left.extend(_arc_points(centre, r_left, a0, a0 + sweep))
            right.extend(_arc_points(centre, r_right, a0, a0 + sweep))
            after_corner = False
        elif kind == "corner":
            if left:
                left.pop()
                right.pop()
            outer, inner = rec["outer_vertex"], rec["inner_vertex"]
            left.append(inner if rec["deg"] > 0 else outer)
            right.append(outer if rec["deg"] > 0 else inner)
            after_corner = True
    return _dedupe(left), _dedupe(right)


def strip_outline(width: float, records: list[dict], from_line=None, to_line=None) -> list[Point]:
    """The closed outline of a channel: the left wall forward, the right wall back."""
    left, right = strip_walls(width, records, from_line, to_line)
    return _dedupe(left + list(reversed(right)))


# -- the wall's frame -----------------------------------------------------------------------


@dataclass
class _WallFrame:
    """A wall in the sketch frame after every reflection: `w0` its upstream end, `flow`
    the unit vector along it, `outward` the unit normal out of the host, `span` its
    length. `point(u, v)` is the WallRef frame of section 3.3. `phi` is the turn that
    makes the wall horizontal (0 when it already lies along an axis): the ops grammar's
    `from` / `to` cuts are axis-aligned lines, so an inclined wall's channel is built
    in the flat frame and turned back with a `rotate` op."""

    ref: EdgeRef
    w0: Point
    flow: Point
    outward: Point
    span: float

    @staticmethod
    def of(ref: EdgeRef, ports: list[PortIntent], flow: str | None, who: str, T: Affine,
           need_flow: bool = True) -> "_WallFrame":
        """A loop's wall needs the flow (E-FLOW on a Rect side with no inlet/outlet or
        flow=); a branch's start or end wall needs only the line, so `need_flow=False`
        there and u runs from the side's counter-clockwise start when no flow is known."""
        wl = wall_line(ref, ports=ports, flow=flow, who=who, need_flow=need_flow)
        w0 = wl.w0 if wl.w0 is not None else wl.a
        fl = wl.flow if wl.flow is not None else wl.direction
        return _WallFrame(ref, _apply(T, w0), _unit(_apply_vec(T, fl)), _unit(_apply_vec(T, wl.outward)), wl.span)

    def point(self, u: float, v: float = 0.0) -> Point:
        return (self.w0[0] + u * self.flow[0] + v * self.outward[0],
                self.w0[1] + u * self.flow[1] + v * self.outward[1])

    def uv(self, p: Point) -> Point:
        d = (p[0] - self.w0[0], p[1] - self.w0[1])
        return (_dot(d, self.flow), _dot(d, self.outward))

    @property
    def hand(self) -> float:
        """+1 when the outward side is the left of the flow (a loop turns left), else -1."""
        return 1.0 if _cross(self.flow, self.outward) > 0 else -1.0

    @property
    def aligned(self) -> bool:
        return abs(self.flow[0]) < _EPS or abs(self.flow[1]) < _EPS

    @property
    def phi(self) -> float:
        return 0.0 if self.aligned else _angle(self.flow)

    def flat(self) -> "_WallFrame":
        if self.aligned:
            return self
        T = _rotate_affine(-self.phi, self.w0)
        return _WallFrame(self.ref, self.w0, _unit(_apply_vec(T, self.flow)), _unit(_apply_vec(T, self.outward)),
                          self.span)

    def axis_value(self) -> tuple[str | None, float | None]:
        """("y", 1.5) for a wall along x, ("x", 0) for one along y, (None, None) inclined."""
        if abs(self.flow[1]) < _EPS:
            return "y", self.w0[1]
        if abs(self.flow[0]) < _EPS:
            return "x", self.w0[0]
        return None, None

    def line_text(self) -> str:
        axis, value = self.axis_value()
        return f"{axis}:{value:.15g}"

    def line(self) -> tuple[Point, Point]:
        """(point, normal) of the wall line."""
        return self.w0, self.outward

    def as_dict(self) -> dict:
        axis, value = self.axis_value()
        return {"axis": axis, "value": value, "flow": list(self.flow), "outward": list(self.outward),
                "span": self.span, "point": list(self.w0), "angle": self.phi}


def _unit(v) -> Point:
    n = math.hypot(v[0], v[1])
    return (v[0] / n, v[1] / n)


# -- the claims a Row refusal reads --------------------------------------------------------


def _claim_rows(claims) -> list[dict]:
    """The claims as dicts, from a ClaimSet, a claims dict or None (duck typed: the
    compiler imports nothing above `sketch`, D27)."""
    if claims is None:
        return []
    rows = claims.get("claims", []) if isinstance(claims, dict) else getattr(claims, "claims", [])
    out = []
    for c in rows:
        if isinstance(c, dict):
            out.append(c)
        else:
            out.append({k: getattr(c, k, None) for k in ("id", "kind", "measure", "of", "value", "min", "max")})
    return out


def _claim_value(c: dict) -> str:
    if c.get("kind") == "range":
        return f"{fmt(c.get('min'))}..{fmt(c.get('max'))}"
    return fmt(c.get("value")) if c.get("value") is not None else "?"


# -- the compiler ---------------------------------------------------------------------------


@dataclass
class _Node:
    """What compiling one feature returns: its op name, the closed-form outlines and edge
    targets beneath it (in the frame of its result: a transform above it moves them), and
    the feature keys (Passage, Serpentine, Bypass, Row) whose sketch-frame solved values --
    leg records, ends, a wall line, a step -- live in that frame and move with it."""
    name: str
    outlines: dict[str, list[list[Point]]] = field(default_factory=dict)
    targets: dict[str, dict] = field(default_factory=dict)
    passages: list[str] = field(default_factory=list)


_BYPASS_FREE = ("return_angle", "outer_radius", "leave_length", "gap")
_ANGLE_TABLE = (45.0, 60.0, 70.0, 80.0, 90.0)


class _Compiler:
    def __init__(self, sketch: Sketch | None, ports: list[PortIntent], claims, gmsh):
        self.sketch = sketch
        self.ports = ports
        self.claims = _claim_rows(claims)
        self.gmsh = gmsh
        self.ops: list[dict] = []
        self.features: dict[str, Solved] = {}
        self.instances: dict[str, dict] = {}
        self.declared_widths: list[float] = []
        self.rows: list[tuple[Row, str, Feature | None]] = []
        self.named: list[tuple[Feature, str]] = []
        self.external: dict | None = None
        self._used: set[str] = set()
        self._counts: dict[str, int] = {}
        self._base: dict[int, str] = {}
        self._suffix = ""

    # -- names ---------------------------------------------------------------------------

    def _opname(self, feature: Feature, root: bool = False) -> str:
        """The feature's own name, `fluid` for an anonymous root, else `<kind><n>`; a
        mirrored copy (`keep=True`) reuses the base its original got, plus the suffix."""
        if root and not feature.name:
            base = "fluid"
        elif feature.name:
            base = feature.name + self._suffix
        elif id(feature) in self._base:
            base = self._base[id(feature)] + self._suffix
        else:
            kind = feature.kind.lower()
            self._counts[kind] = self._counts.get(kind, 0) + 1
            self._base[id(feature)] = f"{kind}{self._counts[kind]}"
            base = self._base[id(feature)] + self._suffix
        name = base
        n = 1
        while name in self._used:
            n += 1
            name = f"{base}~{n}"
        self._used.add(name)
        return name

    def _sub(self, base: str, suffix: str) -> str:
        name = f"{base}.{suffix}"
        self._used.add(name)
        return name

    def _emit(self, op: dict) -> str:
        self.ops.append(_snapped(op))
        return op["name"]

    def _feature_key(self, feature: Feature, opname: str) -> str:
        """The key a feature's Solved, outline and edge targets live under: its name, with
        the `.mirror` suffix while compiling the kept copy of a mirror (so the copy never
        overwrites the original's entry), else its op name."""
        return (feature.name + self._suffix) if feature.name else opname

    # -- the walk ------------------------------------------------------------------------

    def visit(self, f: Feature, T: Affine = IDENTITY, root: bool = False, anchor: float | None = None) -> _Node:
        if isinstance(f, Rect):
            return self._rect(f, T, root)
        if isinstance(f, Disk):
            return self._disk(f, T, root)
        if isinstance(f, Polygon):
            return self._polygon(f, T, root)
        if isinstance(f, Band):
            return self._band(f, T, root)
        if isinstance(f, Outline):
            return self._outline(f, T, root)
        if isinstance(f, Passage):
            return self._passage(f, T, root)
        if isinstance(f, Serpentine):
            return self._serpentine(f, T, root)
        if isinstance(f, Bypass):
            return self._bypass(f, T, root, anchor)
        if isinstance(f, Row):
            return self._row(f, T, root)
        if isinstance(f, (Fuse, Cut, Intersect)):
            return self._boolean(f, T, root)
        if isinstance(f, Moved):
            return self._moved(f, T, root)
        if isinstance(f, Rotated):
            return self._rotated(f, T, root)
        if isinstance(f, Mirrored):
            return self._mirrored(f, T, root)
        if isinstance(f, BodyInBox):
            return self._body_in_box(f, T, root)
        if isinstance(f, RowInstance):
            raise SketchError("E-ARGS", label(f), "a Row instance is placed by its Row and cannot be used on its own",
                              "s.fluid = main | loops   (the Row); loops[k] names an instance in a claim or a port")
        raise SketchError("E-ARGS", label(f), f"{type(f).__name__} is not a feature the compiler knows")

    # -- primitives ----------------------------------------------------------------------

    def _rect(self, f: Rect, T: Affine, root: bool) -> _Node:
        name = self._opname(f, root)
        x0, y0 = f.origin
        x1, y1 = x0 + f.size[0], y0 + f.size[1]
        corners = [_apply(T, p) for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        bx0, by0, bx1, by1 = _bbox(corners)
        self._emit({"op": "rect", "name": name, "origin": [bx0, by0], "size": [bx1 - bx0, by1 - by0],
                    "round": f.round})
        if f.round > 0:
            r = f.round
            loop: list[Point] = []
            for (cx, cy), a0 in (((bx1 - r, by0 + r), -90), ((bx1 - r, by1 - r), 0), ((bx0 + r, by1 - r), 90),
                                 ((bx0 + r, by0 + r), 180)):
                loop.extend(_arc_points((cx, cy), r, a0, a0 + 90))
            loop = _dedupe(loop)
        else:
            loop = _ccw(corners)
        node = _Node(name)
        key = self._feature_key(f, name)
        node.outlines[key] = [loop]
        for which in ("left", "right", "top", "bottom"):
            a, b, outward = f.side_segment(which)
            node.targets[f"{key}.{which}"] = {"kind": "side", "points": [_apply(T, a), _apply(T, b)],
                                              "outward": list(_unit(_apply_vec(T, outward)))}
        self.features[key] = Solved("Rect", {"origin": list(f.origin), "size": list(f.size), "round": f.round},
                                    {"origin": [bx0, by0], "size": [bx1 - bx0, by1 - by0], "round": f.round,
                                     "centre": [(bx0 + bx1) / 2, (by0 + by1) / 2], "width": min(f.size),
                                     "length": max(f.size), "size_x": bx1 - bx0, "size_y": by1 - by0}, [name])
        self.named.append((f, key))
        return node

    def _disk(self, f: Disk, T: Affine, root: bool) -> _Node:
        name = self._opname(f, root)
        c = _apply(T, f.centre)
        self._emit({"op": "disk", "name": name, "center": [c[0], c[1]], "radius": f.radius})
        loop = _arc_points(c, f.radius, 0.0, 360.0)[:-1]
        node = _Node(name)
        key = self._feature_key(f, name)
        node.outlines[key] = [loop]
        node.targets[f"{key}.edge"] = {"kind": "loop", "points": loop, "centre": list(c), "radius": f.radius,
                                       "closed": True}
        self.features[key] = Solved("Disk", {"centre": list(f.centre), "radius": f.radius, "diameter": f.diameter},
                                    {"centre": list(c), "radius": f.radius, "diameter": f.diameter}, [name])
        self.named.append((f, key))
        return node

    def _polygon(self, f: Polygon, T: Affine, root: bool) -> _Node:
        name = self._opname(f, root)
        pts = _ccw([_apply(T, p) for p in f.points])
        self._emit({"op": "polygon", "name": name, "points": [list(p) for p in pts]})
        node = _Node(name)
        key = self._feature_key(f, name)
        node.outlines[key] = [pts]
        node.targets[f"{key}.edge"] = {"kind": "loop", "points": pts, "closed": True}
        self.features[key] = Solved("Polygon", {"points": [list(p) for p in f.points]},
                                    {"points": [list(p) for p in pts]}, [name])
        self.named.append((f, key))
        return node

    def _band(self, f: Band, T: Affine, root: bool) -> _Node:
        name = self._opname(f, root)
        c = _apply(T, f.centre)
        d0, d1 = _apply_vec(T, _dir(f.start_deg)), _apply_vec(T, _dir(f.end_deg))
        start, end = _angle(d0), _angle(d1)
        if _det(T) < 0:
            start, end = end, start
        sweep = f.end_deg - f.start_deg
        end = start + sweep
        if f.full:
            self._emit({"op": "annulus", "name": name, "center": list(c), "r_inner": f.r_inner, "r_outer": f.r_outer})
            outer = _arc_points(c, f.r_outer, 0.0, 360.0)[:-1]
            inner = list(reversed(_arc_points(c, f.r_inner, 0.0, 360.0)[:-1]))
            loops = [outer, inner]
        else:
            self._emit({"op": "band", "name": name, "center": list(c), "r_inner": f.r_inner, "r_outer": f.r_outer,
                        "start": start, "end": end})
            loops = [_dedupe(_arc_points(c, f.r_outer, start, end) + list(reversed(_arc_points(c, f.r_inner, start, end))))]
        node = _Node(name)
        key = self._feature_key(f, name)
        node.outlines[key] = loops
        node.targets[f"{key}.edge"] = {"kind": "loop", "points": loops[0], "loops": loops, "centre": list(c),
                                       "closed": True}
        self.features[key] = Solved("Band", {"centre": list(f.centre), "r_inner": f.r_inner, "r_outer": f.r_outer,
                                             "start_deg": f.start_deg, "end_deg": f.end_deg},
                                    {"centre": list(c), "r_inner": f.r_inner, "r_outer": f.r_outer,
                                     "start_deg": start, "end_deg": end}, [name])
        self.named.append((f, key))
        return node

    def _outline(self, f: Outline, T: Affine, root: bool) -> _Node:
        name = self._opname(f, root)
        mesh2d = _toolbox.load("mesh2d")
        try:
            pts = [tuple(p) for p in mesh2d.outline_points(Path(f.file), f.size, f.aoa)]
        except (SystemExit, OSError, ValueError) as err:
            raise SketchError("E-ARGS", label(f), f"the outline file {f.file!r} could not be read: {err}",
                              "an x,y outline is a csv or a Selig .dat with three or more distinct points") from None
        if T == IDENTITY:
            self._emit({"op": "outline", "name": name, "file": f.file, "size": f.size, "aoa": f.aoa})
            loop = _ccw(pts)
        else:
            loop = _ccw([_apply(T, p) for p in pts])
            self._emit({"op": "polygon", "name": name, "points": [list(p) for p in loop]})
        node = _Node(name)
        key = self._feature_key(f, name)
        node.outlines[key] = [loop]
        node.targets[f"{key}.edge"] = {"kind": "loop", "points": loop, "closed": True}
        self.features[key] = Solved("Outline", {"file": f.file, "size": f.size, "aoa": f.aoa},
                                    {"file": f.file, "size": f.size, "aoa": f.aoa, "points": len(loop)}, [name])
        self.named.append((f, key))
        return node

    # -- channels -------------------------------------------------------------------------

    def _channel_ops(self, name: str, width: float, records: list[dict], leaves: _WallFrame | None,
                     lands: _WallFrame | None, who: str) -> list[str]:
        """The `channel` ops of a passage from its leg records in the sketch frame: one per
        straight run between mitred corners (each adjacent line extended by the mitre,
        D36), fused under `name` when there are several; `from` / `to` cuts spelled on the
        axis-aligned wall lines, in the flat frame plus a `rotate` when a wall is inclined."""
        phi, pivot = 0.0, None
        for frame in (leaves, lands):
            if frame is not None and not frame.aligned:
                turn = _norm360(phi - frame.phi)
                if pivot is not None and min(turn, abs(turn - 180.0), abs(turn - 360.0)) > 1e-9:
                    raise SketchError("E-LINE-TO", who, "the wall it starts on and the wall it ends on are not parallel "
                                      "and neither lies along an axis",
                                      "Phase 1 builds a branch between inclined walls only when they are parallel; "
                                      "end_on a wall along x or y, or rotate the whole sketch afterwards")
                phi, pivot = frame.phi, frame.w0
        if pivot is not None:
            flat = _rotate_affine(-phi, pivot)
            records = _transform_records(records, flat)
            leaves = leaves.flat() if leaves is not None else None
            lands = lands.flat() if lands is not None else None
        pieces: list[dict] = []
        piece: dict | None = None
        pending = 0.0
        for rec in records:
            if rec["kind"] == "corner":
                e = rec["extend"]
                piece["path"][-1] = {"line": piece["path"][-1]["line"] + e}
                pieces.append(piece)
                h = rec["heading_out"]
                d = _dir(h)
                piece = {"start": (rec["at"][0] - e * d[0], rec["at"][1] - e * d[1]), "heading": h, "path": []}
                pending = e
                continue
            if piece is None:
                piece = {"start": rec["from"], "heading": rec["heading"], "path": []}
            if rec["kind"] == "line":
                piece["path"].append({"line": rec["length"] + pending})
            elif rec["kind"] == "arc":
                piece["path"].append({"arc": {"radius": rec["radius"], "angle": rec["sweep"]}})
            else:
                piece["path"].append({"line": {"to": lands.line_text()}})
            pending = 0.0
        pieces.append(piece)
        names = []
        several = len(pieces) > 1
        for i, pc in enumerate(pieces):
            pname = self._sub(name, str(i + 1)) if several else (self._sub(name, "flat") if pivot is not None else name)
            op = {"op": "channel", "name": pname, "width": width, "start": list(pc["start"]), "heading": pc["heading"],
                  "path": pc["path"]}
            if i == 0 and leaves is not None:
                op["from"] = leaves.line_text()
            self._emit(op)
            names.append(pname)
        if several:
            fused = self._sub(name, "flat") if pivot is not None else name
            self._emit({"op": "fuse", "name": fused, "of": names})
            names.append(fused)
        if pivot is not None:
            self._emit({"op": "rotate", "name": name, "target": names[-1], "angle": phi, "about": list(pivot)})
            names.append(name)
        return names

    def _channel_targets(self, f, key: str, width: float, records: list[dict], leaves, lands,
                         node: _Node, T: Affine) -> None:
        from_line = leaves.line() if leaves is not None else None
        to_line = lands.line() if lands is not None else None
        left, right = strip_walls(width, records, from_line, to_line)
        node.outlines[key] = [_dedupe(left + list(reversed(right)))]
        node.targets[f"{key}.start"] = {"kind": "cap", "points": [right[0], left[0]], "width": width}
        node.targets[f"{key}.end"] = {"kind": "cap", "points": [left[-1], right[-1]], "width": width}
        runs = f._runs()
        for run in runs:
            for which in ("left", "right", "top", "bottom"):
                try:
                    wl = wall_line(EdgeRef(f, f"side:{which}", run[0] + 1), ports=self.ports, need_flow=False)
                except SketchError:
                    continue
                target = {"kind": "side", "points": [_apply(T, wl.a), _apply(T, wl.b)],
                          "outward": list(_unit(_apply_vec(T, wl.outward)))}
                for leg in run:
                    node.targets[f"{key}.{which}[{leg + 1}]"] = target
                if len(runs) == 1:
                    node.targets[f"{key}.{which}"] = target

    def _passage(self, f: Passage, T: Affine, root: bool) -> _Node:
        f.check()
        name = self._opname(f, root)
        key = self._feature_key(f, name)
        who = label(f)
        leaves = _WallFrame.of(f.leaves, self.ports, None, who, T, need_flow=False) if f.leaves is not None else None
        lands = _WallFrame.of(f.lands_on, self.ports, None, who, T, need_flow=False) if f.lands_on is not None else None
        tr = f.trace(self.ports)
        records = _transform_records(tr["records"], T)
        ops = self._channel_ops(name, f.width, records, leaves, lands, who)
        node = _Node(name, passages=[key])
        self._channel_targets(f, key, f.width, records, leaves, lands, node, T)
        start = [str(f.leaves), f.start_u] if f.leaves is not None else list(f.start_point)
        self.features[key] = Solved(
            "Passage", {"width": f.width, "start": start, "heading": f.heading, "legs": f.legs},
            {"legs": records, "length": tr["length"], "end": list(_apply(T, tr["end"])),
             "end_heading": _heading_under(T, tr["end_heading"]),
             "leaves": None if f.leaves is None else str(f.leaves),
             "lands_on": None if f.lands_on is None else str(f.lands_on)}, ops)
        self.declared_widths.append(f.width)
        self.named.append((f, key))
        return node

    def _serpentine(self, f: Serpentine, T: Affine, root: bool) -> _Node:
        name = self._opname(f, root)
        key = self._feature_key(f, name)
        tr = f.trace()
        records = _transform_records(tr["records"], T)
        ops = self._channel_ops(name, f.width, records, None, None, label(f))
        node = _Node(name, passages=[key])
        self._channel_targets(f, key, f.width, records, None, None, node, T)
        self.features[key] = Solved(
            "Serpentine", {"width": f.width, "passes": f.passes, "pass_length": f.pass_length,
                           "bend_radius": f.bend_radius, "start": list(f.start_point), "heading": f.heading,
                           "stack": f.stack},
            {"pass_pitch": f.pass_pitch, "wall_between": f.wall_between, "bends": f.bends,
             "end": list(_apply(T, tr["end"])), "end_heading": _heading_under(T, tr["end_heading"]),
             "ends_on_start_side": f.ends_on_start_side, "pass_centrelines": list(f.pass_centrelines)}, ops)
        self.declared_widths.append(f.width)
        self.named.append((f, key))
        return node

    # -- the bypass loop ------------------------------------------------------------------

    def _bypass(self, f: Bypass, T: Affine, root: bool, anchor: float | None) -> _Node:
        who = label(f)
        frame = _WallFrame.of(f.wall, self.ports, f.flow, who, T)
        cf = f.closed_form()
        w = f.width
        if anchor is None:
            if f.at is None:
                up = f.wall.feature.name or "the wall"
                raise SketchError("E-BYPASS-AT", who, f"outside a Row needs at=<u along {f.wall}, 0 at the "
                                  f"inlet end, {fmt(frame.span)} at the outlet end>",
                                  f"Bypass(..., at=30) leaves {f.wall} 30 along it; a Row places its instances itself "
                                  f"(the wall belongs to {up})")
            anchor = f.at
            landing_u = anchor + cf["landing"][0]
            if not -_EPS <= landing_u <= frame.span + _EPS:
                before = landing_u < 0
                steeper = f.closed_form(return_angle=80.0)["lands_upstream_by"] if f.return_angle < 80 else None
                fixes = [f"at >= {fmt(cf['lands_upstream_by'])}"]
                if steeper is not None:
                    fixes.append(f"a steeper return_angle (80 returns {fmt(steeper)} upstream)")
                fixes.append("a Row, which places it")
                raise SketchError(
                    "E-LAND-OFF-WALL", f"{who} at {fmt(anchor)} on {f.wall}",
                    f"it would land at u = {fmt(landing_u)}, "
                    + (f"before the wall's start (0)" if before else f"past the wall's end ({fmt(frame.span)})"),
                    f"the loop returns {fmt(cf['lands_upstream_by'])} upstream of its anchor (leave_angle "
                    f"{fmt(f.leave_angle)}, leave_length {fmt(f.leave_length)}, outer_radius {fmt(f.outer_radius)}, "
                    f"return_angle {fmt(f.return_angle)});",
                    ", or ".join(fixes),
                    numbers={"landing_u": landing_u, "lands_upstream_by": cf["lands_upstream_by"], "span": frame.span})
        elif f.at is not None:
            raise SketchError("E-ROW-ITEM-AT", who, f"{who} gives at={fmt(f.at)}, but the Row places its instances",
                              "drop at= from the Bypass; Row(start=...) fixes the first anchor")
        name = self._opname(f, root)
        key = self._feature_key(f, name)
        flat = frame.flat()
        # the raw channel in the flat frame: heading tL from the flow toward the outward side
        tL, tR = f.leave_angle, f.return_angle
        d_leave = (math.cos(math.radians(tL)) * flat.flow[0] + math.sin(math.radians(tL)) * flat.outward[0],
                   math.cos(math.radians(tL)) * flat.flow[1] + math.sin(math.radians(tL)) * flat.outward[1])
        raw_name = self._sub(name, "raw")
        clip_name = self._sub(name, "clip")
        turned = not frame.aligned
        raw_op = {"op": "channel", "name": self._sub(raw_name, "flat") if turned else raw_name, "width": w,
                  "start": list(flat.point(anchor, 0.0)), "heading": _angle(d_leave), "from": flat.line_text(),
                  "path": [{"line": cf["leave_length"]},
                           {"arc": {"radius": cf["R"], "angle": flat.hand * cf["sweep"]}},
                           {"line": {"to": flat.line_text()}}]}
        self._emit(raw_op)
        ops = [raw_op["name"]]
        if turned:
            self._emit({"op": "rotate", "name": raw_name, "target": raw_op["name"], "angle": frame.phi,
                        "about": list(frame.w0)})
            ops.append(raw_name)
        lo, hi = cf["footprint"]
        reach = hi - lo
        corners = [flat.point(u, v) for u in (-reach, frame.span + reach) for v in (0.0, cf["height"] + reach)]
        cx0, cy0, cx1, cy1 = _bbox(corners)
        clip_op = {"op": "rect", "name": self._sub(clip_name, "flat") if turned else clip_name,
                   "origin": [cx0, cy0], "size": [cx1 - cx0, cy1 - cy0]}
        self._emit(clip_op)
        ops.append(clip_op["name"])
        if turned:
            self._emit({"op": "rotate", "name": clip_name, "target": clip_op["name"], "angle": frame.phi,
                        "about": list(frame.w0)})
            ops.append(clip_name)
        self._emit({"op": "intersect", "name": name, "of": [raw_name, clip_name]})
        ops.append(name)
        # the closed-form outline in the wall's frame, clipped to the outward side, mapped to the sketch
        u0 = anchor
        P1 = (u0 + cf["P1"][0], cf["P1"][1])
        C = (u0 + cf["C"][0], cf["C"][1])
        P2 = (u0 + cf["P2"][0], cf["P2"][1])
        landing = (u0 + cf["landing"][0], 0.0)
        heading_out = 180.0 + tR
        records_uv = [
            {"kind": "line", "from": (u0, 0.0), "to": P1, "heading": tL, "heading_out": tL, "length": cf["leave_length"]},
            {"kind": "arc", "from": P1, "to": P2, "heading": tL, "heading_out": heading_out, "centre": C,
             "radius": cf["R"], "sweep": cf["sweep"]},
            {"kind": "to", "from": P2, "to": landing, "lands": landing, "heading": heading_out,
             "heading_out": heading_out, "length": cf["return_len"], "line": "v=0"},
        ]
        wall_uv = ((0.0, 0.0), (0.0, 1.0))
        left, right = strip_walls(w, records_uv, wall_uv, wall_uv)
        loop_uv = _clip_half_plane(_dedupe(left + list(reversed(right))), *wall_uv)
        loop = [frame.point(u, v) for u, v in loop_uv]
        node = _Node(name, passages=[key])
        node.outlines[key] = [loop]
        node.targets[f"{key}.start"] = {"kind": "cap", "points": [frame.point(*right[0]), frame.point(*left[0])],
                                        "width": w}
        node.targets[f"{key}.end"] = {"kind": "cap", "points": [frame.point(*left[-1]), frame.point(*right[-1])],
                                      "width": w}
        wall_dict = frame.as_dict()
        self.features[key] = Solved(
            "Bypass", {"wall": str(f.wall), "width": w, "leave_angle": tL, "outer_radius": f.outer_radius,
                       "return_angle": tR, "at": anchor, "leave_length": f.leave_length_given, "flow": f.flow},
            {"R": cf["R"], "sweep": cf["sweep"], "P1": list(cf["P1"]), "C": list(cf["C"]), "P2": list(cf["P2"]),
             "landing": list(cf["landing"]), "return_len": cf["return_len"], "lip_u": cf["lip_u"],
             "lip_deg": cf["lip_deg"], "footprint": [lo, hi], "height": cf["height"], "wall_line": wall_dict,
             "theta_leave": tL, "theta_return": tR, "lands_on": str(f.wall), "leave_length": cf["leave_length"],
             "leave_length_default": f.leave_length_default}, ops)
        self.declared_widths.append(w)
        self.named.append((f, key))
        return node

    # -- rows -----------------------------------------------------------------------------

    def _row(self, f: Row, T: Affine, root: bool) -> _Node:
        who = label(f)
        item = f.item
        if isinstance(item, (Row, RowInstance, BodyInBox)):
            raise SketchError("E-ARGS", who, f"a Row repeats a primitive or a loop, not a {item.kind}")
        if isinstance(item, Bypass) and item.at is not None:
            raise SketchError("E-ROW-ITEM-AT", who, f"its item {label(item)} gives at={fmt(item.at)}, but the Row places "
                              "its instances", "drop at= from the Bypass; Row(start=6) fixes the first anchor"
                              .replace("start=6", f"start={fmt(item.at)}"))
        width = item.width
        margin = f.margin if f.margin is not None else (width / 2 if width else 0.0)
        # the direction and the frame the row runs along
        frame: _WallFrame | None = None
        direction: Point | None = None
        if isinstance(item, Bypass):
            if f.along is not None and not isinstance(f.along, EdgeRef):
                raise SketchError("E-ROW-ALONG", who, f"a row of {label(item)} runs along its wall {item.wall}, not a "
                                  f"direction", f"drop along=, or along={item.wall}")
            if isinstance(f.along, EdgeRef) and (f.along.feature is not item.wall.feature or f.along.which != item.wall.which):
                raise SketchError("E-ROW-ALONG", who, f"the {label(item)} leaves {item.wall} but along= names {f.along}",
                                  f"drop along= (the wall is inferred from the Bypass), or along={item.wall}")
            frame = _WallFrame.of(item.wall, self.ports, item.flow, label(item), T)
        elif f.along is None:
            raise SketchError("E-ROW-ALONG", who, f"which way does the row run? the item is a {item.kind}, not a "
                              "Bypass on a wall",
                              "along=main.top (start/align/margin apply), or along=(1, 0) with gap= (the item as "
                              "drawn is the first)")
        elif isinstance(f.along, EdgeRef):
            frame = _WallFrame.of(f.along, self.ports, None, who, T)
        else:
            direction = _unit(_apply_vec(T, f.along))
            along_text = f"({fmt(f.along[0])}, {fmt(f.along[1])})"
            if f.gap is None:
                raise SketchError("E-ROW-GAP-REQUIRED", who, f"{who} along {along_text}: a row along a direction has no "
                                  "wall to fit; give gap=", f"Row(item, count={f.count}, gap=<distance between "
                                  f"instances>, along={along_text})")
            if f.start is not None or f.margin is not None or f.align != "centre":
                first = _bbox(self._item_outline(item, T)) if not isinstance(item, Bypass) else None
                at = (f"at ({fmt((first[0] + first[2]) / 2)}, {fmt((first[1] + first[3]) / 2)})" if first else "")
                raise SketchError("E-ROW-DIRECTION", who, f"{who} along {along_text}: start/align/margin are placed "
                                  "against a wall; a row along a",
                                  f"direction is placed by its item (the first instance is the {item.kind} as drawn "
                                  f"{at})".strip())
        # the footprint along the row and the solve
        if isinstance(item, Bypass):
            cf = item.closed_form()
            lo, hi = cf["footprint"]
            height = cf["height"]
        else:
            pts = self._item_outline(item, T)
            axis = frame.flow if frame is not None else direction
            origin = frame.w0 if frame is not None else (0.0, 0.0)
            proj = [_dot((p[0] - origin[0], p[1] - origin[1]), axis) for p in pts]
            across = [_cross(axis, (p[0] - origin[0], p[1] - origin[1])) for p in pts]
            lo, hi = min(proj), max(proj)
            height = max(across) - min(across)
        footprint = hi - lo
        count = f.count
        gap_declared = f.gap is not None
        table: dict = {}
        free = None
        if frame is not None:
            span = frame.span
            available = span - 2 * margin
            needed = count * footprint + (count - 1) * (f.gap or 0.0)
            floor = width / 10 if width else 0.0
            if f.gap is None:
                gap = (available - count * footprint) / (count - 1) if count > 1 else 0.0
                if (count > 1 and gap < floor - _EPS) or (count == 1 and footprint > available + _EPS):
                    self._refuse_fit(f, item, frame, footprint, lo, hi, margin, available, needed, floor, width)
            else:
                gap = f.gap
                if needed > available + _EPS:
                    self._refuse_fit(f, item, frame, footprint, lo, hi, margin, available, needed, floor, width)
            if isinstance(item, Bypass):
                free, table = self._free_parameter(f, item, frame)
        else:
            span = None
            available = None
            gap = f.gap
        if f.pitch is not None and f.gap is not None and f.pitch < footprint + f.gap - _EPS:
            raise SketchError("E-ROW-PITCH", who, f"pitch {fmt(f.pitch)} is smaller than footprint {fmt(footprint)} + gap "
                              f"{fmt(f.gap)} = {fmt(footprint + f.gap)}",
                              f"pitch >= {fmt(footprint + f.gap)}, or gap <= {fmt(f.pitch - footprint)}, or leave pitch out "
                              "and the gap sets it", partial=item,
                              numbers={"pitch": f.pitch, "footprint": footprint, "gap": f.gap})
        pitch = f.pitch if f.pitch is not None else footprint + gap
        if f.pitch is not None and f.gap is None:
            gap = pitch - footprint
        # placement: an anchor is the u a Bypass hangs from (its footprint lo..hi is relative
        # to it); for any other item the anchor is its footprint's centre along the wall
        # (lo..hi are where the item is drawn), so `start=` fixes that centre
        total = count * footprint + (count - 1) * (pitch - footprint)
        anchor_lo = lo if isinstance(item, Bypass) else -footprint / 2
        if frame is not None:
            if f.start is not None:
                first_lo = f.start + anchor_lo
            elif f.align == "start":
                first_lo = margin
            elif f.align == "end":
                first_lo = span - margin - total
            else:
                first_lo = (span - total) / 2
            anchor0 = first_lo - anchor_lo
            anchors = [anchor0 + k * pitch for k in range(count)]
            row_span = (first_lo, first_lo + total)
            step = (pitch * frame.flow[0], pitch * frame.flow[1])
        else:
            anchors = [(lo + hi) / 2 + k * pitch for k in range(count)]
            row_span = (lo, lo + total)
            step = (pitch * direction[0], pitch * direction[1])
        # the first instance
        name = self._opname(f, root)
        key = self._feature_key(f, name)
        if isinstance(item, Bypass):
            node = self.visit(item, T, anchor=anchors[0])
            first_name = node.name
        else:
            node = self.visit(item, T)
            first_name = node.name
            if frame is not None:
                shift = anchors[0] - (lo + hi) / 2
                if abs(shift) > _EPS:
                    moved = self._sub(name, "first")
                    self._emit({"op": "translate", "name": moved, "target": node.name,
                                "by": [shift * frame.flow[0], shift * frame.flow[1]]})
                    node = self._shift_node(node, _translate_affine(shift * frame.flow[0], shift * frame.flow[1]))
                    first_name = moved
        if count > 1:
            self._emit({"op": "repeat", "name": name, "target": first_name, "count": count, "step": list(step)})
        else:
            self._emit({"op": "translate", "name": name, "target": first_name, "by": [0.0, 0.0]})
        item_key = node.name if node.name in node.outlines else next(iter(node.outlines), None)
        base_loops = node.outlines.get(item_key, [])
        for k in range(count):
            shift = _translate_affine(k * step[0], k * step[1])
            node.outlines[f"{key}[{k}]"] = [[_apply(shift, p) for p in loop] for loop in base_loops]
        along = str(item.wall) if isinstance(item, Bypass) else (str(f.along) if isinstance(f.along, EdgeRef)
                                                                  else [f.along[0], f.along[1]])
        self.features[key] = Solved(
            "Row", {"count": count, "gap": f.gap, "pitch": f.pitch, "along": along, "start": f.start,
                    "align": f.align, "margin": f.margin, "item": item.name},
            {"pitch": pitch, "gap": gap, "gap_declared": gap_declared, "footprint": [lo, hi], "anchors": anchors,
             "span": list(row_span), "margin": margin, "along": along, "free_parameter": free, "table": table},
            [name] + ([f"{name}.first"] if f"{name}.first" in self._used else []))
        self.instances[key] = {"count": count, "step": list(step), "gap": gap, "gap_declared": gap_declared,
                               "footprint": [footprint, height]}
        host = frame.ref.feature if frame is not None else None
        self.rows.append((f, key, host))
        node.name = name
        node.passages.append(key)
        return node

    def _item_outline(self, item: Feature, T: Affine) -> list[Point]:
        """The closed-form outline of a Row's item as drawn (a scratch compile with its
        ops discarded): what a non-Bypass item's footprint along the row is measured on."""
        scratch = _Compiler(self.sketch, self.ports, None, self.gmsh)
        scratch._used = set(self._used)
        node = scratch.visit(item, T)
        pts = [p for loops in node.outlines.values() for loop in loops for p in loop]
        if not pts:
            raise SketchError("E-ARGS", label(item), "the row's item has no outline to measure a footprint on")
        return pts

    @staticmethod
    def _shift_node(node: _Node, T: Affine) -> _Node:
        """The node's outlines and edge targets under an affine: points moved, a side's
        outward normal turned with them."""
        node.outlines = {k: [[_apply(T, p) for p in loop] for loop in loops] for k, loops in node.outlines.items()}
        targets = {}
        for k, t in node.targets.items():
            moved = dict(t, points=[_apply(T, p) for p in t["points"]])
            if t.get("outward") is not None:
                moved["outward"] = list(_unit(_apply_vec(T, t["outward"])))
            if t.get("centre") is not None:
                moved["centre"] = list(_apply(T, t["centre"]))
            targets[k] = moved
        node.targets = targets
        return node

    def _free_parameter(self, row: Row, item: Bypass, frame: _WallFrame) -> tuple[str, dict]:
        """The first of (return_angle, outer_radius, leave_length, gap) no measure/range
        claim fixes on the item, and the item's footprint at five values of it."""
        fixed = self._fixed_by_claims(row, item, frame)
        free = next((p for p in _BYPASS_FREE if p not in fixed), "return_angle")
        table: dict[str, float] = {}
        for value in self._table_values(free, item, row):
            if free == "gap":
                table[fmt(value)] = item.closed_form()["footprint"][1] - item.closed_form()["footprint"][0]
            else:
                lo, hi = item.closed_form(**{free: value})["footprint"]
                table[fmt(value)] = hi - lo
        return free, table

    @staticmethod
    def _table_values(free: str, item: Bypass, row: Row) -> tuple[float, ...]:
        w = item.width
        if free == "return_angle":
            return _ANGLE_TABLE
        if free == "outer_radius":
            return tuple(max(1.1 * w, item.outer_radius * k) for k in (0.6, 0.7, 0.8, 0.9, 1.0))
        if free == "leave_length":
            return tuple(w * k for k in (0.25, 0.5, 1.0, 1.5, 2.0))
        return tuple(w * k for k in (0.1, 0.25, 0.5, 0.75, 1.0))

    def _fixed_by_claims(self, row: Row, item: Bypass, frame: _WallFrame) -> dict[str, str]:
        """parameter -> "outer_radius 6 (c5)" for every claim that fixes one: the item's
        measures (on the item, the row, `row[*]` or `row[k]`), the row's count, gap or
        pitch, and the host wall's length."""
        names = {n for n in (item.name, row.name) if n}
        if row.name:
            names.add(f"{row.name}[*]")
        host = frame.ref.feature.name
        fixed: dict[str, str] = {}
        for c in self.claims:
            of = str(c.get("of") or "")
            kind = c.get("kind")
            measure = c.get("measure") or ""
            cid = c.get("id", "?")
            on_row = of in names or (row.name and of.startswith(f"{row.name}["))
            if kind == "count" and row.name and of == row.name:
                fixed.setdefault("count", f"count {_claim_value(c)} ({cid})")
            elif kind in ("measure", "range") and on_row and measure in ("outer_radius", "leave_angle", "return_angle",
                                                                            "leave_length", "gap", "pitch"):
                fixed.setdefault(measure, f"{measure} {_claim_value(c)} ({cid})")
            elif kind in ("measure", "range") and host and of == host and measure == "length":
                fixed.setdefault("wall", f"wall {_claim_value(c)} ({cid})")
        return fixed

    def _refuse_fit(self, row: Row, item: Feature, frame: _WallFrame, footprint: float, lo: float, hi: float,
                    margin: float, available: float, needed: float, floor: float, width: float | None) -> None:
        who = label(row)
        count = row.count
        span = frame.span
        head = f"{count} x {label(item)} do not fit on {frame.ref} ({fmt(span)} long)"
        lines = [f"each instance spans {fmt(footprint)} along the wall"
                 + (f" ({fmt(-lo)} upstream of its anchor to {fmt(hi)} downstream)" if isinstance(item, Bypass)
                    else f" (drawn at u {fmt(lo)}..{fmt(hi)})")]
        if row.gap is None:
            lines.append(f"{count} x {fmt(footprint)} = {fmt(count * footprint)} needed before any gap; {fmt(span)} - 2 x "
                         f"margin {fmt(margin)} = {fmt(available)} available")
        else:
            lines.append(f"{count} x {fmt(footprint)} + {count - 1} x gap {fmt(row.gap)} = {fmt(needed)} needed; "
                         f"{fmt(span)} - 2 x margin {fmt(margin)} = {fmt(available)} available")
        numbers: dict = {"count": count, "footprint": footprint, "lo": lo, "hi": hi, "needed": needed,
                         "available": available, "span": span, "margin": margin, "floor": floor}
        if isinstance(item, Bypass):
            free, table = self._free_parameter(row, item, frame)
            fixed = self._fixed_by_claims(row, item, frame)
            kept = [p for p in ("outer_radius", "leave_angle", "leave_length", "return_angle") if p != free]
            kept_text = ", ".join(item.leave_length_text() if p == "leave_length" else f"{p} {fmt(getattr(item, p))}"
                                  for p in kept)
            lines.append(f"the footprint is set by {free} ({kept_text} kept):")
            values = list(table)
            fits = []
            for v in values:
                fp = table[v]
                if free == "gap":
                    g = float(v)
                    ok = count * fp + (count - 1) * g <= available + _EPS
                    fits.append(fmt(g) if ok else "--")
                else:
                    g = (available - count * fp) / (count - 1) if count > 1 else available - fp
                    fits.append(f"{g:.2f}" if g >= floor - _EPS else "--")
            lines.append("  " + f"{free}".ljust(14) + " ".join(v.ljust(5) for v in values).rstrip())
            lines.append("  " + "footprint".ljust(14) + " ".join(f"{table[v]:.2f}".ljust(5) for v in values).rstrip())
            lines.append("  " + f"{count} fit at gap".ljust(14) + " ".join(g.ljust(5) for g in fits).rstrip())
            read = [fixed[k] for k in ("count", "outer_radius", "leave_angle", "leave_length", "return_angle",
                                       "gap", "pitch", "wall") if k in fixed]
            if read:
                lines.append(f"fixed by the request: {', '.join(read)}; {free} is free (no claim fixes it)")
            else:
                lines.append(f"fixed by the request: no claims fix the others; {free} is free")
            lines.append("a smaller outer_radius, fewer loops, or a longer wall would also fit")
            numbers.update({"free_parameter": free, "table": table, "fits": dict(zip(values, fits))})
        else:
            lines.append("fewer instances, a smaller item, a smaller gap, or a longer wall would fit")
        raise SketchError("E-ROW-FIT", who, head, *lines, partial=item, numbers=numbers)

    # -- booleans and transforms ----------------------------------------------------------

    def _boolean(self, f, T: Affine, root: bool) -> _Node:
        a = self.visit(f.a, T)
        b = self.visit(f.b, T)
        name = self._opname(f, root)
        if isinstance(f, Cut):
            self._emit({"op": "cut", "name": name, "from": a.name, "take": [b.name]})
        else:
            self._emit({"op": "fuse" if isinstance(f, Fuse) else "intersect", "name": name, "of": [a.name, b.name]})
        node = _Node(name, {**a.outlines, **b.outlines}, {**a.targets, **b.targets}, a.passages + b.passages)
        key = self._feature_key(f, name)
        self.features[key] = Solved(f.kind, {"a": a.name, "b": b.name}, {"of": [a.name, b.name]}, [name])
        return node

    def _moved(self, f: Moved, T: Affine, root: bool) -> _Node:
        inner = self.visit(f.of, T)
        by = _apply_vec(T, (f.dx, f.dy))
        name = self._opname(f, root)
        self._emit({"op": "translate", "name": name, "target": inner.name, "by": list(by)})
        return self._after_transform(inner, name, _translate_affine(*by), 0.0)

    def _rotated(self, f: Rotated, T: Affine, root: bool) -> _Node:
        inner = self.visit(f.of, T)
        deg = -f.deg if _det(T) < 0 else f.deg
        if f.about is not None:
            about = _apply(T, f.about)
        else:
            pts = [p for loops in inner.outlines.values() for loop in loops for p in loop]
            x0, y0, x1, y1 = _bbox(pts)
            about = ((x0 + x1) / 2, (y0 + y1) / 2)
        name = self._opname(f, root)
        self._emit({"op": "rotate", "name": name, "target": inner.name, "angle": deg, "about": list(about)})
        return self._after_transform(inner, name, _rotate_affine(deg, about), deg)

    def _after_transform(self, inner: _Node, name: str, T: Affine, deg: float) -> _Node:
        """A `translate` / `rotate` op moves what was compiled beneath it, so every solved
        value that lives in the sketch frame follows: a Passage's leg records, end and
        heading; a Serpentine's end and heading; a Bypass's wall line; a Row's step. The
        values in a wall's own frame (a Bypass's P1, C, landing, footprint; a Row's anchors)
        are unchanged by construction, and so are a Serpentine's `pass_centrelines`: they
        are each pass's offset along the stack direction from the start, which a rigid
        transform or a mirror moves with the passes (measure reads the same offsets off
        the built legs, so `stacked_in` judges one number either way)."""
        node = self._shift_node(inner, T)
        for key in node.passages:
            feature = self.features[key]
            solved = feature.solved
            if feature.kind == "Bypass":
                wl = solved["wall_line"]
                frame = _WallFrame(None, _apply(T, wl["point"]), _unit(_apply_vec(T, wl["flow"])),
                                   _unit(_apply_vec(T, wl["outward"])), wl["span"])
                solved["wall_line"] = frame.as_dict()
                continue
            if feature.kind == "Row":
                step = self.instances[key]["step"]
                self.instances[key]["step"] = list(_apply_vec(T, step))
                continue
            if "legs" in solved:
                solved["legs"] = _transform_records(solved["legs"], T)
            solved["end"] = list(_apply(T, solved["end"]))
            solved["end_heading"] = solved["end_heading"] + deg
        node.name = name
        return node

    def _mirrored(self, f: Mirrored, T: Affine, root: bool) -> _Node:
        M = _compose(T, _mirror_affine(f.axis, f.at))
        if not f.keep:
            node = self.visit(f.of, M, root)
            return node
        original = self.visit(f.of, T)
        saved = self._suffix
        self._suffix = saved + ".mirror"
        copy = self.visit(f.of, M)
        self._suffix = saved
        name = self._opname(f, root)
        self._emit({"op": "fuse", "name": name, "of": [original.name, copy.name]})
        outlines = dict(original.outlines)
        outlines.update(copy.outlines)
        return _Node(name, outlines, dict(original.targets), original.passages + copy.passages)

    def _body_in_box(self, f: BodyInBox, T: Affine, root: bool) -> _Node:
        body = self.visit(f.body, T)
        name = self._opname(f, root)
        self._emit({"op": "translate", "name": name, "target": body.name, "by": [0.0, 0.0]})
        key = self._feature_key(f, name)
        pts = [p for loops in body.outlines.values() for loop in loops for p in loop]
        x0, y0, x1, y1 = _bbox(pts)
        L = x1 - x0
        box = (x0 - f.ahead * L, y0 - f.below * L, x1 + f.behind * L, y1 + f.above * L)
        node = _Node(name, dict(body.outlines), dict(body.targets), body.passages)
        bx0, by0, bx1, by1 = box
        node.targets[f"{key}.inlet"] = {"kind": "box", "side": "inlet", "points": [(bx0, by0), (bx0, by1)], "box": list(box)}
        node.targets[f"{key}.outlet"] = {"kind": "box", "side": "outlet", "points": [(bx1, by0), (bx1, by1)],
                                         "box": list(box)}
        node.targets[f"{key}.farfield"] = {"kind": "box", "side": "farfield",
                                           "points": [(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)], "box": list(box)}
        body_loops = [loop for loops in body.outlines.values() for loop in loops]
        node.targets[f"{key}.body"] = {"kind": "loop", "points": body_loops[0] if body_loops else [],
                                       "loops": body_loops, "closed": True}
        self.external = {"ahead": f.ahead, "behind": f.behind, "above": f.above, "below": f.below, "far": f.far,
                         "body": body.name}
        self.features[key] = Solved("BodyInBox", {"ahead": f.ahead, "behind": f.behind, "above": f.above,
                                                  "below": f.below, "far": f.far, "body": body.name},
                                    {"box": list(box), "body_extent": [x1 - x0, y1 - y0]}, [name])
        self.named.append((f, key))
        return node

    # -- the plan -------------------------------------------------------------------------

    def finish(self, root: _Node, sketch: Sketch | None, units: str, scale: float) -> Plan:
        if root.name != "fluid":
            self._emit({"op": "translate", "name": "fluid", "target": root.name, "by": [0.0, 0.0]})
        ports = list(sketch.ports) if sketch is not None else []
        rules, targets = self._ports(ports, root)
        apart = self._apart(sketch)
        clip_boxes = [(key, _bbox([p for loop in loops for p in loop]))
                      for key, loops in root.outlines.items()
                      if loops and (self.features.get(key, Solved("", {}, {}, [])).kind == "Bypass"
                                    or (key.endswith("]") and key.rsplit("[", 1)[0] in self.instances))]
        return Plan(units=units, scale=scale, ops=list(self.ops), rules=rules, ports=ports, features=self.features,
                    instances=self.instances, outlines=_jsonable(root.outlines), edge_targets=targets, apart=apart,
                    declared_widths=list(self.declared_widths),
                    expected=tuple(sketch.expected) if sketch is not None else (1, 1),
                    notes=list(sketch.notes) if sketch is not None else [], clip_boxes=clip_boxes,
                    external=self.external)

    def _ports(self, ports: list[PortIntent], root: _Node) -> tuple[list[dict], dict[str, dict]]:
        rules: list[dict] = []
        targets: dict[str, dict] = {}
        for port in ports:
            for edge in port.edges:
                if isinstance(edge, str):
                    rule = _rule_from_string(port, edge)
                    if rule is not None:
                        rules.append(rule)
                    continue
                key = str(edge)
                target = root.targets.get(key)
                if target is None:
                    in_fluid = any(k.split(".")[0] == (edge.feature.name or "") for k in root.targets)
                    what = (f"{label(edge.feature)} has no edge {key.split('.', 1)[-1]!r}" if in_fluid
                            else f"{key} names {label(edge.feature)}, which is not part of s.fluid")
                    raise SketchError("E-PORT-MISSING", port.name, what,
                                      "a port is an edge of a feature in s.fluid: main.start, duct.left, cyl.edge; "
                                      f"edges here: {', '.join(sorted(root.targets)) or 'none'}")
                targets[key] = _jsonable(dict(target, port=port.name, feature=edge.feature.name))
                rules.append(_provisional_rule(port, target))
        return rules, targets

    def _apart(self, sketch: Sketch | None) -> list[tuple[str, str, float | None]]:
        pairs: list[tuple[str, str, float | None]] = []
        seen: set[tuple[str, str]] = set()

        def add(a: str, b: str, gap):
            k = (a, b) if a <= b else (b, a)
            if a != b and k not in seen:
                seen.add(k)
                pairs.append((k[0], k[1], gap))

        # a feature inside a mirrored(keep=True) subtree is compiled twice (under its
        # name and under <name>.mirror), so s.apart(a, b) names every copy of each
        keys_of: dict[int, list[str]] = {}
        for f, k in self.named:
            keys_of.setdefault(id(f), []).append(k)
        for row, key, host in self.rows:
            keys_of.setdefault(id(row), []).append(key)
        if sketch is not None:
            for a, b, gap in sketch._apart:
                ka, kb = keys_of.get(id(a)), keys_of.get(id(b))
                if not ka or not kb:
                    missing = a if not ka else b
                    raise SketchError("E-ARGS", "Sketch", f"s.apart names {label(missing)}, which is not part of s.fluid",
                                      "apart parts are features of the fluid: s.apart(top_loops, bottom_loops)")
                for key_a in ka:
                    for key_b in kb:
                        add(key_a, key_b, gap)
        items = {id(row.item) for row, _, _ in self.rows}
        for row, key, host in self.rows:
            for other, other_key, other_host in self.rows:
                # by key, not identity: a kept mirror compiles the same Row twice
                if other_key != key:
                    add(key, other_key, None)
            for feature, fkey in self.named:
                # a Row's item is the template its instances repeat, not a part beside them
                if id(feature) in items or feature is host:
                    continue
                add(key, fkey, None)
        return pairs


def _rule_from_string(port: PortIntent, text: str) -> dict | None:
    if text.startswith("box:"):
        try:
            box = [float(v) for v in text[4:].replace(";", ",").split(",")]
        except ValueError:
            box = []
        if len(box) != 4:
            raise SketchError("E-ARGS", port.name, f"the rule {text!r} is box:x0,y0,x1,y1")
        return {"name": port.name, "kind": port.kind, "box": box}
    if text.startswith("normal:"):
        return None
    return {"name": port.name, "kind": port.kind, "at": text}


def _provisional_rule(port: PortIntent, target: dict) -> dict:
    """A rule in today's grammar for the record before the build resolves the intent: a
    cap by its midpoint, a side or a box side by a thin box round it, a loop by the point
    at its mid-parameter (for a disk the point at 180 degrees, ON the circle)."""
    pts = target["points"]
    kind = target["kind"]
    if kind == "cap":
        (ax, ay), (bx, by) = pts
        return {"name": port.name, "kind": port.kind, "at": f"near:{(ax + bx) / 2:.15g},{(ay + by) / 2:.15g}"}
    if kind in ("side", "box"):
        x0, y0, x1, y1 = _bbox(pts)
        t = 1e-6 * max(x1 - x0, y1 - y0, 1.0)
        return {"name": port.name, "kind": port.kind, "box": [x0 - t, y0 - t, x1 + t, y1 + t]}
    if target.get("centre") is not None and target.get("radius") is not None:
        cx, cy = target["centre"]
        return {"name": port.name, "kind": port.kind, "at": f"near:{cx - target['radius']:.15g},{cy:.15g}"}
    x0, y0, x1, y1 = _bbox(pts)
    t = 1e-6 * max(x1 - x0, y1 - y0, 1.0)
    return {"name": port.name, "kind": port.kind, "box": [x0 - t, y0 - t, x1 + t, y1 + t]}


# -- the public functions ---------------------------------------------------------------------


def plan(sketch: Sketch, claims: "ClaimSet | None" = None, gmsh=None) -> Plan:
    """Walk the feature tree from `sketch.fluid`, solve every constraint feature (closed
    form; a Row on a non-Bypass item needs `gmsh` for one scratch build of the item), check
    every Row's fit (here, not at construction, D7), assign op names (the feature's `name`
    or `<kind><n>`; Row instances `<name>[k]`, 0-based, D29), and compile to the ops
    grammar. `claims` chooses a Row refusal's free parameter (3.3). Raises SketchError
    with the codes of section 5; a Row refusal carries `partial=item`."""
    fluid = sketch.fluid
    c = _Compiler(sketch, list(sketch.ports), claims, gmsh)
    root = c.visit(fluid, IDENTITY, root=True)
    return c.finish(root, sketch, sketch.units, sketch.scale)


def plan_feature(feature: Feature, gmsh=None) -> Plan:
    """One feature alone as its own fluid (a Row's item after E-ROW-FIT): the cli builds,
    measures and draws it so an rc-5 lap still shows the single instance with its
    footprint (3.15). A Bypass is anchored at u = 0 on its wall."""
    sketch = sk._current_or_none()
    ports = list(sketch.ports) if sketch is not None else []
    c = _Compiler(sketch, ports, None, gmsh)
    anchor = 0.0 if isinstance(feature, Bypass) else None
    root = c.visit(feature, IDENTITY, root=True, anchor=anchor)
    units = sketch.units if sketch is not None else "mm"
    scale = sketch.scale if sketch is not None else sk.SCALE["mm"]
    plan_ = c.finish(root, None, units, scale)
    plan_.notes = []
    host = _host_feature(feature)
    if host is not None:
        # the wall the item leaves is solved beside it (section 7.1 lap 1 prints `main`
        # above `loop`), through its own compiler so none of its ops join the partial's;
        # its Solved carries no ops, which is how build and lint know it was not built
        h = _Compiler(sketch, ports, None, gmsh)
        try:
            h.visit(host, IDENTITY, root=True)
        except SketchError:
            return plan_
        context = {key: Solved(s.kind, s.params, s.solved, []) for key, s in h.features.items()}
        plan_.features = {**context, **{k: v for k, v in plan_.features.items() if k not in context}}
    return plan_


def _host_feature(feature: Feature) -> Feature | None:
    """The feature whose wall a Bypass leaves (`loop.wall.feature`), or None."""
    wall = getattr(feature, "wall", None)
    host = getattr(wall, "feature", None)
    return host if isinstance(host, Feature) and host is not feature else None


def sampled_bounds(gmsh, dim: int, tag: int, n: int = ARC_POINTS) -> tuple[float, float, float, float]:
    """Tight bounds from `getValue` at n parameters per boundary curve (the same rule as
    measure.sampled_bounds; `getBoundingBox` is loose after a transform)."""
    xs: list[float] = []
    ys: list[float] = []
    for d, t in gmsh.model.getBoundary([(dim, tag)], combined=False, oriented=False):
        c = abs(t)
        (t0,), (t1,) = gmsh.model.getParametrizationBounds(1, c)
        params = [t0 + (t1 - t0) * i / n for i in range(n + 1)]
        xyz = gmsh.model.getValue(1, c, params)
        xs.extend(xyz[0::3])
        ys.extend(xyz[1::3])
    return (min(xs), min(ys), max(xs), max(ys))


def oriented_area(gmsh, face: int, n: int = ARC_POINTS) -> float:
    """The face's signed area from its oriented boundary curves (Green's theorem over the
    sampled curves, each taken in the direction gmsh reports): positive for a face the
    kernel built, negative after a mirror (measured: -180 for a mirrored 60 x 3 rect)."""
    total = 0.0
    for d, t in gmsh.model.getBoundary([(2, face)], combined=False, oriented=True):
        c = abs(t)
        (t0,), (t1,) = gmsh.model.getParametrizationBounds(1, c)
        params = [t0 + (t1 - t0) * i / n for i in range(n + 1)]
        xyz = gmsh.model.getValue(1, c, params)
        pts = list(zip(xyz[0::3], xyz[1::3]))
        if t < 0:
            pts.reverse()
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            total += 0.5 * (x0 * y1 - x1 * y0)
    return total


def canary(gmsh, face: int) -> None:
    """D26's check on a built face: a point one span past the sampled bounds along +x must
    be outside, and the oriented boundary must enclose a positive area; else the face
    carries a transform the classifier cannot be trusted on (E-KERNEL-STATE)."""
    x0, y0, x1, y1 = sampled_bounds(gmsh, 2, face)
    span = max(x1 - x0, y1 - y0)
    probe = (x1 + span, (y0 + y1) / 2)
    inside = gmsh.model.isInside(2, face, [probe[0], probe[1], 0.0])
    area = oriented_area(gmsh, face)
    if inside or area <= 0:
        why = (f"the classifier answers inside for a point {fmt(span)} outside the fluid" if inside
               else f"the boundary walk encloses a negative area ({fmt(area)})")
        raise SketchError("E-KERNEL-STATE", "fluid", f"{why}; the face carries a transform",
                          "the kernel cannot trust (an internal error: nothing in the script to fix; reported to the "
                          "desk, rc 3)", numbers={"probe": probe, "inside": int(bool(inside)), "area": area})


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
    mesh2d = _toolbox.load("mesh2d")
    ops, _ = mesh2d.parse_spec({"ops": plan.ops})
    legs: dict = {}
    checks: list[dict] = []
    try:
        face = mesh2d.build_face(gmsh, ops, scale=1.0, legs=legs, checks=checks)
        if plan.external:
            _refuse_a_hollow_body(gmsh, face, plan)
            # the box is cut round a COPY of the body so the body's own face survives
            # the cut: lint.judge_body reads E-PORT-ON-BODY (and E-VOID again) off it
            body_face = face
            (_, tool), = gmsh.model.occ.copy([(2, face)])
            gmsh.model.occ.synchronize()
            face, _ = mesh2d.external_face(gmsh, tool, plan.external)
            for solved in plan.features.values():
                if solved.kind == "BodyInBox":
                    solved.solved["body_face"] = int(body_face)
    except SystemExit as err:
        text = str(err)
        code = "E-DISJOINT" if "separate faces" in text else "E-BUILD"
        raise SketchError(code, "fluid", text.split(":")[0] if code == "E-DISJOINT" else text,
                          *([text.split(": ", 1)[1]] if code == "E-DISJOINT" and ": " in text else [])) from None
    orphans = orphan_landings(plan)
    if orphans:
        # a `to` leg is judged against the wall it lands on (3.6, E-LAND); a feature
        # built alone after a Row refusal (plan_feature) has no host in the plan, so
        # mesh2d's own landing check on it says nothing about the shape and is dropped
        checks[:] = [c for c in checks
                     if not any(str(c.get("what", "")).startswith(f"channel {name!r}: the leg to") for name in orphans)]
    canary(gmsh, face)
    return face, legs, checks


def _refuse_a_hollow_body(gmsh, body: int, plan: Plan) -> None:
    """E-VOID (section 5) on a BodyInBox body that carries an inner loop: the fluid was
    drawn instead of the solid. Judged here, before the box is cut, because the box
    minus a hollow body is two faces and `mesh2d.external_face` refuses with a sentence
    about the box; the lint's own E-VOID (judge_body) needs a face that never gets
    built. The hole's area and centroid are read off its sampled loop."""
    loops, curves = gmsh.model.occ.getCurveLoops(body)
    if len(loops) < 2:
        return
    name = next((key for key, s in plan.features.items() if s.kind == "BodyInBox"), "body")
    measured = []
    for tags in curves:
        pts: list[Point] = []
        for c in tags:
            c = abs(int(c))
            (t0,), (t1,) = gmsh.model.getParametrizationBounds(1, c)
            params = [t0 + (t1 - t0) * i / ARC_POINTS for i in range(ARC_POINTS + 1)]
            xyz = gmsh.model.getValue(1, c, params)
            pts.extend(zip(xyz[0::3], xyz[1::3]))
        area = abs(_signed_area(pts))
        measured.append((area, _centroid(pts)))
    # the outer loop is the largest; every other loop is a hole, the largest named
    measured.sort(key=lambda h: -h[0])
    holes = measured[1:]
    area, (cx, cy) = holes[0]
    raise SketchError("E-VOID", f"BodyInBox '{name}'",
                      f"the body has an enclosed hole of {area:.1f} {plan.units}2 at ({fmt(cx)}, {fmt(cy)})",
                      "the fluid was drawn instead of the solid; the tool cuts the flow box itself",
                      numbers={"area": area, "where": (cx, cy), "holes": len(holes)})


def _centroid(pts: list[Point]) -> Point:
    """The area centroid of a closed sampled loop (the vertex mean is biased by where
    the samples crowd)."""
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(a) < 1e-30:
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    return (cx / (3 * a), cy / (3 * a))


def orphan_landings(plan: Plan) -> set[str]:
    """The channel op names whose recorded `lands_on` names a feature that is not in the
    plan: a partial (a Row's item drawn alone, 3.15) lands on a wall of a host that was
    never compiled with it, so nothing can judge that landing until the whole sketch
    builds. Empty for every whole-sketch plan."""
    out: set[str] = set()
    names = {op.get("name") for op in plan.ops}
    for key, solved in plan.features.items():
        host = str(solved.solved.get("lands_on") or "")
        if not host:
            continue
        host_solved = plan.features.get(host.split(".", 1)[0])
        # a host that is not in the plan at all is an ops-grammar sidecar's (the lint
        # judges the samples, 3.6 clause d is skipped); a host solved with no ops is
        # plan_feature's context, and the landing on it cannot be judged
        if host_solved is not None and not any(name in names for name in host_solved.ops):
            out.update(name for name in solved.ops if any(op.get("name") == name and op.get("op") == "channel"
                                                          for op in plan.ops))
    return out


def _as_dict(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return _jsonable(value)
    if hasattr(value, "as_dict"):
        return value.as_dict()
    return _jsonable(value)


def _rule_dict(rule) -> dict:
    """A resolved rule in today's grammar, whether it arrives as the record's dict or as
    mesh2d's parsed tuple form."""
    out = {"name": rule["name"]}
    if rule.get("kind") and rule["kind"] not in ("inlet", "outlet"):
        out["kind"] = rule["kind"]
    at = rule.get("at")
    if isinstance(at, (tuple, list)) and len(at) == 3:
        axis, kind, value = at
        if kind == "near":
            at = f"near:{value[0]:.15g},{value[1]:.15g}"
        elif kind in ("min", "max"):
            at = f"{'xy'[axis]}:{kind}"
        else:
            at = f"{'xy'[axis]}:{value:.15g}"
    if at is not None:
        out["at"] = at
    if rule.get("box") is not None:
        out["box"] = [float(v) for v in rule["box"]]
    return out


def record(sketch: Sketch, plan: Plan, script: str, m: "Measurements", findings: list["Finding"],
           table: "ComplianceTable | None", resolved_rules: list[dict], *, claims=None) -> dict:
    """The record of section 4.2: a superset of the spec mesh2d.parse_spec reads."""
    try:
        import gmsh  # noqa: WPS433 - the version stamp only; the kernel never needs it here
        version = getattr(gmsh, "__version__", "unknown")
    except ImportError:
        version = "unavailable"
    out = {
        "format": "openreynolds.geometry/1",
        "units": plan.units, "scale": plan.scale,
        "ops": _jsonable(plan.ops),
        "patches": [_rule_dict(r) for r in resolved_rules],
        "ports": [p.as_dict() for p in plan.ports],
        "script": script,
        "script_sha256": hashlib.sha256(script.encode("utf-8")).hexdigest(),
        "features": {k: v.as_dict() for k, v in plan.features.items()},
        "instances": _jsonable(plan.instances),
        "apart": [list(a) for a in plan.apart],
        "claims": _as_dict(claims),
        "compliance": _as_dict(table),
        "lint": [_as_dict(f) for f in findings],
        "measurements": _as_dict(m),
        "built_with": {"gmsh": version, "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
    }
    if plan.external:
        # a BodyInBox record's ops end in the BODY; the flow box round it is built by
        # mesh2d.external_face from these sizes (body lengths) here, and on the instance
        # by `mesh2d.py --spec geometry.json --external --ahead ...`, which the case
        # writer and the rebuild command read from this key
        out["external"] = _jsonable(plan.external)
    return out


def run_script(script: str) -> Sketch:
    """The recorded script through the API namespace (no guard: the runner's child is
    where an untrusted script runs); the one Sketch it made."""
    Sketch._reset()
    namespace = dict(sk.API)
    namespace["__name__"] = "__sketch__"
    exec(compile(script, "script.py", "exec"), namespace)  # noqa: S102 - the recorded script, by design
    return Sketch.only()


def recompile(record: dict) -> list[dict]:
    """Run the recorded script through plan() and return its ops; a test asserts they equal
    the recorded ops (idempotence)."""
    sketch = run_script(record["script"])
    return _jsonable(plan(sketch, record.get("claims")).ops)


def to_json(plan: Plan) -> str:
    return json.dumps(plan.as_dict(), indent=1)
