"""The authoring API: everything the model writes in a script.

Pure Python, no gmsh import: a sketch is a tree of features with the numbers the
request stated, and the closed forms (a Bypass's arc, landing, lip and footprint; a
Row's pitch) live here so a refusal can carry its table before anything is built.
`api_summary()` renders the reference card from these signatures and docstrings, and a
test pins `reference.md` equal to it, so the docstrings are what the model reads.

Conventions: lengths in the sketch's units; angles in degrees, counter-clockwise from
+x as in `mesh2d.py`. `Passage` leg methods mutate and return `self` (D4); every other
feature is a value. The meshed region is `s.fluid` (D5). Skeleton (U0): every public
name below carries its contract's signature and docstring and raises
NotImplementedError until U1 lands.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

_U1 = ("not built in the U0 skeleton: U1 (sketch + compile) implements it against "
       "DESIGN.md section 3.3")


class SketchError(ValueError):
    """A refusal with the fix in the message. `code` is the stable code of section 5.
    `partial` is the feature the refusal is about when it can be drawn alone (a Row's
    item: the cli builds it by `compile.plan_feature` and the lap still gets a picture and
    the FEATURES/LEGS lines for it, section 3.15); `numbers` carries the refusal's table
    (E-ROW-FIT: the five-angle footprints) for the record."""

    def __init__(self, code: str, feature: str, what: str, *lines: str,
                 partial: "Feature | None" = None, numbers: dict | None = None):
        self.code = code
        self.feature = feature
        self.what = what
        self.lines: tuple[str, ...] = tuple(lines)
        self.partial = partial
        self.numbers: dict = dict(numbers or {})
        super().__init__(str(self))

    def __str__(self) -> str:
        head = f"!! ERROR  {self.code}  {self.feature}: {self.what}"
        return "\n".join([head, *(f"          {line}" for line in self.lines)])


Units = Literal["mm", "cm", "m", "in"]
SCALE = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "in": 0.0254}


class Feature:
    """Base of every shape. Values: booleans and transforms return new features and never
    mutate; only Passage's leg methods mutate (D4)."""

    name: str | None
    kind: str

    def __or__(self, other: "Feature") -> "Fuse":
        raise NotImplementedError(_U1)

    def __sub__(self, other: "Feature") -> "Cut":
        raise NotImplementedError(_U1)

    def __and__(self, other: "Feature") -> "Intersect":
        raise NotImplementedError(_U1)

    def moved(self, dx: float = 0.0, dy: float = 0.0) -> "Moved":
        raise NotImplementedError(_U1)

    def rotated(self, deg: float, about: tuple[float, float] | None = None) -> "Rotated":
        raise NotImplementedError(_U1)

    def mirrored(self, axis: Literal["x", "y"], at: float = 0.0, keep: bool = False) -> "Mirrored":
        raise NotImplementedError(_U1)

    def named(self, name: str) -> "Feature":
        """Registers under the new name (a Row instance, a boolean)."""
        raise NotImplementedError(_U1)

    @property
    def width(self) -> float | None:
        """The declared channel width where one exists (Passage, Bypass, Serpentine)."""
        raise NotImplementedError(_U1)


@dataclass(frozen=True)
class EdgeRef:
    """An edge of a feature named by intent; resolved to boundary curves after the build."""

    feature: "Feature"
    which: str
    """Rect: left right top bottom | Passage/Serpentine/Bypass: start end | Disk Band
    Polygon Outline: edge | Passage: side:left side:right side:top side:bottom |
    BodyInBox: inlet outlet farfield body."""
    leg: int | None = None
    """Passage sides: which straight leg (1-based); None = the only one."""

    def __str__(self) -> str:
        name = getattr(self.feature, "name", None) or "?"
        return f"{name}.{self.which}" + (f"[{self.leg}]" if self.leg is not None else "")

    def as_dict(self) -> dict:
        return {"feature": getattr(self.feature, "name", None), "which": self.which, "leg": self.leg}

    @staticmethod
    def from_dict(d: dict) -> "EdgeRef":
        """Needs the sketch's feature registry to turn the name back into a Feature."""
        raise NotImplementedError(_U1)


@dataclass(frozen=True)
class WallRef(EdgeRef):
    """A straight edge with a flow direction along it and an outward side; what a Bypass
    leaves from and lands on, what a branch Passage starts on or ends on. The wall's frame:
    `u` runs from the wall's UPSTREAM end in the flow direction (u = 0 there, u = span at
    the downstream end), `v` outward. `Bypass.at`, `Row.start`, a Passage `start=(wall, u)`,
    every footprint and every E-LAND / E-ROW-FIT number are in u. For a straight Passage
    leg the flow is the leg's heading; for a Rect side it is resolved at compile from the
    rect's inlet/outlet or `flow=`."""

    flow: tuple[float, float] | None = None
    """Unit vector along the wall in the flow direction; None until compile (Rect)."""
    outward: tuple[float, float] | None = None
    """Unit normal pointing out of the host."""
    span: float | None = None
    """The wall's length, None until compile (a Passage leg can grow after the ref is taken, D4)."""

    def as_dict(self) -> dict:
        d = EdgeRef.as_dict(self)
        d.update({"flow": self.flow, "outward": self.outward, "span": self.span})
        return d


@dataclass(frozen=True)
class PortIntent:
    name: str
    kind: Literal["inlet", "outlet", "wall", "slip", "symmetry"]
    edges: tuple[EdgeRef | str, ...]

    def as_dict(self) -> dict:
        """Edges as the record writes them: `main.start` for an EdgeRef, rule strings verbatim."""
        return {"name": self.name, "kind": self.kind,
                "edges": [e if isinstance(e, str) else str(e) for e in self.edges]}

    @staticmethod
    def from_dict(d: dict) -> "PortIntent":
        """EdgeRefs come back as their strings: the record keeps intents for idempotence
        (compile.recompile), not for resolution."""
        return PortIntent(name=d["name"], kind=d["kind"], edges=tuple(d.get("edges", ())))


class Sketch:
    """One sketch per script. `fluid` is the region that is meshed; ports are named by
    intent on features' edges. Lengths are in `units`; angles in degrees, counter-clockwise
    from +x."""

    units: Units
    scale: float
    name: str
    features: dict[str, Feature]
    """Every named feature registered on this sketch."""
    ports: list[PortIntent]
    """What inlet/outlet/wall/patch recorded, in order."""
    expected: tuple[int, int]
    notes: list[str]

    def __init__(self, units: Units = "mm", name: str = "sketch"):
        raise NotImplementedError(_U1)

    @property
    def fluid(self) -> Feature:
        """Assigned once; E-NO-FLUID if never assigned."""
        raise NotImplementedError(_U1)

    @fluid.setter
    def fluid(self, feature: Feature) -> None:
        raise NotImplementedError(_U1)

    # primitives (register the feature; the class constructors do the same via Sketch.current())
    # `center=` is accepted everywhere `centre=` is and recorded under `centre` (D39)

    def rect(self, *, origin: tuple[float, float] | None = None, centre: tuple[float, float] | None = None,
             size: tuple[float, float], round: float = 0.0, name: str | None = None) -> "Rect":
        raise NotImplementedError(_U1)

    def disk(self, *, centre: tuple[float, float], radius: float | None = None,
             diameter: float | None = None, name: str | None = None) -> "Disk":
        raise NotImplementedError(_U1)

    def polygon(self, points: list[tuple[float, float]], *, name: str | None = None) -> "Polygon":
        raise NotImplementedError(_U1)

    def annulus(self, *, centre, r_inner: float, r_outer: float, name=None) -> "Band":
        raise NotImplementedError(_U1)

    def band(self, *, centre, r_inner: float, r_outer: float, start_deg: float, end_deg: float,
             name=None) -> "Band":
        raise NotImplementedError(_U1)

    def outline(self, file: str, *, size: float | None = None, aoa: float = 0.0, name=None) -> "Outline":
        raise NotImplementedError(_U1)

    def passage(self, *, width: float, start: "tuple[float, float] | tuple[WallRef, float]",
                heading: float | None = None, name=None) -> "Passage":
        raise NotImplementedError(_U1)

    def apart(self, *features: Feature, gap: float | None = None) -> None:
        """Declares parts that must not overlap or touch (E-OVERLAP / E-TOUCH between them,
        by intersect area and getDistance on their standalone outlines, D35); two Rows, or
        a Row and any other named feature that is not its host, are apart by default."""
        raise NotImplementedError(_U1)

    # ports, by intent (an EdgeRef) or by rule string ("x:min", "near:0,0", "normal:-x", "box:x0,y0,x1,y1")

    def inlet(self, *edges: EdgeRef | str, name: str = "inlet") -> None:
        raise NotImplementedError(_U1)

    def outlet(self, *edges: EdgeRef | str, name: str = "outlet") -> None:
        raise NotImplementedError(_U1)

    def wall(self, *edges: EdgeRef | str, name: str) -> None:
        raise NotImplementedError(_U1)

    def patch(self, *edges: EdgeRef | str, name: str, kind: Literal["wall", "slip", "symmetry"]) -> None:
        raise NotImplementedError(_U1)

    def expect(self, *, inlets: int = 1, outlets: int = 1) -> None:
        """A manifold declares 1 inlet and 3 outlets; the default is one of each."""
        raise NotImplementedError(_U1)

    def note(self, text: str) -> None:
        """Printed back under SCRIPT; for the model's own reasoning, never parsed."""
        raise NotImplementedError(_U1)

    @staticmethod
    def current() -> "Sketch":
        """The last constructed; E-NO-SKETCH if none."""
        raise NotImplementedError(_U1)

    @staticmethod
    def _reset() -> None:
        """Tests only."""
        raise NotImplementedError(_U1)


class Rect(Feature):
    """origin, size, round; centre property; .left .right .top .bottom -> EdgeRef.
    Measures by role (D36): width = the shorter side, length = the longer, size_x / size_y
    = the page sizes."""

    def __init__(self, *, origin: tuple[float, float] | None = None, centre: tuple[float, float] | None = None,
                 size: tuple[float, float], round: float = 0.0, name: str | None = None, **alias):
        raise NotImplementedError(_U1)

    @property
    def left(self) -> EdgeRef:
        raise NotImplementedError(_U1)

    @property
    def right(self) -> EdgeRef:
        raise NotImplementedError(_U1)

    @property
    def top(self) -> EdgeRef:
        raise NotImplementedError(_U1)

    @property
    def bottom(self) -> EdgeRef:
        raise NotImplementedError(_U1)

    @property
    def centre(self) -> tuple[float, float]:
        raise NotImplementedError(_U1)


class Disk(Feature):
    """centre, radius (diameter= converted); .edge."""

    def __init__(self, *, centre: tuple[float, float] | None = None, radius: float | None = None,
                 diameter: float | None = None, name: str | None = None, **alias):
        raise NotImplementedError(_U1)

    @property
    def edge(self) -> EdgeRef:
        raise NotImplementedError(_U1)


class Polygon(Feature):
    """points (3+, made counter-clockwise); .edge."""

    def __init__(self, points: list[tuple[float, float]], *, name: str | None = None):
        raise NotImplementedError(_U1)

    @property
    def edge(self) -> EdgeRef:
        raise NotImplementedError(_U1)


class Band(Feature):
    """centre, r_inner, r_outer, start_deg, end_deg (annulus 0..360); .edge."""

    def __init__(self, *, centre=None, r_inner: float, r_outer: float, start_deg: float = 0.0,
                 end_deg: float = 360.0, name: str | None = None, **alias):
        raise NotImplementedError(_U1)

    @property
    def edge(self) -> EdgeRef:
        raise NotImplementedError(_U1)


class Outline(Feature):
    """file, size, aoa; .edge."""

    def __init__(self, file: str, *, size: float | None = None, aoa: float = 0.0, name: str | None = None):
        raise NotImplementedError(_U1)

    @property
    def edge(self) -> EdgeRef:
        raise NotImplementedError(_U1)


class Fuse(Feature):
    """a | b. Operands keep their names so EdgeRefs on them resolve."""

    def __init__(self, a: Feature, b: Feature, name: str | None = None):
        raise NotImplementedError(_U1)


class Cut(Feature):
    """a - b. Operands keep their names so EdgeRefs on them resolve."""

    def __init__(self, a: Feature, b: Feature, name: str | None = None):
        raise NotImplementedError(_U1)


class Intersect(Feature):
    """a & b. Operands keep their names so EdgeRefs on them resolve."""

    def __init__(self, a: Feature, b: Feature, name: str | None = None):
        raise NotImplementedError(_U1)


class Moved(Feature):
    """feature.moved(dx, dy). The operand keeps its name so EdgeRefs on it resolve."""

    def __init__(self, of: Feature, dx: float = 0.0, dy: float = 0.0, name: str | None = None):
        raise NotImplementedError(_U1)


class Rotated(Feature):
    """feature.rotated(deg, about). The operand keeps its name so EdgeRefs on it resolve."""

    def __init__(self, of: Feature, deg: float, about: tuple[float, float] | None = None,
                 name: str | None = None):
        raise NotImplementedError(_U1)


class Mirrored(Feature):
    """feature.mirrored(axis, at, keep). Compiled in closed form on the primitives'
    parameters (D30): no `mirror` op ever reaches the fluid."""

    def __init__(self, of: Feature, axis: Literal["x", "y"], at: float = 0.0, keep: bool = False,
                 name: str | None = None):
        raise NotImplementedError(_U1)


class Passage(Feature):
    """A constant-width channel along a centreline: turtle legs (line, arc, turn) and
    absolute legs (line_to, turn_to, u_turn, through, end_on). Leg methods add to the
    passage in place and return it, so `p.line(30)` and `p = p.line(30)` both work.
    `start` is a point, or `(wall, u)`: a branch that leaves `wall` at u along it (D37);
    then `heading` defaults to the wall's outward normal and the branch is cut flush at the
    wall (compiled to `from:`), so no cap corner stands proud of the host."""

    start: EdgeRef
    end: EdgeRef
    legs: list[dict]
    """The leg records in the ops grammar ({"line": L} | {"arc": {...}} | {"line": {"to": "y:1.5"}}
    for end_on only) plus the sketch-side kind "corner" (compiled into the two adjacent
    line legs' lengths)."""
    end_point: tuple[float, float]
    end_heading: float
    """Closed form, for the model's prints."""
    length: float
    """Centreline length, closed form."""
    leaves: WallRef | None
    lands_on: WallRef | None
    """Set by start=(wall, u) / end_on."""

    def __init__(self, *, width: float, start: "tuple[float, float] | tuple[WallRef, float]",
                 heading: float | None = None, name: str | None = None):
        raise NotImplementedError(_U1)

    def line(self, length: float) -> "Passage":
        raise NotImplementedError(_U1)

    def arc(self, *, radius: float, turn: float) -> "Passage":
        """Centreline radius; `turn` in degrees, + left / - right. radius <= width/2 is E-RADIUS."""
        raise NotImplementedError(_U1)

    def turn(self, deg: float) -> "Passage":
        """A sharp (mitred) corner of `deg` degrees, + left / - right, |deg| < 180 (D36):
        the leg before and the leg after are each extended by (w/2) * tan(|deg|/2) past
        the centreline corner so the outer walls meet at a point; recorded as a leg of
        kind `corner` (LEGS prints `corner 90 deg left at (95, 5)`); `sharp_corner`
        reads it. Needs a line leg before and after (E-CORNER otherwise)."""
        raise NotImplementedError(_U1)

    def line_to(self, *, x: float | None = None, y: float | None = None) -> "Passage":
        """Run along the current heading until x=.. or y=..: the length is solved in
        closed form from end_point/heading and the leg is a plain `line` (D28) -- an open
        end, never a landing. Moving away from the line, or running along it, is E-LINE-TO."""
        raise NotImplementedError(_U1)

    def turn_to(self, *, heading: float, radius: float,
                side: Literal["auto", "left", "right"] = "auto") -> "Passage":
        """The arc that brings the heading to an absolute value; auto = the shorter way
        round. A change of exactly 180 (within 1e-6) has no shorter side and is refused,
        E-TURN-AMBIGUOUS: use side=, or u_turn (D38)."""
        raise NotImplementedError(_U1)

    def u_turn(self, *, radius: float, side: Literal["left", "right"] = "left") -> "Passage":
        """A 180-degree arc. `left` stacks the return leg on the +y side for a +x heading
        (the left of the direction of travel)."""
        raise NotImplementedError(_U1)

    def end_on(self, wall: WallRef) -> "Passage":
        """The last leg: run along the current heading until the wall's line and cut flush
        there (the only Passage leg that emits a `to` leg; recorded with lands_on = wall,
        judged by the landing lint, D22/D37). E-LINE-TO when the heading does not meet
        the wall's line."""
        raise NotImplementedError(_U1)

    @classmethod
    def through(cls, *, width: float, points: list[tuple[float, float]], radius: float,
                name=None) -> "Passage":
        """Absolute waypoints with every corner filleted at `radius`; E-FILLET with the
        tangent lengths when a corner does not fit."""
        raise NotImplementedError(_U1)

    @property
    def top(self) -> WallRef:
        """Page words; the only straight leg, else E-SIDE-AMBIGUOUS."""
        raise NotImplementedError(_U1)

    @property
    def bottom(self) -> WallRef:
        """Page words; the only straight leg, else E-SIDE-AMBIGUOUS."""
        raise NotImplementedError(_U1)

    def side(self, which: Literal["left", "right", "top", "bottom"], leg: int | None = None) -> WallRef:
        raise NotImplementedError(_U1)


class Bypass(Feature):
    """A loop that leaves `wall` at `at`, at `leave_angle` from the flow along the wall,
    sweeps round an arc whose OUTER wall has `outer_radius`, and returns onto the same wall
    at `return_angle` measured from the upstream direction, so it always heads against the
    flow. The tool solves the arc, the landing, the lip and the footprint (closed form,
    section 3.4) and prints them; the leave leg is cut flush at the wall and the loop is
    clipped to the wall's outward side, so no cap stands proud of the wall. `at` is u along
    the wall (WallRef's frame); a Bypass outside a Row needs it (E-BYPASS-AT) and inside a
    Row must not give it (E-ROW-ITEM-AT: the Row places its instances; use start=).
    `leave_length` None = one channel width (D40), printed as `leave_length 3 (default:
    one width)`."""

    start: EdgeRef
    end: EdgeRef
    lands_on: WallRef
    """= wall; the landing lint's subject (D22)."""

    def __init__(self, *, wall: WallRef | EdgeRef, width: float, leave_angle: float, outer_radius: float,
                 return_angle: float, at: float | None = None, leave_length: float | None = None,
                 flow: Literal["+x", "-x", "+y", "-y"] | None = None, name: str | None = None):
        # refusals at construction (no kernel): E-RADIUS (outer_radius <= width), E-ANGLE
        # (leave/return not in (0, 90]), E-SELF-CROSS (the return leg's upstream wall meets
        # the leave leg's strip before v = 0: tR < tL with a small R; verified against a
        # built instance before any number is pinned); at compile (the wall's span is known
        # then): E-FLOW (a Rect wall with no inlet/outlet on the rect and no flow=),
        # E-BYPASS-AT, E-ROW-ITEM-AT, E-LAND-OFF-WALL (a bare Bypass whose closed-form
        # landing u is outside the wall's span)
        raise NotImplementedError(_U1)


class Row(Feature):
    """`count` copies of `item` along a wall or a direction. Give `gap=` only when the
    request states one; otherwise the tool solves the largest uniform gap that fits and
    prints it (D7). The pitch is the item's footprint along the row plus the gap; the fit
    against the wall is checked in `compile.plan` (the wall's span is final there, D4),
    before anything is built, and refused with the numbers and, for a Bypass, the footprint
    at five values of the free parameter and the largest gap that fits. Along a direction
    (`along=(dx, dy)`): pitch = footprint along it + `gap` (required: E-ROW-GAP-REQUIRED),
    the first instance is the item as drawn, and `start`/`align`/`margin` are refused
    (E-ROW-DIRECTION); only the built lint judges such a row."""

    def __init__(self, item: Feature, *, count: int, gap: float | None = None, pitch: float | None = None,
                 along: WallRef | tuple[float, float] | None = None, start: float | None = None,
                 align: Literal["centre", "start", "end"] = "centre", margin: float | None = None,
                 name: str | None = None):
        raise NotImplementedError(_U1)

    def __getitem__(self, k: int) -> Feature:
        """Instance k, named f"{name}[{k}]"."""
        raise NotImplementedError(_U1)


class Serpentine(Feature):
    """`passes` straight passes of `pass_length` joined by 180-degree bends of `bend_radius`
    (centreline), stacked along `stack`. Requires 2 * bend_radius > width (E-SERP-RADIUS)."""

    start: EdgeRef
    end: EdgeRef
    pass_pitch: float
    """2 * bend_radius."""
    wall_between: float
    """2 * bend_radius - width."""
    bends: int
    """passes - 1."""
    end_point: tuple[float, float]
    end_heading: float
    ends_on_start_side: bool

    def __init__(self, *, width: float, passes: int, pass_length: float, bend_radius: float,
                 start: tuple[float, float] = (0.0, 0.0), heading: float = 0.0,
                 stack: Literal["+y", "-y", "+x", "-x"] = "+y", name: str | None = None):
        raise NotImplementedError(_U1)

    def side(self, which, leg: int | None = None) -> WallRef:
        raise NotImplementedError(_U1)


class BodyInBox(Feature):
    """A body in a flow box (external), box sizes in body lengths as today's --ahead/--behind/
    --above/--below; declares its own inlet, outlet, farfield and body patches."""

    inlet: EdgeRef
    outlet: EdgeRef
    farfield: EdgeRef
    body_edge: EdgeRef

    def __init__(self, body: Feature, *, ahead: float = 2.0, behind: float = 5.0, above: float = 2.0,
                 below: float = 2.0, far: Literal["slip", "symmetry"] = "slip", name: str | None = None):
        raise NotImplementedError(_U1)


API: dict[str, object] = {
    "Sketch": Sketch, "Rect": Rect, "Disk": Disk, "Polygon": Polygon, "Band": Band,
    "Outline": Outline, "Passage": Passage, "Bypass": Bypass, "Row": Row,
    "Serpentine": Serpentine, "BodyInBox": BodyInBox, "EdgeRef": EdgeRef, "math": math,
}
"""The names bound into the script namespace."""


def api_summary() -> str:
    """The reference card (about 120 lines): one signature and one sentence per name, the
    rule strings, the units, the flow conventions, no shape example (invariant 1)."""
    raise NotImplementedError(_U1)
