"""The authoring API: everything the model writes in a script.

Pure Python, no gmsh import: a sketch is a tree of features with the numbers the
request stated, and the closed forms (a Bypass's arc, landing, lip and footprint; a
Passage's legs, mitres and fillets; a Serpentine's passes) live here so a refusal can
carry its table before anything is built. `api_summary()` renders the reference card
from these signatures and docstrings, and a test pins `reference.md` equal to it, so the
docstrings are what the model reads.

Conventions: lengths in the sketch's units; angles in degrees, counter-clockwise from
+x as in `mesh2d.py`. `Passage` leg methods mutate and return `self` (D4); every other
feature is a value. The meshed region is `s.fluid` (D5). A wall's frame is `u` along the
flow from its upstream end and `v` outward (WallRef). Nothing here draws or builds: the
compiler (`compile.py`) walks the tree, resolves every wall against the ports declared
anywhere in the script, and emits the ops grammar.
"""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Literal


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

_EPS = 1e-9
_SIDES = ("left", "right", "top", "bottom")
_FLOWS = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}


# -- small helpers (numbers, labels, vectors) ------------------------------------------


def fmt(x: float) -> str:
    """A number in a message: up to two decimals, trailing zeros dropped, so a margin
    reads 1.5, a span 57 and a footprint 19.74."""
    if x is None:
        return "?"
    if abs(x) < 5e-3:
        return "0"
    return f"{x:.2f}".rstrip("0").rstrip(".")


def label(feature) -> str:
    """"Bypass 'loop'" for a named feature, "Bypass" for an anonymous one."""
    kind = getattr(feature, "kind", type(feature).__name__)
    name = getattr(feature, "name", None)
    return f"{kind} {name!r}" if name else kind


def _point(value, what: str, who: str) -> tuple[float, float]:
    try:
        x, y = value
        return (float(x), float(y))
    except (TypeError, ValueError):
        raise SketchError("E-ARGS", who, f"{what} must be a point (x, y), not {value!r}") from None


def _number(value, what: str, who: str, positive: bool = False) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise SketchError("E-ARGS", who, f"{what} must be a number, not {value!r}") from None
    if math.isnan(out) or math.isinf(out):
        raise SketchError("E-ARGS", who, f"{what} must be a finite number, not {value!r}")
    if positive and out <= 0:
        raise SketchError("E-ARGS", who, f"{what} must be positive, not {fmt(out)}")
    return out


def _centre_alias(centre, alias: dict, who: str):
    """`center=` is accepted everywhere `centre=` is and recorded under `centre` (D39)."""
    unknown = sorted(set(alias) - {"center"})
    if unknown:
        raise SketchError("E-ARGS", who, f"unknown argument {unknown[0]!r}",
                          "see the reference card for the arguments this constructor takes")
    if "center" in alias:
        if centre is not None:
            raise SketchError("E-ARGS", who, "give centre= or center=, not both")
        centre = alias["center"]
    return centre


def _unit(v: tuple[float, float]) -> tuple[float, float]:
    n = math.hypot(v[0], v[1])
    return (v[0] / n, v[1] / n)


def _dir(deg: float) -> tuple[float, float]:
    return (math.cos(math.radians(deg)), math.sin(math.radians(deg)))


def _left(deg: float) -> tuple[float, float]:
    return (-math.sin(math.radians(deg)), math.cos(math.radians(deg)))


def _sense(deg: float) -> str:
    """"+x" / "-y" / "+x+y": the axis words a heading reads as."""
    dx, dy = _dir(deg)
    parts = []
    if abs(dx) > 1e-9:
        parts.append("+x" if dx > 0 else "-x")
    if abs(dy) > 1e-9:
        parts.append("+y" if dy > 0 else "-y")
    return "".join(parts) or "+x"


def _norm180(deg: float) -> float:
    """An angle folded into (-180, 180]."""
    out = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if abs(out + 180.0) < 1e-12 else out


# -- the registry: one sketch per script ----------------------------------------------


_MADE: list["Sketch"] = []


class Sketch:
    """One sketch per script. `fluid` is the region that is meshed; ports are named by
    intent on features' edges. Lengths are in `units`; angles in degrees, counter-clockwise
    from +x."""

    units: Units
    scale: float
    name: str
    features: dict[str, "Feature"]
    """Every named feature registered on this sketch."""
    ports: list["PortIntent"]
    """What inlet/outlet/wall/patch recorded, in order."""
    expected: tuple[int, int]
    notes: list[str]

    def __init__(self, units: Units = "mm", name: str = "sketch"):
        if units not in SCALE:
            raise SketchError("E-UNITS", "Sketch", f"units={units!r} is not one of mm, cm, m, in",
                              'Sketch(units="mm") for a request in millimetres; the claims use the same unit')
        self.units = units
        self.scale = SCALE[units]
        self.name = str(name)
        self.features = {}
        self.ports = []
        self.expected = (1, 1)
        self.notes = []
        self._fluid: Feature | None = None
        self._apart: list[tuple[Feature, Feature, float | None]] = []
        _MADE.append(self)

    @property
    def fluid(self) -> "Feature":
        """Assigned once; E-NO-FLUID if never assigned."""
        if self._fluid is None:
            raise SketchError("E-NO-FLUID", "Sketch", "the script made a Sketch but never assigned s.fluid",
                              "s.fluid = main | loops   (the one region that is meshed)")
        return self._fluid

    @fluid.setter
    def fluid(self, feature: "Feature") -> None:
        if not isinstance(feature, Feature):
            raise SketchError("E-ARGS", "Sketch", f"s.fluid must be a feature, not {type(feature).__name__}",
                              "s.fluid = main | loops   (the one region that is meshed)")
        self._fluid = feature
        self._register_tree(feature)

    def _register_tree(self, feature: "Feature") -> None:
        todo = [feature]
        while todo:
            f = todo.pop()
            if f.name and f.name not in self.features:
                self.features[f.name] = f
            todo.extend(f.operands())

    # primitives (register the feature; the class constructors do the same via Sketch.current())
    # `center=` is accepted everywhere `centre=` is and recorded under `centre` (D39)

    def rect(self, *, origin: tuple[float, float] | None = None, centre: tuple[float, float] | None = None,
             size: tuple[float, float], round: float = 0.0, name: str | None = None, **alias) -> "Rect":
        return Rect(origin=origin, centre=centre, size=size, round=round, name=name, **alias)

    def disk(self, *, centre: tuple[float, float] | None = None, radius: float | None = None,
             diameter: float | None = None, name: str | None = None, **alias) -> "Disk":
        return Disk(centre=centre, radius=radius, diameter=diameter, name=name, **alias)

    def polygon(self, points: list[tuple[float, float]], *, name: str | None = None) -> "Polygon":
        return Polygon(points, name=name)

    def annulus(self, *, centre=None, r_inner: float, r_outer: float, name=None, **alias) -> "Band":
        return Band(centre=centre, r_inner=r_inner, r_outer=r_outer, start_deg=0.0, end_deg=360.0,
                    name=name, **alias)

    def band(self, *, centre=None, r_inner: float, r_outer: float, start_deg: float, end_deg: float,
             name=None, **alias) -> "Band":
        return Band(centre=centre, r_inner=r_inner, r_outer=r_outer, start_deg=start_deg, end_deg=end_deg,
                    name=name, **alias)

    def outline(self, file: str, *, size: float | None = None, aoa: float = 0.0, name=None) -> "Outline":
        return Outline(file, size=size, aoa=aoa, name=name)

    def passage(self, *, width: float, start: "tuple[float, float] | tuple[WallRef, float]",
                heading: float | None = None, name=None) -> "Passage":
        return Passage(width=width, start=start, heading=heading, name=name)

    def apart(self, *features: "Feature", gap: float | None = None) -> None:
        """Declares parts that must not overlap or touch (E-OVERLAP / E-TOUCH between them,
        by intersect area and getDistance on their standalone outlines, D35); two Rows, or
        a Row and any other named feature that is not its host, are apart by default."""
        parts = list(features)
        if len(parts) < 2:
            raise SketchError("E-ARGS", "Sketch", "s.apart() needs two or more features",
                              "s.apart(top_loops, bottom_loops, gap=1.0)")
        for f in parts:
            if not isinstance(f, Feature):
                raise SketchError("E-ARGS", "Sketch", f"s.apart() takes features, not {type(f).__name__}")
        g = None if gap is None else _number(gap, "gap", "Sketch", positive=True)
        for i, a in enumerate(parts):
            for b in parts[i + 1:]:
                self._apart.append((a, b, g))

    # ports, by intent (an EdgeRef) or by rule string ("x:min", "near:0,0", "normal:-x", "box:x0,y0,x1,y1")

    def _port(self, name: str, kind: str, edges: tuple) -> None:
        if not edges:
            raise SketchError("E-ARGS", "Sketch", f"s.{kind}() names at least one edge",
                              "s.inlet(main.start), s.outlet(duct.right), s.wall(cyl.edge, name=\"cylinder\")")
        for e in edges:
            if not isinstance(e, (EdgeRef, str)):
                raise SketchError("E-ARGS", "Sketch",
                                  f"a port is an edge by intent (main.start, duct.left, cyl.edge) or a rule "
                                  f"string, not {type(e).__name__}")
            if isinstance(e, str) and not any(e.startswith(p) for p in ("x:", "y:", "near:", "box:", "normal:")):
                raise SketchError("E-ARGS", "Sketch", f"the rule string {e!r} is not one of x:min, y:max, x:0.05, "
                                  "near:x,y, box:x0,y0,x1,y1, normal:-x")
        self.ports.append(PortIntent(name=str(name), kind=kind, edges=tuple(edges)))

    def inlet(self, *edges: "EdgeRef | str", name: str = "inlet") -> None:
        self._port(name, "inlet", edges)

    def outlet(self, *edges: "EdgeRef | str", name: str = "outlet") -> None:
        self._port(name, "outlet", edges)

    def wall(self, *edges: "EdgeRef | str", name: str) -> None:
        self._port(name, "wall", edges)

    def patch(self, *edges: "EdgeRef | str", name: str, kind: Literal["wall", "slip", "symmetry"]) -> None:
        if kind not in ("wall", "slip", "symmetry"):
            raise SketchError("E-ARGS", "Sketch", f"patch kind {kind!r} is not one of wall, slip, symmetry")
        self._port(name, kind, edges)

    def expect(self, *, inlets: int = 1, outlets: int = 1) -> None:
        """A trunk with three branches declares 1 inlet and 3 outlets; the default is one
        of each."""
        try:
            self.expected = (int(inlets), int(outlets))
        except (TypeError, ValueError):
            raise SketchError("E-ARGS", "Sketch", "expect() takes whole numbers: expect(inlets=1, outlets=3)") from None

    def note(self, text: str) -> None:
        """Printed back under SCRIPT; for the model's own reasoning, never parsed."""
        self.notes.append(str(text))

    @staticmethod
    def current() -> "Sketch":
        """The last constructed; E-NO-SKETCH if none."""
        if not _MADE:
            raise SketchError("E-NO-SKETCH", "script", "the script made no Sketch",
                              's = Sketch(units="mm") first; every feature belongs to it')
        return _MADE[-1]

    @staticmethod
    def only() -> "Sketch":
        """The one sketch a script made: E-NO-SKETCH for none, E-MANY-SKETCHES for more."""
        if len(_MADE) > 1:
            raise SketchError("E-MANY-SKETCHES", "script", f"the script made {len(_MADE)} Sketch objects; exactly one",
                              "one Sketch per script; a second region is a feature of the same sketch")
        return Sketch.current()

    @staticmethod
    def _reset() -> None:
        """Tests only."""
        _MADE.clear()


def _current_or_none() -> "Sketch | None":
    return _MADE[-1] if _MADE else None


# -- edges by intent -----------------------------------------------------------------------


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
        which = self.which[5:] if self.which.startswith("side:") else self.which
        return f"{name}.{which}" + (f"[{self.leg}]" if self.leg is not None else "")

    def as_dict(self) -> dict:
        return {"feature": getattr(self.feature, "name", None), "which": self.which, "leg": self.leg}

    @staticmethod
    def from_dict(d: dict, features: dict | None = None) -> "EdgeRef":
        """Needs the sketch's feature registry to turn the name back into a Feature: the
        current sketch's when `features` is not given."""
        registry = features if features is not None else Sketch.current().features
        feature = registry.get(d.get("feature"))
        if feature is None:
            raise SketchError("E-ARGS", "EdgeRef", f"no feature named {d.get('feature')!r} on this sketch",
                              f"features: {', '.join(sorted(registry)) or 'none'}")
        if any(k in d for k in ("flow", "outward", "span")):
            flow, outward = d.get("flow"), d.get("outward")
            return WallRef(feature, d["which"], d.get("leg"),
                           flow=None if flow is None else (float(flow[0]), float(flow[1])),
                           outward=None if outward is None else (float(outward[0]), float(outward[1])),
                           span=d.get("span"))
        return EdgeRef(feature, d["which"], d.get("leg"))


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


# -- features -------------------------------------------------------------------------------


class Feature:
    """Base of every shape. Values: booleans and transforms return new features and never
    mutate; only Passage's leg methods mutate (D4)."""

    name: str | None
    kind: str = "Feature"

    def __init__(self, name: str | None = None):
        if name is not None and (not isinstance(name, str) or not name):
            raise SketchError("E-ARGS", self.kind, f"name= must be a non-empty string, not {name!r}")
        self.name = name
        self._register()

    def _register(self) -> None:
        s = _current_or_none()
        if s is not None and self.name:
            s.features[self.name] = self

    def operands(self) -> tuple["Feature", ...]:
        """The features this one is made of (booleans, transforms, a Row's item)."""
        return ()

    def __or__(self, other: "Feature") -> "Fuse":
        return Fuse(self, _as_feature(other, "|"))

    def __sub__(self, other: "Feature") -> "Cut":
        return Cut(self, _as_feature(other, "-"))

    def __and__(self, other: "Feature") -> "Intersect":
        return Intersect(self, _as_feature(other, "&"))

    def moved(self, dx: float = 0.0, dy: float = 0.0) -> "Moved":
        return Moved(self, dx, dy)

    def rotated(self, deg: float, about: tuple[float, float] | None = None) -> "Rotated":
        return Rotated(self, deg, about)

    def mirrored(self, axis: Literal["x", "y"], at: float = 0.0, keep: bool = False) -> "Mirrored":
        return Mirrored(self, axis, at, keep)

    def named(self, name: str) -> "Feature":
        """Registers under the new name (a Row instance, a boolean)."""
        if not isinstance(name, str) or not name:
            raise SketchError("E-ARGS", label(self), f"named() takes a non-empty string, not {name!r}")
        self.name = name
        self._register()
        return self

    @property
    def width(self) -> float | None:
        """The declared channel width where one exists (Passage, Bypass, Serpentine)."""
        return None

    def __repr__(self) -> str:
        return f"<{label(self)}>"


def _as_feature(other, op: str) -> "Feature":
    if not isinstance(other, Feature):
        raise SketchError("E-ARGS", "Feature", f"a feature {op} {type(other).__name__}: both sides of a boolean are features")
    return other


class Rect(Feature):
    """origin, size, round; centre property; .left .right .top .bottom -> EdgeRef.
    Measures by role (D36): width = the shorter side, length = the longer, size_x / size_y
    = the page sizes."""

    kind = "Rect"

    def __init__(self, *, origin: tuple[float, float] | None = None, centre: tuple[float, float] | None = None,
                 size: tuple[float, float], round: float = 0.0, name: str | None = None, **alias):
        who = f"Rect {name!r}" if name else "Rect"
        centre = _centre_alias(centre, alias, who)
        if (origin is None) == (centre is None):
            raise SketchError("E-ARGS", who, "give one of origin= (the lower-left corner) or centre=",
                              "Rect(origin=(0, 0), size=(300, 60)) or Rect(centre=(150, 30), size=(300, 60))")
        self.size = (_number(size[0], "size x", who, positive=True), _number(size[1], "size y", who, positive=True)) \
            if isinstance(size, (tuple, list)) and len(size) == 2 else _point(size, "size", who)
        if origin is not None:
            self.origin = _point(origin, "origin", who)
        else:
            cx, cy = _point(centre, "centre", who)
            self.origin = (cx - self.size[0] / 2, cy - self.size[1] / 2)
        self.round = _number(round, "round", who)
        if self.round < 0 or self.round > min(self.size) / 2 + 1e-12:
            raise SketchError("E-ARGS", who, f"round {fmt(self.round)} must be between 0 and half the shorter side "
                              f"({fmt(min(self.size) / 2)})")
        super().__init__(name)

    @property
    def centre(self) -> tuple[float, float]:
        return (self.origin[0] + self.size[0] / 2, self.origin[1] + self.size[1] / 2)

    @property
    def size_x(self) -> float:
        return self.size[0]

    @property
    def size_y(self) -> float:
        return self.size[1]

    @property
    def width(self) -> float:
        """The shorter side (D36)."""
        return min(self.size)

    @property
    def length(self) -> float:
        """The longer side (D36)."""
        return max(self.size)

    @property
    def left(self) -> EdgeRef:
        return EdgeRef(self, "left")

    @property
    def right(self) -> EdgeRef:
        return EdgeRef(self, "right")

    @property
    def top(self) -> EdgeRef:
        return EdgeRef(self, "top")

    @property
    def bottom(self) -> EdgeRef:
        return EdgeRef(self, "bottom")

    def side_segment(self, which: str) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """(a, b, outward) of a side in the rect's own frame: a -> b runs counter-clockwise."""
        x0, y0 = self.origin
        x1, y1 = x0 + self.size[0], y0 + self.size[1]
        return {"bottom": ((x0, y0), (x1, y0), (0.0, -1.0)), "right": ((x1, y0), (x1, y1), (1.0, 0.0)),
                "top": ((x1, y1), (x0, y1), (0.0, 1.0)), "left": ((x0, y1), (x0, y0), (-1.0, 0.0))}[which]


class Disk(Feature):
    """centre, radius (diameter= converted); .edge."""

    kind = "Disk"

    def __init__(self, *, centre: tuple[float, float] | None = None, radius: float | None = None,
                 diameter: float | None = None, name: str | None = None, **alias):
        who = f"Disk {name!r}" if name else "Disk"
        centre = _centre_alias(centre, alias, who)
        if centre is None:
            raise SketchError("E-ARGS", who, "centre= is required", "Disk(centre=(100, 30), diameter=10)")
        if (radius is None) == (diameter is None):
            raise SketchError("E-ARGS", who, "give one of radius= or diameter=", "Disk(centre=(100, 30), diameter=10)")
        self.centre = _point(centre, "centre", who)
        self.radius = _number(radius, "radius", who, positive=True) if radius is not None \
            else _number(diameter, "diameter", who, positive=True) / 2
        self.diameter_given = diameter is not None
        super().__init__(name)

    @property
    def diameter(self) -> float:
        return 2 * self.radius

    @property
    def edge(self) -> EdgeRef:
        return EdgeRef(self, "edge")


class Polygon(Feature):
    """points (3+, made counter-clockwise); .edge."""

    kind = "Polygon"

    def __init__(self, points: list[tuple[float, float]], *, name: str | None = None):
        who = f"Polygon {name!r}" if name else "Polygon"
        try:
            pts = [_point(p, "a point", who) for p in points]
        except TypeError:
            raise SketchError("E-ARGS", who, "points is a list of (x, y)") from None
        kept: list[tuple[float, float]] = []
        for p in pts:
            if not kept or math.hypot(p[0] - kept[-1][0], p[1] - kept[-1][1]) > _EPS:
                kept.append(p)
        if len(kept) > 1 and math.hypot(kept[0][0] - kept[-1][0], kept[0][1] - kept[-1][1]) <= _EPS:
            kept.pop()
        if len(kept) < 3:
            raise SketchError("E-ARGS", who, f"a polygon needs at least three distinct points, got {len(kept)}")
        if _signed_area(kept) < 0:
            kept.reverse()
        self.points = kept
        super().__init__(name)

    @property
    def edge(self) -> EdgeRef:
        return EdgeRef(self, "edge")


class Band(Feature):
    """centre, r_inner, r_outer, start_deg, end_deg (annulus 0..360); .edge."""

    kind = "Band"

    def __init__(self, *, centre=None, r_inner: float, r_outer: float, start_deg: float = 0.0,
                 end_deg: float = 360.0, name: str | None = None, **alias):
        who = f"Band {name!r}" if name else "Band"
        centre = _centre_alias(centre, alias, who)
        if centre is None:
            raise SketchError("E-ARGS", who, "centre= is required")
        self.centre = _point(centre, "centre", who)
        self.r_inner = _number(r_inner, "r_inner", who, positive=True)
        self.r_outer = _number(r_outer, "r_outer", who, positive=True)
        if self.r_inner >= self.r_outer:
            raise SketchError("E-ARGS", who, f"r_inner {fmt(self.r_inner)} must be smaller than r_outer {fmt(self.r_outer)}")
        self.start_deg = _number(start_deg, "start_deg", who)
        self.end_deg = _number(end_deg, "end_deg", who)
        if not 0 < self.end_deg - self.start_deg <= 360 + 1e-9:
            raise SketchError("E-ARGS", who, f"end_deg - start_deg must be between 0 and 360, not "
                              f"{fmt(self.end_deg - self.start_deg)}")
        super().__init__(name)

    @property
    def edge(self) -> EdgeRef:
        return EdgeRef(self, "edge")

    @property
    def full(self) -> bool:
        return self.end_deg - self.start_deg >= 360 - 1e-9


class Outline(Feature):
    """file, size, aoa; .edge."""

    kind = "Outline"

    def __init__(self, file: str, *, size: float | None = None, aoa: float = 0.0, name: str | None = None):
        who = f"Outline {name!r}" if name else "Outline"
        if not isinstance(file, str) or not file:
            raise SketchError("E-ARGS", who, "file is the path of an x,y outline (csv or Selig .dat)")
        self.file = file
        self.size = None if size is None else _number(size, "size", who, positive=True)
        self.aoa = _number(aoa, "aoa", who)
        super().__init__(name)

    @property
    def edge(self) -> EdgeRef:
        return EdgeRef(self, "edge")


class _Boolean(Feature):
    def __init__(self, a: Feature, b: Feature, name: str | None = None):
        self.a = _as_feature(a, self.kind)
        self.b = _as_feature(b, self.kind)
        super().__init__(name)

    def operands(self) -> tuple[Feature, ...]:
        return (self.a, self.b)

    @property
    def width(self) -> float | None:
        widths = [w for w in (self.a.width, self.b.width) if w is not None]
        return min(widths) if widths else None


class Fuse(_Boolean):
    """a | b. Operands keep their names so EdgeRefs on them resolve."""

    kind = "Fuse"


class Cut(_Boolean):
    """a - b. Operands keep their names so EdgeRefs on them resolve."""

    kind = "Cut"


class Intersect(_Boolean):
    """a & b. Operands keep their names so EdgeRefs on them resolve."""

    kind = "Intersect"


class Moved(Feature):
    """feature.moved(dx, dy). The operand keeps its name so EdgeRefs on it resolve."""

    kind = "Moved"

    def __init__(self, of: Feature, dx: float = 0.0, dy: float = 0.0, name: str | None = None):
        self.of = _as_feature(of, "moved")
        self.dx = _number(dx, "dx", "moved()")
        self.dy = _number(dy, "dy", "moved()")
        super().__init__(name)

    def operands(self) -> tuple[Feature, ...]:
        return (self.of,)

    @property
    def width(self) -> float | None:
        return self.of.width


class Rotated(Feature):
    """feature.rotated(deg, about). The operand keeps its name so EdgeRefs on it resolve.
    `about` None rotates about the operand's own bounding-box centre."""

    kind = "Rotated"

    def __init__(self, of: Feature, deg: float, about: tuple[float, float] | None = None,
                 name: str | None = None):
        self.of = _as_feature(of, "rotated")
        self.deg = _number(deg, "deg", "rotated()")
        self.about = None if about is None else _point(about, "about", "rotated()")
        super().__init__(name)

    def operands(self) -> tuple[Feature, ...]:
        return (self.of,)

    @property
    def width(self) -> float | None:
        return self.of.width


class Mirrored(Feature):
    """feature.mirrored(axis, at, keep). Compiled in closed form on the primitives'
    parameters (D30): no `mirror` op ever reaches the fluid. `axis="x"` mirrors across
    the line x = at, `axis="y"` across y = at; `keep=True` keeps the original too."""

    kind = "Mirrored"

    def __init__(self, of: Feature, axis: Literal["x", "y"], at: float = 0.0, keep: bool = False,
                 name: str | None = None):
        self.of = _as_feature(of, "mirrored")
        if axis not in ("x", "y"):
            raise SketchError("E-ARGS", "mirrored()", f"axis must be \"x\" (the line x = at) or \"y\" (y = at), not {axis!r}")
        self.axis = axis
        self.at = _number(at, "at", "mirrored()")
        self.keep = bool(keep)
        super().__init__(name)

    def operands(self) -> tuple[Feature, ...]:
        return (self.of,)

    @property
    def width(self) -> float | None:
        return self.of.width


# -- walls in closed form -----------------------------------------------------------------


@dataclass
class WallLine:
    """A straight wall resolved in its host's declared frame: the segment `a -> b`, the
    outward normal, and, once the flow is known, `flow`, the upstream end `w0` and `span`.
    The compiler applies the host's placement transform on top."""

    ref: EdgeRef
    a: tuple[float, float]
    b: tuple[float, float]
    outward: tuple[float, float]
    flow: tuple[float, float] | None
    w0: tuple[float, float] | None
    span: float

    @property
    def direction(self) -> tuple[float, float]:
        return _unit((self.b[0] - self.a[0], self.b[1] - self.a[1]))

    def point(self, u: float, v: float = 0.0) -> tuple[float, float]:
        """u along the flow from the upstream end, v outward."""
        fx, fy = self.flow if self.flow is not None else self.direction
        ox, oy = self.outward
        w0 = self.w0 if self.w0 is not None else self.a
        return (w0[0] + u * fx + v * ox, w0[1] + u * fy + v * oy)

    def line(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """(point, direction, normal) of the infinite wall line."""
        return (self.a, self.direction, self.outward)


def as_wall(ref, who: str) -> WallRef:
    """An EdgeRef of a Rect side or a Passage/Serpentine side as a WallRef; anything else
    (a cap, a Disk's edge) is not a straight wall to leave from or land on."""
    if isinstance(ref, WallRef):
        return ref
    if isinstance(ref, EdgeRef):
        f = ref.feature
        if isinstance(f, Rect) and ref.which in _SIDES:
            a, b, outward = f.side_segment(ref.which)
            return WallRef(f, ref.which, None, flow=None, outward=outward, span=math.hypot(b[0] - a[0], b[1] - a[1]))
        if isinstance(f, (Passage, Serpentine)) and ref.which.startswith("side:"):
            return WallRef(f, ref.which, ref.leg)
        raise SketchError("E-ARGS", who, f"{ref} is not a straight wall",
                          "a wall is a Rect side (duct.top) or a straight Passage leg's side (main.top, "
                          "main.side(\"left\", leg=1))")
    raise SketchError("E-ARGS", who, f"wall= takes an edge (main.top, duct.bottom), not {type(ref).__name__}")


def rect_flow(rect: Rect, ports: list[PortIntent] | None, override: str | None) -> tuple[float, float] | None:
    """The flow through a Rect: `flow=` when given, else read from the inlet/outlet the
    script declared on the rect's sides (anywhere in the script, D6); None when neither."""
    if override is not None:
        if override not in _FLOWS:
            raise SketchError("E-ARGS", label(rect), f"flow={override!r} is not one of +x, -x, +y, -y")
        return _FLOWS[override]
    for p in ports or ():
        if p.kind not in ("inlet", "outlet"):
            continue
        for e in p.edges:
            if isinstance(e, EdgeRef) and e.feature is rect and e.which in _SIDES:
                outward = rect.side_segment(e.which)[2]
                return (-outward[0], -outward[1]) if p.kind == "inlet" else outward
    return None


def wall_line(ref: EdgeRef, *, ports: list[PortIntent] | None = None, flow: str | None = None,
              who: str | None = None, need_flow: bool = True) -> WallLine:
    """The wall of an EdgeRef in its host's declared frame. A Passage side needs the leg
    to exist (E-EMPTY-PASSAGE) and to be the only straight one unless `leg=` says which
    (E-SIDE-AMBIGUOUS); a Rect side needs a flow (E-FLOW) when `need_flow`."""
    who = who or str(ref)
    f = ref.feature
    if isinstance(f, Rect) and ref.which in _SIDES:
        a, b, outward = f.side_segment(ref.which)
        span = math.hypot(b[0] - a[0], b[1] - a[1])
        fl = rect_flow(f, ports, flow)
        if fl is None and need_flow:
            raise SketchError("E-FLOW", who, f"which way does the flow run along {ref}?",
                              f"{f.name or 'the rect'} is a Rect; declare s.inlet({f.name or 'duct'}.left) and "
                              f"s.outlet({f.name or 'duct'}.right), or give flow=\"+x\"")
        w0 = None
        if fl is not None:
            d = _unit((b[0] - a[0], b[1] - a[1]))
            along = fl[0] * d[0] + fl[1] * d[1]
            if abs(along) < 0.5:
                raise SketchError("E-FLOW", who, f"the flow runs across {ref}, not along it",
                                  f"a wall a loop leaves is one the flow runs along; the flow through "
                                  f"{f.name or 'the rect'} is {_sense(math.degrees(math.atan2(fl[1], fl[0])))}")
            fl = d if along > 0 else (-d[0], -d[1])
            w0 = a if along > 0 else b
        return WallLine(ref, a, b, outward, fl, w0, span)
    if isinstance(f, (Passage, Serpentine)) and ref.which.startswith("side:"):
        rec = f._straight_leg(ref.leg, ref.which[5:])
        h = rec["heading"]
        outward = f._side_outward(ref.which[5:], h, ref.leg)
        w = f.width
        a = (rec["from"][0] + outward[0] * w / 2, rec["from"][1] + outward[1] * w / 2)
        b = (rec["to"][0] + outward[0] * w / 2, rec["to"][1] + outward[1] * w / 2)
        return WallLine(ref, a, b, outward, _dir(h), a, math.hypot(b[0] - a[0], b[1] - a[1]))
    raise SketchError("E-ARGS", who, f"{ref} is not a straight wall",
                      "a wall is a Rect side (duct.top) or a straight Passage leg's side (main.top)")


def wall_axis_text(ref: EdgeRef) -> str | None:
    """`y:1.5` for a wall whose line is axis-aligned in its host's declared frame (the
    `from` / `to` spelling of the ops grammar), None for an inclined wall."""
    try:
        wall = wall_line(ref, need_flow=False)
    except SketchError:
        return None
    q, d, _ = wall.line()
    if abs(d[1]) < _EPS:
        return f"y:{q[1]:.15g}"
    if abs(d[0]) < _EPS:
        return f"x:{q[0]:.15g}"
    return None


def _line_hit(p: tuple[float, float], d: tuple[float, float], q: tuple[float, float],
              n: tuple[float, float]) -> float | None:
    """t with p + t d on the line through q with normal n; None when parallel."""
    denom = d[0] * n[0] + d[1] * n[1]
    if abs(denom) < _EPS:
        return None
    return ((q[0] - p[0]) * n[0] + (q[1] - p[1]) * n[1]) / denom


def _signed_area(pts: list[tuple[float, float]]) -> float:
    return 0.5 * sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
                     for i in range(len(pts)))


# -- the passage: turtle legs in closed form -------------------------------------------------


def trace_legs(width: float, start: tuple[float, float], heading: float, specs: list[dict], who: str,
               ports: list[PortIntent] | None = None) -> dict:
    """The centreline of a channel from its typed legs: one record per leg in mesh2d's
    shape (kind line | arc | to | corner; from, to, heading, heading_out, length, centre,
    radius, sweep, lands, line), the end point and heading, the centreline length. A
    `corner` leg is the mitre of D36: recorded at the centreline corner with its outer and
    inner vertices; the two adjacent lines are extended by (w/2) tan(|deg|/2) at compile."""
    x, y = start
    h = heading
    records: list[dict] = []
    length = 0.0
    for i, spec in enumerate(specs):
        if "line" in spec:
            L = spec["line"]
            rec = {"kind": "line", "from": (x, y), "heading": h, "length": L}
            x += L * math.cos(math.radians(h))
            y += L * math.sin(math.radians(h))
            rec.update(to=(x, y), heading_out=h)
            length += L
        elif "arc" in spec:
            radius, angle = spec["arc"]["radius"], spec["arc"]["angle"]
            nx, ny = _left(h)
            if angle < 0:
                nx, ny = -nx, -ny
            cx, cy = x + radius * nx, y + radius * ny
            a0 = math.degrees(math.atan2(y - cy, x - cx))
            a1 = a0 + angle
            rec = {"kind": "arc", "from": (x, y), "heading": h, "centre": (cx, cy), "radius": radius,
                   "sweep": angle, "outer_radius": radius + width / 2, "inner_radius": radius - width / 2}
            h += angle
            x, y = cx + radius * math.cos(math.radians(a1)), cy + radius * math.sin(math.radians(a1))
            rec.update(to=(x, y), heading_out=h)
            length += radius * math.radians(abs(angle))
        elif "corner" in spec:
            deg = spec["corner"]
            e = (width / 2) * math.tan(math.radians(abs(deg) / 2))
            d_in = _dir(h)
            n_left = _left(h)
            n_out = (-n_left[0], -n_left[1]) if deg > 0 else n_left
            outer = (x + e * d_in[0] + (width / 2) * n_out[0], y + e * d_in[1] + (width / 2) * n_out[1])
            inner = (x - e * d_in[0] - (width / 2) * n_out[0], y - e * d_in[1] - (width / 2) * n_out[1])
            rec = {"kind": "corner", "from": (x, y), "to": (x, y), "at": (x, y), "heading": h, "deg": deg,
                   "extend": e, "outer_vertex": outer, "inner_vertex": inner}
            h += deg
            rec["heading_out"] = h
        else:
            wall = wall_line(spec["to"], ports=ports, who=who, need_flow=False)
            q, wd, n = wall.line()
            d = _dir(h)
            t = _line_hit((x, y), d, q, n)
            axis_word = "x" if abs(wd[0]) < _EPS else ("y" if abs(wd[1]) < _EPS else None)
            line_text = (f"{axis_word}={fmt(q[0] if axis_word == 'x' else q[1])}" if axis_word
                         else f"the line of {spec['to']}")
            if t is None:
                raise SketchError("E-LINE-TO", who, f"end_on({spec['to']}) from ({fmt(x)}, {fmt(y)}) heading {fmt(h)} "
                                  f"runs along {line_text} and never meets it",
                                  f"the heading is {_sense(h)}; turn first so the leg crosses the wall's line")
            if t <= _EPS:
                raise SketchError("E-LINE-TO", who, f"end_on({spec['to']}) from ({fmt(x)}, {fmt(y)}) heading {fmt(h)} "
                                  f"moves away from {line_text}",
                                  f"the heading is {_sense(h)}; turn first so the leg runs toward the wall")
            lands = (x + t * d[0], y + t * d[1])
            rec = {"kind": "to", "from": (x, y), "heading": h, "length": t, "lands": lands, "to": lands,
                   "line": line_text, "heading_out": h, "lands_on": str(spec["to"])}
            x, y = lands
            length += t
        records.append(rec)
    return {"records": records, "end": (x, y), "end_heading": h, "length": length}


class Passage(Feature):
    """A constant-width channel along a centreline: turtle legs (line, arc, turn) and
    absolute legs (line_to, turn_to, u_turn, through, end_on). Leg methods add to the
    passage in place and return it, so `p.line(30)` and `p = p.line(30)` both work.
    `start` is a point, or `(wall, u)`: a branch that leaves `wall` at u along it (D37);
    then `heading` defaults to the wall's outward normal and the branch is cut flush at the
    wall (compiled to `from:`), so no cap corner stands proud of the host."""

    kind = "Passage"

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
        who = f"Passage {name!r}" if name else "Passage"
        self._width = _number(width, "width", who, positive=True)
        self.leaves: WallRef | None = None
        self.lands_on: WallRef | None = None
        self.start_u: float | None = None
        self._specs: list[dict] = []
        self._closed = False
        if isinstance(start, (tuple, list)) and len(start) == 2 and isinstance(start[0], EdgeRef):
            self.leaves = as_wall(start[0], who)
            self.start_u = _number(start[1], "start u", who)
            self._start_point: tuple[float, float] | None = None
            self._heading_given = None if heading is None else _number(heading, "heading", who)
        else:
            self._start_point = _point(start, "start", who)
            self._heading_given = 0.0 if heading is None else _number(heading, "heading", who)
        self.start = EdgeRef(self, "start")
        self.end = EdgeRef(self, "end")
        super().__init__(name)

    # -- the closed form ---------------------------------------------------------------

    @property
    def width(self) -> float:
        return self._width

    @property
    def legs(self) -> list[dict]:
        out = []
        for spec in self._specs:
            if "line" in spec:
                out.append({"line": spec["line"]})
            elif "arc" in spec:
                out.append({"arc": dict(spec["arc"])})
            elif "corner" in spec:
                out.append({"corner": spec["corner"]})
            else:
                out.append({"line": {"to": wall_axis_text(spec["to"]) or str(spec["to"])}})
        return out

    def start_frame(self, ports: list[PortIntent] | None = None, flow: str | None = None) -> tuple[tuple[float, float], float]:
        """(start point, heading) in the declared frame. A branch on a wall reads the wall's
        point at `u` and, with no heading given, the wall's outward normal; the flow of a
        Rect wall comes from `ports` (the current sketch's when None) or `flow`, and a wall
        whose flow is still unknown is placed from its declared direction until the compiler
        resolves it."""
        if self.leaves is None:
            return self._start_point, self._heading_given
        if ports is None:
            s = _current_or_none()
            ports = s.ports if s is not None else []
        wall = wall_line(self.leaves, ports=ports, flow=flow, who=label(self), need_flow=False)
        p = wall.point(self.start_u, 0.0)
        h = self._heading_given
        if h is None:
            h = math.degrees(math.atan2(wall.outward[1], wall.outward[0]))
        return p, h

    def trace(self, ports: list[PortIntent] | None = None, flow: str | None = None) -> dict:
        """trace_legs over the typed legs from the start frame."""
        p, h = self.start_frame(ports, flow)
        return trace_legs(self._width, p, h, self._specs, label(self), ports)

    @property
    def records(self) -> list[dict]:
        return self.trace()["records"]

    @property
    def end_point(self) -> tuple[float, float]:
        return self.trace()["end"]

    @property
    def end_heading(self) -> float:
        return self.trace()["end_heading"]

    @property
    def length(self) -> float:
        return self.trace()["length"]

    @property
    def heading(self) -> float:
        return self.start_frame()[1]

    @property
    def start_point(self) -> tuple[float, float]:
        return self.start_frame()[0]

    def _current(self) -> tuple[tuple[float, float], float]:
        t = self.trace()
        return t["end"], t["end_heading"]

    def _check_open(self, what: str) -> None:
        if self._closed:
            raise SketchError("E-LEG", label(self), f"{what} after end_on(): end_on is the last leg",
                              "end_on(wall) cuts the passage flush at the wall; nothing follows it")

    def _check_corner_followed(self, what: str) -> None:
        if self._specs and "corner" in self._specs[-1]:
            raise SketchError("E-CORNER", label(self),
                              f"turn({fmt(self._specs[-1]['corner'])}) needs a line leg before it and after it; "
                              f"leg {len(self._specs) + 1} is {what}",
                              "a mitred corner joins two straight legs: .line() on both sides of .turn()")

    def check(self) -> None:
        """The refusals the compiler raises for a passage as a whole: no legs
        (E-EMPTY-PASSAGE), a corner with nothing after it (E-CORNER)."""
        if not self._specs:
            raise SketchError("E-EMPTY-PASSAGE", label(self), f"{label(self)} has no legs",
                              ".line(), .arc(), .turn(), .line_to(), .turn_to(), .u_turn(), .end_on() add legs to "
                              f"the passage in place: {self.name or 'path'}.line(30)")
        if "corner" in self._specs[-1]:
            raise SketchError("E-CORNER", label(self),
                              f"turn({fmt(self._specs[-1]['corner'])}) needs a line leg before it and after it; "
                              f"leg {len(self._specs)} is the last leg",
                              "add the line the corner turns into: .turn(90).line(80)")

    # -- legs (mutate and return self, D4) --------------------------------------------------

    def line(self, length: float) -> "Passage":
        self._check_open("line()")
        L = _number(length, "line length", label(self), positive=True)
        self._specs.append({"line": L})
        return self

    def arc(self, *, radius: float, turn: float) -> "Passage":
        """Centreline radius; `turn` in degrees, + left / - right. radius <= width/2 is E-RADIUS."""
        self._check_open("arc()")
        r = _number(radius, "arc radius", label(self), positive=True)
        t = _number(turn, "arc turn", label(self))
        if r <= self._width / 2 + 1e-12:
            raise SketchError("E-RADIUS", label(self),
                              f"arc radius {fmt(r)} with width {fmt(self._width)} leaves an inner radius of "
                              f"{fmt(max(0.0, r - self._width / 2))}",
                              f"the inner wall folds over; radius > width/2 ({fmt(self._width / 2)}) (inner > 0), "
                              f"and >= width meshes cleanly")
        if abs(t) < 1e-9 or abs(t) > 360:
            raise SketchError("E-LEG", label(self), f"arc turn {fmt(t)} must be non-zero and within 360 degrees")
        self._check_corner_followed("an arc")
        self._specs.append({"arc": {"radius": r, "angle": t}})
        return self

    def turn(self, deg: float) -> "Passage":
        """A sharp (mitred) corner of `deg` degrees, + left / - right, |deg| < 180 (D36):
        the leg before and the leg after are each extended by (w/2) * tan(|deg|/2) past
        the centreline corner so the outer walls meet at a point; recorded as a leg of
        kind `corner` (LEGS prints `corner 90 deg left at (95, 5)`); `sharp_corner`
        reads it. Needs a line leg before and after (E-CORNER otherwise)."""
        self._check_open("turn()")
        d = _number(deg, "turn", label(self))
        if abs(d) < 1e-9 or abs(d) >= 180 - 1e-9:
            raise SketchError("E-CORNER", label(self), f"turn({fmt(d)}): a mitred corner turns by more than 0 and "
                              f"less than 180 degrees",
                              "a U-turn is u_turn(radius=..., side=...); a straight run needs no corner")
        if not self._specs:
            raise SketchError("E-CORNER", label(self), f"turn({fmt(d)}) needs a line leg before it and after it; "
                              "there is no leg before it", "start with .line(L), then .turn(deg), then .line(L)")
        if "line" not in self._specs[-1]:
            what = "an arc" if "arc" in self._specs[-1] else "a corner"
            raise SketchError("E-CORNER", label(self), f"turn({fmt(d)}) needs a line leg before it and after it; "
                              f"leg {len(self._specs)} is {what}",
                              "a mitred corner joins two straight legs: .line() on both sides of .turn()")
        self._specs.append({"corner": d})
        return self

    def line_to(self, *, x: float | None = None, y: float | None = None) -> "Passage":
        """Run along the current heading until x=.. or y=..: the length is solved in
        closed form from end_point/heading and the leg is a plain `line` (D28) -- an open
        end, never a landing. Moving away from the line, or running along it, is E-LINE-TO."""
        self._check_open("line_to()")
        if (x is None) == (y is None):
            raise SketchError("E-LINE-TO", label(self), "line_to() takes exactly one of x= or y=",
                              "line_to(x=60) runs along the heading until x = 60")
        axis = 0 if x is not None else 1
        value = _number(x if x is not None else y, "line_to target", label(self))
        (px, py), h = self._current()
        d = _dir(h)
        word = "xy"[axis]
        here = (px, py)[axis]
        if abs(d[axis]) < _EPS:
            raise SketchError("E-LINE-TO", label(self),
                              f"line_to({word}={fmt(value)}) from ({fmt(px)}, {fmt(py)}) heading {fmt(h)} runs along "
                              f"{word} = {fmt(value)} and never reaches it",
                              f"the heading is {_sense(h)}; line_to({'yx'[axis]}=...) or turn first")
        L = (value - here) / d[axis]
        if L <= _EPS:
            mirrored = 2 * here - value
            raise SketchError("E-LINE-TO", label(self),
                              f"line_to({word}={fmt(value)}) from ({fmt(px)}, {fmt(py)}) heading {fmt(h)} moves away from "
                              f"{word} = {fmt(value)}",
                              f"the heading is {_sense(h)}; turn first, or line_to({word}={fmt(mirrored)})")
        self._check_corner_followed("a line_to")
        self._specs.append({"line": L, "solved_from": f"line_to({word}={fmt(value)})"})
        return self

    def turn_to(self, *, heading: float, radius: float,
                side: Literal["auto", "left", "right"] = "auto") -> "Passage":
        """The arc that brings the heading to an absolute value; auto = the shorter way
        round. A change of exactly 180 (within 1e-6) has no shorter side and is refused,
        E-TURN-AMBIGUOUS: use side=, or u_turn (D38)."""
        self._check_open("turn_to()")
        if side not in ("auto", "left", "right"):
            raise SketchError("E-ARGS", label(self), f"side={side!r} is not one of auto, left, right")
        target = _number(heading, "heading", label(self))
        r = _number(radius, "radius", label(self), positive=True)
        _, h = self._current()
        delta = _norm180(target - h)
        if abs(delta) < 1e-9:
            return self
        if side == "auto":
            if abs(abs(delta) - 180.0) < 1e-6:
                raise SketchError("E-TURN-AMBIGUOUS", label(self),
                                  f"heading {fmt(h % 360)} -> {fmt(target % 360)} is a U-turn either way; there is no "
                                  "shorter side",
                                  f"turn_to(heading={fmt(target)}, radius={fmt(r)}, side=\"left\") or "
                                  f"u_turn(radius={fmt(r)}, side=\"left\") (left = the +y side for a +x heading)")
            turn = delta
        elif side == "left":
            turn = delta % 360.0 or 360.0
        else:
            turn = -((-delta) % 360.0 or 360.0)
        return self.arc(radius=r, turn=turn)

    def u_turn(self, *, radius: float, side: Literal["left", "right"] = "left") -> "Passage":
        """A 180-degree arc. `left` stacks the return leg on the +y side for a +x heading
        (the left of the direction of travel)."""
        if side not in ("left", "right"):
            raise SketchError("E-ARGS", label(self), f"side={side!r} is not one of left, right")
        return self.arc(radius=radius, turn=180.0 if side == "left" else -180.0)

    def end_on(self, wall: WallRef) -> "Passage":
        """The last leg: run along the current heading until the wall's line and cut flush
        there (the only Passage leg that emits a `to` leg; recorded with lands_on = wall,
        judged by the landing lint, D22/D37). E-LINE-TO when the heading does not meet
        the wall's line."""
        self._check_open("end_on()")
        w = as_wall(wall, label(self))
        self._check_corner_followed("an end_on")
        self._specs.append({"to": w})
        try:
            self.trace()
        except SketchError:
            self._specs.pop()
            raise
        self.lands_on = w
        self._closed = True
        return self

    @classmethod
    def through(cls, *, width: float, points: list[tuple[float, float]], radius: float,
                name=None) -> "Passage":
        """Absolute waypoints with every corner filleted at `radius`; E-FILLET with the
        tangent lengths when a corner does not fit."""
        who = f"Passage.through {name!r}" if name else "Passage.through"
        w = _number(width, "width", who, positive=True)
        r = _number(radius, "radius", who, positive=True)
        try:
            pts = [_point(p, "a waypoint", who) for p in points]
        except TypeError:
            raise SketchError("E-ARGS", who, "points is a list of (x, y) waypoints") from None
        if len(pts) < 2:
            raise SketchError("E-ARGS", who, "through() needs at least two waypoints")
        if r <= w / 2 + 1e-12:
            raise SketchError("E-RADIUS", who, f"radius {fmt(r)} with width {fmt(w)} leaves an inner radius of "
                              f"{fmt(max(0.0, r - w / 2))}",
                              f"the inner wall folds over; radius > width/2 ({fmt(w / 2)}) (inner > 0), and >= width "
                              "meshes cleanly")
        segments = []
        for a, b in zip(pts, pts[1:]):
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            if L <= _EPS:
                raise SketchError("E-ARGS", who, f"two consecutive waypoints coincide at ({fmt(a[0])}, {fmt(a[1])})")
            segments.append((a, b, L, math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))))
        turns = []
        for k in range(1, len(pts) - 1):
            phi = _norm180(segments[k][3] - segments[k - 1][3])
            if abs(abs(phi) - 180.0) < 1e-9:
                raise SketchError("E-FILLET", who, f"corner {k + 1} at ({fmt(pts[k][0])}, {fmt(pts[k][1])}) turns back on "
                                  "itself (180 degrees): no fillet fits a reversal",
                                  "a U-turn is two waypoints apart by 2 x radius, or u_turn() on a Passage")
            turns.append((phi, r * math.tan(math.radians(abs(phi) / 2))))
        for k, (a, b, L, _) in enumerate(segments):
            t_before = turns[k - 1][1] if k >= 1 else 0.0
            t_after = turns[k][1] if k < len(turns) else 0.0
            left = L - t_before - t_after
            if left < -1e-6 * r:
                # name the corner at the segment's start (its end when the segment is the first)
                c = k - 1 if k >= 1 else k
                phi, t = turns[c]
                corner = pts[c + 1]
                other = c + 1 if k >= 1 and k < len(turns) else c
                other_t = t_after if k >= 1 and k < len(turns) else t
                lines = [f"each side (tan {fmt(abs(phi) / 2)} deg); the segment ({fmt(a[0])}, {fmt(a[1])})->({fmt(b[0])}, "
                         f"{fmt(b[1])}) is {fmt(L)} long and corner {other + 2} takes {fmt(other_t)} of it,",
                         f"leaving {fmt(left)} -- it needs 0 or more; a radius of {fmt(r * (L / (t_before + t_after)))} would leave 0"]
                raise SketchError("E-FILLET", who,
                                  f"corner {c + 2} at ({fmt(corner[0])}, {fmt(corner[1])}): radius {fmt(r)} needs {fmt(t)} of "
                                  "straight on", *lines,
                                  numbers={"corner": c + 2, "radius": r, "tangent": t, "segment": L, "leaving": left})
        p = cls(width=w, start=pts[0], heading=segments[0][3], name=name)
        for k, (a, b, L, _) in enumerate(segments):
            t_before = turns[k - 1][1] if k >= 1 else 0.0
            t_after = turns[k][1] if k < len(turns) else 0.0
            straight = L - t_before - t_after
            if straight > 1e-9 * max(1.0, L):
                p._specs.append({"line": straight})
            if k < len(turns):
                p._specs.append({"arc": {"radius": r, "angle": turns[k][0]}})
        return p

    # -- sides as walls ---------------------------------------------------------------------

    def _runs(self) -> list[list[int]]:
        """Maximal runs of consecutive straight legs on one heading (0-based leg indices):
        `main.line(40)` then, later, `main.line(20)` is still ONE straight wall of 60, so a
        WallRef taken before the second leg keeps naming it and a Row's fit sees the whole
        wall (D4/D7)."""
        runs: list[list[int]] = []
        prev_heading = None
        for i, rec in enumerate(self.records):
            if rec["kind"] not in ("line", "to"):
                prev_heading = None
                continue
            if runs and prev_heading is not None and abs(_norm180(rec["heading"] - prev_heading)) < 1e-9:
                runs[-1].append(i)
            else:
                runs.append([i])
            prev_heading = rec["heading"]
        return runs

    def _straight_legs(self) -> list[int]:
        """The first leg index of every straight run (what `leg=` names)."""
        return [run[0] for run in self._runs()]

    def _straight_leg(self, leg: int | None, which: str) -> dict:
        """The straight run `.which` names, merged into one record (from, to, heading,
        length): the only run, or the one containing leg `leg` (1-based)."""
        recs = self.records
        runs = self._runs()
        if not runs:
            raise SketchError("E-EMPTY-PASSAGE", label(self), f"{label(self)} has no straight leg for .{which}",
                              f".line() adds one in place: {self.name or 'path'}.line(30)")
        if leg is None:
            if len(runs) > 1:
                name = self.name or "path"
                raise SketchError("E-SIDE-AMBIGUOUS", label(self),
                                  f".{which} names the only straight leg, but {name!r} has {len(runs)} straight legs",
                                  " or ".join(f'{name}.side("{which}", leg={run[0] + 1})' for run in runs))
            run = runs[0]
        else:
            run = next((r for r in runs if (leg - 1) in r), None)
            if run is None:
                raise SketchError("E-SIDE-AMBIGUOUS", label(self),
                                  f'side("{which}", leg={leg}): the straight legs are '
                                  + ", ".join(str(r[0] + 1) for r in runs),
                                  "leg counts every leg from 1 (corners and arcs included) and names a straight one")
        first, last = recs[run[0]], recs[run[-1]]
        return {"kind": "line", "from": first["from"], "to": last["to"], "heading": first["heading"],
                "heading_out": last["heading_out"], "length": sum(recs[i]["length"] for i in run),
                "legs": [i + 1 for i in run]}

    def _side_outward(self, which: str, heading: float, leg: int | None) -> tuple[float, float]:
        n_left = _left(heading)
        if which == "left":
            return n_left
        if which == "right":
            return (-n_left[0], -n_left[1])
        if abs(n_left[1]) < 1e-9:
            raise SketchError("E-SIDE-AMBIGUOUS", label(self),
                              f"leg {leg or 1} runs along y (heading {fmt(heading)}): top and bottom do not name a side of it",
                              f'{self.name or "path"}.side("left") or .side("right") (left = the left of the direction of travel)')
        up = n_left if n_left[1] > 0 else (-n_left[0], -n_left[1])
        return up if which == "top" else (-up[0], -up[1])

    def side(self, which: Literal["left", "right", "top", "bottom"], leg: int | None = None) -> WallRef:
        """A straight leg's wall as a WallRef: page words (top, bottom) or flow words
        (left, right of the direction of travel); `leg` counts every leg from 1."""
        if which not in _SIDES:
            raise SketchError("E-ARGS", label(self), f"side {which!r} is not one of left, right, top, bottom")
        if leg is not None:
            try:
                leg = int(leg)
            except (TypeError, ValueError):
                raise SketchError("E-ARGS", label(self), "leg= is a whole number counted from 1") from None
        straight = self._straight_legs()
        if leg is None and len(straight) > 1:
            self._straight_leg(None, which)
        if straight and (leg is not None or len(straight) == 1):
            rec = self._straight_leg(leg, which)
            h = rec["heading"]
            outward = self._side_outward(which, h, leg)
            return WallRef(self, f"side:{which}", leg, flow=_dir(h), outward=outward, span=rec["length"])
        return WallRef(self, f"side:{which}", leg)

    @property
    def top(self) -> WallRef:
        """Page words; the only straight leg, else E-SIDE-AMBIGUOUS."""
        return self.side("top")

    @property
    def bottom(self) -> WallRef:
        """Page words; the only straight leg, else E-SIDE-AMBIGUOUS."""
        return self.side("bottom")


# -- the bypass loop: closed form ------------------------------------------------------------


def bypass_closed_form(width: float, leave_angle: float, outer_radius: float, return_angle: float,
                       leave_length: float) -> dict:
    """The loop in the wall's frame with its anchor at u = 0 (u along the flow, v outward),
    before any OCC call (section 3.4). Keys: R, sweep, P1, C, P2, landing, return_len,
    lip_u, lip_deg, footprint (lo, hi), height, theta_leave, theta_return, leave_length,
    a0, a1, lands_upstream_by, lip_on_arc."""
    w, tL, tR = width, math.radians(leave_angle), math.radians(return_angle)
    R = outer_radius - w / 2
    L = leave_length
    sweep = 180.0 + return_angle - leave_angle
    P1 = (L * math.cos(tL), L * math.sin(tL))
    C = (P1[0] - R * math.sin(tL), P1[1] + R * math.cos(tL))
    a0 = leave_angle - 90.0
    a1 = a0 + sweep
    P2 = (C[0] - R * math.sin(tR), C[1] + R * math.cos(tR))
    return_len = P2[1] / math.sin(tR)
    landing = (P2[0] - return_len * math.cos(tR), 0.0)
    lo = landing[0] - (w / 2) / math.sin(tR)
    hi = (w / 2) / math.sin(tL)
    if a0 <= 180.0 <= a1:
        lo = min(lo, C[0] - outer_radius)
    if a0 <= 0.0 <= a1:
        hi = max(hi, C[0] + outer_radius)
    height = C[1] + outer_radius if a0 <= 90.0 <= a1 else max(P1[1], P2[1]) + w / 2
    lip_on_arc = L < (w / 2) / math.tan(tL)
    if lip_on_arc:
        ang = math.asin(max(-1.0, min(1.0, -C[1] / outer_radius)))
        lip_u = C[0] + outer_radius * math.cos(ang)
        lip_deg = 90.0 + math.degrees(ang)
    else:
        lip_u = (w / 2) / math.sin(tL)
        lip_deg = leave_angle
    return {"R": R, "sweep": sweep, "P1": P1, "C": C, "P2": P2, "landing": landing, "return_len": return_len,
            "lip_u": lip_u, "lip_deg": lip_deg, "footprint": (lo, hi), "height": height,
            "theta_leave": leave_angle, "theta_return": return_angle, "leave_length": L, "a0": a0, "a1": a1,
            "lands_upstream_by": -landing[0], "lip_on_arc": lip_on_arc}


def _segments_cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float],
                    d: tuple[float, float]) -> tuple[float, float] | None:
    """The proper crossing point of segments a-b and c-d, None when they do not cross."""
    r = (b[0] - a[0], b[1] - a[1])
    s = (d[0] - c[0], d[1] - c[1])
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < _EPS:
        return None
    qp = (c[0] - a[0], c[1] - a[1])
    t = (qp[0] * s[1] - qp[1] * s[0]) / denom
    u = (qp[0] * r[1] - qp[1] * r[0]) / denom
    if _EPS < t < 1 - _EPS and _EPS < u < 1 - _EPS:
        return (a[0] + t * r[0], a[1] + t * r[1])
    return None


def bypass_self_cross(cf: dict, width: float) -> tuple[float, float] | None:
    """Where the return leg's downstream wall cuts through the leave leg's centreline above
    the wall line (u, v in the wall's frame, anchor at 0), or None. A loop whose return leg
    passes through its own leave leg encloses two islands instead of one (measured on a
    built instance at leave 10, return 90, leave_length 20, outer 6, width 3); the mouths
    merging AT the wall line (T01's one opening) is not this: that crossing is below v = 0."""
    tR = math.radians(cf["theta_return"])
    d = (-math.cos(tR), -math.sin(tR))
    n_down = (math.sin(tR), -math.cos(tR))
    a = (cf["P2"][0] + n_down[0] * width / 2, cf["P2"][1] + n_down[1] * width / 2)
    b = (a[0] - a[1] / d[1] * d[0], 0.0)
    hit = _segments_cross(a, b, (0.0, 0.0), cf["P1"])
    if hit is not None and hit[1] > 1e-6 * width:
        return hit
    return None


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

    kind = "Bypass"

    start: EdgeRef
    end: EdgeRef
    lands_on: WallRef
    """= wall; the landing lint's subject (D22)."""

    def __init__(self, *, wall: WallRef | EdgeRef, width: float, leave_angle: float, outer_radius: float,
                 return_angle: float, at: float | None = None, leave_length: float | None = None,
                 flow: Literal["+x", "-x", "+y", "-y"] | None = None, name: str | None = None):
        who = f"Bypass {name!r}" if name else "Bypass"
        self.wall = as_wall(wall, who)
        self._width = _number(width, "width", who, positive=True)
        self.leave_angle = _number(leave_angle, "leave_angle", who)
        self.outer_radius = _number(outer_radius, "outer_radius", who, positive=True)
        self.return_angle = _number(return_angle, "return_angle", who)
        self.at = None if at is None else _number(at, "at", who)
        self.leave_length_given = None if leave_length is None else _number(leave_length, "leave_length", who)
        if self.leave_length_given is not None and self.leave_length_given < 0:
            raise SketchError("E-ARGS", who, f"leave_length {fmt(self.leave_length_given)} must not be negative")
        if flow is not None and flow not in _FLOWS:
            raise SketchError("E-ARGS", who, f"flow={flow!r} is not one of +x, -x, +y, -y")
        self.flow = flow
        w = self._width
        if self.outer_radius <= w + 1e-12:
            raise SketchError("E-RADIUS", who,
                              f"outer_radius {fmt(self.outer_radius)} with width {fmt(w)} leaves an inner radius of "
                              f"{fmt(max(0.0, self.outer_radius - w))}",
                              "the inner wall folds over; outer_radius > width (inner > 0), and >= 1.5 x width meshes cleanly")
        for word, value in (("leave_angle", self.leave_angle), ("return_angle", self.return_angle)):
            if not 0 < value <= 90:
                why = ("return_angle is measured from the upstream direction; 90 is straight back, "
                       f"{fmt(value)} would return with the flow" if word == "return_angle"
                       else "leave_angle is measured from the flow along the wall; 90 is straight out")
                raise SketchError("E-ANGLE", who, f"{word} {fmt(value)} is not in (0, 90]", why)
        cf = self.closed_form()
        hit = bypass_self_cross(cf, w)
        if hit is not None:
            lo, hi = 0.0, self.leave_length
            for _ in range(60):
                mid = (lo + hi) / 2
                if bypass_self_cross(self.closed_form(leave_length=mid), w) is None:
                    lo = mid
                else:
                    hi = mid
            raise SketchError(
                "E-SELF-CROSS", who,
                "the return leg's downstream wall crosses the leave leg before reaching the wall",
                f"the return leg heads {fmt(180 + self.return_angle)} deg from u {fmt(cf['P2'][0])}, v {fmt(cf['P2'][1])} "
                f"and cuts through the leave leg (anchor to u {fmt(cf['P1'][0])}, v {fmt(cf['P1'][1])}) at u {fmt(hit[0])}, "
                f"v {fmt(hit[1])} above the wall:",
                f"the loop would enclose two islands, not one (leave_angle {fmt(self.leave_angle)}, {self.leave_length_text()}, "
                f"outer_radius {fmt(self.outer_radius)}, return_angle {fmt(self.return_angle)})",
                f"leave_length <= {fmt(lo)} keeps the return leg clear of the leave leg; a larger outer_radius or a "
                "shallower return_angle also does",
                numbers={"cross": hit, "P1": cf["P1"], "P2": cf["P2"], "leave_length_max": lo})
        self.start = EdgeRef(self, "start")
        self.end = EdgeRef(self, "end")
        self.lands_on = self.wall
        super().__init__(name)

    @property
    def width(self) -> float:
        return self._width

    @property
    def leave_length(self) -> float:
        """The typed leave length, or one width by default (D40)."""
        return self._width if self.leave_length_given is None else self.leave_length_given

    @property
    def leave_length_default(self) -> bool:
        return self.leave_length_given is None

    def closed_form(self, **override) -> dict:
        """bypass_closed_form on this loop's parameters; `override` varies one of them
        (the E-ROW-FIT table)."""
        params = {"width": self._width, "leave_angle": self.leave_angle, "outer_radius": self.outer_radius,
                  "return_angle": self.return_angle, "leave_length": self.leave_length}
        params.update(override)
        return bypass_closed_form(**params)

    def leave_length_text(self) -> str:
        """`leave_length 3 (default: one width)` or `leave_length 5`."""
        return f"leave_length {fmt(self.leave_length)}" + (" (default: one width)" if self.leave_length_default else "")


# -- rows and instances --------------------------------------------------------------------


class RowInstance(Feature):
    """Instance k of a Row, named `<row>[k]` (D29); placed by the Row at compile."""

    kind = "RowInstance"

    def __init__(self, row: "Row", k: int):
        self.row = row
        self.k = k
        super().__init__(f"{row.name}[{k}]" if row.name else None)

    def operands(self) -> tuple[Feature, ...]:
        return (self.row,)

    @property
    def width(self) -> float | None:
        return self.row.width


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

    kind = "Row"

    def __init__(self, item: Feature, *, count: int, gap: float | None = None, pitch: float | None = None,
                 along: WallRef | tuple[float, float] | None = None, start: float | None = None,
                 align: Literal["centre", "start", "end"] = "centre", margin: float | None = None,
                 name: str | None = None):
        who = f"Row {name!r}" if name else "Row"
        if not isinstance(item, Feature):
            raise SketchError("E-ARGS", who, f"a Row repeats a feature, not {type(item).__name__}")
        self.item = item
        try:
            self.count = int(count)
        except (TypeError, ValueError):
            raise SketchError("E-ARGS", who, f"count must be a whole number, not {count!r}") from None
        if self.count < 1:
            raise SketchError("E-ARGS", who, f"count {self.count} must be at least 1")
        self.gap = None if gap is None else _number(gap, "gap", who)
        self.pitch = None if pitch is None else _number(pitch, "pitch", who, positive=True)
        if self.gap is not None and self.gap < 0:
            raise SketchError("E-ARGS", who, f"gap {fmt(self.gap)} must not be negative")
        if along is None or isinstance(along, EdgeRef):
            self.along = None if along is None else as_wall(along, who)
        else:
            self.along = _point(along, "along", who)
            if math.hypot(*self.along) < _EPS:
                raise SketchError("E-ARGS", who, "along=(dx, dy) must not be the zero vector")
        self.start = None if start is None else _number(start, "start", who)
        if align not in ("centre", "start", "end", "center"):
            raise SketchError("E-ARGS", who, f"align={align!r} is not one of centre, start, end")
        self.align = "centre" if align == "center" else align
        self.margin = None if margin is None else _number(margin, "margin", who)
        super().__init__(name)

    def __getitem__(self, k: int) -> Feature:
        """Instance k, named f"{name}[{k}]"."""
        try:
            k = int(k)
        except (TypeError, ValueError):
            raise SketchError("E-ARGS", label(self), f"an instance index is a whole number from 0, not {k!r}") from None
        if not 0 <= k < self.count:
            raise SketchError("E-ARGS", label(self), f"instance {k} does not exist: the row has {self.count} "
                              f"instances, {self.name or 'row'}[0] to {self.name or 'row'}[{self.count - 1}]")
        return RowInstance(self, k)

    def operands(self) -> tuple[Feature, ...]:
        return (self.item,)

    @property
    def width(self) -> float | None:
        return self.item.width


# -- the serpentine ------------------------------------------------------------------------


class Serpentine(Feature):
    """`passes` straight passes of `pass_length` joined by 180-degree bends of `bend_radius`
    (centreline), stacked along `stack`. Requires 2 * bend_radius > width (E-SERP-RADIUS)."""

    kind = "Serpentine"

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
        who = f"Serpentine {name!r}" if name else "Serpentine"
        self._width = _number(width, "width", who, positive=True)
        try:
            self.passes = int(passes)
        except (TypeError, ValueError):
            raise SketchError("E-ARGS", who, f"passes must be a whole number, not {passes!r}") from None
        if self.passes < 1:
            raise SketchError("E-ARGS", who, f"passes {self.passes} must be at least 1")
        self.pass_length = _number(pass_length, "pass_length", who, positive=True)
        self.bend_radius = _number(bend_radius, "bend_radius", who, positive=True)
        self.start_point = _point(start, "start", who)
        self.heading = _number(heading, "heading", who)
        if stack not in _FLOWS:
            raise SketchError("E-ARGS", who, f"stack={stack!r} is not one of +y, -y, +x, -x")
        self.stack = stack
        w = self._width
        if 2 * self.bend_radius <= w + 1e-12:
            overlap = w - 2 * self.bend_radius
            raise SketchError("E-SERP-RADIUS", who,
                              f"bend_radius {fmt(self.bend_radius)} puts the passes {fmt(2 * self.bend_radius)} apart centre to",
                              f"centre, but they are {fmt(w)} wide: they {'touch' if overlap < 1e-12 else 'overlap by ' + fmt(overlap)}. "
                              f"bend_radius > {fmt(w / 2)} (walls touching), >= {fmt(0.75 * w)}",
                              f"leaves a {fmt(w / 2)} wall between passes")
        n_left = _left(self.heading)
        sv = _FLOWS[stack]
        dot = n_left[0] * sv[0] + n_left[1] * sv[1]
        if abs(abs(dot) - 1.0) > 1e-9:
            raise SketchError("E-ARGS", who, f"stack {stack!r} is not across heading {fmt(self.heading)}",
                              "passes stack across the heading: heading 0 or 180 with +y/-y, 90 or 270 with +x/-x")
        self._first_turn = 1.0 if dot > 0 else -1.0
        self.start = EdgeRef(self, "start")
        self.end = EdgeRef(self, "end")
        super().__init__(name)

    @property
    def width(self) -> float:
        return self._width

    @property
    def pass_pitch(self) -> float:
        return 2 * self.bend_radius

    @property
    def wall_between(self) -> float:
        return 2 * self.bend_radius - self._width

    @property
    def bends(self) -> int:
        return self.passes - 1

    @property
    def legs(self) -> list[dict]:
        """The ops-grammar legs: line, arc(+-180), line, ..."""
        out: list[dict] = []
        for k in range(self.passes):
            out.append({"line": self.pass_length})
            if k < self.passes - 1:
                sign = self._first_turn if k % 2 == 0 else -self._first_turn
                out.append({"arc": {"radius": self.bend_radius, "angle": 180.0 * sign}})
        return out

    def _specs(self) -> list[dict]:
        return self.legs

    def trace(self, ports=None, flow=None) -> dict:
        return trace_legs(self._width, self.start_point, self.heading, self.legs, label(self))

    @property
    def records(self) -> list[dict]:
        return self.trace()["records"]

    @property
    def end_point(self) -> tuple[float, float]:
        return self.trace()["end"]

    @property
    def end_heading(self) -> float:
        return self.trace()["end_heading"]

    @property
    def length(self) -> float:
        return self.trace()["length"]

    @property
    def ends_on_start_side(self) -> bool:
        """An even number of passes ends on the side it started from."""
        return self.passes % 2 == 0

    @property
    def pass_centrelines(self) -> list[float]:
        """The across-stack coordinate of each pass's centreline."""
        sv = _FLOWS[self.stack]
        out = []
        for rec in self.records:
            if rec["kind"] == "line":
                p = rec["from"]
                out.append(p[0] * sv[0] + p[1] * sv[1])
        return out

    _runs = Passage._runs
    _straight_legs = Passage._straight_legs
    _straight_leg = Passage._straight_leg
    _side_outward = Passage._side_outward
    side = Passage.side

    def start_frame(self, ports=None, flow=None) -> tuple[tuple[float, float], float]:
        return self.start_point, self.heading

    def check(self) -> None:
        return None


# -- a body in a flow box -------------------------------------------------------------------


class BodyInBox(Feature):
    """A body in a flow box (external), box sizes in body lengths as today's --ahead/--behind/
    --above/--below; declares its own inlet, outlet, farfield and body patches."""

    kind = "BodyInBox"

    inlet: EdgeRef
    outlet: EdgeRef
    farfield: EdgeRef
    body_edge: EdgeRef

    def __init__(self, body: Feature, *, ahead: float = 2.0, behind: float = 5.0, above: float = 2.0,
                 below: float = 2.0, far: Literal["slip", "symmetry"] = "slip", name: str | None = None):
        who = f"BodyInBox {name!r}" if name else "BodyInBox"
        self.body = _as_feature(body, "BodyInBox")
        self.ahead = _number(ahead, "ahead", who, positive=True)
        self.behind = _number(behind, "behind", who, positive=True)
        self.above = _number(above, "above", who, positive=True)
        self.below = _number(below, "below", who, positive=True)
        if far not in ("slip", "symmetry"):
            raise SketchError("E-ARGS", who, f"far={far!r} is not one of slip, symmetry")
        self.far = far
        self.inlet = EdgeRef(self, "inlet")
        self.outlet = EdgeRef(self, "outlet")
        self.farfield = EdgeRef(self, "farfield")
        self.body_edge = EdgeRef(self, "body")
        super().__init__(name)

    def operands(self) -> tuple[Feature, ...]:
        return (self.body,)


API: dict[str, object] = {
    "Sketch": Sketch, "Rect": Rect, "Disk": Disk, "Polygon": Polygon, "Band": Band,
    "Outline": Outline, "Passage": Passage, "Bypass": Bypass, "Row": Row,
    "Serpentine": Serpentine, "BodyInBox": BodyInBox, "EdgeRef": EdgeRef, "math": math,
}
"""The names bound into the script namespace."""


# -- the reference card ---------------------------------------------------------------------


def _sig(obj, name: str | None = None) -> str:
    """One line: the callable's name and its signature, `self` and `**alias` dropped."""
    fn = obj.__init__ if inspect.isclass(obj) else obj
    params = [p for p in inspect.signature(fn).parameters.values()
              if p.name not in ("self", "cls") and p.kind is not inspect.Parameter.VAR_KEYWORD]
    text = str(inspect.Signature(params)).replace("'", "")
    text = text.replace("tuple[float, float] | tuple[WallRef, float]", "(x, y) | (wall, u)")
    text = text.replace("tuple[float, float]", "(x, y)").replace("list[(x, y)]", "[(x, y), ...]")
    text = text.replace("Literal[", "").replace("]", "]") if "Literal[" not in text else _strip_literal(text)
    text = text.replace(" -> ", " -> ")
    return f"{name or getattr(obj, '__name__', str(obj))}{text}"


def _strip_literal(text: str) -> str:
    out = ""
    i = 0
    while i < len(text):
        if text.startswith("Literal[", i):
            j = text.index("]", i)
            inner = text[i + len("Literal["):j].replace('"', "").replace(" ", "")
            out += inner.replace(",", " | ")
            i = j + 1
        else:
            out += text[i]
            i += 1
    return out


_CARD_LINES = (
    ("Sketch", "one per script; every length in `units`, every angle in degrees counter-clockwise from +x"),
    ("s.fluid = <feature>", "the one region that is meshed (a feature, a boolean of features, a Row)"),
    ("s.rect", "a rectangle by its lower-left origin or its centre; sides .left .right .top .bottom"),
    ("s.disk", "a circle by centre and radius or diameter; .edge names its boundary"),
    ("s.polygon", "a closed polygon from three or more points; .edge"),
    ("s.annulus", "a ring; s.band(...) an arc of a ring between two angles; .edge"),
    ("s.outline", "a closed x,y outline file (csv or Selig .dat), scaled to `size` across x; .edge"),
    ("s.passage", "a constant-width channel; the same as Passage(...)"),
    ("s.apart", "parts that must neither overlap nor touch; two Rows, or a Row and a part that is not its host, are apart by default"),
    ("s.inlet / s.outlet", "the ports by intent (main.start, duct.left) or by rule string; one edge or several"),
    ("s.wall / s.patch", "a named wall (cyl.edge) or a slip / symmetry patch on edges"),
    ("s.expect", "how many inlets and outlets the shape has (default one of each)"),
    ("s.note", "a line printed back under SCRIPT, never parsed"),
)


def api_summary() -> str:
    """The reference card (about 120 lines): one signature and one sentence per name, the
    rule strings, the units, the flow conventions, no shape example (invariant 1)."""
    L: list[str] = []
    L.append("THE GEOMETRY API (reference card)")
    L.append("")
    L.append("A script is plain Python against these names, already bound: Sketch, Rect, Disk, Polygon,")
    L.append("Band, Outline, Passage, Bypass, Row, Serpentine, BodyInBox, EdgeRef, math. No import is")
    L.append("needed and none but math and json is allowed. Lengths are in the sketch's units; angles in")
    L.append("degrees, counter-clockwise from +x (0 = +x, 90 = +y). center= is accepted for centre=.")
    L.append("The tool solves what follows from what is stated (footprints, landings, pitches, fillets)")
    L.append("and prints every number back; a refusal carries the numbers to choose from.")
    L.append("")
    L.append("Sketch")
    L.append(f"  {_sig(Sketch)}")
    for name, sentence in _CARD_LINES:
        L.append(f"  {name:<22} {sentence}")
    L.append(f"  {_sig(Sketch.rect, 's.rect')}")
    L.append(f"  {_sig(Sketch.disk, 's.disk')}")
    L.append(f"  {_sig(Sketch.polygon, 's.polygon')}")
    L.append(f"  {_sig(Sketch.annulus, 's.annulus')}")
    L.append(f"  {_sig(Sketch.band, 's.band')}")
    L.append(f"  {_sig(Sketch.outline, 's.outline')}")
    L.append(f"  {_sig(Sketch.passage, 's.passage')}")
    L.append(f"  {_sig(Sketch.apart, 's.apart')}")
    L.append(f"  {_sig(Sketch.inlet, 's.inlet')}   {_sig(Sketch.outlet, 's.outlet')}")
    L.append(f"  {_sig(Sketch.wall, 's.wall')}   {_sig(Sketch.patch, 's.patch')}")
    L.append(f"  {_sig(Sketch.expect, 's.expect')}   {_sig(Sketch.note, 's.note')}")
    L.append("")
    L.append("Features are values: a | b fuses, a - b cuts, a & b intersects; f.moved(dx, dy), f.rotated(deg,")
    L.append("about=None), f.mirrored(axis, at=0, keep=False) return new features (axis \"x\" mirrors across the")
    L.append("line x = at); f.named(name) registers a name. Every feature a claim names needs name=.")
    L.append("Only a Passage's leg methods change it in place, and they return it, so p.line(30) and")
    L.append("p = p.line(30) both work.")
    L.append("")
    L.append("Rect      " + _sig(Rect))
    L.append("  sides .left .right .top .bottom are edges (ports) and walls (a loop or a branch can hang on")
    L.append("  one when the flow along the rect is known: an inlet/outlet declared on the rect, or flow=).")
    L.append("  Measures by role: width is the shorter side, length the longer, size_x / size_y the page sizes.")
    L.append("Disk      " + _sig(Disk))
    L.append("Polygon   " + _sig(Polygon) + "   (three or more points; made counter-clockwise)")
    L.append("Band      " + _sig(Band) + "   (an annulus when start..end spans 360)")
    L.append("Outline   " + _sig(Outline))
    L.append("")
    L.append("Passage   " + _sig(Passage))
    L.append("  start is a point, or (wall, u): a branch that leaves the wall at u along it, cut flush there;")
    L.append("  heading then defaults to the wall's outward normal. Legs, in order, each returning the passage:")
    L.append(f"  {_sig(Passage.line, '.line'):<44} a straight leg along the current heading")
    L.append(f"  {_sig(Passage.arc, '.arc'):<44} a bend of centreline radius, turn + left / - right (radius > width/2)")
    L.append(f"  {_sig(Passage.turn, '.turn'):<44} a sharp mitred corner between two line legs, |deg| < 180")
    L.append(f"  {_sig(Passage.line_to, '.line_to'):<44} run along the heading until x= or y= (an open end, not a landing)")
    L.append(f"  {_sig(Passage.turn_to, '.turn_to'):<44} the arc to an absolute heading; auto = the shorter way (180 needs side=)")
    L.append(f"  {_sig(Passage.u_turn, '.u_turn'):<44} a 180-degree arc; left = the +y side for a +x heading")
    L.append(f"  {_sig(Passage.end_on, '.end_on'):<44} the last leg: run to the wall's line and cut flush there (a landing)")
    L.append(f"  {_sig(Passage.through, 'Passage.through')}")
    L.append("                                               absolute waypoints, every corner filleted at radius")
    L.append("  .start .end are the caps (ports); .top .bottom (page words) or .side(\"left\"|\"right\"|\"top\"|\"bottom\",")
    L.append("  leg=None) name a straight leg's wall (leg counts every leg from 1; needed when there are several).")
    L.append("  .end_point .end_heading .length are solved for the script's own prints.")
    L.append("")
    L.append("Bypass    " + _sig(Bypass))
    L.append("  a loop that leaves `wall` at `at` (u along the wall) at leave_angle from the flow, sweeps round an")
    L.append("  arc whose OUTER wall has outer_radius, and returns onto the same wall at return_angle measured from")
    L.append("  the upstream direction (90 = straight back), so it always heads against the flow. The tool solves")
    L.append("  the arc, the landing, the lip and the footprint. leave_length None = one width. at= is required")
    L.append("  outside a Row and refused inside one (the Row places its instances). outer_radius > width.")
    L.append("")
    L.append("Row       " + _sig(Row))
    L.append("  count copies of item along a wall (along= inferred from a Bypass's wall) or a direction (dx, dy).")
    L.append("  Give gap= only when the request states one: otherwise the largest uniform gap that fits is solved")
    L.append("  and printed. pitch = footprint along the row + gap. margin defaults to half the item's width;")
    L.append("  align centres the row on the wall, start= fixes the first anchor. When the row does not fit, the")
    L.append("  refusal lists the footprint at five values of the free parameter and the largest gap that fits.")
    L.append("  row[k] is instance k, named row[k] (0-based) everywhere the tool prints.")
    L.append("")
    L.append("Serpentine " + _sig(Serpentine))
    L.append("  passes straight passes joined by 180-degree bends of centreline bend_radius, stacked along stack")
    L.append("  (across the heading); 2 * bend_radius > width. .start .end; .side(...) names a pass's wall.")
    L.append("  An even number of passes ends on the side it started from.")
    L.append("")
    L.append("BodyInBox " + _sig(BodyInBox))
    L.append("  a closed body in a flow box (external flow); box sizes in body lengths; it declares its own")
    L.append("  inlet, outlet, farfield and body patches (.inlet .outlet .farfield .body_edge).")
    L.append("")
    L.append("Ports and rule strings")
    L.append("  s.inlet(main.start), s.outlet(main.end), s.wall(cyl.edge, name=\"cylinder\"), s.patch(duct.top,")
    L.append("  name=\"lid\", kind=\"slip\"). A rule string instead of an edge: \"x:min\" / \"y:max\" (every edge flat")
    L.append("  there), \"x:0.05\" (flat on that line), \"near:x,y\" (the one edge whose middle is nearest), \"box:x0,y0,x1,y1\",")
    L.append("  \"normal:-x\" (the one open end facing that way). Name the ports by intent, never by coordinates.")
    L.append("")
    L.append("Walls and flow")
    L.append("  A wall has a frame: u runs along the flow from its upstream end (0) to its downstream end (its")
    L.append("  span), v outward. Bypass.at, Row.start, a branch's start=(wall, u) and every footprint are in u.")
    L.append("  A Passage leg's flow is its heading; a Rect side's is read from the inlet/outlet declared on the")
    L.append("  rect anywhere in the script, or from flow=\"+x\" (E-FLOW otherwise).")
    L.append("")
    L.append("Refusals (the code, the numbers, the fix; nothing is built): E-ROW-FIT, E-ROW-PITCH, E-ROW-ALONG,")
    L.append("  E-ROW-GAP-REQUIRED, E-ROW-DIRECTION, E-ROW-ITEM-AT, E-BYPASS-AT, E-LAND-OFF-WALL, E-RADIUS, E-ANGLE,")
    L.append("  E-FLOW, E-SIDE-AMBIGUOUS, E-EMPTY-PASSAGE, E-TURN-AMBIGUOUS, E-LINE-TO, E-CORNER, E-FILLET,")
    L.append("  E-SERP-RADIUS, E-NO-FLUID. After the build the lint judges overlap, touch, landings, notches, slivers,")
    L.append("  ports and units, and the compliance table judges every claim.")
    L.append("")
    L.append("Units: Sketch(units=\"mm\" | \"cm\" | \"m\" | \"in\"); use the request's own unit. The case is written in")
    L.append("metres by the tool; the print-back stays in the sketch's units.")
    return "\n".join(L) + "\n"
