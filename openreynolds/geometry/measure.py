"""The instruments: the boundary walk and everything read from the built face.

Every fact about gmsh that shaped this module was run on this box (DESIGN.md 3.5, and
the probes of 2026-09-07): `getBoundary(oriented=True)` carries the wire's signs but is
not promised to be in walk order, and `getCurveLoops` gives the loops unsigned, so
`walk()` chains each loop by end points, takes the first curve's sense from the oriented
boundary, and fixes the direction by signed area; `isInside` answers 1 on the boundary
and 1e-9 outside (0 at 1e-7), and it is wrong on any face that carries a dilate, mirror or
affine (1 for a point 5 m outside a dilated 60 x 3 rect, 1 for 20 mm outside a mirrored
one), so the kernel runs at scale 1 on untransformed faces; a curve's kind is by
curvature, never by the type string (an `addDisk` edge is "Ellipse", every transformed
curve "TrimmedCurve"); bounds come from sampled curves, never `getBoundingBox`, which is
loose after a dilate. gmsh is passed in, never imported here.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

from . import _toolbox  # noqa: F401  (mesh2d.parse_where by path)

if TYPE_CHECKING:
    from .claims import ClaimSet
    from .compile import Plan
    from .lint import Finding

SAMPLES = 64
"""Points per curve in the walk: enough that a 240-degree arc's polyline is within
1e-4 of the arc at the radii the fixtures use, and what the bounds and areas read."""

CUSP_INFO_DEG = 30.0
"""The vertex kinds fluid_cusp / solid_lip are typed at this angle (LintConfig.cusp_info_deg
carries the same number; the lint judges against its own config, the walk only labels)."""

STRAIGHT_DEG = 0.01
"""A turn smaller than this at a vertex is a tangent joint (a line meeting its arc)."""

UNITS_BY_SCALE = {0.001: "mm", 0.01: "cm", 0.0254: "in", 1.0: "m"}
"""A `--spec` fixture's unit, read from its scale (DESIGN.md 3.6, the units row)."""

SCALE_BY_UNIT = {"mm": 1e-3, "cm": 1e-2, "in": 0.0254, "m": 1.0}


def _pt(p) -> tuple[float, float] | None:
    """JSON gives lists back where the field says a point."""
    return None if p is None else (float(p[0]), float(p[1]))


@dataclass
class CurveInfo:
    tag: int
    kind: Literal["line", "arc", "curve"]
    """By curvature at three parameters: |k| < 1e-9/w_min -> line; equal within 1e-6 rel -> arc."""
    length: float
    start: tuple[float, float]
    end: tuple[float, float]
    """In walk direction (fluid on the left)."""
    heading_in: float
    heading_out: float
    radius: float | None
    centre: tuple[float, float] | None
    sweep: float | None
    patch: str
    """Resolved patch name ("walls" until resolved)."""
    loop: int
    """0 = outer, 1.. = holes."""
    midpoint: tuple[float, float]
    """getValue at the mid parameter."""
    inward_normal: tuple[float, float]
    """At the midpoint."""
    samples: list[tuple[float, float]] = field(default_factory=list)
    """64 points along the curve."""

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "CurveInfo":
        return CurveInfo(tag=d["tag"], kind=d["kind"], length=d["length"], start=_pt(d["start"]),
                         end=_pt(d["end"]), heading_in=d["heading_in"], heading_out=d["heading_out"],
                         radius=d.get("radius"), centre=_pt(d.get("centre")), sweep=d.get("sweep"),
                         patch=d.get("patch", "walls"), loop=d.get("loop", 0), midpoint=_pt(d["midpoint"]),
                         inward_normal=_pt(d["inward_normal"]),
                         samples=[_pt(p) for p in d.get("samples", [])])


@dataclass
class VertexInfo:
    at: tuple[float, float]
    curves: tuple[int, int]
    """Incoming, outgoing."""
    interior_deg: float
    """Of the FLUID: 180 - signed turn, in (0, 360)."""
    solid_deg: float
    """360 - interior_deg: the wall's angle at a lip."""
    kind: Literal["convex", "straight", "reflex", "fluid_cusp", "solid_lip"]
    """fluid_cusp: interior < cfg.cusp_info_deg (W-CUSP / I-CUSP: a wedge of fluid);
    solid_lip: solid_deg < cfg.cusp_info_deg (I-LIP: a knife-edge wall, the Tesla lip at
    28.9, an aerofoil trailing edge); on T01 every mouth corner is reflex (interior
    200..340) and no fluid_cusp exists (D31)."""

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "VertexInfo":
        return VertexInfo(at=_pt(d["at"]), curves=(int(d["curves"][0]), int(d["curves"][1])),
                          interior_deg=d["interior_deg"], solid_deg=d["solid_deg"], kind=d["kind"])


@dataclass
class Walk:
    loops: list[list[int]]
    """Curve tags per loop in walk order, outer first, fluid on the left after repair."""
    curves: dict[int, CurveInfo]
    vertices: list[VertexInfo]
    signed_area: float
    """Of the outer loop's sampling, positive after repair (a test walks a mirrored rect)."""
    repaired: bool
    """True when a loop's direction had to be reversed (a transformed face; the canary then refuses)."""

    def as_dict(self) -> dict:
        return {"loops": self.loops, "curves": {str(k): v.as_dict() for k, v in self.curves.items()},
                "vertices": [v.as_dict() for v in self.vertices], "signed_area": self.signed_area,
                "repaired": self.repaired}

    @staticmethod
    def from_dict(d: dict) -> "Walk":
        return Walk(loops=[list(loop) for loop in d.get("loops", [])],
                    curves={int(k): CurveInfo.from_dict(v) for k, v in d.get("curves", {}).items()},
                    vertices=[VertexInfo.from_dict(v) for v in d.get("vertices", [])],
                    signed_area=d["signed_area"], repaired=bool(d.get("repaired", False)))

    def vertex_after(self, curve: int) -> VertexInfo | None:
        """The vertex at the end of `curve` in walk direction."""
        return next((v for v in self.vertices if v.curves[0] == curve), None)

    def vertex_before(self, curve: int) -> VertexInfo | None:
        """The vertex at the start of `curve` in walk direction."""
        return next((v for v in self.vertices if v.curves[1] == curve), None)

    def bounds(self) -> tuple[float, float, float, float]:
        """Tight bounds of every sample (Finding F)."""
        xs = [p[0] for c in self.curves.values() for p in c.samples]
        ys = [p[1] for c in self.curves.values() for p in c.samples]
        return (min(xs), min(ys), max(xs), max(ys))

    def span(self) -> float:
        b = self.bounds()
        return max(b[2] - b[0], b[3] - b[1])

    def loop_area(self, k: int) -> float:
        """Signed area of loop k's sampled polygon."""
        return _signed_area(self.loop_points(k))

    def loop_points(self, k: int) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        for c in self.loops[k]:
            pts += self.curves[c].samples[:-1]
        return pts


@dataclass
class OpenEnd:
    curve: int
    centre: tuple[float, float]
    length: float
    outward_normal: tuple[float, float]
    depth: float
    """Fluid behind it along the inward normal, capped at 3 * length."""
    corner_angles: tuple[float, float]
    name: str | None = None
    """After rule resolution."""

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "OpenEnd":
        return OpenEnd(curve=d["curve"], centre=_pt(d["centre"]), length=d["length"],
                       outward_normal=_pt(d["outward_normal"]), depth=d["depth"],
                       corner_angles=(d["corner_angles"][0], d["corner_angles"][1]), name=d.get("name"))


@dataclass
class Junction:
    channel: str
    where: tuple[float, float]
    wall_line: str
    kind: Literal["leave", "return"]
    typed_angle: float | None
    """From the leg record (heading against the wall)."""
    measured_angle: float
    """The angle between the wall's flow direction and the leg's STRAIGHT WALL CURVE on
    the walk (the curve whose samples lie on the compiled leg's wall line within touch
    tol; it may be on the island loop): 20.0 for T01's leave leg (its upstream wall), 80.0
    for the return leg (its upstream wall from (1.50, 1.5) to (2.61, 7.80)). NEVER from
    vertices on the wall line: two of T01's four mouth corners are interior points because
    the mouths merge (3.3)."""
    lip: tuple[tuple[float, float], float] | None
    """The downstream lip vertex on the wall line and its solid_deg (28.9 at (11.42, 1.5)
    for the T01 leave junction), None when the corner merged into another mouth."""
    heading: float
    against_flow: bool | None
    """Sign of the heading's component along the flow; None when the flow is unknown."""

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Junction":
        lip = d.get("lip")
        return Junction(channel=d["channel"], where=_pt(d["where"]), wall_line=d["wall_line"], kind=d["kind"],
                        typed_angle=d.get("typed_angle"), measured_angle=d["measured_angle"],
                        lip=None if lip is None else (_pt(lip[0]), float(lip[1])),
                        heading=d["heading"], against_flow=d.get("against_flow"))


@dataclass
class PassageWidths:
    min: float
    where_min: tuple[float, float]
    p10: float
    median: float
    n: int

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "PassageWidths":
        return PassageWidths(min=d["min"], where_min=_pt(d["where_min"]), p10=d["p10"], median=d["median"],
                             n=d["n"])


@dataclass
class RowMeasure:
    name: str
    count: int
    pitch: float
    footprint: tuple[float, float]
    gap: float | None
    """Declared (or solved) bbox gap."""
    gap_measured: float | None
    """Least wall distance between consecutive instances."""
    anchors: list[float]
    span: tuple[float, float]

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "RowMeasure":
        return RowMeasure(name=d["name"], count=d["count"], pitch=d["pitch"], footprint=_pt(d["footprint"]),
                          gap=d.get("gap"), gap_measured=d.get("gap_measured"), anchors=list(d.get("anchors", [])),
                          span=_pt(d["span"]))


@dataclass
class HoleMeasure:
    loop: int
    area: float
    centroid: tuple[float, float]
    radius: float | None
    """When the loop is one curve of constant curvature: 1/k."""
    circumference: float

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "HoleMeasure":
        return HoleMeasure(loop=d["loop"], area=d["area"], centroid=_pt(d["centroid"]), radius=d.get("radius"),
                           circumference=d["circumference"])


@dataclass
class Measurements:
    units: str
    scale: float
    bounds: tuple[float, float, float, float]
    """From sampled curves (Finding F), never getBoundingBox."""
    extent: tuple[float, float]
    area: float
    islands: int
    n_curves: int
    shortest_edge: tuple[float, tuple[float, float]]
    patches: dict[str, dict]
    """name -> {"curves": [...], "n": int, "length": float, "midpoints": [...]}."""
    legs: dict[str, list[dict]]
    """Channel name -> leg records (mesh2d's, unchanged)."""
    features: dict[str, dict]
    """Per feature: the keys claims.MEASURES reads (the table in DESIGN.md 3.5)."""
    rows: dict[str, RowMeasure]
    open_ends: list[OpenEnd]
    junctions: list[Junction]
    vertices: list[VertexInfo]
    passage: PassageWidths | None
    reference_width: float
    reference_width_from: str
    """"declared by 'main'" | "claim c1" | "hole diameter (cyl)" | "p10 of passage samples" | "extent / 10"."""
    holes: list[HoleMeasure]
    flow: tuple[float, float] | None
    """The inlet's inward normal, once resolved."""

    def as_dict(self) -> dict:
        return {
            "units": self.units, "scale": self.scale, "bounds": list(self.bounds), "extent": list(self.extent),
            "area": self.area, "islands": self.islands, "n_curves": self.n_curves,
            "shortest_edge": [self.shortest_edge[0], list(self.shortest_edge[1])],
            "patches": self.patches, "legs": self.legs, "features": self.features,
            "rows": {k: v.as_dict() for k, v in self.rows.items()},
            "open_ends": [e.as_dict() for e in self.open_ends],
            "junctions": [j.as_dict() for j in self.junctions],
            "vertices": [v.as_dict() for v in self.vertices],
            "passage": None if self.passage is None else self.passage.as_dict(),
            "reference_width": self.reference_width, "reference_width_from": self.reference_width_from,
            "holes": [h.as_dict() for h in self.holes],
            "flow": None if self.flow is None else list(self.flow),
        }

    @staticmethod
    def from_dict(d: dict) -> "Measurements":
        b = d["bounds"]
        return Measurements(
            units=d["units"], scale=d["scale"], bounds=(b[0], b[1], b[2], b[3]), extent=_pt(d["extent"]),
            area=d["area"], islands=d["islands"], n_curves=d["n_curves"],
            shortest_edge=(d["shortest_edge"][0], _pt(d["shortest_edge"][1])),
            patches=dict(d.get("patches", {})), legs=dict(d.get("legs", {})), features=dict(d.get("features", {})),
            rows={k: RowMeasure.from_dict(v) for k, v in d.get("rows", {}).items()},
            open_ends=[OpenEnd.from_dict(e) for e in d.get("open_ends", [])],
            junctions=[Junction.from_dict(j) for j in d.get("junctions", [])],
            vertices=[VertexInfo.from_dict(v) for v in d.get("vertices", [])],
            passage=None if d.get("passage") is None else PassageWidths.from_dict(d["passage"]),
            reference_width=d["reference_width"], reference_width_from=d["reference_width_from"],
            holes=[HoleMeasure.from_dict(h) for h in d.get("holes", [])],
            flow=_pt(d.get("flow")),
        )


# -- plain geometry ------------------------------------------------------------------


def _signed_area(pts) -> float:
    n = len(pts)
    if n < 3:
        return 0.0
    return 0.5 * sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n))


def _centroid(pts) -> tuple[float, float]:
    a = _signed_area(pts)
    n = len(pts)
    if abs(a) < 1e-300:
        return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)
    cx = cy = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        w = x0 * y1 - x1 * y0
        cx += (x0 + x1) * w
        cy += (y0 + y1) * w
    return (cx / (6 * a), cy / (6 * a))


def _norm180(deg: float) -> float:
    """Into (-180, 180]."""
    d = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def _heading(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dy, dx)) % 360.0


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _unit(v) -> tuple[float, float]:
    n = math.hypot(v[0], v[1])
    return (v[0] / n, v[1] / n) if n > 0 else (0.0, 0.0)


def point_segment_distance(p, a, b) -> float:
    """Distance from p to the segment a-b."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 <= 0:
        return _dist(p, a)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / l2))
    return _dist(p, (ax + t * dx, ay + t * dy))


def point_line_distance(p, a, direction) -> float:
    """Distance from p to the infinite line through a with a unit `direction`."""
    return abs((p[0] - a[0]) * direction[1] - (p[1] - a[1]) * direction[0])


def _polyline_distance(p, pts: list, closed: bool) -> float:
    n = len(pts)
    rng = range(n) if closed else range(n - 1)
    return min(point_segment_distance(p, pts[i], pts[(i + 1) % n]) for i in rng)


def _turning_deg(pts) -> float:
    """Total signed turning of an open polyline, degrees, + left."""
    total = 0.0
    prev = None
    for i in range(len(pts) - 1):
        seg = (pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        if math.hypot(*seg) <= 0:
            continue
        h = _heading(*seg)
        if prev is not None:
            total += _norm180(h - prev)
        prev = h
    return total


# -- gmsh readers --------------------------------------------------------------------


def _params(gmsh, curve: int, n: int) -> list[float]:
    b = gmsh.model.getParametrizationBounds(1, curve)
    t0, t1 = float(b[0][0]), float(b[1][0])
    return [t0 + (t1 - t0) * i / (n - 1) for i in range(n)]


def _values(gmsh, curve: int, ts: list[float]) -> list[tuple[float, float]]:
    xyz = gmsh.model.getValue(1, curve, ts)
    return [(float(xyz[3 * i]), float(xyz[3 * i + 1])) for i in range(len(ts))]


def _tangent(gmsh, curve: int, t: float) -> tuple[float, float]:
    d = gmsh.model.getDerivative(1, curve, [t])
    return _unit((float(d[0]), float(d[1])))


def _face_of(gmsh, wk: Walk) -> int:
    """The face the walk was taken on: the upward adjacency of its first curve (the walk
    dataclass carries no face tag, and after `build_face` the stray faces are gone)."""
    up, _ = gmsh.model.getAdjacencies(1, wk.loops[0][0])
    return int(up[0])


def sampled_bounds(gmsh, dim: int, tag: int, n: int = 64) -> tuple[float, float, float, float]:
    """Tight bounds from `getValue` at n parameters per boundary curve plus the vertices.
    The same function is added to mesh2d.py (section 3.17); the two are pinned equal by a test."""
    curves = [tag] if dim == 1 else [abs(int(t)) for _, t in gmsh.model.getBoundary([(2, tag)], combined=False,
                                                                                  oriented=False)]
    xs: list[float] = []
    ys: list[float] = []
    for c in curves:
        for x, y in _values(gmsh, c, _params(gmsh, c, n)):
            xs.append(x)
            ys.append(y)
        for _, v in gmsh.model.getBoundary([(1, c)], oriented=False):
            xyz = gmsh.model.getValue(0, abs(int(v)), [])
            xs.append(float(xyz[0]))
            ys.append(float(xyz[1]))
    return (min(xs), min(ys), max(xs), max(ys))


def walk(gmsh, face: int, w_min: float) -> Walk:
    """The boundary walk, computed once and shared by lint and measure: loops from
    `occ.getCurveLoops`, each chained by end points (the vertex dimTags of
    `getBoundary([(1, c)])`, which come back unsigned), the first curve of each loop taken
    in the sense the oriented face boundary gives it, every next curve oriented so the
    chain closes, then the direction fixed by signed area (outer positive, holes negative)
    and `repaired` set when a reversal was needed (preamble). On an untransformed face the
    wire's senses already put the fluid on the left (T01: +651 / -34.4; a rect minus a disk:
    outer +, hole -); a mirrored rect's wire runs clockwise and is repaired."""
    occ = gmsh.model.occ
    _, loops = occ.getCurveLoops(face)
    senses = {abs(int(t)): (1 if int(t) > 0 else -1)
              for _, t in gmsh.model.getBoundary([(2, face)], combined=False, oriented=True)}
    raw: dict[int, dict] = {}
    for loop in loops:
        for t in loop:
            c = abs(int(t))
            ts = _params(gmsh, c, SAMPLES)
            pts = _values(gmsh, c, ts)
            raw[c] = {"ts": ts, "pts": pts}
    span = 0.0
    xs = [p[0] for r in raw.values() for p in r["pts"]]
    ys = [p[1] for r in raw.values() for p in r["pts"]]
    if xs:
        span = max(max(xs) - min(xs), max(ys) - min(ys))
    tol = 1e-6 * max(span, 1e-12) + 1e-12

    chains: list[list[tuple[int, bool]]] = []
    for loop in loops:
        tags = [abs(int(t)) for t in loop]
        first = tags[0]
        forward = senses.get(first, 1) > 0
        chain = [(first, forward)]
        end = raw[first]["pts"][-1] if forward else raw[first]["pts"][0]
        remaining = tags[1:]
        while remaining:
            best = None
            for c in remaining:
                p0, p1 = raw[c]["pts"][0], raw[c]["pts"][-1]
                d0, d1 = _dist(end, p0), _dist(end, p1)
                cand = (d0, c, True) if d0 <= d1 else (d1, c, False)
                if best is None or cand[0] < best[0]:
                    best = cand
            _, c, fwd = best
            chain.append((c, fwd))
            end = raw[c]["pts"][-1] if fwd else raw[c]["pts"][0]
            remaining.remove(c)
        chains.append(chain)

    def chain_points(chain):
        pts = []
        for c, fwd in chain:
            p = raw[c]["pts"]
            pts += (p if fwd else p[::-1])[:-1]
        return pts

    areas = [_signed_area(chain_points(ch)) for ch in chains]
    order = sorted(range(len(chains)), key=lambda i: -abs(areas[i]))
    repaired = False
    ordered: list[list[tuple[int, bool]]] = []
    for rank, i in enumerate(order):
        chain = chains[i]
        want_positive = rank == 0
        if (areas[i] > 0) != want_positive:
            chain = [(c, not fwd) for c, fwd in reversed(chain)]
            repaired = True
        ordered.append(chain)

    curves: dict[int, CurveInfo] = {}
    loops_out: list[list[int]] = []
    line_k = 1e-9 / max(w_min, 1e-12)
    for k, chain in enumerate(ordered):
        loops_out.append([c for c, _ in chain])
        for c, fwd in chain:
            ts = raw[c]["ts"]
            pts = raw[c]["pts"] if fwd else raw[c]["pts"][::-1]
            t_mid = (ts[0] + ts[-1]) / 2
            t_start, t_end = (ts[0], ts[-1]) if fwd else (ts[-1], ts[0])
            sign = 1.0 if fwd else -1.0
            tan_in = _tangent(gmsh, c, t_start)
            tan_out = _tangent(gmsh, c, t_end)
            tan_mid = _tangent(gmsh, c, t_mid)
            heading_in = _heading(sign * tan_in[0], sign * tan_in[1])
            heading_out = _heading(sign * tan_out[0], sign * tan_out[1])
            ks = [abs(float(v)) for v in gmsh.model.getCurvature(1, c, [ts[0], t_mid, ts[-1]])]
            kmax, kmin = max(ks), min(ks)
            if kmax < line_k:
                kind = "line"
            elif kmax - kmin <= 1e-6 * kmax:
                kind = "arc"
            else:
                kind = "curve"
            mid = _values(gmsh, c, [t_mid])[0]
            left = (-sign * tan_mid[1], sign * tan_mid[0])
            length = float(gmsh.model.occ.getMass(1, c))
            radius = centre = sweep = None
            if kind == "arc":
                radius = 1.0 / ((kmax + kmin) / 2)
                # The magnitude is arc length over radius: the turning of the sampled
                # chords misses one chord's share (240 deg reads 236.2 at 64 samples);
                # the chords give only the sense.
                sweep = math.copysign(math.degrees(length / radius), _turning_deg(pts) or 1.0)
                if abs(abs(sweep) - 360.0) < 1.0 and _dist(pts[0], pts[-1]) < tol:
                    sweep = math.copysign(360.0, sweep)
                side = 1.0 if sweep >= 0 else -1.0
                centre = (mid[0] + side * radius * left[0], mid[1] + side * radius * left[1])
            curves[c] = CurveInfo(tag=c, kind=kind, length=length,
                                  start=pts[0], end=pts[-1], heading_in=heading_in, heading_out=heading_out,
                                  radius=radius, centre=centre, sweep=sweep, patch="walls", loop=k,
                                  midpoint=mid, inward_normal=left, samples=pts)

    vertices: list[VertexInfo] = []
    for loop in loops_out:
        n = len(loop)
        for i, c in enumerate(loop):
            nxt = loop[(i + 1) % n]
            turn = _norm180(curves[nxt].heading_in - curves[c].heading_out)
            interior = 180.0 - turn
            if interior <= 0:
                interior += 360.0
            if interior >= 360.0:
                interior -= 360.0
            solid = 360.0 - interior
            if abs(turn) < STRAIGHT_DEG:
                kind = "straight"
            elif interior < CUSP_INFO_DEG:
                kind = "fluid_cusp"
            elif solid < CUSP_INFO_DEG:
                kind = "solid_lip"
            elif interior < 180.0:
                kind = "convex"
            else:
                kind = "reflex"
            vertices.append(VertexInfo(at=curves[c].end, curves=(c, nxt), interior_deg=interior, solid_deg=solid,
                                       kind=kind))
    wk = Walk(loops=loops_out, curves=curves, vertices=vertices, signed_area=0.0, repaired=repaired)
    wk.signed_area = wk.loop_area(0) if loops_out else 0.0
    return wk


def ray_depth(gmsh, face: int, p: tuple[float, float], n: tuple[float, float], max_len: float,
              steps: int, until: Literal["outside", "inside"] = "outside", resolution: float | None = None) -> float:
    """March p + (i/steps) * max_len * n, i = 1..steps, isInside per point, until the
    first point that is outside (until="outside": how much fluid lies ahead) or inside
    (until="inside": how much solid lies ahead, the notch predicate marching OUTWARD);
    then bisect between the last point before the change and the first after it down to
    `resolution` (default 1e-3 * max_len / 3; ten extra calls), and return that distance,
    or max_len when nothing changes. A point within 1e-7 of the boundary counts as on it
    (measured: isInside answers 1 on the boundary and 1e-9 outside, 0 at 1e-7).
    Without the bisection the coarse march is quantised to a step and biased one step
    high (T02's 2 mm channel read 2.33 at 18 steps over 3 w; T01's 3 mm read 3.5)."""
    if resolution is None:
        resolution = 1e-3 * max_len / 3.0
    want_inside = until == "inside"

    def inside(d: float) -> bool:
        return bool(gmsh.model.isInside(2, face, [p[0] + d * n[0], p[1] + d * n[1], 0.0]))

    lo = 0.0
    hi = None
    for i in range(1, steps + 1):
        d = max_len * i / steps
        if inside(d) == want_inside:
            hi = d
            break
        lo = d
    if hi is None:
        return max_len
    while hi - lo > resolution:
        mid = (lo + hi) / 2
        if inside(mid) == want_inside:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def open_ends(gmsh, wk: Walk, w_min: float, flow_box: tuple | None,
              angle_tol_deg: float = 30.0, depth_rel: float = 1.0) -> list[OpenEnd]:
    """A candidate is a curve that is (1) a line, (2) whose two end vertices are convex with
    interior angle within 90 +/- 30 deg, (3) with depth >= its own length along the inward
    normal (ray_depth until="outside"), (4) not on the flow box. A dead-end cap is a
    candidate; the rules decide."""
    face = _face_of(gmsh, wk)
    tol = 1e-6 * wk.span() + 1e-12
    out: list[OpenEnd] = []
    for loop in wk.loops:
        for c in loop:
            info = wk.curves[c]
            if info.kind != "line":
                continue
            before, after = wk.vertex_before(c), wk.vertex_after(c)
            if before is None or after is None:
                continue
            angles = (before.interior_deg, after.interior_deg)
            if any(abs(a - 90.0) > angle_tol_deg for a in angles):
                continue
            if flow_box is not None and _on_box_side(info.samples, flow_box, tol):
                continue
            depth = ray_depth(gmsh, face, info.midpoint, info.inward_normal, 3.0 * info.length, 18)
            if depth < depth_rel * info.length:
                continue
            out.append(OpenEnd(curve=c, centre=info.midpoint, length=info.length,
                               outward_normal=(-info.inward_normal[0], -info.inward_normal[1]),
                               depth=depth, corner_angles=angles))
    return out


def _on_box_side(samples, box, tol) -> bool:
    x0, y0, x1, y1 = box
    for axis, value in ((0, x0), (0, x1), (1, y0), (1, y1)):
        if all(abs(p[axis] - value) <= tol for p in samples):
            return True
    return False


def body_open_ends(gmsh, wk: Walk, w_min: float, angle_tol_deg: float = 30.0,
                   depth_rel: float = 1.0) -> list[OpenEnd]:
    """The open ends of a BodyInBox BODY, judged on the body's own face before the flow
    box is cut (E-PORT-ON-BODY): a passage drawn INTO the solid. On the body a slot's
    bottom edge has 270-degree ends (the solid wraps round it) and the slot's walls are
    at least as long as it is; marching OUTWARD from that edge, the ray runs through the
    slot and never re-enters the body. A plain rect body has no such edge, and on the
    fluid the same slot reads as a dead-end candidate on the hole loop (`open_ends`:
    convex fluid corners, deep behind), which `lint.judge` reports the same way."""
    face = _face_of(gmsh, wk)
    out: list[OpenEnd] = []
    for loop in wk.loops:
        for c in loop:
            info = wk.curves[c]
            if info.kind != "line":
                continue
            before, after = wk.vertex_before(c), wk.vertex_after(c)
            if before is None or after is None:
                continue
            angles = (before.interior_deg, after.interior_deg)
            if any(abs(a - 270.0) > angle_tol_deg for a in angles):
                continue
            walls = (wk.curves[before.curves[0]], wk.curves[after.curves[1]])
            if any(wall.length < depth_rel * info.length for wall in walls):
                continue
            outward = (-info.inward_normal[0], -info.inward_normal[1])
            free = ray_depth(gmsh, face, info.midpoint, outward, 3.0 * info.length, 18, until="inside")
            if free < 3.0 * info.length:
                continue
            out.append(OpenEnd(curve=c, centre=info.midpoint, length=info.length, outward_normal=outward,
                               depth=min(wall.length for wall in walls), corner_angles=angles))
    return out


def host_wall_span(wk: Walk, axis: int, value: float, outward: tuple[float, float],
                   flow: tuple[float, float], tol: float) -> tuple[float, tuple[float, float]] | None:
    """The span of a host wall on the BUILT face, for the landing's u clause (D22 c):
    every straight curve on the line axis=value whose inward normal points into the
    host (against `outward`; a return leg's flush cut hanging in the air lies on the
    same line with the fluid on the other side and is left out), projected on `flow`.
    Returns (span, upstream point) with u = 0 at the upstream end, or None when no
    curve of the walk lies on the line."""
    us: list[float] = []
    pts: list[tuple[float, float]] = []
    for info in wk.curves.values():
        if info.kind != "line" or any(abs(p[axis] - value) > tol for p in info.samples):
            continue
        if info.inward_normal[0] * outward[0] + info.inward_normal[1] * outward[1] > -0.5:
            continue
        for p in (info.start, info.end):
            us.append(p[0] * flow[0] + p[1] * flow[1])
            pts.append(p)
    if not us:
        return None
    lo = min(range(len(us)), key=lambda i: us[i])
    return max(us) - us[lo], pts[lo]


def passage_widths(gmsh, face: int, wk: Walk, w_min: float) -> PassageWidths:
    """Samples along every wall curve of length >= w_min at spacing w_min/4, skipping samples
    within w_min/2 of a vertex with interior < 90 deg; ray_depth inward, max 3 w_min, 18
    steps plus the bisection, resolution 1e-3 w_min. About 400 rays of up to 28 isInside
    calls on a valve at ~0.4 ms per call: ~4 s on this box; the sample spacing is halved
    (w_min/2) when the wall length sum exceeds 200 w_min so the budget stays under 5 s.
    The resolution is printed in MEASURED ("narrowest passage 3.000 (+/- 0.003)").
    Numbers for T01/T02 are pinned by `test_passage_widths_on_t01_and_t02` from the first
    run of this implementation."""
    total = sum(c.length for c in wk.curves.values() if c.length >= w_min)
    spacing = w_min / 4.0 if total <= 200 * w_min else w_min / 2.0
    sharp = [v.at for v in wk.vertices if v.interior_deg < 90.0]
    # A cap's inward ray runs along its channel and measures a length, not a width (a
    # 60 x 3 rect's two caps added eight rays of 9 to the sample); only wall curves are
    # sampled, and the open-end candidates are the caps whatever they are named.
    caps = {e.curve for e in open_ends(gmsh, wk, w_min, None)}
    depths: list[tuple[float, tuple[float, float]]] = []
    for info in wk.curves.values():
        if info.length < w_min or info.tag in caps:
            continue
        pts = info.samples
        cum = [0.0]
        for i in range(1, len(pts)):
            cum.append(cum[-1] + _dist(pts[i - 1], pts[i]))
        n = max(1, int(info.length / spacing))
        for j in range(n):
            s = (j + 0.5) * info.length / n
            i = max(1, min(len(pts) - 1, next((k for k in range(1, len(pts)) if cum[k] >= s), len(pts) - 1)))
            f = (s - cum[i - 1]) / max(cum[i] - cum[i - 1], 1e-300)
            p = (pts[i - 1][0] + f * (pts[i][0] - pts[i - 1][0]), pts[i - 1][1] + f * (pts[i][1] - pts[i - 1][1]))
            if any(_dist(p, v) < w_min / 2 for v in sharp):
                continue
            seg = _unit((pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
            normal = (-seg[1], seg[0])
            depths.append((ray_depth(gmsh, face, p, normal, 3.0 * w_min, 18, resolution=1e-3 * w_min), p))
    if not depths:
        return PassageWidths(min=3.0 * w_min, where_min=(0.0, 0.0), p10=3.0 * w_min, median=3.0 * w_min, n=0)
    depths.sort(key=lambda d: d[0])
    values = [d for d, _ in depths]
    n = len(values)
    return PassageWidths(min=values[0], where_min=depths[0][1], p10=values[min(n - 1, int(0.1 * n))],
                         median=values[n // 2], n=n)


# -- ports -------------------------------------------------------------------------------


def _rule_where(rule: dict):
    """A rule's `at`, parsed (mesh2d.parse_where) whether the plan carries the raw string
    or the parsed tuple."""
    at = rule.get("at")
    if at is None:
        return None
    if isinstance(at, str):
        return _toolbox.load("mesh2d").parse_where(at)
    return tuple(at)


def _rule_box(rule: dict):
    box = rule.get("box")
    return None if box is None else tuple(float(v) for v in box)


def _flat_curves(wk: Walk, axis: int, value: float, tol: float) -> list[int]:
    return [c for c, info in wk.curves.items() if all(abs(p[axis] - value) <= tol for p in info.samples)]


def _curves_on_polyline(wk: Walk, pts: list, closed: bool, tol: float) -> list[int]:
    """The walk curves whose samples all lie on the target polyline. A straight curve is
    held to `tol`; an arc is held to the sagitta of the polyline's longest chord at the
    arc's own radius (a 64-point outline of a radius-5 circle is 0.006 off the true
    circle, four orders above 1e-6 of a 300 mm span), so a Disk's `edge` resolves and a
    neighbouring straight wall does not."""
    out = []
    n = len(pts)
    chord = max((_dist(pts[i], pts[(i + 1) % n]) for i in (range(n) if closed else range(n - 1))), default=0.0)
    for c, info in wk.curves.items():
        allow = tol
        if info.kind != "line" and info.radius and chord <= 0.25 * info.radius:
            allow = max(tol, 1.5 * chord * chord / (8.0 * info.radius))
        if all(_polyline_distance(p, pts, closed) <= allow for p in info.samples):
            out.append(c)
    return out


def _normal_rule(text: str) -> tuple[float, float] | None:
    """'normal:-x' -> (-1, 0); None when the string is not a normal rule."""
    if not isinstance(text, str) or not text.startswith("normal:"):
        return None
    word = text.split(":", 1)[1].strip()
    table = {"+x": (1.0, 0.0), "x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "y": (0.0, 1.0),
             "-y": (0.0, -1.0)}
    return table.get(word)


def _port_kind(name: str, kind: str | None) -> str:
    if kind:
        return kind
    return name if name in ("inlet", "outlet") else "wall"


def resolve_ports(gmsh, wk: Walk, plan: "Plan", open_ends: list[OpenEnd]) -> tuple[dict[int, str], list[dict], list["Finding"]]:
    """EdgeRefs and rule strings -> (curve -> patch name, the resolved rules in today's
    grammar for the record, the findings E-PORT-MISSING / E-PORT-RULE / E-PORT-COUNT /
    E-PORT-OPEN). Reads `plan.edge_targets` for every EdgeRef (section 3.4's table) and
    never `plan.features`. Port intents are read first; a plan with no intents (an ops
    fixture, `cli check --spec`) is read from its `rules` in mesh2d's grammar. The
    automatic reading (flat at the two ends of the longest axis) is used only when no rule
    names inlet or outlet AND the topology finds exactly two candidates with opposite
    normals along the longest axis (an elbow's -x / +y ends refuse it: E-PORT-COUNT with
    the two ends named). The resolved rules name each curve by `near:` at its
    mid-parameter point (a point ON the curve), so `mesh2d.py` on the instance reproduces
    the classification with no new grammar (D8)."""
    from .lint import Finding, render

    findings: list[Finding] = []
    span = wk.span()
    tol = 1e-6 * span + 1e-12
    bounds = wk.bounds()
    curve_patch: dict[int, str] = {c: "walls" for c in wk.curves}
    kinds: dict[str, str] = {}
    ends_by_curve = {e.curve: e for e in open_ends}
    named_any = False

    def assign(curves, name):
        for c in curves:
            if curve_patch[c] == "walls":
                curve_patch[c] = name

    def edges_of(port) -> list:
        return list(port.edges) if hasattr(port, "edges") else []

    entries: list[tuple[str, str, object]] = []
    ports = list(getattr(plan, "ports", []) or [])
    if ports:
        for port in ports:
            for edge in edges_of(port):
                entries.append((port.name, port.kind, edge))
    else:
        for rule in list(getattr(plan, "rules", []) or []):
            name = rule["name"]
            kind = _port_kind(name, rule.get("kind"))
            if rule.get("box") is not None:
                entries.append((name, kind, ("box", _rule_box(rule))))
            else:
                at = rule.get("at")
                entries.append((name, kind, at if isinstance(at, str) and at.startswith("normal:") else ("at", _rule_where(rule))))

    for name, kind, edge in entries:
        kinds[name] = kind
        named_any = named_any or kind in ("inlet", "outlet")
        hit: list[int] = []
        if isinstance(edge, tuple) and edge and edge[0] == "box":
            x0, y0, x1, y1 = edge[1]
            hit = [c for c, info in wk.curves.items()
                   if all(x0 - tol <= p[0] <= x1 + tol and y0 - tol <= p[1] <= y1 + tol for p in info.samples)]
            if not hit:
                findings.append(render("error", "E-PORT-RULE", name, rule=f"box:{x0:g},{y0:g},{x1:g},{y1:g}",
                                       n=0, ends=_ends_text(open_ends), fix=_rule_fix(open_ends)))
        elif isinstance(edge, tuple) and edge and edge[0] == "at":
            axis, where, value = edge[1]
            if where == "near":
                px, py = value
                nearest = min(wk.curves, key=lambda c: _dist(wk.curves[c].midpoint, (px, py)))
                hit = [nearest]
            else:
                if where == "min":
                    value = bounds[axis]
                elif where == "max":
                    value = bounds[axis + 2]
                hit = _flat_curves(wk, axis, value, max(tol, 1e-4 * span))
                if not hit:
                    findings.append(render("error", "E-PORT-RULE", name, rule=f"{'xy'[axis]}:{where if where != 'at' else value:g}" if where == "at" else f"{'xy'[axis]}:{where}",
                                           n=0, ends=_ends_text(open_ends), fix=_rule_fix(open_ends)))
        elif isinstance(edge, str) and _normal_rule(edge) is not None:
            nx, ny = _normal_rule(edge)
            cands = [e for e in open_ends
                     if e.outward_normal[0] * nx + e.outward_normal[1] * ny >= math.cos(math.radians(20.0))]
            if len(cands) != 1:
                findings.append(render("error", "E-PORT-RULE", name, rule=edge, n=len(cands),
                                       ends=_ends_text(cands or open_ends), fix=_rule_fix(cands or open_ends)))
            else:
                hit = [cands[0].curve]
        elif isinstance(edge, str):
            entry = plan.edge_targets.get(edge) if getattr(plan, "edge_targets", None) else None
            if entry is None:
                try:
                    parsed = ("at", _toolbox.load("mesh2d").parse_where(edge))
                except SystemExit:
                    parsed = None
                if parsed is not None:
                    entries.append((name, kind, parsed))
                    continue
                findings.append(render("error", "E-PORT-MISSING", name, n_ends=len(open_ends), edge=edge, at="?",
                                       why="no edge target was compiled for it", host="the fluid"))
                continue
            hit = _resolve_target(wk, entry, ends_by_curve, tol, span)
            if not hit:
                pts = entry.get("points", [])
                at = _mid(pts)
                findings.append(render("error", "E-PORT-MISSING", name, n_ends=len(open_ends), edge=edge,
                                       at=f"({at[0]:.4g}, {at[1]:.4g})",
                                       why="the cap there was fused into its host", host="its host"))
        else:
            entry = plan.edge_targets.get(str(edge)) if getattr(plan, "edge_targets", None) else None
            if entry is None:
                findings.append(render("error", "E-PORT-MISSING", name, n_ends=len(open_ends), edge=str(edge), at="?",
                                       why="no edge target was compiled for it", host="the fluid"))
                continue
            hit = _resolve_target(wk, entry, ends_by_curve, tol, span)
            if not hit:
                at = _mid(entry.get("points", []))
                findings.append(render("error", "E-PORT-MISSING", name, n_ends=len(open_ends), edge=str(edge),
                                       at=f"({at[0]:.4g}, {at[1]:.4g})",
                                       why="the cap there was fused into its host", host="its host"))
        assign(hit, name)

    expected = tuple(getattr(plan, "expected", (1, 1)) or (1, 1))
    automatic_refused = False
    if not named_any:
        lo, hi = _automatic_ends(open_ends, bounds)
        if lo is not None and hi is not None:
            assign([lo.curve], "inlet")
            assign([hi.curve], "outlet")
            kinds.setdefault("inlet", "inlet")
            kinds.setdefault("outlet", "outlet")
        else:
            automatic_refused = True

    for e in open_ends:
        e.name = curve_patch.get(e.curve) if curve_patch.get(e.curve) != "walls" else None

    for want in ("inlet", "outlet"):
        names = [n for n, k in kinds.items() if k == want] or [want]
        edges = [c for c, n in curve_patch.items() if n in names]
        need = expected[0] if want == "inlet" else expected[1]
        if len(edges) == need:
            continue
        spots = [wk.curves[c].midpoint for c in edges]
        if automatic_refused and not edges:
            spots = [e.centre for e in open_ends]
        findings.append(render("error", "E-PORT-COUNT", want, want=want, n=len(edges), need=need,
                               ends=_ends_text_points(spots) if spots else "none",
                               where=spots[0] if spots else None,
                               fix=_count_fix(want, open_ends, automatic_refused)))
    if not automatic_refused:
        for e in open_ends:
            if e.name is None and wk.curves[e.curve].loop == 0:
                facing = _facing(e.outward_normal)
                findings.append(render("error", "E-PORT-OPEN", "fluid", length=e.length, x=e.centre[0], y=e.centre[1],
                                       facing=facing, where=e.centre))

    for c, name in curve_patch.items():
        wk.curves[c].patch = name
    resolved: list[dict] = []
    for name in dict.fromkeys(curve_patch.values()):
        if name == "walls" and "walls" not in kinds:
            continue
        for c, n in curve_patch.items():
            if n != name:
                continue
            mx, my = wk.curves[c].midpoint
            resolved.append({"name": name, "kind": kinds.get(name, _port_kind(name, None)), "at": f"near:{mx:.6g},{my:.6g}"})
    return curve_patch, resolved, findings


def _mid(pts) -> tuple[float, float]:
    if not pts:
        return (0.0, 0.0)
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _facing(normal) -> str:
    nx, ny = normal
    if abs(nx) >= abs(ny):
        return "+x" if nx > 0 else "-x"
    return "+y" if ny > 0 else "-y"


def _ends_text(ends: list[OpenEnd]) -> str:
    return _ends_text_points([e.centre for e in ends])


def _ends_text_points(pts) -> str:
    words = [f"({x:.4g}, {y:.4g})" for x, y in pts]
    if len(words) <= 1:
        return words[0] if words else "none"
    return ", ".join(words[:-1]) + " and " + words[-1]


def _rule_fix(ends: list[OpenEnd]) -> str:
    if not ends:
        return "name an open end of the fluid by intent (main.end) or near:x,y"
    e = ends[0]
    return f"near:{e.centre[0]:.4g},{e.centre[1]:.4g} names the one at ({e.centre[0]:.4g}, {e.centre[1]:.4g})"


def _count_fix(want: str, ends: list[OpenEnd], refused: bool) -> str:
    if refused and ends:
        where = ", ".join(f"({e.centre[0]:.4g}, {e.centre[1]:.4g}) facing {_facing(e.outward_normal)}" for e in ends)
        return (f"the open ends are {where}; they are not the two ends of one axis, so name them: "
                f"s.inlet(<feature>.start) and s.outlet(<feature>.end), or near:x,y")
    return f"s.inlet(<feature>.start) and s.outlet(<feature>.end) tell the ends apart; s.expect(...) states a count other than one {want}"


def _automatic_ends(ends: list[OpenEnd], bounds) -> tuple[OpenEnd | None, OpenEnd | None]:
    """The two candidates facing -axis and +axis along the longest axis, when exactly one each."""
    axis = 0 if bounds[2] - bounds[0] >= bounds[3] - bounds[1] else 1
    cos_tol = math.cos(math.radians(30.0))
    lo = [e for e in ends if -e.outward_normal[axis] >= cos_tol]
    hi = [e for e in ends if e.outward_normal[axis] >= cos_tol]
    if len(lo) == 1 and len(hi) == 1 and len(ends) == 2:
        return lo[0], hi[0]
    return None, None


def _resolve_target(wk: Walk, entry: dict, ends_by_curve: dict, tol: float, span: float) -> list[int]:
    kind = entry.get("kind")
    pts = [tuple(p) for p in entry.get("points", [])]
    if kind == "cap":
        centre = _mid(pts)
        width = float(entry.get("width", 0.0))
        for c, e in ends_by_curve.items():
            if _dist(e.centre, centre) <= 1e-6 * span + 1e-9 and abs(e.length - width) <= 1e-6 * span + 1e-9:
                return [c]
        for c, e in ends_by_curve.items():
            if _dist(e.centre, centre) <= 1e-3 * span:
                return [c]
        return []
    if kind == "side":
        return _curves_on_polyline(wk, pts, closed=False, tol=tol)
    if kind == "loop":
        return _curves_on_polyline(wk, pts, closed=True, tol=tol)
    return _curves_on_polyline(wk, pts, closed=len(pts) > 2, tol=tol)


# -- the reference width ------------------------------------------------------------------


def _claims_list(claims) -> list:
    if claims is None:
        return []
    if isinstance(claims, dict):
        return list(claims.get("claims", []))
    return list(getattr(claims, "claims", []) or [])


def _claim_get(claim, key, default=None):
    if isinstance(claim, dict):
        return claim.get(key, default)
    return getattr(claim, key, default)


def reference_width(plan: "Plan", claims: "ClaimSet | None", holes: list[HoleMeasure], bounds) -> tuple[float, str]:
    """The scale every relative threshold and every ray uses. In this order: min of
    plan.declared_widths ("declared by 'main'"); else the smallest `width` claim ("claim
    c1"); else the smallest hole diameter -- a `diameter` claim on a Disk, or the smallest
    HoleMeasure with a radius -- ("hole diameter (cyl)": T05 gets 10); else the fluid's
    shorter extent / 10 ("extent / 10"). Never the passage minimum (the minimum is the
    sliver we are looking for) and never undefined: a Rect/Disk fluid with no width claim
    had no branch in the first draft. `passage_widths` runs AFTER this, with its cap and
    spacing from the result; the p10 of its samples is printed beside the reference as
    I-REFERENCE, not used as the reference."""
    declared = [float(w) for w in (getattr(plan, "declared_widths", None) or []) if w and w > 0]
    if declared:
        w = min(declared)
        owner = next((op.get("name") for op in getattr(plan, "ops", []) or []
                      if op.get("op") == "channel" and abs(float(op.get("width", 0)) - w) <= 1e-9 * w), None)
        if owner is None:
            owner = next((name for name, s in (getattr(plan, "features", None) or {}).items()
                          if abs(float(s.params.get("width", 0) or 0) - w) <= 1e-9 * w), None)
        return w, f"declared by '{owner}'" if owner else "declared"
    widths = [(float(_claim_get(c, "value")), _claim_get(c, "id"))
              for c in _claims_list(claims)
              if _claim_get(c, "kind") == "measure" and _claim_get(c, "measure") == "width"
              and _claim_get(c, "value") is not None and float(_claim_get(c, "value")) > 0]
    if widths:
        w, cid = min(widths)
        return w, f"claim {cid}"
    diameters = [(float(_claim_get(c, "value")), _claim_get(c, "of"))
                 for c in _claims_list(claims)
                 if _claim_get(c, "kind") == "measure" and _claim_get(c, "measure") == "diameter"
                 and _claim_get(c, "value") is not None and float(_claim_get(c, "value")) > 0]
    if diameters:
        d, of = min(diameters)
        return d, f"hole diameter ({of})"
    round_holes = [h for h in holes if h.radius]
    if round_holes:
        h = min(round_holes, key=lambda h: h.radius)
        name = None
        for fname, s in (getattr(plan, "features", None) or {}).items():
            centre = s.params.get("centre") or s.params.get("center") or s.solved.get("centre")
            if s.kind == "Disk" and centre is not None and _dist(centre, h.centroid) <= 1e-3 * (h.radius or 1.0) + 1e-9:
                name = fname
        if name is None:
            for op in getattr(plan, "ops", []) or []:
                if op.get("op") == "disk" and _dist(tuple(op.get("center", (1e300, 1e300))), h.centroid) <= 1e-3 * h.radius + 1e-9:
                    name = op.get("name")
        return 2.0 * h.radius, f"hole diameter ({name or f'loop {h.loop}'})"
    x0, y0, x1, y1 = bounds
    return min(x1 - x0, y1 - y0) / 10.0, "extent / 10"


# -- holes, instances, junctions -----------------------------------------------------------


def holes_of(wk: Walk) -> list[HoleMeasure]:
    out: list[HoleMeasure] = []
    for k in range(1, len(wk.loops)):
        pts = wk.loop_points(k)
        curves = [wk.curves[c] for c in wk.loops[k]]
        radius = None
        if all(c.kind == "arc" and c.radius for c in curves):
            radii = [c.radius for c in curves]
            if max(radii) - min(radii) <= 1e-6 * max(radii):
                radius = sum(radii) / len(radii)
        out.append(HoleMeasure(loop=k, area=abs(_signed_area(pts)), centroid=_centroid(pts), radius=radius,
                               circumference=sum(c.length for c in curves)))
    return out


def _op_inputs(op: dict) -> list[str]:
    names: list[str] = []
    for key in ("of", "take"):
        v = op.get(key)
        if isinstance(v, list):
            names += [str(n) for n in v]
    for key in ("from", "target"):
        v = op.get(key)
        if isinstance(v, str) and op.get("op") != "channel":
            names.append(v)
    return names


def instances_of(plan: "Plan", channel: str) -> tuple[str | None, list[tuple[float, float]], float | None]:
    """(row name, offsets, pitch) for the built copies of a channel op: the repeat op that
    (transitively) takes it, read from `plan.ops`; a bare channel is one instance at (0, 0)
    with no row. A rotating repeat is reported with its translation offsets only, so its
    instances are not measured as a row."""
    ops = list(getattr(plan, "ops", []) or [])
    by_name = {op.get("name"): op for op in ops}
    reach = {channel}
    changed = True
    while changed:
        changed = False
        for op in ops:
            name = op.get("name")
            if name in reach:
                continue
            if any(n in reach for n in _op_inputs(op)):
                reach.add(name)
                changed = True
    for op in ops:
        if op.get("op") == "repeat" and op.get("target") in reach and not op.get("angle"):
            step = tuple(op.get("step", (0.0, 0.0)) or (0.0, 0.0))
            count = int(op.get("count", 1))
            offsets = [(k * float(step[0]), k * float(step[1])) for k in range(count)]
            return op.get("name"), offsets, math.hypot(float(step[0]), float(step[1]))
    return None, [(0.0, 0.0)], None


def _shift(p, off):
    return (p[0] + off[0], p[1] + off[1])


def _wall_frame(line: str, legs: dict, channel: str, flow_hint: tuple[float, float] | None):
    """(axis, value, flow unit vector, outward unit normal) of the wall line a leg starts
    on or lands on. The flow along the wall is the heading of another channel's straight
    leg whose wall lies on the line; else the fluid's flow projected on it; else +axis."""
    axis = 0 if line.startswith("x") else 1
    value = float(line.split("=", 1)[1])
    along = (0.0, 1.0) if axis == 0 else (1.0, 0.0)
    flow = None
    for name, records in legs.items():
        if name == channel:
            continue
        width = next((r["width"] for r in records if r["kind"] == "ports"), None)
        for r in records:
            if r["kind"] not in ("line", "to") or width is None:
                continue
            h = math.radians(r["heading"])
            d = (math.cos(h), math.sin(h))
            if abs(d[axis]) > 1e-9:
                continue
            centre = r["from"][axis]
            if abs(abs(centre - value) - width / 2) <= 1e-6 * max(1.0, abs(value)):
                flow = d
                break
        if flow:
            break
    if flow is None and flow_hint is not None and abs(flow_hint[0] * along[0] + flow_hint[1] * along[1]) > 0.5:
        s = flow_hint[0] * along[0] + flow_hint[1] * along[1]
        flow = along if s > 0 else (-along[0], -along[1])
    if flow is None:
        flow = along
    return axis, value, flow


def _leg_wall_lines(record: dict, width: float, off):
    """The two wall lines of a straight leg: (point, unit direction) each, in the order
    (left of the heading, right of the heading)."""
    h = math.radians(record["heading"])
    d = (math.cos(h), math.sin(h))
    left = (-d[1], d[0])
    p = _shift(record["from"], off)
    return ((p[0] + width / 2 * left[0], p[1] + width / 2 * left[1]), d), \
           ((p[0] - width / 2 * left[0], p[1] - width / 2 * left[1]), d)


def _curve_on_line(wk: Walk, point, direction, tol: float) -> CurveInfo | None:
    """The longest straight curve of the walk whose samples all lie on the line."""
    best = None
    for info in wk.curves.values():
        if info.kind != "line":
            continue
        if all(point_line_distance(p, point, direction) <= tol for p in info.samples):
            if best is None or info.length > best.length:
                best = info
    return best


def junctions_of(wk: Walk, plan: "Plan", legs: dict, flow_hint: tuple[float, float] | None) -> list[Junction]:
    """One Junction per leg that starts on a wall (`from_line`) or lands on one (a `to`
    leg), per built instance of its channel. The measured angle is read from the leg's
    upstream wall curve on the walk (the fluid flows away from the host along a leave leg
    and toward it along a return leg, which fixes the sense the wall curve is read in)."""
    out: list[Junction] = []
    span = wk.span()
    tol = 1e-6 * span + 1e-9
    for channel, records in legs.items():
        width = next((r["width"] for r in records if r["kind"] == "ports"), None)
        if width is None:
            continue
        row, offsets, _ = instances_of(plan, channel)
        to_legs = [r for r in records if r["kind"] == "to"]
        first = next((r for r in records if r["kind"] != "ports"), None)
        for k, off in enumerate(offsets):
            instance = f"{row}[{k}]" if row else channel
            for r in records:
                if r["kind"] in ("line", "to") and r.get("from_line"):
                    out.append(_junction(wk, instance, r, r["from_line"], _shift(r["from"], off), "leave",
                                         width, off, legs, channel, flow_hint, tol))
                if r["kind"] == "to":
                    out.append(_junction(wk, instance, r, r["line"], _shift(r["lands"], off), "return",
                                         width, off, legs, channel, flow_hint, tol))
            # A loop drawn without `from` (the 4025 valve) still leaves the wall its
            # return leg lands on: its leave junction is read at the first leg's start
            # against that line, so the built leave angle is measured, not assumed.
            if to_legs and first is not None and first["kind"] == "line" and not first.get("from_line"):
                line = to_legs[-1]["line"]
                axis = 0 if line.startswith("x") else 1
                value = float(line.split("=", 1)[1])
                start = _shift(first["from"], off)
                if abs(start[axis] - value) <= width:
                    where = (value, start[1]) if axis == 0 else (start[0], value)
                    out.append(_junction(wk, instance, first, line, where, "leave", width, off, legs, channel,
                                         flow_hint, tol))
    return out


def _junction(wk, instance, record, line, where, kind, width, off, legs, channel, flow_hint, tol) -> Junction:
    axis, value, flow = _wall_frame(line, legs, channel, flow_hint)
    outward = None
    h = math.radians(record["heading"])
    d = (math.cos(h), math.sin(h))
    normal = (0.0, 1.0) if axis == 1 else (1.0, 0.0)
    into_host = d[0] * normal[0] + d[1] * normal[1]
    leaves = into_host > 0 if kind == "leave" else into_host < 0
    outward = normal if leaves else (-normal[0], -normal[1])
    upstream = (-flow[0], -flow[1])
    typed = math.degrees(math.acos(max(-1.0, min(1.0, (d[0] * flow[0] + d[1] * flow[1]) if kind == "leave"
                                                  else (d[0] * upstream[0] + d[1] * upstream[1])))))
    left_line, right_line = _leg_wall_lines(record, width, off)
    walls = [left_line, right_line]
    walls.sort(key=lambda w: (w[0][0] - where[0]) * flow[0] + (w[0][1] - where[1]) * flow[1])
    upstream_wall = walls[0]
    curve = _curve_on_line(wk, upstream_wall[0], upstream_wall[1], max(tol, 1e-6 * width))
    heading = record["heading"] % 360.0
    measured = typed
    if curve is not None:
        c_dir = (math.cos(math.radians(curve.heading_in)), math.sin(math.radians(curve.heading_in)))
        s = c_dir[0] * outward[0] + c_dir[1] * outward[1]
        want_out = kind == "leave"
        if (s > 0) != want_out:
            c_dir = (-c_dir[0], -c_dir[1])
        ref = flow if kind == "leave" else upstream
        measured = math.degrees(math.acos(max(-1.0, min(1.0, c_dir[0] * ref[0] + c_dir[1] * ref[1]))))
        if kind == "return":
            heading = _heading(*c_dir)
    corner = None
    if kind == "leave":
        s = abs(d[0] * normal[0] + d[1] * normal[1])
        corner = (where[0] + flow[0] * width / 2 / max(s, 1e-9), where[1] + flow[1] * width / 2 / max(s, 1e-9))
    lip = None
    if corner is not None:
        best = None
        for v in wk.vertices:
            if abs(v.at[axis] - value) > max(tol, 1e-6 * width) or v.solid_deg >= 180.0:
                continue
            dd = _dist(v.at, corner)
            if dd <= width / 2 and (best is None or dd < best[0]):
                best = (dd, v)
        if best is not None:
            lip = (best[1].at, best[1].solid_deg)
    against = None
    if flow is not None:
        comp = math.cos(math.radians(heading)) * flow[0] + math.sin(math.radians(heading)) * flow[1]
        against = comp < 0
    return Junction(channel=instance, where=where, wall_line=line, kind=kind, typed_angle=typed,
                    measured_angle=measured, lip=lip, heading=heading, against_flow=against)


# -- measure --------------------------------------------------------------------------------


def _arc_chain(wk: Walk, centre, radius, tol) -> tuple[float | None, float]:
    """(measured radius, total |sweep|) of every arc curve of the walk about `centre` at
    `radius` (both within tol): a band's wall is split at the circle's seam, so the sweep
    is the sum over the pieces."""
    radii = []
    sweep = 0.0
    for info in wk.curves.values():
        if info.kind != "arc" or info.centre is None or info.radius is None:
            continue
        if _dist(info.centre, centre) <= tol and abs(info.radius - radius) <= tol:
            radii.append(info.radius)
            sweep += abs(info.sweep or 0.0)
    return (sum(radii) / len(radii) if radii else None), sweep


def _channel_shape(records: list[dict]) -> dict:
    """The centreline measures of a channel from its leg records."""
    legs = [r for r in records if r["kind"] != "ports"]
    ports = next((r for r in records if r["kind"] == "ports"), {})
    lines = [r for r in legs if r["kind"] in ("line", "to")]
    arcs = [r for r in legs if r["kind"] == "arc"]
    corners = [r for r in legs if r["kind"] == "corner"]
    length = sum(float(r.get("length", 0.0)) for r in lines) + sum(
        float(r["radius"]) * math.radians(abs(float(r["sweep"]))) for r in arcs)
    out = {"width": ports.get("width"), "length": length, "legs": len(legs), "legs_straight": len(lines),
           "corners": len(corners), "bends": len(arcs), "start": tuple(ports.get("start", (0, 0))),
           "end": tuple(ports.get("end", (0, 0))), "start_heading": float(ports.get("heading_in", 0.0)) % 360.0,
           "end_heading": float(ports.get("heading_out", 0.0)) % 360.0}
    if arcs:
        out["bend_radius"] = float(arcs[0]["radius"])
        out["bend_sweep"] = abs(float(arcs[0]["sweep"]))
        out["bend_sweeps"] = [abs(float(r["sweep"])) for r in arcs]
    straight = sorted(lines, key=lambda r: -float(r.get("length", 0.0)))
    if len(straight) >= 2:
        a, b = straight[0], straight[1]
        ha, hb = float(a["heading"]) % 360.0, float(b["heading"]) % 360.0
        if abs(_norm180(ha - hb)) < 1e-6 or abs(abs(_norm180(ha - hb)) - 180.0) < 1e-6:
            d = (math.cos(math.radians(ha)), math.sin(math.radians(ha)))
            out["spacing"] = point_line_distance(b["from"], a["from"], d)
    sx, sy = out["start"]
    ex, ey = out["end"]
    sh = math.radians(out["start_heading"])
    along = (ex - sx) * math.cos(sh) + (ey - sy) * math.sin(sh)
    out["end_side"] = "same" if abs(along) < float(ports.get("width", 1.0)) else "opposite"
    if arcs and lines and all(abs(abs(float(r["sweep"])) - 180.0) < 1e-6 for r in arcs) and len(lines) == len(arcs) + 1:
        r0 = float(arcs[0]["radius"])
        w = float(ports.get("width", 0.0))
        out["passes"] = len(lines)
        out["pass_length"] = float(lines[0].get("length", 0.0))
        out["pass_pitch"] = 2 * r0
        out["wall_between"] = 2 * r0 - w
        stack = (-math.sin(sh), math.cos(sh)) if float(arcs[0]["sweep"]) > 0 else (math.sin(sh), -math.cos(sh))
        out["pass_centrelines"] = [round((r["from"][0] - sx) * stack[0] + (r["from"][1] - sy) * stack[1], 9) for r in lines]
    return out


def measure(gmsh, face: int, plan: "Plan", wk: Walk, legs: dict, curve_patch: dict[int, str],
            claims: "ClaimSet | None") -> Measurements:
    """Everything the print-back, the claims and the preview read. Per-feature keys: the
    table of DESIGN.md 3.5 (Rect: width/length/size_x/size_y/height/centre/origin; Disk:
    centre/radius/diameter/circumference from the hole's curvature; Passage; Bypass -- every
    one READ FROM THE WALK, the typed value printed beside a measured one when they differ;
    Row; Serpentine; BodyInBox; `fluid`; a patch name). A channel op that both starts on a
    wall and lands on one is measured as a Bypass whether or not the plan names it so, and
    its built copies are `<row>[k]` (D29) with a `<row>[*]` aggregate of lists."""
    units = getattr(plan, "units", None) or UNITS_BY_SCALE.get(float(getattr(plan, "scale", 1.0)), "units")
    scale = float(getattr(plan, "scale", 1.0))
    bounds = wk.bounds()
    extent = (bounds[2] - bounds[0], bounds[3] - bounds[1])
    span = max(extent)
    area = float(gmsh.model.occ.getMass(2, face))
    for c, name in curve_patch.items():
        if c in wk.curves:
            wk.curves[c].patch = name
    shortest = min(wk.curves.values(), key=lambda c: c.length)
    patches: dict[str, dict] = {}
    for c, info in wk.curves.items():
        p = patches.setdefault(info.patch, {"curves": [], "n": 0, "length": 0.0, "midpoints": []})
        p["curves"].append(c)
        p["n"] += 1
        p["length"] += info.length
        p["midpoints"].append(list(info.midpoint))
    flow = None
    inlet_curves = [c for c, n in curve_patch.items() if n == "inlet" and c in wk.curves]
    if inlet_curves:
        fx = sum(wk.curves[c].inward_normal[0] for c in inlet_curves)
        fy = sum(wk.curves[c].inward_normal[1] for c in inlet_curves)
        if math.hypot(fx, fy) > 0:
            flow = _unit((fx, fy))
    holes = holes_of(wk)
    ref, ref_from = reference_width(plan, claims, holes, bounds)
    passage = passage_widths(gmsh, face, wk, ref)
    ends = open_ends(gmsh, wk, ref, _flow_box(plan))
    for e in ends:
        e.name = curve_patch.get(e.curve) if curve_patch.get(e.curve) != "walls" else None
    junctions = junctions_of(wk, plan, legs, flow)
    features: dict[str, dict] = {}
    rows: dict[str, RowMeasure] = {}
    solved = getattr(plan, "features", None) or {}
    ops = list(getattr(plan, "ops", []) or [])
    tol = 1e-6 * span + 1e-9

    for op in ops:
        name = op.get("name")
        if op.get("op") == "rect" and "origin" in op:
            features[name] = _rect_measures(op["origin"], op["size"])
        elif op.get("op") == "rect" and "center" in op:
            sx, sy = op["size"]
            cx, cy = op["center"]
            features[name] = _rect_measures((cx - sx / 2, cy - sy / 2), (sx, sy))
        elif op.get("op") == "disk":
            features[name] = _disk_measures(tuple(op["center"]), float(op["radius"]), holes, wk)
    for name, s in solved.items():
        params, sol = dict(s.params), dict(s.solved)
        if s.kind == "Rect":
            origin = sol.get("origin") or params.get("origin")
            size = sol.get("size") or params.get("size")
            if origin is None and size is not None:
                c = sol.get("centre") or params.get("centre") or params.get("center")
                origin = (c[0] - size[0] / 2, c[1] - size[1] / 2)
            if origin is not None and size is not None:
                features[name] = _rect_measures(tuple(origin), tuple(size))
        elif s.kind == "Disk":
            centre = sol.get("centre") or params.get("centre") or params.get("center")
            radius = sol.get("radius") or params.get("radius") or (params.get("diameter", 0) / 2)
            if centre is not None:
                features[name] = _disk_measures(tuple(centre), float(radius or 0.0), holes, wk)
        elif s.kind == "BodyInBox":
            features[name] = {k: params.get(k, sol.get(k)) for k in ("box", "ahead", "behind", "above", "below")}

    for channel, records in legs.items():
        ports = next((r for r in records if r["kind"] == "ports"), None)
        if ports is None:
            continue
        shape = _channel_shape(records)
        feature_name, kind = _feature_of(solved, channel)
        row, offsets, pitch = instances_of(plan, channel)
        first = records[0] if records and records[0]["kind"] != "ports" else None
        is_bypass = kind == "Bypass" or (first is not None and first["kind"] == "line"
                                         and any(r["kind"] == "arc" for r in records)
                                         and any(r["kind"] == "to" for r in records))
        base = feature_name or channel
        if not is_bypass:
            entry = dict(shape)
            if kind == "Serpentine" or "passes" in shape:
                entry["extent"] = extent if len(legs) == 1 else _legs_extent(records, ports["width"])
            else:
                entry["extent"] = extent if len(legs) == 1 else _legs_extent(records, ports["width"])
            features[base] = entry
            if row:
                for k in range(len(offsets)):
                    features[f"{row}[{k}]"] = dict(entry)
            continue
        per: list[dict] = []
        for k, off in enumerate(offsets):
            instance = f"{row}[{k}]" if row else base
            per.append(_bypass_measures(wk, instance, records, ports, off, junctions, holes, tol, solved.get(base)))
            features[instance] = per[-1]
        if row:
            features[f"{row}[*]"] = {key: [d.get(key) for d in per] for key in per[0]}
            if base != row and base not in features:
                features[base] = dict(per[0])
        elif base != channel:
            features[channel] = features[base]

    for name, inst in (getattr(plan, "instances", None) or {}).items():
        count = int(inst.get("count", 0))
        step = inst.get("step") or (0.0, 0.0)
        pitch = float(inst.get("pitch") or math.hypot(float(step[0]), float(step[1])))
        gap = inst.get("gap")
        footprint = tuple(inst.get("footprint") or (0.0, 0.0))
        anchors = list(inst.get("anchors") or [])
        rspan = tuple(inst.get("span") or (0.0, 0.0))
        gap_measured = inst.get("gap_measured")
        rows[name] = RowMeasure(name=name, count=count, pitch=pitch, footprint=footprint, gap=gap,
                                gap_measured=gap_measured, anchors=anchors, span=rspan)
        features[name] = {"count": count, "pitch": pitch, "gap": gap, "gap_measured": gap_measured, "span": rspan,
                          "anchors": anchors}

    features["fluid"] = {"extent_x": extent[0], "extent_y": extent[1], "area": area, "islands": len(wk.loops) - 1,
                         "edges": len(wk.curves), "narrowest_passage": passage.min}
    for name, p in patches.items():
        entry = {"edges": p["n"], "length": p["length"],
                 "midpoint": tuple(p["midpoints"][0]) if p["n"] == 1 else _mid(p["midpoints"])}
        entry["width"] = p["length"] if p["n"] == 1 else p["length"]
        features.setdefault(name, entry)

    return Measurements(units=units, scale=scale, bounds=bounds, extent=extent, area=area,
                        islands=len(wk.loops) - 1, n_curves=len(wk.curves),
                        shortest_edge=(shortest.length, shortest.midpoint), patches=patches, legs=legs,
                        features=features, rows=rows, open_ends=ends, junctions=junctions,
                        vertices=list(wk.vertices), passage=passage, reference_width=ref,
                        reference_width_from=ref_from, holes=holes, flow=flow)


def _flow_box(plan) -> tuple | None:
    for s in (getattr(plan, "features", None) or {}).values():
        if s.kind == "BodyInBox":
            box = s.solved.get("box") or s.params.get("box")
            if box is not None:
                return tuple(box)
    return None


def _feature_of(solved: dict, channel: str) -> tuple[str | None, str | None]:
    for name, s in solved.items():
        if channel in (s.ops or []) or name == channel:
            return name, s.kind
    stem = channel.split(".")[0]
    if stem in solved:
        return stem, solved[stem].kind
    return None, None


def _rect_measures(origin, size) -> dict:
    sx, sy = float(size[0]), float(size[1])
    return {"width": min(sx, sy), "length": max(sx, sy), "size_x": sx, "size_y": sy, "height": sy,
            "centre": (origin[0] + sx / 2, origin[1] + sy / 2), "origin": (float(origin[0]), float(origin[1]))}


def _disk_measures(centre, typed_radius: float, holes: list[HoleMeasure], wk: Walk) -> dict:
    hole = None
    for h in holes:
        if h.radius and _dist(h.centroid, centre) <= 1e-3 * max(typed_radius, h.radius):
            hole = h
    if hole is not None:
        r = hole.radius
        return {"centre": hole.centroid, "radius": r, "diameter": 2 * r, "circumference": hole.circumference,
                "typed_radius": typed_radius}
    arcs = [c for c in wk.curves.values() if c.kind == "arc" and c.centre is not None
            and _dist(c.centre, centre) <= 1e-3 * max(typed_radius, 1e-9)]
    if arcs:
        r = sum(c.radius for c in arcs) / len(arcs)
        return {"centre": centre, "radius": r, "diameter": 2 * r, "circumference": sum(c.length for c in arcs),
                "typed_radius": typed_radius}
    return {"centre": centre, "radius": typed_radius, "diameter": 2 * typed_radius,
            "circumference": 2 * math.pi * typed_radius, "typed_radius": typed_radius}


def _legs_extent(records, width) -> tuple[float, float]:
    xs, ys = [], []
    for r in records:
        if r["kind"] == "ports":
            continue
        for key in ("from", "to", "lands"):
            if key in r:
                xs.append(r[key][0])
                ys.append(r[key][1])
        if r["kind"] == "arc":
            xs += [r["centre"][0] - r["outer_radius"], r["centre"][0] + r["outer_radius"]]
            ys += [r["centre"][1] - r["outer_radius"], r["centre"][1] + r["outer_radius"]]
    return (max(xs) - min(xs) + width, max(ys) - min(ys) + width)


def _bypass_measures(wk: Walk, instance: str, records, ports, off, junctions, holes, tol, s) -> dict:
    width = float(ports["width"])
    arcs = [r for r in records if r["kind"] == "arc"]
    to_leg = next((r for r in records if r["kind"] == "to"), None)
    leave = next((j for j in junctions if j.channel == instance and j.kind == "leave"), None)
    ret = next((j for j in junctions if j.channel == instance and j.kind == "return"), None)
    out: dict = {"width": width}
    typed: dict = {}
    if arcs:
        a = arcs[0]
        centre = _shift(a["centre"], off)
        outer_r, outer_sweep = _arc_chain(wk, centre, a["outer_radius"], max(tol, 1e-6 * width))
        inner_r, inner_sweep = _arc_chain(wk, centre, a["inner_radius"], max(tol, 1e-6 * width))
        out["outer_radius"] = outer_r
        out["inner_radius"] = inner_r
        out["radius"] = (outer_r + inner_r) / 2 if outer_r and inner_r else (outer_r or inner_r)
        out["sweep"] = inner_sweep if inner_sweep else outer_sweep
        out["centre"] = centre
        typed.update(outer_radius=a["outer_radius"], inner_radius=a["inner_radius"], radius=a["radius"],
                     sweep=abs(a["sweep"]))
        island = min(holes, key=lambda h: _dist(h.centroid, centre)) if holes else None
        if island is not None and _dist(island.centroid, centre) <= a["outer_radius"]:
            out["island_area"] = island.area
    if leave is not None:
        out["leave_angle"] = leave.measured_angle
        typed["leave_angle"] = leave.typed_angle
        if leave.lip is not None:
            out["lip_at"], out["lip_angle"] = leave.lip
    if ret is not None:
        out["return_angle"] = ret.measured_angle
        out["return_heading"] = ret.heading
        out["against_flow"] = ret.against_flow
        typed["return_angle"] = ret.typed_angle
        typed["return_heading"] = float(to_leg["heading"]) % 360.0 if to_leg else ret.heading
        out["lands_at"] = ret.where
        if leave is not None:
            flow = _flow_of(leave, ret)
            out["lands_upstream_by"] = -((ret.where[0] - leave.where[0]) * flow[0] + (ret.where[1] - leave.where[1]) * flow[1])
    inst_curves = _instance_curves(wk, records, off, width, tol)
    if inst_curves and leave is not None:
        flow = _flow_of(leave, ret) if ret is not None else (1.0, 0.0)
        us = [(p[0] - leave.where[0]) * flow[0] + (p[1] - leave.where[1]) * flow[1] for c in inst_curves for p in c.samples]
        axis = 1 if abs(flow[0]) >= abs(flow[1]) else 0
        vs = [abs(p[axis] - leave.where[axis]) for c in inst_curves for p in c.samples]
        out["footprint"] = (min(us), max(us))
        out["height"] = max(vs)
    if s is not None:
        for key in ("outer_radius", "leave_angle", "return_angle", "sweep", "lip_deg"):
            if key in s.params:
                typed[key] = s.params[key]
    out["typed"] = typed
    return out


def _flow_of(leave: Junction, ret: Junction | None) -> tuple[float, float]:
    axis = 0 if leave.wall_line.startswith("x") else 1
    along = (0.0, 1.0) if axis == 0 else (1.0, 0.0)
    h = math.radians(leave.heading)
    d = (math.cos(h), math.sin(h))
    s = d[0] * along[0] + d[1] * along[1]
    if leave.against_flow is not None:
        return along if (s > 0) != leave.against_flow else (-along[0], -along[1])
    return along if s >= 0 else (-along[0], -along[1])


def _instance_curves(wk: Walk, records, off, width, tol) -> list[CurveInfo]:
    """The walk curves that belong to one built copy of a channel: the arcs about its
    arc centres and the straight curves on its legs' wall lines."""
    out: list[CurveInfo] = []
    seen: set[int] = set()
    for r in records:
        if r["kind"] == "arc":
            centre = _shift(r["centre"], off)
            for info in wk.curves.values():
                if info.tag in seen or info.kind != "arc" or info.centre is None:
                    continue
                if _dist(info.centre, centre) <= max(tol, 1e-6 * width) and abs(info.radius - r["outer_radius"]) <= max(tol, 1e-6 * width) \
                        or _dist(info.centre, centre) <= max(tol, 1e-6 * width) and abs(info.radius - r["inner_radius"]) <= max(tol, 1e-6 * width):
                    out.append(info)
                    seen.add(info.tag)
        elif r["kind"] in ("line", "to"):
            for point, direction in _leg_wall_lines(r, width, off):
                for info in wk.curves.values():
                    if info.tag in seen or info.kind != "line":
                        continue
                    if all(point_line_distance(p, point, direction) <= max(tol, 1e-6 * width) for p in info.samples):
                        start = _shift(r["from"], off)
                        end = _shift(r.get("lands", r.get("to", r["from"])), off)
                        lo = min((start[0] - point[0]) * direction[0] + (start[1] - point[1]) * direction[1],
                                 (end[0] - point[0]) * direction[0] + (end[1] - point[1]) * direction[1]) - width
                        hi = max((start[0] - point[0]) * direction[0] + (start[1] - point[1]) * direction[1],
                                 (end[0] - point[0]) * direction[0] + (end[1] - point[1]) * direction[1]) + width
                        u = (info.midpoint[0] - point[0]) * direction[0] + (info.midpoint[1] - point[1]) * direction[1]
                        if lo <= u <= hi:
                            out.append(info)
                            seen.add(info.tag)
    return out
