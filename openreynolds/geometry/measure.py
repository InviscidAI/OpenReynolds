"""The instruments: the boundary walk and everything read from the built face.

Every fact about gmsh that shaped these signatures was run on this box (DESIGN.md 3.5):
`getBoundary(oriented=True)` is not in walk order and `getCurveLoops` gives the loops
unsigned, so `walk()` chains each loop by end points and fixes the direction by signed
area; `isInside` is wrong on any face that carries a dilate, mirror or affine, so the
kernel runs at scale 1 on untransformed faces; a curve's kind is by curvature, never by
the type string; bounds come from sampled curves, never `getBoundingBox`. gmsh is passed
in, never imported here. Skeleton (U0): dataclasses with their fields and round trips;
the instruments raise NotImplementedError until U2 lands.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

from . import _toolbox  # noqa: F401  (mesh2d.leg_lines by path)

if TYPE_CHECKING:
    from .claims import ClaimSet
    from .compile import Plan
    from .lint import Finding

_U2 = ("not built in the U0 skeleton: U2 (measure + lint) implements it against "
       "DESIGN.md section 3.5")


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


def walk(gmsh, face: int, w_min: float) -> Walk:
    """The boundary walk, computed once and shared by lint and measure: loops from
    `occ.getCurveLoops`, each chained by end points (the vertex dimTags of
    `getBoundary([(1, c)])`, which come back unsigned), each curve oriented so the chain
    closes, then the direction fixed by signed area (outer positive, holes negative) and
    `repaired` set when a reversal was needed (preamble)."""
    raise NotImplementedError(_U2)


def sampled_bounds(gmsh, dim: int, tag: int, n: int = 64) -> tuple[float, float, float, float]:
    """Tight bounds from `getValue` at n parameters per boundary curve plus the vertices.
    The same function is added to mesh2d.py (section 3.17); the two are pinned equal by a test."""
    raise NotImplementedError(_U2)


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
    raise NotImplementedError(_U2)


def resolve_ports(gmsh, wk: Walk, plan: "Plan", open_ends: list[OpenEnd]) -> tuple[dict[int, str], list[dict], list["Finding"]]:
    """EdgeRefs and rule strings -> (curve -> patch name, the resolved rules in today's
    grammar for the record, the findings E-PORT-MISSING / E-PORT-RULE / E-PORT-COUNT /
    E-PORT-OPEN). Reads `plan.edge_targets` for every EdgeRef (section 3.4's table) and
    never `plan.features`. The automatic reading (flat at the two ends of the longest
    axis) is used only when no rule names inlet or outlet AND the topology finds exactly
    two candidates with opposite normals along the longest axis (an elbow's -x / +y ends
    refuse it: E-PORT-COUNT with the two ends named)."""
    raise NotImplementedError(_U2)


def open_ends(gmsh, wk: Walk, w_min: float, flow_box: tuple | None) -> list[OpenEnd]:
    """A candidate is a curve that is (1) a line, (2) whose two end vertices are convex with
    interior angle within 90 +/- 30 deg, (3) with depth >= its own length along the inward
    normal (ray_depth until="outside"), (4) not on the flow box. A dead-end cap is a
    candidate; the rules decide."""
    raise NotImplementedError(_U2)


def ray_depth(gmsh, face: int, p: tuple[float, float], n: tuple[float, float], max_len: float,
              steps: int, until: Literal["outside", "inside"] = "outside", resolution: float | None = None) -> float:
    """March p + (i/steps) * max_len * n, i = 1..steps, isInside per point, until the
    first point that is outside (until="outside": how much fluid lies ahead) or inside
    (until="inside": how much solid lies ahead, the notch predicate marching OUTWARD);
    then bisect between the last point before the change and the first after it down to
    `resolution` (default 1e-3 * max_len / 3; ten extra calls), and return that distance,
    or max_len when nothing changes. A point within 1e-7 of the boundary counts as on it.
    Without the bisection the coarse march is quantised to a step and biased one step
    high (T02's 2 mm channel read 2.33 at 18 steps over 3 w; T01's 3 mm read 3.5)."""
    raise NotImplementedError(_U2)


def passage_widths(gmsh, face: int, wk: Walk, w_min: float) -> PassageWidths:
    """Samples along every wall curve of length >= w_min at spacing w_min/4, skipping samples
    within w_min/2 of a vertex with interior < 90 deg; ray_depth inward, max 3 w_min, 18
    steps plus the bisection, resolution 1e-3 w_min. About 7,000 rays on a valve at ~0.35
    ms per isInside call: ~2.5 s coarse plus ~2.5 s bisection on this box; the sample
    spacing is halved (w_min/2) when the wall length sum exceeds 200 w_min so the budget
    stays under 5 s. The resolution is printed in the docstring and in MEASURED
    ("narrowest passage 3.000 (+/- 0.003)"). Numbers for T01/T02 are pinned only after this
    implementation runs (U2 sets them; none is asserted in this document)."""
    raise NotImplementedError(_U2)


def measure(gmsh, face: int, plan: "Plan", wk: Walk, legs: dict, curve_patch: dict[int, str],
            claims: "ClaimSet | None") -> Measurements:
    """Everything the print-back, the claims and the preview read. Per-feature keys: the
    table of DESIGN.md 3.5 (Rect: width/length/size_x/size_y/height/centre/origin; Disk:
    centre/radius/diameter/circumference from the hole's curvature; Passage; Bypass -- every
    one READ FROM THE WALK, the typed value printed beside a measured one when they differ;
    Row; Serpentine; BodyInBox; `fluid`; a patch name)."""
    raise NotImplementedError(_U2)
