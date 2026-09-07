"""Claims: what the request states, written before any geometry, and the compliance
table the built shape is judged against.

The schema is DESIGN.md 4.1 (`id`, `says`, `of`, top-level `unit`/`kind`/`flow`, an
explicit `kind` per claim, D12); an unknown predicate or measure is `not_measurable`,
never an error; a claim naming an absent feature is a `fail` row listing the script's
features. `can_commit` is pure and is where D10 (warnings do not block in Phase 1), D11
(disagrees must equal the failing set) and D33 (a table needs a checkable row) live.

Everything here reads `Measurements` and `Finding` by their fields and `Plan` by
`features`/`instances`/`ops`; nothing here touches gmsh, so the desk can rerun a
compliance table over a recorded result without a kernel.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

from .lint import Finding, errors as _errors, warnings as _warnings
from .measure import Measurements

if TYPE_CHECKING:
    from .compile import Plan
    from .desk import Reply

KINDS = ("measure", "range", "count", "predicate", "patch", "report", "not_measurable")

SCHEMA = "openreynolds.geometry/claims-1"
UNITS = ("mm", "cm", "m", "in")
SET_KINDS = ("passage", "body_in_flow")
FLOWS = ("+x", "-x", "+y", "-y")
FLOW_VECTORS = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}

MEASURES: dict[str, str] = {
    # Rect (by role, D36)
    "width": "the channel width; for a Rect the shorter side, for a Passage/Bypass/Serpentine the declared width read back from the walls",
    "length": "the centreline length of a Passage; for a Rect the longer side",
    "size_x": "a Rect's size along x on the page",
    "size_y": "a Rect's size along y on the page",
    "height": "a Rect's size along y (a channel '60 mm high'); for a Bypass the loop's height above the wall",
    "centre": "the centre point (x, y) of a Rect or a Disk",
    "centre_x": "the x of the centre",
    "centre_y": "the y of the centre",
    "origin": "a Rect's lower-left corner",
    # Disk
    "radius": "a Disk's radius, from the curvature of the hole's curve, never the typed value",
    "diameter": "a Disk's diameter, twice the measured radius",
    "circumference": "a Disk's boundary length",
    # Passage
    "legs": "the number of legs of a Passage",
    "legs_straight": "the number of straight legs of a Passage",
    "corners": "the number of mitred corners of a Passage",
    "bends": "the number of arc legs (a Passage) or U-bends (a Serpentine)",
    "start": "the start point (x, y)",
    "end": "the end point (x, y)",
    "start_heading": "the heading at the start, degrees counter-clockwise from +x",
    "end_heading": "the heading at the end, degrees counter-clockwise from +x",
    "heading": "a leg's heading, degrees counter-clockwise from +x (name.legs[k])",
    "extent": "the (x, y) extent of the feature",
    "spacing": "the centreline distance between the two longest parallel straight legs",
    "end_side": "whether the end is on the same side as the start along the start's axis ('same' / 'opposite')",
    "bend_radius": "the centreline radius of the first arc (a Passage) or of the bends (a Serpentine)",
    "bend_sweep": "the sweep of the first arc, degrees",
    # Bypass
    "leave_angle": "the angle between the wall's flow and the leave leg's wall, measured on the built face",
    "return_angle": "the angle between the wall's flow and the return leg's wall, measured on the built face",
    "return_heading": "the heading of the return leg, degrees counter-clockwise from +x",
    "outer_radius": "the outer arc's radius, from its curvature",
    "inner_radius": "the inner arc's radius, from its curvature",
    "sweep": "the arc's sweep, degrees, from the walk",
    "lands_at": "where the return leg meets the wall, u along the wall",
    "lands_upstream_by": "how far upstream of its anchor the loop lands, along the wall",
    "footprint": "the loop's extent along the wall (lo, hi) about its anchor",
    "island_area": "the area of the island the loop encloses",
    "lip_angle": "the solid angle of the lip where the outer arc meets the wall, degrees",
    "lip_at": "the lip vertex (x, y)",
    # Row
    "count": "the number of instances of a Row",
    "pitch": "the distance between consecutive anchors of a Row",
    "gap": "the declared or solved bounding-box gap between consecutive instances of a Row",
    "gap_measured": "the least wall distance between consecutive instances of a Row",
    "span": "the (lo, hi) of a Row along its wall",
    "anchors": "the anchors of a Row, u along its wall",
    # Serpentine
    "passes": "the number of straight passes of a Serpentine",
    "pass_length": "the length of a Serpentine's passes",
    "pass_pitch": "the centreline distance between consecutive passes",
    "wall_between": "the solid wall thickness between consecutive passes",
    "pass_centrelines": "the across-stack coordinate of each pass",
    # BodyInBox
    "box": "the flow box (x0, y0, x1, y1)",
    "ahead": "the box length ahead of the body, in body lengths",
    "behind": "the box length behind the body, in body lengths",
    "above": "the box height above the body, in body lengths",
    "below": "the box height below the body, in body lengths",
    # fluid
    "extent_x": "the fluid's extent along x",
    "extent_y": "the fluid's extent along y",
    "area": "the fluid's area",
    "islands": "the number of islands (holes) in the fluid",
    "edges": "the number of boundary curves",
    "narrowest_passage": "the narrowest passage found by rays from the walls",
    # patches
    "midpoint": "a patch's midpoint (x, y)",
}
"""Measure key -> one sentence (the keys of DESIGN.md 3.5's per-feature table)."""

LENGTH_MEASURES: frozenset[str] = frozenset({
    "width", "height", "length", "extent_x", "extent_y", "outer_radius", "inner_radius", "radius",
    "diameter", "pitch", "gap", "gap_measured", "spacing", "pass_length", "bend_radius", "pass_pitch",
    "wall_between", "leave_length", "lands_upstream_by", "centre_x", "centre_y", "footprint",
    "circumference",
})

ANGLE_MEASURES: frozenset[str] = frozenset({
    "leave_angle", "return_angle", "return_heading", "start_heading", "end_heading", "heading",
    "lip_angle", "sweep", "bend_sweep",
})
COUNT_MEASURES: frozenset[str] = frozenset({
    "legs", "legs_straight", "corners", "bends", "count", "passes", "islands", "edges",
})
POINT_MEASURES: frozenset[str] = frozenset({"centre", "origin", "start", "end", "lip_at", "midpoint"})
SWEEP_MEASURES: frozenset[str] = frozenset({"sweep", "bend_sweep"})
PAIR_MEASURES: frozenset[str] = POINT_MEASURES | frozenset({"footprint", "span", "extent", "box"})
"""Measures whose value is one pair; any other list value is one number per pass / bend /
instance (a Serpentine's `pass_length` is four numbers), judged all together and printed
`30.00 (x4)`."""
LIST_MEASURES: frozenset[str] = frozenset({"anchors", "pass_centrelines"})
"""Measures whose value is a list that is reported whole, never judged per element."""

COUNT_WORDS = ("islands", "open_ends", "inlets", "outlets", "bends", "passes", "holes", "edges", "corners")
"""`of` words a count claim may name without a feature (DESIGN.md 4.1)."""

ANGLE_TOL_DEG = 2.0
COUNT_EXACT = 0.0

WARNINGS_BLOCK_COMMIT: bool = False
"""D10: revisit after the benchmark measures how often a warning fires on a correct shape.
Plan R2 says warnings print and are drawn, not that they block; the benchmark records how
often a warning fires on a correct shape and the flip is decided from that number."""


class ClaimsError(ValueError):
    """A malformed claims file: the message names the claim id and the field."""


@dataclass
class Verdict:
    ok: bool | None
    """None = not measurable."""
    measured: str
    """"heading 260 deg, 100 deg from the flow (+x): component -0.17"."""
    where: tuple[float, float] | None = None
    numbers: dict = field(default_factory=dict)
    expected: str = ""
    """The predicate's own margin as the table prints it ("<= 30 deg"); the claim cannot narrow it."""
    detail: str = ""
    """The explanation under a FAIL, when the predicate has one."""

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Verdict":
        where = d.get("where")
        return Verdict(ok=d.get("ok"), measured=d.get("measured", ""),
                       where=None if where is None else (float(where[0]), float(where[1])),
                       numbers=dict(d.get("numbers", {})), expected=d.get("expected", ""),
                       detail=d.get("detail", ""))


Predicate = Callable[[Measurements, str, dict], Verdict]
"""(m: Measurements, of: str, args: dict) -> Verdict. `args` carries the claim's own
arguments plus `_flow` (the claims file's flow word, or None) and `_plan_features` (the
plan's feature kinds by name) that `comply` adds; a predicate reads the rest by name."""


@dataclass
class Claim:
    id: str
    says: str
    kind: str
    measure: str = ""
    of: str = ""
    value: float | None = None
    tol: float | None = None
    tol_rel: float | None = None
    min: float | None = None
    max: float | None = None
    predicate: str = ""
    args: dict = field(default_factory=dict)
    patch: str = ""
    at: str | list | None = None
    side: str | None = None
    patch_kind: str = ""
    not_measurable: str = ""
    contradictory: bool = False
    """The benchmark's own claims files use it; the desk's never set it."""

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Claim":
        return Claim(**{k: v for k, v in d.items() if k in Claim.__dataclass_fields__})


@dataclass
class ClaimSet:
    unit: str
    kind: str
    flow: str | None
    claims: list[Claim]
    largest_length: float | None = None
    """Derived: the largest value/max among LENGTH_MEASURES claims."""

    def as_dict(self) -> dict:
        return {"unit": self.unit, "kind": self.kind, "flow": self.flow,
                "claims": [c.as_dict() for c in self.claims], "largest_length": self.largest_length}

    @staticmethod
    def from_dict(d: dict) -> "ClaimSet":
        return ClaimSet(unit=d["unit"], kind=d["kind"], flow=d.get("flow"),
                        claims=[Claim.from_dict(c) for c in d.get("claims", [])],
                        largest_length=d.get("largest_length"))

    def by_id(self, claim_id: str) -> Claim | None:
        return next((c for c in self.claims if c.id == claim_id), None)


# ----------------------------------------------------------------------------- parsing

_REQUIRED: dict[str, tuple[str, ...]] = {
    "measure": ("measure", "of", "value"),
    "range": ("measure", "of", "min", "max"),
    "count": ("of", "value"),
    "predicate": ("predicate", "of"),
    "patch": ("patch",),
    "report": ("measure", "of"),
    "not_measurable": ("not_measurable",),
}
_NUMERIC = ("value", "tol", "tol_rel", "min", "max")


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not (isinstance(v, float) and math.isnan(v))


def largest_length(claims: list[Claim]) -> float | None:
    """The largest value/max among measure and range claims on a length-typed measure;
    the lint's units test compares it with the fluid's span (D13)."""
    best = None
    for c in claims:
        if c.kind not in ("measure", "range") or c.measure not in LENGTH_MEASURES:
            continue
        for v in (c.value, c.max):
            if _is_number(v) and (best is None or abs(v) > best):
                best = abs(float(v))
    return best


def parse(payload: dict) -> tuple[ClaimSet, list[str]]:
    """Validate the schema of section 4.1. Returns the set and one line per claim for the
    claims-lap answer: 'c4 outer_radius of loops[*]: measurable' / 'c10: not measurable
    here (the finish reports it)' / 'c6: no predicate named sweeps_round; known: ...'. An
    unknown predicate or measure is NOT an error: the claim becomes not_measurable with the
    reason. Malformed JSON, a duplicate id, a missing `says`, a `kind` outside KINDS, or a
    `kind` whose required fields are absent raise ClaimsError naming the claim id and field.
    `kind: body_in_flow` with no BodyInBox in the script (known at the first script lap) is
    a warning line in the next print-back, not a failure: 'c0: kind body_in_flow but the
    script builds no BodyInBox; judged as a passage' (`kind_note`, printed by `comply`)."""
    if not isinstance(payload, dict):
        raise ClaimsError("the claims reply is not a JSON object; the schema is an object with "
                          "schema, unit, kind, flow and a claims list")
    schema = payload.get("schema")
    if schema != SCHEMA:
        raise ClaimsError(f"schema: {schema!r} is not {SCHEMA!r}; the claims object starts with "
                          f'"schema": "{SCHEMA}"')
    unit = payload.get("unit")
    if unit not in UNITS:
        raise ClaimsError(f"unit: {unit!r} is not one of {', '.join(UNITS)}; the unit the request uses")
    kind = payload.get("kind")
    if kind not in SET_KINDS:
        raise ClaimsError(f"kind: {kind!r} is not one of {', '.join(SET_KINDS)}; body_in_flow only when "
                          "the script builds a BodyInBox, else passage")
    flow = payload.get("flow")
    if flow is not None and flow not in FLOWS:
        raise ClaimsError(f"flow: {flow!r} is not one of {', '.join(FLOWS)} or null (null: the inlet's "
                          "inward normal is the flow)")
    raw = payload.get("claims")
    if not isinstance(raw, list) or not raw:
        raise ClaimsError("claims: a non-empty list of claims is required; every clause of the request "
                          "becomes one (a not_measurable claim when no predicate fits)")

    claims: list[Claim] = []
    lines: list[str] = []
    seen: set[str] = set()
    for i, d in enumerate(raw):
        if not isinstance(d, dict):
            raise ClaimsError(f"claims[{i}]: not an object; each claim is an object with id, kind, says")
        cid = d.get("id")
        if not isinstance(cid, str) or not cid.strip():
            raise ClaimsError(f"claims[{i}]: id is required (c1, c2, ... in the request's order)")
        cid = cid.strip()
        if cid in seen:
            raise ClaimsError(f"{cid}: duplicate id; every claim has its own")
        seen.add(cid)
        ckind = d.get("kind")
        if ckind not in KINDS:
            raise ClaimsError(f"{cid}: kind {ckind!r} is not one of {', '.join(KINDS)}")
        says = d.get("says")
        if not isinstance(says, str) or not says.strip():
            raise ClaimsError(f"{cid}: says is required (the request's own words for this claim)")
        for name in _REQUIRED[ckind]:
            if d.get(name) is None or d.get(name) == "":
                raise ClaimsError(f"{cid}: {ckind} claims need {name}")
        if ckind == "patch" and d.get("at") is None and d.get("side") is None:
            raise ClaimsError(f"{cid}: patch claims need at ([x, y] or a rule string) or side (left right top bottom)")
        if ckind == "patch" and d.get("side") is not None and d["side"] not in ("left", "right", "top", "bottom"):
            raise ClaimsError(f"{cid}: side {d['side']!r} is not one of left, right, top, bottom")
        for name in _NUMERIC:
            v = d.get(name)
            if v is not None and not _is_number(v):
                raise ClaimsError(f"{cid}: {name} must be a number, not {v!r}")
        if ckind == "range" and d["min"] > d["max"]:
            raise ClaimsError(f"{cid}: min {d['min']} is above max {d['max']}")
        if d.get("args") is not None and not isinstance(d["args"], dict):
            raise ClaimsError(f"{cid}: args must be an object of named arguments")
        claim = Claim.from_dict({**d, "id": cid, "says": says.strip()})
        if claim.args is None:
            claim.args = {}
        lines.append(_answer_line(claim))
        claims.append(claim)
    cs = ClaimSet(unit=unit, kind=kind, flow=flow, claims=claims, largest_length=largest_length(claims))
    return cs, lines


def _known(names) -> str:
    return ", ".join(sorted(names))


def _answer_line(claim: Claim) -> str:
    """The claims-lap answer for one claim (DESIGN.md 6.2); an unknown predicate or measure
    turns the claim into `not_measurable` with the reason, never an error."""
    cid = claim.id
    if claim.kind == "not_measurable":
        return f"{cid}: not measurable here ({claim.not_measurable})"
    if claim.kind == "predicate":
        if claim.predicate not in PREDICATES:
            reason = f"no predicate named {claim.predicate}; known: {_known(PREDICATES)}"
            claim.not_measurable = reason
            claim.kind = "not_measurable"
            return f"{cid}: {reason}"
        return f"{cid} {claim.predicate} of {claim.of}: measurable"
    if claim.kind == "patch":
        where = claim.side if claim.side is not None else _fmt_at(claim.at)
        return f"{cid} patch {claim.patch} {where}: measurable"
    if claim.kind == "count":
        what = claim.measure or ("count" if claim.of not in COUNT_WORDS else claim.of)
        return f"{cid} {what} of {claim.of}: measurable" if claim.of not in COUNT_WORDS else f"{cid} {claim.of}: measurable"
    if claim.measure not in MEASURES:
        reason = f"no measure named {claim.measure}; known: {_known(MEASURES)}"
        claim.not_measurable = reason
        claim.kind = "not_measurable"
        return f"{cid}: {reason}"
    if claim.kind == "report":
        return f"{cid} {claim.measure} of {claim.of}: reported (measured and printed, not judged)"
    return f"{cid} {claim.measure} of {claim.of}: measurable"


def answer(claims: ClaimSet, lines: list[str]) -> str:
    """The claims-lap answer of DESIGN.md 6.2: the counts line, the per-claim lines, and
    the instruction to write the script."""
    measurable = sum(1 for c in claims.claims if c.kind in CHECKABLE_KINDS)
    reported = sum(1 for c in claims.claims if c.kind == "report")
    unmeasurable = sum(1 for c in claims.claims if c.kind == "not_measurable")
    head = f"{len(claims.claims)} claims: {measurable} measurable"
    if reported:
        head += f", {reported} reported"
    if unmeasurable:
        head += f", {unmeasurable} not measurable"
    body = "\n".join(f"  {line}" for line in lines)
    return f"{head}\n{body}\nNow the script. Reply with ONLY a ```python block against the reference card."


def kind_note(claims: ClaimSet, plan: "Plan | None") -> str | None:
    """'c0: kind body_in_flow but the script builds no BodyInBox; judged as a passage' when
    the claims say body_in_flow and the plan has no BodyInBox feature (4.1); None otherwise."""
    if claims.kind != "body_in_flow" or plan is None:
        return None
    if any(getattr(f, "kind", "") == "BodyInBox" for f in plan.features.values()):
        return None
    return "c0: kind body_in_flow but the script builds no BodyInBox; judged as a passage"


# ----------------------------------------------------------------------------- formatting

def fmt_len(v: float) -> str:
    """Lengths to four significant digits with the trailing zeros kept: 3.000, 60.00, 300.0;
    from 1000 up one decimal (1500.0), never a bare trailing point or an exponent, since a
    duct in mm is easily four digits long."""
    text = f"{float(v):#.4g}"
    if "e" in text or text.endswith("."):
        return f"{float(v):.1f}"
    return text


def fmt_angle(v: float) -> str:
    return f"{float(v):.1f}"


ZERO_BELOW = 1e-9
"""A printed number this close to zero is zero: the kernel's turtle leaves sin(pi) =
1.2e-16 on coordinates that are 0, and a table that reads (-1.225e-16, 18) says nothing
a person or the model can use (7.2 prints (0, 18))."""


def _clean(v: float) -> float:
    v = float(v)
    return 0.0 if abs(v) < ZERO_BELOW else v


def fmt_g(v: float) -> str:
    return f"{_clean(v):.4g}"


def fmt_point(p) -> str:
    return f"({_clean(p[0]):.4g}, {_clean(p[1]):.4g})"


def _fmt_at(at) -> str:
    if isinstance(at, (list, tuple)) and len(at) == 2:
        return fmt_point(at)
    return str(at)


def fmt_value(measure: str, v) -> str:
    """One measured value as the table prints it, by the measure's type."""
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(_is_number(x) for x in v):
        return fmt_point(v) if measure in POINT_MEASURES else f"{fmt_g(v[0])}..{fmt_g(v[1])}"
    if isinstance(v, str):
        return v
    if v is None:
        return "none"
    if measure in SWEEP_MEASURES:
        return f"{fmt_g(v)} deg"
    if measure in ANGLE_MEASURES:
        return fmt_angle(v)
    if measure in COUNT_MEASURES:
        return str(int(v))
    if measure in LENGTH_MEASURES:
        return fmt_len(v)
    if isinstance(v, int) and not isinstance(v, bool):
        return str(int(v))
    return fmt_g(v)


def default_tol(measure: str, value: float, span: float) -> float:
    """Judge 5.6: lengths, radii, positions tol = max(1 % of the value, 1e-6 span);
    angles +/- 2 deg; counts exact; extent_* 2 %; area 5 %."""
    if measure in COUNT_MEASURES:
        return COUNT_EXACT
    if measure in ANGLE_MEASURES:
        return ANGLE_TOL_DEG
    if measure in ("extent_x", "extent_y", "extent"):
        return 0.02 * abs(value)
    if measure == "area":
        return 0.05 * abs(value)
    return max(0.01 * abs(value), 1e-6 * span)


def _span(m: Measurements) -> float:
    return max(m.extent[0], m.extent[1], 1e-12)


def _expected_tol(value: float, tol: float) -> str:
    return f"{value:g}" if tol == 0 else f"{value:g} +/- {tol:g}"


# ----------------------------------------------------------------------------- addressing

def feature_names(m: Measurements, plan: "Plan | None") -> list[str]:
    """The script's feature names, for the fail row of a claim on an absent feature: the
    plan's features in script order, then any measured feature the plan does not list,
    then fluid."""
    names: list[str] = []
    if plan is not None:
        names.extend(plan.features.keys())
    for name in m.features:
        if "[" not in name and name not in names and name != "fluid" and name not in m.patches:
            names.append(name)
    if "fluid" not in names:
        names.append("fluid")
    return names


def _row_count(name: str, m: Measurements, plan: "Plan | None") -> int | None:
    if name in m.rows:
        return m.rows[name].count
    if plan is not None and name in plan.instances:
        return int(plan.instances[name].get("count", 0))
    feat = m.features.get(name)
    if feat and _is_number(feat.get("count")):
        return int(feat["count"])
    k = 0
    while f"{name}[{k}]" in m.features:
        k += 1
    return k or None


def resolve(of: str, m: Measurements, plan: "Plan | None") -> tuple[list[tuple[str, dict]] | None, str]:
    """`of` -> the (label, feature dict) instances it names, or (None, why) when nothing is.
    `name[*]` is every instance of a Row; `name[k]` one; `name.legs[k]` one leg of a
    Passage (length, heading, start, end, radius, sweep); `fluid`; a patch name."""
    of = (of or "").strip()
    if not of:
        return None, "the claim names no feature (of is empty)"
    if of == "fluid":
        fluid = dict(m.features.get("fluid", {}))
        fluid.setdefault("extent_x", m.extent[0])
        fluid.setdefault("extent_y", m.extent[1])
        fluid.setdefault("area", m.area)
        fluid.setdefault("islands", m.islands)
        fluid.setdefault("edges", m.n_curves)
        if m.passage is not None:
            fluid.setdefault("narrowest_passage", m.passage.min)
        return [("fluid", fluid)], ""
    if ".legs[" in of and of.endswith("]"):
        name, _, rest = of.partition(".legs[")
        try:
            k = int(rest[:-1])
        except ValueError:
            return None, f"'{of}' is not a leg address; name.legs[k] with k 0-based"
        legs = _legs_of(name, m, plan)
        if legs is None:
            return None, f"no feature named '{name}'; features: {', '.join(feature_names(m, plan))}"
        if not 0 <= k < len(legs):
            return None, f"'{name}' has {len(legs)} legs (legs[0]..legs[{len(legs) - 1}]); no legs[{k}]"
        r = legs[k]
        return [(of, {"length": r.get("length"), "heading": r.get("heading"), "start": r.get("from"),
                      "end": r.get("to"), "radius": r.get("radius"), "sweep": r.get("sweep"),
                      "kind": r.get("kind")})], ""
    if of.endswith("[*]"):
        name = of[:-3]
        count = _row_count(name, m, plan)
        if count is None:
            if f"{name}[*]" in m.features:
                return [(of, m.features[of])], ""
            return None, f"no feature named '{name}'; features: {', '.join(feature_names(m, plan))}"
        found = [(f"{name}[{k}]", m.features[f"{name}[{k}]"]) for k in range(count) if f"{name}[{k}]" in m.features]
        if found:
            return found, ""
        if f"{name}[*]" in m.features:
            return [(of, m.features[of])], ""
        return None, f"'{name}' has no measured instances (a Row's instances are {name}[0]..{name}[{count - 1}])"
    if of.endswith("]") and "[" in of:
        name, _, rest = of.partition("[")
        if of in m.features:
            return [(of, m.features[of])], ""
        count = _row_count(name, m, plan)
        if count is None:
            return None, f"no feature named '{name}'; features: {', '.join(feature_names(m, plan))}"
        return None, f"'{name}' has {count} instances ({name}[0]..{name}[{count - 1}]); no {of}"
    if of in m.features:
        feat = dict(m.features[of])
        if of in m.rows:
            row = m.rows[of]
            for key in ("count", "pitch", "gap", "gap_measured", "anchors", "span"):
                feat.setdefault(key, getattr(row, key))
        return [(of, feat)], ""
    if of in m.rows:
        row = m.rows[of]
        return [(of, {"count": row.count, "pitch": row.pitch, "gap": row.gap, "gap_measured": row.gap_measured,
                      "anchors": row.anchors, "span": row.span})], ""
    if of in m.patches:
        p = m.patches[of]
        feat = {"edges": p.get("n", len(p.get("curves", []))), "length": p.get("length"),
                "midpoint": _patch_midpoint(p), "curves": p.get("curves", [])}
        if _is_number(p.get("width")):
            feat["width"] = p["width"]
        elif feat["edges"] == 1 and _is_number(p.get("length")):
            feat["width"] = p["length"]
        return [(of, feat)], ""
    return None, f"no feature named '{of}'; features: {', '.join(feature_names(m, plan))}"


def _legs_of(name: str, m: Measurements, plan: "Plan | None") -> list[dict] | None:
    if name in m.legs:
        return [r for r in m.legs[name] if r.get("kind") != "ports"]
    if plan is not None and name in plan.features:
        solved = plan.features[name].solved
        if isinstance(solved.get("legs"), list):
            return solved["legs"]
        for op in plan.features[name].ops:
            if op in m.legs:
                return [r for r in m.legs[op] if r.get("kind") != "ports"]
    return None


def _patch_midpoint(p: dict) -> tuple[float, float] | None:
    mids = p.get("midpoints") or []
    if not mids:
        return None
    if len(mids) == 1:
        return (float(mids[0][0]), float(mids[0][1]))
    return (sum(float(q[0]) for q in mids) / len(mids), sum(float(q[1]) for q in mids) / len(mids))


# ----------------------------------------------------------------------------- predicates

def _flow_vector(m: Measurements, args: dict) -> tuple[tuple[float, float] | None, str]:
    word = args.get("_flow")
    if word in FLOW_VECTORS:
        return FLOW_VECTORS[word], word
    if m.flow is not None:
        fx, fy = m.flow
        label = _dir_word((fx, fy))
        return (fx, fy), label
    return None, "unknown"


def _dir_word(v: tuple[float, float]) -> str:
    x, y = v
    if abs(x) >= abs(y):
        return "+x" if x > 0 else "-x"
    return "+y" if y > 0 else "-y"


def _unit(deg: float) -> tuple[float, float]:
    return (math.cos(math.radians(deg)), math.sin(math.radians(deg)))


def _first(feat: dict, *keys):
    for k in keys:
        if k in feat and feat[k] is not None:
            return feat[k]
    return None


def _junction(m: Measurements, label: str, kind: str):
    """The Junction of `kind` ("leave" / "return") recorded for the channel `label`, for a
    feature whose dict carries no angle of its own (a Passage that starts on a wall or
    ends on one has its built angle only in `m.junctions`)."""
    for j in m.junctions:
        if j.kind == kind and (j.channel == label or j.channel.split("[")[0] == label.split("[")[0]):
            return j
    return None


def _junction_angle(m: Measurements, label: str, feat: dict, kind: str):
    key = "leave_angle" if kind == "leave" else "return_angle"
    angle = _first(feat, key, "junction_angle")
    if angle is not None:
        return float(angle)
    j = _junction(m, label, kind)
    return None if j is None else float(j.measured_angle)


def _multi(values: list[str], count: int) -> str:
    """`3.000 (x4)` when every instance agrees, else the values listed."""
    if count == 1:
        return values[0]
    if len(set(values)) == 1:
        return f"{values[0]} (x{count})"
    return ", ".join(values)


def _instances(m: Measurements, of: str, args: dict) -> tuple[list[tuple[str, dict]] | None, str]:
    return resolve(of, m, args.get("_plan"))


def _margin(args: dict, name: str, default: float, stricter) -> tuple[float, str]:
    """A claim may not narrow a predicate margin (3.7): the margin judged is the stricter
    of the default and the claim's (`stricter` picks it: min for an upper bound, max for a
    lower one), and the note says so when the claim's was set aside."""
    given = args.get(name)
    if not _is_number(given):
        return default, ""
    chosen = stricter(float(given), default)
    if chosen != float(given):
        return chosen, f" ({name} {given:g} given; the predicate's margin is {default:g} and is not narrowed)"
    return chosen, ""


def returns_against_flow(m: Measurements, of: str, args: dict) -> Verdict:
    """The last leg's heading makes more than 90 degrees with the flow direction (the
    inlet's inward normal, or the claims' flow word); the margin is the predicate's."""
    return _returns(m, of, args, against=True)


def returns_with_flow(m: Measurements, of: str, args: dict) -> Verdict:
    """The complement of returns_against_flow: the last leg heads within 90 degrees of the flow."""
    return _returns(m, of, args, against=False)


def _returns(m: Measurements, of: str, args: dict, against: bool) -> Verdict:
    insts, why = _instances(m, of, args)
    if insts is None:
        return Verdict(None, why)
    flow, label = _flow_vector(m, args)
    if flow is None:
        return Verdict(None, "the flow direction is unknown: no inlet resolved and the claims give no flow")
    if against:
        min_deg, note = _margin(args, "min_deg", 90.0, max)
    else:
        min_deg, note = _margin(args, "max_deg", 90.0, min)
    texts, oks, senses = [], [], []
    numbers = {}
    for label_i, feat in insts:
        heading = _first(feat, "return_heading", "end_heading")
        if heading is None:
            j = _junction(m, label_i, "return")
            heading = None if j is None else j.heading
        if heading is None:
            return Verdict(None, f"{label_i} has no return heading to judge (no Bypass or Passage end)")
        hx, hy = _unit(float(heading))
        comp = hx * flow[0] + hy * flow[1]
        angle = math.degrees(math.acos(max(-1.0, min(1.0, comp))))
        ok = angle > min_deg if against else angle < min_deg
        oks.append(ok)
        texts.append(f"heading {float(heading):.4g} deg, {angle:.4g} deg from the flow ({label}): component {comp:.2f}")
        if against != (comp < 0):
            senses.append("against" if comp < 0 else "WITH")
        numbers = {"heading": float(heading), "angle": angle, "component": comp}
    expected = (f"> {min_deg:g} deg" if against else f"< {min_deg:g} deg")
    sense = f" ({senses[0]} the flow)" if senses else ""
    return Verdict(all(oks), _multi(texts, len(insts)) + sense + note, numbers=numbers, expected=expected)


def shallow_angle(m: Measurements, of: str, args: dict) -> Verdict:
    """The BUILT leg angle at the junction (the leg's wall against the host wall's flow)
    is at most `max` degrees (30); the lip angle is printed beside it, never judged."""
    insts, why = _instances(m, of, args)
    if insts is None:
        return Verdict(None, why)
    limit, note = _margin(args, "max", 30.0, min)
    texts, oks, numbers = [], [], {}
    for label, feat in insts:
        angle = _junction_angle(m, label, feat, "leave")
        if angle is None:
            return Verdict(None, f"{label} has no junction angle (no leg leaves a wall)")
        oks.append(angle <= limit)
        texts.append(fmt_angle(angle))
        numbers = {"angle": angle}
    lip = _first(insts[0][1], "lip_angle")
    if lip is None:
        j = _junction(m, insts[0][0], "leave")
        lip = None if j is None or j.lip is None else j.lip[1]
    tail = " built at the leg's wall" + (f"; lip {fmt_angle(lip)} on the arc" if lip is not None else "")
    return Verdict(all(oks), f"{of}.leave_angle = {_multi(texts, len(insts))}{tail}{note}", numbers=numbers,
                   expected=f"<= {limit:g} deg")


def steep_angle(m: Measurements, of: str, args: dict) -> Verdict:
    """The BUILT leg angle at the junction is at least `min` degrees (60)."""
    insts, why = _instances(m, of, args)
    if insts is None:
        return Verdict(None, why)
    limit, note = _margin(args, "min", 60.0, max)
    texts, oks, numbers = [], [], {}
    for label, feat in insts:
        angle = _junction_angle(m, label, feat, "return")
        if angle is None:
            angle = _junction_angle(m, label, feat, "leave")
        if angle is None:
            return Verdict(None, f"{label} has no junction angle (no leg meets a wall)")
        oks.append(angle >= limit)
        texts.append(fmt_angle(angle))
        numbers = {"angle": angle}
    return Verdict(all(oks), f"{of} angle at the wall = {_multi(texts, len(insts))} built at the leg's wall{note}",
                   numbers=numbers, expected=f">= {limit:g} deg")


def lands_on_wall(m: Measurements, of: str, args: dict) -> Verdict:
    """Every leg recorded as landing on a wall passed the landing lint (D22): no E-LAND
    Finding names the feature."""
    insts, why = _instances(m, of, args)
    if insts is None:
        return Verdict(None, why)
    findings = args.get("_findings") or []
    name = of.split("[")[0]
    hits = [f for f in findings if f.code == "E-LAND" and name in f.subject]
    if hits:
        f = hits[0]
        return Verdict(False, f"{f.subject}: {f.what}", where=f.where, expected="lands on the wall")
    return Verdict(True, f"{of} lands on its wall (no E-LAND)", expected="lands on the wall")


def stacked_in(m: Measurements, of: str, args: dict) -> Verdict:
    """Pass headings alternate 0/180 (axis y) or 90/270 (axis x) within 5 degrees and the
    pass centrelines are spaced by a constant within 5 %."""
    insts, why = _instances(m, of, args)
    if insts is None:
        return Verdict(None, why)
    axis = str(args.get("axis", "y"))
    feat = insts[0][1]
    lines = feat.get("pass_centrelines")
    headings = feat.get("pass_headings")
    if headings is None:
        legs = _legs_of(of, m, args.get("_plan")) or []
        headings = [float(r["heading"]) % 360 for r in legs if r.get("kind") == "line"]
    if not headings or lines is None:
        return Verdict(None, f"{of} has no passes to judge (a Serpentine or a Passage of straight legs)")
    base = (0.0, 180.0) if axis == "y" else (90.0, 270.0)
    ok_headings = all(min(abs((h - b + 180) % 360 - 180) for b in base) <= 5.0 for h in headings)
    alternate = all(abs((headings[i] - headings[i - 1] + 180) % 360 - 180) > 175 for i in range(1, len(headings)))
    spacings = [float(lines[i]) - float(lines[i - 1]) for i in range(1, len(lines))]
    even = (not spacings) or all(abs(s - spacings[0]) <= 0.05 * abs(spacings[0]) for s in spacings)
    count = args.get("count")
    count_ok = count is None or int(count) == len(headings)
    text = (f"headings {'/'.join(f'{h:g}' for h in headings)}; centrelines {axis} = "
            f"{', '.join(f'{float(v):g}' for v in lines)}"
            + (f" (spacing {spacings[0]:g})" if spacings else ""))
    if not count_ok:
        text += f"; {len(headings)} passes, not {count}"
    return Verdict(ok_headings and alternate and even and count_ok, text,
                   expected=f"stacked in {axis}" + (f", {count} passes" if count is not None else ""))


def bends_are_u(m: Measurements, of: str, args: dict) -> Verdict:
    """Every arc leg sweeps 180 +/- 2 degrees."""
    legs = _legs_of(of, m, args.get("_plan"))
    if legs is None:
        insts, why = _instances(m, of, args)
        return Verdict(None, why if insts is None else f"{of} has no leg table")
    sweeps = [abs(float(r["sweep"])) for r in legs if r.get("kind") == "arc" and _is_number(r.get("sweep"))]
    if not sweeps:
        return Verdict(False, f"{of} has no arc legs", expected="180 +/- 2")
    ok = all(abs(s - 180.0) <= 2.0 for s in sweeps)
    word = "sweep" if len(sweeps) == 1 else "sweeps"
    return Verdict(ok, f"{word} {', '.join(f'{s:g}' for s in sweeps)} deg", expected="180 +/- 2",
                   numbers={"sweeps": sweeps})


def parallel_legs(m: Measurements, of: str, args: dict) -> Verdict:
    """Two straight legs with headings 180 degrees apart within 2 degrees, their centrelines
    `spacing` apart within `tol` (1 %)."""
    legs = _legs_of(of, m, args.get("_plan"))
    if legs is None:
        return Verdict(None, f"no feature named '{of}' with a leg table")
    lines = [(i + 1, r) for i, r in enumerate(legs) if r.get("kind") == "line"]
    spacing = args.get("spacing")
    best = None
    for a in range(len(lines)):
        for b in range(a + 1, len(lines)):
            ia, ra = lines[a]
            ib, rb = lines[b]
            dh = abs((float(ra["heading"]) - float(rb["heading"]) + 180) % 360 - 180)
            if abs(dh - 180.0) > 2.0:
                continue
            ux, uy = _unit(float(ra["heading"]))
            dx = float(rb["from"][0]) - float(ra["from"][0])
            dy = float(rb["from"][1]) - float(ra["from"][1])
            dist = abs(dx * -uy + dy * ux)
            if best is None or (spacing is not None and abs(dist - float(spacing)) < abs(best[2] - float(spacing))):
                best = (ia, ib, dist, float(ra["heading"]) % 360, float(rb["heading"]) % 360)
    if best is None:
        return Verdict(False, f"{of} has no two straight legs 180 deg apart",
                       expected=f"{float(spacing):g} +/- {0.01 * float(spacing):g}" if _is_number(spacing) else "parallel")
    ia, ib, dist, ha, hb = best
    text = f"legs {ia} and {ib}: headings {ha:g} / {hb:g}, {dist:.2f} apart"
    if not _is_number(spacing):
        return Verdict(True, text, expected="parallel")
    tol = float(args["tol"]) if _is_number(args.get("tol")) else 0.01 * float(spacing)
    tol = max(tol, 0.01 * float(spacing))
    return Verdict(abs(dist - float(spacing)) <= tol, text, expected=f"{float(spacing):g} +/- {tol:g}",
                   numbers={"spacing": dist})


def _open_end_axis_word(n: tuple[float, float]) -> str:
    return _dir_word(n)


def _ends_by_name(m: Measurements) -> list:
    return list(m.open_ends)


def ends_on_same_side(m: Measurements, of: str, args: dict) -> Verdict:
    """Both open ends' outward normals lie within 20 degrees of the same axis direction
    (`side`, when given: -x, +x, -y, +y)."""
    ends = sorted(_ends_by_name(m), key=lambda e: (round(e.centre[0], 6), round(e.centre[1], 6)))
    if len(ends) < 2:
        return Verdict(False, f"{len(ends)} open end(s); two are needed", expected=str(args.get("side", "same side")))
    words = [_dir_word(e.outward_normal) for e in ends]
    within = [_within_deg(e.outward_normal, FLOW_VECTORS[w], 20.0) for e, w in zip(ends, words)]
    side = args.get("side")
    same = len(set(words)) == 1 and all(within)
    if side is not None:
        same = same and words[0] == side
    text = " and ".join(f"open ends at {fmt_point(e.centre)} facing {w}" if i == 0 else f"{fmt_point(e.centre)} facing {w}"
                        for i, (e, w) in enumerate(zip(ends, words)))
    return Verdict(same, text, expected=str(side) if side is not None else "same side")


def ends_on_opposite_sides(m: Measurements, of: str, args: dict) -> Verdict:
    """The two open ends face opposite ways along one axis, each within 20 degrees of it."""
    ends = _ends_by_name(m)
    if len(ends) < 2:
        return Verdict(False, f"{len(ends)} open end(s); two are needed", expected="opposite sides")
    words = [_dir_word(e.outward_normal) for e in ends]
    opposite = (len(words) == 2 and words[0][1] == words[1][1] and words[0][0] != words[1][0]
                and all(_within_deg(e.outward_normal, FLOW_VECTORS[w], 20.0) for e, w in zip(ends, words)))
    text = " and ".join(f"{fmt_point(e.centre)} facing {w}" for e, w in zip(ends, words))
    return Verdict(opposite, "open ends at " + text, expected="opposite sides")


def _within_deg(a: tuple[float, float], b: tuple[float, float], deg: float) -> bool:
    dot = a[0] * b[0] + a[1] * b[1]
    return math.degrees(math.acos(max(-1.0, min(1.0, dot)))) <= deg


def _port_faces(m: Measurements, port: str, args: dict) -> Verdict:
    d = args.get("dir")
    if d not in FLOW_VECTORS:
        return Verdict(None, f"dir must be one of {', '.join(FLOWS)}", expected=str(d))
    ends = [e for e in m.open_ends if e.name == port]
    if not ends:
        return Verdict(False, f"no open end is named {port}", expected=f"{d} within 20 deg")
    n = ends[0].outward_normal
    ok = _within_deg(n, FLOW_VECTORS[d], 20.0)
    return Verdict(ok, f"{port}'s outward normal ({n[0]:.4g}, {n[1]:.4g})", expected=f"{d} within 20 deg")


def inlet_faces(m: Measurements, of: str, args: dict) -> Verdict:
    """The inlet's outward normal is within 20 degrees of `dir`."""
    return _port_faces(m, "inlet", args)


def outlet_faces(m: Measurements, of: str, args: dict) -> Verdict:
    """The outlet's outward normal is within 20 degrees of `dir`."""
    return _port_faces(m, "outlet", args)


def _centroid(feat: dict):
    c = _first(feat, "centre", "centroid", "midpoint")
    return None if c is None else (float(c[0]), float(c[1]))


def at_distance_from(m: Measurements, of: str, args: dict) -> Verdict:
    """The feature's centroid is `value` from `patch` (inlet) along the flow, within `tol` (1 %)."""
    insts, why = _instances(m, of, args)
    if insts is None:
        return Verdict(None, why)
    patch = str(args.get("patch", "inlet"))
    value = args.get("value")
    if not _is_number(value):
        return Verdict(None, "at_distance_from needs value")
    p = m.patches.get(patch)
    origin = _patch_midpoint(p) if p else None
    if origin is None:
        return Verdict(False, f"no patch named {patch}", expected=f"{float(value):g}")
    flow, _ = _flow_vector(m, args)
    if flow is None:
        flow = (1.0, 0.0)
    tol = max(float(args["tol"]) if _is_number(args.get("tol")) else 0.0, 0.01 * abs(float(value)))
    c = _centroid(insts[0][1])
    if c is None:
        return Verdict(None, f"{of} has no centre to measure from")
    dist = (c[0] - origin[0]) * flow[0] + (c[1] - origin[1]) * flow[1]
    axis = "x" if abs(flow[0]) >= abs(flow[1]) else "y"
    at = origin[0] if axis == "x" else origin[1]
    return Verdict(abs(dist - float(value)) <= tol,
                   f"{of}.centre {fmt_point(c)}; {patch} at {axis} = {at:.4g}: {dist:.1f} from it",
                   expected=f"{float(value):g} +/- {tol:g}", numbers={"distance": dist})


def centred_in(m: Measurements, of: str, args: dict) -> Verdict:
    """The feature's centroid across the parent equals the parent's mid within 1 % of the
    parent's size across (`in`, `axis`)."""
    insts, why = _instances(m, of, args)
    if insts is None:
        return Verdict(None, why)
    parent_name = str(args.get("in") or args.get("of") or "")
    parents, why = resolve(parent_name, m, args.get("_plan"))
    if parents is None:
        return Verdict(False, why, expected="centred")
    axis = str(args.get("axis", "y"))
    parent = parents[0][1]
    pc = _centroid(parent)
    size = _first(parent, "size_y" if axis == "y" else "size_x", "height" if axis == "y" else "length",
                  "extent_y" if axis == "y" else "extent_x")
    c = _centroid(insts[0][1])
    if pc is None or size is None or c is None:
        return Verdict(None, f"{parent_name} has no centre and size along {axis} to compare with")
    i = 1 if axis == "y" else 0
    tol = 0.01 * float(size)
    word = "mid-height" if axis == "y" else "mid-length"
    return Verdict(abs(c[i] - pc[i]) <= tol,
                   f"{of}.centre.{axis} = {c[i]:.1f} = {parent_name} {word} {pc[i]:.1f}",
                   expected=f"+/- {tol:g}", numbers={"offset": c[i] - pc[i]})


def at_mid_height(m: Measurements, of: str, args: dict) -> Verdict:
    """centred_in(in=of, axis="y") by another name."""
    return centred_in(m, of, {**args, "in": args.get("of") or args.get("in"), "axis": "y"})


def own_patch(m: Measurements, of: str, args: dict) -> Verdict:
    """The named patch's curves are exactly the curves of `args.of`'s resolved edge."""
    p = m.patches.get(of)
    if p is None:
        return Verdict(False, f"no patch named '{of}'; patches: {', '.join(m.patches)}", expected="own patch")
    feature = str(args.get("of", ""))
    curves = list(p.get("curves", []))
    target = m.features.get(feature, {}).get("curves")
    if target is None:
        return Verdict(True if curves else False,
                       f"patch '{of}' = the {len(curves)} curve{'s' if len(curves) != 1 else ''} of {feature}.edge",
                       expected="own patch")
    ok = sorted(curves) == sorted(target)
    return Verdict(ok, f"patch '{of}' = the {len(curves)} curve{'s' if len(curves) != 1 else ''} of {feature}.edge"
                   + ("" if ok else f" (the edge has {len(target)})"), expected="own patch")


def walls_are(m: Measurements, of: str, args: dict) -> Verdict:
    """The named patch is every curve that is not inlet/outlet and not in `except`."""
    p = m.patches.get(of)
    if p is None:
        return Verdict(False, f"no patch named '{of}'; patches: {', '.join(m.patches)}", expected="the rest")
    excluded = set(args.get("except", [])) | {"inlet", "outlet"}
    others = set()
    for name, q in m.patches.items():
        if name != of and name not in excluded:
            others.update(q.get("curves", []))
    mine = set(p.get("curves", []))
    ok = not (mine & others) and len(mine) + sum(len(m.patches[n].get("curves", [])) for n in excluded if n in m.patches) == m.n_curves
    names = "/".join(["inlet", "outlet"] + [e for e in args.get("except", [])])
    return Verdict(ok, f"patch '{of}' = every curve not {names} ({len(mine)})", expected="the rest")


def sharp_corner(m: Measurements, of: str, args: dict) -> Verdict:
    """The passage has a `corner` leg within one width of `near` whose typed turn is
    `angle`, and the walk has two line-line vertices within one width of `near`, one with
    interior `angle +/- 3` (the outer corner) and one with `360 - angle +/- 3` (the inner)
    (D36); a Fuse of rects is judged by the walk alone."""
    near = args.get("near")
    angle = float(args.get("angle", 90))
    if not (isinstance(near, (list, tuple)) and len(near) == 2):
        return Verdict(None, "sharp_corner needs near: [x, y]")
    nx, ny = float(near[0]), float(near[1])
    width = m.reference_width
    feat = m.features.get(of, {})
    w = float(feat.get("width") or width)
    legs = _legs_of(of, m, args.get("_plan")) or []
    corner_legs = [r for r in legs if r.get("kind") == "corner"
                   and math.dist((float(r["from"][0]), float(r["from"][1])), (nx, ny)) <= w
                   and abs(abs(float(r.get("turn", r.get("sweep", 0)))) - angle) <= 3.0]
    outer = [v for v in m.vertices if math.dist(v.at, (nx, ny)) <= w and abs(v.interior_deg - angle) <= 3.0
             and _both_lines(m, v)]
    inner = [v for v in m.vertices if math.dist(v.at, (nx, ny)) <= w and abs(v.interior_deg - (360 - angle)) <= 3.0
             and _both_lines(m, v)]
    ok = bool(outer) and bool(inner) and (bool(corner_legs) or not legs)
    parts = [f"corner at ({nx:.4g}, {ny:.4g})"]
    if outer:
        parts.append(f"outer vertex {fmt_point(outer[0].at)} interior {outer[0].interior_deg:.1f}")
    else:
        parts.append("no outer vertex of that angle within a width")
    if inner:
        parts.append(f"inner {fmt_point(inner[0].at)} interior {inner[0].interior_deg:.1f}")
    else:
        parts.append("no inner vertex within a width")
    if outer and inner:
        parts.append("both lines")
    if legs and not corner_legs:
        parts.append("no corner leg there")
    return Verdict(ok, ": ".join([parts[0], ", ".join(parts[1:])]), expected=f"{angle:g} +/- 3",
                   where=(nx, ny))


def _both_lines(m: Measurements, v) -> bool:
    kinds = []
    for tag in v.curves:
        c = _curve_kind(m, tag)
        kinds.append(c)
    return all(k in (None, "line") for k in kinds)


def _curve_kind(m: Measurements, tag: int) -> str | None:
    for p in m.patches.values():
        kinds = p.get("kinds")
        if kinds and tag in p.get("curves", []):
            return kinds[p["curves"].index(tag)]
    return None


def rounded_corner(m: Measurements, of: str, args: dict) -> Verdict:
    """An arc leg within w/2 of `near` with `radius +/- 1 %`."""
    near = args.get("near")
    radius = args.get("radius")
    if not (isinstance(near, (list, tuple)) and len(near) == 2) or not _is_number(radius):
        return Verdict(None, "rounded_corner needs near: [x, y] and radius")
    legs = _legs_of(of, m, args.get("_plan")) or []
    feat = m.features.get(of, {})
    w = float(feat.get("width") or m.reference_width)
    nx, ny = float(near[0]), float(near[1])
    arcs = [r for r in legs if r.get("kind") == "arc" and r.get("centre") is not None
            and math.dist((float(r["centre"][0]), float(r["centre"][1])), (nx, ny)) <= max(w / 2, float(radius)) + w / 2]
    if not arcs:
        return Verdict(False, f"no arc within a width of ({nx:.4g}, {ny:.4g})", expected=f"r {float(radius):g} +/- 1 %")
    r = float(arcs[0]["radius"])
    return Verdict(abs(r - float(radius)) <= 0.01 * float(radius), f"arc r {r:.4g} centre {fmt_point(arcs[0]['centre'])}",
                   expected=f"r {float(radius):g} +/- 1 %", numbers={"radius": r})


def one_inlet_one_outlet(m: Measurements, of: str, args: dict) -> Verdict:
    """Exactly one open end named inlet and one named outlet."""
    inlets = [e for e in m.open_ends if e.name == "inlet"]
    outlets = [e for e in m.open_ends if e.name == "outlet"]
    if not m.open_ends:
        inlets = [1] if m.patches.get("inlet", {}).get("n") == 1 else []
        outlets = [1] if m.patches.get("outlet", {}).get("n") == 1 else []
    return Verdict(len(inlets) == 1 and len(outlets) == 1,
                   f"{len(inlets)} inlet, {len(outlets)} outlet", expected="1 inlet, 1 outlet")


def count_islands_equals_count(m: Measurements, of: str, args: dict) -> Verdict:
    """The fluid's islands equal the Row's count (one island inside each loop)."""
    count = _row_count(of, m, args.get("_plan"))
    if count is None:
        return Verdict(None, f"no Row named '{of}'")
    return Verdict(m.islands == count, f"islands = {m.islands}, {of}.count = {count}", expected=f"islands = {count}")


def symmetric_about(m: Measurements, of: str, args: dict) -> Verdict:
    """Symmetry by a mirror-boolean: not in Phase 1 (D20)."""
    return Verdict(None, "symmetric_about is not in Phase 1 (symmetry by mirror-boolean waits for the library shapes that need it)")


def all_hex(m: Measurements, of: str, args: dict) -> Verdict:
    """Every cell a hexahedron: judged by the fitness table after the mesh, not in Phase 1."""
    return Verdict(None, "all_hex is not in Phase 1 (the fitness table reports hex_fraction after the mesh)")


PREDICATES: dict[str, Predicate] = {
    "returns_against_flow": returns_against_flow,
    "returns_with_flow": returns_with_flow,
    "shallow_angle": shallow_angle,
    "steep_angle": steep_angle,
    "lands_on_wall": lands_on_wall,
    "stacked_in": stacked_in,
    "bends_are_u": bends_are_u,
    "parallel_legs": parallel_legs,
    "ends_on_same_side": ends_on_same_side,
    "ends_on_opposite_sides": ends_on_opposite_sides,
    "inlet_faces": inlet_faces,
    "outlet_faces": outlet_faces,
    "at_distance_from": at_distance_from,
    "centred_in": centred_in,
    "at_mid_height": at_mid_height,
    "own_patch": own_patch,
    "walls_are": walls_are,
    "sharp_corner": sharp_corner,
    "rounded_corner": rounded_corner,
    "one_inlet_one_outlet": one_inlet_one_outlet,
    "count_islands_equals_count": count_islands_equals_count,
    "symmetric_about": symmetric_about,
    "all_hex": all_hex,
}
"""name -> callable. The Phase 1 predicates of DESIGN.md 3.7's table; `symmetric_about`
and `all_hex` answer not measurable until Phase 2."""


# ----------------------------------------------------------------------------- the table

@dataclass
class ComplianceRow:
    id: str
    says: str
    kind: str
    expected: str
    """"3 +/- 0.03", "4", "<= 30 deg", "inlet at (0, 0) +/- 0.5"."""
    measured: str
    """"3.000 (main.width)" | "no feature named 'loops'; features: main, loop, row, fluid"."""
    verdict: Literal["pass", "fail", "not_measurable", "disagreed"]
    where: tuple[float, float] | None = None
    numbers: dict = field(default_factory=dict)
    detail: str = ""
    """The explanation under a FAIL (T02's c8 sentence)."""

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ComplianceRow":
        where = d.get("where")
        return ComplianceRow(id=d["id"], says=d["says"], kind=d["kind"], expected=d.get("expected", ""),
                             measured=d.get("measured", ""), verdict=d["verdict"],
                             where=None if where is None else (float(where[0]), float(where[1])),
                             numbers=dict(d.get("numbers", {})), detail=d.get("detail", ""))

    def text(self) -> str:
        """The row and its detail lines, the way the CLAIMS block prints them."""
        return "\n".join(_row_lines(self))


CHECKABLE_KINDS = frozenset({"measure", "range", "count", "predicate", "patch"})

SAYS_WIDTH = 34
MEASURED_WIDTH = 32
EXPECTED_WIDTH = 16
ID_WIDTH = 5
DETAIL_INDENT = " " * ID_WIDTH


def _cell(text: str, width: int, overflowed: bool) -> tuple[str, bool]:
    """A column of the CLAIMS block: padded to its width when it fits with a space to
    spare, or followed by three spaces once a cell has overflowed (the row then reads as
    prose, still in column order). A `says` is never abbreviated: the request's own words
    are the point of the column, and the worked T02/T05 rows overflow in full."""
    if overflowed or len(text) >= width:
        return text + "   ", True
    return text.ljust(width), False


def _row_lines(r: ComplianceRow) -> list[str]:
    head = r.id.ljust(ID_WIDTH)
    says, over = _cell(r.says, SAYS_WIDTH, False)
    if r.verdict == "not_measurable":
        lines = [head + says + r.measured]
    else:
        measured, over = _cell(r.measured, MEASURED_WIDTH, over)
        expected, over = _cell(r.expected, EXPECTED_WIDTH, over)
        verdict = {"pass": "pass", "fail": "FAIL", "disagreed": "disagreed"}[r.verdict]
        lines = [(head + says + measured + expected + verdict).rstrip()]
    for line in (r.detail or "").splitlines():
        if line.strip():
            lines.append(DETAIL_INDENT + line.rstrip())
    return lines


@dataclass
class ComplianceTable:
    rows: list[ComplianceRow]
    notes: list[str] = field(default_factory=list)
    """Lines printed under the CLAIMS header before the rows: the body_in_flow note of 4.1."""

    @property
    def passed(self) -> int:
        """Every row whose verdict is `pass`, report rows included (a report row was
        measured and printed, which is its pass); the summary line counts those under
        `reported` instead."""
        return sum(1 for r in self.rows if r.verdict == "pass")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.rows if r.verdict == "fail")

    @property
    def unmeasurable(self) -> int:
        return sum(1 for r in self.rows if r.verdict == "not_measurable")

    @property
    def disagreed(self) -> int:
        return sum(1 for r in self.rows if r.verdict == "disagreed")

    @property
    def reported(self) -> int:
        return sum(1 for r in self.rows if r.kind == "report")

    @property
    def checkable(self) -> int:
        """Rows of kind measure / range / count / predicate / patch (report and
        not_measurable do not count)."""
        return sum(1 for r in self.rows if r.kind in CHECKABLE_KINDS)

    def ok(self) -> bool:
        """failed == 0 AND checkable >= 1 (D33: an empty or report-only table never passes)."""
        return self.failed == 0 and self.checkable >= 1

    def failing(self) -> list[ComplianceRow]:
        return [r for r in self.rows if r.verdict == "fail"]

    def failing_ids(self) -> list[str]:
        return [r.id for r in self.failing()]

    def by_id(self, claim_id: str) -> ComplianceRow | None:
        return next((r for r in self.rows if r.id == claim_id), None)

    def summary(self) -> str:
        """'12 claims: 9 pass, 0 FAIL, 1 not measurable, 2 reported'. The FAIL count is
        printed whenever some row is neither a pass nor not-measurable (a failure, a
        disagreement, a report row), so a table with reported rows still says '0 FAIL';
        a table where every row passes reads '9 claims: 9 pass', and one of passes and
        not-measurable rows alone reads '12 claims: 11 pass, 1 not measurable'."""
        n = len(self.rows)
        passed = sum(1 for r in self.rows if r.verdict == "pass" and r.kind != "report")
        parts = [f"{passed} pass"]
        if passed + self.unmeasurable != n:
            parts.append(f"{self.failed} FAIL")
        if self.disagreed:
            parts.append(f"{self.disagreed} disagreed")
        if self.unmeasurable:
            parts.append(f"{self.unmeasurable} not measurable")
        if self.reported:
            parts.append(f"{self.reported} reported")
        return f"{n} claim{'s' if n != 1 else ''}: {', '.join(parts)}"

    def lines(self) -> list[str]:
        """The CLAIMS block of section 6: the summary, the notes, one line per row with its
        detail lines under it."""
        out = [self.summary()]
        out.extend(self.notes)
        for r in self.rows:
            out.extend(_row_lines(r))
        return out

    def disagree(self, ids: list[str]) -> str:
        """Marks the named failing rows `disagreed` and returns their text verbatim
        (`GeometryResult.disagreement_text`, D18)."""
        texts = []
        for r in self.rows:
            if r.id in ids and r.verdict == "fail":
                r.verdict = "disagreed"
                texts.append(r.text())
        return "\n".join(texts)

    def as_dict(self) -> dict:
        return {"rows": [r.as_dict() for r in self.rows], "notes": list(self.notes)}

    @staticmethod
    def from_dict(d: dict) -> "ComplianceTable":
        return ComplianceTable(rows=[ComplianceRow.from_dict(r) for r in d.get("rows", [])],
                               notes=list(d.get("notes", [])))


# ----------------------------------------------------------------------------- comply

def comply(claims: ClaimSet, m: Measurements, findings: list[Finding], plan: "Plan | None") -> ComplianceTable:
    """One row per claim. `of` joins on the script's `name=`: `name[*]` = every instance of
    a Row (all must pass; the measured column says `3.0 (x4)`), `name[k]` = one instance;
    `fluid`, a patch name, or `inlet`/`outlet`. A claim naming an absent feature is a
    `fail` row whose `measured` lists the script's feature names (D12)."""
    rows = [_row_for(c, claims, m, findings, plan) for c in claims.claims]
    note = kind_note(claims, plan)
    return ComplianceTable(rows=rows, notes=[note] if note else [])


def _row(c: Claim, verdict: str, measured: str, expected: str = "", where=None, numbers=None, detail: str = "") -> ComplianceRow:
    return ComplianceRow(id=c.id, says=c.says, kind=c.kind, expected=expected, measured=measured, verdict=verdict,
                         where=where, numbers=dict(numbers or {}), detail=detail)


def _row_for(c: Claim, claims: ClaimSet, m: Measurements, findings: list[Finding], plan: "Plan | None") -> ComplianceRow:
    if c.kind == "not_measurable":
        reason = (c.not_measurable or "").split(";")[0].strip()
        return _row(c, "not_measurable", f"not measurable here: {reason}" if reason else "not measurable here")
    if c.kind == "predicate":
        return _predicate_row(c, claims, m, findings, plan)
    if c.kind == "patch":
        return _patch_row(c, m, plan)
    if c.kind == "count":
        return _count_row(c, m, plan)
    insts, why = resolve(c.of, m, plan)
    if insts is None:
        return _row(c, "fail", why, _expected_for(c, m))
    if c.measure not in MEASURES:
        return _row(c, "not_measurable", f"no measure named {c.measure}; known: {_known(MEASURES)}")
    values = []
    for label, feat in insts:
        if c.measure not in feat or feat[c.measure] is None:
            kind = _kind_of(label, plan)
            return _row(c, "not_measurable",
                        f"no measure named {c.measure} on {kind} '{label}'; it has: {', '.join(sorted(k for k in feat if not k.startswith('_')))}")
        values.extend(_per_instance(c.measure, feat[c.measure]))
    label = c.of
    count = len(values)
    if c.kind == "report":
        text = f"{label}.{c.measure} = {_multi([fmt_value(c.measure, v) for v in values], count)}"
        return _row(c, "pass", text, "reported", numbers={"values": _plain(values)})
    if not all(_is_number(v) for v in values):
        text = f"{label}.{c.measure} = {_multi([fmt_value(c.measure, v) for v in values], count)}"
        return _row(c, "not_measurable", f"{text} is not a number to compare with {c.value}")
    span = _span(m)
    if c.kind == "range":
        lo, hi = float(c.min), float(c.max)
        ok = all(lo <= float(v) <= hi for v in values)
        text = f"{label}.{c.measure} = {_multi([fmt_value(c.measure, v) for v in values], count)}"
        return _row(c, "pass" if ok else "fail", text, f"{lo:g}..{hi:g}", numbers={"values": _plain(values)})
    value = float(c.value)
    tol = _tol_for(c, value, span)
    ok = all(abs(float(v) - value) <= tol for v in values)
    note = _curvature_note(c.measure, _kind_of(insts[0][0], plan))
    if c.measure == "legs_straight" and count == 1:
        note = _straight_lengths(label, m, plan)
    shown = [fmt_value(c.measure, v) for v in values]
    if count > 1 and len(set(shown)) == 1:
        body = f"{shown[0]} (x{count}{', ' + note if note else ''})"
    elif count > 1:
        body = ", ".join(shown) + (f" ({note})" if note else "")
    else:
        body = shown[0] + (f" ({note})" if note else "")
    text = f"{label}.{c.measure} = {body}"
    detail = "" if ok else _measure_detail(c, values, value, tol)
    return _row(c, "pass" if ok else "fail", text, _expected_tol(value, tol), numbers={"values": _plain(values)},
                detail=detail)


def _plain(values):
    return [float(v) if _is_number(v) else v for v in values]


def _per_instance(measure: str, value) -> list:
    """A feature value as the values a claim judges: a list of numbers on a measure that
    is not a pair or a whole list is one number per pass / bend (T02's `pass_length` is
    judged four times and printed `30.00 (x4)`); anything else is the one value."""
    if (isinstance(value, (list, tuple)) and measure not in PAIR_MEASURES and measure not in LIST_MEASURES
            and value and all(_is_number(v) for v in value)):
        return list(value)
    return [value]


def _straight_lengths(label: str, m: Measurements, plan: "Plan | None") -> str:
    """The straight legs' lengths beside a `legs_straight` count (7.3's `2 (60.0, 60.0)`)."""
    legs = _legs_of(label, m, plan) or []
    lengths = [float(r["length"]) for r in legs if r.get("kind") == "line" and _is_number(r.get("length"))]
    return ", ".join(f"{v:.1f}" for v in lengths) if lengths else ""


def _kind_of(label: str, plan: "Plan | None") -> str:
    name = label.split("[")[0].split(".")[0]
    if plan is not None and name in plan.features:
        return plan.features[name].kind
    return "feature"


def _curvature_note(measure: str, kind: str) -> str:
    """The note beside a Bypass radius (7.1's c5); a Disk's radius says where it came from
    on its FEATURES line and its claims row is the bare number (7.4's c1)."""
    if measure == "outer_radius":
        return "from the outer arc's curvature"
    if measure == "inner_radius":
        return "from the inner arc's curvature"
    return ""


def _tol_for(c: Claim, value: float, span: float) -> float:
    if _is_number(c.tol):
        return float(c.tol)
    if _is_number(c.tol_rel):
        return float(c.tol_rel) * abs(value)
    return default_tol(c.measure, value, span)


def _expected_for(c: Claim, m: Measurements) -> str:
    if c.kind == "measure" and _is_number(c.value):
        return _expected_tol(float(c.value), _tol_for(c, float(c.value), _span(m)))
    if c.kind == "range":
        return f"{float(c.min):g}..{float(c.max):g}"
    if c.kind == "count" and _is_number(c.value):
        return f"{float(c.value):g}"
    if c.kind == "report":
        return "reported"
    return ""


def _measure_detail(c: Claim, values, value: float, tol: float) -> str:
    worst = max(values, key=lambda v: abs(float(v) - value))
    off = float(worst) - value
    return (f"{c.measure} reads {fmt_value(c.measure, worst)}, {abs(off):.4g} {'above' if off > 0 else 'below'} the "
            f"{value:g} the request states (tolerance {tol:g})")


def _count_row(c: Claim, m: Measurements, plan: "Plan | None") -> ComplianceRow:
    value = int(round(float(c.value)))
    expected = f"{value}"
    of = (c.of or "").strip()
    if of in COUNT_WORDS and not c.measure:
        n = _count_word(of, m, plan)
        if n is None:
            return _row(c, "not_measurable", f"{of} is not counted on this shape")
        return _row(c, "pass" if n == value else "fail", f"{of} = {n}", expected, numbers={"count": n},
                    detail="" if n == value else f"the built shape has {n} {of}, the request says {value}")
    insts, why = resolve(of, m, plan)
    if insts is None:
        return _row(c, "fail", why, expected)
    key = c.measure or "count"
    label, feat = insts[0]
    n = feat.get(key)
    if n is None and key == "count":
        n = _row_count(of, m, plan)
    if n is None:
        return _row(c, "not_measurable", f"no count named {key} on '{label}'; it has: {', '.join(sorted(feat))}")
    n = int(round(float(n)))
    return _row(c, "pass" if n == value else "fail", f"{label}.{key} = {n}", expected, numbers={"count": n},
                detail="" if n == value else f"the built shape has {n}, the request says {value}")


def _count_word(word: str, m: Measurements, plan: "Plan | None") -> int | None:
    if word == "islands" or word == "holes":
        return m.islands if word == "islands" else len(m.holes)
    if word == "open_ends":
        return len(m.open_ends)
    if word in ("inlets", "outlets"):
        name = word[:-1]
        if m.open_ends:
            n = sum(1 for e in m.open_ends if e.name == name)
            if n or name not in m.patches:
                return n
        p = m.patches.get(name)
        return 0 if p is None else int(p.get("n", len(p.get("curves", []))))
    if word == "edges":
        return m.n_curves
    if word in ("bends", "passes", "corners"):
        total, seen = 0, False
        for name, feat in m.features.items():
            if "[" in name or name == "fluid":
                continue
            if _is_number(feat.get(word)):
                total += int(feat[word])
                seen = True
        return total if seen else None
    return None


def _predicate_row(c: Claim, claims: ClaimSet, m: Measurements, findings: list[Finding], plan: "Plan | None") -> ComplianceRow:
    fn = PREDICATES.get(c.predicate)
    if fn is None:
        return _row(c, "not_measurable", f"no predicate named {c.predicate}; known: {_known(PREDICATES)}")
    args = dict(c.args or {})
    args["_flow"] = claims.flow
    args["_plan"] = plan
    args["_findings"] = findings
    if c.of and c.of not in ("fluid",) and not c.of.startswith("fluid") and not _names_something(c.of, m, plan):
        return _row(c, "fail", resolve(c.of, m, plan)[1], "")
    v = fn(m, c.of, args)
    if v.ok is None:
        return _row(c, "not_measurable", f"not measurable here: {v.measured}", v.expected, v.where, v.numbers)
    return _row(c, "pass" if v.ok else "fail", v.measured, v.expected, v.where, v.numbers, v.detail)


def _names_something(of: str, m: Measurements, plan: "Plan | None") -> bool:
    if of in m.patches or of in m.rows or of in m.features:
        return True
    if plan is not None and of.split("[")[0].split(".")[0] in plan.features:
        return True
    insts, _ = resolve(of, m, plan)
    return insts is not None


def _patch_row(c: Claim, m: Measurements, plan: "Plan | None" = None) -> ComplianceRow:
    p = m.patches.get(c.patch)
    if p is None:
        return _row(c, "fail", f"no patch named '{c.patch}'; patches: {', '.join(m.patches) or 'none'}",
                    _patch_expected(c, m))
    mid = _patch_midpoint(p)
    n = int(p.get("n", len(p.get("curves", []))))
    if c.side is not None:
        x0, y0, x1, y1 = m.bounds
        span = _span(m)
        edge = {"left": ("x", x0, 0), "right": ("x", x1, 0), "bottom": ("y", y0, 1), "top": ("y", y1, 1)}[c.side]
        axis, line, i = edge
        mids = p.get("midpoints") or ([] if mid is None else [mid])
        on = all(abs(float(q[i]) - line) <= 1e-6 * span + 1e-9 for q in mids) and bool(mids)
        at = fmt_point(mid) if mid else "no midpoint"
        if on:
            text = f"{c.patch} at {at}, the {c.side} side"
            return _row(c, "pass", text, c.side, mid)
        lo, hi = (x0, x1) if axis == "x" else (y0, y1)
        # the end is named along the claim's own axis (left / right for x, bottom / top for
        # y): the request said "right", so the answer is which of the two it is
        if mid is None:
            actual = "nowhere on the extent"
        else:
            low, high = ("left", "right") if axis == "x" else ("bottom", "top")
            actual = low if float(mid[i]) <= (lo + hi) / 2 else high
        # the extent's ends to three figures: the sampled bounds (3.5) miss an arc's apex
        # by a thousandth, and "-4..34" is the range a person reads, not "-3.999..34"
        text = (f"{c.patch} at {at}: the {actual.upper()} end ({axis} = {fmt_g(mid[i]) if mid else 0} of "
                f"{_clean(lo):.3g}..{_clean(hi):.3g})")
        return _row(c, "fail", text, c.side, mid, detail=_serpentine_side_detail(c, m, plan))
    at = c.at
    if isinstance(at, (list, tuple)) and len(at) == 2:
        tol = float(c.tol) if _is_number(c.tol) else m.reference_width / 2
        if mid is None:
            return _row(c, "fail", f"{c.patch} has no curves", f"{fmt_point(at)} +/- {tol:g}")
        dist = math.dist(mid, (float(at[0]), float(at[1])))
        ok = dist <= tol
        text = f"{c.patch} at {fmt_point(mid)}" + ("" if ok else f", {dist:.4g} from {fmt_point(at)}")
        return _row(c, "pass" if ok else "fail", text, f"{fmt_point(at)} +/- {tol:g}", mid, {"distance": dist})
    rule = str(at)
    ok = _rule_matches(rule, p, m)
    if ok is None:
        return _row(c, "not_measurable", f"the rule '{rule}' is not judged here (at: [x, y] or side: is)")
    text = f"{c.patch} ({n} edge{'s' if n != 1 else ''}) at {fmt_point(mid) if mid else '?'}" + ("" if ok else f" is not {rule}")
    return _row(c, "pass" if ok else "fail", text, rule, mid)


def _serpentine_side_detail(c: Claim, m: Measurements, plan: "Plan | None") -> str:
    """The explanation under T02's c9 (section 5, 7.2): the outlet of a Serpentine with an
    even number of passes is on the inlet's side of the pass axis, whatever the request
    says, and the counts that would end on the other side are named. Empty for any other
    side failure."""
    if plan is None or c.patch != "outlet" or c.side is None:
        return ""
    for name, sol in plan.features.items():
        if getattr(sol, "kind", "") != "Serpentine":
            continue
        feat = m.features.get(name, {})
        passes = feat.get("passes", sol.params.get("passes"))
        stack = str(sol.params.get("stack") or "+y")
        along_x = stack.endswith("y")
        if not _is_number(passes) or int(passes) % 2 or (c.side in ("left", "right")) != along_x:
            continue
        n = int(passes)
        return (f"an even number of passes ends on the inlet's side; {n} passes with {n - 1} bends cannot end "
                f"at the {c.side}.\n{n + 1} passes ({n} bends) or {n - 1} passes ({n - 2} bends) end on the "
                f"{c.side}; the request fixes {n} and {n - 1}.")
    return ""


def _patch_expected(c: Claim, m: Measurements) -> str:
    if c.side is not None:
        return c.side
    if isinstance(c.at, (list, tuple)) and len(c.at) == 2:
        tol = float(c.tol) if _is_number(c.tol) else m.reference_width / 2
        return f"{fmt_point(c.at)} +/- {tol:g}"
    return str(c.at)


def _side_word(mid, bounds) -> str:
    x0, y0, x1, y1 = bounds
    d = {"left": abs(mid[0] - x0), "right": abs(mid[0] - x1), "bottom": abs(mid[1] - y0), "top": abs(mid[1] - y1)}
    return min(d, key=d.get)


def _rule_matches(rule: str, p: dict, m: Measurements) -> bool | None:
    axis, _, what = rule.partition(":")
    if axis not in ("x", "y") or what not in ("min", "max"):
        return None
    x0, y0, x1, y1 = m.bounds
    i = 0 if axis == "x" else 1
    line = {("x", "min"): x0, ("x", "max"): x1, ("y", "min"): y0, ("y", "max"): y1}[(axis, what)]
    mids = p.get("midpoints") or []
    return bool(mids) and all(abs(float(q[i]) - line) <= 1e-6 * _span(m) + 1e-9 for q in mids)


# ----------------------------------------------------------------------------- can_commit

def _where(f: Finding) -> str:
    return f" at {fmt_point(f.where)}" if f.where is not None else ""


def _accepted(w: Finding, accepts) -> bool:
    for entry in accepts or ():
        code, where = entry[0], entry[1] if len(entry) > 1 else None
        if code != w.code:
            continue
        if where is None or w.where is None:
            return True
        if math.dist(where, w.where) <= 0.1:
            return True
    return False


def unaccepted_warnings(findings: list[Finding], reply) -> list[Finding]:
    """The warnings the reply did not accept by code and `where` (within 0.1 mm)."""
    accepts = getattr(reply, "accepts", None) or []
    return [w for w in _warnings(findings) if not _accepted(w, accepts)]


def can_commit(findings: list[Finding], table: ComplianceTable | None, reply: "Reply") -> tuple[bool, str]:
    """Pure. (ok, reason). No table -> refused. A table with `checkable == 0` -> refused:
    'COMMIT refused: no checkable claim was recorded; the claims lap failed twice: <parse
    error>' (D33). Lint errors -> refused naming codes and coordinates. reply.disagrees must
    equal table.failing_ids() as sets (D11). Warnings: refused when WARNINGS_BLOCK_COMMIT and
    any warning is not in reply.accepts (matched by code and `where` within 0.1 mm);
    otherwise never blocking."""
    if table is None:
        return False, ("COMMIT refused: no compliance table was recorded; no script lap has built against "
                       "the claims yet")
    if table.checkable == 0:
        reasons = [r.measured for r in table.rows if r.kind == "not_measurable" and r.measured]
        return False, ("COMMIT refused: no checkable claim was recorded; the claims lap failed twice"
                       + (f" or every claim is report / not_measurable ({'; '.join(reasons)})" if reasons
                          else " (no measure, range, count, predicate or patch claim in the table)"))
    errs = _errors(findings)
    if errs:
        named = ", ".join(f"{f.code}{_where(f)}" for f in errs)
        return False, (f"COMMIT refused: lint errors block: {named}; a lint error can never be disagreed with "
                       "-- fix the shape")
    named_ids = getattr(reply, "disagrees", None) or []
    if isinstance(named_ids, str):
        named_ids = named_ids.replace(";", ",").split(",")
    disagrees = [d.strip() for d in named_ids if d and d.strip()]
    named = set(disagrees)
    failing = set(table.failing_ids())
    known = {r.id for r in table.rows}
    for cid in sorted(named - failing):
        if cid not in known:
            return False, f"COMMIT refused: nothing to disagree with on {cid}; it is not a claim in the table"
        return False, f"COMMIT refused: nothing to disagree with on {cid}; it passes"
    missing = sorted(failing - named, key=_id_key)
    if missing:
        every = ", ".join(sorted(failing, key=_id_key))
        return False, (f"COMMIT refused: {', '.join(missing)} FAIL{'S' if len(missing) == 1 else ''} and "
                       f"{'is' if len(missing) == 1 else 'are'} not named; fix it, or reply "
                       f"`COMMIT disagrees: {every}` naming every failing claim")
    open_warnings = unaccepted_warnings(findings, reply)
    if WARNINGS_BLOCK_COMMIT and open_warnings:
        named_w = ", ".join(f"{w.code}{_where(w)}" for w in open_warnings)
        return False, (f"COMMIT refused: warnings not accepted: {named_w}; accept each by name "
                       "(COMMIT accepts: W-CODE at (x, y) -- why) or fix it")
    reason = "COMMIT accepted"
    if disagrees:
        reason += f", disagreeing with {', '.join(sorted(named, key=_id_key))}"
    if open_warnings:
        reason += f", with {len(open_warnings)} unaccepted warning{'s' if len(open_warnings) != 1 else ''}"
    return True, reason


def _id_key(cid: str):
    digits = "".join(ch for ch in cid if ch.isdigit())
    return (int(digits) if digits else 0, cid)
