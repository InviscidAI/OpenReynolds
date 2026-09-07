"""The judge: predicates over the walk and the measurements, each a Finding with a
stable code from DESIGN.md section 5.

Every threshold is a multiple of the reference width or the span (never an absolute
length), every message is rendered from `TEXTS` so it names the feature, the numbers and
what changes it, and every instance reference is `<row>[k]`, 0-based (D29). `from_legacy`
turns mesh2d's check dicts into Findings so the toolbox's own text survives only inside
`mesh2d.check_lines`. The landing predicate runs only on legs recorded as landing on a
wall (D22/D28); the notch predicate marches OUTWARD from a short wall edge and asks how
much solid stands before the fluid on its far side (D9); cusps are judged on the fluid's
angle and lips reported on the solid's (D31); parts declared apart are checked by
intersect area and distance on their standalone outlines (D35).
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

from . import _toolbox
from .measure import (  # noqa: F401
    SCALE_BY_UNIT, Measurements, Walk, _dist, _face_of, _feature_of, _shift, body_open_ends, holes_of,
    host_wall_span, instances_of, open_ends, point_line_distance, ray_depth, resolve_ports,
)

if TYPE_CHECKING:
    from .claims import ClaimSet
    from .compile import Plan

Level = Literal["error", "warn", "info"]

_LEAD = {"error": "!! ERROR ", "warn": "!!       ", "info": "check    "}


@dataclass
class Finding:
    level: Level
    code: str
    """Section 5's codes: E-OVERLAP, W-SHORT-EDGE, I-GAP, ..."""
    subject: str
    """"Row 'loops'", "Bypass 'loop[2]'", "inlet", "fluid"."""
    what: str
    """One sentence with the number, the threshold and the coordinates."""
    fix: str = ""
    """What changes it (may be empty for info)."""
    where: tuple[float, float] | None = None
    """Sketch units; drawn as a red (error) / orange (warn) cross."""
    numbers: dict[str, float] = field(default_factory=dict)
    draw: list[dict] = field(default_factory=list)
    """{"segment": [(x,y),(x,y)]} | {"circle": (cx, cy, r)} | {"ray": ...}."""

    def text(self) -> str:
        """"!! ERROR  CODE  subject: what\\n          fix" (section 5 format); warnings lead
        with `!!        W-CODE`, info with `check     I-CODE`."""
        head = f"{_LEAD[self.level]} {self.code}  {self.subject}: {self.what}"
        return head if not self.fix else f"{head}\n          {self.fix}"

    def as_legacy(self) -> dict:
        """{"level", "where", "what"} for today's regexes."""
        return {"level": self.level, "where": self.subject, "what": self.what}

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Finding":
        where = d.get("where")
        return Finding(level=d["level"], code=d["code"], subject=d["subject"], what=d["what"],
                       fix=d.get("fix", ""), where=None if where is None else (float(where[0]), float(where[1])),
                       numbers=dict(d.get("numbers", {})), draw=list(d.get("draw", [])))


TEXTS: dict[str, str] = {
    "E-OVERLAP": (
        "{a} and {b} overlap by {area:.1f} {units}2 ({pct:.0f} % of an instance's {inst:.0f} {units}2) at "
        "({x:.1f}, {y:.1f}); the pitch {pitch:.4g} is smaller than the footprint {footprint:.4g} plus a gap\n"
        "(a pitch was given explicitly -- remove it, or raise it to >= {footprint:.4g} + gap)"),
    "E-OVERLAP:between": (
        "overlap by {area:.1f} {units}2 at ({x:.1f}, {y:.1f})\n"
        "(parts that are not each other's host are apart by default; s.apart(a, b, gap=...) states the gap)"),
    "E-OVERLAP:legacy": "{what}\n(a pitch was given explicitly -- remove it, or raise it to the footprint plus a gap)",
    "E-TOUCH": (
        "{a} and {b} touch (walls {gap:.3f} apart at ({x:.1f}, {y:.1f}))\n"
        "two walls meeting at a knife edge mesh into slivers; gap >= {floor:.3g} (a tenth of the width) is the floor"),
    "E-TOUCH:between": (
        "touch (walls {gap:.3f} apart at ({x:.1f}, {y:.1f}))\n"
        "two walls meeting at a knife edge mesh into slivers; gap >= {floor:.3g} (a tenth of the width) is the floor"),
    "E-TOUCH:legacy": "{what}\ntwo walls meeting at a knife edge mesh into slivers; a tenth of the width is the floor",
    "E-ROW-GAP": (
        "{a} and {b} are {gap:.2f} apart where the row declares gap {declared:g}\n"
        "(the row was built with pitch {pitch:.4g}; pitch >= footprint {footprint:.4g} + {declared:g} = {need:.4g})"),
    "E-APART-GAP": "are {gap:.2g} apart where s.apart declares gap {declared:g}",
    "E-LAND": (
        "the leg to {line} lands at ({lx:.4g}, {ly:.4g}); {k} of {n} points across its mouth, a tenth of a width "
        "past the wall line, are outside the fluid ({span_text})\n"
        "it runs off the part it was meant to rejoin. Row start >= {start:.4g}, or align=\"start\""),
    "E-LAND:legacy": (
        "the leg to {line} lands at ({lx:.4g}, {ly:.4g}) and nothing of the fluid is there (1 of 1 point a tenth "
        "of a width past the line is outside)\nit runs off the part it was meant to join"),
    "E-LAND:short": (
        "the leg to {line} stops {short:.4g} short of {host} ({host_line}): {k} of {n} points a tenth of a width "
        "past {line} are outside the fluid (a gap of {short:.4g} between the cap and the wall)\n"
        "end_on({host}) runs to the wall's own line; a to-line is not a wall"),
    "E-LAND:capped": (
        "the leg to {line} reaches {host} but its mouth is closed: {k} of {n} points a tenth of a width on the "
        "leg's side of {line} are outside the fluid (a cap of {width:.4g} at ({lx:.4g}, {ly:.4g}))\n"
        "the leg was cut where it started, not where it lands; end_on({host}) cuts it flush"),
    "E-LAND:span": (
        "the leg to {line} lands at u = {u:.4g}, outside {host}'s span {u0:g}..{u1:g}\n"
        "it runs off the part it was meant to rejoin. Row start >= {start:.4g}, or align=\"start\""),
    "E-NOTCH": (
        "wall edge of {edge:.3g} at ({x:.4g}, {y:.4g}) has {solid:.2f} of solid behind it before the fluid on "
        "its far side (a third of the {w:g} passage, {floor:.3g}, is the floor): a cap or a corner is poking "
        "through a wall; the cells there will be slivers\n"
        "a Passage that leaves a wall starts on it, start=(main.top, 30); a Bypass does this itself"),
    "E-SHORT-EDGE": (
        "edge of {edge:.2g} at ({x:.4g}, {y:.4g}): shorter than a thirtieth of the {w:g} passage ({floor:.2g})\n"
        "a sliver no cell can sit on"),
    "W-SHORT-EDGE": (
        "edge of {edge:.2g} at ({x:.4g}, {y:.4g}): shorter than {floor:.3g} (a third of the narrowest passage "
        "{w:g}); {n} such edges\n"
        "a corner poking through a wall, or a leg shorter than the channel is wide; the cells there will be poor"),
    "W-GAP": (
        "{a} and {b} are {gap:.2g} apart, a solid wall thinner than a tenth of the {w:g} passage\n"
        "if the wall is meant, say so at COMMIT (accepts: W-GAP at (x, y) -- why)"),
    "I-GAP": "neighbours' walls {gap:.2f} apart{solved_text}",
    "I-GAP:legacy": "{what}",
    "W-PASSAGE": (
        "the narrowest passage measured {min:.3f} (+/- {res:.3f}) at ({x:.4g}, {y:.4g}), less than half the "
        "{w:g} {source}\n"
        "if a throat is meant, declare its width (Passage(width={min:.3g})) so the mesh is sized for it"),
    "W-CUSP": "a {deg:.3g} deg wedge of fluid at ({x:.4g}, {y:.4g}): two walls nearly tangent; the mesher will make slivers there",
    "I-CUSP": (
        "a {deg:.3g} deg wedge of fluid at ({x:.4g}, {y:.4g}): two walls closing on each other (information; "
        "W-CUSP below {warn:g} deg)"),
    "I-LIP": (
        "a knife-edge lip of {deg:.1f} deg (solid) at ({x:.4g}, {y:.4g}), {n} such lips: an arc or a leg's wall "
        "meets its host wall at a knife edge (information; the mesh follows the lip)"),
    "I-OVERLAP-NOISE": (
        "{a} and {b} intersect by {area:.2g} {units}2, below {rel:g} of an instance (OCC fuzz on a tangent pair)"),
    "I-REFERENCE": "passage width {w:g} ({source}); p10 of the passage samples {p10:.3f} at ({x:.4g}, {y:.4g})",
    "W-UNITS": (
        "extent {span:.4g} against a largest stated length of {L:.4g} ({r:.3g}x): the built shape is much "
        "{sense} than the request's numbers"),
    "E-UNITS": (
        "the fluid spans {span:.4g} in a sketch in {units}; the claims' largest length is {L:.4g} {cunit} = "
        "{Lconv:.4g} {units} (ratio {r:.3g})\n"
        "Sketch(units=\"{cunit}\"), or the claims are in {units} -- one of them is wrong by {factor:g}x"),
    "E-PORT-MISSING": (
        "the end of {edge} at {at} is not an open end of the fluid\n"
        "{why}; an inlet must be a free edge of the outline (a body in a channel: the inlet is the channel's "
        "own end, duct.left); {host} has {n_ends} open ends"),
    "E-PORT-COUNT": "{n} {want} edges: at {ends}; a passage has exactly {need}\n{fix}",
    "E-PORT-OPEN": (
        "an open end of {length:.3g} at ({x:.4g}, {y:.4g}) facing {facing} that no port names; it would mesh "
        "as a wall\ns.outlet(<feature>.end), or s.wall(<feature>.end, name=\"cap\") if it is meant to be closed"),
    "E-PORT-RULE": "\"{rule}\" names {n} open ends, at {ends}; a rule names exactly one\n{fix}",
    "E-PORT-ON-BODY": (
        "the body has an open end of {length:.3g} at ({x:.4g}, {y:.4g}) facing {facing}\n"
        "the body was drawn as a passage; a body in a flow is a closed outline"),
    "E-DISJOINT": "fluid is {n} separate faces: {pieces} do not touch\n{why}",
    "E-VOID": (
        "the body has an enclosed hole of {area:.4g} {units}2 at ({x:.4g}, {y:.4g})\n"
        "the fluid was drawn instead of the solid; the tool cuts the flow box itself"),
    "E-MIRROR-BAKE": (
        "the fluid's last op is `{op}` ({n} ops in); the classifier is wrong on a {adjective} face\n"
        "fuse it with something after the {op}, or {op} the primitives' coordinates (the API does this itself)"),
    "E-KERNEL-STATE": (
        "the classifier answers inside for a point {d:.4g} outside the fluid; the face carries a transform the "
        "kernel cannot trust (an internal error: nothing in the script to fix; reported to the desk, rc 3)"),
}
"""code -> the message template of section 5 (every Finding's text is rendered from it;
invariant 1's readable surface scans this dict, and a test pins that every template names
a number placeholder and never "looks"/"seems"/"probably"). A key with a `:variant`
suffix is one code's other wording; the Finding's `code` is the part before the colon."""


def render(level: Level, code: str, subject: str, where: tuple[float, float] | None = None,
           draw: list[dict] | None = None, variant: str | None = None, **fields) -> Finding:
    """A Finding whose `what` and `fix` are the template of `TEXTS` filled with `fields`;
    every numeric field lands in `numbers` so a test reads the value, not the text."""
    key = f"{code}:{variant}" if variant else code
    template = TEXTS[key]
    what, _, fix = template.partition("\n")
    numbers = {k: float(v) for k, v in fields.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return Finding(level=level, code=code, subject=subject, what=what.format(**fields), fix=fix.format(**fields),
                   where=where, numbers=numbers, draw=list(draw or []))


@dataclass
class LintConfig:
    overlap_area_rel: float = 1e-6
    """Of the smaller instance's area (OCC fuzzy ~1e-7 of size; 3 orders above noise, 4
    below any real overlap); applied in from_legacy and overlap_between."""
    touch_rel: float = 1e-6
    """Of span (60 nm on a 60 mm valve)."""
    thin_wall_rel: float = 0.1
    """Of w_min: undeclared gap below this warns (the accepted T01 sits at 0.8 on 3)."""
    landing_probe_rel: float = 0.1
    """Of w_min, into the parent."""
    landing_samples: int = 5
    notch_edge_rel: float = 1.0
    """Of w_min: candidate edge length (the square cap is exactly w/2)."""
    notch_solid_rel: float = 1.0 / 3
    """Of w_min: SOLID behind the edge along the outward normal before the ray re-enters the fluid (D9)."""
    short_warn_rel: float = 1.0 / 3
    """Phase 0's rule, kept."""
    short_error_rel: float = 1.0 / 30
    """Below the finest cell Phase 2 makes (w/20, w/30 with layers)."""
    cusp_info_deg: float = 30.0
    """FLUID angle below this: I-CUSP; SOLID angle below this: I-LIP (D31)."""
    cusp_warn_deg: float = 5.0
    """FLUID angle below this: W-CUSP."""
    passage_warn_rel: float = 0.5
    open_end_angle_tol_deg: float = 30.0
    open_end_depth_rel: float = 1.0
    units_warn: tuple[float, float] = (0.5, 3.0)
    units_error: tuple[float, float] = (1 / 8, 8.0)

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "LintConfig":
        cfg = LintConfig(**{k: v for k, v in d.items() if k not in ("units_warn", "units_error")})
        if "units_warn" in d:
            cfg.units_warn = (d["units_warn"][0], d["units_warn"][1])
        if "units_error" in d:
            cfg.units_error = (d["units_error"][0], d["units_error"][1])
        return cfg


_OVERLAP = re.compile(r"repeat '([^']+)': copies (\d+) and (\d+) overlap by ([-+0-9.eE]+)")
_TOUCH = re.compile(r"repeat '([^']+)': copies (\d+) and (\d+) touch")
_GAP = re.compile(r"repeat '([^']+)': gap between copies ([-+0-9.eE]+)")
_LANDS = re.compile(r"channel '([^']+)': the leg to (\S+) lands at \(([-+0-9.eE]+), ([-+0-9.eE]+)\)")


def from_legacy(checks: list[dict], row: str, cfg: LintConfig, units: str = "mm",
                w: float | None = None) -> list[Finding]:
    """mesh2d.py's check dicts (from build_face's repeat branch: overlap, touch, the gap
    info) -> Findings with the codes E-OVERLAP, E-TOUCH, I-GAP. The numeric keys `pair`,
    `gap`, `overlap`, `instance_area` (section 3.17 edit 2) are read; the subject and the
    text are rewritten to `<row>[k]` 0-based from `pair` (D29: mesh2d's "copies 1 and 2"
    becomes `loops[0]` and `loops[1]`); an overlap below `cfg.overlap_area_rel *
    instance_area` is demoted to I-OVERLAP-NOISE (OCC fuzz on a tangent pair; measured
    empty on tangent disks, so the branch is a guard, not a path the fixtures take).
    Checks from an rc-2 build with a missing `pair` (a spec older than edit 2) keep the
    legacy text verbatim. A dict that already carries a `code` (compile.build's
    E-DISJOINT) passes through; mesh2d's landing dict becomes E-LAND in its legacy wording,
    and `judge` drops it for a leg the package judged itself."""
    out: list[Finding] = []
    for check in checks:
        what = str(check.get("what", ""))
        where = check.get("where")
        where = None if where is None else (float(where[0]), float(where[1]))
        level = check.get("level", "error")
        if check.get("code"):
            out.append(Finding(level=level, code=check["code"], subject=check.get("subject", "fluid"), what=what,
                               fix=check.get("fix", ""), where=where, numbers=dict(check.get("numbers", {}))))
            continue
        m = _OVERLAP.search(what)
        if m:
            name = m.group(1) if row is None else row
            pair = check.get("pair")
            area = float(check.get("overlap", m.group(4)))
            inst = check.get("instance_area")
            if pair is None:
                out.append(render("error", "E-OVERLAP", f"Row '{name}'", where=where, variant="legacy", what=what,
                                  area=area))
                continue
            a, b = f"{name}[{int(pair[0]) - 1}]", f"{name}[{int(pair[1]) - 1}]"
            fp = check.get("footprint") or (0.0, 0.0)
            pitch = float(check.get("pitch") or 0.0)
            if inst and area <= cfg.overlap_area_rel * float(inst):
                out.append(render("info", "I-OVERLAP-NOISE", f"Row '{name}'", where=where, a=a, b=b, area=area,
                                  units=units, rel=cfg.overlap_area_rel))
                continue
            out.append(render("error", "E-OVERLAP", f"Row '{name}'", where=where, a=a, b=b, area=area, units=units,
                              pct=100.0 * area / float(inst) if inst else 0.0, inst=float(inst or 0.0),
                              x=where[0] if where else 0.0, y=where[1] if where else 0.0, pitch=pitch,
                              footprint=float(fp[0]), instance_area=float(inst or 0.0),
                              pair_a=int(pair[0]) - 1, pair_b=int(pair[1]) - 1))
            continue
        m = _TOUCH.search(what)
        if m:
            name = m.group(1) if row is None else row
            pair = check.get("pair")
            if pair is None:
                out.append(render("error", "E-TOUCH", f"Row '{name}'", where=where, variant="legacy", what=what, gap=0.0))
                continue
            a, b = f"{name}[{int(pair[0]) - 1}]", f"{name}[{int(pair[1]) - 1}]"
            out.append(render("error", "E-TOUCH", f"Row '{name}'", where=where, a=a, b=b,
                              gap=float(check.get("gap") or 0.0), x=where[0] if where else 0.0,
                              y=where[1] if where else 0.0, floor=cfg.thin_wall_rel * float(w or 0.0)))
            continue
        m = _GAP.search(what)
        if m:
            name = m.group(1) if row is None else row
            gap = float(check.get("gap", m.group(2)))
            pair = check.get("pair")
            if pair is None:
                out.append(render("info", "I-GAP", f"Row '{name}'", where=where, variant="legacy", what=what, gap=gap))
                continue
            fp = check.get("footprint") or (0.0, 0.0)
            out.append(render("info", "I-GAP", f"Row '{name}'", where=where, gap=gap, solved_text="",
                              pair_a=int(pair[0]) - 1, pair_b=int(pair[1]) - 1,
                              pitch=float(check.get("pitch") or 0.0), footprint_u=float(fp[0])))
            continue
        m = _LANDS.search(what)
        if m:
            out.append(render("error", "E-LAND", f"channel '{m.group(1)}'", where=where, variant="legacy",
                              line=m.group(2), lx=float(m.group(3)), ly=float(m.group(4)), channel=m.group(1)))
            continue
        out.append(Finding(level=level, code={"error": "E-CHECK", "warn": "W-CHECK", "info": "I-CHECK"}[level],
                           subject="fluid", what=what, where=where))
    return out


def kernel_findings(ops: list[dict]) -> list[Finding]:
    """`cli check --spec`: the fluid's last op is `mirror` (the grammar's one transform the
    classifier is blind to; `dilate`/`affine` have no op) with no fuse after it (D30) ->
    E-MIRROR-BAKE. A `mirror` with `keep` fuses the copy and is clean (measured)."""
    if not ops:
        return []
    body = next((op for op in ops if op.get("name") == "body"), ops[-1])
    if body.get("op") == "mirror" and not body.get("keep"):
        return [render("error", "E-MIRROR-BAKE", "spec", op="mirror", adjective="mirrored", n=len(ops))]
    return []


def kernel_state(gmsh, face: int, wk: Walk) -> Finding | None:
    """The canary of D26: a point one span past the sampled bounds along +x must be
    outside, and the walk must not have needed a repair; else E-KERNEL-STATE."""
    x0, y0, x1, y1 = wk.bounds()
    span = max(x1 - x0, y1 - y0)
    probe = (x1 + span, (y0 + y1) / 2)
    if wk.repaired or gmsh.model.isInside(2, face, [probe[0], probe[1], 0.0]):
        return render("error", "E-KERNEL-STATE", "fluid", where=probe, d=span)
    return None


def void_findings(gmsh, body_face: int, name: str, units: str) -> list[Finding]:
    """E-VOID on a BodyInBox body face that carries an inner loop (the fluid was drawn
    instead of the solid): run on the body before the flow box is cut, since the box minus
    a hollow body is two faces and never reaches the walk."""
    out = []
    wk = walk_or_none(gmsh, body_face)
    if wk is None:
        return out
    for h in holes_of(wk):
        out.append(render("error", "E-VOID", f"BodyInBox '{name}'", where=h.centroid, area=h.area, units=units,
                          x=h.centroid[0], y=h.centroid[1]))
    return out


def judge_body(gmsh, body_face: int, name: str, units: str, w_min: float,
               cfg: LintConfig = LintConfig()) -> list[Finding]:
    """The two predicates that are asked of a BodyInBox BODY before the flow box is cut:
    E-VOID (an enclosed hole: the fluid was drawn instead of the solid) and
    E-PORT-ON-BODY (an open end of a passage drawn into the solid). Whoever builds the
    body (compile.build for a BodyInBox, the cli's --external path) calls this on the
    body's face; `judge` repeats the open-end reading on the fluid's hole loops when the
    plan carries a BodyInBox whose `solved` has no `body_face`."""
    out = void_findings(gmsh, body_face, name, units)
    wk = walk_or_none(gmsh, body_face)
    if wk is not None:
        for e in body_open_ends(gmsh, wk, w_min, cfg.open_end_angle_tol_deg, cfg.open_end_depth_rel):
            out.append(render("error", "E-PORT-ON-BODY", f"BodyInBox '{name}'", where=e.centre, length=e.length,
                              x=e.centre[0], y=e.centre[1], facing=_facing(e.outward_normal)))
    return out


def walk_or_none(gmsh, face: int) -> Walk | None:
    from .measure import walk
    try:
        return walk(gmsh, face, 1.0)
    except Exception:  # noqa: BLE001 - a face with no loops is no walk
        return None


_LEVEL_ORDER = {"error": 0, "warn": 1, "info": 2}


def judge(gmsh, face: int, plan: "Plan", wk: Walk, m: Measurements, legacy_checks: list[dict],
          claims: "ClaimSet | None", cfg: LintConfig = LintConfig()) -> list[Finding]:
    """Every predicate of DESIGN.md 3.6's table, in this order, errors first in the
    result. Returns after E-DISJOINT (no walk of a multi-face fluid)."""
    units = m.units if m is not None else "mm"
    legacy = _legacy_by_row(legacy_checks, cfg, units, m.reference_width if m is not None else None)
    if wk is None or any(f.code == "E-DISJOINT" for f in legacy):
        return _ordered(legacy)
    w = m.reference_width
    span = max(m.extent)
    external = any(s.kind == "BodyInBox" for s in (getattr(plan, "features", None) or {}).values())
    findings: list[Finding] = []
    findings += _void_and_body(gmsh, plan, wk, m, cfg, units)
    findings += _drop_judged_landings(legacy, plan, m)
    findings += _row_gaps(plan, m, legacy, cfg, w)
    findings += _apart(gmsh, plan, cfg, units, span, w)
    findings += _landings(gmsh, face, plan, wk, m, cfg, w, span)
    notched = set()
    for f in _notches(gmsh, face, wk, cfg, w):
        findings.append(f)
        notched.add(int(f.numbers.get("curve", -1)))
    findings += _short_edges(wk, cfg, w, notched)
    findings += _cusps_and_lips(wk, cfg)
    findings += _passage(m, cfg, w)
    findings += _ports(gmsh, wk, plan, m, cfg, w)
    findings += _units(m, claims, cfg, _body_span(wk) if external else span)
    findings += kernel_findings(list(getattr(plan, "ops", []) or []))
    canary = kernel_state(gmsh, face, wk)
    if canary is not None:
        findings.append(canary)
    if m.passage is not None and m.passage.n:
        findings.append(render("info", "I-REFERENCE", "fluid", w=w, source=m.reference_width_from,
                               p10=m.passage.p10, x=m.passage.where_min[0], y=m.passage.where_min[1]))
    return _ordered(findings)


def _ordered(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: _LEVEL_ORDER.get(f.level, 3))


def _body_span(wk: Walk) -> float:
    """The span the units check judges on a BodyInBox: the body's, not the flow box's
    (3.6, the units row). The body is the fluid's hole loops; a 20 mm body in a box five
    lengths long is judged against the request's 20, not the box's 70. With no hole
    (the body reached the box's edge) the fluid's own span stands."""
    pts = [p for k in range(1, len(wk.loops)) for c in wk.loops[k] for p in wk.curves[c].samples]
    if not pts:
        return wk.span()
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def has_errors(findings: list[Finding]) -> bool:
    return any(f.level == "error" for f in findings)


def errors(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.level == "error"]


def warnings(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.level == "warn"]


# -- the predicates ---------------------------------------------------------------------


def _legacy_by_row(checks: list[dict], cfg: LintConfig, units: str, w: float | None) -> list[Finding]:
    out: list[Finding] = []
    for check in checks or []:
        m = re.search(r"repeat '([^']+)'", str(check.get("what", "")))
        out += from_legacy([check], m.group(1) if m else None, cfg, units, w)
    return out


def _drop_judged_landings(legacy: list[Finding], plan, m: Measurements) -> list[Finding]:
    """mesh2d's landing dicts are kept only for channels the package does not judge
    itself (no `lands_on` feature); the package's E-LAND carries the landing point."""
    judged = {name for name, _ in _landing_channels(plan, m)}
    out = []
    for f in legacy:
        if f.code == "E-LAND" and f.subject.startswith("channel '"):
            channel = f.subject[len("channel '"):-1]
            if channel in judged:
                continue
        out.append(f)
    return out


def _void_and_body(gmsh, plan, wk: Walk, m: Measurements, cfg: LintConfig, units: str) -> list[Finding]:
    out: list[Finding] = []
    for name, s in (getattr(plan, "features", None) or {}).items():
        if s.kind != "BodyInBox":
            continue
        body_face = s.solved.get("body_face")
        if body_face is not None:
            if walk_or_none(gmsh, int(body_face)) is None:
                # compile.build keeps the body's face alive through the box cut; a tag
                # that no longer walks is an internal error, never a clean body
                out.append(Finding(level="error", code="E-KERNEL-STATE", subject=f"BodyInBox '{name}'",
                                   what=f"the body face {int(body_face)} recorded for the void check is not in the "
                                        "model, so E-VOID and E-PORT-ON-BODY were not judged",
                                   fix="(an internal error: nothing in the script to fix; reported to the desk)"))
                continue
            out += judge_body(gmsh, int(body_face), name, units, m.reference_width, cfg)
            continue
        # On the fluid the body is a hole loop, and a passage drawn into it reads as an
        # open end there: convex mouth corners, deep behind (a rect body's sides end at
        # the fluid's 270-degree corners and never qualify).
        for e in m.open_ends:
            if wk.curves[e.curve].loop > 0:
                out.append(render("error", "E-PORT-ON-BODY", f"BodyInBox '{name}'", where=e.centre,
                                  length=e.length, x=e.centre[0], y=e.centre[1], facing=_facing(e.outward_normal)))
    return out


def _facing(normal) -> str:
    nx, ny = normal
    if abs(nx) >= abs(ny):
        return "+x" if nx > 0 else "-x"
    return "+y" if ny > 0 else "-y"


def _row_gaps(plan, m: Measurements, legacy: list[Finding], cfg: LintConfig, w: float) -> list[Finding]:
    """Declared gap: measured wall distance < 0.99 x gap is E-ROW-GAP; undeclared: below
    thin_wall_rel x w is W-GAP; else the I-GAP that from_legacy already carries."""
    out: list[Finding] = []
    for name, inst in (getattr(plan, "instances", None) or {}).items():
        gaps = [f for f in legacy if f.code == "I-GAP" and f.subject == f"Row '{name}'"]
        if not gaps:
            continue
        measured = min(f.numbers.get("gap", math.inf) for f in gaps)
        first = gaps[0]
        a = f"{name}[{int(first.numbers.get('pair_a', 0))}]"
        b = f"{name}[{int(first.numbers.get('pair_b', 1))}]"
        declared = inst.get("gap")
        step = inst.get("step") or (0.0, 0.0)
        pitch = float(inst.get("pitch") or math.hypot(float(step[0]), float(step[1])) or first.numbers.get("pitch", 0.0))
        footprint = float((inst.get("footprint") or (0.0, 0.0))[0]) or float(first.numbers.get("footprint_u", 0.0))
        if inst.get("gap_declared") and declared is not None and measured < 0.99 * float(declared):
            out.append(render("error", "E-ROW-GAP", f"Row '{name}'", where=first.where, a=a, b=b, gap=measured,
                              declared=float(declared), pitch=pitch, footprint=footprint,
                              need=footprint + float(declared)))
        elif not inst.get("gap_declared") and measured < cfg.thin_wall_rel * w:
            out.append(render("warn", "W-GAP", f"Row '{name}'", where=first.where, a=a, b=b, gap=measured, w=w))
        if declared is not None and not inst.get("gap_declared"):
            first.what = TEXTS["I-GAP"].format(gap=measured, solved_text=f" (solved gap {float(declared):.2f} is along the wall's bounding footprint)")
        # The build measured the wall distance with OCC (`_instances_apart`, edit 2) and
        # `measure` never sees the check dicts, so the row's `gap_measured` is written
        # here, where both are in hand; the claims (c11's "walls 2.44 apart") read it.
        if math.isfinite(measured):
            row = m.rows.get(name)
            if row is not None and row.gap_measured is None:
                row.gap_measured = measured
            entry = m.features.get(name)
            if isinstance(entry, dict) and entry.get("gap_measured") is None:
                entry["gap_measured"] = measured
    return out


def _part_faces(occ, ops: dict[str, dict], name: str, made: dict[str, list[int]]) -> list[int]:
    """One part of the fluid rebuilt alone from its ops in a scratch model, through the
    same OCC primitives `mesh2d.build_face` uses (by path), except that a repeat's copies
    stay separate faces: a Row apart from another part is its instances, and a fuse of
    copies that do not touch is not a face. Exact OCC faces, so the 1e-6 area threshold
    and `getDistance` mean what they say (a 64-point polygon of an arc is 0.05 mm2 off)."""
    if name in made:
        return made[name]
    m2d = _toolbox.load("mesh2d")
    op = ops[name]
    kind = op["op"]

    def tags(n):
        return [(2, t) for t in _part_faces(occ, ops, n, made)]

    if kind == "rect":
        faces = [occ.addRectangle(op["origin"][0], op["origin"][1], 0, op["size"][0], op["size"][1],
                                  roundedRadius=op["round"])]
    elif kind == "disk":
        faces = [occ.addDisk(op["center"][0], op["center"][1], 0, op["radius"], op["ry"])]
    elif kind == "annulus":
        faces = m2d._band(occ, op["center"], op["r_inner"], op["r_outer"], 0.0, 360.0)
    elif kind == "band":
        faces = m2d._band(occ, op["center"], op["r_inner"], op["r_outer"], op["start"], op["end"])
    elif kind == "polygon":
        faces = [m2d._polygon_face(occ, m2d.case_gen.as_ccw(op["points"]))]
    elif kind == "outline":
        from pathlib import Path
        faces = [m2d._polygon_face(occ, m2d.outline_points(Path(op["file"]), op["size"], op["aoa"]))]
    elif kind == "channel":
        faces = m2d._channel(occ, op["width"], op["start"], op["heading"], op["path"], None, from_line=op.get("from"))
    elif kind in ("fuse", "intersect"):
        first, rest = op["of"][0], op["of"][1:]
        joined = occ.fuse if kind == "fuse" else occ.intersect
        out, _ = joined(tags(first), [t for n in rest for t in tags(n)])
        faces = [t for d, t in out if d == 2]
    elif kind == "cut":
        out, _ = occ.cut(tags(op["from"]), [t for n in op["take"] for t in tags(n)])
        faces = [t for d, t in out if d == 2]
    elif kind == "translate":
        occ.translate(tags(op["target"]), op["by"][0], op["by"][1], 0)
        faces = made[op["target"]]
    elif kind == "rotate":
        about = op["about"] or (0.0, 0.0)
        occ.rotate(tags(op["target"]), about[0], about[1], 0, 0, 0, 1, math.radians(op["angle"]))
        faces = made[op["target"]]
    elif kind == "mirror":
        a, b, d = (1.0, 0.0, -op["at"]) if op["axis"] == "x" else (0.0, 1.0, -op["at"])
        target = tags(op["target"])
        if op["keep"]:
            copy = occ.copy(target)
            occ.mirror(copy, a, b, 0, d)
            faces = made[op["target"]] + [t for _, t in copy]
        else:
            occ.mirror(target, a, b, 0, d)
            faces = made[op["target"]]
    elif kind == "copy":
        faces = [t for _, t in occ.copy(tags(op["target"]))]
    elif kind == "repeat":
        base = tags(op["target"])
        about = op["about"] or (0.0, 0.0)
        faces = [t for _, t in base]
        for k in range(1, op["count"]):
            c = occ.copy(base)
            if op["step"] != (0.0, 0.0):
                occ.translate(c, k * op["step"][0], k * op["step"][1], 0)
            if op["angle"]:
                occ.rotate(c, about[0], about[1], 0, 0, 0, 1, math.radians(k * op["angle"]))
            faces += [t for _, t in c]
    else:
        faces = []
    made[name] = faces
    return faces


def _apart(gmsh, plan, cfg: LintConfig, units: str, span: float, w: float) -> list[Finding]:
    """D35: every pair in `plan.apart`, each part rebuilt standalone from its OPS in a
    scratch model (`_part_faces`): intersect area (E-OVERLAP), distance (E-TOUCH), a
    declared gap (E-APART-GAP). A feature's part is the last op it owns (`Solved.ops`),
    else the op of its name."""
    pairs = list(getattr(plan, "apart", []) or [])
    raw_ops = list(getattr(plan, "ops", []) or [])
    if not pairs or not raw_ops:
        return []
    m2d = _toolbox.load("mesh2d")
    try:
        parsed, _ = m2d.parse_spec({"ops": raw_ops})
    except SystemExit:
        return []
    ops = {op["name"]: op for op in parsed}
    solved = getattr(plan, "features", None) or {}

    def op_of(name: str) -> str | None:
        s = solved.get(name)
        if s is not None and s.ops and s.ops[-1] in ops:
            return s.ops[-1]
        return name if name in ops else None

    out: list[Finding] = []
    current = gmsh.model.getCurrent()
    gmsh.model.add("openreynolds_apart")
    try:
        occ = gmsh.model.occ
        made: dict[str, list[int]] = {}

        def faces_of(name: str) -> list[int]:
            op_name = op_of(name)
            if op_name is None:
                return []
            faces = _part_faces(occ, ops, op_name, {})
            occ.synchronize()
            return faces

        for a, b, gap in pairs:
            fa, fb = faces_of(a), faces_of(b)
            if not fa or not fb:
                continue
            area_a = sum(occ.getMass(2, t) for t in fa)
            area_b = sum(occ.getMass(2, t) for t in fb)
            ca = occ.copy([(2, t) for t in fa])
            cb = occ.copy([(2, t) for t in fb])
            common, _ = occ.intersect(ca, cb, removeObject=True, removeTool=True)
            occ.synchronize()
            area = sum(occ.getMass(2, t) for d, t in common if d == 2)
            centre = None
            if common:
                weighted = [(occ.getMass(2, t), occ.getCenterOfMass(2, t)) for d, t in common if d == 2]
                if area > 0:
                    centre = (sum(m_ * c[0] for m_, c in weighted) / area, sum(m_ * c[1] for m_, c in weighted) / area)
                occ.remove(common, recursive=True)
                occ.synchronize()
            label_a, label_b = _apart_label(plan, a), _apart_label(plan, b)
            if area > cfg.overlap_area_rel * min(area_a, area_b) and centre is not None:
                out.append(render("error", "E-OVERLAP", f"{label_a} and {label_b}", where=centre, variant="between",
                                  a=label_a, b=label_b, area=area, units=units, x=centre[0], y=centre[1]))
                continue
            dist = math.inf
            nearest = None
            for ta in fa:
                for tb in fb:
                    d, x1, y1, _, x2, y2, _ = occ.getDistance(2, ta, 2, tb)
                    if d < dist:
                        dist, nearest = d, ((x1 + x2) / 2, (y1 + y2) / 2)
            if dist <= cfg.touch_rel * span:
                out.append(render("error", "E-TOUCH", f"{label_a} and {label_b}", where=nearest, variant="between",
                                  a=label_a, b=label_b, gap=dist, x=nearest[0], y=nearest[1],
                                  floor=cfg.thin_wall_rel * w))
            elif gap is not None and dist < float(gap):
                out.append(render("error", "E-APART-GAP", f"{label_a} and {label_b}", where=nearest, a=label_a,
                                  b=label_b, gap=dist, declared=float(gap)))
    finally:
        gmsh.model.remove()
        if current:
            gmsh.model.setCurrent(current)
    return out


def _apart_label(plan, name: str) -> str:
    s = (getattr(plan, "features", None) or {}).get(name)
    return f"{s.kind} '{name}'" if s is not None else f"'{name}'"


def _landing_channels(plan, m: Measurements) -> list[tuple[str, dict]]:
    """(channel op name, Solved.solved) for every channel whose feature records `lands_on`."""
    solved = getattr(plan, "features", None) or {}
    out = []
    for channel, records in m.legs.items():
        if not any(r["kind"] == "to" for r in records):
            continue
        name, _ = _feature_of(solved, channel)
        if name is None:
            continue
        s = solved[name]
        host = str(s.solved.get("lands_on") or "")
        # a landing is judged against its host; a partial built alone (3.15) carries the
        # host as context (solved, with none of its ops in the plan), and a landing on a
        # wall that was not built cannot be judged. A host absent from the features is
        # an ops-grammar sidecar's: the samples judge it (3.6, clause d skipped).
        if not host:
            continue
        host_solved = solved.get(host.split(".", 1)[0])
        built_names = {op.get("name") for op in (getattr(plan, "ops", None) or [])}
        if host_solved is not None and not any(op in built_names for op in host_solved.ops):
            continue
        out.append((channel, s.solved))
    return out


def _landings(gmsh, face: int, plan, wk: Walk, m: Measurements, cfg: LintConfig, w: float,
              span: float) -> list[Finding]:
    """D22, per leg recorded with `lands_on`, per built instance: (a) five samples across
    the mouth a tenth of a width PAST the `to` line must be inside; (b) the same five a
    tenth on the leg's side must be inside (else the mouth is capped); (c) the landing's
    u within the host wall's span, the span read off the built face (`host_wall_span`:
    the record's `wall_line` carries axis, value, flow and outward, not a u origin);
    (d) the `to` line on the host wall's line (closed form; a line outside the host
    is refused, a line inside it is a flush cut after the fuse and the samples judge it).
    (d) is judged before the samples, since a leg that stopped short fails (a) for that
    reason; the record's `wall_line` is read as a dict or as 3.4's tuple (`_wall_line`)."""
    out: list[Finding] = []
    solved = getattr(plan, "features", None) or {}
    tol = cfg.touch_rel * span + 1e-12
    for channel, sol in _landing_channels(plan, m):
        records = m.legs[channel]
        name, kind = _feature_of(solved, channel)
        width = next((r["width"] for r in records if r["kind"] == "ports"), w)
        row, offsets, _ = instances_of(plan, channel)
        host = str(sol.get("lands_on"))
        wall = sol.get("wall_line")
        for k, off in enumerate(offsets):
            instance = f"{row}[{k}]" if row else name
            subject = f"{kind or 'channel'} '{instance}'"
            for r in records:
                if r["kind"] != "to":
                    continue
                lands = _shift(r["lands"], off)
                h = math.radians(r["heading"])
                d = (math.cos(h), math.sin(h))
                probe = cfg.landing_probe_rel * width
                n = cfg.landing_samples
                axis = 0 if r["line"].startswith("x") else 1
                to_value = float(r["line"].split("=", 1)[1])
                # The mouth is the segment of the `to` line the leg cuts: half a width
                # over the sine of the leg's angle to the line, either side of the landing
                # ALONG the line (across the leg's own width two of five points would sit
                # inside the leg above the line, and a landing on nothing would read 3 of 5).
                half = width / 2 / max(abs(d[axis]), 1e-9)
                along = (0.0, 1.0) if axis == 0 else (1.0, 0.0)
                offsets_s = [-0.9 + 1.8 * i / (n - 1) for i in range(n)] if n > 1 else [0.0]

                def sample(sign):
                    outside = 0
                    for s in offsets_s:
                        p = (lands[0] + s * half * along[0] + sign * probe * d[0],
                             lands[1] + s * half * along[1] + sign * probe * d[1])
                        if not gmsh.model.isInside(2, face, [p[0], p[1], 0.0]):
                            outside += 1
                    return outside

                past, leg_side = sample(+1), sample(-1)
                fields = dict(line=r["line"], lx=lands[0], ly=lands[1], n=n, host=host, width=width)
                span_text = f"the mouth spans {lands[axis ^ 1] - half:.4g}..{lands[axis ^ 1] + half:.4g} on {r['line']}"
                start = half
                wall_frame = _wall_line(wall, axis, to_value)
                if wall_frame is not None and wall_frame[0] == axis:
                    w_axis, w_value, flow, outward = wall_frame
                    host_line = f"{'xy'[w_axis]} = {w_value:g}"
                    # Clause (d), closed form: the `to` line on the wall's line. It refuses
                    # only a line OUTSIDE the host (the cap stands proud, a gap between it
                    # and the wall); a line inside the host is cut where the fuse buries
                    # the cap and the outline is a flush cut (8.1 row 6: not a defect),
                    # so the samples judge it like any landing.
                    if (to_value - w_value) * outward[w_axis] > tol:
                        out.append(render("error", "E-LAND", subject, where=lands, variant="short",
                                          short=abs(w_value - to_value), host_line=host_line, k=past, **fields))
                        continue
                    measured = host_wall_span(wk, w_axis, w_value, outward, flow, max(tol, 1e-6 * width))
                    if measured is not None:
                        host_span, upstream = measured
                        u = (lands[0] - upstream[0]) * flow[0] + (lands[1] - upstream[1]) * flow[1]
                        span_text = f"{host} spans u 0..{host_span:.4g}; the landing is at u = {u:.4g}"
                        start = _row_start(plan, row, u, half)
                        # The samples are the judge; the u clause is the cross-check for a
                        # leg whose mouth is in fluid that is not its host's wall (the host
                        # wall as built cannot show the part a straddling mouth ate).
                        if not past and not leg_side and (u < -half or u > host_span + half):
                            out.append(render("error", "E-LAND", subject, where=lands, variant="span", u=u, u0=0.0,
                                              u1=host_span, start=start, **fields))
                            continue
                if past:
                    out.append(render("error", "E-LAND", subject, where=lands, k=past, span_text=span_text, start=start,
                                      **fields))
                    continue
                if leg_side:
                    out.append(render("error", "E-LAND", subject, where=lands, variant="capped", k=leg_side, **fields))
    return out


def _axis_index(axis, default: int) -> int:
    """The record's `wall_line.axis` is the letter of the wall's line ("y" for a wall on
    y = 1.5, section 4.2); an ops-grammar sidecar may write the index. Either reads to
    0 (x) or 1 (y); None is the leg's own axis."""
    if axis is None:
        return default
    if isinstance(axis, str):
        letter = axis.strip().lower()
        if letter in ("x", "y"):
            return "xy".index(letter)
        return int(letter)
    return int(axis)


def _wall_line(wall, axis: int, value: float) -> tuple[int, float, tuple, tuple] | None:
    """The record's `wall_line` as (axis, value, flow, outward), whether U1 wrote it as a
    dict with those keys or as the tuple 3.4 lists (axis, value, flow, outward, span);
    None when the record has none. A missing flow runs +x along a horizontal wall and +y
    along a vertical one; a missing outward points +y / +x."""
    if wall is None:
        return None
    if isinstance(wall, dict):
        w_axis = _axis_index(wall.get("axis"), axis)
        w_value = float(wall.get("value", value))
        flow, outward = wall.get("flow"), wall.get("outward")
    else:
        parts = list(wall)
        w_axis = _axis_index(parts[0] if len(parts) > 0 else None, axis)
        w_value = float(parts[1]) if len(parts) > 1 and parts[1] is not None else value
        flow = parts[2] if len(parts) > 2 else None
        outward = parts[3] if len(parts) > 3 else None
    flow = tuple(float(v) for v in flow) if flow else ((1.0, 0.0) if w_axis == 1 else (0.0, 1.0))
    outward = tuple(float(v) for v in outward) if outward else ((0.0, 1.0) if w_axis == 1 else (1.0, 0.0))
    return w_axis, w_value, flow, outward


def _row_start(plan, row, u, half) -> float:
    """The Row start that puts the mouth's upstream corner at the wall's start (u = 0)."""
    inst = (getattr(plan, "instances", None) or {}).get(row or "", {})
    anchors = inst.get("anchors") or []
    anchor = float(anchors[0]) if anchors else 0.0
    return anchor - (u - half)


def _notches(gmsh, face: int, wk: Walk, cfg: LintConfig, w: float) -> list[Finding]:
    out: list[Finding] = []
    for info in wk.curves.values():
        if info.kind != "line" or info.patch not in ("walls", "body") or info.length >= cfg.notch_edge_rel * w:
            continue
        outward = (-info.inward_normal[0], -info.inward_normal[1])
        solid = ray_depth(gmsh, face, info.midpoint, outward, w, 12, until="inside", resolution=1e-3 * w)
        if solid < cfg.notch_solid_rel * w:
            out.append(render("error", "E-NOTCH", "fluid", where=info.midpoint,
                              draw=[{"ray": [list(info.midpoint), [info.midpoint[0] + solid * outward[0],
                                                                   info.midpoint[1] + solid * outward[1]]]}],
                              edge=info.length, x=info.midpoint[0], y=info.midpoint[1], solid=solid, w=w,
                              floor=cfg.notch_solid_rel * w, curve=info.tag))
    return out


def _short_edges(wk: Walk, cfg: LintConfig, w: float, notched: set[int]) -> list[Finding]:
    out: list[Finding] = []
    short = [c for c in wk.curves.values() if c.length < cfg.short_warn_rel * w and c.tag not in notched]
    for info in wk.curves.values():
        if info.length < cfg.short_error_rel * w:
            out.append(render("error", "E-SHORT-EDGE", "fluid", where=info.midpoint, edge=info.length,
                              x=info.midpoint[0], y=info.midpoint[1], w=w, floor=cfg.short_error_rel * w,
                              curve=info.tag))
        elif info.length < cfg.short_warn_rel * w and info.tag not in notched:
            out.append(render("warn", "W-SHORT-EDGE", "fluid", where=info.midpoint, edge=info.length,
                              x=info.midpoint[0], y=info.midpoint[1], w=w, floor=cfg.short_warn_rel * w,
                              n=len(short), curve=info.tag))
    return out


def _cusps_and_lips(wk: Walk, cfg: LintConfig) -> list[Finding]:
    out: list[Finding] = []
    lips = [v for v in wk.vertices if v.kind != "straight" and v.solid_deg < cfg.cusp_info_deg]
    for v in wk.vertices:
        if v.kind == "straight":
            continue
        if v.interior_deg < cfg.cusp_warn_deg:
            out.append(render("warn", "W-CUSP", "fluid", where=v.at, deg=v.interior_deg, x=v.at[0], y=v.at[1]))
        elif v.interior_deg < cfg.cusp_info_deg:
            out.append(render("info", "I-CUSP", "fluid", where=v.at, deg=v.interior_deg, x=v.at[0], y=v.at[1],
                              warn=cfg.cusp_warn_deg))
        elif v.solid_deg < cfg.cusp_info_deg:
            out.append(render("info", "I-LIP", "fluid", where=v.at, deg=v.solid_deg, x=v.at[0], y=v.at[1],
                              n=len(lips)))
    return out


def _passage(m: Measurements, cfg: LintConfig, w: float) -> list[Finding]:
    p = m.passage
    if p is None or not p.n or p.min >= cfg.passage_warn_rel * w:
        return []
    return [render("warn", "W-PASSAGE", "fluid", where=p.where_min, min=p.min, res=1e-3 * w, x=p.where_min[0],
                   y=p.where_min[1], w=w, source=m.reference_width_from)]


def _ports(gmsh, wk: Walk, plan, m: Measurements, cfg: LintConfig, w: float) -> list[Finding]:
    _, _, findings = resolve_ports(gmsh, wk, plan, list(m.open_ends))
    external = any(s.kind == "BodyInBox" for s in (getattr(plan, "features", None) or {}).values())
    if external:
        findings = [f for f in findings if f.code not in ("E-PORT-OPEN",)]
    return findings


_LENGTH_MEASURES = frozenset({
    "width", "height", "length", "extent_x", "extent_y", "outer_radius", "inner_radius", "radius", "diameter",
    "pitch", "gap", "gap_measured", "spacing", "pass_length", "bend_radius", "pass_pitch", "wall_between",
    "leave_length", "lands_upstream_by", "centre_x", "centre_y", "footprint", "circumference", "size_x", "size_y",
})
"""The length-typed measures the units check reads when the claims carry no
`largest_length` (claims.LENGTH_MEASURES is the same set; lint cannot import claims)."""


def _largest_length(claims) -> float | None:
    if claims is None:
        return None
    given = claims.get("largest_length") if isinstance(claims, dict) else getattr(claims, "largest_length", None)
    if given:
        return float(given)
    items = claims.get("claims", []) if isinstance(claims, dict) else getattr(claims, "claims", []) or []
    best = None
    for c in items:
        get = (lambda k: c.get(k)) if isinstance(c, dict) else (lambda k: getattr(c, k, None))
        if get("kind") not in ("measure", "range") or get("measure") not in _LENGTH_MEASURES:
            continue
        for key in ("value", "max", "min"):
            v = get(key)
            if v is not None and (best is None or float(v) > best):
                best = float(v)
    return best


def _units(m: Measurements, claims, cfg: LintConfig, span: float) -> list[Finding]:
    """r = span / L_req with the claims' largest length CONVERTED into the sketch's units
    through the metre: the sketch's unit is its `scale` (metres per unit), so a `--spec`
    fixture at scale 0.004 is mm x 4 against mm claims (fixture 15) and a mm sketch
    against claims in m is off by 1000 (fixture 14)."""
    largest = _largest_length(claims)
    if not largest:
        return []
    cunit = (claims.get("unit") if isinstance(claims, dict) else getattr(claims, "unit", None)) or m.units
    if cunit not in SCALE_BY_UNIT:
        return []
    sketch_scale = float(m.scale) if m.scale else SCALE_BY_UNIT.get(m.units)
    if not sketch_scale:
        return []
    factor = SCALE_BY_UNIT[cunit] / sketch_scale
    converted = largest * factor
    r = span / converted
    lo, hi = cfg.units_error
    if r < lo or r > hi:
        return [render("error", "E-UNITS", "fluid", span=span, units=m.units, L=largest, cunit=cunit, Lconv=converted,
                       r=r, factor=factor if factor >= 1 else 1 / factor)]
    lo, hi = cfg.units_warn
    if r < lo or r > hi:
        return [render("warn", "W-UNITS", "fluid", span=span, L=converted, r=r,
                       sense="larger" if r > 1 else "smaller")]
    return []
