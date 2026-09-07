"""The print-back: the one place text is assembled (DESIGN.md section 6).

Fixed section order (SCRIPT, LINT, FEATURES, LEGS, MEASURED, PATCHES, CLAIMS, REFERENCE,
VERDICT), a fixed left column of eleven characters, all lengths in the sketch's units with
the metre equivalent once on the extent line; `verdict` is computed from the tables and
the desk reads its bool, never the text. The same function renders the three shapes a lap
can take: the full print-back (rc 0 and 2), the print-back of a `partial` with the refusal
under LINT and 'CLAIMS not evaluated' (rc 5 with a Row's item drawn alone), and the
refusal alone under one SCRIPT line (rc 3, and rc 5 without a partial).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from . import _toolbox  # noqa: F401  (mesh2d.leg_lines by path, for the plain leg table)
from .claims import ComplianceTable, fmt_angle, fmt_point
from .lint import Finding, errors as _errors, warnings as _warnings
from .measure import Measurements

if TYPE_CHECKING:
    from .compile import Plan
    from .library import ReferenceMatch

COL = 11
"""The left column: a section word padded to eleven characters."""
INDENT = " " * COL
FEATURE_INDENT = " " * (COL + 10 + 11)
"""Continuation lines of a FEATURES entry start under the summary column."""
LEG_INDENT = " " * (COL + 2)
FINDING_INDENT = " " * (COL + 10)
"""Continuation lines of a finding start under its text (the section-5 format's ten
spaces, moved right by the column)."""

_LEAD = {"error": "!! ERROR ", "warn": "!!       ", "info": "check    "}
_METRES = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "in": 0.0254}


def _section(word: str, first: str) -> str:
    return f"{word:<{COL}}{first}"


def _g(v) -> str:
    return f"{float(v):.4g}"


def _f2(v) -> str:
    return f"{float(v):.2f}"


def _pt(p) -> str:
    return fmt_point(p)


def _sense(heading: float, axis: int = 0) -> str:
    """mesh2d's sense word for a heading: +x, -x, or across x."""
    along = "xy"[axis]
    h = float(heading) % 360
    comp = math.cos(math.radians(h)) if axis == 0 else math.sin(math.radians(h))
    return f"+{along}" if comp > 1e-9 else (f"-{along}" if comp < -1e-9 else f"across {along}")


# ----------------------------------------------------------------------------- SCRIPT

def script_lines(printed: str, notes: list[str], seconds: float) -> list[str]:
    prints = [line for line in (printed or "").splitlines() if line.strip()]
    parts = []
    if prints:
        parts.append(f"{len(prints)} print{'s' if len(prints) != 1 else ''}")
    if notes:
        parts.append(f"{len(notes)} note{'s' if len(notes) != 1 else ''}")
    head = f"ran in {seconds:.1f} s; " + (", ".join(parts) if parts else "no prints")
    out = [_section("SCRIPT", head)]
    out.extend(f"{INDENT}> {line}" for line in prints)
    out.extend(f"{INDENT}note: {note}" for note in notes)
    return out


# ----------------------------------------------------------------------------- LINT

def finding_lines(f: Finding) -> list[str]:
    """A finding in the section-5 format under the LINT column: the head line, then every
    continuation line (the fix, a wrapped sentence, a table) under the text. A finding
    with no subject (E-NOTCH, E-SHORT-EDGE, I-LIP) prints without the colon."""
    if f.subject:
        head = f"{_LEAD[f.level]} {f.code}  {f.subject}: {f.what}"
    else:
        head = f"{_LEAD[f.level]} {f.code}  {f.what}"
    body = head.split("\n")
    if f.fix:
        body.extend(f.fix.split("\n"))
    out = [f"{INDENT}{body[0]}"]
    for line in body[1:]:
        if line.startswith(" " * 10):
            line = line[10:]
        out.append(f"{FINDING_INDENT}{line}" if line.strip() else "")
    return [line for line in out if line != ""]


def lint_lines(findings: list[Finding]) -> list[str]:
    errs = _errors(findings)
    warns = _warnings(findings)
    infos = [f for f in findings if f.level == "info"]
    parts = []
    if errs:
        parts.append(f"{len(errs)} error{'s' if len(errs) != 1 else ''}")
    if warns:
        parts.append(f"{len(warns)} warning{'s' if len(warns) != 1 else ''}")
    head = ", ".join(parts) if parts else "clean"
    if errs:
        head += " (errors block)"
    out = [_section("LINT", head)]
    for f in errs + warns + infos:
        out.extend(finding_lines(f))
    return out


# ----------------------------------------------------------------------------- FEATURES

def _feature_row(name: str, kind: str, summary: list[str]) -> list[str]:
    first = summary[0] if summary else ""
    out = [f"{INDENT}{name:<10}{kind:<11}{first}".rstrip()]
    out.extend(f"{FEATURE_INDENT}{line}" for line in summary[1:])
    return out


def _repeat_targets(plan: "Plan") -> dict[str, str]:
    """item name -> row name, from the plan's repeat ops."""
    return {op.get("target"): op.get("name") for op in plan.ops if op.get("op") == "repeat"}


def _flow_word(v) -> str:
    if not v:
        return ""
    x, y = float(v[0]), float(v[1])
    if abs(x) >= abs(y):
        return "+x" if x > 0 else "-x"
    return "+y" if y > 0 else "-y"


def _bypass_summary(name: str, sol, m: Measurements | None, built: bool) -> list[str]:
    s, p = sol.solved, sol.params
    width = p.get("width")
    outer = p.get("outer_radius")
    R = s.get("R", (outer - width / 2) if outer is not None and width is not None else None)
    inner = (outer - width) if outer is not None and width is not None else None
    wall = s.get("wall_line") or {}
    flow = wall.get("flow") or (1.0, 0.0)
    outward = wall.get("outward") or (0.0, 1.0)
    phi = math.degrees(math.atan2(float(flow[1]), float(flow[0])))
    left = (float(flow[0]) * float(outward[1]) - float(flow[1]) * float(outward[0])) > 0
    t_return = s.get("theta_return", p.get("return_angle"))
    heading = None
    if t_return is not None:
        heading = (phi + (180 + float(t_return)) * (1 if left else -1)) % 360
    feat = (m.features.get(name) if m is not None else None) or {}
    if built and feat.get("return_heading") is not None:
        heading = float(feat["return_heading"]) % 360
    parts = []
    if R is not None:
        parts.append(f"R {_g(R)} (outer {_g(outer)}, inner {_g(inner)})")
    if s.get("sweep") is not None:
        parts.append(f"sweep {_g(s['sweep'])}")
    if heading is not None:
        if built:
            comp = math.cos(math.radians(heading - phi))
            parts.append(f"return heading {_g(heading)} ({'against' if comp < 0 else 'with'} {_flow_word(flow)})")
        else:
            parts.append(f"return heading {_g(heading)} ({_sense(heading)})")
    if s.get("leave_length") is not None:
        parts.append(f"leave_length {_g(s['leave_length'])}" + (" (default: one width)" if s.get("leave_length_default") else ""))
    first = ("" if built else "solved: ") + ", ".join(parts) + ("," if parts else "")
    second = []
    landing = s.get("landing")
    if landing is not None:
        u = float(landing[0])
        second.append(f"lands {_f2(abs(u))} {'upstream' if u < 0 else 'downstream'} of its anchor")
    if built and feat.get("lip_angle") is not None and s.get("lip_u") is not None:
        second.append(f"lip {_f2(s['lip_u'])} downstream at {fmt_angle(feat['lip_angle'])} deg")
    fp = s.get("footprint")
    if fp is not None:
        second.append(f"footprint {_f2(fp[0])}..{float(fp[1]):+.2f}")
    if s.get("height") is not None:
        second.append(f"height {_g(s['height'])}")
    lines = [first]
    if second:
        lines.append(", ".join(second))
    return lines


def _row_summary(name: str, sol, plan: "Plan", m: Measurements | None) -> list[str]:
    s, p = sol.solved, sol.params
    if not s or s.get("pitch") is None:
        return ["NOT SOLVED"]
    count = p.get("count", plan.instances.get(name, {}).get("count"))
    pitch = s.get("pitch")
    fp = s.get("footprint")
    gap = s.get("gap")
    fp_along = (float(fp[1]) - float(fp[0])) if isinstance(fp, (list, tuple)) else (float(fp) if fp is not None else None)
    how = "declared" if s.get("gap_declared") else "solved"
    first = f"{count} at pitch {_f2(pitch)}"
    if fp_along is not None and gap is not None:
        first += f" ({_f2(fp_along)} + {_f2(gap)} {how})"
    anchors = s.get("anchors") or []
    if anchors:
        first += f", anchors u = {', '.join(_f2(a) for a in anchors)},"
    span = s.get("span")
    second = ""
    if span is not None:
        second = f"span {_f2(span[0])}..{_f2(span[1])}"
        along = s.get("along")
        if isinstance(along, str) and along:
            wall_span = _wall_span_for(name, plan)
            second += f" on {along}" + (f" (0..{_g(wall_span)})" if wall_span is not None else "")
        align = p.get("align")
        if align == "centre":
            second += ", centred"
        elif align in ("start", "end"):
            second += f", aligned to the {align}"
        elif p.get("start") is not None:
            second += f", first anchor at {_g(p['start'])}"
    return [first] + ([second] if second else [])


def _wall_span_for(row: str, plan: "Plan") -> float | None:
    for op in plan.ops:
        if op.get("op") == "repeat" and op.get("name") == row:
            item = plan.features.get(op.get("target"))
            if item is not None:
                wall = item.solved.get("wall_line") or {}
                if wall.get("span") is not None:
                    return float(wall["span"])
    return None


def _passage_summary(name: str, sol, m: Measurements | None) -> list[str]:
    s, p = sol.solved, sol.params
    feat = (m.features.get(name) if m is not None else None) or {}
    width = p.get("width", feat.get("width"))
    legs = [r for r in (s.get("legs") or []) if r.get("kind") != "ports"]
    start = p.get("start", feat.get("start"))
    heading = p.get("heading", feat.get("start_heading"))
    end = s.get("end", feat.get("end"))
    end_heading = s.get("end_heading", feat.get("end_heading"))
    if len(legs) == 1 and legs[0].get("kind") == "line":
        line = f"width {_g(width)}, one leg {_g(legs[0].get('length'))} along {_sense(legs[0].get('heading', 0))}"
        if start is not None and end is not None:
            line += f", start {_pt(start)} end {_pt(end)}"
        return [line]
    straight = sum(1 for r in legs if r.get("kind") == "line")
    corners = sum(1 for r in legs if r.get("kind") == "corner")
    detail = f" ({straight} straight, {corners} corner{'s' if corners != 1 else ''})" if corners else ""
    line = f"width {_g(width)}, {len(legs)} legs{detail}"
    if s.get("length") is not None:
        line += f", centreline {_g(s['length'])}"
    tail = []
    if start is not None:
        tail.append(f"start {_pt(start)}" + (f" heading {_g(heading)}" if heading is not None else ""))
    if end is not None:
        tail.append(f"end {_pt(end)}" + (f" heading {_g(end_heading)}" if end_heading is not None else ""))
    if tail:
        line += "; " + ", ".join(tail)
    out = [line]
    extras = []
    if feat.get("spacing") is not None:
        extras.append(f"parallel straight legs {_g(feat['spacing'])} apart")
    if feat.get("end_side") == "same" and start is not None:
        extras.append(f"both ends on x = {_g(start[0])}")
    if extras:
        out[0] += ";"
        out.append("; ".join(extras))
    return out


def _serpentine_summary(name: str, sol, m: Measurements | None) -> list[str]:
    s, p = sol.solved, sol.params
    width = p.get("width")
    r = p.get("bend_radius")
    outer = inner = None
    if r is not None and width is not None:
        outer, inner = r + width / 2, r - width / 2
    first = f"{p.get('passes')} passes of {_g(p.get('pass_length'))}, {s.get('bends')} bends r {_g(r)}"
    if outer is not None:
        first += f" (outer {_g(outer)}, inner {_g(inner)})"
    if s.get("pass_pitch") is not None:
        first += f", pass pitch {_g(s['pass_pitch'])}"
    if s.get("wall_between") is not None:
        first += f", wall between passes {_g(s['wall_between'])}"
    second = f"start {_pt(p.get('start', (0, 0)))} heading {_g(p.get('heading', 0))}"
    if s.get("end") is not None:
        second += f"; end {_pt(s['end'])} heading {_g(s.get('end_heading', 0))}"
        if s.get("ends_on_start_side") is not None:
            second += ": on the start's side" if s["ends_on_start_side"] else ": opposite the start"
    return [first + ";", second]


def _rect_summary(name: str, sol, m: Measurements | None) -> list[str]:
    p = sol.params
    feat = (m.features.get(name) if m is not None else None) or {}
    size = p.get("size") or (feat.get("size_x"), feat.get("size_y"))
    origin = p.get("origin", feat.get("origin"))
    centre = p.get("centre", feat.get("centre"))
    line = f"{_g(size[0])} x {_g(size[1])}"
    if origin is not None:
        line += f" at origin {_pt(origin)}"
    if centre is not None:
        line += f", centre {_pt(centre)}"
    if feat.get("width") is not None and feat.get("length") is not None:
        line += f"; width {_g(feat['width'])} (the shorter side), length {_g(feat['length'])}"
    return [line]


def _disk_summary(name: str, sol, m: Measurements | None) -> list[str]:
    p = sol.params
    feat = (m.features.get(name) if m is not None else None) or {}
    centre = feat.get("centre", p.get("centre"))
    line = f"centre {_pt(centre)}" if centre is not None else ""
    if feat.get("radius") is not None:
        line += f", radius {float(feat['radius']):.3f} (diameter {_g(feat.get('diameter', 2 * float(feat['radius'])))}, from the hole's curvature)"
    elif p.get("radius") is not None:
        line += f", radius {_g(p['radius'])} (typed; not built)"
    if feat.get("circumference") is not None:
        line += f", circumference {_g(feat['circumference'])}"
    return [line.lstrip(", ")]


def _generic_summary(name: str, sol, m: Measurements | None) -> list[str]:
    feat = (m.features.get(name) if m is not None else None) or {}
    parts = []
    for key in ("width", "length", "extent", "islands", "area"):
        if key in feat and feat[key] is not None:
            v = feat[key]
            parts.append(f"{key} {_pt(v) if isinstance(v, (list, tuple)) else _g(v)}")
    if not parts:
        parts = [f"{k} {_pt(v) if isinstance(v, (list, tuple)) and len(v) == 2 else _g(v) if isinstance(v, (int, float)) else v}"
                 for k, v in sol.params.items() if isinstance(v, (int, float, list, tuple, str))][:4]
    return [", ".join(parts)] if parts else [""]


def feature_lines(plan: "Plan | None", m: Measurements | None, findings: list[Finding], built: bool) -> list[str]:
    if plan is None:
        return []
    out: list[str] = []
    not_solved = {}
    for f in _errors(findings):
        if f.code.startswith("E-ROW") and f.subject.startswith("Row '"):
            not_solved[f.subject[5:].split("'")[0]] = f.code
    seen = set()
    for name, sol in plan.features.items():
        seen.add(name)
        kind = sol.kind
        if kind == "Bypass":
            summary = _bypass_summary(name, sol, m, built)
        elif kind == "Row":
            summary = _row_summary(name, sol, plan, m)
            if summary == ["NOT SOLVED"]:
                summary = [f"NOT SOLVED ({not_solved.get(name, 'refused')})"]
        elif kind == "Passage":
            summary = _passage_summary(name, sol, m)
        elif kind == "Serpentine":
            summary = _serpentine_summary(name, sol, m)
        elif kind == "Rect":
            summary = _rect_summary(name, sol, m)
        elif kind == "Disk":
            summary = _disk_summary(name, sol, m)
        else:
            summary = _generic_summary(name, sol, m)
        out.extend(_feature_row(name, kind, summary))
    for name, code in not_solved.items():
        if name not in seen:
            out.extend(_feature_row(name, "Row", [f"NOT SOLVED ({code})"]))
    return [_section("FEATURES", out[0][COL:])] + out[1:] if out else []


# ----------------------------------------------------------------------------- LEGS

def _leg_line(i: int, r: dict) -> str:
    kind = r.get("kind")
    h = float(r.get("heading", 0)) % 360
    if kind == "arc":
        turn = "left" if float(r.get("sweep", 0)) > 0 else "right"
        outer = r.get("outer_radius")
        inner = r.get("inner_radius")
        radii = f"r {_g(r['radius'])}"
        if outer is not None and inner is not None:
            radii += f" (outer {_g(outer)}, inner {_g(inner)})"
        return (f"leg {i}  arc  {radii} centre {_pt(r['centre'])} {_g(abs(float(r['sweep'])))} deg {turn}: "
                f"heading {_g(h)} -> {_g(float(r.get('heading_out', h)) % 360)}")
    if kind == "corner":
        turn = float(r.get("turn", r.get("sweep", 0)))
        line = f"leg {i}  corner {_g(abs(turn))} deg {'left' if turn > 0 else 'right'} at {_pt(r['from'])}"
        if r.get("outer") is not None and r.get("inner") is not None:
            line += f": outer vertex {_pt(r['outer'])}, inner vertex {_pt(r['inner'])}"
        return line
    if kind == "to":
        what = f"to {str(r.get('line', '')).replace(':', '=')}"
    else:
        what = f"line {_g(r.get('length', 0))}"
    line = (f"leg {i}  {what:<9}  from {_pt(r['from'])} heading {_g(h)} deg ({_sense(h)}) "
            f"to {_pt(r['to'])}")
    if r.get("lands"):
        line += f"   lands on {r['lands']}"
    elif r.get("note"):
        line += f"   ({r['note']})"
    return line


def leg_lines(legs: dict, wall_frame: dict | None = None) -> list[str]:
    """The leg tables in the print-back's format (mesh2d's numbers and words, the `what`
    column padded so the `from` points line up), one block per channel: a header line and
    one line per leg. `wall_frame` (optional) maps a channel name to how its header
    reads: {"<channel>": {"row": "loops", "count": 4, "anchor": 7.24, "pitch": 14.66}}
    prints `loops[0] (anchor 7.24; loops[1..3] identical, shifted by 14.66)`;
    {"<channel>": {"name": "loop", "anchor": 0}} prints `loop (one instance, anchor at
    u = 0 for the table)`; every instance reference is `<row>[k]`, 0-based (D29)."""
    frames = wall_frame or {}
    out: list[str] = []
    for channel, records in legs.items():
        frame = frames.get(channel) or {}
        if frame.get("row"):
            row, count = frame["row"], int(frame.get("count", 1))
            head = f"{row}[0]"
            extra = []
            if frame.get("anchor") is not None:
                extra.append(f"anchor {_f2(frame['anchor'])}")
            if count > 1 and frame.get("pitch") is not None:
                rest = f"{row}[1]" if count == 2 else f"{row}[1..{count - 1}]"
                extra.append(f"{rest} identical, shifted by {_f2(frame['pitch'])}")
            if extra:
                head += f" ({'; '.join(extra)})"
        elif frame.get("name") and frame.get("single"):
            anchor = frame.get("anchor", 0)
            head = f"{frame['name']} (one instance, anchor at u = {_g(anchor)} for the table)"
        else:
            head = frame.get("name") or channel
        out.append(head)
        i = 0
        for r in records:
            if r.get("kind") == "ports":
                continue
            i += 1
            out.append(f"  {_leg_line(i, r)}")
    return out


def leg_frames(plan: "Plan | None", legs: dict) -> dict:
    """How each channel's leg table is headed, from the plan: a Row's item channel is the
    row's first instance; a Bypass outside a Row is a single instance; a Passage or
    Serpentine is itself."""
    frames: dict[str, dict] = {}
    if plan is None:
        return frames
    targets = _repeat_targets(plan)
    for channel in legs:
        owner = None
        for name, sol in plan.features.items():
            if channel == name or channel in sol.ops:
                owner = (name, sol)
                break
        if owner is None:
            continue
        name, sol = owner
        row = targets.get(name)
        if row is not None and row in plan.features:
            rs = plan.features[row].solved
            anchors = rs.get("anchors") or []
            frames[channel] = {"row": row, "count": int(plan.features[row].params.get("count", len(anchors) or 1)),
                               "anchor": anchors[0] if anchors else None, "pitch": rs.get("pitch")}
        elif sol.kind == "Bypass":
            frames[channel] = {"name": name, "single": True, "anchor": sol.params.get("at") or 0}
        else:
            frames[channel] = {"name": name}
    return frames


def _legs_block(plan: "Plan | None", legs: dict) -> list[str]:
    if not legs:
        return []
    lines = leg_lines(legs, leg_frames(plan, legs))
    out = [_section("LEGS", lines[0])]
    for line in lines[1:]:
        out.append(f"{INDENT}{line}" if not line.startswith("  ") else f"{INDENT}{line}")
    return out


# ----------------------------------------------------------------------------- MEASURED

def _metres(v: float, units: str, scale: float) -> str:
    factor = scale if scale else _METRES.get(units, 1.0)
    return _g(float(v) * factor)


def extent_line(m: Measurements) -> str:
    ex, ey = m.extent
    return (f"extent {_g(ex)} x {_g(ey)} {m.units} "
            f"({_metres(ex, m.units, m.scale)} x {_metres(ey, m.units, m.scale)} m)")


def measured_lines(m: Measurements, plan: "Plan | None") -> list[str]:
    parts = [extent_line(m), f"area {_g(m.area)} {m.units}2", f"islands {m.islands}"]
    if m.passage is not None:
        res = 1e-3 * m.reference_width
        parts.append(f"narrowest passage {m.passage.min:.3f} (+/- {res:.3f}) at {_pt(m.passage.where_min)}")
    else:
        parts.append("narrowest passage not measured")
    length, at = m.shortest_edge
    parts.append(f"{m.n_curves} edges, shortest {_f2(length)} at {_pt(at)}")
    out = [_section("MEASURED", "   ".join(parts))]
    if m.holes:
        where = ", ".join(_pt(h.centroid) + (f" r {_g(h.radius)}" if h.radius is not None else "") for h in m.holes)
        line = f"islands at {where}"
        rows = [r for r in m.rows.values()]
        if plan is not None and not rows:
            rows = [type("R", (), {"count": int(i.get("count", 0))})() for i in plan.instances.values()]
        if len(rows) == 1 and rows[0].count == len(m.holes) == m.islands:
            line += ", one inside each loop"
        out.append(f"{INDENT}{line}")
    return out


# ----------------------------------------------------------------------------- PATCHES

def _port_source(name: str, plan: "Plan | None") -> str:
    if plan is None:
        return ""
    for port in plan.ports:
        if port.name == name:
            return ", ".join(str(e) for e in port.edges)
    return ""


def _patch_text(name: str, p: dict, plan: "Plan | None") -> str:
    n = int(p.get("n", len(p.get("curves", []))))
    length = p.get("length")
    text = f"{name} ({n} edge{'s' if n != 1 else ''}" + (f", {float(length):.1f})" if length is not None else ")")
    mids = p.get("midpoints") or []
    if n == 1 and mids:
        text += f" at {_pt(mids[0])}"
    source = _port_source(name, plan)
    if source:
        text += f" = {source}"
    return text


def patch_lines(m: Measurements, plan: "Plan | None") -> list[str]:
    if not m.patches:
        return [_section("PATCHES", "none resolved")]
    names = list(m.patches)
    first = [n for n in ("inlet", "outlet") if n in names]
    rest = [n for n in names if n not in first and n != "walls"]
    if "walls" in names:
        rest.append("walls")
    lines = []
    if first:
        lines.append("     ".join(_patch_text(n, m.patches[n], plan) for n in first))
    lines.extend(_patch_text(n, m.patches[n], plan) for n in rest)
    return [_section("PATCHES", lines[0])] + [f"{INDENT}{line}" for line in lines[1:]]


# ----------------------------------------------------------------------------- CLAIMS / REFERENCE / VERDICT

def claims_lines(table: ComplianceTable | None, findings: list[Finding], built: bool) -> list[str]:
    if table is None:
        if not built or _errors(findings):
            return [_section("CLAIMS", "not evaluated: the sketch did not build")]
        return [_section("CLAIMS", "not evaluated: no claims")]
    lines = table.lines()
    return [_section("CLAIMS", lines[0])] + [f"{INDENT}{line}" for line in lines[1:]]


def reference_lines(reference: "ReferenceMatch | None", m: Measurements | None) -> list[str]:
    if reference is None:
        return [_section("REFERENCE", "none (no approved library entry matches)")]
    meas = reference.measurements or {}
    params = meas.get("params") or {}
    head = f"library {reference.entry} (preset {reference.preset}, approved {reference.approved_at or 'undated'})"
    if params:
        head += ": " + ", ".join(f"{k} {_g(v) if isinstance(v, (int, float)) else v}" for k, v in params.items())
    out = [_section("REFERENCE", head)]
    if m is not None:
        rx = meas.get("extent")
        parts = []
        if rx:
            parts.append(f"extent {_g(rx[0])} x {_g(rx[1])} vs {_g(m.extent[0])} x {_g(m.extent[1])}")
        if meas.get("islands") is not None:
            parts.append(f"islands {meas['islands']} vs {m.islands}")
        h = "not measured" if reference.hausdorff is None else _g(reference.hausdorff)
        out.append(f"{INDENT}reference vs candidate: {', '.join(parts)}; Hausdorff {h}" if parts
                   else f"{INDENT}reference vs candidate: Hausdorff {h}")
    return out


def verdict(findings: list[Finding], table: ComplianceTable | None) -> tuple[str, bool]:
    """('ready to COMMIT', True) | ('not ready: LINT E-ROW-FIT; claims c6 FAIL   (COMMIT
    disagrees: c6 records it)', False). Computed, never parsed; the desk uses the bool."""
    codes = []
    for f in _errors(findings):
        if f.code not in codes:
            codes.append(f.code)
    parts = []
    if codes:
        parts.append(f"LINT {', '.join(codes)}")
    failing = table.failing_ids() if table is not None else []
    if failing:
        parts.append(f"claims {', '.join(failing)} FAIL")
    if table is None and not codes:
        parts.append("no claims")
    elif table is not None and not table.checkable and not codes and not failing:
        parts.append("no checkable claim")
    if not parts:
        return "ready to COMMIT", True
    text = "not ready: " + "; ".join(parts)
    if failing:
        text += f"   (COMMIT disagrees: {', '.join(failing)} records it)"
    return text, False


# ----------------------------------------------------------------------------- the print-back

def text(printed: str, notes: list[str], seconds: float, findings: list[Finding], plan: "Plan | None",
         m: Measurements | None, legs: dict, table: ComplianceTable | None,
         reference: "ReferenceMatch | None", refusal: str | None = None) -> str:
    """The print-back of section 6, in the fixed section order, all lengths in the
    sketch's units with the metre equivalent once on the extent line. Renders only the
    sections it has: at rc 5 with a `partial` (3.15) `plan`/`m` are the partial's, CLAIMS
    reads 'not evaluated: the sketch did not build' and VERDICT 'not ready: LINT <code>';
    at rc 5 without a partial and at rc 3 the whole text is `refusal` under one SCRIPT line
    (the one rule for rc 3/5 text; 3.10 and 6.1 say the same)."""
    out = script_lines(printed, notes, seconds)
    if m is None and plan is None:
        if refusal:
            out.extend(f"{INDENT}{line}" if line.strip() else "" for line in refusal.rstrip().splitlines())
        elif findings:
            out.extend(lint_lines(findings))
        return "\n".join(out)
    built = table is not None or not _errors(findings)
    out.extend(lint_lines(findings))
    out.extend(feature_lines(plan, m, findings, built=table is not None))
    out.extend(_legs_block(plan, legs or {}))
    if m is not None:
        out.extend(measured_lines(m, plan))
        out.extend(patch_lines(m, plan))
    out.extend(claims_lines(table, findings, built))
    out.extend(reference_lines(reference, m))
    out.append(_section("VERDICT", verdict(findings, table)[0]))
    return "\n".join(out)
