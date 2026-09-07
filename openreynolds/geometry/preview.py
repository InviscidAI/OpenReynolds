"""The annotated picture: what the model sees first each lap and what a person sees
in the tool result.

matplotlib is imported inside the functions, never at module level: the child
interpreter imports it after the script has run (the import guard is gone by then) and
`child_env` gives it an MPLCONFIGDIR. Everything drawn comes from the same records the
print-back prints (the leg tables, the measurements, the findings, the compliance
table), so the picture and the text cannot disagree; the tests read the Figure back
through `_figure()` (inset titles, marker positions, the caption) and never compare
pixels.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from . import _toolbox
from .claims import ComplianceTable
from .lint import Finding
from .measure import Measurements

if TYPE_CHECKING:
    from .compile import Plan
    from .library import ReferenceMatch

DPI = 110
"""The picture is 1024 px wide at this dpi; the model reads it at that size."""

MAX_INSETS = 8
"""Junction windows tiled along the top edge; past this the title list says '...'."""

CAPTION_LINES = 16
"""Caption lines shown under the picture before '... and N more'."""

FILL = "#d5d9e0"
FILL_EDGE = "#c3c7ce"
ANNOTATION = "#3b4a6b"
GOLDEN = "#8c8c8c"
LEVEL_COLOURS = {"error": "#d62728", "warn": "#ff7f0e"}
EXTRA_COLOURS = ("#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22")
"""One colour per named patch beyond mesh2d.COLOURS, in a stable order."""

GOLDEN_DIR = Path(__file__).resolve().parent / "library" / "golden"
"""Where the library keeps <name>.png; read by path so the preview never imports the
library (the library imports the preview)."""


# -- reading the face ---------------------------------------------------------------


def _curve_points(gmsh, tag: int, n: int = 64) -> list[tuple[float, float]]:
    """n points along a curve, in parameter order."""
    t0, t1 = gmsh.model.getParametrizationBounds(1, tag)
    ts = [t0[0] + (t1[0] - t0[0]) * i / (n - 1) for i in range(n)]
    xyz = gmsh.model.getValue(1, tag, ts)
    return [(xyz[3 * i], xyz[3 * i + 1]) for i in range(n)]


def _curve_length(gmsh, tag: int) -> float:
    return float(gmsh.model.occ.getMass(1, tag))


def _face_curves(gmsh, face: int) -> list[int]:
    return sorted({abs(t) for _, t in gmsh.model.getBoundary([(2, face)], combined=False, oriented=False)})


def _bounds(points) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _triangles(gmsh, face: int, span: float) -> list[list[tuple[float, float]]]:
    """A coarse triangulation of the face, drawn and thrown away; the real mesh is made
    afterwards with its own options, which are restored here."""
    option = gmsh.option
    saved = {name: option.getNumber(name) for name in
             ("Mesh.MeshSizeMin", "Mesh.MeshSizeMax", "Mesh.MeshSizeFromCurvature", "Mesh.Algorithm")}
    try:
        option.setNumber("Mesh.MeshSizeMin", 0)
        option.setNumber("Mesh.MeshSizeMax", span / 60)
        option.setNumber("Mesh.MeshSizeFromCurvature", 20)
        option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
        ntags, coords, _ = gmsh.model.mesh.getNodes()
        pos = {int(t): (coords[3 * i], coords[3 * i + 1]) for i, t in enumerate(ntags)}
        _, enodes = gmsh.model.mesh.getElementsByType(2, face)
        return [[pos[int(n)] for n in enodes[i:i + 3]] for i in range(0, len(enodes), 3)]
    finally:
        gmsh.model.mesh.clear()
        for name, value in saved.items():
            option.setNumber(name, value)


def outline_polyline(gmsh, face: int, spacing: float) -> list[list[tuple[float, float]]]:
    """The boundary as ordered loops of points at `spacing` arc length (the golden artefact).
    `getCurveLoops` gives each loop's curves unordered, so every loop is chained by matching
    end points, each curve reversed where the chain needs it; the outer loop is made
    counter-clockwise and every hole clockwise so two goldens of one shape agree point
    for point, not only as sets."""
    occ = gmsh.model.occ
    loops, curve_sets = occ.getCurveLoops(face)
    out: list[list[tuple[float, float]]] = []
    for tags in curve_sets:
        pending = {}
        for tag in tags:
            tag = abs(int(tag))
            length = max(_curve_length(gmsh, tag), 1e-12)
            n = max(2, int(math.ceil(length / max(spacing, 1e-12))) + 1)
            pending[tag] = _curve_points(gmsh, tag, n)
        if not pending:
            continue
        span = max(max(_bounds(p)[2] - _bounds(p)[0], _bounds(p)[3] - _bounds(p)[1]) for p in pending.values())
        tol = 1e-6 * max(span, 1.0) + 1e-9
        first = min(pending)
        chain = list(pending.pop(first))
        while pending:
            end = chain[-1]
            picked = None
            for tag, pts in pending.items():
                if math.dist(pts[0], end) <= tol:
                    picked, seq = tag, pts
                    break
                if math.dist(pts[-1], end) <= tol:
                    picked, seq = tag, list(reversed(pts))
                    break
            if picked is None:
                # a loop the chain cannot close (a degenerate curve); keep what was walked
                nearest = min(pending, key=lambda t: min(math.dist(pending[t][0], end), math.dist(pending[t][-1], end)))
                seq = pending[nearest]
                if math.dist(seq[-1], end) < math.dist(seq[0], end):
                    seq = list(reversed(seq))
                picked = nearest
            pending.pop(picked)
            chain.extend(seq[1:])
        if len(chain) > 1 and math.dist(chain[0], chain[-1]) <= tol:
            chain.pop()
        out.append(chain)
    if not out:
        return out
    areas = [_signed_area(loop) for loop in out]
    outer = max(range(len(out)), key=lambda i: abs(areas[i]))
    ordered = [out[outer]] + [loop for i, loop in enumerate(out) if i != outer]
    if _signed_area(ordered[0]) < 0:
        ordered[0] = list(reversed(ordered[0]))
    for i in range(1, len(ordered)):
        if _signed_area(ordered[i]) > 0:
            ordered[i] = list(reversed(ordered[i]))
    return ordered


def _signed_area(loop) -> float:
    total = 0.0
    n = len(loop)
    for i in range(n):
        x0, y0 = loop[i]
        x1, y1 = loop[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return total / 2.0


def hausdorff(a: list, b: list) -> float:
    """Symmetric Hausdorff distance between two outlines (lists of loops of points), the
    number the caption prints beside a reference; numpy, chunked so a 20,000-point golden
    does not allocate a 20,000 x 20,000 table."""
    import numpy as np

    pa = np.array([p for loop in a for p in loop], dtype=float).reshape(-1, 2)
    pb = np.array([p for loop in b for p in loop], dtype=float).reshape(-1, 2)
    if len(pa) == 0 or len(pb) == 0:
        return float("inf")

    def directed(p, q):
        worst = 0.0
        for i in range(0, len(p), 500):
            d = np.sqrt(((p[i:i + 500, None, :] - q[None, :, :]) ** 2).sum(axis=2))
            worst = max(worst, float(d.min(axis=1).max()))
        return worst

    return max(directed(pa, pb), directed(pb, pa))


# -- what to annotate, read from the records -----------------------------------------


def _curve_patch_of(m: Measurements) -> dict[int, str]:
    out: dict[int, str] = {}
    for name, info in (m.patches or {}).items():
        for c in info.get("curves", []):
            out[int(c)] = name
    return out


def _heading_sense(h: float, axis: int = 0) -> str:
    """The sense along the fluid's longest axis, mesh2d's leg-table convention: T01's
    return heading 260 reads '(-x)', the way the LEGS block prints it, so the picture and
    the text say the same thing about the same leg."""
    comp = (math.cos(math.radians(h)), math.sin(math.radians(h)))[axis]
    along = "xy"[axis]
    if comp > 1e-9:
        return f"+{along}"
    if comp < -1e-9:
        return f"-{along}"
    return f"across {along}"


def _channel_width(records: list[dict]) -> float | None:
    for r in records:
        if r.get("kind") == "ports":
            return float(r["width"])
    return None


def _rows_of(plan: "Plan | None") -> dict[str, dict]:
    """Row name -> {"count", "step", "channels": [op names]}: which channel op each repeat
    copies (through an intersect or a fuse), so junction insets and the pitch dimension
    can be placed for every instance from the one recorded leg table."""
    if plan is None:
        return {}
    by_name = {op.get("name"): op for op in plan.ops}

    def channels(name: str, seen: set[str]) -> list[str]:
        op = by_name.get(name)
        if op is None or name in seen:
            return []
        seen.add(name)
        if op.get("op") == "channel":
            return [name]
        found: list[str] = []
        for key in ("target", "from"):
            if isinstance(op.get(key), str):
                found += channels(op[key], seen)
        for key in ("of", "take"):
            for n in op.get(key) or []:
                if isinstance(n, str):
                    found += channels(n, seen)
        return found

    rows: dict[str, dict] = {}
    for op in plan.ops:
        if op.get("op") != "repeat":
            continue
        info = dict(plan.instances.get(op["name"], {}))
        info.setdefault("count", op.get("count", 1))
        info.setdefault("step", tuple(op.get("step") or (0.0, 0.0)))
        info["channels"] = channels(op["target"], set())
        rows[op["name"]] = info
    return rows


def _instance_label(channel: str, rows: dict[str, dict]) -> tuple[str, list[tuple[float, float]]]:
    """('loops', [(0, 0), (14.66, 0), ...]) for a channel a Row copies, else (channel, [(0, 0)])."""
    for row, info in rows.items():
        if channel in info["channels"]:
            sx, sy = info["step"]
            return row, [(k * sx, k * sy) for k in range(int(info["count"]))]
    stem = channel
    for suffix in (".raw", "_raw"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem, [(0.0, 0.0)]


def _junction_windows(m: Measurements, plan: "Plan | None") -> list[tuple[str, tuple[float, float], float]]:
    """(title, centre, half width) for every inset: the measured junctions when the
    instruments recorded them, else the leg tables' leave points and landings (a `from`
    line, a `to` leg) shifted to every instance of a Row; then every lip and every fluid
    cusp among the vertices."""
    windows: list[tuple[str, tuple[float, float], float]] = []
    w_ref = float(m.reference_width or 1.0)
    rows = _rows_of(plan)
    widths = {name: (_channel_width(recs) or w_ref) for name, recs in (m.legs or {}).items()}
    if m.junctions:
        for j in m.junctions:
            width = widths.get(j.channel.split("[")[0], w_ref)
            what = "leave" if j.kind == "leave" else "landing"
            windows.append((f"{j.channel} {what} ({j.where[0]:.4g}, {j.where[1]:.4g})", tuple(j.where), 1.5 * width))
    else:
        for channel, records in (m.legs or {}).items():
            width = widths.get(channel, w_ref)
            label, shifts = _instance_label(channel, rows)
            many = len(shifts) > 1
            for k, (dx, dy) in enumerate(shifts):
                name = f"{label}[{k}]" if many else label
                for i, r in enumerate(records):
                    if r.get("kind") == "ports":
                        continue
                    if i == 0 and r.get("from_line"):
                        x, y = r["from"][0] + dx, r["from"][1] + dy
                        windows.append((f"{name} leave ({x:.4g}, {y:.4g})", (x, y), 1.5 * width))
                    if r.get("kind") == "to" and r.get("lands"):
                        x, y = r["lands"][0] + dx, r["lands"][1] + dy
                        windows.append((f"{name} landing ({x:.4g}, {y:.4g})", (x, y), 1.5 * width))
    for v in m.vertices or []:
        if v.kind == "solid_lip":
            windows.append((f"lip {v.solid_deg:.3g} deg ({v.at[0]:.4g}, {v.at[1]:.4g})", tuple(v.at), 1.5 * w_ref))
        elif v.kind == "fluid_cusp":
            windows.append((f"cusp {v.interior_deg:.3g} deg ({v.at[0]:.4g}, {v.at[1]:.4g})", tuple(v.at), 1.5 * w_ref))
    return windows


def _reference_shift(m: Measurements, reference: "ReferenceMatch") -> tuple[float, float]:
    """Translate the golden so the two inlet centres coincide; without an inlet on either
    side, the outlines' lower-left corners."""
    mine = None
    inlet = (m.patches or {}).get("inlet") or {}
    if inlet.get("midpoints"):
        mine = tuple(inlet["midpoints"][0])
    theirs = None
    ref_patches = (reference.measurements or {}).get("patches") or {}
    if isinstance(ref_patches.get("inlet"), dict) and ref_patches["inlet"].get("midpoints"):
        theirs = tuple(ref_patches["inlet"]["midpoints"][0])
    if mine is None or theirs is None:
        points = [p for loop in reference.outline for p in loop]
        if not points:
            return (0.0, 0.0)
        rb = _bounds(points)
        return (m.bounds[0] - rb[0], m.bounds[1] - rb[1])
    return (mine[0] - theirs[0], mine[1] - theirs[1])


def reference_distance(gmsh, face: int, m: Measurements, reference: "ReferenceMatch") -> float:
    """The symmetric Hausdorff distance between the built outline and the golden's,
    after the golden is translated so the inlet centres coincide (the number the caption
    and the REFERENCE line print; `cli build` stores it in result.json)."""
    span = max(m.extent) if m.extent else 1.0
    mine = outline_polyline(gmsh, face, max(span, 1e-9) / 500)
    dx, dy = _reference_shift(m, reference)
    theirs = [[(px + dx, py + dy) for px, py in loop] for loop in reference.outline]
    return hausdorff(mine, theirs)


def _caption(findings: list[Finding], table: ComplianceTable | None, hausdorff_d: float | None,
             reference: "ReferenceMatch | None") -> list[str]:
    lines: list[str] = []
    order = {"error": 0, "warn": 1, "info": 2}
    for f in sorted(findings, key=lambda f: order.get(f.level, 3)):
        lines += f.text().splitlines()
    if table is not None:
        failing = table.failing()
        if failing:
            lines.append(f"CLAIMS     {len(failing)} FAIL of {len(table.rows)}")
            for r in failing:
                lines.append(f"           {r.id:<4} {r.says[:34]:<34} {r.measured[:40]:<40} {r.expected[:16]:<16} FAIL")
    if reference is not None:
        if hausdorff_d is None:
            lines.append(f"REFERENCE  {reference.entry} ({reference.preset}): Hausdorff not measured")
        else:
            lines.append(f"REFERENCE  {reference.entry} ({reference.preset}): Hausdorff {hausdorff_d:.4g} from the golden outline")
    return lines


# -- the figure ---------------------------------------------------------------------


def _figure(gmsh, face: int, plan: "Plan | None", m: Measurements, findings: list[Finding],
            table: ComplianceTable | None, reference: "ReferenceMatch | None" = None,
            width_px: int = 1024):
    """The Figure and what was drawn on it: {"insets": titles, "ellipsis": bool, "marks":
    [(x, y, level, code)], "caption": lines, "reference_shift": (dx, dy), "hausdorff":
    float | None}. `draw` saves it; the tests read it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Polygon as MplPolygon

    mesh2d = _toolbox.load("mesh2d")
    curves = _face_curves(gmsh, face)
    curve_pts = {c: _curve_points(gmsh, c) for c in curves}
    all_pts = [p for pts in curve_pts.values() for p in pts]
    x0, y0, x1, y1 = _bounds(all_pts) if all_pts else m.bounds
    w_ext, h_ext = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    span = max(w_ext, h_ext)
    w_ref = float(m.reference_width or span / 10)
    tris = _triangles(gmsh, face, span)
    curve_patch = _curve_patch_of(m)
    colours = dict(mesh2d.COLOURS)
    extra = iter(EXTRA_COLOURS)
    for name in sorted(set(curve_patch.values())):
        if name not in colours:
            colours[name] = next(extra, "#9467bd")

    windows = _junction_windows(m, plan)
    shown_windows = windows[:MAX_INSETS]
    ellipsis = len(windows) > MAX_INSETS
    hd = None
    shift = (0.0, 0.0)
    if reference is not None:
        shift = _reference_shift(m, reference)
        hd = reference.hausdorff
        if hd is None and reference.outline:
            mine = outline_polyline(gmsh, face, span / 500)
            theirs = [[(px + shift[0], py + shift[1]) for px, py in loop] for loop in reference.outline]
            hd = hausdorff(mine, theirs)
    caption = _caption(findings, table, hd, reference)
    shown_caption = caption[:CAPTION_LINES]
    if len(caption) > CAPTION_LINES:
        shown_caption.append(f"... and {len(caption) - CAPTION_LINES} more lines in the report")

    # layout, inches: title strip, inset strip, the main axes (and the panel), the caption
    fig_w = width_px / DPI
    inset_h = 1.45 if shown_windows else 0.0
    caption_h = 0.12 + 0.135 * len(shown_caption)
    main_w_frac = 0.6 if reference is not None else 1.0
    main_h = min(6.5, max(1.8, (fig_w * main_w_frac - 1.0) * h_ext / w_ext * 1.25))
    top_h = 0.45
    fig_h = min(15.0, top_h + inset_h + main_h + 0.7 + caption_h)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
    main_bottom = (caption_h + 0.55) / fig_h
    main_height = main_h / fig_h
    ax = fig.add_axes([0.07, main_bottom, main_w_frac - 0.09, main_height])

    def paint(axes, lw: float):
        if tris:
            axes.add_collection(PolyCollection(tris, facecolors=FILL, edgecolors=FILL_EDGE, linewidths=0.2))
        for c, pts in curve_pts.items():
            name = curve_patch.get(c, "walls")
            axes.plot([p[0] for p in pts], [p[1] for p in pts], color=colours.get(name, "#9467bd"), lw=lw)

    paint(ax, 1.4)
    pad = 0.08 * span
    # the two dimension lines for the extent
    dim_y = y0 - 0.9 * pad
    ax.annotate("", xy=(x1, dim_y), xytext=(x0, dim_y),
                arrowprops=dict(arrowstyle="<->", color=ANNOTATION, lw=0.8, shrinkA=0, shrinkB=0))
    ax.text((x0 + x1) / 2, dim_y - 0.15 * pad, f"{w_ext:.4g}", ha="center", va="top", fontsize=7.5, color=ANNOTATION)
    dim_x = x0 - 0.9 * pad
    ax.annotate("", xy=(dim_x, y1), xytext=(dim_x, y0),
                arrowprops=dict(arrowstyle="<->", color=ANNOTATION, lw=0.8, shrinkA=0, shrinkB=0))
    ax.text(dim_x - 0.15 * pad, (y0 + y1) / 2, f"{h_ext:.4g}", ha="right", va="center", fontsize=7.5,
            color=ANNOTATION, rotation=90)

    rows = _rows_of(plan)
    axis = 0 if w_ext >= h_ext else 1
    # per leg: heading arrows, arc centres and radii
    for channel, records in (m.legs or {}).items():
        width = _channel_width(records) or w_ref
        for r in records:
            kind = r.get("kind")
            if kind in ("line", "to"):
                (fx, fy), (tx, ty) = r["from"], r["to"]
                mx, my = (fx + tx) / 2, (fy + ty) / 2
                h = float(r["heading"]) % 360
                dx, dy = math.cos(math.radians(h)), math.sin(math.radians(h))
                L = 0.45 * width
                ax.annotate("", xy=(mx + dx * L, my + dy * L), xytext=(mx - dx * L, my - dy * L),
                            arrowprops=dict(arrowstyle="->", color=ANNOTATION, lw=1.0))
                ax.text(mx - dy * 0.55 * width, my + dx * 0.55 * width, f"{h:.4g} deg ({_heading_sense(h, axis)})",
                        fontsize=6.5, color=ANNOTATION, ha="center", va="center")
            elif kind == "arc":
                cx, cy = r["centre"]
                fx, fy = r["from"]
                a0 = math.degrees(math.atan2(fy - cy, fx - cx))
                mid = math.radians(a0 + float(r["sweep"]) / 2)
                radius = float(r["radius"])
                px, py = cx + radius * math.cos(mid), cy + radius * math.sin(mid)
                ax.plot([cx], [cy], marker="o", ms=3, color=ANNOTATION)
                ax.plot([cx, px], [cy, py], color=ANNOTATION, lw=0.7, ls=":")
                outer = r.get("outer_radius", radius + width / 2)
                ax.text(px, py, f"r {radius:.4g} (outer {outer:.4g}) {abs(float(r['sweep'])):.4g} deg",
                        fontsize=6.5, color=ANNOTATION, ha="center", va="bottom")
        # a Row's pitch between its first two instances
        for row, info in rows.items():
            if channel not in info["channels"] or int(info["count"]) < 2:
                continue
            anchor = next((tuple(r["from"]) for r in records if r.get("kind") != "ports"), None)
            if anchor is None:
                continue
            sx, sy = info["step"]
            pitch = math.hypot(sx, sy)
            label = f"pitch {pitch:.4g}"
            fp, gap = info.get("footprint"), info.get("gap")
            if fp is not None and gap is not None:
                along = fp[0] if isinstance(fp, (list, tuple)) else fp
                label += f" = {float(along):.4g} + {float(gap):.4g}" + ("" if info.get("gap_declared") else " (solved)")
            yy = y1 + 0.35 * pad
            ax.annotate("", xy=(anchor[0] + sx, yy), xytext=(anchor[0], yy),
                        arrowprops=dict(arrowstyle="<->", color=ANNOTATION, lw=0.8, shrinkA=0, shrinkB=0))
            ax.text(anchor[0] + sx / 2, yy + 0.1 * pad, label, ha="center", va="bottom", fontsize=7, color=ANNOTATION)
    # a Disk's diameter, a Row refusal's footprint
    footprint_wanted = any(f.code == "E-ROW-FIT" for f in findings)
    for name, solved in ((plan.features if plan else {}) or {}).items():
        params = {**(solved.solved or {}), **(solved.params or {})}
        if solved.kind == "Disk" and params.get("centre") is not None:
            cx, cy = params["centre"]
            radius = params.get("radius") or (float(params.get("diameter") or 0) / 2)
            if radius:
                ax.plot([cx - radius, cx + radius], [cy, cy], color=ANNOTATION, lw=0.7, ls="--")
                ax.text(cx, cy, f"d {2 * radius:.4g}", fontsize=6.5, color=ANNOTATION, ha="center", va="bottom")
        if solved.kind == "Bypass" and footprint_wanted:
            box = _footprint_box(solved.solved or {}, m)
            if box:
                ax.add_patch(MplPolygon(box, closed=True, fill=False, ls="--", lw=0.9, color=ANNOTATION))
                lo, hi = solved.solved["footprint"]
                ax.text(box[3][0], box[3][1], f"footprint {float(hi) - float(lo):.4g}", fontsize=7,
                        color=ANNOTATION, ha="left", va="bottom")
    # findings: crosses with the code, and their own marks
    marks: list[tuple[float, float, str, str]] = []
    for f in findings:
        for d in f.draw or []:
            if "segment" in d:
                (ax_, ay_), (bx_, by_) = d["segment"]
                ax.plot([ax_, bx_], [ay_, by_], color=LEVEL_COLOURS.get(f.level, ANNOTATION), lw=1.2)
            elif "circle" in d:
                cx, cy, r = d["circle"]
                ax.add_patch(Circle((cx, cy), r, fill=False, color=LEVEL_COLOURS.get(f.level, ANNOTATION), lw=1.0))
            elif "ray" in d:
                ray = d["ray"]
                (px, py), (dx, dy) = ray.get("from", (0, 0)), ray.get("dir", (1, 0))
                L = float(ray.get("length", w_ref))
                ax.annotate("", xy=(px + dx * L, py + dy * L), xytext=(px, py),
                            arrowprops=dict(arrowstyle="->", color=LEVEL_COLOURS.get(f.level, ANNOTATION), lw=1.0))
        if f.where is None or f.level not in LEVEL_COLOURS:
            continue
        x, y = f.where
        ax.plot([x], [y], marker="x", ms=11, mew=2.2, color=LEVEL_COLOURS[f.level], zorder=6, gid=f"finding:{f.code}")
        ax.annotate(f.code, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=6.5,
                    color=LEVEL_COLOURS[f.level], bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=LEVEL_COLOURS[f.level], lw=0.6),
                    zorder=7)
        marks.append((x, y, f.level, f.code))
    # the golden outline, translated so the inlets coincide
    if reference is not None:
        for loop in reference.outline:
            xs = [p[0] + shift[0] for p in loop] + [loop[0][0] + shift[0]]
            ys = [p[1] + shift[1] for p in loop] + [loop[0][1] + shift[1]]
            ax.plot(xs, ys, color=GOLDEN, lw=1.0, ls="--", zorder=4, gid="reference-outline")
    ax.set_aspect("equal")
    ax.set_xlim(x0 - 1.6 * pad, x1 + 0.4 * pad)
    ax.set_ylim(y0 - 1.6 * pad, y1 + (0.9 if rows else 0.4) * pad)
    ax.grid(True, lw=0.3, alpha=0.6)
    ax.tick_params(labelsize=7)
    seen = {curve_patch.get(c, "walls") for c in curves}
    ax.legend([Line2D([0], [0], color=colours.get(n, "#9467bd"), lw=2) for n in sorted(seen)], sorted(seen),
              loc="upper right", fontsize=7, frameon=True)
    units = m.units or ""
    ax.set_title(f"extent {w_ext:.4g} x {h_ext:.4g} {units}   area {m.area:.4g} {units}2   islands {m.islands}",
                 fontsize=9, loc="left")

    # the reference panel to the right, at the same scale
    if reference is not None:
        ax_ref = fig.add_axes([main_w_frac + 0.02, main_bottom, 1.0 - main_w_frac - 0.04, main_height])
        png = GOLDEN_DIR / f"{reference.entry}.png"
        drawn_png = False
        if png.exists():
            try:
                ax_ref.imshow(plt.imread(str(png)))
                ax_ref.set_axis_off()
                drawn_png = True
            except (OSError, ValueError):
                drawn_png = False
        if not drawn_png:
            for loop in reference.outline:
                xs = [p[0] for p in loop] + [loop[0][0]]
                ys = [p[1] for p in loop] + [loop[0][1]]
                ax_ref.plot(xs, ys, color=GOLDEN, lw=1.0)
            ax_ref.set_aspect("equal")
            ax_ref.set_xlim(ax.get_xlim()[0] - shift[0], ax.get_xlim()[1] - shift[0])
            ax_ref.set_ylim(ax.get_ylim()[0] - shift[1], ax.get_ylim()[1] - shift[1])
            ax_ref.tick_params(labelsize=6)
        approved = f" -- approved {reference.approved_at}" if reference.approved_at else ""
        ax_ref.set_title(f"library: {reference.entry} ({reference.preset}){approved}", fontsize=8, loc="left")

    # the insets along the top edge
    titles: list[str] = []
    if shown_windows:
        n = len(shown_windows)
        gap = 0.012
        avail = 0.94 - gap * (n - 1)
        iw = avail / n
        ih = (inset_h - 0.35) / fig_h
        ib = 1.0 - (top_h + inset_h - 0.2) / fig_h
        for i, (title, (cx, cy), half) in enumerate(shown_windows):
            axi = fig.add_axes([0.03 + i * (iw + gap), ib, iw, ih])
            paint(axi, 1.0)
            for f in findings:
                if f.where is not None and f.level in LEVEL_COLOURS and abs(f.where[0] - cx) <= half and abs(f.where[1] - cy) <= half:
                    axi.plot([f.where[0]], [f.where[1]], marker="x", ms=8, mew=1.8, color=LEVEL_COLOURS[f.level], zorder=6)
            axi.set_xlim(cx - half, cx + half)
            axi.set_ylim(cy - half, cy + half)
            axi.set_aspect("equal")
            axi.set_xticks([])
            axi.set_yticks([])
            axi.set_title(title.replace(" (", "\n(", 1), fontsize=6, pad=2)
            titles.append(title)
        if ellipsis:
            fig.text(0.985, ib + ih / 2, "...", fontsize=10, ha="right", va="center", color=ANNOTATION)
            titles.append("...")
    if shown_caption:
        fig.text(0.01, 0.01, "\n".join(shown_caption), family="monospace", fontsize=7.5, va="bottom")
    meta = {"insets": titles, "ellipsis": ellipsis, "marks": marks, "caption": caption,
            "reference_shift": shift, "hausdorff": hd, "bounds": (x0, y0, x1, y1)}
    return fig, meta


def _footprint_box(solved: dict, m: Measurements) -> list[tuple[float, float]] | None:
    """The refused Bypass's footprint in the sketch frame: lo..hi along the wall's flow
    from its anchor, 0..height outward. The anchor is the first leg's start in the leg
    table (the compiled instance), the frame the solved `wall_line` carries."""
    fp, height, wall = solved.get("footprint"), solved.get("height"), solved.get("wall_line") or {}
    if not fp or height is None:
        return None
    flow = tuple(wall.get("flow") or (1.0, 0.0))
    outward = tuple(wall.get("outward") or (-flow[1], flow[0]))
    anchor = None
    for records in (m.legs or {}).values():
        first = next((r for r in records if r.get("kind") != "ports"), None)
        if first is not None:
            anchor = tuple(first["from"])
            break
    if anchor is None:
        return None
    lo, hi = float(fp[0]), float(fp[1])

    def at(u, v):
        return (anchor[0] + flow[0] * u + outward[0] * v, anchor[1] + flow[1] * u + outward[1] * v)

    return [at(lo, 0.0), at(hi, 0.0), at(hi, float(height)), at(lo, float(height))]


def draw(gmsh, face: int, plan: "Plan", m: Measurements, findings: list[Finding],
         table: ComplianceTable | None, out: Path, reference: "ReferenceMatch | None" = None,
         width_px: int = 1024) -> Path:
    """The annotated picture (matplotlib Agg, 1024 px wide, dpi 110). Main axes: the face
    filled by a coarse triangulation (Mesh.Algorithm 6, thrown away), every boundary curve
    coloured by patch (mesh2d.COLOURS plus one colour per named patch), equal aspect, light
    grid, two dimension lines for the extent. Per feature: a heading arrow at each leg's
    midpoint labelled in degrees ('260 deg (-x)'); an arc's centre dot and radius line
    labelled 'r 4.5 (outer 6)' and its sweep; a Row's pitch dimension between the first two
    instances labelled 'pitch 14.66 = 13.02 + 1.64 (solved)'; a Disk's diameter line; a Row
    refusal's footprint as a dashed box (the `partial` path). Findings: red crosses (error)
    / orange (warn) at `where` with the code in a label box; `draw` marks. Insets: an
    inset_axes window three widths across at every junction (a Bypass's leave and landing,
    every `lands_on` leg, every `start=(wall, u)` branch), every lip and every fluid cusp,
    tiled along the top edge, titled 'loops[1] landing (17.68, 1.5)', cap 8 then '...'.
    Reference panel: when given, a second axes to the right at the same scale with the
    golden PNG, titled 'library: tesla_valve (t01) -- approved <date>', and the golden
    outline polyline overlaid in grey on the main axes translated so the inlet centres
    coincide; the Hausdorff distance in the caption. Caption: the LINT block and the
    CLAIMS FAIL rows, monospace."""
    import matplotlib.pyplot as plt

    fig, _ = _figure(gmsh, face, plan, m, findings, table, reference, width_px)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(str(out), dpi=DPI)
    finally:
        plt.close(fig)
    return out


def draw_plain(gmsh, built, curve_patch, out: Path, title: str, caption, marks) -> Path:
    """mesh2d.preview by path, unchanged; `cli preview --plain`."""
    mesh2d = _toolbox.load("mesh2d")
    return mesh2d.preview(gmsh, built, curve_patch, Path(out), title, list(caption or []), list(marks or []))
