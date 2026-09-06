"""A runnable 2D OpenFOAM case from any closed planar outline, in one call -- drawn
and measured before it is meshed.

The case directory on the workspace is where a 2D geometry is usually built by hand:
a gmsh script written blind, run, read back as an error, rewritten. This script takes
the shape as a short composition of 2D primitives and booleans (or an x,y outline
file), builds the face with gmsh's OpenCASCADE kernel, draws it with its patches
coloured, measures it (extent, area, how many islands, every patch's edge count and
length), and only then meshes it in quads, extrudes it one cell thick into hexahedra,
names the edge patches, and writes the case with `frontAndBack` typed `empty` in the
mesh and in every field. `Allmesh` runs gmshToFoam and checkMesh.

    python3 mesh2d.py CASE --spec shape.json --scale 0.001 --cell 0.0003 --mesh --preview CASE/outline.png
    python3 mesh2d.py --spec shape.json --scale 0.001 --dry-run --preview outline.png   # picture + report only
    python3 mesh2d.py CASE --outline profile.dat --size 0.1 --aoa 4 --external --mesh   # a body in a flow box

THE SPEC is a JSON list of ops, or {"ops": [...], "patches": [...]}. Later ops name
earlier ones; the op named "body" is the shape, else the last one. Lengths are metres
unless --scale is given (0.001 for a spec in mm); angles are degrees, counter-clockwise
from +x.

  rect      origin|center [x,y], size [w,h], round r (optional rounded corners)
  disk      center, radius, ry (optional: an ellipse)
  annulus   center, r_inner, r_outer
  band      center, r_inner, r_outer, start, end           an arc of annulus, 0 < end-start <= 360
  polygon   points [[x,y], ...]  (3 or more)
  outline   file (csv / Selig .dat of x,y), size (scale to this width), aoa (degrees, leading edge up)
  channel   width, start [x,y], heading, path [{"line": L}, {"arc": {"radius": R, "angle": +-deg}}, ...]
            a constant-width passage swept along a centreline; an arc continues the
            leg before it tangentially (positive angle turns left). Any internal
            passage is this: a bypass, a serpentine, a manifold branch. A leg written
            {"line": {"to": "y:1.5"}} runs along its heading until it reaches that
            line and is cut flush there -- the way to bring a return leg back onto a
            wall without working out its length. A channel that leaves a wall at an
            angle says so with "from": "y:1.5" (the line it starts on) and is cut flush
            there too; without it, one corner of its square start sticks out of the wall.
            The first leg of a channel with "from" is a line or a "to" leg.
  fuse      of [names...]        cut  from, take [names...]        intersect  of [names...]
  translate target, by [dx,dy]   rotate  target, angle, about [x,y]   copy  target
  mirror    target, axis x|y, at (the line x=at or y=at, default 0), keep (also keep the original)
  repeat    target, count, step [dx,dy], angle (per copy), about        copies, fused together
  patches   [{"name": "inlet", "at": "x:min"}, {"name": "outlet", "at": "near:0,6"},
             {"name": "lid", "box": [x0,y0,x1,y1], "kind": "slip"}]
            names an edge by where it sits, before the automatic reading below: "at" is
            x:min, y:max, x:0.012 (every edge flat on that line) or near:x,y (the one
            edge whose middle is nearest that point -- how a U-bend's two ends, both on
            x=0, are told apart); a position or box is in the spec's own units, like
            the ops (--scale applies)

Without a `patches` entry the edges are read off the shape: for a passage, the edges
flat at the low end of its longest axis are the inlet, flat at the high end the outlet,
everything else `walls`. With --external the shape is a body and a flow box is put
round it (inlet/outlet/farfield/body). The two z faces are always `frontAndBack`, empty.

There is deliberately no worked example of a named shape here: an example written by
hand is a shape nobody measured, and one such example was copied into a wrong Tesla
valve by every author that read it. The report is the check instead, and it has three
parts. The measurements: extent, area, islands (enclosed inner loops), edges per patch.
The leg table: for every `channel`, each leg's start, end and absolute heading, each
arc's centre, radius and sweep, and where a `to` leg lands -- so "leaves at 25 degrees",
"outer radius 6", "returns against the flow" are read off numbers, not off the picture.
The checks: copies of a `repeat` that overlap or touch each other, a `to` leg that lands
off the body, a passage with other than one inlet and one outlet, the open end of a
channel read as a wall, and edges shorter than a third of the narrowest channel (the
notches and slivers that make bad cells). A check that fails is printed with its
coordinates, `!!` marks it, and the script exits 2 after the preview is drawn, so the
picture of the failure is still there to look at. The mesh is all hexahedra when
checkMesh's `hexahedra:` equals its `cells:`, and the summary says so, or says how many
prisms were left where a quad could not be made.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sibling(name: str):
    """A sibling script, loaded by path: the toolbox is a directory, not a package."""
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


case_gen = sibling("case_gen")
snappy_gen = sibling("snappy_gen")
cad_gen = sibling("cad_gen")

OPS = ("rect", "disk", "annulus", "band", "polygon", "outline", "channel",
       "fuse", "cut", "intersect", "translate", "rotate", "mirror", "copy", "repeat", "fillet")
KINDS = ("inlet", "outlet", "wall", "slip", "symmetry")
EMPTY = "frontAndBack"


# -- the spec: pure parsing, so a bad one stops before the kernel starts -----------


def vec2(value, what: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SystemExit(f"{what} must be [x, y]")
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        raise SystemExit(f"{what} must be two numbers") from None


def number(value, what: str, positive: bool = False) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise SystemExit(f"{what} must be a number") from None
    if positive and out <= 0:
        raise SystemExit(f"{what} must be positive")
    return out


def parse_where(text) -> tuple:
    """'x:min', 'y:max', 'x:0.012' -> (axis, 'min'|'max'|'at', value);
    'near:0,6' -> (None, 'near', (x, y)): the one edge whose middle is nearest that
    point -- the way to name an end when two ends sit on the same line (a U-bend)."""
    if not isinstance(text, str) or ":" not in text:
        raise SystemExit(f"a position is 'x:min', 'y:max', 'x:0.012' or 'near:x,y', not {text!r}")
    axis_name, _, where = text.partition(":")
    axis_name = axis_name.strip()
    if axis_name == "near":
        parts = [p for p in where.replace(";", ",").split(",") if p.strip()]
        if len(parts) != 2:
            raise SystemExit(f"'near' wants a point, near:x,y, not {text!r}")
        return None, "near", (number(parts[0], f"x in {text!r}"), number(parts[1], f"y in {text!r}"))
    axis = {"x": 0, "y": 1}.get(axis_name)
    if axis is None:
        raise SystemExit(f"the axis in {text!r} must be x or y (or 'near:x,y')")
    where = where.strip()
    if where in ("min", "max"):
        return axis, where, None
    return axis, "at", number(where, f"the position in {text!r}")


def parse_rules(rules) -> list[dict]:
    if rules is None:
        return []
    if not isinstance(rules, list):
        raise SystemExit('"patches" must be a list of {"name": ..., "at": ... | "box": ...}')
    out = []
    for i, rule in enumerate(rules):
        if not isinstance(rule, dict) or not isinstance(rule.get("name"), str) or not rule["name"]:
            raise SystemExit(f"patches entry {i} needs a name")
        name = rule["name"]
        if ("at" in rule) == ("box" in rule):
            raise SystemExit(f"patches entry {i} ({name}) needs one of \"at\" or \"box\"")
        kind = rule.get("kind")
        if kind is None:
            kind = name if name in ("inlet", "outlet") else "wall"
        if kind not in KINDS:
            raise SystemExit(f"patches entry {i} ({name}): kind must be one of {', '.join(KINDS)}")
        item = {"name": name, "kind": kind, "at": None, "box": None}
        if "at" in rule:
            item["at"] = parse_where(rule["at"])
        else:
            box = rule["box"]
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise SystemExit(f"patches entry {i} ({name}): box is [x0, y0, x1, y1]")
            item["box"] = tuple(number(v, f"box of {name}") for v in box)
        out.append(item)
    return out


def scale_rules(rules: list[dict], scale: float) -> list[dict]:
    """The spec's patch positions and boxes, in the units the shape is built in.

    A spec in mm with `"box": [-0.6, -1.2, 0.6, 1.2]` round its inlet was, before this,
    matched against the shape after --scale 0.001 -- a box of 1.2 m round a 38 mm
    serpentine, so every edge was the inlet. Rules share the ops' units."""
    if scale == 1.0:
        return rules
    out = []
    for rule in rules:
        item = dict(rule)
        if rule["at"] is not None and rule["at"][1] == "at":
            axis, where, value = rule["at"]
            item["at"] = (axis, where, value * scale)
        elif rule["at"] is not None and rule["at"][1] == "near":
            (px, py) = rule["at"][2]
            item["at"] = (None, "near", (px * scale, py * scale))
        if rule["box"] is not None:
            item["box"] = tuple(v * scale for v in rule["box"])
        out.append(item)
    return out


def parse_spec(payload) -> tuple[list[dict], list[dict]]:
    """Validate and normalise. Returns (ops, rules). Errors name the entry."""
    if isinstance(payload, dict):
        entries, rules = payload.get("ops"), payload.get("patches")
    elif isinstance(payload, list):
        entries, rules = payload, None
    else:
        raise SystemExit('the spec is a JSON list of ops, or {"ops": [...], "patches": [...]}')
    if not isinstance(entries, list) or not entries:
        raise SystemExit("the spec needs at least one op")
    ops: list[dict] = []
    names: set[str] = set()

    def ref(i, op, key, value):
        if not isinstance(value, str) or value not in names:
            raise SystemExit(f"spec entry {i} ({op}): {key} must name an earlier op, not {value!r}")
        return value

    for i, raw in enumerate(entries):
        if not isinstance(raw, dict) or not isinstance(raw.get("op"), str):
            raise SystemExit(f"spec entry {i} must be an object with a string \"op\"")
        kind = raw["op"]
        if kind not in OPS:
            raise SystemExit(f"spec entry {i}: no op called {kind!r} (have {', '.join(OPS)})")
        name = raw.get("name") or f"op{i}"
        if not isinstance(name, str):
            raise SystemExit(f"spec entry {i} ({kind}): name must be a string")
        if name in names:
            raise SystemExit(f"spec entry {i} ({kind}): the name {name!r} is already used")
        op: dict = {"op": kind, "name": name}
        w = f"spec entry {i} ({kind})"
        if kind == "rect":
            if ("origin" in raw) == ("center" in raw):
                raise SystemExit(f"{w}: one of origin or center")
            op["size"] = vec2(raw.get("size"), f"{w}: size")
            if op["size"][0] <= 0 or op["size"][1] <= 0:
                raise SystemExit(f"{w}: size must be positive")
            if "origin" in raw:
                op["origin"] = vec2(raw["origin"], f"{w}: origin")
            else:
                c = vec2(raw["center"], f"{w}: center")
                op["origin"] = (c[0] - op["size"][0] / 2, c[1] - op["size"][1] / 2)
            op["round"] = number(raw.get("round", 0.0), f"{w}: round")
            if op["round"] < 0:
                raise SystemExit(f"{w}: round must not be negative")
        elif kind == "disk":
            op["center"] = vec2(raw.get("center"), f"{w}: center")
            op["radius"] = number(raw.get("radius"), f"{w}: radius", positive=True)
            op["ry"] = number(raw["ry"], f"{w}: ry", positive=True) if "ry" in raw else op["radius"]
        elif kind in ("annulus", "band"):
            op["center"] = vec2(raw.get("center"), f"{w}: center")
            op["r_inner"] = number(raw.get("r_inner"), f"{w}: r_inner", positive=True)
            op["r_outer"] = number(raw.get("r_outer"), f"{w}: r_outer", positive=True)
            if op["r_inner"] >= op["r_outer"]:
                raise SystemExit(f"{w}: r_inner must be smaller than r_outer")
            if kind == "band":
                op["start"] = number(raw.get("start"), f"{w}: start")
                op["end"] = number(raw.get("end"), f"{w}: end")
                sweep = op["end"] - op["start"]
                if not 0 < sweep <= 360:
                    raise SystemExit(f"{w}: end - start must be between 0 and 360 degrees")
        elif kind == "polygon":
            pts = raw.get("points")
            if not isinstance(pts, list):
                raise SystemExit(f"{w}: points is a list of [x, y]")
            pts = case_gen.drop_repeats([vec2(p, f"{w}: a point") for p in pts])
            if len(pts) < 3:
                raise SystemExit(f"{w}: a polygon needs at least three distinct points")
            op["points"] = pts
        elif kind == "outline":
            if not isinstance(raw.get("file"), str):
                raise SystemExit(f"{w}: file is the path of an x,y outline")
            op["file"] = raw["file"]
            op["size"] = number(raw["size"], f"{w}: size", positive=True) if "size" in raw else None
            op["aoa"] = number(raw.get("aoa", 0.0), f"{w}: aoa")
        elif kind == "channel":
            op["width"] = number(raw.get("width"), f"{w}: width", positive=True)
            op["start"] = vec2(raw.get("start"), f"{w}: start")
            op["heading"] = number(raw.get("heading", 0.0), f"{w}: heading")
            op["from"] = None
            if raw.get("from") is not None:
                op["from"] = parse_where(raw["from"])
                if op["from"][1] != "at":
                    raise SystemExit(f"{w}: from is the line the channel starts on, x:0.05 or y:1.5")
            path = raw.get("path")
            if not isinstance(path, list) or not path:
                raise SystemExit(f"{w}: path is a non-empty list of {{\"line\": L}} / {{\"arc\": ...}}")
            legs = []
            for j, seg in enumerate(path):
                if isinstance(seg, dict) and "line" in seg and "arc" not in seg:
                    if isinstance(seg["line"], dict):
                        to = seg["line"].get("to")
                        if to is None:
                            raise SystemExit(f"{w}: path[{j}] line is a length, or {{\"to\": \"y:1.5\"}}")
                        legs.append({"to": parse_where(to)})
                    else:
                        legs.append({"line": number(seg["line"], f"{w}: path[{j}] line", positive=True)})
                elif isinstance(seg, dict) and "arc" in seg and isinstance(seg["arc"], dict):
                    arc = seg["arc"]
                    radius = number(arc.get("radius"), f"{w}: path[{j}] arc radius", positive=True)
                    angle = number(arc.get("angle"), f"{w}: path[{j}] arc angle")
                    if angle == 0 or abs(angle) > 360:
                        raise SystemExit(f"{w}: path[{j}] arc angle must be non-zero and within 360")
                    if radius <= op["width"] / 2:
                        raise SystemExit(f"{w}: path[{j}] arc radius must exceed half the width "
                                         f"({op['width'] / 2:g}), or the inner wall folds over")
                    legs.append({"arc": {"radius": radius, "angle": angle}})
                else:
                    raise SystemExit(f"{w}: path[{j}] is {{\"line\": L}} or {{\"arc\": {{\"radius\", \"angle\"}}}}")
            op["path"] = legs
        elif kind in ("fuse", "intersect"):
            of = raw.get("of")
            if not isinstance(of, list) or len(of) < 2:
                raise SystemExit(f"{w}: of needs two or more names")
            op["of"] = [ref(i, kind, "of", n) for n in of]
        elif kind == "cut":
            op["from"] = ref(i, kind, "from", raw.get("from"))
            take = raw.get("take")
            if not isinstance(take, list) or not take:
                raise SystemExit(f"{w}: take needs one or more names")
            op["take"] = [ref(i, kind, "take", n) for n in take]
        elif kind == "translate":
            op["target"] = ref(i, kind, "target", raw.get("target"))
            op["by"] = vec2(raw.get("by"), f"{w}: by")
        elif kind == "rotate":
            op["target"] = ref(i, kind, "target", raw.get("target"))
            op["angle"] = number(raw.get("angle"), f"{w}: angle")
            op["about"] = vec2(raw["about"], f"{w}: about") if "about" in raw else None
        elif kind == "mirror":
            op["target"] = ref(i, kind, "target", raw.get("target"))
            if raw.get("axis") not in ("x", "y"):
                raise SystemExit(f"{w}: axis must be x or y")
            op["axis"] = raw["axis"]
            op["at"] = number(raw.get("at", 0.0), f"{w}: at")
            op["keep"] = bool(raw.get("keep", False))
        elif kind == "copy":
            op["target"] = ref(i, kind, "target", raw.get("target"))
        elif kind == "repeat":
            op["target"] = ref(i, kind, "target", raw.get("target"))
            try:
                op["count"] = int(raw.get("count"))
            except (TypeError, ValueError):
                raise SystemExit(f"{w}: count must be an integer") from None
            if op["count"] < 2:
                raise SystemExit(f"{w}: count must be at least 2")
            op["step"] = vec2(raw["step"], f"{w}: step") if "step" in raw else (0.0, 0.0)
            op["angle"] = number(raw.get("angle", 0.0), f"{w}: angle")
            op["about"] = vec2(raw["about"], f"{w}: about") if "about" in raw else None
            if op["step"] == (0.0, 0.0) and op["angle"] == 0.0:
                raise SystemExit(f"{w}: a step or an angle, or the copies land on each other")
        elif kind == "fillet":
            op["target"] = ref(i, kind, "target", raw.get("target"))
            op["radius"] = number(raw.get("radius"), f"{w}: radius", positive=True)
        ops.append(op)
        names.add(name)
    return ops, parse_rules(rules)


def body_name(ops: list[dict]) -> str:
    return "body" if any(op["name"] == "body" for op in ops) else ops[-1]["name"]


# -- geometry helpers, pure ----------------------------------------------------------


def wedge_polygon(center, reach: float, start: float, end: float) -> list[tuple[float, float]]:
    """The centre, then points on a circle of `reach` from `start` to `end` degrees, at
    most 60 degrees apart, counter-clockwise: a sector that a band is cut from."""
    cx, cy = center
    pts = [(cx, cy)]
    n = max(2, int(math.ceil((end - start) / 60.0)) + 1)
    for i in range(n):
        a = math.radians(start + (end - start) * i / (n - 1))
        pts.append((cx + reach * math.cos(a), cy + reach * math.sin(a)))
    return pts


def outline_points(path: Path, size: float | None, aoa: float) -> list[tuple[float, float]]:
    """An x,y outline file as a closed counter-clockwise polygon, scaled to `size`
    across x if given and turned by -aoa (leading edge up, as the profile template)."""
    pts = case_gen.as_ccw(case_gen.drop_repeats([tuple(p) for p in case_gen.read_profile(path)]))
    if len(pts) < 3:
        raise SystemExit(f"{path} holds fewer than three distinct points")
    xs = [p[0] for p in pts]
    if size:
        span = max(xs) - min(xs)
        if span <= 0:
            raise SystemExit(f"{path} has no extent in x to scale")
        k = size / span
        pts = [(x * k, y * k) for x, y in pts]
    if aoa:
        a = math.radians(-aoa)
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        pts = [(cx + (x - cx) * math.cos(a) - (y - cy) * math.sin(a),
                cy + (x - cx) * math.sin(a) + (y - cy) * math.cos(a)) for x, y in pts]
    return pts


def longest_axis2d(extent) -> int:
    return 0 if extent[0] >= extent[1] else 1


def flat_at(bounds, axis: int, value: float, tol: float) -> bool:
    """The edge lies along the line axis=value (both ends within tol of it)."""
    return abs(bounds[axis] - value) < tol and abs(bounds[axis + 2] - value) < tol


def classify_edge(bounds, domain, axis: int, tol: float) -> str:
    """An internal passage: flat at the domain's low end along `axis` -> inlet, at its
    high end -> outlet, else a wall. An arc that only touches an end is a wall."""
    if flat_at(bounds, axis, domain[axis], tol):
        return "inlet"
    if flat_at(bounds, axis, domain[axis + 2], tol):
        return "outlet"
    return "walls"


def classify_external(bounds, box, tol: float) -> str:
    """A body in a flow box: the box's low-x side is the inlet, high-x the outlet, its
    y sides the farfield, and every other edge belongs to the body."""
    if flat_at(bounds, 0, box[0], tol):
        return "inlet"
    if flat_at(bounds, 0, box[2], tol):
        return "outlet"
    if flat_at(bounds, 1, box[1], tol) or flat_at(bounds, 1, box[3], tol):
        return "farfield"
    return "body"


def near_targets(rules: list[dict], centres: dict[int, tuple[float, float]]) -> dict[int, str]:
    """entity -> patch name for every `near` rule: the one entity whose centre is
    nearest the rule's point. Nearest, not within a tolerance: gmsh's bounding box of
    a circle is a little off its centre (0.09948 for a circle at 0.1), so a target
    derived from the curve never met the extruded surface within any tolerance, and
    a cylinder named `near` its centre came out as `walls` (study
    20260907-011928-d6c4). Called once for the curves and once for the lateral
    surfaces; each pass picks its own nearest."""
    out: dict[int, str] = {}
    for rule in rules:
        if rule["at"] is not None and rule["at"][1] == "near" and centres:
            px, py = rule["at"][2]
            nearest = min(centres, key=lambda tag: math.hypot(centres[tag][0] - px, centres[tag][1] - py))
            out.setdefault(nearest, rule["name"])
    return out


def edge_rule(bounds, rules: list[dict], domain, tol: float) -> str | None:
    """The first rule the edge satisfies, by name; None when none does. `near` rules
    are resolved by `near_targets` before this is asked."""
    for rule in rules:
        if rule["at"] is not None and rule["at"][1] == "near":
            continue
        if rule["at"] is not None:
            axis, where, value = rule["at"]
            if where == "min":
                value = domain[axis]
            elif where == "max":
                value = domain[axis + 2]
            if flat_at(bounds, axis, value, tol):
                return rule["name"]
        else:
            x0, y0, x1, y1 = rule["box"]
            if (x0 - tol <= bounds[0] and bounds[2] <= x1 + tol
                    and y0 - tol <= bounds[1] and bounds[3] <= y1 + tol):
                return rule["name"]
    return None


def patch_for_edge(bounds, domain, axis: int, tol: float, rules: list[dict],
                   external: bool) -> str:
    named = edge_rule(bounds, rules, domain, tol)
    if named:
        return named
    return classify_external(bounds, domain, tol) if external else classify_edge(bounds, domain, axis, tol)


def mesh_sizes2d(extent, opts) -> tuple[float, float, float]:
    """(cell, wall cell, thickness). The cell defaults to a thirtieth of the short
    side -- on a channel with loops, some ten cells across the channel -- the wall cell
    to the cell, and the one cell in z to the cell so it is not a sliver."""
    cell = float(opts.get("cell") or min(extent) / 30.0)
    wall = float(opts.get("wall_cell") or cell)
    thickness = float(opts.get("thickness") or cell)
    if cell <= 0 or wall <= 0 or thickness <= 0:
        raise SystemExit("--cell, --wall-cell and --thickness must be positive")
    return cell, wall, thickness


class Built2D:
    """The measured face -- what the report and the classification read.

    A plain class, not a dataclass: the toolbox is loaded by path, unregistered in
    sys.modules, and a dataclass with string annotations cannot resolve them there."""

    def __init__(self, face, bounds, extent, area, islands, curves, lengths, short, domain,
                 external=False):
        self.face = face
        self.bounds = tuple(bounds)     # x0, y0, x1, y1
        self.extent = tuple(extent)     # width, height
        self.area = float(area)
        self.islands = int(islands)
        self.curves = list(curves)      # curve tags
        self.lengths = dict(lengths)    # curve -> length
        self.short = list(short)        # (curve, length) shorter than 1e-3 of the span
        self.domain = tuple(domain)     # what classification runs against (the box, external)
        self.external = bool(external)


class Patches2D:
    """What case_gen's writers ask of a mesh: the patch list with a face count each,
    the one cell's thickness (forceCoeffs' Aref) and the smallest cell (deltaT)."""

    def __init__(self, faces: dict[str, int], thickness: float, smallest_cell: float):
        self.patch_faces = dict(faces)
        self.thickness = float(thickness)
        self.smallest_cell = float(smallest_cell)

    def patch_face_counts(self) -> dict[str, int]:
        return dict(self.patch_faces)


def build_roles(patches, rules: list[dict], axis: int, far: str, external: bool) -> dict:
    """name -> role, in the shape case_gen's field writers read. The inlet points along
    the passage's axis (or +x for a box); frontAndBack is `empty`, which every writer
    already knows how to write."""
    kinds = {rule["name"]: rule["kind"] for rule in rules}
    direction = (1.0, 0.0, 0.0) if (external or axis == 0) else (0.0, 1.0, 0.0)
    roles = {}
    for name in patches:
        kind = kinds.get(name)
        if name == EMPTY:
            roles[name] = {"kind": "empty"}
        elif kind == "inlet" or (kind is None and name == "inlet"):
            roles[name] = {"kind": "inlet", "direction": direction}
        elif kind == "outlet" or (kind is None and name == "outlet"):
            roles[name] = {"kind": "outlet"}
        elif kind in ("slip", "symmetry"):
            roles[name] = {"kind": kind}
        elif name == "farfield":
            roles[name] = {"kind": far}
        else:
            roles[name] = {"kind": "wall"}
    return roles


def wall_patches(roles: dict) -> list[str]:
    return [n for n, r in roles.items() if r.get("kind") == "wall"]


def empty_patches(roles: dict) -> list[str]:
    return [n for n, r in roles.items() if r.get("kind") == "empty"]


def allmesh(roles: dict, cores: int) -> str:
    """gmshToFoam names every patch `patch`; the walls are retyped `wall` and the two
    z faces `empty` before anything reads the mesh, then renumber and checkMesh."""
    lines = ["#!/bin/sh", "set -e", 'cd "$(dirname "$0")"',
             "gmshToFoam body.msh > log.gmshToFoam 2>&1"]
    for name in wall_patches(roles):
        lines.append(f"foamDictionary constant/polyMesh/boundary -entry entry0/{name}/type "
                     f"-set wall >> log.gmshToFoam 2>&1")
    for name in empty_patches(roles):
        lines.append(f"foamDictionary constant/polyMesh/boundary -entry entry0/{name}/type "
                     f"-set empty >> log.gmshToFoam 2>&1")
    lines += ["renumberMesh -overwrite > log.renumberMesh 2>&1",
              "checkMesh > log.checkMesh 2>&1 || true",
              "grep -E 'cells:|hexahedra:|prisms:|Mesh OK|Failed|non-orthogonality|skewness' log.checkMesh",
              ""]
    return "\n".join(lines)


def case_files(plan, flow, opts, model: str, study: str, external: bool) -> dict[str, str]:
    """Every file of the case. The schemes and solvers are case_gen's hex-tuned set:
    a recombined, extruded mesh is hexahedra, not the tetrahedra cad_gen detunes for."""
    solver = case_gen.SOLVERS.get(study, "")
    files = {
        "system/controlDict": case_gen.control_dict(plan, flow, opts, study),
        "system/fvSchemes": case_gen.fv_schemes(study, model),
        "system/fvSolution": case_gen.fv_solution(study, model),
        "system/decomposeParDict": snappy_gen.decompose_dict(int(opts["cores"])),
        "constant/transportProperties": case_gen.transport_properties(flow),
        "constant/turbulenceProperties": case_gen.turbulence_properties(model),
        "0/U": case_gen.field_U(plan, flow),
        "0/p": case_gen.field_p(plan),
        "Allmesh": allmesh(plan.roles, int(opts["cores"])),
        "Allrun": cad_gen.allrun(solver, int(opts["cores"])),
        "case.foam": "",
    }
    fields = case_gen.turbulence_fields(model)
    if fields:
        intensity = opts.get("turbulent_intensity")
        if intensity is None:
            intensity = case_gen.FREE_STREAM_INTENSITY if external else case_gen.DUCT_INTENSITY
        mixing = opts.get("mixing_length") or 0.07 * plan.length
        ratio = None
        if external and not opts.get("mixing_length"):
            ratio = opts.get("viscosity_ratio") or case_gen.FREE_STREAM_VISCOSITY_RATIO
        writers = {
            "k": lambda: case_gen.field_k(plan, flow, intensity),
            "omega": lambda: case_gen.field_omega(plan, flow, intensity, mixing, nu=flow.nu, ratio=ratio),
            "epsilon": lambda: case_gen.field_epsilon(plan, flow, intensity, mixing),
            "nuTilda": lambda: case_gen.field_nu_tilda(plan, flow),
            "gammaInt": lambda: case_gen.field_gamma_int(plan),
            "ReThetat": lambda: case_gen.field_re_theta(plan, intensity),
        }
        for name in fields:
            files[f"0/{name}"] = writers[name]()
        files["0/nut"] = case_gen.field_nut(plan, model)
    return files


def report_lines(built: Built2D, curve_patch: dict, source: str, legs: dict | None = None,
                 checks: list | None = None, axis: int = 0, gmsh=None) -> list[str]:
    """The measured description of the shape -- what a picture cannot say in numbers:
    the measurements, the leg table, the checks."""
    w, h = built.extent
    lines = [f"geometry   {source}",
             f"extent     {w:.4g} x {h:.4g} m   area {built.area:.4g} m2   "
             f"islands {built.islands} (inner boundary loops)   {len(built.curves)} edges"]
    per: dict[str, list[float]] = {}
    for c in built.curves:
        per.setdefault(curve_patch.get(c, "?"), []).append(built.lengths[c])
    order = case_gen.patch_order(per)
    lines.append("patches    " + ", ".join(
        f"{name} ({len(per[name])} edge{'s' if len(per[name]) != 1 else ''}, {sum(per[name]):.4g} m)"
        for name in order))
    if legs:
        lines += leg_lines(legs, axis)
    groups: dict[str, list] = {}
    for c, length in built.short:
        groups.setdefault(f"{length:.2g}", []).append(c)
    for key, cs in groups.items():
        where = ""
        if gmsh:
            spots = [curve_midpoint(gmsh, c) for c in cs[:4]]
            where = " at " + ", ".join(f"({x:.4g}, {y:.4g})" for x, y in spots) + (" ..." if len(cs) > 4 else "")
        plural = "s" if len(cs) != 1 else ""
        lines.append(f"!!         {len(cs)} edge{plural} of {key} m{where}: short against the narrowest "
                     "channel -- a sliver, a corner poking through a wall, or a leg shorter than the "
                     "channel is wide; the cells there will be poor")
    if checks:
        lines += check_lines(checks)
    return lines


# -- gmsh-bound ---------------------------------------------------------------------


def _polygon_face(occ, pts) -> int:
    p = [occ.addPoint(x, y, 0) for x, y in pts]
    lines = [occ.addLine(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]
    return occ.addPlaneSurface([occ.addCurveLoop(lines)])


def _band(occ, center, r_in, r_out, start, end) -> list[int]:
    cx, cy = center
    outer = occ.addDisk(cx, cy, 0, r_out, r_out)
    inner = occ.addDisk(cx, cy, 0, r_in, r_in)
    ring, _ = occ.cut([(2, outer)], [(2, inner)])
    if end - start >= 360 - 1e-9:
        return [t for d, t in ring]
    wedge = _polygon_face(occ, wedge_polygon(center, 2.5 * r_out, start, end))
    out, _ = occ.intersect(ring, [(2, wedge)])
    return [t for d, t in out]


def _leg(occ, x, y, heading, length, width) -> int:
    r = occ.addRectangle(x, y - width / 2, 0, length, width)
    occ.rotate([(2, r)], x, y, 0, 0, 0, 1, math.radians(heading))
    return r


def _half_plane(occ, axis: int, value: float, sign: float, reach: float) -> int:
    """Everything on the `sign` side of the line axis=value, as a face to cut with."""
    if axis == 1:
        y0 = value if sign > 0 else value - reach
        return occ.addRectangle(-reach, y0, 0, 2 * reach, reach)
    x0 = value if sign > 0 else value - reach
    return occ.addRectangle(x0, -reach, 0, reach, 2 * reach)


def _channel(occ, width, start, heading, path, legs: list | None = None,
             from_line=None) -> list[int]:
    """The centreline compiled to faces: a rect per line leg, a band per arc (its centre
    at the leg end plus R along the normal, so the arc continues the leg tangentially);
    all fused. Every transition is tangent by construction, so there is nothing to
    round at a joint: the joint disks an earlier version added there stuck out past the
    outer wall of every arc by a sliver and made twelve notches on one valve.

    Every leg is recorded into `legs` (start, end, absolute headings, an arc's centre and
    radius, where a `to` leg lands) -- the numbers a request states and a picture only
    suggests. A `to` leg is cut off flush at the line it runs to, and a channel given
    `from` (the line it starts on, a wall it leaves) is cut flush there too: a square
    cap on a leg that meets a wall at an angle has one corner outside the wall, a notch."""
    x, y = start
    h = heading
    parts: list[int] = []
    reach = 1e3 * (width + math.hypot(*start) + sum(
        (seg["line"] if isinstance(seg.get("line"), (int, float)) else 0)
        + (seg["arc"]["radius"] if "arc" in seg else 0) for seg in path) + 1.0)
    for i, seg in enumerate(path):
        if "line" in seg or "to" in seg:
            record = {"kind": "line", "from": (x, y), "heading": h}
            back = 0.0
            if i == 0 and from_line is not None:
                axis, _, value = from_line
                d = (math.cos(math.radians(h)), math.sin(math.radians(h)))[axis]
                if abs(d) < 1e-9:
                    raise SystemExit(f"a channel heading {h:g} degrees runs along {'xy'[axis]}={value:g}, "
                                     "so it cannot start from it")
                back = width / abs(d)
                record["from_line"] = f"{'xy'[axis]}={value:g}"
            if "to" in seg:
                # Run along the heading until the line x=v or y=v: this is the one
                # arithmetic a passage's return leg keeps getting wrong when it is
                # typed as a length. Overshoot enough for both corners to cross the
                # line, then cut the leg flush at it.
                axis, kind, value = seg["to"]
                if kind != "at":
                    raise SystemExit("a leg's `to` is a coordinate, x:0.05 or y:1.5, not min/max")
                d = (math.cos(math.radians(h)), math.sin(math.radians(h)))[axis]
                if abs(d) < 1e-9:
                    raise SystemExit(f"a leg heading {h:g} degrees never reaches {'xy'[axis]}={value:g}")
                length = (value - (x, y)[axis]) / d
                if length <= 0:
                    raise SystemExit(f"a leg heading {h:g} degrees from ({x:g}, {y:g}) moves away from "
                                     f"{'xy'[axis]}={value:g}")
                lands = (x + length * math.cos(math.radians(h)), y + length * math.sin(math.radians(h)))
                x0, y0 = x - back * math.cos(math.radians(h)), y - back * math.sin(math.radians(h))
                leg = _leg(occ, x0, y0, h, back + length + width / abs(d), width)
                cut, _ = occ.cut([(2, leg)], [(2, _half_plane(occ, axis, value, math.copysign(1.0, d), reach))])
                parts += [t for dd, t in cut]
                record.update(kind="to", line=f"{'xy'[axis]}={value:g}", lands=lands, length=length)
                x, y = lands
            else:
                length = seg["line"]
                x0, y0 = x - back * math.cos(math.radians(h)), y - back * math.sin(math.radians(h))
                parts.append(_leg(occ, x0, y0, h, back + length, width))
                x += length * math.cos(math.radians(h))
                y += length * math.sin(math.radians(h))
                record["length"] = length
            if back:
                axis, _, value = from_line
                d = (math.cos(math.radians(h)), math.sin(math.radians(h)))[axis]
                cut, _ = occ.cut([(2, parts[-1])],
                                 [(2, _half_plane(occ, axis, value, -math.copysign(1.0, d), reach))])
                parts[-1:] = [t for dd, t in cut]
            record["to"] = (x, y)
            record["heading_out"] = h
        else:
            radius, angle = seg["arc"]["radius"], seg["arc"]["angle"]
            nx, ny = -math.sin(math.radians(h)), math.cos(math.radians(h))   # left normal
            if angle < 0:
                nx, ny = -nx, -ny
            cx, cy = x + radius * nx, y + radius * ny
            a0 = math.degrees(math.atan2(y - cy, x - cx))
            a1 = a0 + angle
            lo, hi = (a0, a1) if a1 > a0 else (a1, a0)
            parts += _band(occ, (cx, cy), radius - width / 2, radius + width / 2, lo, hi)
            record = {"kind": "arc", "from": (x, y), "heading": h, "centre": (cx, cy),
                      "radius": radius, "sweep": angle,
                      "outer_radius": radius + width / 2, "inner_radius": radius - width / 2}
            h += angle
            x, y = cx + radius * math.cos(math.radians(a1)), cy + radius * math.sin(math.radians(a1))
            record.update(to=(x, y), heading_out=h)
        if legs is not None:
            legs.append(record)
    tags = [(2, t) for t in parts]
    if len(tags) > 1:
        tags, _ = occ.fuse([tags[0]], tags[1:])
    if legs is not None:
        legs.append({"kind": "ports", "start": tuple(start), "end": (x, y), "width": width,
                     "heading_in": heading, "heading_out": h})
    return [t for d, t in tags]


def _instances_apart(occ, gmsh, base, copies, name: str, checks: list[dict]) -> None:
    """Consecutive copies of a repeat must neither overlap nor touch: they are meant as
    separate instances, and a fuse would quietly absorb either. Measured, not assumed --
    the loops of one 'correct' Tesla valve overlapped by a quarter of their area and
    passed every count."""
    instances = [base] + [list(c) for c in copies]
    for k in range(len(instances) - 1):
        a = occ.copy(instances[k])
        b = occ.copy(instances[k + 1])
        common, _ = occ.intersect(a, b, removeObject=True, removeTool=True)
        occ.synchronize()
        area = sum(occ.getMass(2, t) for d, t in common if d == 2)
        if common:
            occ.remove(common, recursive=True)
            occ.synchronize()
        if area > 0:
            x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(2, instances[k][0][1])
            checks.append({"level": "error", "where": ((x0 + x1) / 2, (y0 + y1) / 2),
                           "what": f"repeat {name!r}: copies {k + 1} and {k + 2} overlap by {area:.4g} "
                                   f"(units squared); the step is smaller than a copy's footprint "
                                   f"({x1 - x0:.4g} x {y1 - y0:.4g}) plus a gap"})
            continue
        try:
            gap = occ.getDistance(2, instances[k][0][1], 2, instances[k + 1][0][1])[0]
        except Exception:  # noqa: BLE001 - an older kernel without getDistance
            gap = None
        if gap is not None and gap <= 1e-9:
            x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(2, instances[k][0][1])
            checks.append({"level": "error", "where": ((x0 + x1) / 2, (y0 + y1) / 2),
                           "what": f"repeat {name!r}: copies {k + 1} and {k + 2} touch (gap 0); two "
                                   f"walls meeting at a knife edge mesh into slivers"})
        elif gap is not None and k == 0:
            checks.append({"level": "info", "where": None,
                           "what": f"repeat {name!r}: gap between copies {gap:.4g}"})


def build_face(gmsh, ops: list[dict], scale: float = 1.0, rotate: float = 0.0,
               legs: dict | None = None, checks: list | None = None) -> int:
    """Run the ops through OpenCASCADE and return the one face that is the body.
    Everything else the ops made along the way is removed, so nothing but the body
    is meshed. `legs` (channel name -> its leg records) and `checks` (the lint
    findings) are filled in when given; both are in the spec's own units."""
    occ = gmsh.model.occ
    made: dict[str, list[int]] = {}
    legs = legs if legs is not None else {}
    checks = checks if checks is not None else []

    def tags(name):
        return [(2, t) for t in made[name]]

    for op in ops:
        kind, name = op["op"], op["name"]
        if kind == "rect":
            x, y = op["origin"]
            made[name] = [occ.addRectangle(x, y, 0, op["size"][0], op["size"][1],
                                           roundedRadius=op["round"])]
        elif kind == "disk":
            cx, cy = op["center"]
            made[name] = [occ.addDisk(cx, cy, 0, op["radius"], op["ry"])]
        elif kind == "annulus":
            made[name] = _band(occ, op["center"], op["r_inner"], op["r_outer"], 0.0, 360.0)
        elif kind == "band":
            made[name] = _band(occ, op["center"], op["r_inner"], op["r_outer"], op["start"], op["end"])
        elif kind == "polygon":
            made[name] = [_polygon_face(occ, case_gen.as_ccw(op["points"]))]
        elif kind == "outline":
            made[name] = [_polygon_face(occ, outline_points(Path(op["file"]), op["size"], op["aoa"]))]
        elif kind == "channel":
            legs[name] = []
            made[name] = _channel(occ, op["width"], op["start"], op["heading"], op["path"], legs[name],
                                  from_line=op.get("from"))
        elif kind == "fuse":
            first, rest = op["of"][0], op["of"][1:]
            out, _ = occ.fuse(tags(first), [t for n in rest for t in tags(n)])
            made[name] = [t for d, t in out if d == 2]
        elif kind == "cut":
            out, _ = occ.cut(tags(op["from"]), [t for n in op["take"] for t in tags(n)])
            made[name] = [t for d, t in out if d == 2]
        elif kind == "intersect":
            first, rest = op["of"][0], op["of"][1:]
            out, _ = occ.intersect(tags(first), [t for n in rest for t in tags(n)])
            made[name] = [t for d, t in out if d == 2]
        elif kind == "translate":
            occ.translate(tags(op["target"]), op["by"][0], op["by"][1], 0)
            made[name] = made[op["target"]]
        elif kind == "rotate":
            about = op["about"]
            if about is None:
                x0, y0, _, x1, y1, _ = occ.getBoundingBox(2, made[op["target"]][0])
                about = ((x0 + x1) / 2, (y0 + y1) / 2)
            occ.rotate(tags(op["target"]), about[0], about[1], 0, 0, 0, 1, math.radians(op["angle"]))
            made[name] = made[op["target"]]
        elif kind == "mirror":
            a, b, d = (1.0, 0.0, -op["at"]) if op["axis"] == "x" else (0.0, 1.0, -op["at"])
            if op["keep"]:
                copy = occ.copy(tags(op["target"]))
                occ.mirror(copy, a, b, 0, d)
                out, _ = occ.fuse(tags(op["target"]), copy)
                made[name] = [t for dd, t in out if dd == 2]
            else:
                occ.mirror(tags(op["target"]), a, b, 0, d)
                made[name] = made[op["target"]]
        elif kind == "copy":
            made[name] = [t for d, t in occ.copy(tags(op["target"]))]
        elif kind == "repeat":
            base = tags(op["target"])
            about = op["about"]
            if about is None and op["angle"]:
                x0, y0, _, x1, y1, _ = occ.getBoundingBox(2, made[op["target"]][0])
                about = ((x0 + x1) / 2, (y0 + y1) / 2)
            copies = []
            for k in range(1, op["count"]):
                c = occ.copy(base)
                if op["step"] != (0.0, 0.0):
                    occ.translate(c, k * op["step"][0], k * op["step"][1], 0)
                if op["angle"]:
                    occ.rotate(c, about[0], about[1], 0, 0, 0, 1, math.radians(k * op["angle"]))
                copies.append(c)
            occ.synchronize()
            _instances_apart(occ, gmsh, base, copies, name, checks)
            out, _ = occ.fuse(base, [t for c in copies for t in c])
            made[name] = [t for d, t in out if d == 2]
        elif kind == "fillet":
            raise SystemExit(f"op {name!r}: fillet is not in this version; a rounded corner is "
                             "`rect` with `round`, and a rounded passage is a `channel`")
    body = body_name(ops)
    if len(made[body]) != 1:
        raise SystemExit(f"the body ({body!r}) is {len(made[body])} separate faces: the parts do "
                         "not touch. A fuse joins parts that overlap; a repeat's copies must "
                         "each reach the part they are meant to join")
    face = made[body][0]
    if scale != 1.0:
        occ.dilate([(2, face)], 0, 0, 0, scale, scale, scale)
    if rotate:
        occ.rotate([(2, face)], 0, 0, 0, 0, 0, 1, math.radians(rotate))
    occ.removeAllDuplicates()
    occ.synchronize()
    strays = [(d, t) for d, t in gmsh.model.getEntities(2) if t != face]
    if strays:
        occ.remove(strays, recursive=True)
        occ.synchronize()
    return face


def measure2d(gmsh, face: int, domain=None, external: bool = False,
              short_below: float | None = None) -> Built2D:
    """`short_below`: an edge shorter than this is a sliver or a notch. The caller
    passes a third of the narrowest channel when there is one; a thousandth of the
    span (the old rule) missed every notch on a 3 mm channel."""
    occ = gmsh.model.occ
    x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(2, face)
    bounds = (x0, y0, x1, y1)
    extent = (x1 - x0, y1 - y0)
    span = max(extent)
    loops, _ = occ.getCurveLoops(face)
    curves = [abs(t) for d, t in gmsh.model.getBoundary([(2, face)], combined=False, oriented=False)]
    lengths = {c: occ.getMass(1, c) for c in curves}
    threshold = short_below if short_below else 1e-3 * span
    short = [(c, length) for c, length in lengths.items() if length < threshold]
    return Built2D(face, bounds, extent, occ.getMass(2, face), len(loops) - 1, curves, lengths,
                   short, domain or bounds, external)


def narrowest_channel(ops: list[dict]) -> float | None:
    widths = [op["width"] for op in ops if op["op"] == "channel"]
    return min(widths) if widths else None


def curve_midpoint(gmsh, curve: int) -> tuple[float, float]:
    x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(1, curve)
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def landing_checks(gmsh, face: int, legs: dict, scale: float, checks: list) -> None:
    """A `to` leg lands on the body or it is an error: a point a tenth of a width past
    the line it ran to, along its heading, has to be inside the face."""
    for name, records in legs.items():
        width = next((r["width"] for r in records if r["kind"] == "ports"), 0.0)
        for r in records:
            if r["kind"] != "to":
                continue
            h = math.radians(r["heading"])
            px = (r["lands"][0] + 0.1 * width * math.cos(h)) * scale
            py = (r["lands"][1] + 0.1 * width * math.sin(h)) * scale
            if not gmsh.model.isInside(2, face, [px, py, 0.0]):
                checks.append({"level": "error", "where": (px, py),
                               "what": f"channel {name!r}: the leg to {r['line']} lands at "
                                       f"({r['lands'][0]:.4g}, {r['lands'][1]:.4g}) and nothing of the body "
                                       f"is there; it runs off the part it was meant to join"})


def port_checks(gmsh, built: Built2D, curve_patch: dict, legs: dict, scale: float,
                checks: list) -> None:
    """A channel's open end that the classification read as a wall is almost always a
    missing inlet or outlet (the U-bend whose both ends sit on the same side, the
    serpentine that ends where it began)."""
    tol = max(built.extent) * 1e-3 + 1e-12
    for name, records in legs.items():
        ports = next((r for r in records if r["kind"] == "ports"), None)
        if ports is None:
            continue
        for label, point in (("start", ports["start"]), ("end", ports["end"])):
            px, py = point[0] * scale, point[1] * scale
            for c in built.curves:
                mx, my = curve_midpoint(gmsh, c)
                if math.hypot(mx - px, my - py) < tol and abs(built.lengths[c] - ports["width"] * scale) < tol:
                    if curve_patch.get(c) == "walls":
                        checks.append({"level": "warn", "where": (px, py),
                                       "what": f"channel {name!r}: its open {label} at ({point[0]:.4g}, "
                                               f"{point[1]:.4g}) was read as a wall; if it is the inlet or "
                                               f"the outlet, name it (patches, --inlet, --outlet)"})


def count_checks(gmsh, built: Built2D, curve_patch: dict, rules: list[dict], checks: list) -> None:
    """A passage has one inlet edge and one outlet edge. Two inlets is a U-bend read
    off its bounding box; none is an end that is not flat to the axis. Either was a
    note before, and the case was written anyway."""
    if built.external:
        return
    kinds = {r["name"]: r["kind"] for r in rules}
    for want in ("inlet", "outlet"):
        names = [n for n, k in kinds.items() if k == want] or [want]
        edges = [c for c, n in curve_patch.items() if n in names]
        if len(edges) == 1:
            continue
        where = [curve_midpoint(gmsh, c) for c in edges]
        spots = ", ".join(f"({x:.4g}, {y:.4g})" for x, y in where) or "none"
        end = "low" if want == "inlet" else "high"
        checks.append({"level": "error", "where": where[0] if where else None,
                       "what": f"{len(edges)} {want} edges (at {spots}); a passage has exactly one. "
                               f"The automatic reading takes the edges flat at the {end} end of the "
                               f"longest axis; --{want} x:min / y:0.05 or a patches rule names the right one"})


def leg_lines(legs: dict, axis: int) -> list[str]:
    """The leg table: the numbers a request states, read off what was built."""
    lines = []
    along = "xy"[axis]
    for name, records in legs.items():
        lines.append(f"channel    {name!r} (spec units)")
        for i, r in enumerate(records):
            if r["kind"] == "ports":
                continue
            h = r["heading"] % 360
            if r["kind"] == "arc":
                turn = "left" if r["sweep"] > 0 else "right"
                lines.append(f"  leg {i + 1}  arc  r {r['radius']:.4g} (outer {r['outer_radius']:.4g}, inner "
                             f"{r['inner_radius']:.4g}) centre ({r['centre'][0]:.4g}, {r['centre'][1]:.4g}) "
                             f"{abs(r['sweep']):.4g} deg {turn}: heading {h:.4g} -> {r['heading_out'] % 360:.4g}")
            else:
                dx, dy = math.cos(math.radians(h)), math.sin(math.radians(h))
                comp = dx if axis == 0 else dy
                sense = f"+{along}" if comp > 1e-9 else (f"-{along}" if comp < -1e-9 else f"across {along}")
                what = f"to {r['line']}" if r["kind"] == "to" else f"line {r['length']:.4g}"
                lines.append(f"  leg {i + 1}  {what}  from ({r['from'][0]:.4g}, {r['from'][1]:.4g}) heading "
                             f"{h:.4g} deg ({sense}) to ({r['to'][0]:.4g}, {r['to'][1]:.4g})")
    return lines


def check_lines(checks: list) -> list[str]:
    lines = []
    for c in checks:
        mark = {"error": "!! ERROR", "warn": "!!", "info": "check"}[c["level"]]
        lines.append(f"{mark:<10} {c['what']}")
    return lines


def has_errors(checks: list) -> bool:
    return any(c["level"] == "error" for c in checks)


def curve_bounds(gmsh, curve: int) -> tuple:
    x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(1, curve)
    return (x0, y0, x1, y1)


def classify_curves(gmsh, built: Built2D, axis: int, rules: list[dict]) -> dict[int, str]:
    tol = max(built.extent) * 1e-4 + 1e-12
    named = near_targets(rules, {c: curve_midpoint(gmsh, c) for c in built.curves})
    return {c: named.get(c) or patch_for_edge(curve_bounds(gmsh, c), built.domain, axis, tol, rules,
                                              built.external)
            for c in built.curves}


def external_face(gmsh, body: int, opts) -> tuple[int, tuple]:
    """The flow box minus the body, in body lengths; returns (face, box bounds)."""
    occ = gmsh.model.occ
    x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(2, body)
    L = x1 - x0
    bx0, by0 = x0 - float(opts["ahead"]) * L, y0 - float(opts["below"]) * L
    bx1, by1 = x1 + float(opts["behind"]) * L, y1 + float(opts["above"]) * L
    box = occ.addRectangle(bx0, by0, 0, bx1 - bx0, by1 - by0)
    out, _ = occ.cut([(2, box)], [(2, body)])
    faces = [t for d, t in out if d == 2]
    occ.removeAllDuplicates()
    occ.synchronize()
    if len(faces) != 1:
        raise SystemExit("the flow box minus the body is not one face: the body reaches or "
                         "leaves the box")
    return faces[0], (bx0, by0, bx1, by1)


COLOURS = {"inlet": "#1f77b4", "outlet": "#d62728", "walls": "#111111", "body": "#2ca02c",
           "farfield": "#7f7f7f"}


def preview(gmsh, built: Built2D, curve_patch: dict, out: Path, title: str,
            caption: list[str], marks: list | None = None) -> Path:
    """The outline, filled, with every edge coloured by the patch it was read as, and
    the measured report under it; every failed check is a red cross where it is, so the
    eye lands on the notch or the overlap and not on the four islands. A coarse triangle
    mesh is drawn and thrown away; the real mesh is made afterwards with its own options."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D

    span = max(built.extent)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", span / 60)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    ntags, coords, _ = gmsh.model.mesh.getNodes()
    pos = {int(n): (coords[3 * i], coords[3 * i + 1]) for i, n in enumerate(ntags)}
    w, h = built.extent
    fig_w = 12.0
    fig_h = max(3.0, min(9.0, fig_w * h / max(w, 1e-12) + 1.6))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _, enodes = gmsh.model.mesh.getElementsByType(2, built.face)
    if len(enodes):
        tris = [[pos[int(n)] for n in enodes[i:i + 3]] for i in range(0, len(enodes), 3)]
        ax.add_collection(PolyCollection(tris, facecolors="#d5d9e0", edgecolors="#c3c7ce", linewidths=0.2))
    seen = {}
    for c in built.curves:
        name = curve_patch.get(c, "?")
        colour = COLOURS.get(name, "#9467bd")
        seen[name] = colour
        _, en = gmsh.model.mesh.getElementsByType(1, c)
        for i in range(0, len(en), 2):
            a, b = pos[int(en[i])], pos[int(en[i + 1])]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=colour, lw=1.6)
    for x, y, level in (marks or []):
        colour = "#d62728" if level == "error" else "#ff7f0e"
        ax.plot([x], [y], marker="x", ms=11, mew=2.2, color=colour, zorder=5)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.grid(True, lw=0.3)
    ax.set_title(title)
    ax.legend([Line2D([0], [0], color=col, lw=2) for col in seen.values()], list(seen),
              loc="upper right", fontsize=8, frameon=True)
    shown = caption[:14] + ([f"... and {len(caption) - 14} more lines in the report"] if len(caption) > 14 else [])
    fig.text(0.01, 0.01, "\n".join(shown), family="monospace", fontsize=8, va="bottom")
    fig.subplots_adjust(bottom=min(0.6, 0.06 + 0.035 * len(shown)))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    gmsh.model.mesh.clear()
    return out


class MeshResult2D:
    def __init__(self, patches=None, hexes=0, prisms=0, tets=0, base_quads=0, base_tris=0,
                 nodes=0, seconds=0.0, volume=0):
        self.patches = dict(patches or {})   # name -> surface tags
        self.hexes, self.prisms, self.tets = int(hexes), int(prisms), int(tets)
        self.base_quads, self.base_tris = int(base_quads), int(base_tris)
        self.nodes, self.seconds, self.volume = int(nodes), float(seconds), int(volume)


def generate(gmsh, built: Built2D, axis: int, rules: list[dict], curve_patch: dict,
             cell: float, wall_cell: float, thickness: float, threads: int | None) -> MeshResult2D:
    """Quads on the face, one cell of extrusion, physical groups from the edges."""
    occ = gmsh.model.occ
    gmsh.model.mesh.clear()
    walls = [c for c, name in curve_patch.items() if name not in ("inlet", "outlet", "farfield")]
    fld = gmsh.model.mesh.field
    if walls:
        fld.add("Distance", 1)
        fld.setNumbers(1, "CurvesList", walls)
        fld.setNumber(1, "Sampling", 100)
        fld.add("Threshold", 2)
        fld.setNumber(2, "InField", 1)
        fld.setNumber(2, "SizeMin", wall_cell)
        fld.setNumber(2, "SizeMax", cell)
        fld.setNumber(2, "DistMin", 2 * wall_cell)
        fld.setNumber(2, "DistMax", 10 * wall_cell)
        fld.setAsBackgroundMesh(2)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)
    gmsh.option.setNumber("Mesh.MeshSizeMin", wall_cell * 0.5)
    gmsh.option.setNumber("Mesh.MeshSizeMax", cell)
    gmsh.option.setNumber("Mesh.Algorithm", 8)               # Frontal-Delaunay for quads
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)  # Blossom full-quad
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 0)
    gmsh.option.setNumber("Mesh.RecombineOptimizeTopology", 5)
    gmsh.option.setNumber("Mesh.Smoothing", 10)
    if threads:
        gmsh.option.setNumber("General.NumThreads", int(threads))
    gmsh.model.mesh.setRecombine(2, built.face)
    out = occ.extrude([(2, built.face)], 0, 0, thickness, numElements=[1], recombine=True)
    occ.synchronize()
    volume = [t for d, t in out if d == 3][0]
    tol = max(built.extent) * 1e-4 + 1e-12
    boxes = {s: gmsh.model.getBoundingBox(2, s)
             for d, s in gmsh.model.getBoundary([(3, volume)], combined=True, oriented=False)}
    lateral = {s: ((b[0] + b[3]) / 2, (b[1] + b[4]) / 2) for s, b in boxes.items() if abs(b[5] - b[2]) >= tol}
    named = near_targets(rules, lateral)
    patches: dict[str, list[int]] = {}
    for s, (x0, y0, z0, x1, y1, z1) in boxes.items():
        if abs(z1 - z0) < tol:
            name = EMPTY
        else:
            name = named.get(s) or patch_for_edge((x0, y0, x1, y1), built.domain, axis, tol, rules,
                                                  built.external)
        patches.setdefault(name, []).append(s)
    for name, tags in patches.items():
        g = gmsh.model.addPhysicalGroup(2, tags)
        gmsh.model.setPhysicalName(2, g, name)
    g = gmsh.model.addPhysicalGroup(3, [volume])
    gmsh.model.setPhysicalName(3, g, "fluid")
    t0 = time.time()
    gmsh.model.mesh.generate(3)
    seconds = time.time() - t0
    count = lambda kind, *tag: len(gmsh.model.mesh.getElementsByType(kind, *tag)[0])  # noqa: E731
    return MeshResult2D(
        patches=patches,
        hexes=count(5), prisms=count(6), tets=count(4),
        base_quads=count(3, built.face), base_tris=count(2, built.face),
        nodes=len(gmsh.model.mesh.getNodes()[0]), seconds=seconds, volume=volume,
    )


def summary(built: Built2D, curve_patch: dict, source: str, mesh: MeshResult2D | None,
            flow, model: str, why: str, study: str, sizes, length_note: str,
            notes: list[str], drawn: Path | None, legs: dict | None = None,
            checks: list | None = None, axis: int = 0, gmsh=None) -> list[str]:
    cell, wall_cell, thickness = sizes
    lines = report_lines(built, curve_patch, source, legs, checks, axis, gmsh)
    axis_name = "x" if built.domain[2] - built.domain[0] >= built.domain[3] - built.domain[1] else "y"
    if built.external:
        lines.append(f"domain     a flow box {built.domain[2] - built.domain[0]:.4g} x "
                     f"{built.domain[3] - built.domain[1]:.4g} m round the body; {length_note}")
    else:
        lines.append(f"domain     a passage along {axis_name}; {length_note}")
    lines.append(f"mesh       cell {cell:.4g} m, {wall_cell:.4g} m at the walls, one cell of "
                 f"{thickness:.4g} m in z (frontAndBack empty)")
    if mesh is not None:
        prisms = f" + {mesh.prisms} prisms !!" if mesh.prisms else ""
        tets = f" + {mesh.tets} tetrahedra !!" if mesh.tets else ""
        lines.append(f"cells      {mesh.hexes:,} hexahedra{prisms}{tets}   "
                     f"({mesh.base_quads:,} quads on the face"
                     + (f", {mesh.base_tris} triangles" if mesh.base_tris else "")
                     + f")   {mesh.nodes:,} nodes   {mesh.seconds:.1f} s")
        lines.append("patches    " + ", ".join(
            f"{n} ({len(mesh.patches[n])})" for n in case_gen.patch_order(mesh.patches)))
    if flow is not None:
        lines.append(f"flow       {flow.line()}")
        lines.append(f"turbulence {model}   ({why})")
        solver = case_gen.SOLVERS.get(study, "")
        lines.append(f"run        {study}" + (f"   {solver}" if solver else "   mesh only, no solver"))
    for note in notes:
        lines.append(f"note       {note}")
    if drawn is not None:
        lines.append(f"drawn to   {drawn}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", type=Path, nargs="?", help="Directory to write the case into.")
    src = ap.add_argument_group("the shape")
    src.add_argument("--spec", type=Path, default=None, help="JSON of ops (see above).")
    src.add_argument("--outline", type=Path, default=None, help="An x,y outline file (csv / .dat).")
    src.add_argument("--size", type=float, default=None, help="--outline: scale it to this width, m.")
    src.add_argument("--aoa", type=float, default=0.0, help="--outline: degrees of incidence.")
    src.add_argument("--scale", type=float, default=1.0, help="Multiply the shape by this (0.001 for mm).")
    src.add_argument("--rotate", type=float, default=0.0, help="Turn the shape, degrees about the origin.")
    dom = ap.add_argument_group("the domain")
    dom.add_argument("--external", action="store_true",
                     help="The shape is a body; put a flow box round it (inlet at -x).")
    dom.add_argument("--ahead", type=float, default=2.0, help="Box ahead of the body, in body lengths.")
    dom.add_argument("--behind", type=float, default=5.0)
    dom.add_argument("--above", type=float, default=2.0)
    dom.add_argument("--below", type=float, default=2.0)
    dom.add_argument("--far", default="slip", choices=["slip", "symmetry"])
    dom.add_argument("--along", default="auto", choices=["auto", "x", "y"],
                     help="A passage's axis (default: its longest).")
    dom.add_argument("--inlet", default=None, help="Where the inlet is, e.g. y:max (default: read off the shape).")
    dom.add_argument("--outlet", default=None, help="Where the outlet is, e.g. x:0.06.")
    msh = ap.add_argument_group("the mesh")
    msh.add_argument("--cell", type=float, default=None, help="Cell size, m (default: short side / 30).")
    msh.add_argument("--wall-cell", type=float, default=None, dest="wall_cell",
                     help="Cell size at the walls, m (default: the cell).")
    msh.add_argument("--thickness", type=float, default=None, help="The one cell in z, m (default: the cell).")
    msh.add_argument("--threads", type=int, default=None)
    flw = ap.add_argument_group("the flow")
    flw.add_argument("--speed", type=float, default=1.0)
    flw.add_argument("--nu", type=float, default=None, help="m2/s (default 1.5e-5, air).")
    flw.add_argument("--reynolds", type=float, default=None)
    flw.add_argument("--length", type=float, default=None,
                     help="Reference length for Re (default: a passage's hydraulic diameter 4A/P, "
                          "a body's x extent).")
    flw.add_argument("--turbulence", default="auto",
                     choices=["auto", "laminar", "kOmegaSST", "kOmegaSSTLM", "kEpsilon", "SpalartAllmaras"])
    flw.add_argument("--turbulent-intensity", type=float, default=None, dest="turbulent_intensity")
    flw.add_argument("--viscosity-ratio", type=float, default=None, dest="viscosity_ratio")
    flw.add_argument("--mixing-length", type=float, default=None, dest="mixing_length")
    run = ap.add_argument_group("the run")
    run.add_argument("--study", default="steady", choices=["mesh", "steady", "transient"])
    run.add_argument("--iterations", type=int, default=1000)
    run.add_argument("--end-time", type=float, default=1.0, dest="end_time")
    run.add_argument("--delta-t", type=float, default=None, dest="delta_t")
    run.add_argument("--courant", type=float, default=0.9)
    run.add_argument("--writes", type=int, default=10)
    run.add_argument("--cores", type=int, default=4)
    run.add_argument("--purge", type=int, default=None, help="Keep only the last N time directories.")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="Build, measure and report; mesh and write nothing.")
    ap.add_argument("--preview", type=Path, default=None, help="Draw the outline into this PNG.")
    ap.add_argument("--mesh", action="store_true", help="After writing, run Allmesh here.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if (args.spec is None) == (args.outline is None):
        ap.error("one of --spec or --outline")
    if args.case is None and not args.dry_run:
        ap.error("a directory to write the case into, or --dry-run")
    opts = vars(args)
    opts["body_patch"] = "body"
    if opts["nu"] is None and opts["reynolds"] is None:
        opts["nu"] = 1.5e-5
    notes: list[str] = []

    if args.spec is not None:
        ops, rules = parse_spec(json.loads(args.spec.read_text(encoding="utf-8")))
        rules = scale_rules(rules, args.scale)  # the spec's rules, in the spec's units
        source = f"{args.spec} ({len(ops)} ops)"
    else:
        ops = [{"op": "outline", "name": "body", "file": str(args.outline), "size": args.size, "aoa": args.aoa}]
        rules = []
        source = f"{args.outline}" + (f" scaled to {args.size:g} m" if args.size else "")
    for flag in ("inlet", "outlet"):
        if opts.get(flag):
            rules.insert(0, {"name": flag, "kind": flag, "at": parse_where(opts[flag]), "box": None})

    try:
        import gmsh
    except ImportError:
        raise SystemExit("the gmsh Python module is not importable here; on the instance it is, "
                         "and `python3 -c 'import gmsh'` is the check")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if args.verbose else 0)
    gmsh.model.add("mesh2d")
    try:
        legs: dict = {}
        checks: list = []
        body = build_face(gmsh, ops, scale=args.scale, rotate=args.rotate, legs=legs, checks=checks)
        for c in checks:          # found while building, before the scale was applied
            if c.get("where"):
                c["where"] = (c["where"][0] * args.scale, c["where"][1] * args.scale)
        narrow = narrowest_channel(ops)
        short_below = narrow * args.scale / 3.0 if narrow else None
        external = bool(args.external)
        if external:
            face, box = external_face(gmsh, body, opts)
            built = measure2d(gmsh, face, domain=box, external=True, short_below=short_below)
        else:
            built = measure2d(gmsh, body, short_below=short_below)
        axis = {"auto": longest_axis2d(built.extent), "x": 0, "y": 1}[args.along]
        curve_patch = classify_curves(gmsh, built, axis, rules)
        if not external:
            landing_checks(gmsh, built.face, legs, args.scale, checks)
            port_checks(gmsh, built, curve_patch, legs, args.scale, checks)
            count_checks(gmsh, built, curve_patch, rules, checks)
        sizes = mesh_sizes2d(built.extent, opts)
        cell, wall_cell, thickness = sizes
        if external:
            bx0, _, bx1, _ = built.bounds
            length = float(opts.get("length") or (bx1 - bx0))
            length_note = f"reference length {length:.4g} m (the body's x extent)"
        else:
            walls_len = sum(built.lengths[c] for c, n in curve_patch.items() if n not in ("inlet", "outlet"))
            dh = 4.0 * built.area / walls_len if walls_len > 0 else max(built.extent)
            length = float(opts.get("length") or dh)
            length_note = f"reference length {length:.4g} m (hydraulic diameter 4A/P of the walls)"
        drawn = None
        if args.preview is not None:
            caption = report_lines(built, curve_patch, source, legs, checks, axis, gmsh)
            marks = [(*c["where"], c["level"]) for c in checks if c.get("where") and c["level"] != "info"]
            marks += [(*curve_midpoint(gmsh, c), "warn") for c, _ in built.short]
            drawn = preview(gmsh, built, curve_patch, args.preview, Path(source.split(" ")[0]).name,
                            caption, marks)

        flow = model = why = None
        if args.study != "mesh" or not args.dry_run:
            flow = case_gen.derive_flow(opts, length)
            model, why = case_gen.turbulence_model(opts, flow)

        if args.dry_run or has_errors(checks):
            for line in summary(built, curve_patch, source, None, flow, model or "", why or "",
                                args.study, sizes, length_note, notes, drawn, legs, checks, axis, gmsh):
                print(line)
            if has_errors(checks):
                print(f"not written: {sum(1 for c in checks if c['level'] == 'error')} check(s) failed "
                      "(the !! ERROR lines above, and the red crosses on the preview)")
                return 2
            return 0

        target: Path = args.case
        if (target / "Allmesh").exists() and not args.force:
            raise SystemExit(f"{target} already holds a case; --force to write over it")
        target.mkdir(parents=True, exist_ok=True)
        (target / "constant" / "geometry").mkdir(parents=True, exist_ok=True)
        if args.spec is not None:
            # The record is the spec as written, with the scale and rotation it was
            # built at: the same file rebuilds the case, which the normalised ops (tuples,
            # resolved `to` legs) could not.
            raw = json.loads(args.spec.read_text(encoding="utf-8"))
            record = dict(raw) if isinstance(raw, dict) else {"ops": raw}
            record.update(scale=args.scale, rotate=args.rotate)
            (target / "constant" / "geometry" / "body.json").write_text(
                json.dumps(record, indent=1), encoding="utf-8")
        with cad_gen.quiet_fd1():
            gmsh.write(str(target / "constant" / "geometry" / "body.step"))

        mesh = generate(gmsh, built, axis, rules, curve_patch, cell, wall_cell, thickness, args.threads)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.write(str(target / "body.msh"))
        if mesh.tets:
            notes.append(f"!! {mesh.tets} tetrahedra in an extruded mesh means the extrusion did not "
                         "take; the case is not the one asked for")
        if mesh.prisms:
            notes.append(f"{mesh.prisms} prisms where a quad could not be made; checkMesh's "
                         "hexahedra count will fall short of its cells by that many")

        patches = Patches2D({name: len(tags) for name, tags in mesh.patches.items()}, thickness, wall_cell)
        roles = build_roles(patches.patch_faces, rules, axis, args.far, external)
        plan = case_gen.Plan(patches, roles, length, {"source": source})
        files = case_files(plan, flow, opts, model, args.study, external)
        written = case_gen.write_case(target, files)
        (target / "Allmesh").chmod(0o755)
        (target / "Allrun").chmod(0o755)

        for line in summary(built, curve_patch, source, mesh, flow, model, why, args.study, sizes,
                            length_note, notes, drawn, legs, checks, axis, gmsh):
            print(line)
        print(f"wrote {len(written)} files and body.msh into {target}")
        print(f"next: sh {target}/Allmesh   (gmshToFoam, retype walls and frontAndBack, checkMesh)")
    finally:
        gmsh.finalize()

    if args.mesh:
        if shutil.which("gmshToFoam") is None:
            print("Allmesh not run: gmshToFoam is not on PATH here")
            return 0
        proc = subprocess.run(["sh", "./Allmesh"], cwd=str(target), capture_output=True, text=True)
        print(proc.stdout[-3000:], end="")
        if proc.returncode != 0:
            print(f"Allmesh exited {proc.returncode}; log.gmshToFoam and log.checkMesh say why")
            print(proc.stderr[-2000:], end="")
            return 1
        log = target / "log.checkMesh"
        if log.exists():
            digest = sibling("mesh_digest")
            for line in digest.report(digest.parse(log.read_text(encoding="utf-8", errors="replace"))).splitlines():
                print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
