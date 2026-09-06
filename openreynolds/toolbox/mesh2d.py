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
            line (and half a width past it, to join what it meets) -- the way to bring
            a return leg back onto a wall without working out its length.
  fuse      of [names...]        cut  from, take [names...]        intersect  of [names...]
  translate target, by [dx,dy]   rotate  target, angle, about [x,y]   copy  target
  mirror    target, axis x|y, at (the line x=at or y=at, default 0), keep (also keep the original)
  repeat    target, count, step [dx,dy], angle (per copy), about        copies, fused together
  patches   [{"name": "inlet", "at": "x:min"}, {"name": "lid", "box": [x0,y0,x1,y1], "kind": "slip"}]
            names an edge by where it sits, before the automatic reading below; a
            position or box is in the spec's own units, like the ops (--scale applies)

Without a `patches` entry the edges are read off the shape: for a passage, the edges
flat at the low end of its longest axis are the inlet, flat at the high end the outlet,
everything else `walls`. With --external the shape is a body and a flow box is put
round it (inlet/outlet/farfield/body). The two z faces are always `frontAndBack`, empty.

A Tesla valve is one rect, one channel and a repeat -- four bypasses that leave the
channel steeply, sweep round, and come back down onto the channel's top wall (y=1.5)
against the -x direction, each leaving a teardrop island (mm; run with --scale 0.001).
The return leg is written `to` the wall rather than as a length, so it always lands:

  {"ops": [
    {"op": "rect", "name": "main", "origin": [0, -1.5], "size": [60, 3]},
    {"op": "channel", "name": "bypass", "width": 3, "start": [9, 1.5], "heading": 60,
     "path": [{"line": 2}, {"arc": {"radius": 3.5, "angle": 240}}, {"line": {"to": "y:1.5"}}]},
    {"op": "repeat", "name": "bypasses", "target": "bypass", "count": 4, "step": [12, 0]},
    {"op": "fuse", "name": "body", "of": ["main", "bypasses"]}],
   "patches": [{"name": "inlet", "at": "x:min"}, {"name": "outlet", "at": "x:max"}]}

The report it prints -- extent, area, islands, edges per patch -- is the check that the
shape is the one that was meant, made before a mesh exists; `islands 4` is what four
enclosed bypasses look like as a number. The mesh is all hexahedra when checkMesh's
`hexahedra:` equals its `cells:`, and the summary says so, or says how many prisms
were left where a quad could not be made.
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


def parse_where(text) -> tuple[int, str, float | None]:
    """'x:min', 'y:max' or 'x:0.012' -> (axis, 'min'|'max'|'at', value)."""
    if not isinstance(text, str) or ":" not in text:
        raise SystemExit(f"a position is 'x:min', 'y:max' or 'x:0.012', not {text!r}")
    axis_name, _, where = text.partition(":")
    axis = {"x": 0, "y": 1}.get(axis_name.strip())
    if axis is None:
        raise SystemExit(f"the axis in {text!r} must be x or y")
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


def edge_rule(bounds, rules: list[dict], domain, tol: float) -> str | None:
    """The first rule the edge satisfies, by name; None when none does."""
    for rule in rules:
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


def report_lines(built: Built2D, curve_patch: dict, source: str) -> list[str]:
    """The measured description of the shape -- what a picture cannot say in numbers."""
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
    for c, length in built.short:
        lines.append(f"!! a {length:.3g} m edge (curve {c}); the quads there will be poor -- "
                     "usually two parts meeting at a tangent or a copy landing on a wall")
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


def _channel(occ, width, start, heading, path) -> list[int]:
    """The centreline compiled to faces: a rect per line leg, a band per arc (its centre
    at the leg end plus R along the normal, so the arc continues the leg tangentially),
    a disk of w/2 at each joint so nothing meets at a knife edge; all fused."""
    x, y = start
    h = heading
    parts: list[int] = []
    joints: list[tuple[float, float]] = []
    for seg in path:
        if "line" in seg or "to" in seg:
            if "to" in seg:
                # Run along the heading until the line x=v or y=v, and half a width
                # beyond it so the leg joins what it meets cleanly. This is the one
                # arithmetic a passage's return leg keeps getting wrong when it is
                # typed as a length: where the arc left it, and how far the wall is.
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
                length += width / 2
            else:
                length = seg["line"]
            parts.append(_leg(occ, x, y, h, length, width))
            x += length * math.cos(math.radians(h))
            y += length * math.sin(math.radians(h))
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
            h += angle
            x, y = cx + radius * math.cos(math.radians(a1)), cy + radius * math.sin(math.radians(a1))
        joints.append((x, y))
    for jx, jy in joints[:-1]:
        parts.append(occ.addDisk(jx, jy, 0, width / 2, width / 2))
    tags = [(2, t) for t in parts]
    if len(tags) > 1:
        tags, _ = occ.fuse([tags[0]], tags[1:])
    return [t for d, t in tags]


def build_face(gmsh, ops: list[dict], scale: float = 1.0, rotate: float = 0.0) -> int:
    """Run the ops through OpenCASCADE and return the one face that is the body.
    Everything else the ops made along the way is removed, so nothing but the body
    is meshed."""
    occ = gmsh.model.occ
    made: dict[str, list[int]] = {}

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
            made[name] = _channel(occ, op["width"], op["start"], op["heading"], op["path"])
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
                copies += c
            out, _ = occ.fuse(base, copies)
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


def measure2d(gmsh, face: int, domain=None, external: bool = False) -> Built2D:
    occ = gmsh.model.occ
    x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(2, face)
    bounds = (x0, y0, x1, y1)
    extent = (x1 - x0, y1 - y0)
    span = max(extent)
    loops, _ = occ.getCurveLoops(face)
    curves = [abs(t) for d, t in gmsh.model.getBoundary([(2, face)], combined=False, oriented=False)]
    lengths = {c: occ.getMass(1, c) for c in curves}
    short = [(c, length) for c, length in lengths.items() if length < 1e-3 * span]
    return Built2D(face, bounds, extent, occ.getMass(2, face), len(loops) - 1, curves, lengths,
                   short, domain or bounds, external)


def curve_bounds(gmsh, curve: int) -> tuple:
    x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(1, curve)
    return (x0, y0, x1, y1)


def classify_curves(gmsh, built: Built2D, axis: int, rules: list[dict]) -> dict[int, str]:
    tol = max(built.extent) * 1e-4 + 1e-12
    return {c: patch_for_edge(curve_bounds(gmsh, c), built.domain, axis, tol, rules, built.external)
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
            caption: list[str]) -> Path:
    """The outline, filled, with every edge coloured by the patch it was read as, and
    the measured report under it. A coarse triangle mesh is drawn and thrown away; the
    real mesh is made afterwards with its own options."""
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
    ax.set_aspect("equal")
    ax.autoscale()
    ax.grid(True, lw=0.3)
    ax.set_title(title)
    ax.legend([Line2D([0], [0], color=col, lw=2) for col in seen.values()], list(seen),
              loc="upper right", fontsize=8, frameon=True)
    fig.text(0.01, 0.01, "\n".join(caption), family="monospace", fontsize=8, va="bottom")
    fig.subplots_adjust(bottom=0.06 + 0.035 * len(caption))
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
    patches: dict[str, list[int]] = {}
    for d, s in gmsh.model.getBoundary([(3, volume)], combined=True, oriented=False):
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, s)
        if abs(z1 - z0) < tol:
            name = EMPTY
        else:
            name = patch_for_edge((x0, y0, x1, y1), built.domain, axis, tol, rules, built.external)
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
            notes: list[str], drawn: Path | None) -> list[str]:
    cell, wall_cell, thickness = sizes
    lines = report_lines(built, curve_patch, source)
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
        body = build_face(gmsh, ops, scale=args.scale, rotate=args.rotate)
        external = bool(args.external)
        if external:
            face, box = external_face(gmsh, body, opts)
            built = measure2d(gmsh, face, domain=box, external=True)
        else:
            built = measure2d(gmsh, body)
        axis = {"auto": longest_axis2d(built.extent), "x": 0, "y": 1}[args.along]
        curve_patch = classify_curves(gmsh, built, axis, rules)
        names = set(curve_patch.values())
        if not external:
            for want in ("inlet", "outlet"):
                if want not in names:
                    end = "low" if want == "inlet" else "high"
                    notes.append(f"!! no edge sits flat at the passage's {end} end along "
                                 f"{'xy'[axis]}, so there is no {want}; --{want} names one")
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
        for c, ln in built.short:
            notes.append(f"a {ln:.3g} m edge at curve {c} will make poor quads")

        drawn = None
        if args.preview is not None:
            caption = report_lines(built, curve_patch, source)
            drawn = preview(gmsh, built, curve_patch, args.preview, Path(source.split(" ")[0]).name, caption)

        flow = model = why = None
        if args.study != "mesh" or not args.dry_run:
            flow = case_gen.derive_flow(opts, length)
            model, why = case_gen.turbulence_model(opts, flow)

        if args.dry_run:
            for line in summary(built, curve_patch, source, None, flow, model or "", why or "",
                                args.study, sizes, length_note, notes, drawn):
                print(line)
            return 0

        target: Path = args.case
        if (target / "Allmesh").exists() and not args.force:
            raise SystemExit(f"{target} already holds a case; --force to write over it")
        target.mkdir(parents=True, exist_ok=True)
        (target / "constant" / "geometry").mkdir(parents=True, exist_ok=True)
        if args.spec is not None:
            (target / "constant" / "geometry" / "body.json").write_text(
                json.dumps({"ops": ops, "patches": [
                    {"name": r["name"], "kind": r["kind"],
                     **({"at": f"{'xy'[r['at'][0]]}:{r['at'][1] if r['at'][1] != 'at' else r['at'][2]}"}
                        if r["at"] else {"box": list(r["box"])})} for r in rules],
                            "scale": args.scale, "rotate": args.rotate}, indent=1),
                encoding="utf-8")
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
                            length_note, notes, drawn):
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
