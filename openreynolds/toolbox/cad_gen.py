"""A runnable case from a solid built out of primitives, or from a STEP file.

    python3 cad_gen.py CASE --spec body.json --speed 10 --nu 1.5e-5
    python3 cad_gen.py CASE --step body.step --scale 0.001 --speed 10
    python3 cad_gen.py CASE --spec body.json --layers 5 --y-plus 30    # prism layers
    python3 cad_gen.py CASE --spec body.json --symmetry y              # half the body
    python3 cad_gen.py CASE --spec body.json --study thermal --wall-temperature 350
    python3 cad_gen.py CASE --spec body.json --dry-run          # geometry and estimates only
    python3 cad_gen.py CASE --spec body.json --mesh             # and run Allmesh here

The other two generators start from a hand-drawn 2D outline (`case_gen.py`) or an
uploaded triangle surface (`snappy_gen.py`). This one starts from a *description of a
solid*: boxes, cylinders, spheres, cones, prisms, joined, subtracted, moved and turned,
in a small JSON list -- or any STEP file, which is what build123d, CadQuery, FreeCAD and
every commercial CAD package write. A penne is two cylinders and a subtraction; a house
is a box, a prism and a subtraction; a laptop is two boxes, two cylinders and a fillet.
None of them is its own template, because a laptop is not a primitive: it is a
composition, and the composition is what gets described.

The mesh is body-fitted and made by gmsh's OpenCASCADE kernel straight from the B-rep --
no STL export, no snappyHexMesh, no snapping to triangles. The CAD edges are where the
mesh sits by construction. On a 2026-09 penne that whole step took under a second where
snappyHexMesh took three minutes plus twelve minutes of writing dictionaries by hand.
`--layers N` grows N prism layers off the walls (sized from `--y-plus` or
`--layer-first`), extruded normal to the surface before the tetrahedra are filled in, so
they are real prisms and not a shrink-wrap; the planar patches the body touches -- a
floor, a symmetry plane, a pipe's inlet -- are re-made around the layer's rim.

The spec, one op per entry, each optionally `"name"`d so a later op can refer to it;
the op named `"body"` is the solid, otherwise the last one is:

    [{"op": "cylinder", "name": "outer", "base": [0,0,0], "axis": [0.04,0,0], "radius": 0.005},
     {"op": "cylinder", "name": "bore",  "base": [0,0,0], "axis": [0.04,0,0], "radius": 0.004},
     {"op": "cut", "name": "tube", "from": "outer", "take": ["bore"]},
     {"op": "rotate", "name": "body", "target": "tube", "axis": [0,1,0], "angle": 20,
      "about": [0.02,0,0]}]

    box       origin|center, size                      sphere   center, radius
    cylinder  base, axis (length is its norm), radius  cone     base, axis, r1, r2
    torus     center, radius, tube                     prism    plane xy|xz|yz, polygon, from, to
    fuse      of [..]        cut  from, take [..]      intersect  of [..]
    translate target, by     rotate target, axis, angle (deg), about
    fillet    target, radius (every edge; a radius that does not fit stops with the reason)

`constant/geometry/` keeps the STEP it meshed and the spec it was built from, so the
geometry is a file a person or a CAD program can open and change, and `Allmesh` turns
the `.msh` into an OpenFOAM mesh (`gmshToFoam`, patch types, `checkMesh`). The 0/,
system/ and constant/ files come from the same writers `snappy_gen.py` uses, including
its thermal (`--study thermal`) and rotating-zone (`--mrf`) ones.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sibling(name: str):
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


case_gen = sibling("case_gen")
snappy_gen = sibling("snappy_gen")

PRIMITIVES = ("box", "cylinder", "sphere", "cone", "torus", "prism")
BOOLEANS = ("fuse", "cut", "intersect")
TRANSFORMS = ("translate", "rotate", "fillet")
OPS = PRIMITIVES + BOOLEANS + TRANSFORMS

PLANES = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}
"""For a prism: which two axes the polygon is drawn in, and the third it is extruded along."""

AXES = {"x": 0, "y": 1, "z": 2}
THERMAL = ("thermal", "thermal-transient")
TRANSIENT = ("transient", "thermal-transient")


# -- the spec ----------------------------------------------------------------------

def vec3(value, what: str) -> tuple[float, float, float]:
    try:
        x, y, z = (float(v) for v in value)
    except (TypeError, ValueError):
        raise SystemExit(f"{what} wants three numbers, got {value!r}")
    return (x, y, z)


def parse_spec(entries) -> list[dict]:
    """The op list checked and normalised: every op known, every name unique, every
    reference to an earlier name. A bad spec stops here, with the entry named, rather
    than inside the CAD kernel with a tag number."""
    if not isinstance(entries, list) or not entries:
        raise SystemExit("the spec is a non-empty JSON list of ops")
    ops: list[dict] = []
    names: set[str] = set()
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict) or "op" not in raw:
            raise SystemExit(f"spec entry {index} has no \"op\"")
        op = dict(raw)
        kind = op["op"]
        if kind not in OPS:
            raise SystemExit(f"spec entry {index}: unknown op {kind!r}; one of {', '.join(OPS)}")
        name = op.get("name") or f"op{index}"
        if name in names:
            raise SystemExit(f"spec entry {index}: the name {name!r} is already used")
        op["name"] = name
        refs = []
        if kind == "fuse" or kind == "intersect":
            refs = list(op.get("of") or [])
            if len(refs) < 2:
                raise SystemExit(f"spec entry {index} ({kind}): \"of\" wants two or more names")
        elif kind == "cut":
            if "from" not in op or not op.get("take"):
                raise SystemExit(f"spec entry {index} (cut): wants \"from\" and \"take\"")
            refs = [op["from"]] + list(op["take"])
        elif kind in TRANSFORMS:
            if "target" not in op:
                raise SystemExit(f"spec entry {index} ({kind}): wants \"target\"")
            refs = [op["target"]]
        for ref in refs:
            if ref not in names:
                raise SystemExit(f"spec entry {index} ({kind}): {ref!r} is not an earlier op")
        if kind == "box":
            if "origin" not in op and "center" not in op:
                raise SystemExit(f"spec entry {index} (box): wants \"origin\" or \"center\"")
            op["size"] = vec3(op.get("size"), f"spec entry {index} (box) size")
        elif kind in ("cylinder", "cone"):
            op["base"] = vec3(op.get("base"), f"spec entry {index} ({kind}) base")
            op["axis"] = vec3(op.get("axis"), f"spec entry {index} ({kind}) axis")
            if math.sqrt(sum(c * c for c in op["axis"])) <= 0:
                raise SystemExit(f"spec entry {index} ({kind}): the axis has no length")
        elif kind == "prism":
            plane = op.get("plane", "xy")
            if plane not in PLANES:
                raise SystemExit(f"spec entry {index} (prism): plane is one of xy, xz, yz")
            polygon = op.get("polygon") or []
            if len(polygon) < 3:
                raise SystemExit(f"spec entry {index} (prism): the polygon wants three or more points")
            op["plane"] = plane
            op["polygon"] = [(float(p[0]), float(p[1])) for p in polygon]
            op["from"], op["to"] = float(op.get("from", 0.0)), float(op.get("to", 1.0))
            if op["to"] == op["from"]:
                raise SystemExit(f"spec entry {index} (prism): from and to are the same")
        elif kind == "rotate":
            op["axis"] = vec3(op.get("axis"), f"spec entry {index} (rotate) axis")
            op["angle"] = float(op.get("angle", 0.0))
            op["about"] = vec3(op.get("about", (0.0, 0.0, 0.0)), f"spec entry {index} (rotate) about")
        elif kind == "translate":
            op["by"] = vec3(op.get("by"), f"spec entry {index} (translate) by")
        elif kind == "fillet":
            op["radius"] = float(op.get("radius", 0.0))
            if op["radius"] <= 0:
                raise SystemExit(f"spec entry {index} (fillet): radius wants to be positive")
        names.add(name)
        ops.append(op)
    return ops


def body_name(ops: list[dict]) -> str:
    """The op called `body`, or the last one."""
    for op in ops:
        if op["name"] == "body":
            return "body"
    return ops[-1]["name"]


# -- building it -------------------------------------------------------------------

class Built:
    """What the kernel handed back: the body's dimTags, its bounds, its volume."""

    def __init__(self, tags, bounds, volume: float, extent):
        self.tags = list(tags)
        self.bounds = tuple(bounds)          # x0, y0, z0, x1, y1, z1
        self.volume = float(volume)
        self.extent = tuple(extent)          # dx, dy, dz

    @property
    def centre(self) -> tuple[float, float, float]:
        b = self.bounds
        return ((b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2)


def build_solid(gmsh, ops: list[dict]) -> list[tuple[int, int]]:
    """Run the spec through gmsh's OpenCASCADE kernel; returns the body's dimTags."""
    occ = gmsh.model.occ
    made: dict[str, list[tuple[int, int]]] = {}
    for op in ops:
        kind, name = op["op"], op["name"]
        if kind == "box":
            dx, dy, dz = op["size"]
            if "center" in op:
                cx, cy, cz = vec3(op["center"], "box center")
                x, y, z = cx - dx / 2, cy - dy / 2, cz - dz / 2
            else:
                x, y, z = vec3(op["origin"], "box origin")
            made[name] = [(3, occ.addBox(x, y, z, dx, dy, dz))]
        elif kind == "cylinder":
            (x, y, z), (dx, dy, dz) = op["base"], op["axis"]
            made[name] = [(3, occ.addCylinder(x, y, z, dx, dy, dz, float(op["radius"])))]
        elif kind == "sphere":
            x, y, z = vec3(op.get("center"), "sphere center")
            made[name] = [(3, occ.addSphere(x, y, z, float(op["radius"])))]
        elif kind == "cone":
            (x, y, z), (dx, dy, dz) = op["base"], op["axis"]
            made[name] = [(3, occ.addCone(x, y, z, dx, dy, dz, float(op["r1"]), float(op["r2"])))]
        elif kind == "torus":
            x, y, z = vec3(op.get("center"), "torus center")
            made[name] = [(3, occ.addTorus(x, y, z, float(op["radius"]), float(op["tube"])))]
        elif kind == "prism":
            a, b, along = PLANES[op["plane"]]
            points = []
            for u, v in op["polygon"]:
                coord = [0.0, 0.0, 0.0]
                coord[a], coord[b], coord[along] = u, v, op["from"]
                points.append(occ.addPoint(*coord))
            lines = [occ.addLine(points[i], points[(i + 1) % len(points)])
                     for i in range(len(points))]
            surface = occ.addPlaneSurface([occ.addCurveLoop(lines)])
            step = [0.0, 0.0, 0.0]
            step[along] = op["to"] - op["from"]
            out = occ.extrude([(2, surface)], *step)
            made[name] = [e for e in out if e[0] == 3]
        elif kind == "fuse":
            parts = [made[n] for n in op["of"]]
            out, _ = occ.fuse(parts[0], [t for p in parts[1:] for t in p])
            made[name] = out
        elif kind == "cut":
            out, _ = occ.cut(made[op["from"]], [t for n in op["take"] for t in made[n]])
            made[name] = out
        elif kind == "intersect":
            parts = [made[n] for n in op["of"]]
            out, _ = occ.intersect(parts[0], [t for p in parts[1:] for t in p])
            made[name] = out
        elif kind == "translate":
            occ.translate(made[op["target"]], *op["by"])
            made[name] = made[op["target"]]
        elif kind == "rotate":
            (ax, ay, az), (px, py, pz) = op["axis"], op["about"]
            occ.rotate(made[op["target"]], px, py, pz, ax, ay, az, math.radians(op["angle"]))
            made[name] = made[op["target"]]
        elif kind == "fillet":
            occ.synchronize()
            target = made[op["target"]]
            edges = [t for d, t in gmsh.model.getBoundary(target, combined=False,
                                                          oriented=False, recursive=True)
                     if d == 1]
            try:
                out = occ.fillet([t for _, t in target], sorted(set(edges)),
                                 [op["radius"]], removeVolume=True)
            except Exception as exc:  # the kernel's own message names the edge
                raise SystemExit(f"fillet {name!r} at radius {op['radius']:g} did not build: {exc}")
            made[name] = out
        if not made[name]:
            raise SystemExit(f"op {name!r} ({kind}) produced no solid -- an empty boolean, usually")
    occ.synchronize()
    return made[body_name(ops)]


def measure(gmsh, tags) -> Built:
    occ = gmsh.model.occ
    boxes = [gmsh.model.getBoundingBox(d, t) for d, t in tags]
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes); z0 = min(b[2] for b in boxes)
    x1 = max(b[3] for b in boxes); y1 = max(b[4] for b in boxes); z1 = max(b[5] for b in boxes)
    volume = sum(occ.getMass(d, t) for d, t in tags)
    return Built(tags, (x0, y0, z0, x1, y1, z1), volume, (x1 - x0, y1 - y0, z1 - z0))


# -- symmetry ----------------------------------------------------------------------

def parse_symmetry(spec, bounds) -> list[dict]:
    """`--symmetry y,z:max` -> the planes to cut on. A bare axis is the body's own
    mid-plane; `:min`/`:max` its own extreme; `:0.125` a coordinate. A plane that
    bisects the body halves the reference area; one at its edge does not -- the same
    reading snappy_gen makes, for the same reason."""
    if not spec or spec == "none":
        return []
    planes = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        axis_name, _, where = item.partition(":")
        if axis_name not in AXES:
            raise SystemExit(f"--symmetry: '{axis_name}' is not one of x, y, z")
        axis = AXES[axis_name]
        lo, hi = bounds[axis], bounds[axis + 3]
        if where in ("", "mid"):
            plane = (lo + hi) / 2.0
        elif where == "max":
            plane = hi
        elif where == "min":
            plane = lo
        else:
            try:
                plane = float(where)
            except ValueError:
                raise SystemExit(f"--symmetry {item}: an axis, or axis:min, axis:max, or axis:<coordinate>")
        margin = 1e-6 * max(hi - lo, 1e-12)
        keep = "low" if plane >= hi - margin else "high"
        planes.append({"axis": axis, "name": axis_name, "plane": plane,
                       "bisects": lo + margin < plane < hi - margin, "keep": keep})
    return planes


def symmetry_patch_name(planes: list[dict], index: int) -> str:
    return "symmetry" if len(planes) == 1 else f"symmetry{planes[index]['name'].upper()}"


def cut_to_half(gmsh, tags, plane: dict, reach: float):
    """Take away the side of the body the symmetry plane discards."""
    occ = gmsh.model.occ
    lo = [-reach, -reach, -reach]
    size = [2 * reach, 2 * reach, 2 * reach]
    if plane["keep"] == "high":
        lo[plane["axis"]] = plane["plane"] - 2 * reach
        size[plane["axis"]] = 2 * reach
    else:
        lo[plane["axis"]] = plane["plane"]
        size[plane["axis"]] = 2 * reach
    box = occ.addBox(lo[0], lo[1], lo[2], size[0], size[1], size[2])
    out, _ = occ.cut(tags, [(3, box)])
    occ.synchronize()
    if not out:
        raise SystemExit(f"the {plane['name']} symmetry plane at {plane['plane']:.6g} leaves no body")
    return out


# -- the domain and its patches ----------------------------------------------------

def domain_bounds(built: Built, opts, planes=()) -> tuple[float, ...]:
    """The flow box, in body lengths, around the body -- the same convention as
    snappy_gen: --ahead/--behind along x, --side in y, --above/--below in z. A
    symmetry plane replaces the box side on its discarded half."""
    x0, y0, z0, x1, y1, z1 = built.bounds
    L = max(built.extent[0], 1e-12)
    b = [x0 - opts["ahead"] * L, y0 - opts["side"] * L,
         z0 if opts.get("ground") else z0 - opts["below"] * L,
         x1 + opts["behind"] * L, y1 + opts["side"] * L, z1 + opts["above"] * L]
    for plane in planes:
        if plane["keep"] == "high":
            b[plane["axis"]] = plane["plane"]
        else:
            b[plane["axis"] + 3] = plane["plane"]
    return tuple(b)


def on_plane(bounds, axis: int, value: float, tol: float) -> bool:
    return abs(bounds[axis] - value) < tol and abs(bounds[axis + 3] - value) < tol


def classify(surface_bounds, box, tol: float, ground: bool, planes=()) -> str:
    """Which patch a surface of the fluid volume belongs to, from where it sits:
    flat against the upstream box face is the inlet, downstream the outlet, a
    symmetry plane is the patch named for it, the floor is `ground` when there is
    one, the other box faces are `farfield`, and anything not on the box is the body."""
    a0, b0, c0, a1, b1, c1 = surface_bounds
    bx0, by0, bz0, bx1, by1, bz1 = box
    for index, plane in enumerate(planes):
        if on_plane(surface_bounds, plane["axis"], plane["plane"], tol):
            return symmetry_patch_name(list(planes), index)
    if abs(a0 - bx0) < tol and abs(a1 - bx0) < tol:
        return "inlet"
    if abs(a0 - bx1) < tol and abs(a1 - bx1) < tol:
        return "outlet"
    if ground and abs(c0 - bz0) < tol and abs(c1 - bz0) < tol:
        return "ground"
    for lo, hi, v0, v1 in ((by0, by1, b0, b1), (bz0, bz1, c0, c1)):
        if (abs(v0 - lo) < tol and abs(v1 - lo) < tol) or (abs(v0 - hi) < tol and abs(v1 - hi) < tol):
            return "farfield"
    return "body"


def longest_axis(extent) -> int:
    """0, 1 or 2: the axis a passage runs along, read off its extents."""
    return max(range(3), key=lambda i: extent[i])


def classify_internal(surface_bounds, bounds, axis: int, tol: float) -> str:
    """For a passage meshed on its own volume: the face flat at the body's low end
    along `axis` is the inlet, the one flat at its high end the outlet, everything
    else a wall. A passage whose ends are not flat to an axis gets no inlet, and
    the summary says so."""
    lo, hi = surface_bounds[axis], surface_bounds[axis + 3]
    b_lo, b_hi = bounds[axis], bounds[axis + 3]
    if abs(lo - b_lo) < tol and abs(hi - b_lo) < tol:
        return "inlet"
    if abs(lo - b_hi) < tol and abs(hi - b_hi) < tol:
        return "outlet"
    return "walls"


def build_roles(patches: dict[str, list[int]], opts, body_patch: str,
                direction=(1.0, 0.0, 0.0)) -> dict:
    far = opts.get("far") or "slip"
    roles = {}
    for name in patches:
        if name == "inlet":
            roles[name] = {"kind": "inlet", "direction": tuple(direction)}
        elif name == "outlet":
            roles[name] = {"kind": "outlet"}
        elif name == "ground":
            roles[name] = {"kind": "wall"}
        elif name.startswith("symmetry"):
            roles[name] = {"kind": "symmetry"}
        elif name == "farfield":
            roles[name] = {"kind": far}
        else:
            roles[name] = {"kind": "wall"}
    roles[body_patch] = {"kind": "wall"}
    return roles


def wall_patches(roles: dict) -> list[str]:
    return [name for name, role in roles.items() if role.get("kind") == "wall"]


def symmetry_patches(roles: dict) -> list[str]:
    return [name for name, role in roles.items() if role.get("kind") == "symmetry"]


# -- prism layers ------------------------------------------------------------------

def cumulative_heights(first: float, ratio: float, layers: int) -> list[float]:
    return [snappy_gen.stack_thickness(first, i + 1, ratio) for i in range(layers)]


def curves_of(gmsh, surfaces, combined=False) -> list[int]:
    return [t for d, t in gmsh.model.getBoundary([(2, s) for s in surfaces], combined=combined,
                                                 oriented=False) if d == 1]


def touched_surfaces(gmsh, walls: list[int], others: list[int]) -> list[int]:
    """The boundary surfaces the wall set shares an edge with -- a floor under a
    building, a symmetry plane through a body, the inlet of a passage."""
    wall_curves = set(curves_of(gmsh, walls))
    return [s for s in others if any(c in wall_curves for c in curves_of(gmsh, [s]))]


def add_prism_layers(gmsh, walls: list[int], others: list[int], first: float, ratio: float,
                     layers: int) -> list[int]:
    """Prisms extruded off a free-standing wall set, then the fluid volume rebuilt
    from the other boundary surfaces plus the layer's outer shell. Returns the
    volume tags.

    Only for a body that touches nothing: where the wall set has an open edge the
    extrusion leaves lateral faces in the touched plane, and re-making that plane
    around the layer's rim on entities that have no geometry until they are meshed
    did not close a volume in two afternoons of trying. Those bodies get their layers
    from snappyHexMesh instead (`snappy_layers_dict`), which was built for junctions."""
    geo = gmsh.model.geo
    out = geo.extrudeBoundaryLayer([(2, s) for s in walls], [1] * layers,
                                   cumulative_heights(first, ratio, layers), recombine=True)
    geo.synchronize()
    tops, volumes = [], []
    i = 0
    while i < len(out):
        d, t = out[i]
        if d == 2 and i + 1 < len(out) and out[i + 1][0] == 3:
            tops.append(t); volumes.append(out[i + 1][1]); i += 2
        else:
            i += 1
    loop = geo.addSurfaceLoop(list(others) + tops)
    volume = geo.addVolume([loop])
    geo.synchronize()
    return [volume] + volumes


LAYER_QUALITY = snappy_gen.MESH_QUALITY.replace("maxNonOrtho         65;", "maxNonOrtho         70;") \
    .replace("maxNonOrtho     75;", "maxNonOrtho     80;")
"""snappy's quality gate, loosened by five degrees at both levels: a Delaunay tet mesh
already carries cells at 65-68 degrees, and a gate set under what is there refuses
every layer next to them."""


def snappy_layers_dict(layer: dict, wall_name: str, point) -> str:
    """snappyHexMesh with only the third phase on: the mesh exists, the layers are
    grown on it. castellatedMesh and snap are off, and their control blocks are the
    minimum the dictionary reader insists on."""
    lines = ["castellatedMesh false;", "snap            false;", "addLayers       true;", "",
             "geometry", "{", "}", "",
             "castellatedMeshControls", "{",
             "    maxLocalCells       2000000;", "    maxGlobalCells      8000000;",
             "    minRefinementCells  0;", "    nCellsBetweenLevels 1;",
             "    features            ();", "    refinementSurfaces  {}",
             "    resolveFeatureAngle 30;", "    refinementRegions   {}",
             f"    locationInMesh      ({point[0]:.8g} {point[1]:.8g} {point[2]:.8g});",
             "    allowFreeStandingZoneFaces true;", "}", "",
             "snapControls", "{", "    nSmoothPatch    3;", "    tolerance       2.0;",
             "    nSolveIter      30;", "    nRelaxIter      5;", "}", "",
             "addLayersControls", "{",
             "    relativeSizes   false;   // firstLayerThickness is in metres",
             "    layers", "    {", f"        {wall_name}", "        {",
             f"            nSurfaceLayers  {layer['layers']};", "        }", "    }", "",
             f"    firstLayerThickness {layer['first']:.6g};",
             f"    expansionRatio  {layer['ratio']:g};",
             f"    minThickness    {layer['first'] * 0.1:.6g};",
             "    nGrow           0;", "    featureAngle    130;", "    slipFeatureAngle 30;",
             "    nRelaxIter      5;", "    nSmoothSurfaceNormals 1;", "    nSmoothNormals  3;",
             "    nSmoothThickness 10;", "    maxFaceThicknessRatio 0.5;",
             "    maxThicknessToMedialRatio 0.6;", "    minMedialAxisAngle 90;",
             "    nBufferCellsNoExtrude 0;", "    nLayerIter      50;", "    nRelaxedIter    25;",
             "}", "", "meshQualityControls", "{", '    #include "meshQualityDict"', "}", "",
             "writeFlags", "(", "    layerSets", "    layerFields", ");", "",
             "mergeTolerance  1e-6;"]
    return case_gen.foam_file("dictionary", "snappyHexMeshDict", "\n".join(lines), "system")


def inside_point(gmsh) -> tuple[float, float, float]:
    """The centroid of the first tetrahedron: a point that is in the fluid by
    construction, which a guessed box centre is not for a hollow body."""
    _, nodes = gmsh.model.mesh.getElementsByType(4)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    xyz = {int(t): (coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]) for i, t in enumerate(tags)}
    corners = [xyz[int(nodes[k])] for k in range(4)]
    return tuple(sum(c[i] for c in corners) / 4.0 for i in range(3))


# -- the mesh ----------------------------------------------------------------------

def mesh_sizes(built: Built, opts) -> tuple[float, float]:
    L = max(built.extent[0], 1e-12)
    near = opts.get("surface_cell") or L / 40.0
    far = opts.get("far_cell") or max(near * 8.0, L / 4.0)
    return float(near), float(far)


def projected_and_wetted(gmsh, surfaces: list[int]) -> tuple[float, float]:
    """The body's frontal area (projected along x) and wetted area, off the surface
    triangles that were actually meshed on it. The frontal projection counts front and
    back, so it is halved."""
    tags, coords, _ = gmsh.model.mesh.getNodes()
    xyz = {int(t): (coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]) for i, t in enumerate(tags)}
    frontal = wetted = 0.0
    for s in surfaces:
        for kind in (2, 3):     # triangles, and quads where a layer recombined them
            _, nodes = gmsh.model.mesh.getElementsByType(kind, s)
            n = 3 if kind == 2 else 4
            for i in range(0, len(nodes), n):
                corners = [xyz[int(nodes[i + k])] for k in range(n)]
                tris = [(corners[0], corners[1], corners[2])]
                if n == 4:
                    tris.append((corners[0], corners[2], corners[3]))
                for a, b, c in tris:
                    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
                    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
                    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
                    wetted += 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)
                    frontal += 0.5 * abs(nx)
    return frontal / 2.0, wetted


class MeshResult:
    def __init__(self, patches, tets, prisms, nodes, seconds, frontal, wetted,
                 layer_route=None, touched=(), inside=(0.0, 0.0, 0.0)):
        self.patches = patches      # name -> [surface tags]
        self.tets = int(tets)
        self.prisms = int(prisms)
        self.nodes = int(nodes)
        self.seconds = float(seconds)
        self.frontal = float(frontal)
        self.wetted = float(wetted)
        self.layer_route = layer_route   # None, "gmsh" (prisms in body.msh) or "snappy" (Allmesh)
        self.touched = list(touched)     # patch names the body shares an edge with
        self.inside = tuple(inside)


def generate(gmsh, built: Built, opts, body_patch: str, planes=(), layer=None):
    """Cut the fluid out (or keep the body's own volume), name the patches, grow the
    prism layers if asked, size the mesh finer on the body, and mesh in 3D."""
    occ = gmsh.model.occ
    internal = bool(opts.get("internal"))
    tol = max(built.extent) * 1e-4 + 1e-12
    if internal:
        fluid = list(built.tags)
        box = None
    else:
        box = domain_bounds(built, opts, planes)
        bx0, by0, bz0, bx1, by1, bz1 = box
        tag = occ.addBox(bx0, by0, bz0, bx1 - bx0, by1 - by0, bz1 - bz0)
        fluid, _ = occ.cut([(3, tag)], built.tags)
        occ.synchronize()
        if not fluid:
            raise SystemExit("the flow box minus the body is empty; the body fills or escapes it")
    occ.synchronize()

    axis = longest_axis(built.extent)

    def patch_of(surface: int) -> str:
        sb = gmsh.model.getBoundingBox(2, surface)
        name = classify_internal(sb, built.bounds, axis, tol) if internal \
            else classify(sb, box, tol, bool(opts.get("ground")), planes)
        return body_patch if name == "body" else name

    patches: dict[str, list[int]] = {}
    for d, s in gmsh.model.getBoundary(fluid, combined=True, oriented=False):
        patches.setdefault(patch_of(s), []).append(s)
    wall_name = "walls" if internal else body_patch
    walls = patches.get(wall_name, [])

    volumes = [t for _, t in fluid]
    layer_route = None
    touched: list[str] = []
    if layer and layer["layers"] > 0 and walls:
        home = {s: name for name, tags in patches.items() for s in tags}
        others = [s for name, tags in patches.items() if name != wall_name for s in tags]
        touched = sorted({home[s] for s in touched_surfaces(gmsh, walls, others)})
        if touched:
            layer_route = "snappy"      # grown on the OpenFOAM mesh, in Allmesh
        else:
            occ.remove(fluid, recursive=False)
            occ.synchronize()
            try:
                volumes = add_prism_layers(gmsh, walls, others, layer["first"], layer["ratio"],
                                           layer["layers"])
            except Exception as exc:
                raise SystemExit(f"prism layers did not build on this body: {exc}; "
                                 f"--layers 0 meshes it without")
            layer_route = "gmsh"

    for name, tags in patches.items():
        g = gmsh.model.addPhysicalGroup(2, tags)
        gmsh.model.setPhysicalName(2, g, name)
    g = gmsh.model.addPhysicalGroup(3, volumes)
    gmsh.model.setPhysicalName(3, g, "fluid")

    near, far = mesh_sizes(built, opts)
    L = max(built.extent[0], 1e-12)
    field = gmsh.model.mesh.field
    field.add("Distance", 1)
    field.setNumbers(1, "SurfacesList", walls)
    field.add("Threshold", 2)
    field.setNumber(2, "InField", 1)
    field.setNumber(2, "SizeMin", near)
    field.setNumber(2, "SizeMax", far)
    field.setNumber(2, "DistMin", 0.5 * L * opts.get("growth", 1.0))
    field.setNumber(2, "DistMax", 4.0 * L * opts.get("growth", 1.0))
    field.setAsBackgroundMesh(2)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
    gmsh.option.setNumber("Mesh.MeshSizeMin", near * 0.25)
    gmsh.option.setNumber("Mesh.MeshSizeMax", far)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)     # HXT: parallel Delaunay
    # Both optimisers: the Netgen pass costs about as long again as the meshing and
    # takes the worst tets off the top of the non-orthogonality distribution, which
    # is where a segregated solver on tets gets into trouble first.
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0 if layer_route == "gmsh" else 1)
    gmsh.option.setNumber("General.NumThreads", int(opts.get("threads") or os.cpu_count() or 1))

    started = time.time()
    gmsh.model.mesh.generate(3)
    seconds = time.time() - started
    tets = len(gmsh.model.mesh.getElementsByType(4)[0])
    prisms = len(gmsh.model.mesh.getElementsByType(6)[0]) + len(gmsh.model.mesh.getElementsByType(7)[0])
    nodes = len(gmsh.model.mesh.getNodes()[0])
    frontal, wetted = projected_and_wetted(gmsh, walls)
    return MeshResult(patches, tets, prisms, nodes, seconds, frontal, wetted,
                      layer_route, touched, inside_point(gmsh))


# -- the files ---------------------------------------------------------------------

def allmesh(roles: dict, cores: int, mrf: bool = False, snappy_layers: bool = False) -> str:
    """One command that turns body.msh into constant/polyMesh: gmshToFoam names the
    patches after the physical groups and calls every one of them `patch`, so the
    walls and the symmetry planes are retyped before anything reads the mesh; then
    the layers where snappy grows them, renumber, check, and the rotating zone when
    there is one. checkMesh's findings are printed, not treated as failure."""
    lines = ["#!/bin/sh", "set -e", "cd \"$(dirname \"$0\")\"",
             "gmshToFoam body.msh > log.gmshToFoam 2>&1"]
    for name in wall_patches(roles):
        lines.append(f"foamDictionary constant/polyMesh/boundary -entry entry0/{name}/type "
                     f"-set wall >> log.gmshToFoam 2>&1")
    for name in symmetry_patches(roles):
        lines.append(f"foamDictionary constant/polyMesh/boundary -entry entry0/{name}/type "
                     f"-set symmetry >> log.gmshToFoam 2>&1")
    if snappy_layers:
        lines.append("snappyHexMesh -overwrite > log.snappyHexMesh 2>&1")
    lines += ["renumberMesh -overwrite > log.renumberMesh 2>&1",
              "checkMesh > log.checkMesh 2>&1 || true"]
    if mrf:
        lines.append("topoSet > log.topoSet 2>&1")
    lines.append("grep -E 'cells:|Mesh OK|Failed|non-orthogonality|skewness' log.checkMesh")
    if snappy_layers:
        # The achieved layers, not the asked-for ones: snappy drops what does not fit
        # and says so only in this table.
        lines += ["echo '---- layers: got of asked, thickness percent ----'",
                  "grep -A40 'patch  *faces  *layers' log.snappyHexMesh 2>/dev/null"
                  " | grep -E '^[A-Za-z_][A-Za-z0-9_.]*  *[0-9]' | tail -6 || true"]
    lines.append("")
    return "\n".join(lines)


def allrun(solver: str, cores: int, thermal: bool = False) -> str:
    """potentialFoam first: a divergence-free start is what a tetrahedral mesh most
    wants, and on the 2026-09-06 penne the segregated solver started from a uniform
    field and had negative omega on its first iteration. The compressible solver
    starts from its own thermo, so it skips that."""
    if not solver:
        return "#!/bin/sh\n# a mesh-only case: nothing to run\n"
    lines = ["#!/bin/sh", "set -e", "cd \"$(dirname \"$0\")\""]
    if not thermal:
        lines.append("potentialFoam -writePhi > log.potentialFoam 2>&1")
    lines += ["decomposePar -force > log.decomposePar 2>&1",
              f"mpirun -np {cores} {solver} -parallel > log.{solver} 2>&1",
              "reconstructPar -latestTime > log.reconstructPar 2>&1", ""]
    return "\n".join(lines)


def fv_schemes(opts) -> str:
    """Schemes for a tetrahedral mesh, which is not the hex-dominant mesh snappy makes.

    Tets from a Delaunay mesher sit at 60-70 degrees of non-orthogonality at their
    worst, and the hex-tuned set -- second-order turbulence convection, a third of a
    non-orthogonal correction -- drove omega negative on the first iteration of the
    first case this generator produced. So: upwind on the turbulence equations (their
    accuracy is not where a drag or a pressure drop lives), half the correction on the
    Laplacians, and limited gradients everywhere."""
    steady = opts["study"] not in TRANSIENT
    thermal = opts["study"] in THERMAL
    ddt = "steadyState" if steady else "Euler"
    bounded = "bounded " if steady else ""
    lines = ["ddtSchemes", "{", f"    default         {ddt};", "}", "",
             "gradSchemes", "{", "    default         cellLimited Gauss linear 1;", "}", "",
             "divSchemes", "{", "    default         none;",
             f"    div(phi,U)      {bounded}Gauss linearUpwind grad(U);"]
    if thermal:
        for field in ("h", "K", "e", "Ekp"):
            lines.append(f"    div(phi,{field})      {bounded}Gauss upwind;")
    for field in ("k", "omega", "epsilon", "nuTilda", "gammaInt", "ReThetat"):
        lines.append(f"    div(phi,{field}) {bounded}Gauss upwind;")
    lines.append("    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;" if thermal
                 else "    div((nuEff*dev2(T(grad(U))))) Gauss linear;")
    lines += ["}", "",
              "laplacianSchemes", "{", "    default         Gauss linear limited corrected 0.5;",
              "}", "", "interpolationSchemes", "{", "    default         linear;", "}", "",
              "snGradSchemes", "{", "    default         limited corrected 0.5;", "}", "",
              "wallDist", "{", "    method          meshWave;", "}"]
    return case_gen.foam_file("dictionary", "fvSchemes", "\n".join(lines), "system")


def fv_solution(opts, model: str) -> str:
    """Plain SIMPLE with the classic factors rather than SIMPLEC: on tets the higher
    pressure factor buys speed the mesh cannot carry. Two non-orthogonal correctors,
    and the Phi solver and potentialFlow block that Allrun's potentialFoam reads."""
    transient = opts["study"] in TRANSIENT
    thermal = opts["study"] in THERMAL
    turb = list(case_gen.turbulence_fields(model))
    p_name = "p_rgh" if thermal else "p"
    lines = ["solvers", "{"]
    if thermal and transient:
        lines += ['    "rho.*"', "    {", "        solver          diagonal;", "    }", ""]
    lines += [f"    {p_name}", "    {",
              "        solver          GAMG;", "        tolerance       1e-7;",
              "        relTol          0.01;", "        smoother        GaussSeidel;", "    }", ""]
    if transient:
        lines += [f"    {p_name}Final", "    {", "        solver          GAMG;",
                  "        tolerance       1e-7;", "        relTol          0;",
                  "        smoother        GaussSeidel;", "    }", ""]
    elif not thermal:
        lines += case_gen.solver_entry("Phi", case_gen.PHI_SOLVER)
    fields = ["U"] + turb + (["h", "e"] if thermal else [])
    lines += [f'    "({"|".join(fields)})"', "    {", "        solver          smoothSolver;",
              "        smoother        symGaussSeidel;", "        tolerance       1e-8;",
              "        relTol          0.1;", "    }", ""]
    if transient:
        lines += [f'    "({"|".join(fields)})Final"', "    {", "        solver          smoothSolver;",
                  "        smoother        symGaussSeidel;", "        tolerance       1e-8;",
                  "        relTol          0;", "    }", ""]
    lines += ["}", ""]
    thermo = (["    pRefCell        0;", f"    pRefValue       {opts['pressure']:g};",
               "    rhoMin          0.2;", "    rhoMax          2.0;"] if thermal else [])
    if transient:
        lines += ["PIMPLE", "{", "    nOuterCorrectors 2;", "    nCorrectors     2;",
                  "    nNonOrthogonalCorrectors 2;", "    consistent      no;"] + thermo + ["}", ""]
    else:
        if not thermal:
            lines += ["potentialFlow", "{", "    nNonOrthogonalCorrectors 10;", "}", ""]
        lines += ["SIMPLE", "{", "    nNonOrthogonalCorrectors 2;", "    consistent      no;"]
        lines += thermo + ["", "    residualControl", "    {"]
        controls = [(p_name, 1e-4), ("U", 1e-5)] + [(f, 1e-5) for f in turb]
        if thermal:
            controls.append(("h", 1e-5))
        for name, value in controls:
            lines.append(f"        {name:<12s} {value:g};")
        lines += ["    }", "}", ""]
    lines += ["relaxationFactors", "{", "    fields", "    {"]
    lines += ['        ".*"            1;'] if transient else [f"        {p_name:<15s} 0.3;"]
    lines += ["    }", "    equations", "    {"]
    if transient:
        lines += ['        ".*"            1;']
    else:
        lines += ["        U               0.7;", '        ".*"            0.7;']
    lines += ["    }", "}"]
    return case_gen.foam_file("dictionary", "fvSolution", "\n".join(lines), "system")


def case_files(plan, flow, opts, model: str, body_patch: str, layer=None,
               mesh: MeshResult | None = None) -> dict[str, str]:
    solver = snappy_gen.SOLVERS[opts["study"]]
    thermal = opts["study"] in THERMAL
    snappy_layers = bool(mesh is not None and mesh.layer_route == "snappy" and layer)
    files = {
        "system/controlDict": snappy_gen.control_dict(opts, flow, body_patch),
        "system/fvSchemes": fv_schemes(opts),
        "system/fvSolution": fv_solution(opts, model),
        "system/decomposeParDict": snappy_gen.decompose_dict(opts["cores"]),
        "constant/turbulenceProperties": case_gen.turbulence_properties(model),
        "0/U": case_gen.field_U(plan, flow),
        "Allmesh": allmesh(plan.roles, opts["cores"], bool(opts.get("mrf")), snappy_layers),
        "Allrun": allrun(solver, opts["cores"], thermal),
        "case.foam": "",
    }
    if snappy_layers:
        files["system/snappyHexMeshDict"] = snappy_layers_dict(layer, body_patch, mesh.inside)
        files["system/meshQualityDict"] = case_gen.foam_file("dictionary", "meshQualityDict",
                                                             LAYER_QUALITY, "system")
    if thermal:
        files["constant/thermophysicalProperties"] = snappy_gen.thermo_properties(opts, flow)
        files["constant/g"] = snappy_gen.gravity_file()
        files["0/p"] = snappy_gen.field_p_thermal(plan, opts, "p")
        files["0/p_rgh"] = snappy_gen.field_p_thermal(plan, opts, "p_rgh")
        files["0/T"] = snappy_gen.field_T(plan, opts)
        files["0/alphat"] = snappy_gen.field_alphat(plan, opts)
    else:
        files["constant/transportProperties"] = case_gen.transport_properties(flow)
        files["0/p"] = case_gen.field_p(plan)
    if opts.get("mrf"):
        files["constant/MRFProperties"] = snappy_gen.mrf_properties(opts)
        files["system/topoSetDict"] = snappy_gen.topo_set_dict(opts)
    fields = case_gen.turbulence_fields(model)
    if fields:
        intensity = opts.get("turbulent_intensity")
        if intensity is None:
            intensity = case_gen.FREE_STREAM_INTENSITY
        mixing = opts.get("mixing_length") or 0.07 * plan.length
        ratio = None if opts.get("mixing_length") else (
            opts.get("viscosity_ratio") or case_gen.FREE_STREAM_VISCOSITY_RATIO)
        writers = {
            "k": lambda: case_gen.field_k(plan, flow, intensity),
            "omega": lambda: case_gen.field_omega(plan, flow, intensity, mixing,
                                                  nu=flow.nu, ratio=ratio),
            "epsilon": lambda: case_gen.field_epsilon(plan, flow, intensity, mixing),
            "nuTilda": lambda: case_gen.field_nu_tilda(plan, flow),
            "gammaInt": lambda: case_gen.field_gamma_int(plan),
            "ReThetat": lambda: case_gen.field_re_theta(plan, intensity),
        }
        for field in fields:
            files[f"0/{field}"] = writers[field]()
        files["0/nut"] = case_gen.field_nut(plan, model)
    return files


# -- the summary -------------------------------------------------------------------

def summary(built: Built, mesh: MeshResult | None, flow, opts, model: str, why: str,
            source: str, notes: list[str], planes=(), layer=None) -> list[str]:
    dx, dy, dz = built.extent
    lines = [f"geometry   {source}",
             f"           extent {dx:.4g} x {dy:.4g} x {dz:.4g} m, volume {built.volume:.4g} m3"]
    if opts.get("internal"):
        along = "xyz"[longest_axis(built.extent)]
        lines.append(f"domain     the body's own volume (internal flow), running along {along}: "
                     f"inlet at low {along}, outlet at high {along}")
    else:
        bx0, by0, bz0, bx1, by1, bz1 = domain_bounds(built, opts, planes)
        lines.append(f"domain     x {bx0:.4g}..{bx1:.4g}  y {by0:.4g}..{by1:.4g}  z {bz0:.4g}..{bz1:.4g}"
                     + ("  (floor is a wall)" if opts.get("ground") else ""))
    for index, plane in enumerate(planes):
        lines.append(f"symmetry   {symmetry_patch_name(list(planes), index)}: {plane['name']} = "
                     f"{plane['plane']:.6g} m, "
                     + ("cuts the body in half; the measured area is the half's"
                        if plane["bisects"] else "at the body's edge; nothing of it removed"))
    near, far = mesh_sizes(built, opts)
    lines.append(f"cells      {near:.4g} m on the body, {far:.4g} m far away")
    if layer and layer["layers"] > 0:
        line = (f"layers     {layer['layers']} prisms, first {layer['first']:.3g} m, ratio "
                f"{layer['ratio']:g}, stack {layer['total']:.3g} m")
        if mesh is not None and mesh.layer_route == "snappy":
            line += (f" -- grown by snappyHexMesh in Allmesh, because the body meets "
                     f"{', '.join(mesh.touched)}; its table there says how many were built")
        elif mesh is not None:
            line += " -- extruded in gmsh, in body.msh"
        lines.append(line)
    else:
        lines.append("layers     none (--layers N grows prisms off the walls)")
    if mesh is not None:
        cells = f"{mesh.tets:,} tetrahedra" + (f" + {mesh.prisms:,} prisms" if mesh.prisms else "")
        lines.append(f"mesh       {cells}, {mesh.nodes:,} nodes, {mesh.seconds:.2f} s to make")
        lines.append("patches    " + ", ".join(f"{k} ({len(v)})" for k, v in mesh.patches.items()))
        lines.append(f"area       frontal {mesh.frontal:.4g} m2, wetted {mesh.wetted:.4g} m2 "
                     f"(off the surface mesh)")
    lines.append(f"flow       {flow.speed:g} m/s, L {flow.length:.4g} m, nu {flow.nu:.3g} -> "
                 f"Re {flow.reynolds:,.0f} ({flow.derived} follows)")
    lines.append(f"model      {model} -- {why}")
    first_cell = layer["first"] if layer and layer["layers"] > 0 else near / 2.0
    y_plus = case_gen.estimate_y_plus(first_cell, flow)
    lines.append(f"y+         about {y_plus:,.0f} at the first cell; "
                 + ("wall functions bridge it" if y_plus >= 30 else
                    "under the log layer, so the wall function is nutUSpalding's blend"))
    solver = snappy_gen.SOLVERS[opts["study"]]
    lines.append(f"run        {opts['study']}" + (f" -> {solver}" if solver else ""))
    if opts.get("thermal"):
        lines.append(f"heat       wall {opts['wall_temperature']:g} K, inlet "
                     f"{opts['inlet_temperature']:g} K, Pr {opts['prandtl']:g}")
    if opts.get("mrf"):
        p1, p2, r = opts["mrf_p1"], opts["mrf_p2"], opts["mrf_radius"]
        lines.append(f"rotor      {opts['mrf_rpm']:g} rpm about {opts['mrf_axis']}, zone radius "
                     f"{r:.4g} m from {tuple(round(v, 4) for v in p1)} to {tuple(round(v, 4) for v in p2)}")
    for note in notes:
        lines.append(f"note       {note}")
    return lines


# -- main --------------------------------------------------------------------------

@contextlib.contextmanager
def quiet_fd1():
    """OpenCASCADE's STEP writer reports to file descriptor 1 directly, past any
    gmsh verbosity setting; the summary is the only thing wanted on stdout."""
    try:
        sys.stdout.flush()
        fd = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return
    saved = os.dup(fd)
    try:
        with open(os.devnull, "w") as null:
            os.dup2(null.fileno(), fd)
            yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, fd)
        os.close(saved)


def parse_rotation(spec: str) -> list[tuple[float, float, float, float]]:
    """'y:20' or 'y:20,z:-30' -> [(ax, ay, az, angle), ...], applied in order."""
    turns = []
    for part in [p for p in (spec or "").split(",") if p.strip()]:
        axis, _, angle = part.partition(":")
        axis = axis.strip().lower()
        if axis not in ("x", "y", "z"):
            raise SystemExit(f"--rotate wants x:, y: or z:, got {part!r}")
        vector = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[axis]
        turns.append(vector + (float(angle),))
    return turns


def preview(gmsh, built: Built, out: Path, title: str) -> Path:
    """The solid alone, surface-meshed and drawn by geometry_view's four fixed views
    -- the picture a person checks a spec against before anything is meshed in 3D.
    geometry_view reads surface files, so the surface goes through an STL beside
    the PNG; pyvista is imported only here, so the rest of the script needs none."""
    import tempfile
    geometry_view = sibling("geometry_view")
    span = max(built.extent) or 1.0
    # Coarse on purpose: the facets are the outline the picture is read by, and at a
    # sixtieth of the span they merge into a black surface.
    gmsh.option.setNumber("Mesh.MeshSizeMax", span / 25.0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)
    gmsh.model.mesh.generate(2)
    with tempfile.TemporaryDirectory() as tmp:
        surface = Path(tmp) / "preview.stl"
        gmsh.write(str(surface))
        mesh = geometry_view.load(surface)
        out = Path(out)
        geometry_view.draw([("body", mesh)], out, title)
    gmsh.model.mesh.clear()
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
    return out


def layer_plan(opts, flow, length: float) -> dict:
    """How thick the first prism is, and the stack: from --layer-first, else from
    --y-plus through the flat-plate correlation snappy_gen uses."""
    layers = int(opts.get("layers") or 0)
    ratio = float(opts.get("layer_ratio") or 1.2)
    if layers <= 0:
        return {"layers": 0, "first": 0.0, "ratio": ratio, "total": 0.0}
    if opts.get("layer_first"):
        first = float(opts["layer_first"])
    else:
        first, _ = snappy_gen.first_layer_thickness(float(opts.get("y_plus") or 50.0), flow, length)
    return {"layers": layers, "first": first, "ratio": ratio,
            "total": snappy_gen.stack_thickness(first, layers, ratio)}


def mrf_zone(built: Built, opts) -> list[str]:
    """The rotating cylinder of cells, sized to CONTAIN the rotor, as snappy_gen sizes it."""
    axis = AXES[opts["mrf_axis"]]
    centre = built.centre
    span = max(built.extent[i] for i in range(3) if i != axis)
    radius = opts.get("mrf_radius") or 0.6 * span
    thickness = opts.get("mrf_thickness") or 1.5 * built.extent[axis]
    notes = []
    if thickness < built.extent[axis]:
        notes.append(f"!! the MRF zone is {thickness:.4g} m thick along {opts['mrf_axis']} but the "
                     f"rotor spans {built.extent[axis]:.4g} m there; --mrf-thickness")
    if 2 * radius < span:
        notes.append(f"!! the MRF zone is {2 * radius:.4g} m across but the rotor spans {span:.4g} m; "
                     f"--mrf-radius")
    p1, p2 = list(centre), list(centre)
    p1[axis] -= thickness / 2.0
    p2[axis] += thickness / 2.0
    opts["mrf_p1"], opts["mrf_p2"], opts["mrf_radius"] = tuple(p1), tuple(p2), radius
    return notes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", type=Path, nargs="?", help="Directory to write the case into.")
    src = ap.add_argument_group("the solid")
    src.add_argument("--spec", type=Path, default=None, help="JSON list of ops (see above).")
    src.add_argument("--step", type=Path, default=None, help="A STEP (or BREP/IGES) file.")
    src.add_argument("--scale", type=float, default=1.0,
                     help="Multiply the solid by this (0.001 for a file in mm).")
    src.add_argument("--rotate", default="",
                     help="Turn the solid before meshing: 'y:90' or 'y:90,z:-30', degrees, "
                          "about the origin, in order. The inlet is always -x.")
    src.add_argument("--body-name", default="body", dest="body_name",
                     help="Patch name for the solid's surface (default body).")
    src.add_argument("--internal", action="store_true",
                     help="Mesh the solid's own volume as the passage -- a pipe, a duct -- "
                          "with the inlet at its low face and the outlet at its high face "
                          "along its longest axis, rather than the air around it.")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="Build the solid and report; mesh and write nothing.")
    ap.add_argument("--preview", type=Path, default=None,
                    help="Draw the solid from geometry_view's four views into this PNG "
                         "(with --dry-run: the picture and the summary, nothing else).")
    ap.add_argument("--mesh", action="store_true",
                    help="After writing, run Allmesh here (needs gmshToFoam on PATH).")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="Let gmsh print what it is doing.")

    flow = ap.add_argument_group("the flow")
    flow.add_argument("--speed", type=float, default=1.0)
    flow.add_argument("--nu", type=float, default=None,
                      help="Kinematic viscosity, m2/s (default 1.5e-5, air at 20 C).")
    flow.add_argument("--reynolds", type=float, default=None)
    flow.add_argument("--length", type=float, default=None,
                      help="Reference length for Re (default: the body's x extent).")
    flow.add_argument("--density", type=float, default=1.205)
    flow.add_argument("--turbulence", default="auto",
                      choices=["auto", "laminar", "kOmegaSST", "kOmegaSSTLM", "kEpsilon",
                               "SpalartAllmaras"])
    flow.add_argument("--turbulent-intensity", type=float, default=None, dest="turbulent_intensity")
    flow.add_argument("--viscosity-ratio", type=float, default=None, dest="viscosity_ratio")
    flow.add_argument("--mixing-length", type=float, default=None, dest="mixing_length")

    run = ap.add_argument_group("the run")
    run.add_argument("--study", default="steady",
                     choices=["mesh", "steady", "transient", "thermal", "thermal-transient"],
                     help="thermal is buoyantSimpleFoam with the body at --wall-temperature.")
    run.add_argument("--iterations", type=int, default=1000)
    run.add_argument("--end-time", type=float, default=1.0, dest="end_time")
    run.add_argument("--delta-t", type=float, default=None, dest="delta_t")
    run.add_argument("--courant", type=float, default=5.0)
    run.add_argument("--writes", type=int, default=10)
    run.add_argument("--cores", type=int, default=4)

    box = ap.add_argument_group("the domain, in body lengths (external flow)")
    box.add_argument("--ahead", type=float, default=2.0)
    box.add_argument("--behind", type=float, default=5.0)
    box.add_argument("--side", type=float, default=2.0)
    box.add_argument("--above", type=float, default=2.0)
    box.add_argument("--below", type=float, default=2.0)
    box.add_argument("--far", default="slip", choices=["slip", "symmetry"])
    box.add_argument("--ground", action="store_true",
                     help="The floor sits at the body's underside and is a wall.")
    box.add_argument("--symmetry", default="none",
                     help="Cut on one or more planes, comma separated: 'y' (the body's own "
                          "mid-plane), 'z:max' / 'z:min' (its own extreme), or 'y:0.125' (a "
                          "coordinate). A plane that bisects the body halves the measured area.")

    mesh = ap.add_argument_group("the mesh")
    mesh.add_argument("--surface-cell", type=float, default=None, dest="surface_cell",
                      help="Cell size on the body, m (default: body length / 40).")
    mesh.add_argument("--far-cell", type=float, default=None, dest="far_cell",
                      help="Cell size at the far boundary, m (default: 8x the surface cell).")
    mesh.add_argument("--growth", type=float, default=1.0,
                      help="Stretch the refinement zone around the body by this factor.")
    mesh.add_argument("--threads", type=int, default=None, help="gmsh threads (default: all).")
    mesh.add_argument("--layers", type=int, default=0,
                      help="Prism layers grown off the walls (default 0: none).")
    mesh.add_argument("--layer-first", type=float, default=None, dest="layer_first",
                      help="First prism thickness, m (default: from --y-plus).")
    mesh.add_argument("--layer-ratio", type=float, default=1.2, dest="layer_ratio")
    mesh.add_argument("--y-plus", type=float, default=50.0, dest="y_plus",
                      help="Target y+ for the first prism when --layers is set (default 50).")

    forces = ap.add_argument_group("the forces")
    forces.add_argument("--area", default="frontal", choices=["frontal", "wetted"])
    forces.add_argument("--ref-area", type=float, default=None, dest="ref_area")

    rot = ap.add_argument_group("a rotating zone (MRF)")
    rot.add_argument("--mrf", action="store_true")
    rot.add_argument("--mrf-rpm", type=float, default=0.0, dest="mrf_rpm")
    rot.add_argument("--mrf-axis", default="x", choices=["x", "y", "z"], dest="mrf_axis")
    rot.add_argument("--mrf-radius", type=float, default=None, dest="mrf_radius")
    rot.add_argument("--mrf-thickness", type=float, default=None, dest="mrf_thickness")

    heat = ap.add_argument_group("heat")
    heat.add_argument("--wall-temperature", type=float, default=333.15, dest="wall_temperature")
    heat.add_argument("--inlet-temperature", type=float, default=293.15, dest="inlet_temperature")
    heat.add_argument("--prandtl", type=float, default=0.71)
    heat.add_argument("--turbulent-prandtl", type=float, default=0.85, dest="turbulent_prandtl")
    heat.add_argument("--cp", type=float, default=1005.0)
    heat.add_argument("--pressure", type=float, default=101325.0)

    args = ap.parse_args(argv)
    if (args.spec is None) == (args.step is None):
        ap.error("one of --spec or --step")
    if args.case is None and not args.dry_run:
        ap.error("a directory to write the case into, or --dry-run")
    if args.internal and args.symmetry != "none":
        ap.error("--symmetry applies to the flow around a body, not to --internal")

    opts = vars(args)
    opts["thermal"] = args.study in THERMAL
    if opts["nu"] is None and opts["reynolds"] is None:
        opts["nu"] = 1.5e-5
    notes: list[str] = []

    try:
        import gmsh  # the image carries the Python module; the binary alone is not enough
    except ImportError:
        raise SystemExit("the gmsh Python module is not importable here; on the instance it "
                         "is, and `python3 -c 'import gmsh'` is the check")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if args.verbose else 0)
    gmsh.model.add("cad_gen")
    occ = gmsh.model.occ
    try:
        if args.spec is not None:
            ops = parse_spec(json.loads(args.spec.read_text(encoding="utf-8")))
            tags = build_solid(gmsh, ops)
            source = f"{args.spec.name}: {len(ops)} ops, body is {body_name(ops)!r}"
        else:
            if not args.step.exists():
                raise SystemExit(f"no such file: {args.step}")
            ops = []
            shapes = occ.importShapes(str(args.step))
            occ.synchronize()
            tags = [e for e in shapes if e[0] == 3]
            if not tags:
                raise SystemExit(f"{args.step} holds no solid (surfaces only?)")
            if len(tags) > 1:
                tags, _ = occ.fuse([tags[0]], tags[1:])
                occ.synchronize()
                notes.append(f"{args.step.name} held {len(shapes)} shapes; fused into one body")
            source = f"{args.step.name}"
        if args.scale != 1.0:
            occ.dilate(tags, 0, 0, 0, args.scale, args.scale, args.scale)
            occ.synchronize()
            notes.append(f"the solid was scaled by {args.scale:g}")
        for ax, ay, az, angle in parse_rotation(args.rotate):
            occ.rotate(tags, 0, 0, 0, ax, ay, az, math.radians(angle))
            occ.synchronize()
        if args.rotate:
            notes.append(f"the solid was turned {args.rotate} before meshing")

        whole = measure(gmsh, tags)
        if whole.volume <= 0:
            raise SystemExit("the solid has no volume")
        opts["_l_ref"] = args.length or whole.extent[0]
        flow_state = case_gen.derive_flow(opts, opts["_l_ref"])
        model, why = case_gen.turbulence_model(opts, flow_state)
        body_patch = "walls" if args.internal else args.body_name
        opts["_body_patch"] = body_patch
        planes = parse_symmetry(args.symmetry, whole.bounds)
        built = whole
        for plane in planes:
            if plane["bisects"]:
                tags = cut_to_half(gmsh, tags, plane, 10.0 * max(whole.extent))
                built = measure(gmsh, tags)
        layer = layer_plan(opts, flow_state, opts["_l_ref"])
        if layer["layers"]:
            notes += snappy_gen.y_plus_notes(float(args.y_plus), layer["layers"]) \
                if not args.layer_first else []
            near, _ = mesh_sizes(built, opts)
            if layer["total"] > near:
                # Not a warning: a stack thicker than the cells around it is extruded
                # into itself on any concave wall -- the bore of a tube, the inside of a
                # corner -- and the mesher does not fail, it runs without end.
                raise SystemExit(
                    f"the prism stack ({layer['total']:.3g} m: {layer['layers']} layers from "
                    f"{layer['first']:.3g} m at {layer['ratio']:g}) is thicker than the surface "
                    f"cell ({near:.3g} m). On a body this size that first layer is what "
                    f"--y-plus {opts['y_plus']:g} comes to; --layer-first sets it directly, "
                    f"or fewer layers, a lower ratio, or --surface-cell larger.")
        if args.mrf:
            notes += mrf_zone(built, opts)

        if args.preview is not None:
            preview(gmsh, built, args.preview, source)
            notes.append(f"drawn to {args.preview}")

        if args.dry_run:
            for line in summary(built, None, flow_state, opts, model, why, source, notes,
                                planes, layer):
                print(line)
            return 0

        target = Path(args.case)
        if (target / "Allmesh").exists() and not args.force:
            raise SystemExit(f"{target} already holds a case; --force to write over it")
        target.mkdir(parents=True, exist_ok=True)
        geometry = target / "constant" / "geometry"
        geometry.mkdir(parents=True, exist_ok=True)
        with quiet_fd1():
            gmsh.write(str(geometry / "body.step"))     # the B-rep, for a person or a CAD program
        if ops:
            (geometry / "body.json").write_text(json.dumps(ops, indent=1), encoding="utf-8")

        result = generate(gmsh, built, opts, body_patch, planes, layer)
        if args.internal and "inlet" not in result.patches:
            notes.append("!! no face sits flat at the passage's low end, so there is no inlet")
        if not args.internal and body_patch not in result.patches:
            raise SystemExit("no surface was left for the body; it sits outside the flow box")

        halved = sum(1 for p in planes if p["bisects"])
        if args.ref_area:
            factor = 0.5 ** halved
            opts["_a_ref"] = args.ref_area * factor
            opts["_a_ref_why"] = "--ref-area" + (f" x {factor:g} for the symmetry cut" if halved else "")
        elif args.area == "wetted":
            opts["_a_ref"], opts["_a_ref_why"] = result.wetted, "wetted, measured" + (
                " on the half" if halved else "")
        else:
            opts["_a_ref"], opts["_a_ref_why"] = result.frontal, "frontal, measured" + (
                " on the half" if halved else "")

        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.write(str(target / "body.msh"))

        direction = [0.0, 0.0, 0.0]
        direction[longest_axis(built.extent) if args.internal else 0] = 1.0
        plan = case_gen.Plan(snappy_gen.PatchList(list(result.patches)),
                             build_roles(result.patches, opts, body_patch, tuple(direction)),
                             opts["_l_ref"], {}, notes)
        files = case_files(plan, flow_state, opts, model, body_patch, layer, result)
        case_gen.write_case(target, files)
        (target / "Allmesh").chmod(0o755)
        (target / "Allrun").chmod(0o755)

        for line in summary(built, result, flow_state, opts, model, why, source, notes,
                            planes, layer):
            print(line)
        print(f"wrote      {target} ({len(files)} files + body.msh)")
    finally:
        gmsh.finalize()

    if args.mesh:
        if shutil.which("gmshToFoam") is None:
            print("Allmesh not run: gmshToFoam is not on PATH here")
            return 0
        print()
        proc = subprocess.run(["sh", "./Allmesh"], cwd=str(target), capture_output=True, text=True)
        sys.stdout.write(proc.stdout[-3000:])
        if proc.returncode != 0:
            sys.stdout.write(proc.stderr[-2000:])
            print(f"Allmesh exited {proc.returncode}; log.gmshToFoam and log.checkMesh say why")
            return proc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
