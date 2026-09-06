"""A runnable case from a solid built out of primitives, or from a STEP file.

    python3 cad_gen.py CASE --spec body.json --speed 10 --nu 1.5e-5
    python3 cad_gen.py CASE --step body.step --scale 0.001 --speed 10
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
What it does not do (yet) is prism layers: the first cell is roughly half the surface
cell, and the summary says what y+ that comes to, which is the number to judge it by.

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
system/ and constant/ files come from the same writers `snappy_gen.py` uses.
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


# -- the domain and its patches ----------------------------------------------------

def domain_bounds(built: Built, opts) -> tuple[float, ...]:
    """The flow box, in body lengths, around the body -- the same convention as
    snappy_gen: --ahead/--behind along x, --side in y, --above/--below in z."""
    x0, y0, z0, x1, y1, z1 = built.bounds
    L = max(built.extent[0], 1e-12)
    bx0, bx1 = x0 - opts["ahead"] * L, x1 + opts["behind"] * L
    by0, by1 = y0 - opts["side"] * L, y1 + opts["side"] * L
    bz0 = z0 if opts.get("ground") else z0 - opts["below"] * L
    bz1 = z1 + opts["above"] * L
    return (bx0, by0, bz0, bx1, by1, bz1)


def classify(surface_bounds, box, tol: float, ground: bool) -> str:
    """Which patch a surface of the fluid volume belongs to, from where it sits:
    flat against the upstream box face is the inlet, downstream the outlet, the
    floor is `ground` when there is one, the other box faces are `farfield`, and
    anything not on the box is the body."""
    a0, b0, c0, a1, b1, c1 = surface_bounds
    bx0, by0, bz0, bx1, by1, bz1 = box
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
        elif name == "farfield":
            roles[name] = {"kind": far}
        else:
            roles[name] = {"kind": "wall"}
    roles[body_patch] = {"kind": "wall"}
    return roles


def wall_patches(roles: dict) -> list[str]:
    return [name for name, role in roles.items() if role.get("kind") == "wall"]


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
        _, nodes = gmsh.model.mesh.getElementsByType(2, s)
        for i in range(0, len(nodes), 3):
            p, q, r = (xyz[int(nodes[i + k])] for k in range(3))
            ux, uy, uz = q[0] - p[0], q[1] - p[1], q[2] - p[2]
            vx, vy, vz = r[0] - p[0], r[1] - p[1], r[2] - p[2]
            nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
            wetted += 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)
            frontal += 0.5 * abs(nx)
    return frontal / 2.0, wetted


class MeshResult:
    def __init__(self, patches, tets, nodes, seconds, frontal, wetted):
        self.patches = patches      # name -> [surface tags]
        self.tets = int(tets)
        self.nodes = int(nodes)
        self.seconds = float(seconds)
        self.frontal = float(frontal)
        self.wetted = float(wetted)


def generate(gmsh, built: Built, opts, body_patch: str):
    """Cut the fluid out (or keep the body's own volume), name the patches, size the
    mesh finer on the body, and mesh in 3D. Returns the patches and the counts."""
    occ = gmsh.model.occ
    internal = bool(opts.get("internal"))
    tol = max(built.extent) * 1e-4 + 1e-12
    if internal:
        fluid = built.tags
        box = None
    else:
        box = domain_bounds(built, opts)
        bx0, by0, bz0, bx1, by1, bz1 = box
        tag = occ.addBox(bx0, by0, bz0, bx1 - bx0, by1 - by0, bz1 - bz0)
        fluid, _ = occ.cut([(3, tag)], built.tags)
        occ.synchronize()
        if not fluid:
            raise SystemExit("the flow box minus the body is empty; the body fills or escapes it")
    occ.synchronize()

    axis = longest_axis(built.extent)
    patches: dict[str, list[int]] = {}
    for d, s in gmsh.model.getBoundary(fluid, combined=True, oriented=False):
        sb = gmsh.model.getBoundingBox(d, s)
        name = classify_internal(sb, built.bounds, axis, tol) if internal \
            else classify(sb, box, tol, bool(opts.get("ground")))
        if name == "body":
            name = body_patch
        patches.setdefault(name, []).append(s)
    for name, tags in patches.items():
        g = gmsh.model.addPhysicalGroup(2, tags)
        gmsh.model.setPhysicalName(2, g, name)
    g = gmsh.model.addPhysicalGroup(3, [t for _, t in fluid])
    gmsh.model.setPhysicalName(3, g, "fluid")

    near, far = mesh_sizes(built, opts)
    refine_on = patches.get("walls" if internal else body_patch, [])
    L = max(built.extent[0], 1e-12)
    field = gmsh.model.mesh.field
    field.add("Distance", 1)
    field.setNumbers(1, "SurfacesList", refine_on)
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
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
    gmsh.option.setNumber("General.NumThreads", int(opts.get("threads") or os.cpu_count() or 1))

    started = time.time()
    gmsh.model.mesh.generate(3)
    seconds = time.time() - started
    tets = len(gmsh.model.mesh.getElementsByType(4)[0])
    nodes = len(gmsh.model.mesh.getNodes()[0])
    frontal, wetted = projected_and_wetted(gmsh, refine_on)
    return MeshResult(patches, tets, nodes, seconds, frontal, wetted)


# -- the files ---------------------------------------------------------------------

def allmesh(roles: dict, cores: int) -> str:
    """One command that turns body.msh into constant/polyMesh: gmshToFoam names the
    patches after the physical groups and calls every one of them `patch`, so the
    walls are retyped before anything reads the mesh; then renumber and check."""
    lines = ["#!/bin/sh", "set -e", "cd \"$(dirname \"$0\")\"",
             "gmshToFoam body.msh > log.gmshToFoam 2>&1"]
    for name in wall_patches(roles):
        lines.append(f"foamDictionary constant/polyMesh/boundary -entry entry0/{name}/type "
                     f"-set wall >> log.gmshToFoam 2>&1")
    lines += ["renumberMesh -overwrite > log.renumberMesh 2>&1",
              "checkMesh > log.checkMesh 2>&1",
              "grep -E 'cells:|Mesh OK|Failed|non-orthogonality|skewness' log.checkMesh",
              ""]
    return "\n".join(lines)


def allrun(solver: str, cores: int) -> str:
    if not solver:
        return "#!/bin/sh\n# a mesh-only case: nothing to run\n"
    return "\n".join([
        "#!/bin/sh", "set -e", "cd \"$(dirname \"$0\")\"",
        "decomposePar -force > log.decomposePar 2>&1",
        f"mpirun -np {cores} {solver} -parallel > log.{solver} 2>&1",
        "reconstructPar -latestTime > log.reconstructPar 2>&1", ""])


def case_files(plan, flow, opts, model: str, body_patch: str) -> dict[str, str]:
    solver = snappy_gen.SOLVERS[opts["study"]]
    files = {
        "system/controlDict": snappy_gen.control_dict(opts, flow, body_patch),
        "system/fvSchemes": snappy_gen.fv_schemes(opts),
        "system/fvSolution": snappy_gen.fv_solution(opts, model),
        "system/decomposeParDict": snappy_gen.decompose_dict(opts["cores"]),
        "constant/turbulenceProperties": case_gen.turbulence_properties(model),
        "constant/transportProperties": case_gen.transport_properties(flow),
        "0/p": case_gen.field_p(plan),
        "0/U": case_gen.field_U(plan, flow),
        "Allmesh": allmesh(plan.roles, opts["cores"]),
        "Allrun": allrun(solver, opts["cores"]),
        "case.foam": "",
    }
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
            source: str, notes: list[str]) -> list[str]:
    dx, dy, dz = built.extent
    lines = [f"geometry   {source}",
             f"           extent {dx:.4g} x {dy:.4g} x {dz:.4g} m, volume {built.volume:.4g} m3"]
    if opts.get("internal"):
        along = "xyz"[longest_axis(built.extent)]
        lines.append(f"domain     the body's own volume (internal flow), running along {along}: "
                     f"inlet at low {along}, outlet at high {along}")
    else:
        bx0, by0, bz0, bx1, by1, bz1 = domain_bounds(built, opts)
        lines.append(f"domain     x {bx0:.4g}..{bx1:.4g}  y {by0:.4g}..{by1:.4g}  z {bz0:.4g}..{bz1:.4g}"
                     + ("  (floor is a wall)" if opts.get("ground") else ""))
    near, far = mesh_sizes(built, opts)
    lines.append(f"cells      {near:.4g} m on the body, {far:.4g} m far away; no prism layers")
    if mesh is not None:
        lines.append(f"mesh       {mesh.tets:,} tetrahedra, {mesh.nodes:,} nodes, "
                     f"{mesh.seconds:.2f} s to make")
        lines.append("patches    " + ", ".join(f"{k} ({len(v)})" for k, v in mesh.patches.items()))
        lines.append(f"area       frontal {mesh.frontal:.4g} m2, wetted {mesh.wetted:.4g} m2 "
                     f"(off the surface triangles)")
    lines.append(f"flow       {flow.speed:g} m/s, L {flow.length:.4g} m, nu {flow.nu:.3g} -> "
                 f"Re {flow.reynolds:,.0f} ({flow.derived} follows)")
    lines.append(f"model      {model} -- {why}")
    y_plus = case_gen.estimate_y_plus(near / 2.0, flow)
    lines.append(f"y+         about {y_plus:,.0f} at the first cell (half the surface cell); "
                 + ("wall functions bridge it" if y_plus >= 30 else
                    "under the log layer, so the wall function is nutUSpalding's blend"))
    solver = snappy_gen.SOLVERS[opts["study"]]
    lines.append(f"run        {opts['study']}" + (f" -> {solver}" if solver else ""))
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
                          "with the inlet at its low-x face and the outlet at its high-x face, "
                          "rather than the air around it.")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="Build the solid and report; mesh and write nothing.")
    ap.add_argument("--mesh", action="store_true",
                    help="After writing, run Allmesh here (needs gmshToFoam on PATH).")
    ap.add_argument("--force", action="store_true")

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
    run.add_argument("--study", default="steady", choices=["mesh", "steady", "transient"])
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

    mesh = ap.add_argument_group("the mesh")
    mesh.add_argument("--surface-cell", type=float, default=None, dest="surface_cell",
                      help="Cell size on the body, m (default: body length / 40).")
    mesh.add_argument("--far-cell", type=float, default=None, dest="far_cell",
                      help="Cell size at the far boundary, m (default: 8x the surface cell).")
    mesh.add_argument("--growth", type=float, default=1.0,
                      help="Stretch the refinement zone around the body by this factor.")
    mesh.add_argument("--threads", type=int, default=None, help="gmsh threads (default: all).")

    forces = ap.add_argument_group("the forces")
    forces.add_argument("--area", default="frontal", choices=["frontal", "wetted"])
    forces.add_argument("--ref-area", type=float, default=None, dest="ref_area")

    args = ap.parse_args(argv)
    if (args.spec is None) == (args.step is None):
        ap.error("one of --spec or --step")
    if args.case is None and not args.dry_run:
        ap.error("a directory to write the case into, or --dry-run")

    opts = vars(args)
    opts["thermal"] = False
    if opts["nu"] is None and opts["reynolds"] is None:
        opts["nu"] = 1.5e-5
    notes: list[str] = []

    try:
        import gmsh  # the image carries the Python module; the binary alone is not enough
    except ImportError:
        raise SystemExit("the gmsh Python module is not importable here; on the instance it "
                         "is, and `python3 -c 'import gmsh'` is the check")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
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

        built = measure(gmsh, tags)
        if built.volume <= 0:
            raise SystemExit("the solid has no volume")
        opts["_l_ref"] = args.length or built.extent[0]
        flow_state = case_gen.derive_flow(opts, opts["_l_ref"])
        model, why = case_gen.turbulence_model(opts, flow_state)
        body_patch = "walls" if args.internal else args.body_name

        if args.dry_run:
            for line in summary(built, None, flow_state, opts, model, why, source, notes):
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

        result = generate(gmsh, built, opts, body_patch)
        if args.internal and "inlet" not in result.patches:
            notes.append("!! no face sits flat at the body's low-x end, so there is no inlet")
        if not args.internal and body_patch not in result.patches:
            raise SystemExit("no surface was left for the body; it sits outside the flow box")

        if args.ref_area:
            opts["_a_ref"], opts["_a_ref_why"] = args.ref_area, "--ref-area"
        elif args.area == "wetted":
            opts["_a_ref"], opts["_a_ref_why"] = result.wetted, "wetted, measured"
        else:
            opts["_a_ref"], opts["_a_ref_why"] = result.frontal, "frontal, measured"

        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.write(str(target / "body.msh"))

        direction = [0.0, 0.0, 0.0]
        direction[longest_axis(built.extent) if args.internal else 0] = 1.0
        plan = case_gen.Plan(snappy_gen.PatchList(list(result.patches)),
                             build_roles(result.patches, opts, body_patch, tuple(direction)),
                             opts["_l_ref"], {}, notes)
        files = case_files(plan, flow_state, opts, model, body_patch)
        case_gen.write_case(target, files)
        (target / "Allmesh").chmod(0o755)
        (target / "Allrun").chmod(0o755)

        for line in summary(built, result, flow_state, opts, model, why, source, notes):
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
