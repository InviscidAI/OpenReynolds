#!/usr/bin/env python3
"""Named patches out of one build123d shape, tessellated once so the seams weld.

    from cad_export import export_patches
    export_patches(fluid, {"inlet": inlet_faces, "outlet": outlet_faces,
                           "walls": ...}, tolerance=2.5e-4)

`fluid` is one shape. Each patch names **faces of that shape** -- `fluid.faces()`, or a
selection off it -- and exactly one patch may be `...`, meaning whatever is left. Every
face lands in exactly one patch or the call refuses, before a triangle exists.

**Why this exists rather than a loop over `export_stl`.** `export_stl` meshes the shape
you hand it. Hand it the patches one at a time and each one is meshed on its own, so two
patches that share an edge discretise it separately and the union they form has a torn
seam down every boundary between them -- open edges snappyHexMesh reads as holes,
triangles wound against their neighbours, and a `checkMesh` that passes anyway because
the volume mesh is closed by construction whatever the surface did. Measured on one
fused solid of 27 faces, welding at exact coordinates:

    one export_stl per group, same tolerance     0 open edges
    one export_stl per group, tolerance differs  4,536
    one export_stl per face                      4,284
    a patch built as a loose Face, not off the solid    882

The first row is the trap. It is not correct, it is *lucky*: OpenCASCADE caches a
triangulation on the shape, so a second call at the same deflection reuses the edge
polygons the first one wrote. Change the tolerance between patches, split per face, or
introduce a face that is not a face of the solid, and the luck runs out -- which is why
this failure recurs on 15 of 26 cases across the sweeps and never reproduces cleanly.

Here the shape is meshed **once** and each patch is written off that single
triangulation, so the seams weld by construction and not by cache. There is no per-patch
tolerance to get wrong, and a face that is not a face of the shape is refused with its
distance to the nearest real one.

It also reports what it wrote: triangles and area per patch, the bounding box in metres,
and the open-edge and winding counts of the union. Those last two are printed, never
enforced -- a zero-thickness baffle is open on purpose, and a gate that cannot tell it
from a leak is a gate that blocks correct work.

`patches.json` goes in beside the files, naming which file is which patch.

There is no command line: the argument is a live build123d shape and its faces, which
only exist inside the session that built them. The whole of it is readable without one --

    python3 -c "import cad_export; help(cad_export.export_patches)"
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

__all__ = ["export_patches", "Refused", "render"]

DEFAULT_OUT = "constant/triSurface"
"""Where OpenFOAM looks. Named so the common call takes no path at all."""

ANGULAR_DEFAULT = 0.1
"""Radians of surface turn per facet -- OpenCASCADE's angular deflection.

It is what puts chords round a circle too tight for `tolerance` to notice, the same job
`Mesh.MeshSizeFromCurvature` does on the gmsh path. 0.1 rad is ~63 facets on a full
circle."""


class Refused(Exception):
    """An export that would have written a plausible-looking wrong surface."""


# -- the partition, asserted before anything is tessellated -------------------------


def _faces_of(shape) -> list:
    """Every face of the shape, in OpenCASCADE's order, as build123d wraps them."""
    try:
        return list(shape.faces())
    except AttributeError as exc:
        raise Refused(
            f"refused: {type(shape).__name__} has no .faces(). The first argument is one "
            "build123d shape -- the solid you are about to mesh -- not a list of faces "
            "and not a path."
        ) from exc


def _index(shape):
    """A map from a face to its position in `shape`, and the identity test with it.

    `TopTools_IndexedMapOfShape` rather than a centroid match or a face index: it answers
    *is this face part of this shape* exactly, in one call, and returns 0 for a face that
    is not. A centroid match would have to pick a tolerance, and a face index is
    OpenCASCADE's own ordering, which changes when anything upstream of the shape does.
    """
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape

    mapping = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape.wrapped, TopAbs_FACE, mapping)
    return mapping


def _centre(face) -> tuple:
    try:
        point = face.center()
        return (point.X, point.Y, point.Z)
    except Exception:  # noqa: BLE001 - only ever used to write a refusal message
        return (float("nan"),) * 3


def _assign(shape, patches: dict) -> list[dict]:
    """Faces to patches: exhaustive, disjoint, and every face proved to be this shape's.

    Returns one entry per patch in the order given. Raises rather than guessing, in the
    three cases where guessing produces a surface that meshes and is wrong:

    * a face that belongs to some other shape. It cannot share an edge with anything
      here, so it cannot weld, and the union gets a hole the length of its boundary.
      This is the loose-`Face` patch of the header table, and it is what a duct wall
      built with `Wire.make_polygon` beside the solid is;
    * a face in two patches -- two surfaces in one place, and snappyHexMesh snaps to
      whichever it reads last;
    * a face in none. It is not dropped: it lands in the mesher's default patch and
      takes that patch's boundary condition, and the run finishes.
    """
    if not isinstance(patches, dict) or not patches:
        raise Refused(
            "refused: `patches` is a dict of patch name to the faces of that patch -- "
            '{"inlet": [...], "walls": ...}. One patch may be `...`, which means '
            "whatever is left over."
        )

    mapping = _index(shape)
    everything = _faces_of(shape)
    claimed: dict[int, str] = {}
    out: list[dict] = []
    remainder = None

    for name, given in patches.items():
        name = str(name)
        if "/" in name or not name.strip():
            raise Refused(f"refused: {name!r} is not usable as a patch name or a filename.")
        if given is Ellipsis or given is None:
            if remainder is not None:
                raise Refused(
                    f"refused: {remainder!r} and {name!r} both ask for whatever is left. "
                    "Which of them an unclaimed face goes to would be a coin toss."
                )
            remainder = name
            out.append({"name": name, "indices": [], "remainder": True})
            continue
        if not isinstance(given, (list, tuple, set)):
            given = [given]
        indices: list[int] = []
        for face in given:
            wrapped = getattr(face, "wrapped", face)
            position = mapping.FindIndex(wrapped)
            if position == 0:
                where = _centre(face)
                near = min(
                    (math.dist(where, _centre(other)), other) for other in everything
                ) if everything else None
                hint = "" if near is None else (
                    f" The nearest face of the shape is {near[0]:.6g} m away."
                )
                raise Refused(
                    f"refused: patch {name!r} was given a face at "
                    f"({', '.join(f'{v:.6g}' for v in where)}) that is not a face of the "
                    f"shape being exported.{hint}\n\n"
                    "A face built separately -- Face(Wire.make_polygon(...)), or a face "
                    "of some earlier shape the booleans have since replaced -- shares no "
                    "edge with this solid, so its triangles cannot weld to anything and "
                    "the union gets a hole the length of its boundary. Select the patch "
                    "off the shape you are exporting: the inlet is one of "
                    "`shape.faces()`, not a rectangle drawn where the inlet is."
                )
            if position in claimed:
                raise Refused(
                    f"refused: the face at "
                    f"({', '.join(f'{v:.6g}' for v in _centre(face))}) is in both "
                    f"{claimed[position]!r} and {name!r}. A face in two patches is two "
                    "surfaces in one place."
                )
            claimed[position] = name
            indices.append(position)
        if not indices:
            raise Refused(
                f"refused: patch {name!r} was given no faces. An empty patch file is a "
                "patch the mesher will not find, and the boundary it was meant to be "
                "becomes somebody else's default. Drop the name, or select for it again."
            )
        out.append({"name": name, "indices": indices, "remainder": False})

    unclaimed = [i for i in range(1, mapping.Extent() + 1) if i not in claimed]
    if remainder is not None:
        for entry in out:
            if entry["remainder"]:
                entry["indices"] = unclaimed
        if not unclaimed:
            raise Refused(
                f"refused: patch {remainder!r} asks for whatever is left and the other "
                "patches already name every face, so it would be written empty."
            )
        unclaimed = []
    if unclaimed:
        listed = "\n".join(
            f"  ({', '.join(f'{v:.6g}' for v in _centre(everything[i - 1]))})"
            f"  area {everything[i - 1].area:.6g} m^2"
            for i in unclaimed[:20]
        )
        raise Refused(
            f"refused: {len(unclaimed)} of {mapping.Extent()} faces are in no patch.\n\n"
            f"Their centres, in metres:\n{listed}"
            + ("\n  ..." if len(unclaimed) > 20 else "")
            + "\n\nAn unnamed face is not dropped -- it lands in whatever patch the "
            "mesher defaults to and takes that patch's boundary condition, and the run "
            "finishes looking fine. Name it, or give one patch `...` to take the rest."
        )
    return out


# -- the one tessellation ------------------------------------------------------------


def _triangulate(shape, tolerance: float, angular_tolerance: float) -> None:
    """Mesh the whole shape once. Everything written afterwards reads this.

    `BRepMesh_IncrementalMesh` meshes each edge once and hands the same polygon to both
    faces that share it, which is precisely the property the per-patch loop loses.
    `isRelative=False` so `tolerance` is metres of chord deviation and not a fraction of
    a bounding box that changes with the model.
    """
    from OCP.BRepMesh import BRepMesh_IncrementalMesh

    BRepMesh_IncrementalMesh(shape.wrapped, tolerance, False, angular_tolerance, True)


def _triangles(face) -> list[tuple]:
    """This face's share of the shape's triangulation, in world coordinates.

    Wound to match the face's own orientation: OpenCASCADE stores a triangulation
    against the underlying surface, and a face the booleans left `REVERSED` has to be
    written the other way round or its patch points into the solid while its neighbours
    point out -- which is the winding failure, arriving by a different door.
    """
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_REVERSED
    from OCP.TopLoc import TopLoc_Location

    location = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation_s(face.wrapped, location)
    if triangulation is None:
        return []
    transform = location.Transformation()
    reversed_face = face.wrapped.Orientation() == TopAbs_REVERSED
    nodes = [
        triangulation.Node(i).Transformed(transform)
        for i in range(1, triangulation.NbNodes() + 1)
    ]
    out = []
    for i in range(1, triangulation.NbTriangles() + 1):
        a, b, c = triangulation.Triangle(i).Get()
        if reversed_face:
            b, c = c, b
        out.append(tuple(
            (nodes[k - 1].X(), nodes[k - 1].Y(), nodes[k - 1].Z()) for k in (a, b, c)
        ))
    return out


def _normal(triangle) -> tuple:
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = triangle
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length == 0.0:
        return (0.0, 0.0, 0.0)
    return (nx / length, ny / length, nz / length)


def _area(triangle) -> float:
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = triangle
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    return 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)


def _write(path: Path, name: str, triangles: list, binary: bool) -> None:
    if binary:
        with path.open("wb") as handle:
            handle.write(name.encode("ascii", "replace")[:80].ljust(80, b"\0"))
            handle.write(struct.pack("<I", len(triangles)))
            for triangle in triangles:
                handle.write(struct.pack("<3f", *_normal(triangle)))
                for vertex in triangle:
                    handle.write(struct.pack("<3f", *vertex))
                handle.write(b"\0\0")
        return
    with path.open("w") as handle:
        handle.write(f"solid {name}\n")
        for triangle in triangles:
            handle.write("facet normal {:.9e} {:.9e} {:.9e}\n outer loop\n".format(*_normal(triangle)))
            for vertex in triangle:
                handle.write("  vertex {:.9e} {:.9e} {:.9e}\n".format(*vertex))
            handle.write(" endloop\nendfacet\n")
        handle.write(f"endsolid {name}\n")


# -- what the union turned out to be -------------------------------------------------


def _union_topology(per_patch: dict) -> dict:
    """Open edges and winding of the whole patch set, welded at exact coordinates.

    Exact rather than within a tolerance, and that is the point: these triangles came
    out of one triangulation, so a shared seam is bit-identical and no tolerance is
    needed to see it. A non-zero count here is a real feature of the solid -- an open
    shell, a zero-thickness baffle -- and not a seam artefact, which is what makes it
    worth printing.
    """
    vertices: dict[tuple, int] = {}
    edges: dict[tuple, int] = {}
    directed: dict[tuple, int] = {}
    for triangles in per_patch.values():
        for triangle in triangles:
            keys = []
            for vertex in triangle:
                key = vertices.setdefault(vertex, len(vertices))
                keys.append(key)
            for a, b in ((keys[0], keys[1]), (keys[1], keys[2]), (keys[2], keys[0])):
                edges[(min(a, b), max(a, b))] = edges.get((min(a, b), max(a, b)), 0) + 1
                directed[(a, b)] = directed.get((a, b), 0) + 1
    return {
        "vertices": len(vertices),
        "open_edges": sum(1 for n in edges.values() if n == 1),
        "non_manifold_edges": sum(1 for n in edges.values() if n > 2),
        "flipped_edges": sum(1 for n in directed.values() if n > 1),
    }


# -- the export ----------------------------------------------------------------------


def export_patches(
    shape,
    patches: dict,
    out_dir=DEFAULT_OUT,
    *,
    tolerance: float | None = None,
    angular_tolerance: float = ANGULAR_DEFAULT,
    location_in_mesh=None,
    binary: bool = True,
    clean: bool = True,
    quiet: bool = False,
) -> dict:
    """One shape to one STL per named patch, tessellated once, with the numbers printed.

        export_patches(fluid, {"inlet": ins, "outlet": outs, "walls": ...},
                       tolerance=2.5e-4)

    `shape`      one build123d shape -- the solid whose surface is the mesh boundary.
    `patches`    patch name -> the faces of `shape` in it. Exactly one patch may be
                 `...`, meaning every face the others did not name.
    `out_dir`    `constant/triSurface` unless you say otherwise.
    `tolerance`  max chord deviation in **metres**: how far a facet may sit from the
                 surface it stands for. Follow the mesh -- roughly a quarter of the
                 finest surface cell -- and no default is offered, because faceting
                 coarser than the mesh becomes the geometry and reaches the pressure
                 field looking like physics.
    `clean`      clear STLs this call did not write out of `out_dir`. A file left from
                 an abandoned route is welded in by every downstream reader as though
                 it were part of the surface.

    Returns the report and prints it. Raises `Refused` -- never a silent guess.
    """
    out_dir = Path(out_dir)
    if tolerance is None or tolerance <= 0:
        raise Refused(
            "refused: no `tolerance` given, and there is no default worth guessing.\n\n"
            "It is the largest distance a facet may sit from the surface it stands for, "
            "in metres, and it is the decision this call exists to make explicit. Facet "
            "finer than the mesh and you pay for triangles snappyHexMesh cannot resolve; "
            "facet coarser and the faceting *is* the geometry -- the mesh reproduces the "
            "flat spots faithfully and they arrive in the pressure field looking like "
            "physics. A quarter of the finest surface cell is a reasonable start."
        )

    assignment = _assign(shape, patches)
    _triangulate(shape, float(tolerance), float(angular_tolerance))

    everything = _faces_of(shape)
    per_patch: dict[str, list] = {}
    for entry in assignment:
        triangles: list = []
        for position in entry["indices"]:
            triangles.extend(_triangles(everything[position - 1]))
        per_patch[entry["name"]] = triangles

    empty = [name for name, triangles in per_patch.items() if not triangles]
    if empty:
        raise Refused(
            f"refused: {', '.join(repr(n) for n in empty)} came out of the tessellation "
            "with no triangles at all, so the file would be an empty patch the mesher "
            "never finds. The faces are there and OpenCASCADE produced nothing for them "
            "-- usually a face far below `tolerance` in every direction. Mesh finer, or "
            "fold it into a neighbouring patch."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    written = {f"{entry['name']}.stl" for entry in assignment}
    removed = []
    if clean:
        for stale in sorted(out_dir.glob("*.stl")):
            if stale.name not in written:
                stale.unlink()
                removed.append(stale.name)

    report: dict = {
        "out_dir": str(out_dir),
        "tolerance": float(tolerance),
        "angular_tolerance": float(angular_tolerance),
        "faces": len(everything),
        "removed": removed,
        "patches": [],
    }
    low = [math.inf] * 3
    high = [-math.inf] * 3
    for entry in assignment:
        name = entry["name"]
        triangles = per_patch[name]
        path = out_dir / f"{name}.stl"
        _write(path, name, triangles, binary)
        for triangle in triangles:
            for vertex in triangle:
                for axis in range(3):
                    low[axis] = min(low[axis], vertex[axis])
                    high[axis] = max(high[axis], vertex[axis])
        report["patches"].append({
            "name": name,
            "file": path.name,
            "faces": len(entry["indices"]),
            "triangles": len(triangles),
            "area_m2": sum(_area(t) for t in triangles),
            "bytes": path.stat().st_size,
        })

    report["triangles"] = sum(p["triangles"] for p in report["patches"])
    report["area_m2"] = sum(p["area_m2"] for p in report["patches"])
    report["bounds_m"] = ([round(v, 12) for v in low + high]
                          if report["triangles"] else [])
    report["extent_m"] = ([round(high[i] - low[i], 12) for i in range(3)]
                          if report["triangles"] else [])
    report["union"] = _union_topology(per_patch)
    if location_in_mesh is not None:
        location_in_mesh = [float(v) for v in location_in_mesh]
        if len(location_in_mesh) != 3:
            raise Refused(
                f"refused: location_in_mesh is {location_in_mesh!r}, which is not one "
                "(x, y, z) in metres."
            )
    report["location_in_mesh"] = location_in_mesh

    manifest = {
        "unit_metres": 1.0,
        "source": "cad_export.export_patches",
        "patches": [
            {"name": p["name"], "file": p["file"], "triangles": p["triangles"],
             "area_m2": p["area_m2"]}
            for p in report["patches"]
        ],
        "location_in_mesh": location_in_mesh,
    }
    (out_dir / "patches.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report["manifest"] = str(out_dir / "patches.json")

    if not quiet:
        print(render(report))
    return report


def render(report: dict) -> str:
    """The report as text. This is what has to reach the transcript."""
    lines = [
        f"{report['out_dir']}  <-  {report['faces']} faces, one tessellation",
        f"  tolerance           {report['tolerance']:.6g} m chord deviation, "
        f"{report['angular_tolerance']:g} rad angular",
    ]
    for patch in report["patches"]:
        lines.append(
            f"    {patch['name']:<22} {patch['faces']:>4} faces  "
            f"{patch['triangles']:>9,} triangles  {patch['area_m2']:.6g} m^2  "
            f"-> {patch['file']}"
        )
    extent = report["extent_m"]
    if extent:
        lines.append(
            f"  extent              {extent[0]:.6g} x {extent[1]:.6g} x {extent[2]:.6g} m"
            "   <- metres, so check these against the request"
        )
    union = report["union"]
    lines.append(
        f"  union               {report['triangles']:,} triangles, "
        f"{union['open_edges']} open edges, {union['flipped_edges']} flipped, "
        f"{union['non_manifold_edges']} non-manifold"
    )
    if union["open_edges"] or union["flipped_edges"] or union["non_manifold_edges"]:
        lines.append(
            "                      reported, not enforced: on one tessellation a seam "
            "cannot tear, so this is"
        )
        lines.append(
            "                      the solid itself -- an open shell, a baffle, a "
            "self-touching boolean. Say which."
        )
    if report["removed"]:
        lines.append(
            f"  cleared             {', '.join(report['removed'])} "
            "(not written by this call)"
        )
    point = report.get("location_in_mesh")
    lines.append(
        "  location_in_mesh    "
        + (", ".join(f"{v:.6g}" for v in point) if point else "not given")
    )
    lines.append(f"  manifest            {report['manifest']}")
    return "\n".join(lines)
