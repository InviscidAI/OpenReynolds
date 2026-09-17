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


def _index(shape):
    """Every face of the shape, once, and the identity test against it, in one object.

    `TopTools_IndexedMapOfShape` rather than a centroid match or a face index:
    `FindIndex` answers *is this face part of this shape* exactly, in one call, and
    returns 0 for a face that is not. A centroid match would have to pick a tolerance,
    and a bare face index is OpenCASCADE's own ordering, which changes when anything
    upstream of the shape does.

    **The face list is read back out of this same map**, by `FindKey`, rather than taken
    from `shape.faces()`. The two agree today on everything measured. Relying on that is
    the mistake this file exists to avoid: they are two independent traversals, one of
    them de-duplicates a face shared by two solids and the other does not, and an export
    keyed on their agreeing is correct until a conjugate case turns up and silently wrong
    afterwards.
    """
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape

    if not hasattr(shape, "wrapped"):
        raise Refused(
            f"refused: the first argument is a {type(shape).__name__}. It is one "
            "build123d shape -- the solid you are about to mesh -- not a list of faces, "
            "not a path, and not a Compound you assembled out of loose faces."
        )
    mapping = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape.wrapped, TopAbs_FACE, mapping)
    if mapping.Extent() == 0:
        raise Refused(
            "refused: that shape has no faces at all, so there is nothing to tessellate."
        )
    return mapping


def _faces_from(mapping) -> list:
    """The map's faces in index order, downcast from `TopoDS_Shape` to `TopoDS_Face`.

    `FindKey` hands back the base class; `BRep_Tool.Triangulation_s` will not take it.
    """
    from OCP.TopoDS import TopoDS

    return [TopoDS.Face_s(mapping.FindKey(i)) for i in range(1, mapping.Extent() + 1)]


def _wrap(topods):
    """A raw `TopoDS_Face` back as build123d, for the two questions only it can answer."""
    import build123d as bd

    return bd.Face(topods)


def _centre(face) -> tuple:
    """Centre of mass, for a refusal message. Never for deciding anything."""
    try:
        point = _wrap(getattr(face, "wrapped", face)).center()
        return (point.X, point.Y, point.Z)
    except Exception:  # noqa: BLE001 - only ever used to write a refusal message
        return (float("nan"),) * 3


def _assign(shape, patches: dict, mapping=None) -> list[dict]:
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

    if mapping is None:
        mapping = _index(shape)
    everything = _faces_from(mapping)
    try:
        scale = max(abs(v) for v in shape.bounding_box().size)
    except Exception:  # noqa: BLE001 - only ever scales a refusal's tolerance
        scale = 1.0
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
                    (math.dist(where, _centre(other)), i)
                    for i, other in enumerate(everything)
                ) if everything else None
                identical = near is not None and near[0] <= 1e-9 * (scale or 1.0)
                hint = "" if near is None else (
                    f" The nearest face of the shape is {near[0]:.6g} m away."
                )
                why = (
                    # Distance zero and still a different face: same model, different
                    # build. A desk that rebuilds a shape in a later cell and keeps the
                    # face handles from the earlier one lands here, and the geometry is
                    # right, so every message about drawn rectangles would mislead.
                    "That face is in the same place as one of this shape's and is still "
                    "not it, which means you are holding faces from an earlier build of "
                    "the same model -- the shape was rebuilt, or this is a copy. Face "
                    "identity does not survive that, and it is why selectors are "
                    "re-derived rather than recorded. Select the patches off the very "
                    "object you are passing as `shape`, in the same cell."
                    if identical else
                    "A face built separately -- Face(Wire.make_polygon(...)), or a face "
                    "of some earlier shape the booleans have since replaced -- shares no "
                    "edge with this solid, so its triangles cannot weld to anything and "
                    "the union gets a hole the length of its boundary. Select the patch "
                    "off the shape you are exporting: the inlet is one of "
                    "`shape.faces()`, not a rectangle drawn where the inlet is."
                )
                raise Refused(
                    f"refused: patch {name!r} was given a face at "
                    f"({', '.join(f'{v:.6g}' for v in where)}) that is not a face of the "
                    f"shape being exported.{hint}\n\n{why}"
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
            f"  area {_wrap(everything[i - 1]).area:.6g} m^2"
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

    face = getattr(face, "wrapped", face)
    location = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation_s(face, location)
    if triangulation is None:
        return []
    transform = location.Transformation()
    reversed_face = face.Orientation() == TopAbs_REVERSED
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


def _f32(value: float) -> float:
    """One coordinate as the STL will carry it.

    **STL is a single-precision format** -- a binary file stores three floats per vertex
    and nothing else -- so the surface a mesher reads is the float32 one, and that is the
    surface every number here is about. Everything is rounded once, here, before it is
    either written or counted, which is what keeps the printed topology and the files on
    disk the same object.

    Measured, this is not a detail. OpenCASCADE stores each face's nodes against that
    face's own location, so the two sides of a shared seam come back as doubles that
    differ in the last bits -- median 6.9e-18 m on a plenum with eight runners, against a
    float32 spacing of 1.9e-9 m at that scale. Counted at double precision the union of
    that patch set has 1,036 open edges; written to disk it has none, because the write
    rounds them together. Reporting the first would have told the desk its surface was
    torn while handing the mesher one that is not, and reporting it in a format-dependent
    way -- 9 significant figures in ASCII, 24 bits in binary -- would have made the same
    export two different surfaces depending on a flag.
    """
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _quantise(triangles: list) -> tuple[list, int]:
    """To float32, and without the triangles that have no area once they get there.

    OpenCASCADE emits a degenerate triangle at each pole of a sphere or a cone -- two of
    its three corners are the same point -- and rounding to float32 can collapse another
    one anywhere the mesh is finer than single precision. They carry no area and no
    information, and every downstream count is wrong while they are in: a box with a
    spherical void, closed by construction, reports **2 open edges and 2 non-manifold
    edges** purely from its two pole triangles, and with them dropped reports none.

    That number is not a nuisance, it is a trap. It is small, it is stable, it looks
    exactly like a real leak, and the corpus has three runs on record that spent their
    whole remaining step budget chasing a surface warning. A tool that manufactures one
    is worse than no tool. So they go here, and the count goes in the report, because a
    triangle silently discarded is its own way of lying.
    """
    out = []
    dropped = 0
    for triangle in triangles:
        rounded = tuple(tuple(_f32(v) for v in vertex) for vertex in triangle)
        if len(set(rounded)) < 3 or _area(rounded) == 0.0:
            dropped += 1
            continue
        out.append(rounded)
    return out, dropped


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
                # `%.17g`, not `%.9e`. The values are already float32, so nine
                # significant digits reproduce the same *float32* -- but a reader parses
                # an STL into doubles, and `preflight.read_triangles` is such a reader,
                # so nine digits hands the probes a different set of doubles than the
                # binary file would and the same export becomes two surfaces depending
                # on a flag. Seventeen round-trips a double exactly, which is what makes
                # `binary=` a choice of file format and not of geometry.
                handle.write("  vertex {:.17g} {:.17g} {:.17g}\n".format(*vertex))
            handle.write(" endloop\nendfacet\n")
        handle.write(f"endsolid {name}\n")


# -- what the union turned out to be -------------------------------------------------


def _union_topology(per_patch: dict) -> dict:  # noqa: C901 - one pass, kept flat
    """Open edges and winding of the whole patch set, welded at exact coordinates.

    Exact rather than within a tolerance, and taken on the float32 values that were
    written, so this is a statement about the files and not about an intermediate nobody
    reads. No tolerance is needed: these triangles came out of one triangulation, so the
    two sides of a shared seam round to the same three floats. A non-zero count is then a
    real feature of the solid -- an open shell, a zero-thickness baffle, two solids of a
    conjugate pair sharing a wall -- and not a seam artefact, which is what makes it
    worth printing rather than enforcing.
    """
    import numpy as np

    flat = [t for triangles in per_patch.values() for t in triangles]
    if not flat:
        return {"vertices": 0, "open_edges": 0, "non_manifold_edges": 0,
                "flipped_edges": 0}
    corners = np.asarray(flat, dtype=np.float32).reshape(-1, 3)
    _, keys = np.unique(corners, axis=0, return_inverse=True)
    keys = keys.reshape(-1, 3)
    directed = np.concatenate([keys[:, [0, 1]], keys[:, [1, 2]], keys[:, [2, 0]]])
    undirected = np.sort(directed, axis=1)
    _, walks = np.unique(undirected, axis=0, return_counts=True)
    _, same_way = np.unique(directed, axis=0, return_counts=True)
    return {
        "vertices": int(keys.max()) + 1,
        "open_edges": int((walks == 1).sum()),
        "non_manifold_edges": int((walks > 2).sum()),
        "flipped_edges": int((same_way > 1).sum()),
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

    mapping = _index(shape)
    assignment = _assign(shape, patches, mapping)
    _triangulate(shape, float(tolerance), float(angular_tolerance))

    everything = _faces_from(mapping)
    per_patch: dict[str, list] = {}
    dropped: dict[str, int] = {}
    for entry in assignment:
        triangles: list = []
        for position in entry["indices"]:
            triangles.extend(_triangles(everything[position - 1]))
        per_patch[entry["name"]], dropped[entry["name"]] = _quantise(triangles)

    empty = [name for name, triangles in per_patch.items() if not triangles]
    if empty:
        raise Refused(
            f"refused: {', '.join(repr(n) for n in empty)} came out of the tessellation "
            "with no triangles of any area, so the file would be an empty patch the "
            "mesher never finds. The faces are there and the tessellation produced "
            "nothing usable for them -- usually a face far below `tolerance` in every "
            "direction. Mesh finer, or fold it into a neighbouring patch."
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
        "faces": mapping.Extent(),
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
            "degenerate_dropped": dropped[name],
            "bytes": path.stat().st_size,
        })

    report["triangles"] = sum(p["triangles"] for p in report["patches"])
    report["degenerate_dropped"] = sum(dropped.values())
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
    if report["degenerate_dropped"]:
        lines.append(
            f"  dropped             {report['degenerate_dropped']} degenerate triangle(s) "
            "with no area -- OpenCASCADE puts one at each"
        )
        lines.append(
            "                      pole of a sphere or cone. Not a defect in your "
            "geometry, and not in the surface either."
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
