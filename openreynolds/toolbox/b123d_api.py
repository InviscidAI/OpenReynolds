"""What build123d offers, grouped by what each thing is for.

    python3 b123d_api.py                 # write b123d_api.md beside this script
    python3 b123d_api.py --out FILE      # somewhere else
    python3 b123d_api.py --check         # names that no longer resolve, exit 2 if any
    python3 b123d_api.py --stdout        # print it instead of writing it

The provenance claim is **curated selection, introspected content, verified live**, not
"generated, never hand-written". The selection below is a reading list: somebody decided
that `fillet` and `filter_by` matter to this desk and that `BRepAlgoAPI_Defeaturing`,
`cos` and `BuildPart` do not. What is introspected is the *content* of each entry -- the
call signature comes off the installed library, so it cannot describe a parameter that was
renamed two releases ago -- and `--check` is the verification half: every name here is
resolved against the library that is actually importable, and one that has gone is an
error rather than a plausible-looking line.

Three tiers, and this file is the middle one. The brief carries one line pointing here.
This file, at ~15-20 KB, is read on demand and searched with `grep`; there is deliberately
no search script, because `grep` is already in the kernel. And `help(build123d.fillet)`
and `inspect.signature(build123d.fillet)` in the kernel are the third tier -- the exact
call, the full docstring, live against the installed version, which is why the job of this
file is **discovery** (knowing a callable exists, in order to introspect it) rather than
reference.

Three cuts made on purpose, all of them the same decision:

* **Builder mode is absent.** No `BuildPart`, `BuildSketch`, `BuildLine`, and the `mode=`
  parameter is stripped out of every signature below. Algebra mode was chosen because
  every operation names its operands and there is no ambient context to replay; offering
  the builder surface beside it invites exactly the hidden-state failures the choice was
  made to avoid.
* **Raw OCCT is absent.** `BRep*`, `Bnd_*`, `TopoDS*`, `gp_*` are the kernel underneath,
  reachable and occasionally necessary, and not what this list is for.
* **Standard-library leakage is absent.** `cos`, `sqrt`, `dataclass`, `Any` arrive in
  `dir(build123d)` by re-export and are not build123d's API.

The boolean operators are here and could not have been discovered by any introspection
pass, because they are dunders: `+`, `-` and `&` *are* algebra mode.
"""

from __future__ import annotations

import argparse
import enum
import importlib
import inspect
import re
import sys
from pathlib import Path

OCCT_PATTERN = re.compile(r"\b(BRep\w*|Bnd_\w*|TopoDS\w*|gp_\w+)\b")

DEFAULT_OUT = Path(__file__).resolve().parent / "b123d_api.md"

HEADER = """\
# build123d, grouped by what it is for

A reading list, not a reference. Every name below is checked to exist in the build123d
that is installed here; the signatures are read off that same library. What it does not
carry is the exact typed signature and the full docstring, because those are one call
away in your own kernel and cannot go stale there:

```python
import build123d as bd, inspect
inspect.signature(bd.fillet)      # the exact call, annotations and all
help(bd.ShapeList.filter_by)      # the whole docstring, with its examples
```

So use this to find out that something exists, then ask the kernel what it takes.

`grep -n fillet /work/.toolbox/b123d_api.md` is the search; there is no search script
because `!grep` in a cell is already better than one.

Signatures below are shown as parameter names and defaults, with annotations dropped for
width and the builder-mode `mode=` parameter dropped entirely. This is algebra mode:
operations name their operands, nothing is pending, and nothing is inherited from an
enclosing context.
"""

# -- the curated selection -----------------------------------------------------
#
# (display, target, purpose). `target` is the dotted name resolved against build123d;
# where it is "" the display is the name. Every one of these is checked by --check.

OPERATORS = [
    ("`a + b`", "Shape.__add__",
     "fuse: the union of two shapes, as one shape. `Part() + Box(...)` grows material."),
    ("`a - b`", "Shape.__sub__",
     "cut: a with b removed. Does not commute -- `a - b` is not `b - a` -- and this is "
     "the operation the fluid domain is extracted with."),
    ("`a & b`", "Shape.__and__",
     "intersect: only what is in both. Interference between two solids is "
     "`(a & b).volume`, and zero means they do not touch."),
    ("`loc * shape`", "Location.__mul__",
     "place: `Pos(0, 0, 10) * part` moves a copy, `Plane.XZ * sketch` puts a sketch on a "
     "plane. The shape on the right is not modified."),
]

GROUPS: list[tuple[str, str, list[tuple[str, str, str]]]] = [
    (
        "Part operations",
        "Each takes its operands by name and returns a new shape. Nothing is pending and "
        "nothing is consumed from an earlier line.",
        [
            ("extrude", "", "a face or sketch grown into a solid, by `amount` or `until` "
                            "another object."),
            ("revolve", "", "a profile swept about an axis; `revolution_arc` short of 360 "
                            "gives a wedge."),
            ("loft", "", "a solid through a series of sections in order."),
            ("sweep", "", "a section carried along a path; `is_frenet` for a path that "
                          "twists."),
            ("offset", "", "a shell, a hollow, or a grown/shrunk copy: `openings` names "
                           "the faces to leave off, which is how a box becomes a duct."),
            ("fillet", "", "a radius on the named edges or vertices -- pass the "
                           "`ShapeList` you selected, not an index."),
            ("chamfer", "", "a flat on the named edges; `length2` and `angle` for an "
                            "asymmetric one."),
            ("draft", "", "a taper applied to faces for moulding."),
            ("thicken", "", "a face or shell given a wall thickness, normal to itself."),
            ("split", "", "a shape cut by a plane; `keep` says which side survives, or "
                          "both."),
            ("section", "", "the planar faces where a plane crosses a solid -- the cheap "
                            "way to see the inside of an import."),
            ("mirror", "", "a reflected copy about a plane. Symmetry built rather than "
                           "re-authored."),
            ("scale", "", "a resized copy, uniform or per-axis. Not the unit fix: a "
                          "millimetre import is scaled by 0.001 here, once, early."),
            ("project", "", "faces, edges or points thrown onto a plane or a target "
                            "shape along a direction."),
            ("make_face", "", "a closed set of edges or wires turned into a face."),
            ("make_hull", "", "the convex hull of edges -- a quick outer envelope for a "
                              "flow box."),
            ("full_round", "", "a tangent-continuous round replacing two edges and the "
                               "face between them."),
            ("add", "", "a shape folded into another, type by type."),
            ("Until", "", "argument to `extrude`: NEXT, LAST, PREVIOUS, FIRST -- extrude "
                          "to geometry rather than to a number you measured by hand."),
            ("Keep", "", "argument to `split`: TOP, BOTTOM, BOTH, INSIDE, OUTSIDE."),
            ("Kind", "", "argument to `offset`: ARC, INTERSECTION, TANGENT -- how corners "
                         "are carried out."),
            ("Side", "", "argument to `offset` on 2D: LEFT, RIGHT, BOTH."),
        ],
    ),
    (
        "Solid primitives",
        "Combined with the operators above. Every one takes `align` and `rotation`, so "
        "placement is part of the call rather than a later move.",
        [
            ("Box", "", "length, width, height."),
            ("Cylinder", "", "radius and height; `arc_size` under 360 for a pie slice."),
            ("Sphere", "", "radius, and the three arc angles for a partial one."),
            ("Cone", "", "bottom radius, top radius, height -- top radius 0 for a point."),
            ("Torus", "", "major and minor radius."),
            ("Wedge", "", "a tapered box, six lengths."),
            ("Hole", "", "a through hole as a shape to subtract."),
            ("CounterBoreHole", "", "a hole with a flat-bottomed recess, as a shape to "
                                    "subtract."),
            ("CounterSinkHole", "", "a hole with a conical recess, as a shape to "
                                    "subtract."),
        ],
    ),
    (
        "Sketch primitives (2D)",
        "A sketch is a face. Place it with `Plane.XZ * sketch`, then extrude or revolve "
        "it.",
        [
            ("Rectangle", "", "width and height."),
            ("RectangleRounded", "", "the same with a corner radius, without a fillet "
                                     "call."),
            ("Circle", "", "radius."),
            ("Ellipse", "", "two radii."),
            ("Polygon", "", "an explicit list of points."),
            ("RegularPolygon", "", "radius and side count; `major_radius` says whether "
                                   "that is across corners or across flats."),
            ("Triangle", "", "any three of sides and angles -- the rest solved for you."),
            ("Trapezoid", "", "width, height, and the two base angles."),
            ("SlotOverall", "", "a slot by its total length and height."),
            ("SlotCenterToCenter", "", "a slot by the distance between its end centres."),
            ("SlotCenterPoint", "", "a slot from a centre, a point and a height."),
            ("SlotArc", "", "a slot following an arc."),
            ("Text", "", "characters as a face -- labels cut into a part."),
        ],
    ),
    (
        "Curves and edges",
        "Built into a `Curve`, closed, and turned into a face with `make_face` when the "
        "outline is not one of the primitives above.",
        [
            ("Line", "", "two points."),
            ("Polyline", "", "a run of points; `close=True` to shut it."),
            ("Spline", "", "through points, with optional end tangents."),
            ("CenterArc", "", "centre, radius, start angle, arc size."),
            ("ThreePointArc", "", "through three points."),
            ("RadiusArc", "", "two points and a radius."),
            ("TangentArc", "", "two points, leaving the first along a given tangent."),
            ("EllipticalCenterArc", "", "an elliptical arc about a centre."),
            ("JernArc", "", "an arc that starts tangent to where you are -- the one that "
                            "chains without recomputing a centre."),
            ("Helix", "", "pitch, height, radius; the path for a threaded sweep."),
            ("Bezier", "", "control points, with weights for a rational curve."),
            ("FilletPolyline", "", "a polyline with every corner already rounded."),
            ("trace", "", "a line given a width, as a face."),
        ],
    ),
    (
        "Placement: planes, axes, locations",
        "Where a thing is, said once at the call rather than fixed up afterwards.",
        [
            ("Plane.XY", "Plane.XY", "the ground plane; also `Plane.XZ`, `Plane.YZ`."),
            ("Plane.front", "Plane.front", "named views -- `front`, `back`, `top`, "
                                           "`bottom`, `left`, `right`, `isometric`."),
            ("Plane(face)", "Plane", "a plane taken from a planar face, so a sketch can "
                                     "be placed on geometry you selected rather than on "
                                     "coordinates you assumed."),
            ("Plane.offset", "Plane.offset", "a parallel plane a distance along its own "
                                             "normal."),
            ("Plane.rotated", "Plane.rotated", "a plane turned about its own axes."),
            ("Plane.location", "Plane.location", "the `Location` a plane represents."),
            ("Axis.X", "Axis.X", "the world axes; also `Axis.Y`, `Axis.Z`."),
            ("Axis(origin, direction)", "Axis", "an arbitrary axis -- the argument "
                                                "`revolve`, `filter_by` and `sort_by` all "
                                                "take."),
            ("Location", "", "a position and an orientation; multiply it onto a shape to "
                             "place a copy."),
            ("Pos", "", "shorthand `Location` for a pure translation: `Pos(0, 0, 10) * "
                        "part`."),
            ("Rot", "", "shorthand `Location` for a pure rotation, in degrees."),
            ("Rotation", "", "a rotation about X, Y, Z in degrees, usable as a `Location`."),
            ("Vector", "", "a 3D vector with `length`, `normalized`, `cross`, `dot` -- "
                           "what every `center()` and `normal_at` returns."),
            ("Align", "", "MIN, CENTER, MAX per axis: where a primitive sits relative to "
                          "its own origin."),
        ],
    ),
    (
        "Selecting: ShapeList and its query methods",
        "This is the tagging idiom, and the one that has to be stable. Every selector is "
        "re-derived from geometry on each run -- a face index from a previous session is "
        "not a selector and does not survive a rebuild.",
        [
            ("faces", "Shape.faces", "every face, as a `ShapeList`. Also `edges`, "
                                     "`vertices`, `solids`, `wires`, `shells`."),
            ("face", "Shape.face", "the single face, raising if there is not exactly one. "
                                   "Also `edge`, `vertex`, `solid`."),
            ("ShapeList.filter_by", "", "keep what matches: an `Axis` (faces whose normal "
                                        "is along it), a `GeomType`, a property, or any "
                                        "predicate."),
            ("ShapeList.filter_by_position", "", "keep what lies between two coordinates "
                                                 "along an axis."),
            ("ShapeList.sort_by", "", "ordered along an `Axis` or by a `SortBy` property "
                                      "-- AREA, LENGTH, VOLUME, RADIUS, DISTANCE."),
            ("ShapeList.sort_by_distance", "", "ordered by distance from a point or a "
                                               "shape."),
            ("ShapeList.group_by", "", "grouped into a `GroupBy` of `ShapeList`s by axis "
                                       "position or property -- this is how you get "
                                       "\"all the faces at the same height\" without a "
                                       "tolerance you invented."),
            ("ShapeList.first", "", "the first of the list after sorting."),
            ("ShapeList.last", "", "the last of the list after sorting."),
            ("ShapeList.center", "", "the centre of everything in the list."),
            ("GeomType", "", "PLANE, CYLINDER, CONE, SPHERE, TORUS, BSPLINE... the "
                             "argument `filter_by` takes to find fillets: cylindrical "
                             "and toroidal faces are what defeaturing removes."),
            ("SortBy", "", "LENGTH, RADIUS, AREA, VOLUME, DISTANCE: what `sort_by` and "
                           "`group_by` order on."),
            ("Select", "", "ALL or LAST, where a query offers the choice."),
        ],
    ),
    (
        "Measuring a shape",
        "Every binding can be measured. This is what makes a named-binding script "
        "inspectable: you do not have to render to find out what you built.",
        [
            ("Shape.bounding_box", "", "the extents, with `.min`, `.max`, `.size` and "
                                       "`.diagonal`."),
            ("Shape.area", "", "surface area, in whatever unit the numbers are in -- "
                               "which is metres by the time it is exported."),
            ("Solid.volume", "", "volume of a solid; on the fluid domain this is the "
                                 "number the mesh cell count is predicted from."),
            ("Solid.center", "", "centre of mass, bounding box or geometry, by "
                                 "`CenterOf`."),
            ("Shape.is_valid", "", "OCCT's own verdict on the shape -- worth asking after "
                                   "every boolean on an import."),
            ("Shape.distance_to", "", "the shortest distance to another shape: the gap, "
                                      "measured rather than eyeballed."),
            ("Shape.closest_points", "", "the two points where that shortest distance "
                                         "is."),
            ("Shape.clean", "", "coincident faces merged after a boolean; run it before "
                                "counting faces."),
            ("Shape.show_topology", "", "the tree -- compound, solids, shells, faces -- "
                                        "printed. The first thing to run on an import "
                                        "that came back as something unexpected."),
            ("Shape.children", "", "the parts of a compound, which is what a multi-solid "
                                   "import arrives as."),
            ("Shape.tessellate", "", "vertices and triangles at a tolerance, without "
                                     "writing a file."),
            ("Face.normal_at", "", "the outward normal at a point on a face -- which side "
                                   "is the fluid on."),
            ("Edge.length", "", "arc length of an edge."),
        ],
    ),
    (
        "Moving a shape",
        "`moved` and `located` return copies; `move` and `locate` change in place. In a "
        "script that has to re-run from empty, prefer the copies.",
        [
            ("Shape.moved", "", "a copy displaced by a `Location`."),
            ("Shape.located", "", "a copy placed at a `Location` absolutely."),
            ("Shape.move", "", "in place, relative."),
            ("Shape.locate", "", "in place, absolute."),
            ("Shape.rotate", "", "a copy turned about an `Axis` by an angle in degrees."),
            ("Shape.translate", "", "a copy displaced by a vector."),
        ],
    ),
    (
        "Reading and writing files",
        "The import side is the whole reason OCCT is underneath this.",
        [
            ("import_step", "", "a STEP file as a `Part` -- a compound when the file "
                                "holds several solids, which is the normal case and not "
                                "an error. `.children` or `.solids()` to get at them."),
            ("import_brep", "", "OCCT's own format; the fastest checkpoint format."),
            ("import_stl", "", "a triangulated surface as a `Shape`. Triangles, not "
                               "B-rep: no faces to tag and no fillets to remove."),
            ("export_step", "", "B-rep out, exactly. This is the checkpoint an expensive "
                                "import-and-repair stage writes so later cells load it "
                                "instead of redoing it."),
            ("export_brep", "", "the same, in OCCT's format."),
            ("export_stl", "", "one triangulated file, with a linear and an angular "
                               "tolerance."),
            ("export_gltf", "", "for viewing elsewhere."),
            ("Mesher", "", "3MF/glTF with per-object names kept."),
        ],
    ),
    (
        "The types you will see",
        "Mostly so a `repr` or a traceback reads.",
        [
            ("Part", "", "a 3D result. `Part()` is the empty one to add to."),
            ("Sketch", "", "a 2D face or set of faces."),
            ("Curve", "", "1D edges, before they are a face."),
            ("Compound", "", "several shapes as one -- what a multi-solid STEP import "
                             "is."),
            ("Solid", "", "one closed volume."),
            ("Shell", "", "connected faces; closed, it bounds a solid."),
            ("Face", "", "one bounded surface -- the unit a patch name is attached to."),
            ("Wire", "", "connected edges."),
            ("Edge", "", "one bounded curve."),
            ("Vertex", "", "a point in the topology."),
            ("ShapeList", "", "what every selector returns: a list with the query methods "
                              "above on it."),
            ("Shape", "", "the base class; where `bounding_box`, `moved` and the "
                          "operators live."),
            ("Color", "", "per-shape colour, carried through glTF and 3MF export."),
        ],
    ),
]


# -- introspection -------------------------------------------------------------


def library():
    """The installed build123d, imported by name so this file parses without it."""
    return importlib.import_module("build123d")


def resolve(module, dotted: str):
    """The object a dotted entry names, or AttributeError saying which step failed."""
    obj = module
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _format_default(value) -> str:
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, tuple) and value and isinstance(value[0], enum.Enum):
        return "(" + ", ".join(_format_default(item) for item in value) + ")"
    text = repr(value)
    if len(text) > 24 or OCCT_PATTERN.search(text):
        return "..."
    return text


def signature_of(obj) -> str:
    """Parameter names and defaults, annotations dropped, `mode=` dropped.

    `mode=` is builder-mode machinery: it says which pending context an operation adds
    itself to, and in algebra mode there is no context. Leaving it in every line would
    both waste the width and advertise the surface this file exists not to offer.
    """
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return ""
    rendered = []
    for name, parameter in signature.parameters.items():
        if name in ("self", "cls", "mode"):
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            rendered.append(f"*{name}")
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            rendered.append(f"**{name}")
        elif parameter.default is inspect.Parameter.empty:
            rendered.append(name)
        else:
            rendered.append(f"{name}={_format_default(parameter.default)}")
    return "(" + ", ".join(rendered) + ")"


def _entry_line(module, display: str, target: str, purpose: str) -> str:
    name = target or display
    obj = resolve(module, name)
    call = ""
    if not display.endswith(")") and not display.startswith("`"):
        call = signature_of(obj)
    label = display if display.startswith("`") else f"`{display}{call}`"
    return f"- {label}\n  {purpose}"


def render(module=None) -> str:
    """The whole index. Every entry is resolved here, so a stale name raises."""
    module = module or library()
    version = getattr(module, "__version__", "unknown")
    out = [HEADER, f"\nChecked against build123d {version}.\n"]

    out.append("\n## The operators -- this is algebra mode\n")
    out.append(
        "Three dunders, invisible to any `dir()` pass, and the whole reason the surface "
        "reads. Every one returns a new shape and leaves its operands alone, so a line "
        "can be understood without replaying the ones above it.\n"
    )
    for display, target, purpose in OPERATORS:
        resolve(module, target)
        out.append(f"- {display}\n  {purpose}")
    out.append(
        "\nOrder matters and cannot be made not to: difference does not commute, and "
        "mixed compositions do not associate. Name the intermediate results.\n"
    )

    for title, blurb, entries in GROUPS:
        out.append(f"\n## {title}\n")
        out.append(blurb + "\n")
        for display, target, purpose in entries:
            out.append(_entry_line(module, display, target, purpose))
    out.append("")
    return "\n".join(out)


def unresolved(module=None) -> list[str]:
    """Entries that no longer name anything. The verification half of the claim."""
    module = module or library()
    gone = []
    for _, target, _ in OPERATORS:
        try:
            resolve(module, target)
        except AttributeError:
            gone.append(target)
    for _, _, entries in GROUPS:
        for display, target, _ in entries:
            name = target or display
            try:
                resolve(module, name)
            except AttributeError:
                gone.append(name)
    return gone


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"where to write the index (default {DEFAULT_OUT})")
    parser.add_argument("--check", action="store_true",
                        help="only report entries that no longer resolve")
    parser.add_argument("--stdout", action="store_true", help="print instead of writing")
    args = parser.parse_args(argv)

    try:
        module = library()
    except ImportError as exc:
        print(f"refused: build123d is not importable here ({exc}); this reads the "
              "installed library and cannot be written without it", file=sys.stderr)
        return 2

    gone = unresolved(module)
    if args.check:
        for name in gone:
            print(f"gone: {name}")
        print(f"{len(gone)} of the curated entries no longer resolve")
        return 2 if gone else 0
    if gone:
        print(f"refused: {len(gone)} entries no longer resolve ({', '.join(gone)}); "
              "the selection needs editing, not the output", file=sys.stderr)
        return 2

    text = render(module)
    if args.stdout:
        print(text)
        return 0
    args.out.write_text(text, encoding="utf-8")
    print(f"{args.out}: {len(text.encode('utf-8')) / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
