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


Checked against build123d 0.11.1.


## The operators -- this is algebra mode

Three dunders, invisible to any `dir()` pass, and the whole reason the surface reads. Every one returns a new shape and leaves its operands alone, so a line can be understood without replaying the ones above it.

- `a + b`
  fuse: the union of two shapes, as one shape. `Part() + Box(...)` grows material.
- `a - b`
  cut: a with b removed. Does not commute -- `a - b` is not `b - a` -- and this is the operation the fluid domain is extracted with.
- `a & b`
  intersect: only what is in both. Interference between two solids is `(a & b).volume`, and zero means they do not touch.
- `loc * shape`
  place: `Pos(0, 0, 10) * part` moves a copy, `Plane.XZ * sketch` puts a sketch on a plane. The shape on the right is not modified.

Order matters and cannot be made not to: difference does not commute, and mixed compositions do not associate. Name the intermediate results.


## Part operations

Each takes its operands by name and returns a new shape. Nothing is pending and nothing is consumed from an earlier line.

- `extrude(to_extrude=None, amount=None, dir=None, until=None, target=None, both=False, taper=0.0, clean=True)`
  a face or sketch grown into a solid, by `amount` or `until` another object.
- `revolve(profiles=None, axis=..., revolution_arc=360.0, clean=True)`
  a profile swept about an axis; `revolution_arc` short of 360 gives a wedge.
- `loft(sections=None, ruled=False, clean=True)`
  a solid through a series of sections in order.
- `sweep(sections=None, path=None, multisection=False, is_frenet=False, transition=Transition.TRANSFORMED, normal=None, binormal=None, clean=True)`
  a section carried along a path; `is_frenet` for a path that twists.
- `offset(objects=None, amount=0, openings=None, kind=Kind.ARC, side=Side.BOTH, closed=True, min_edge_length=None)`
  a shell, a hollow, or a grown/shrunk copy: `openings` names the faces to leave off, which is how a box becomes a duct.
- `fillet(objects, radius)`
  a radius on the named edges or vertices -- pass the `ShapeList` you selected, not an index.
- `chamfer(objects, length, length2=None, angle=None, reference=None)`
  a flat on the named edges; `length2` and `angle` for an asymmetric one.
- `draft(faces, neutral_plane, angle)`
  a taper applied to faces for moulding.
- `thicken(to_thicken=None, amount=None, normal_override=None, both=False, clean=True)`
  a face or shell given a wall thickness, normal to itself.
- `split(objects=None, bisect_by=..., keep=Keep.TOP)`
  a shape cut by a plane; `keep` says which side survives, or both.
- `section(obj=None, section_by=..., height=0.0, clean=True)`
  the planar faces where a plane crosses a solid -- the cheap way to see the inside of an import.
- `mirror(objects=None, about=...)`
  a reflected copy about a plane. Symmetry built rather than re-authored.
- `scale(objects=None, by=1, about=None)`
  a resized copy, uniform or per-axis. Not the unit fix: a millimetre import is scaled by 0.001 here, once, early.
- `project(objects=None, workplane=None, target=None)`
  faces, edges or points thrown onto a plane or a target shape along a direction.
- `make_face(edges=None)`
  a closed set of edges or wires turned into a face.
- `make_hull(edges=None)`
  the convex hull of edges -- a quick outer envelope for a flow box.
- `full_round(edge, invert=False, voronoi_point_count=100)`
  a tangent-continuous round replacing two edges and the face between them.
- `add(objects, rotation=None, clean=True)`
  a shape folded into another, type by type.
- `Until(*values)`
  argument to `extrude`: NEXT, LAST, PREVIOUS, FIRST -- extrude to geometry rather than to a number you measured by hand.
- `Keep(*values)`
  argument to `split`: TOP, BOTTOM, BOTH, INSIDE, OUTSIDE.
- `Kind(*values)`
  argument to `offset`: ARC, INTERSECTION, TANGENT -- how corners are carried out.
- `Side(*values)`
  argument to `offset` on 2D: LEFT, RIGHT, BOTH.

## Solid primitives

Combined with the operators above. Every one takes `align` and `rotation`, so placement is part of the call rather than a later move.

- `Box(length, width, height, rotation=(0, 0, 0), align=(Align.CENTER, Align.CENTER, Align.CENTER))`
  length, width, height.
- `Cylinder(radius, height, arc_size=360, rotation=(0, 0, 0), align=(Align.CENTER, Align.CENTER, Align.CENTER))`
  radius and height; `arc_size` under 360 for a pie slice.
- `Sphere(radius, arc_size1=-90, arc_size2=90, arc_size3=360, rotation=(0, 0, 0), align=(Align.CENTER, Align.CENTER, Align.CENTER))`
  radius, and the three arc angles for a partial one.
- `Cone(bottom_radius, top_radius, height, arc_size=360, rotation=(0, 0, 0), align=(Align.CENTER, Align.CENTER, Align.CENTER))`
  bottom radius, top radius, height -- top radius 0 for a point.
- `Torus(major_radius, minor_radius, minor_start_angle=0, minor_end_angle=360, major_angle=360, rotation=(0, 0, 0), align=(Align.CENTER, Align.CENTER, Align.CENTER))`
  major and minor radius.
- `Wedge(xsize, ysize, zsize, xmin, zmin, xmax, zmax, rotation=(0, 0, 0), align=(Align.CENTER, Align.CENTER, Align.CENTER))`
  a tapered box, six lengths.
- `Hole(radius, depth=None)`
  a through hole as a shape to subtract.
- `CounterBoreHole(radius, counter_bore_radius, counter_bore_depth, depth=None)`
  a hole with a flat-bottomed recess, as a shape to subtract.
- `CounterSinkHole(radius, counter_sink_radius, depth=None, counter_sink_angle=82)`
  a hole with a conical recess, as a shape to subtract.

## Sketch primitives (2D)

A sketch is a face. Place it with `Plane.XZ * sketch`, then extrude or revolve it.

- `Rectangle(width, height, rotation=0, align=(Align.CENTER, Align.CENTER))`
  width and height.
- `RectangleRounded(width, height, radius, rotation=0, align=(Align.CENTER, Align.CENTER))`
  the same with a corner radius, without a fillet call.
- `Circle(radius, arc_size=360.0, align=(Align.CENTER, Align.CENTER))`
  radius.
- `Ellipse(x_radius, y_radius, rotation=0, align=(Align.CENTER, Align.CENTER))`
  two radii.
- `Polygon(*pts, rotation=0, align=(Align.NONE, Align.NONE))`
  an explicit list of points.
- `RegularPolygon(radius, side_count, major_radius=True, rotation=0, align=(Align.CENTER, Align.CENTER))`
  radius and side count; `major_radius` says whether that is across corners or across flats.
- `Triangle(a=None, b=None, c=None, A=None, B=None, C=None, align=None, rotation=0)`
  any three of sides and angles -- the rest solved for you.
- `Trapezoid(width, height, left_side_angle, right_side_angle=None, rotation=0, align=(Align.CENTER, Align.CENTER))`
  width, height, and the two base angles.
- `SlotOverall(width, height, rotation=0, align=(Align.CENTER, Align.CENTER))`
  a slot by its total length and height.
- `SlotCenterToCenter(center_separation, height, rotation=0)`
  a slot by the distance between its end centres.
- `SlotCenterPoint(center, point, height, rotation=0)`
  a slot from a centre, a point and a height.
- `SlotArc(arc, height, rotation=0)`
  a slot following an arc.
- `Text(txt, font_size, font='Arial', font_path=None, font_style=FontStyle.REGULAR, text_align=(TextAlign.CENTER, TextAlign.CENTER), align=None, path=None, position_on_path=0.0, single_line_width=None, rotation=0.0)`
  characters as a face -- labels cut into a part.

## Curves and edges

Built into a `Curve`, closed, and turned into a face with `make_face` when the outline is not one of the primitives above.

- `Line(*pts)`
  two points.
- `Polyline(*pts, close=False)`
  a run of points; `close=True` to shut it.
- `Spline(*pts, tangents=None, tangent_scalars=None, periodic=False)`
  through points, with optional end tangents.
- `CenterArc(center, radius, start_angle, arc_size)`
  centre, radius, start angle, arc size.
- `ThreePointArc(*pts)`
  through three points.
- `RadiusArc(start_point, end_point, radius, short_sagitta=True)`
  two points and a radius.
- `TangentArc(*pts, tangent, tangent_from_first=True)`
  two points, leaving the first along a given tangent.
- `EllipticalCenterArc(center, x_radius, y_radius, start_angle=0.0, end_angle=None, arc_size=90.0, rotation=0.0, angular_direction=None)`
  an elliptical arc about a centre.
- `JernArc(start, tangent, radius, arc_size)`
  an arc that starts tangent to where you are -- the one that chains without recomputing a centre.
- `Helix(pitch, height, radius, center=(0, 0, 0), direction=(0, 0, 1), cone_angle=0, lefthand=False)`
  pitch, height, radius; the path for a threaded sweep.
- `Bezier(*cntl_pnts, weights=None)`
  control points, with weights for a rational curve.
- `FilletPolyline(*pts, radius, close=False)`
  a polyline with every corner already rounded.
- `trace(lines=None, line_width=1)`
  a line given a width, as a face.

## Placement: planes, axes, locations

Where a thing is, said once at the call rather than fixed up afterwards.

- `Plane.XY`
  the ground plane; also `Plane.XZ`, `Plane.YZ`.
- `Plane.front`
  named views -- `front`, `back`, `top`, `bottom`, `left`, `right`, `isometric`.
- `Plane(face)`
  a plane taken from a planar face, so a sketch can be placed on geometry you selected rather than on coordinates you assumed.
- `Plane.offset(amount)`
  a parallel plane a distance along its own normal.
- `Plane.rotated(rotation=(0, 0, 0), ordering=None)`
  a plane turned about its own axes.
- `Plane.location`
  the `Location` a plane represents.
- `Axis.X`
  the world axes; also `Axis.Y`, `Axis.Z`.
- `Axis(origin, direction)`
  an arbitrary axis -- the argument `revolve`, `filter_by` and `sort_by` all take.
- `Location(*args, **kwargs)`
  a position and an orientation; multiply it onto a shape to place a copy.
- `Pos(*args, **kwargs)`
  shorthand `Location` for a pure translation: `Pos(0, 0, 10) * part`.
- `Rot(*args, **kwargs)`
  shorthand `Location` for a pure rotation, in degrees.
- `Rotation(*args, **kwargs)`
  a rotation about X, Y, Z in degrees, usable as a `Location`.
- `Vector(*args, **kwargs)`
  a 3D vector with `length`, `normalized`, `cross`, `dot` -- what every `center()` and `normal_at` returns.
- `Align(*values)`
  MIN, CENTER, MAX per axis: where a primitive sits relative to its own origin.

## Selecting: ShapeList and its query methods

This is the tagging idiom, and the one that has to be stable. Every selector is re-derived from geometry on each run -- a face index from a previous session is not a selector and does not survive a rebuild.

- `faces()`
  every face, as a `ShapeList`. Also `edges`, `vertices`, `solids`, `wires`, `shells`.
- `face()`
  the single face, raising if there is not exactly one. Also `edge`, `vertex`, `solid`.
- `ShapeList.filter_by(filter_by, reverse=False, tolerance=1e-05)`
  keep what matches: an `Axis` (faces whose normal is along it), a `GeomType`, a property, or any predicate.
- `ShapeList.filter_by_position(axis, minimum, maximum, inclusive=(True, True))`
  keep what lies between two coordinates along an axis.
- `ShapeList.sort_by(sort_by=..., reverse=False)`
  ordered along an `Axis` or by a `SortBy` property -- AREA, LENGTH, VOLUME, RADIUS, DISTANCE.
- `ShapeList.sort_by_distance(other, reverse=False)`
  ordered by distance from a point or a shape.
- `ShapeList.group_by(group_by=..., reverse=False, tol_digits=6)`
  grouped into a `GroupBy` of `ShapeList`s by axis position or property -- this is how you get "all the faces at the same height" without a tolerance you invented.
- `ShapeList.first`
  the first of the list after sorting.
- `ShapeList.last`
  the last of the list after sorting.
- `ShapeList.center()`
  the centre of everything in the list.
- `GeomType(*values)`
  PLANE, CYLINDER, CONE, SPHERE, TORUS, BSPLINE... the argument `filter_by` takes to find fillets: cylindrical and toroidal faces are what defeaturing removes.
- `SortBy(*values)`
  LENGTH, RADIUS, AREA, VOLUME, DISTANCE: what `sort_by` and `group_by` order on.
- `Select(*values)`
  ALL or LAST, where a query offers the choice.

## Measuring a shape

Every binding can be measured. This is what makes a named-binding script inspectable: you do not have to render to find out what you built.

- `Shape.bounding_box(tolerance=None, optimal=True)`
  the extents, with `.min`, `.max`, `.size` and `.diagonal`.
- `Shape.area`
  surface area, in whatever unit the numbers are in -- which is metres by the time it is exported.
- `Solid.volume`
  volume of a solid; on the fluid domain this is the number the mesh cell count is predicted from.
- `Solid.center(center_of=CenterOf.MASS)`
  centre of mass, bounding box or geometry, by `CenterOf`.
- `Shape.is_valid`
  OCCT's own verdict on the shape -- worth asking after every boolean on an import.
- `Shape.distance_to(other)`
  the shortest distance to another shape: the gap, measured rather than eyeballed.
- `Shape.closest_points(other)`
  the two points where that shortest distance is.
- `Shape.clean()`
  coincident faces merged after a boolean; run it before counting faces.
- `Shape.show_topology(limit_class='Vertex', show_center=None)`
  the tree -- compound, solids, shells, faces -- printed. The first thing to run on an import that came back as something unexpected.
- `Shape.children`
  the parts of a compound, which is what a multi-solid import arrives as.
- `Shape.tessellate(tolerance, angular_tolerance=0.1)`
  vertices and triangles at a tolerance, without writing a file.
- `Face.normal_at(*args, **kwargs)`
  the outward normal at a point on a face -- which side is the fluid on.
- `Edge.length`
  arc length of an edge.

## Moving a shape

`moved` and `located` return copies; `move` and `locate` change in place. In a script that has to re-run from empty, prefer the copies.

- `Shape.moved(loc)`
  a copy displaced by a `Location`.
- `Shape.located(loc)`
  a copy placed at a `Location` absolutely.
- `Shape.move(loc)`
  in place, relative.
- `Shape.locate(loc)`
  in place, absolute.
- `Shape.rotate(axis, angle, transform=False)`
  a copy turned about an `Axis` by an angle in degrees.
- `Shape.translate(vector, transform=False)`
  a copy displaced by a vector.

## Reading and writing files

The import side is the whole reason OCCT is underneath this.

- `import_step(filename)`
  a STEP file as a `Part` -- a compound when the file holds several solids, which is the normal case and not an error. `.children` or `.solids()` to get at them.
- `import_brep(file_name)`
  OCCT's own format; the fastest checkpoint format.
- `import_stl(file_name, model_unit=Unit.MM)`
  a triangulated surface as a `Shape`. Triangles, not B-rep: no faces to tag and no fillets to remove.
- `export_step(to_export, file_path, unit=Unit.MM, write_pcurves=True, precision_mode=PrecisionMode.AVERAGE, timestamp=None)`
  B-rep out, exactly. This is the checkpoint an expensive import-and-repair stage writes so later cells load it instead of redoing it.
- `export_brep(to_export, file_path)`
  the same, in OCCT's format.
- `export_stl(to_export, file_path, tolerance=0.001, angular_tolerance=0.1, ascii_format=False)`
  one triangulated file, with a linear and an angular tolerance.
- `export_gltf(to_export, file_path, unit=Unit.MM, binary=False, linear_deflection=0.001, angular_deflection=0.1)`
  for viewing elsewhere.
- `Mesher(unit=Unit.MM)`
  3MF/glTF with per-object names kept.

## The types you will see

Mostly so a `repr` or a traceback reads.

- `Part(obj=None, label='', color=None, material='', joints=None, parent=None, children=None)`
  a 3D result. `Part()` is the empty one to add to.
- `Sketch(obj=None, label='', color=None, material='', joints=None, parent=None, children=None)`
  a 2D face or set of faces.
- `Curve(obj=None, label='', color=None, material='', joints=None, parent=None, children=None)`
  1D edges, before they are a face.
- `Compound(obj=None, label='', color=None, material='', joints=None, parent=None, children=None)`
  several shapes as one -- what a multi-solid STEP import is.
- `Solid(obj=None, label='', color=None, material='', joints=None, parent=None)`
  one closed volume.
- `Shell(obj=None, label='', color=None, parent=None)`
  connected faces; closed, it bounds a solid.
- `Face(*args, **kwargs)`
  one bounded surface -- the unit a patch name is attached to.
- `Wire(*args, **kwargs)`
  connected edges.
- `Edge(obj=None, label='', color=None, parent=None)`
  one bounded curve.
- `Vertex(*args, **kwargs)`
  a point in the topology.
- `ShapeList(iterable=())`
  what every selector returns: a list with the query methods above on it.
- `Shape(obj=None, label='', color=None, parent=None)`
  the base class; where `bounding_box`, `moved` and the operators live.
- `Color(*args, **kwargs)`
  per-shape colour, carried through glTF and 3MF export.
