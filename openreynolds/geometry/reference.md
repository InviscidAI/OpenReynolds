THE GEOMETRY API (reference card)

A script is plain Python against these names, already bound: Sketch, Rect, Disk, Polygon,
Band, Outline, Passage, Bypass, Row, Serpentine, BodyInBox, EdgeRef, math. No import is
needed and none but math and json is allowed. Lengths are in the sketch's units; angles in
degrees, counter-clockwise from +x (0 = +x, 90 = +y). center= is accepted for centre=.
The tool solves what follows from what is stated (footprints, landings, pitches, fillets)
and prints every number back; a refusal carries the numbers to choose from.

Sketch
  Sketch(units: Units = mm, name: str = sketch)
  Sketch                 one per script; every length in `units`, every angle in degrees counter-clockwise from +x
  s.fluid = <feature>    the one region that is meshed (a feature, a boolean of features, a Row)
  s.rect                 a rectangle by its lower-left origin or its centre; sides .left .right .top .bottom
  s.disk                 a circle by centre and radius or diameter; .edge names its boundary
  s.polygon              a closed polygon from three or more points; .edge
  s.annulus              a ring; s.band(...) an arc of a ring between two angles; .edge
  s.outline              a closed x,y outline file (csv or Selig .dat), scaled to `size` across x; .edge
  s.passage              a constant-width channel; the same as Passage(...)
  s.apart                parts that must neither overlap nor touch; two Rows, or a Row and a part that is not its host, are apart by default
  s.inlet / s.outlet     the ports by intent (main.start, duct.left) or by rule string; one edge or several
  s.wall / s.patch       a named wall (cyl.edge) or a slip / symmetry patch on edges
  s.expect               how many inlets and outlets the shape has (default one of each)
  s.note                 a line printed back under SCRIPT, never parsed
  s.rect(*, origin: (x, y) | None = None, centre: (x, y) | None = None, size: (x, y), round: float = 0.0, name: str | None = None)
  s.disk(*, centre: (x, y) | None = None, radius: float | None = None, diameter: float | None = None, name: str | None = None)
  s.polygon(points: [(x, y), ...], *, name: str | None = None)
  s.annulus(*, centre=None, r_inner: float, r_outer: float, name=None)
  s.band(*, centre=None, r_inner: float, r_outer: float, start_deg: float, end_deg: float, name=None)
  s.outline(file: str, *, size: float | None = None, aoa: float = 0.0, name=None)
  s.passage(*, width: float, start: "(x, y) | (wall, u)", heading: float | None = None, name=None)
  s.apart(*features: "Feature", gap: float | None = None)
  s.inlet(*edges: "EdgeRef | str", name: str = inlet)   s.outlet(*edges: "EdgeRef | str", name: str = outlet)
  s.wall(*edges: "EdgeRef | str", name: str)   s.patch(*edges: "EdgeRef | str", name: str, kind: "wall | slip | symmetry")
  s.expect(*, inlets: int = 1, outlets: int = 1)   s.note(text: str)

Features are values: a | b fuses, a - b cuts, a & b intersects; f.moved(dx, dy), f.rotated(deg,
about=None), f.mirrored(axis, at=0, keep=False) return new features (axis "x" mirrors across the
line x = at); f.named(name) registers a name. Every feature a claim names needs name=.
Only a Passage's leg methods change it in place, and they return it, so p.line(30) and
p = p.line(30) both work.

Rect      Rect(*, origin: (x, y) | None = None, centre: (x, y) | None = None, size: (x, y), round: float = 0.0, name: str | None = None)
  sides .left .right .top .bottom are edges (ports) and walls (a loop or a branch can hang on
  one when the flow along the rect is known: an inlet/outlet declared on the rect, or flow=).
  Measures by role: width is the shorter side, length the longer, size_x / size_y the page sizes.
Disk      Disk(*, centre: (x, y) | None = None, radius: float | None = None, diameter: float | None = None, name: str | None = None)
Polygon   Polygon(points: [(x, y), ...], *, name: str | None = None)   (three or more points; made counter-clockwise)
Band      Band(*, centre=None, r_inner: float, r_outer: float, start_deg: float = 0.0, end_deg: float = 360.0, name: str | None = None)   (an annulus when start..end spans 360)
Outline   Outline(file: str, *, size: float | None = None, aoa: float = 0.0, name: str | None = None)

Passage   Passage(*, width: float, start: "(x, y) | (wall, u)", heading: float | None = None, name: str | None = None)
  start is a point, or (wall, u): a branch that leaves the wall at u along it, cut flush there;
  heading then defaults to the wall's outward normal. Legs, in order, each returning the passage:
  .line(length: float)                         a straight leg along the current heading
  .arc(*, radius: float, turn: float)          a bend of centreline radius, turn + left / - right (radius > width/2)
  .turn(deg: float)                            a sharp mitred corner between two line legs, |deg| < 180
  .line_to(*, x: float | None = None, y: float | None = None) run along the heading until x= or y= (an open end, not a landing)
  .turn_to(*, heading: float, radius: float, side: "auto | left | right" = auto) the arc to an absolute heading; auto = the shorter way (180 needs side=)
  .u_turn(*, radius: float, side: "left | right" = left) a 180-degree arc; left = the +y side for a +x heading
  .end_on(wall: WallRef)                       the last leg: run to the wall's line and cut flush there (a landing)
  Passage.through(*, width: float, points: [(x, y), ...], radius: float, name=None)
                                               absolute waypoints, every corner filleted at radius
  .start .end are the caps (ports); .top .bottom (page words) or .side("left"|"right"|"top"|"bottom",
  leg=None) name a straight leg's wall (leg counts every leg from 1; needed when there are several).
  .end_point .end_heading .length are solved for the script's own prints.

Bypass    Bypass(*, wall: WallRef | EdgeRef, width: float, leave_angle: float, outer_radius: float, return_angle: float, at: float | None = None, leave_length: float | None = None, flow: "+x | -x | +y | -y | None" = None, name: str | None = None)
  a loop that leaves `wall` at `at` (u along the wall) at leave_angle from the flow, sweeps round an
  arc whose OUTER wall has outer_radius, and returns onto the same wall at return_angle measured from
  the upstream direction (90 = straight back), so it always heads against the flow. The tool solves
  the arc, the landing, the lip and the footprint. leave_length None = one width. at= is required
  outside a Row and refused inside one (the Row places its instances). outer_radius > width.

Row       Row(item: Feature, *, count: int, gap: float | None = None, pitch: float | None = None, along: WallRef | (x, y) | None = None, start: float | None = None, align: "centre | start | end" = centre, margin: float | None = None, name: str | None = None)
  count copies of item along a wall (along= inferred from a Bypass's wall) or a direction (dx, dy).
  Give gap= only when the request states one: otherwise the largest uniform gap that fits is solved
  and printed. pitch = footprint along the row + gap. margin defaults to half the item's width;
  align centres the row on the wall, start= fixes the first anchor. When the row does not fit, the
  refusal lists the footprint at five values of the free parameter and the largest gap that fits.
  row[k] is instance k, named row[k] (0-based) everywhere the tool prints.

Serpentine Serpentine(*, width: float, passes: int, pass_length: float, bend_radius: float, start: (x, y) = (0.0, 0.0), heading: float = 0.0, stack: "+y | -y | +x | -x" = +y, name: str | None = None)
  passes straight passes joined by 180-degree bends of centreline bend_radius, stacked along stack
  (across the heading); 2 * bend_radius > width. .start .end; .side(...) names a pass's wall.
  An even number of passes ends on the side it started from.

BodyInBox BodyInBox(body: Feature, *, ahead: float = 2.0, behind: float = 5.0, above: float = 2.0, below: float = 2.0, far: "slip | symmetry" = slip, name: str | None = None)
  a closed body in a flow box (external flow); box sizes in body lengths; it declares its own
  inlet, outlet, farfield and body patches (.inlet .outlet .farfield .body_edge).

Ports and rule strings
  s.inlet(main.start), s.outlet(main.end), s.wall(cyl.edge, name="cylinder"), s.patch(duct.top,
  name="lid", kind="slip"). A rule string instead of an edge: "x:min" / "y:max" (every edge flat
  there), "x:0.05" (flat on that line), "near:x,y" (the one edge whose middle is nearest), "box:x0,y0,x1,y1",
  "normal:-x" (the one open end facing that way). Name the ports by intent, never by coordinates.

Walls and flow
  A wall has a frame: u runs along the flow from its upstream end (0) to its downstream end (its
  span), v outward. Bypass.at, Row.start, a branch's start=(wall, u) and every footprint are in u.
  A Passage leg's flow is its heading; a Rect side's is read from the inlet/outlet declared on the
  rect anywhere in the script, or from flow="+x" (E-FLOW otherwise).

Refusals (the code, the numbers, the fix; nothing is built): E-ROW-FIT, E-ROW-PITCH, E-ROW-ALONG,
  E-ROW-GAP-REQUIRED, E-ROW-DIRECTION, E-ROW-ITEM-AT, E-BYPASS-AT, E-LAND-OFF-WALL, E-RADIUS, E-ANGLE,
  E-FLOW, E-SIDE-AMBIGUOUS, E-EMPTY-PASSAGE, E-TURN-AMBIGUOUS, E-LINE-TO, E-CORNER, E-FILLET,
  E-SERP-RADIUS, E-NO-FLUID. After the build the lint judges overlap, touch, landings, notches, slivers,
  ports and units, and the compliance table judges every claim.

Units: Sketch(units="mm" | "cm" | "m" | "in"); use the request's own unit. The case is written in
metres by the tool; the print-back stays in the sketch's units.
