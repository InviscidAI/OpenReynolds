# T10 — a thin trailing edge, thinner than the cell

**Capability:** authoring, thin sharp features at the cell scale.
**Fixture:** none — authored from the request.

## Request

> Mesh the air passage between two blades of an axial turbine stage. The blades sit on a
> hub of 45 mm radius and stand 45 mm tall radially, so the tip is at 90 mm. There are 28
> blades evenly spaced around the hub, each a cambered aerofoil of 22 mm chord and 6%
> maximum thickness, staggered 30 degrees at the root and twisting a further 35 degrees
> from root to tip. Mesh the volume of one passage: bounded by the pressure side of one
> blade, the suction side of its neighbour, the hub below, a shroud at 90 mm radius, and
> inlet and outlet planes one chord upstream and one chord downstream. Name the two blade
> faces, the hub, the shroud, the inlet, the outlet, and the two periodic side faces
> separately.

## Properties the desk must measure and print

- the blade chord and maximum thickness, measured on the meshed blade patches at root, mid-span and tip
- the trailing edge thickness, and the cell size there, and how many cells resolve it
- the passage pitch at the hub and at the tip, measured between the two periodic faces
- the blade twist from root to tip, in degrees, measured off the geometry
- the two periodic faces, each with its area, and whether they match face for face
- every exported patch, exhaustive and disjoint over the domain's faces

## What this catches

**The feature that is smaller than the mesh and gets silently rounded.** A 6%-thick
aerofoil of 22 mm chord closes to a trailing edge of well under a millimetre. Below the
cell size, `snappyHexMesh` will either round it into a blunt stub or open it into the
neighbouring passage, and the delivered mesh passes `checkMesh` in both cases. The blade
loading is then wrong by an amount nothing in the record can see.

**`scale`, which the last sweep found broken on exactly this input.** §4.1 records
`scale` skipping with *"the case states no dimension"* on three cases that state
dimensions plainly. This case states six, at three spanwise stations, so a skip here is
the same defect firing again and a verdict here is evidence it was fixed.

**Periodic faces claimed rather than matched.** T8 is the one case in the corpus whose
interface conformality is backed by a cross-region computation. This asks the same
question of a periodic pair within one region, where per-region `checkMesh` cannot see it
either.

## A pass that is really a failure

The desk builds the passage from its own parameters, measures the chord and pitch by
re-printing the numbers it set, and delivers a clean mesh whose trailing edge has been
rounded to the cell size. Every printed number agrees with the request because every
printed number came from the request. The trailing edge thickness measured *on the mesh*
is the one number that cannot be produced that way, which is why it is listed.

## Provenance

Adapted from the axial turbine blisk benchmark (12) in
[Adam-CAD/CADAM](https://github.com/Adam-CAD/CADAM), which asks for 28 twisted aerofoil
blades on a keyed hub and is scored on whether the blades are present and twisted. Its
published OpenSCAD solution applies `offset(r=0.6)` to the aerofoil polygon explicitly to
"guarantee manifold trailing edges" — that is, it thickens the trailing edge by 1.2 mm to
make it printable. That fix is the failure this case exists to catch: a trailing edge
thickened to survive the mesher is a blade with different aerodynamics. The blade
dimensions are theirs; the passage, the periodicity and the properties are authored here.
