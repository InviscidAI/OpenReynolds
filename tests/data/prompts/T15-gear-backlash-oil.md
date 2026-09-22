# T15 — oil in a gear mesh, where the gap varies with rotation

**Capability:** authoring, two solids in near contact, a clearance that is not constant.
**Fixture:** none — authored from the request.

## Request

> Mesh the oil in a flooded bevel gear box. Two straight bevel gears on shafts crossing at
> 90 degrees: the pinion 20 teeth, the wheel 30 teeth, 3 mm module, 20 degree pressure
> angle, meshing with 0.3 mm of backlash measured along the pitch line. They sit in a
> rectangular cavity 160 x 140 x 120 mm, both shafts leaving through bores in the walls. The
> cavity is filled with oil, so the oil is the cavity minus the two gears, and it is one
> connected region because the backlash and the tip clearances leave a path everywhere.
> Mesh it. Name the pinion surface, the wheel surface, the cavity walls and the two shaft
> bores separately, and export one STL per patch to `constant/triSurface` before meshing.

**Largest extent:** `0.16` m — the largest dimension the exported surface should
span, for the `scale` probe. The whole union, not the part: an external-flow case
exports its far-field box too. Order of magnitude is enough; the probe fires
outside a factor of a hundred.

## Properties, measured on the delivered mesh

- the backlash at the pitch line, measured between the two meshed gear patches, at the engaged tooth pair
- how many cells span the backlash gap, counted on the mesh
- the oil volume, against the cavity volume minus both gear volumes

## What this catches

**Two solids that nearly touch, where nearly is the whole point.** A boolean subtraction of
two gears from a box is the case where OCCT either fuses the two gears through their contact
point — collapsing the backlash and disconnecting the oil on each side of the mesh — or
leaves a sliver face that no mesher can resolve. `checkMesh`'s region count sees the first.
Nothing in the corpus currently has two solids in near contact.

**A count taken on the mesh.** Tooth count is the cheapest possible measured-versus-restated
test: the desk typed 20 and 30 into the geometry, and recovering them from the meshed patch
requires actually interrogating it. If the desk prints `teeth: 20, 30` without a query, the
pattern the last two sweeps found four-for-four is confirmed a fifth time on the easiest
possible instance.

**A clearance that varies.** T9's clearance is constant by construction, so one number
describes it. Backlash along a gear flank is a function of angular position, which means the
honest answer names where it was measured.

## A pass that is really a failure

The gears are built at their nominal pitch diameters with no backlash applied at all — the
flanks coincident — and OCCT's subtraction quietly welds them into one solid. The oil meshes,
`checkMesh` reports one region because the cavity is still connected around the outside of
the gear pair, and the backlash is reported as the 0.3 mm that was requested. The gear mesh
itself is sealed and the only evidence is a gap measured between the two flank patches.

## Provenance

Adapted from the bevel gear drive (07) in
[Adam-CAD/CADAM](https://github.com/Adam-CAD/CADAM) — "meshing bevel gear pair at 90°" — and
the planetary gear stage (09) there, which is P10 in
[Pan-Chera/Multi-Agent-CAD](https://github.com/Pan-Chera/Multi-Agent-CAD) and its largest
benchmark at 21 features. Both score the gears as meshing if they look meshed; neither asks
what the gap between the flanks measures, because for a printed gear pair the answer is
whether it turns.
