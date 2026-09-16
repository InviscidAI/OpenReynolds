# T24 — an angle, measured off the mesh

**Capability:** authoring, tangential entry, a direction as a measured quantity.
**Fixture:** none — authored from the request.

## Request

> Mesh the gas in a swirl chamber. A cylindrical vessel 120 mm internal diameter and 200 mm
> tall, closed at the bottom, with a 40 mm diameter outlet through the centre of the top. Four
> rectangular inlet ports 20 mm wide and 30 mm tall enter through the cylindrical wall 40 mm
> above the floor, evenly spaced around it, each angled 15 degrees off the local tangent so
> the incoming gas sets up a swirl. Mesh the gas inside the vessel and the four inlet ducts,
> each duct 50 mm long measured along its own axis. Name the four inlets, the outlet, the
> cylinder wall, the floor, the top and the four duct walls separately, and export one STL
> per patch to `constant/triSurface` before meshing.

**Largest extent:** `0.2` m — the largest dimension the exported surface should
span, for the `scale` probe. The whole union, not the part: an external-flow case
exports its far-field box too. Order of magnitude is enough; the probe fires
outside a factor of a hundred.

## Properties, measured on the delivered mesh

- each inlet port's face normal, and the angle between it and the local tangent at that port, measured on the mesh
- each inlet's area and its height above the floor, measured on the mesh
- the chamber volume and the four duct volumes separately, measured on the mesh

## What this catches

**A property that is a direction, which the corpus has never asked for.** Every measured
property to date is a length, an area, a volume or a count. A face normal is a vector, the
15-degree tangential offset is the only thing that makes this a swirl chamber rather than a
tank, and it is a number the desk cannot restate from a dimension — it has to come from the
mesh's face normals. The baseline's T2 was asked for "the four patches' areas and mean
normals" and printed one area and no normal at all, after the per-patch loop died on a
`TypeError` and was never retried.

**Four features at 90 degrees, where the error is in the rotation.** A desk that builds one
duct and rotates it three times gets the angular positions right and can still get all four
tangential angles wrong in the same direction — or get one right and mirror the others. The
spread is what shows it.

**A sign that matters.** Fifteen degrees off tangent one way swirls; the other way it also
swirls, in the opposite sense, and both mesh identically. Reporting the angle without its sense
relative to the axis is half a measurement.

## A pass that is really a failure

The ducts enter radially — 0 degrees off the radius rather than 15 off the tangent, a confusion
the two conventions invite — and the chamber meshes perfectly. Volumes, areas, heights and
angular positions are all exactly right. There is no swirl, which is the entire purpose of the
vessel, and no length, area or volume in the case can see it.

## Provenance

Adapted from the plasma reactor (S9) in the show gallery of
[Pan-Chera/Multi-Agent-CAD](https://github.com/Pan-Chera/Multi-Agent-CAD). Dimensions and the
tangential entry are authored here; what the source supplies is the class of object, a vessel
whose function is the direction its inlets point rather than any of its sizes.
