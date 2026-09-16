# T22 — thirty-six passages, and a count that must come off the mesh

**Capability:** authoring, a high-count radial array with through flow.
**Fixture:** none — authored from the request.

## Request

> Mesh the cooling air through a vented brake disc. Two annular friction faces, each 320 mm
> outside diameter, 190 mm inside diameter and 10 mm thick, held 12 mm apart by 36 radial
> vanes evenly spaced around the disc, each vane 8 mm wide and spanning the full radial depth
> between the inner and outer diameters. Air is drawn in at the inner diameter and thrown out
> at the outer by rotation. Mesh the air in the 36 passages between the vanes. Name the two
> friction-face inner surfaces, the vane surfaces, the inner inlet and the outer outlet
> separately, and export one STL per patch to `constant/triSurface` before meshing.

## Properties the desk must measure and print

- the number of open passages, counted on the mesh, against the 36 the request names
- one passage's width at the inner diameter and at the outer diameter, measured on the meshed vane patches
- the air volume, against the annular gap volume minus 36 vanes

## What this catches

**A count high enough that a wrong one is plausible.** T11's honeycomb has 105 cells and the
desk got them all; this has 36 passages that are open at both ends, so losing one leaves 35
passages and a region count of 1 either way — the passages are all connected through the inner
and outer plena. T11's boolean was a subtraction of prisms from a box; this is 36 vanes
*unioned into* a gap, which is the direction rank 4 of the baseline failed in.

**A width that varies along the passage by a known factor.** The passage is bounded by two
radial vanes, so its width grows linearly with radius: at 190 mm inside diameter the pitch is
about 16.6 mm and at 320 mm it is about 27.9 mm, minus 8 mm of vane in both cases. A desk
measuring at one radius and reporting one width has answered a different question than the
brief asked, and a desk restating its inputs cannot produce either number.

**An area ratio that is a design quantity.** Inlet area at the inner diameter against outlet
area at the outer sets whether the disc pumps. Both are measurable on the mesh and neither is
an input.

## A pass that is really a failure

Thirty-five vanes are placed and the thirty-sixth lands on top of its neighbour, or two vanes
merge — the disc meshes, one region, the volume is off by one vane out of 36, and the passage
count is reported as 36 because that is what was asked for. The friction faces, the thickness
and the diameters are all exactly right. A count taken on the mesh is the only thing that
finds it, and this is the fifth case to ask for one.

## Provenance

Adapted from the brake disc (S10) in the show gallery of
[Pan-Chera/Multi-Agent-CAD](https://github.com/Pan-Chera/Multi-Agent-CAD), and cousin to the
centrifugal impeller that is P8 there and benchmark 08 in CADAM — already in this corpus as
T12. A vented disc is the same radial-array-with-through-flow problem as an impeller with the
blade curvature removed, which is deliberate: T12's failure can then be attributed to the
curvature or not.
