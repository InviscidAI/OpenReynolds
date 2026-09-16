# T11 — many internal passages, and a boolean that must close all of them

**Capability:** authoring, high face count, boolean closure at scale.
**Fixture:** none — authored from the request.

## Request

> Mesh the air path through a honeycomb flow straightener. The core is a 80 x 80 mm plate,
> 20 mm thick in the flow direction, filled with a hexagonal honeycomb of 8 mm cell width
> across the flats and 1 mm wall thickness, the cells running through the full 20 mm so the
> plate is open front to back. The core sits in a square duct of the same 80 x 80 mm
> section that runs 40 mm upstream and 80 mm downstream of it. Mesh the air: the duct
> volume with the honeycomb walls subtracted. Name the inlet, the outlet, the duct walls
> and the honeycomb walls separately.

## Properties the desk must measure and print

- the number of open honeycomb cells through the core, counted on the mesh, against the number the geometry should give
- the cell width across the flats and the wall thickness, measured on the meshed honeycomb patches
- the open area ratio of the core — honeycomb open area over duct section — measured on the mesh, not derived from inputs

## What this catches

**A boolean subtraction that loses some of its cutters.** Around a hundred hexagonal
prisms subtracted from one box is the case where OCCT drops a cutter or fuses two walls
and nothing downstream notices: the mesh is clean, the flow is through a straightener
with three cells blocked, and the pressure drop is wrong. The last sweep's rank 4,
`boolean_union_of_swept_bands_loses_material`, is the same failure on four bypass loops.
This asks it of a hundred, which is where it should be far more likely — and if it is
*not* more likely, that is a result too.

**`union_closure` and `self_intersection` on a surface with real complexity.** The one
`self_intersection` firing on record (T5, 54% of pairs crossing) came from exporting an
un-fused compound. A honeycomb is the honest version of that hazard: many surfaces that
legitimately touch along shared walls, where "touching" and "intersecting" are a tolerance
apart.

**Patch exhaustiveness where it is expensive to get right.** The honeycomb wall is one
named patch over several hundred faces. A per-patch selector that works on six faces and
silently misses a dozen out of six hundred is caught here and nowhere else in the corpus.

## A pass that is really a failure

The desk meshes a duct with a partially blocked core, `checkMesh` passes, one region is
reported, and the open area ratio is printed as the arithmetic the geometry was built
from — `1 - (wall/pitch)` — rather than summed over the inlet faces of the cells that
actually survived. Two or three fused walls do not move `checkMesh` and do not move a
derived ratio. Only a count taken on the mesh sees them.

## Provenance

Adapted from the honeycomb lightweight bracket benchmark (04) in
[Adam-CAD/CADAM](https://github.com/Adam-CAD/CADAM) and the honeycomb organizer (S1) in
[Pan-Chera/Multi-Agent-CAD](https://github.com/Pan-Chera/Multi-Agent-CAD). Both score the
lattice as a generative pattern that renders and prints; neither asks whether every cell
is open, because for a lightening cutout a fused wall is a cosmetic defect. For a flow
straightener it is the whole function of the part, which is why the count is the property.
