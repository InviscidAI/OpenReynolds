# T17 — one plenum, eight runners, and whether they are all open

**Capability:** authoring, branching internal flow, per-branch measurement.
**Fixture:** none — authored from the request.

## Request

> Mesh the air inside a V8 intake manifold. A plenum 300 mm long, 120 mm wide and 80 mm
> tall sits in the valley between the banks, fed by a single 70 mm diameter throttle inlet
> at one end. Eight runners leave it, four from each side, evenly spaced along its length:
> each is a 38 mm diameter round tube that leaves the plenum wall horizontally, turns
> through 90 degrees on a 60 mm centreline radius, and ends 90 mm from the plenum wall at a
> flat port face. Mesh the air in the plenum and all eight runners as one volume. Name the
> throttle inlet, the plenum walls, the runner walls, and each of the eight port faces
> separately, and export one STL per patch to `constant/triSurface` before meshing.

## Properties the desk must measure and print

- the number of runners open to the plenum, established on the mesh — one connected region does not establish it
- each runner's cross-sectional area at its port face, measured on the mesh, and the spread across the eight
- the plenum volume and the total runner volume, separately, measured on the mesh

## What this catches

**Nine patches that must each be found and named, of which eight are nearly identical.**
Geometric selectors that key on "the flat face at the end of a tube" have eight candidates
here, and the corpus's patch-naming failures to date — T1's patches named by bounding box,
T10's nearest-centroid collision that put two surfaces on one face — are exactly the failures
that scale with candidate count. Eight is enough for a heuristic to collide and few enough
that a collision is unmistakable.

**A boolean union of eight tubes into one plenum**, where losing one is a blocked cylinder
rather than a cosmetic defect. Rank 4 of the baseline is
`boolean_union_of_swept_bands_loses_material` on four bypass loops; T11 asked it of a hundred
hex prisms and the desk got all 105. Eight swept round tubes meeting a flat wall at a
tangency is a different and harder boolean than either.

**A spread, which is a measurement no single number can fake.** Eight runners that should be
identical give eight areas; printing the spread requires having measured each. A desk
restating its inputs prints one number eight times.

## A pass that is really a failure

Seven runners union cleanly and the eighth meets the plenum wall tangentially, leaving a
sliver that the mesher closes. The air meshes, `checkMesh` reports one region — because seven
open runners and a plenum are still one region — and the eighth port face is still exported,
still named, and no longer connected to anything. Only a count of runners open *to the
plenum*, taken on the mesh, sees it.

## Provenance

Adapted from the V8 engine benchmark (13) in
[Adam-CAD/CADAM](https://github.com/Adam-CAD/CADAM), its largest at 22 dimensions and 8
colours — "an intake manifold in the valley" — reduced to the manifold's internal air path,
which is the only part of a V8 with a wetted flow domain. Dimensions are authored here.
