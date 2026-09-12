# T9 — a sub-cell clearance that decides the region count

**Capability:** authoring, internal flow across a gap narrower than the cell.
**Fixture:** none — authored from the request.

## Request

> Model a flooded gyroscope bearing and mesh the oil inside it. An outer ring, 30 mm
> outer radius and 23 mm inner radius, 10 mm tall, centred on the XY plane. Inside it an
> inner spinner, 22 mm outer radius and 15 mm inner radius, **8 mm tall and centred on the
> same plane**, so a 1.0 mm radial clearance separates it from the ring and a 1.0 mm axial
> gap separates each of its ends from the ring's. The bore through the spinner, the
> annular clearance and the two end gaps are all filled with oil and connect through those
> gaps, so the oil is one continuous region. Mesh it. Name the outer ring's wetted face,
> the spinner's wetted face, and the ring's two annular end faces separately, and
> **export one STL per patch to `constant/triSurface` before meshing.**

## Properties the desk must measure and print

- the radial clearance, measured between the two meshed wall patches, not restated from the input
- the axial end gap, measured the same way, at both ends
- the number of connected mesh regions `checkMesh` reports, and the number expected
- the cell size across the clearance, and how many cells span it, counted on the mesh
- the oil volume, against the ring's bore swept volume minus the spinner's
- every exported patch, exhaustive and disjoint over the domain's faces

## What this catches

**A gap the mesher closes silently.** At any cell size above 1 mm the clearance is
narrower than one cell, and `snappyHexMesh` will bridge it — the two chambers fuse, the
oil becomes one region for the wrong reason, and the mesh looks perfect. The distinction
between "one region because the clearance is meshed" and "one region because the
clearance was erased" is invisible in `checkMesh`'s region count alone, and the only thing
that separates them is the cells-across-the-gap number this case requires printed.

**`coverage`, which has never returned a verdict.** The sweep of 2026-09-12 records that
`coverage` and `scale` have never been anything but `skipped` in any case of any sweep —
§6 candidate 3. This case exports a surface and names a wall-normal resolution, so both
have an input.

**An STL-exporting case that is not a STEP ingest.** §6 candidate 4: five of eight cases
cannot reach `union_closure`, `normals` or `self_intersection` by construction, because
they never write `constant/triSurface`. This one does, from authored geometry.

## A pass that is really a failure

The desk meshes the oil, `checkMesh` reports one region, and it reports success —
having never resolved the clearance at all. One region is the expected answer, so the
number agrees with the request while the geometry does not. The cells-across-the-clearance
count is what tells the two apart, and it is the property most likely to be replaced by
arithmetic on the requested cell size rather than measured on the mesh.

## Corrected on 2026-09-12, after the first sweep

**The first version of this brief could not be passed.** It gave the spinner the same 10 mm
height as the ring, flush, and then asserted that the bore and the annular clearance "all
connect through the clearance". With both parts flush there is no axial path between them:
two regions is the geometrically correct answer, and sweep
`core+bench12-20260912-123315-8815` recorded the desk building exactly what was described
and `checkMesh` reporting `*Number of regions: 2` — correctly, against a brief that expected
one. The 8 mm spinner height above is the fix, and it makes the one-region expectation true.

The original also named patches without requiring them exported, so no probe could read the
surface. Four cases made that mistake and all four returned `n/a` on every probe. The export
is now required in the request.

## Provenance

Adapted from the print-in-place articulable gyroscope in
[Pan-Chera/Multi-Agent-CAD](https://github.com/Pan-Chera/Multi-Agent-CAD), whose
benchmark turns on holding a 0.4–1 mm clearance so the parts move freely off the build
plate. That repo scores the clearance as a printability feature; here the same clearance
is scored as a flow path, because a gap that prints is not the same claim as a gap that
meshes. The dimensions are theirs; the fluid domain, the patches and the properties are
authored for this corpus.
