# T9 — a sub-cell clearance that decides the region count

**Capability:** authoring, internal flow across a gap narrower than the cell.
**Fixture:** none — authored from the request.

## Request

> Model a flooded gyroscope bearing and mesh the oil inside it. An outer ring, 30 mm
> outer radius and 23 mm inner radius, 10 mm tall, centred on the XY plane. Inside it an
> inner spinner, 22 mm outer radius and 15 mm inner radius, the same 10 mm tall, so a
> 1.0 mm radial clearance separates the two. The bore through the spinner, the annular
> clearance and the volume outside the spinner but inside the ring are all filled with
> oil and all connect through the clearance. Mesh that oil as one continuous fluid
> region. Name the outer ring's wetted face, the spinner's wetted face, and the two
> annular end faces at the top and bottom separately.

## Properties the desk must measure and print

- the radial clearance, measured between the two meshed wall patches, not restated from the input
- the number of connected mesh regions `checkMesh` reports, and the number expected
- the cell size across the clearance, and how many cells span it
- the oil volume, against the ring's swept volume minus the spinner's
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

## Provenance

Adapted from the print-in-place articulable gyroscope in
[Pan-Chera/Multi-Agent-CAD](https://github.com/Pan-Chera/Multi-Agent-CAD), whose
benchmark turns on holding a 0.4–1 mm clearance so the parts move freely off the build
plate. That repo scores the clearance as a printability feature; here the same clearance
is scored as a flow path, because a gap that prints is not the same claim as a gap that
meshes. The dimensions are theirs; the fluid domain, the patches and the properties are
authored for this corpus.
