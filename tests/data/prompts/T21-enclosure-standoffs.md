# T21 — blind holes that must not become through holes

**Capability:** authoring, an open-topped cavity with blind features.
**Fixture:** none — authored from the request.

## Request

> Mesh the air in an open-topped electronics enclosure being cooled by a fan above it. The
> box is 160 x 100 mm and 40 mm deep internally, with 3 mm walls and no lid. Four cylindrical
> standoffs 10 mm diameter and 25 mm tall rise from the floor, 15 mm in from each corner,
> each with a 4 mm diameter blind hole 12 mm deep down its axis from the top. Air enters
> through the open top and leaves through a 30 x 20 mm vent in one short wall, 10 mm above the
> floor. Mesh the air inside: around the standoffs and inside their blind holes. Name the open
> top, the vent, the enclosure walls and floor, the standoff outer surfaces, and the blind
> hole surfaces separately, and export one STL per patch to `constant/triSurface` before
> meshing.

## Properties the desk must measure and print

- each blind hole's depth, measured on the mesh, and the 13 mm of standoff remaining below it
- the blind hole diameters, measured on the meshed hole patches
- the four standoffs' heights and centre positions, measured on the mesh, against 15 mm in from each corner
- the vent's area and its height above the floor, measured on the mesh
- the air volume, against the internal box volume minus four standoffs plus four blind holes
- the number of connected mesh regions `checkMesh` reports, and the number expected

## What this catches

**A blind hole, which is a through hole that stopped.** The baseline failure list has nothing
about depth. A hole drilled 12 mm into a 25 mm standoff leaves 13 mm of solid below it; drilled
through, the air volume changes by four small cylinders and the region count does not move,
because the holes open upward into the same air either way. **`checkMesh` cannot see this at
all** — the topology is identical — and the volume difference is under 1%. Only the measured
depth finds it, which makes this the corpus's cleanest test of whether a measured length is
actually measured.

**Four features whose positions are stated relative to something.** "15 mm in from each
corner" is a position that must be differenced against the measured wall, which is exactly the
form of the baseline's T7 failure: *"5 mm above the floor" is `GAP_Z` restated in prose, never
differenced against the cavity extent*. T7 is the cheapest run in the corpus and skipped the
property that would have caught its false pass. This asks the same question of four features
and a vent.

**An open boundary that is not an inlet pipe.** The whole top face is open to the atmosphere,
which is a patch with no solid behind it — a different thing from T2's far field and from an
inlet, and something the corpus has not asked for.

## A pass that is really a failure

The standoffs are modelled as tubes rather than posts with blind holes — open top and bottom —
and the air meshes, one region, every external dimension correct, volume within a per cent. Or
the holes are omitted entirely and the standoffs are solid, which changes the volume by the
same tiny amount in the other direction. The depth is the only number that separates the three
possibilities, and it is the number most easily printed from the input.

## Provenance

Adapted from the open-top enclosure that is P5 in
[Pan-Chera/Multi-Agent-CAD](https://github.com/Pan-Chera/Multi-Agent-CAD) — "open-top
enclosure + 4 standoffs + blind holes + outer fillets", 12 features. It is the one prompt
where the `cad` skill failed a feature (item 11, fillet scope) and MAC passed 12/12. The
standoffs and blind holes are theirs; the cooling flow, the vent and the measured depths are
authored here. The outer fillets are dropped, being outside the wetted surface.
