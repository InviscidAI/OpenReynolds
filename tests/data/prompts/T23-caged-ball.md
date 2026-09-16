# T23 — a solid that touches nothing

**Capability:** authoring, a free-floating internal body.
**Fixture:** none — authored from the request.

## Request

> Mesh the water in a flooded ball cage. A 40 mm cube has a spherical hollow of 16 mm radius
> at its centre, and a 15 mm radius solid ball sits inside that hollow, free — 1 mm of water
> all around it. Six 12 mm radius holes, one through the centre of each cube face, connect
> the hollow to the outside. The cage and the ball are both solid; the water fills the 1 mm
> shell around the ball, the rest of the hollow, and the six holes. Mesh that water. Name the
> ball's surface, the hollow's wall, the six hole walls and the six outer openings
> separately, and export one STL per patch to `constant/triSurface` before meshing.

**Largest extent:** `0.04` m — the largest dimension the exported surface should
span, for the `scale` probe. The whole union, not the part: an external-flow case
exports its far-field box too. Order of magnitude is enough; the probe fires
outside a factor of a hundred.

## Properties, measured on the delivered mesh

- the ball's diameter, measured on the meshed ball patch, and its centre position
- the water gap between ball and hollow wall, measured between the two meshed patches, at a hole axis and at a diagonal
- the water volume, against the hollow plus six holes minus the ball

## What this catches

**An internal boundary with no connection to the outer one.** Every solid in this corpus so
far is either the domain's outer wall or attached to it. The ball is a closed surface floating
inside the fluid, which is the topology `snappyHexMesh` handles by refining onto a surface that
does not bound the region it is refining — and the case where a `locationInMesh` point can
easily end up *inside the ball* rather than in the water. That would mesh the ball and discard
the water, producing a clean mesh of exactly the wrong volume.

**`location_in_mesh`, the third time and the first time it should be meaningful.** §4.1
established the probe as a false positive on external flow, where `outside` is correct. Here
there is a genuinely correct answer — in the water, not in the ball, not in the cage — and a
genuinely wrong one. If the probe cannot distinguish them, the finding is about the probe. With
T2 and T14 as the external-flow pair and this as the internal case, the three together settle
what the probe can and cannot do.

**A gap measured in two different directions.** The 1 mm shell is uniform, so measuring it
along a hole axis and along a cube diagonal should give the same number — and will not if the
ball has been snapped to the mesh or displaced by the boolean.

## A pass that is really a failure

The `locationInMesh` point lands inside the ball. `snappyHexMesh` meshes the ball's interior,
`checkMesh` is clean, one region, the ball patch is exported and closed, and every dimension
of the ball is exactly right. The volume is 14,137 mm³ — the ball's — where the water is about
28,600: the hollow's 17,157, less the ball, plus about 25,600 in the six holes. **A factor of
two is only obvious if the expected number is printed beside it**, and the brief asks for
exactly that comparison because a lone `volume 1.41e-05 m³` gives the desk nothing to notice
with. A gap measured between two patches that both exist is the other tell; in this failure
there is only one patch.

## Provenance

Adapted from the ball-in-cage print-in-place demo in
[Pan-Chera/Multi-Agent-CAD](https://github.com/Pan-Chera/Multi-Agent-CAD), which states it as
the harder of its two articulable cases: "not only must the pipeline model each body
separately, it must also precisely control clearances". Their dimensions are used as given —
40 mm cube, 16 mm hollow, 15 mm ball, 1 mm clearance, six 12 mm holes. Their criterion is that
the ball rattles and cannot escape; here it is that the water around it is what gets meshed.
