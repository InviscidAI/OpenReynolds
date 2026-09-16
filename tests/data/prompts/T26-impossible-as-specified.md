# T26 — a spec that cannot be built as written

**Capability:** authoring, noticing that the request is self-inconsistent.
**Fixture:** none — authored from the request.

## Request

> Mesh the air in a stairwell. A cylindrical shaft 2400 mm internal diameter and 3000 mm tall.
> A central column 200 mm in diameter runs its full height. Twenty treads spiral around the
> column, each 1100 mm long radially, 250 mm deep, 40 mm thick, rising 150 mm per tread and
> turning 18 degrees per tread, with each tread's inner end reaching to the column so it is
> supported. Mesh the air in the shaft, around the column and between the treads. Name the
> shaft wall, the column, the tread surfaces, the floor, and the opening at the top
> separately, and export one STL per patch to `constant/triSurface` before meshing.

**Largest extent:** `3.0` m — the largest dimension the exported surface should
span, for the `scale` probe. The whole union, not the part: an external-flow case
exports its far-field box too. Order of magnitude is enough; the probe fires
outside a factor of a hundred.

## Properties, measured on the delivered mesh

- the gap or overlap between each tread's inner end and the column, measured on the mesh, with its sign
- the number of treads whose inner end is in contact with the column, established on the mesh
- the air volume, against the shaft minus column minus twenty treads

## What this catches

**A request whose numbers do not close, where the right answer is to say so.** The column is
200 mm in diameter, so its surface is at 100 mm radius. A tread 1100 mm long radially reaching
the 1200 mm shaft wall starts at 100 mm — exactly tangent to the column, touching it along a
single line with no solid overlap. As a solid model that is a zero-thickness connection: the
treads are not supported, the geometry is either non-manifold or disconnected depending on
tolerance, and "so it is supported" is false of the numbers given.

**The desk has three defensible responses and one indefensible one.** It may refuse and say
the numbers do not close; it may correct the tread length or the column diameter, state the
correction and its reason, and proceed; or it may build the tangency and report that the
treads are unsupported. What it must not do is build it, let the tolerance decide whether the
tangency fuses, and report success — because then whether the stairwell is one region or
twenty-one is decided by a tolerance nobody chose.

**The gap's sign, which is the whole measurement.** Zero, positive and negative are three
different geometries here and all three mesh. The corpus has no other property whose sign
carries the finding.

## A pass that is really a failure

OCCT's boolean fuses the tangent treads to the column — as it often will, since the tangency is
within tolerance — and the air meshes as one region with twenty supported treads. Every
dimension is exactly as requested. The model is correct *and* the specification was
impossible, so the desk has silently resolved an ambiguity in the customer's favour and left no
record that there was one. The next such spec resolves the other way and nothing explains why.

## Provenance

**This is the case where an external pipeline is on record doing the right thing.** The
miniature spiral staircase is P9 in
[Pan-Chera/Multi-Agent-CAD](https://github.com/Pan-Chera/Multi-Agent-CAD), 16 features, and its
published report logs a **defensive correction** on item 8: the original "tread inner end
approaches column" design "causes floating geometry (the tread and column are only tangent at a
single point, with no solid connection), and the model would fracture at that point during 3D
printing. MAC proactively corrected this parameter based on physical common sense, keeping a
safe overlap." The `cad` skill baseline, on the same prompt, produced the disconnected version —
their report shows the two side by side.

So this case has an external result in both directions: one pipeline noticed and fixed it, one
did not. The dimensions here are scaled to a full-size stairwell and the flow domain is
authored, but the tangency is theirs and it is reproduced deliberately.
