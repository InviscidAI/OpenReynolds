# T18 — twelve fin gaps, and the one that is measured

**Capability:** authoring, external flow through a repeated narrow gap.
**Fixture:** none — authored from the request.

## Request

> Mesh the cooling air around a radial engine cylinder. The barrel is 90 mm outside
> diameter and 120 mm tall, carrying 12 annular cooling fins evenly spaced along it: each
> fin 140 mm outside diameter and 3 mm thick, leaving a 6 mm air gap between neighbouring
> fins. A domed head closes the top. The cylinder sits in a 300 x 300 mm square duct
> running 150 mm below and 300 mm above it, with air blown along the cylinder axis. Mesh the
> air outside the cylinder and between the fins. Name the barrel, the fins, the head, the
> duct walls, the inlet and the outlet separately, and export one STL per patch to
> `constant/triSurface` before meshing.

**Largest extent:** `0.62` m — the largest dimension the exported surface should
span, for the `scale` probe. The whole union, not the part: an external-flow case
exports its far-field box too. Order of magnitude is enough; the probe fires
outside a factor of a hundred.

## Properties, measured on the delivered mesh

- the fin gap, measured between two adjacent meshed fin patches, at the innermost and outermost radius
- the number of fins, counted on the meshed fin patch
- how many cells span one fin gap, counted on the mesh

## What this catches

**Twelve instances of the same narrow gap, where the corpus has only ever had one.** T9 has a
single 1 mm clearance and T13 a single helical one. Twelve 6 mm gaps in series is the case
where a mesher's refinement decisions are made once and applied everywhere, so a gap that
closes closes twelve times, and the wetted area — the quantity the part exists for — drops by
a measurable fraction rather than vanishing.

**Wetted area as an independently checkable number.** Twelve annuli of 140 mm outside and
90 mm inside diameter have an analytic area. A fin merged into its neighbour removes two
faces from that total, which is visible in a number the desk can compute two ways — the
comparison T11 did well and the rest of the corpus did not.

**A repeated feature counted rather than asserted.** Five cases now ask for a count taken on
the mesh. If the desk prints `fins: 12` from its input on this one too, the pattern is no
longer a finding about four cases but about the desk.

## A pass that is really a failure

The fins are built as a single revolved solid with grooves cut into it rather than as twelve
separate annuli, or the boolean merges two adjacent fins across their 6 mm gap. Either way the
air meshes, the region count is right, the barrel and head dimensions are right, and the fin
gap is reported as the 6 mm requested. The wetted area is short by two annuli and nothing
downstream can see it.

## Provenance

Adapted from the 9-cylinder radial aircraft engine (10) in
[Adam-CAD/CADAM](https://github.com/Adam-CAD/CADAM) — "each cylinder with stacked cooling
fins and a domed cylinder head" — and from the radial-engine cylinder that is P7 in
[Pan-Chera/Multi-Agent-CAD](https://github.com/Pan-Chera/Multi-Agent-CAD), where it scores 17
features including "12 fins". P7 is one of the prompts MAC passed 17/17, so the geometry is
known to be buildable by a pipeline of this kind; what is asked here is what the gaps measure.
