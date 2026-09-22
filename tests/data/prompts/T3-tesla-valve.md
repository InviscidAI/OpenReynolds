# T3 — Tesla valve

**Capability:** authoring. **Runtime expectation:** the hardest of the set.

## Request

> A Tesla valve: a straight main channel with four teardrop-shaped bypass loops branching off
> it and rejoining it downstream, so that flow one way runs straight through and flow the
> other way is diverted into the loops and opposes itself. Channel width 6 mm, overall length
> about 120 mm, plane (2D). Inlet at one end, outlet at the other.

**Largest extent:** `0.12` m — the largest dimension the exported surface should
span, for the `scale` probe. The whole union, not the part: an external-flow case
exports its far-field box too. Order of magnitude is enough; the probe fires
outside a factor of a hundred.

## Properties, measured on the delivered mesh

- number of bypass loops, 4, counted on the mesh
- channel width, 6 mm, measured on the mesh in the main channel and in a loop
- **every loop both branches from and rejoins the main channel** — established on the mesh. A region count cannot decide this: a loop that dead-ends is still one region

## What this catches

**The failure that ended the previous stack**, in its author's words: "you ask it to make a
tesla valve / it gives circles connected by a line / making edits was hell." This is the
shape the deleted 63,943-line geometry stack could not build, and it is in the set for that
reason rather than because a Tesla valve is a common request.

**Connectivity that no count can see.** Four loops of the right width in the right places,
one of which never rejoins the channel, passes every dimensional check. The connected-region
count is the check that sees it, and nothing else in the finish check does.

## A pass that is really a failure

Four loops, correct width, correct length, and the device does not rectify because the loops
meet the channel at the wrong angle. The checks cannot see this. It is the J-Geo against
J-Sem gap, and it is what the reviewer agent's closing condition is looking for.
