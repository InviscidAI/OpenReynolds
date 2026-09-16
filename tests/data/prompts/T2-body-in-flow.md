# T2 — body in a flow box

**Capability:** authoring. **Runtime expectation:** the longest of the authoring set.

## Request

> A 40 mm diameter sphere in an open air flow, for an external aerodynamics solve. Put it in
> a domain that extends 5 diameters upstream, 15 downstream, and 5 to each side and above
> and below. Name the inlet, the outlet, the far field and the sphere separately. Resolve the
> boundary layer on the sphere with prism layers.

**Largest extent:** `0.8` m — the largest dimension the exported surface should
span, for the `scale` probe. The whole union, not the part: an external-flow case
exports its far-field box too. Order of magnitude is enough; the probe fires
outside a factor of a hundred.

## Properties, measured on the delivered mesh

- sphere diameter, 40 mm, measured on the meshed sphere patch
- achieved layer coverage on the sphere, as a percentage, **read off the mesh**
- the fluid volume, against the domain box minus the sphere

## What this catches

**Layer coverage claimed rather than measured.** snappy's layer stage "inserts prisms then
deletes them where a quality metric fails — on one hull it reached 88.6% coverage on its
first iteration, eroded to 66.7% by its fiftieth, printed 66.7% and exited 0". A desk that
reports what it asked for rather than what it got passes this prompt with a mesh that has no
boundary layer over a third of the body. cfMesh's `cartesianMesh` covered 100.0% of the same
wall and is the better first try where layers matter.

**Scale.** A sphere authored in millimetres and left there gives a 40 m sphere, at a
thousandth of the Reynolds number, with `checkMesh` and the picture both perfectly happy.

## A pass that is really a failure

Layer coverage of 60% reported as "layers added". The number is the deliverable, not the
fact that the stage ran.
