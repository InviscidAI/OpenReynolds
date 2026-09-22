# T4 — cold plate with fins

**Capability:** authoring. **Runtime expectation:** long; the boolean is the cost.

## Request

> A liquid cold plate: a 60 mm x 40 mm x 8 mm aluminium block with a serpentine channel
> milled through it, 4 mm wide and 5 mm deep, making three passes across the plate with 3 mm
> of metal between passes. Build the aluminium block and cut the channel out of it, then take
> the water as what the cut removed -- I want the solid to exist, because the 3 mm webs are
> where I expect trouble. Mesh the water side only. Inlet and outlet are the two open ends
> of the channel.

**Largest extent:** `0.06` m — the largest dimension the exported surface should
span, for the `scale` probe. The whole union, not the part: an external-flow case
exports its far-field box too. Order of magnitude is enough; the probe fires
outside a factor of a hundred.

## Properties, measured on the delivered mesh

- channel width 4 mm and depth 5 mm, measured on the meshed channel walls at more than one station
- number of passes, 3, counted on the mesh
- minimum local width of the fluid domain, measured on the mesh
- the metal web between adjacent passes, 3 mm, measured on the aluminium solid the
  channel was cut from

## What this catches

**Thin-wall booleans, OCCT's known weak case.** §2: "The fluid-domain boolean is a large
subtraction against thin-walled fin geometry, OCCT's known weak case." 3 mm of metal between
5 mm-deep passes is that case deliberately. If it fails here it fails on a customer's
heatsink, and this is where the Manifold fallback held in reserve gets its evidence.

**Minimum width, which nothing in the repo could measure before.** `#11` keeps the
definition because of cases like this one: "local width is well posed as twice the inscribed
-sphere radius [...] the same field on the solid gives minimum wall thickness". A plate whose
wall has been eaten to 0.4 mm somewhere by a boolean that nearly failed looks correct in
every picture.

## A pass that is really a failure

Every dimension correct and the channel in the wrong place — "a cold plate can pass every
check and still have the channel in the wrong place", §1's own example of where v1 is
exposed. Named here so the reviewer's closing condition has somewhere to look.
