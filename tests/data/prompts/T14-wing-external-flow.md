# T14 — an aerofoil section measured at three spanwise stations

**Capability:** authoring, external flow, a lofted surface measured rather than restated.
**Fixture:** none — authored from the request.

## Request

> Mesh the air around a tapered wing section. The wing uses a NACA 2412 profile throughout:
> 120 mm chord at the root tapering to 80 mm at the tip over a 200 mm span, with no sweep
> and no twist, the root fixed to a wall. Put it in a flow domain extending 4 root chords
> upstream, 8 downstream, and 3 to each side and above. Mesh the air outside the wing.
> Name the wing surface, the root wall, the inlet, the outlet and the far field separately,
> and export one STL per patch to `constant/triSurface` before meshing.

**Largest extent:** `1.56` m — the largest dimension the exported surface should
span, for the `scale` probe. The whole union, not the part: an external-flow case
exports its far-field box too. Order of magnitude is enough; the probe fires
outside a factor of a hundred.

## Properties, measured on the delivered mesh

- the chord at root and at tip, measured on the meshed wing patch
- the maximum thickness at those same stations, measured on the meshed wing patch, and the thickness-to-chord ratio
- the fluid volume, against the domain box minus the wing

## What this catches

**A profile the desk generated and can therefore restate.** NACA 2412 has a closed-form
thickness distribution, so the desk knows 12% of chord without measuring anything. The brief
asks for the thickness *on the meshed patch* at three stations, which is the one form of the
number that a taper error, a loft defect or a snapping artefact would move.

**Blockage, which the baseline's T2 never printed** — the word did not appear in that run at
all, and T2 is the corpus's other external-flow case.

**`location_in_mesh`'s known false positive, a second time.** §4.1 established that for
external flow the probe necessarily reports `outside` because it reads only
`constant/triSurface` and has no notion of the background box. One case cannot distinguish a
probe defect from a case defect. Two unrelated external-flow cases both reporting `outside`
while delivering a correct fluid-outside mesh can.

## A pass that is really a failure

The desk builds the wing from the NACA equations, prints `t/c = 0.120` at all three stations
because that is what the equations say, and delivers a clean mesh whose tip section was
scaled rather than re-lofted — or whose trailing edge snapped shut. Every printed number
agrees with the profile because every printed number came from it.

## Provenance

Adapted from the NACA 2412 tapered wing (05) in
[Adam-CAD/CADAM](https://github.com/Adam-CAD/CADAM) — "true airfoil from the NACA equations,
tapered loft", 120 mm root chord to 80 mm tip over a 200 mm span. Its spar tubes and
lightening holes are dropped, being internal structure with no wetted surface. The flow
domain and the measured-at-stations properties are authored here.
