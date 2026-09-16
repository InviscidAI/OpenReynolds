# T16 — an annular duct, and its area at five stations

**Capability:** authoring, annular internal flow, area as a function of position.
**Fixture:** none — authored from the request.

## Request

> Mesh the bypass air path of a turbofan. An outer cowl whose inner surface runs from 450 mm
> radius at the fan exit, to 430 mm at mid-duct, to 410 mm at the nozzle, over 900 mm of
> length. Inside it a core cowl whose outer surface runs from 280 mm radius at the fan exit,
> to 310 mm at mid-duct, to 250 mm at the nozzle, over the same length. Both are surfaces of
> revolution about the engine axis, blended smoothly between the three radii given. Mesh the
> air in the annulus between them. Name the outer cowl, the core cowl, the fan-exit inlet and
> the nozzle outlet separately, and export one STL per patch to `constant/triSurface` before
> meshing.

## Properties the desk must measure and print

- the hub and tip radius at three equally spaced axial stations, measured on the two meshed cowl patches
- the flow area at each of those stations, measured on the mesh, and the area ratio nozzle to inlet
- the air volume, against the analytic integral of the measured areas along the axis

## What this catches

**An area schedule, which is the quantity the part exists to produce.** A bypass duct is
defined by how its flow area varies along the axis — it has a bulge at mid-duct here, where
the core cowl grows faster than the outer cowl shrinks — and that schedule is what sets the
pressure gradient. The desk knows the six radii it was given; the brief asks for the area at
five stations, which requires evaluating the blended surfaces rather than the inputs.

**A non-monotonic dimension.** The annulus height is 170 mm at the inlet, 120 mm at mid-duct
and 160 mm at the nozzle. A desk that linearly interpolates between endpoints, as is the
easiest way to loft two surfaces, gets mid-duct wrong by 25% and every printed number at the
endpoints right. No other case in the corpus has a property that cannot be got right by
interpolating the extremes.

**A large-radius, thin-annulus mesh** — 450 mm outer with a 120 mm gap — where the cell
aspect ratio is the thing most likely to make `checkMesh` complain, and the corpus has
nothing else at this scale.

## A pass that is really a failure

The two cowls are lofted linearly from fan exit to nozzle, ignoring the mid-duct radii. The
duct meshes perfectly, the inlet and outlet areas are exactly right, the volume is close, and
the mid-duct bulge — the only interesting feature of the geometry — is absent. Measuring the
area at five stations is what finds it; measuring it at two cannot.

## Provenance

Adapted from the turbofan jet engine (11) in
[Adam-CAD/CADAM](https://github.com/Adam-CAD/CADAM) — "a bypass cowl, an internal core with
compressor/turbine stages" — reduced to the bypass annulus alone, which is the part with a
wetted flow path. Radii and the mid-duct bulge are authored here; CADAM's model has two
dimensions and ten colours, which is a statement about what it was scored on.
