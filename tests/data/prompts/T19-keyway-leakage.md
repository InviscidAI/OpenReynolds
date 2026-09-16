# T19 — an annular seal, and the keyway that defeats it

**Capability:** authoring, a leak path around an otherwise sealed annulus.
**Fixture:** none — authored from the request.

## Request

> Mesh the oil leaking past a shaft seal. A stepped shaft runs through a housing bore: the
> shaft is 40 mm diameter over the sealed length, stepping up to 50 mm at one end and down
> to 30 mm at the other, and the bore is 40.1 mm diameter over 60 mm of length, so the
> radial clearance is 0.05 mm. A 12 mm wide, 5 mm deep keyway is cut along the full sealed
> length of the shaft. Oil at the 50 mm end leaks to the 30 mm end through the clearance and
> the keyway together. Mesh the oil in that path. Name the shaft surface, the keyway floor
> and flanks, the bore surface, the high-pressure end face and the low-pressure end face
> separately, and export one STL per patch to `constant/triSurface` before meshing.

## Properties the desk must measure and print

- the radial clearance, measured between the meshed shaft and bore patches, at three angular positions away from the keyway
- how many cells span the 0.05 mm clearance, counted on the mesh
- the keyway depth and width, measured on the meshed keyway patches

## What this catches

**Two flow paths in parallel whose areas differ by an order of magnitude.** The annular
clearance is 40 mm circumference by 0.05 mm — about 6 mm² — and the keyway is 12 by 5, or
60 mm². Nearly all the leakage goes through the keyway, which means a desk that meshes the
keyway well and the clearance badly gets a plausible answer for the wrong reason, and one
that closes the clearance entirely still reports a connected region. **The region count
cannot fail here**, which makes the cells-across-the-clearance count the only property that
distinguishes the two.

**The smallest gap in the corpus by a factor of twenty.** T9's clearance is 1 mm, T13's
0.2 mm; this is 0.05 mm against a 40 mm diameter, a ratio of 1:800. Whether the desk chooses a
cell size that can resolve it, or declines and says why, is the case.

**A dimension that is 40.1 against 40.** Two radii that differ in the third significant
figure is where a desk that rounds, or that works in a unit it has not stated, produces
either zero clearance or a negative one. T6 is the corpus's unit case and it is about a
declared unit; this is about precision within one.

## A pass that is really a failure

The clearance is meshed one cell thick, or collapses entirely, and the keyway carries all the
flow. The oil is one region, `checkMesh` is clean, the leak path exists, and the computed
leakage is within a few per cent of correct — because the keyway really does dominate. The
mesh is nonetheless wrong about the annulus, and the next case has no keyway.

## Provenance

Adapted from the stepped shaft with keyway and end chamfer that is P4 in
[Pan-Chera/Multi-Agent-CAD](https://github.com/Pan-Chera/Multi-Agent-CAD), 11 features,
passed 11/11 by MAC and 11/11 by the `cad` skill. The shaft and keyway are theirs; the
housing bore, the 0.05 mm clearance and the leak path are authored here. That both pipelines
scored it full marks as a solid is the reason it is interesting as a fluid domain: the part is
easy and the gap around it is not.
