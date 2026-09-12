# T12 — the published failure, asked of this desk

**Capability:** authoring, swept blades, root fillets at the cell scale.
**Fixture:** none — authored from the request.

## Request

> Mesh the water passage through a centrifugal impeller. A backplate of 60 mm radius and
> 4 mm thickness, a hub of 12 mm radius rising 18 mm from it, and 7 backward-curved blades
> 3 mm thick standing 14 mm off the backplate, sweeping from 16 mm radius at the inlet to
> 58 mm radius at the outlet, curving back 30 degrees against the direction of rotation.
> The blades meet the backplate and the hub with 2 mm fillets at the root. A shroud closes
> the blade tips 14 mm above the backplate. Mesh the water in the passages between the
> blades: entering axially down the hub bore, turning, and leaving radially at 58 mm.
> Remove the root fillets if they will not survive the cell size, and say so if you do.
> Name the inlet, the radial outlet, the blade surfaces, the backplate, the hub and the
> shroud separately.

## Properties the desk must measure and print

- the number of blade passages meshed, against the 7 the request names
- the blade thickness and the blade wrap angle, measured on the meshed blade patches
- the root fillet radius as meshed, or the statement that the fillets were removed and the volume that added
- the passage width at the inlet radius and at the outlet radius, measured on the mesh
- the water volume, against the shroud-to-backplate swept volume minus hub and blades
- every exported patch, exhaustive and disjoint over the domain's faces

## What this catches

**The one failure both open-source pipelines put on record.** The centrifugal impeller is
benchmark 08 in CADAM and P8 in MAC, and P8 item 13 — *"blade root fillets (where blades
meet the backplate and hub)"* — is MAC's **only** failure across 141 features: the fillet
code is present, OCCT's fillet generation conflicts on the geometry, and five repair
iterations cannot resolve it. This case is the same geometry asked of this desk, so the
corpus has one point where its result is directly comparable to a published number rather
than only to its own baseline.

**A defeature that is allowed, if it is declared.** T5's defeature cell was cut at the
240 s cap having printed nothing, and rank 9 of the last sweep is
`defeature_call_cut_without_printing_anything`. Here removing the fillets is a legitimate
answer — the request says so — and the failure is removing them silently. That makes this
the one case in the corpus that distinguishes a defeature that happened from a defeature
that was reported.

**A long-running single cell**, §6 candidate 6. Seven filleted swept blades booleaned
against a hub is the operation most likely to run past a cap legitimately. Nothing in the
corpus currently tests a slow operation that is working.

## A pass that is really a failure

The fillets fail to apply — as they do in MAC — the desk carries on with sharp roots, and
the passage meshes cleanly with `checkMesh` reporting nothing wrong, because a sharp root
is easier to mesh than a filleted one. The delivered mesh is better-conditioned than the
correct geometry would have been, the volume is off by the fillet volume, and the only
evidence is the fillet radius measured on the mesh against the 2 mm the request names.
A desk that prints `fillets: 2 mm` from its own parameter passes this without having built
one.

## Provenance

The geometry is the centrifugal impeller common to benchmark 08 in
[Adam-CAD/CADAM](https://github.com/Adam-CAD/CADAM) ("7 swept backward-curved blades", 10
dims · 1 colour) and P8 in
[Pan-Chera/Multi-Agent-CAD](https://github.com/Pan-Chera/Multi-Agent-CAD) ("centrifugal
impeller + 12 backward-curved blades + root fillets", 15 features), the latter sharing its
prompt with [earthtojake/text-to-cad](https://github.com/earthtojake/text-to-cad). Blade
count and radii are set here to CADAM's 7 rather than MAC's 12 so the passages stay wide
enough that the *fillet* is the marginal feature and not the passage. Fluid domain, patches
and properties are authored for this corpus.

**The two external results on this geometry disagree**, which is worth stating beside the
claim above: MAC failed item 13 and the `cad` skill scored P8 15/15 at ¥32.75 — 27× MAC's
cost. So the fillet is not simply beyond a pipeline of this kind; it is what MAC's cheaper
architecture gave up to get there. `docs/cad-benchmark-provenance.md` has the full map.
