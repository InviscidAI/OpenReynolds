# T5 — STEP assembly, fluid domain extracted

**Capability:** STEP prep. **Fixture:** C1's committed multi-solid STEP.

## Request

> Here is a STEP file of an assembly. Take the fluid volume inside it — the space the air
> actually flows through — cap the openings, and mesh it. Remove the small fillets that will
> not survive the cell size. Name the inlet, the outlet and the walls separately.

`geometry`: the committed fixture path.

**Largest extent:** `0.2` m — the largest dimension the exported surface should
span, for the `scale` probe. The whole union, not the part: an external-flow case
exports its far-field box too. Order of magnitude is enough; the probe fires
outside a factor of a hundred.

## Properties, measured on the delivered mesh

- extracted fluid volume, against the enclosure minus the solids
- the capping faces, each planar, with its area, measured on the meshed patches

## What this catches

**The capability §1 leads with**, end to end: "Ingest customer STEP/IGES, repair, defeature,
extract the fluid domain by boolean subtraction with capping, tag boundary faces using
geometric selectors re-derived on each run, and export per-patch STL."

**The untested thing `#14` refuses to call retired.** Whether output from a real CAD system
imports cleanly at all — "degenerate faces, what `healShapes` in fact repairs, how assembly
structure arrives, and whether units are declared per component". Supporting multi-solid
"moves the discovery from before implementation to during it". This prompt is where the
discovery lands.

**Prep has an unambiguous verifier** (§1): the geometry meshes or it does not. Unlike the
authoring prompts, a pass here means more than it does elsewhere.

## A pass that is really a failure

The fluid domain is extracted from the wrong enclosure — the space around the assembly
rather than the space inside it — and meshes perfectly. `locationInMesh` validation is what
catches it; nothing else does.
