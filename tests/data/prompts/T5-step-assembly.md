# T5 — STEP assembly in an external flow

**Capability:** STEP prep. **Fixture:** C1's committed multi-solid STEP.

## Request

> Here is a STEP file of an assembly — a lidar unit, three bodies, in whatever units the
> file declares. Mesh the air flowing around it. Put it in a domain reaching 150 mm
> upstream, 400 mm downstream, and 150 mm to each side, above and below. Remove the small
> fillets that will not survive the cell size. Name the inlet, the outlet, the far field
> and the device separately.

`geometry`: the committed fixture path.

**Largest extent:** `0.6` m — the largest dimension the exported surface should
span, for the `scale` probe. The whole union, not the part: an external-flow case
exports its far-field box too. Order of magnitude is enough; the probe fires
outside a factor of a hundred.

## Properties, measured on the delivered mesh

- the device's overall size, measured on the meshed device patches, against the dimensions
  the file declares in its own units
- the number of separate bodies in the device surface, counted on the mesh, against the
  three the file contains
- the fluid volume, against the domain box minus the three solids

## What this catches

**The capability §1 leads with**, minus one step: "Ingest customer STEP/IGES, repair,
defeature, … tag boundary faces using geometric selectors re-derived on each run, and
export per-patch STL." Extraction by boolean subtraction with capping is **not** tested
here and is not tested anywhere; see below.

**The untested thing `#14` refuses to call retired.** Whether output from a real CAD system
imports cleanly at all — "degenerate faces, what `healShapes` in fact repairs, how assembly
structure arrives, and whether units are declared per component". Supporting multi-solid
"moves the discovery from before implementation to during it". This prompt is where the
discovery lands.

**Defeaturing against a real part.** The fixture carries fillets at r = 0.5 mm and
r = 0.1 mm — genuinely below any cell size this domain will carry — on a device 54 mm
across. Whether they survive, and whether removing them costs the shape, is the question.

**Prep has an unambiguous verifier** (§1): the geometry meshes or it does not.

## A pass that is really a failure

One of the three bodies is silently dropped — by a `healShapes` call, by a selector that
found two solids where there are three, or by an import that took only the first — and the
remaining two mesh perfectly. Volume moves by a few percent and nothing else complains. The
body count on the meshed device surface is what catches it.

The other one: `locationInMesh` lands inside a body rather than in the air, and the mesh is
of the device's interior. That probe reads `inside` for an internal case and `outside` for
this one, so **`outside` is the right answer here** and a run that reports it is not failing.

## History — why this case is external flow

Until 2026-09-17 this case asked for the fluid volume *inside* the assembly, capped at its
openings. **There is no such volume.** Measured on the fixture: three solids, every one a
single shell, base-to-body and body-to-emitter both touching at 0.000 mm, and a padded box
minus the assembly is one connected region. The only cylinders are Ø5 mm corner fillets and
Ø5.5 / Ø2.6 mm mounting holes through a 3.5 mm plate.

**Ten runs across every sweep on record, four models, and not one reached `checkmesh_ok`.**
The `core+cad_export` run refused, correctly, and was scored `passed: false` because the
case expected `done`. The case had been grading the right answer as a failure for its whole
life.

The search for a replacement fixture with a real internal passage came up empty: NIST's
MBE PMI set is public domain but every part is single-shell — FTC-07 reads as a box and
cuts open as a C-section channel — Ultimaker's parts are CC-BY-NC, and the one genuine duct
found, Prusa's print-fan shroud, is GPL-2.0. So the case was reposed to what this fixture
can actually support.

**Extract-the-internal-domain-by-boolean-with-capping is now an uncovered capability.** It
left with this rewrite and nothing replaced it. That is recorded here so it is not
rediscovered as a surprise.
