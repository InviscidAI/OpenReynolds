# `tests/data/cad/` — where these files came from, and what they are not

Two sets, with opposite properties on purpose.

`real/` is one file a CAD system wrote. Nobody knows its volume in closed form and that
is the point: it is the input whose surprises are the reason for the chunk. `synthetic/`
is four files this repository wrote, where every number is exact by construction and
`build_synthetic.py` checks each one against the solid before the file is written.

---

## `real/ldrobot_ld19_lidar.step`

| | |
|---|---|
| What it is | The LDROBOT LD19 / D300 spinning lidar module — a moulded base, a rotor and a cover. |
| Taken from | <https://github.com/Myzhar/ldrobot-lidar-ros2>, path `3d_model/LD19.step`, at commit `296d3956b416553935ac65fc5cd5863d958d383a` (2021-09-21). Committed here byte-identical; only the filename changed. |
| Licence | **Apache-2.0**, the licence of the repository it was taken from. Redistribution is permitted with the notice; the Apache-2.0 text and the attribution above are that notice. Nothing here is sublicensed. |
| SHA-256 | `368bed304152e7b5669b1c60511aa2c82b8f152d0740c2d4a16db14562e00ab1` |
| Size | 110,029 bytes |
| Schema | AP214 (`AUTOMOTIVE_DESIGN`), written by ST-Developer v18.1 |
| Units | **Declares metres** — `#2301 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.) )` — and is 53.95 units across. A lidar module 54 m wide is not a thing, so the declaration is wrong by a factor of 1000 and the numbers are millimetres. See probe 3; this is a real file behaving badly and it is kept exactly as found. |

**Not written by OpenCASCADE.** `#14`'s complaint about the original probe was that "the
probe's file was written by OCC itself, the friendly case". This one was not: it was
modelled in a commercial CAD system and exported through STEP Tools' ST-Developer, the
translator most of the industry's STEP passes through. Its `originating_system` field is
two spaces — which is exactly why the gate does not read that field and measures the
geometry instead.

**What it carries**, measured rather than claimed (`scripts/cad_probes.py`, probe
`fixture` and probe 4):

- **3 solids** — `gmsh.model.occ.importShapes` returns three volumes, which is the
  multi-solid case the gate requires.
- **78 faces**: 43 planar, 28 cylindrical, 2 toroidal, 1 conical and **4 B-spline
  surfaces** — a genuinely curved surface that is not a canonical primitive.
- **Fillets of known radius**: two toroidal faces of minor radius 0.5 mm, and cylindrical
  faces at 0.5 mm and 0.1 mm. Probe 1 runs `BRepAlgoAPI_Defeaturing` against them and
  reports the outcome as an observation.
- Bounding box 53.95 × 46.84 × 31.35 (file units; millimetres in fact, metres by
  declaration); volume 41,193.7 mm³ = 4.11937 × 10⁻⁵ m³.
- **A wrong unit declaration that nothing refuses.** `cad_convert.py` reads "metre" off
  this file and converts without ceremony, because its refusal fires on a *missing*
  declaration and this one is present and false. Probe 3 records it. It is the most
  useful thing this fixture carries and it was not looked for.

### The hard case is only half met, and this is the half that is not

The plan's preferred gate assertion is that the fixture "carries at least one face OCCT's
checker flags **before** `healShapes` and not after — proving there was something real to
repair". It does not.

**OCCT's checker flags no face on this fixture before healShapes.** Zero invalid faces,
zero free edges, three closed solids; `BRepCheck_Analyzer` returns valid for the whole
shape. So this fixture is a real-CAD file and *not* a broken one, and there was nothing
here for repair to repair.

Running `healShapes` on it anyway is not a no-op, which is probe 4's finding and not a
property of the fixture: with gmsh's default arguments the three solids become **none**
and 354 free edges appear. `sewFaces=False` is the single argument that changes it. The
file is fine; the repair is what breaks it.

Five other candidate real-CAD STEP files were imported and checked the same way while
looking for one with a defect — an Onshape lidar, an ST-Developer PCB assembly, a u-blox
module library, a CERN-OHL machine frame and an E3D part. Every one of them came back with
zero faces flagged. A vendor STEP that OCCT rejects is not a thing one finds on demand.

**The consequences, stated rather than papered over:**

- The degenerate-face case is **unmet**. There is no degenerate face in this fixture.
- `healShapes`'s repair behaviour is **untested on real broken input**. Probe 4 records
  what it does on a healthy file, which is nothing, and that is all it records.
- C5's thin-wall and degenerate-face outcomes must be reported as **untested on real
  input**, not as passes. The plan says this in as many words and it is repeated here so
  that the chunk that reads this file does not have to go and find it.
- `scripts/cad_probes.py` enforces the admission rather than trusting it: when probe 4
  measures zero flagged faces before healing it requires the sentence in bold above to be
  present in this file verbatim, and exits non-zero if it is not. Removing the admission
  makes the probe run red.

### What *is* met

The real-CAD requirement itself. This is not the fallback the plan allowed for — an
OCC-authored multi-solid fixture is **not** what is committed here, and the sentence the
plan reserved for that case ("the real-CAD case is unmet") does not apply. A file from a
real CAD system, multi-solid, with fillets and B-spline surfaces, redistributable, is in
the tree. What is unmet is narrower and named above: it is clean, so it does not prove
repair.

---

## `synthetic/` — built by `build_synthetic.py`, and checked by it

Rebuild with:

    python3 tests/data/cad/build_synthetic.py

Every fixture is modelled in metres, written out in **millimetres with the unit
declared** (which is what a CAD system emits, and what `cad_convert.py`'s declared-unit
path exists for), then read back and its volume re-checked through that conversion.
`synthetic/answers.json` holds the answers in SI. They are not transcribed by hand: the
builder computes each one analytically, compares it against the solid, and refuses to
write the file if the two disagree.

| file | the exact answer |
|---|---|
| `box_with_duct.step` | 0.20 × 0.10 × 0.10 m box, a duct of r = 0.02 m bored the full length. **Fluid volume π r² L = 2.513274122871835 × 10⁻⁴ m³**; the solid that is left is 1.748672587712817 × 10⁻³ m³. One solid, seven faces. |
| `three_solids.step` | Three 0.1 m cubes, deliberately **not fused**. The second is offset 0.08 m in x, so it **overlaps the first by 2 × 10⁻⁴ m³** exactly; the third stands 0.4 m away and touches nothing. |
| `hollow_box.step` | Outer 0.100 × 0.080 × 0.060 m, void 0.090 × 0.074 × 0.052 m, centred. Walls are 5, 3 and 4 mm, so **the minimum wall thickness is 0.003 m** and it is a choice rather than a symmetry. Volume 1.3368 × 10⁻⁴ m³, twelve faces. |
| `filleted_block.step` | 0.08 × 0.05 × 0.06 m block, one fillet of **r = 0.005 m** on the through edge at x = 0, y = 0 of **length L = 0.06 m**. That edge is a **90° convex edge by construction**, which is the only case in which (1 − π/4)·r²·L is the volume the fillet removed: **3.219027549038276 × 10⁻⁷ m³**. Exactly one cylindrical face. |

The right angle is not incidental. Probe 1's whole assertion is that defeaturing recovers
that number to within 1%, and the formula is true for 90° and false for anything else — so
the angle is pinned in the builder, asserted there, and stated here.

### Licence

Written by this repository, MIT, like the rest of it.

---

## Anything else that came near this directory

Nothing else was committed. The search for a real-CAD fixture also passed over the NIST
MBE PMI validation models (public domain, and genuinely NX/Creo/CATIA/SolidWorks output —
but every one of the eleven geometry-only files is a **single** solid, so none of them
meets the multi-solid gate) and the AS1 assembly that ships in gmsh's demo tree (18 solids,
Unigraphics-authored, and with **no licence stated anywhere** — which is not the same as
permission).
