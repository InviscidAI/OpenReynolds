# The CAD probes and their answers (C1)

Four measurements the CAD plan is built on and none of which had been taken: whether
defeaturing is reachable, whether `Mesh.StlLinearDeflection` does anything on our path,
what the pinned gmsh/OCC does about unit declarations, and what `healShapes` in fact
repairs on a file a CAD system wrote.

Run them:

    python3 scripts/cad_probes.py              # run all, compare with the record, exit 0/1
    python3 scripts/cad_probes.py --dump       # print the measurements as JSON
    python3 scripts/cad_probes.py --only probe2_deflection
    python3 scripts/cad_probes.py --no-pytest  # skip probe 3's suite re-run (the slow half)

About a minute, most of it probe 3 re-running `tests/test_toolbox_cad_convert.py` and
probe 1 defeaturing the real fixture. The machine copy of every answer below lives in
`RECORDED` at the top of `scripts/cad_probes.py`; a re-run that disagrees exits non-zero
and names the probe and the path that moved.

## These answers are NOT from the image, and all six still owe an image run

| | |
|---|---|
| Measured on | the dev machine, 2026-09-11 |
| gmsh | 4.15.2 — **the image's pin**, so gmsh-side answers are expected to carry over |
| OCCT via `cadquery-ocp-novtk` | **7.9.3.1.1** — the image is **7.8.1**, and this is the difference that matters |
| build123d | 0.11.1 |

`#14` reserved the unit-declaration re-run precisely because that behaviour "is exactly
the class of thing that moves between OCC versions". Certifying 7.8.1 from a 7.9.3 run
would repeat that error one version later, so nothing here is certified for the image.
`versions` is compared like every other recorded answer: **on the image the run will go
red on the OCC version first**, which is the correct first thing for it to say. Re-record
there with `--dump` and, where the two differ, keep both rows with the versions beside
them rather than overwriting.

| probe | owes an image run | why it might move |
|---|---|---|
| `fixture` | yes | solid and face counts come out of OCCT's STEP reader |
| `probe1_defeaturing` | yes | `BRepAlgoAPI_Defeaturing` is OCCT's, and its tolerances are absolute |
| `probe2_deflection` | yes, but least likely to move | gmsh's own option plumbing, and gmsh is pinned identically |
| `probe3_units` | **yes, and this is the one `#14` named** | OCC's STEP reader and its unit statics |
| `probe3b_unit_leak` | yes | an OCC static's lifetime, and OCC is the version that differs |
| `probe4_healshapes` | yes | `healShapes` is a thin wrapper over OCCT's `ShapeFix`/sewing |

---

## The fixtures these were measured on

`tests/data/cad/real/ldrobot_ld19_lidar.step` — a real-CAD multi-solid STEP, Apache-2.0,
three solids, 78 faces, fillets and four B-spline surfaces. `tests/data/cad/synthetic/` —
four files built and self-checked by `tests/data/cad/build_synthetic.py`, whose every
number is exact by construction. `tests/data/cad/PROVENANCE.md` has both, including what
the real fixture does **not** prove.

`importShapes` returns **3 solids and 78 surfaces** from the real fixture, which is the
multi-solid gate, asserted rather than eyeballed.

---

## Probe 1 — defeaturing is reachable, and it does the arithmetic

`BRepAlgoAPI_Defeaturing` imports from `OCP.BRepAlgoAPI` and runs. That the symbol exists
is the cheap half; §2's argument for keeping B-rep all the way to export rests on the
operation *working*, so the probe removes a fillet whose answer is known in closed form.

The fixture is `synthetic/filleted_block.step`: an 0.08 × 0.05 × 0.06 m block with one
fillet of **r = 0.005 m** on a through edge of **L = 0.06 m**, and that edge is a **90°
convex edge by construction** — the formula (1 − π/4)·r²·L is true for a right angle and
false for anything else, which is why the assertion is pinned to a fixture whose angle
was ours to choose.

| | |
|---|---|
| Cylindrical faces before → after | **1 → 0** (exactly the one removed) |
| Faces before → after | 7 → 6 |
| Volume before → after | 2.3967809724509637 × 10⁻⁴ → 2.4 × 10⁻⁴ m³ |
| Volume increase | **3.2190275490364 × 10⁻⁷ m³** |
| (1 − π/4)·r²·L | **3.219027549038276 × 10⁻⁷ m³** |
| Relative error | 6 × 10⁻¹² (the gate allows 1%) |

The recovered volume is the plain box to the last figure: defeaturing did not approximate
the fillet away, it reconstructed the corner.

**On the real fixture, as an observation only.** No analytic answer applies to those
edges and none is asserted.

| | |
|---|---|
| Fillet faces offered (tori of minor radius ≤ 1 mm, cylinders of radius ≤ 0.6 mm) | 9 |
| Succeeded | **yes** |
| Faces before → after | **78 → 73** |
| Volume change | −1.459 × 10⁻¹⁰ m³ on 4.119 × 10⁻⁵ m³, i.e. −3.5 × 10⁻⁴ % |

Nine faces offered and five removed: defeaturing took what it could and left the rest,
which is the behaviour to expect and is worth knowing before C5 relies on it.

**A scale trap, found by falling into it.** Run at the file's own scale — the fixture is
54 units across, see probe 3 — the same call had not returned after fifteen minutes. Run
at 1/1000, which is the model's true size in metres, the whole probe -- synthetic
half included -- takes under half a minute. OCCT's
defeaturing tolerances are absolute, so the size of the numbers is not a presentational
detail. Anything downstream that defeatures should convert to metres first.

---

## Probe 2 — `Mesh.StlLinearDeflection`, and the verdict C5 needs

Max chord deviation measured off the B-rep on the duct wall of
`synthetic/box_with_duct.step` — a cylinder of **r = 0.02 m exactly**, so the distance
from any meshed point to the true surface is `r − hypot(Δy, Δz)` in closed form and no
projection is involved. Taken at the midpoint of every triangle edge, which is where a
chord departs furthest from the arc it replaces. `Mesh.MeshSizeMax` fixed at **0.01 m**
throughout. gmsh's defaults on this build: `StlLinearDeflection` 0.001,
`StlLinearDeflectionRelative` 1, `StlAngularDeflection` 0.3.

### The table

| `StlLinearDeflection` | `MeshSizeFromCurvature = 0` | `MeshSizeFromCurvature = 2` |
|---|---|---|
| **1 × 10⁻⁴ m** | 8.803844165120762 × 10⁻⁴ m (636 triangles) | 8.803844165120762 × 10⁻⁴ m (636) |
| **1 × 10⁻³ m** | 8.803844165120762 × 10⁻⁴ m (636) | 8.803844165120762 × 10⁻⁴ m (636) |
| **1 × 10⁻² m** | 8.803844165120762 × 10⁻⁴ m (636) | 8.803844165120762 × 10⁻⁴ m (636) |

Not approximately equal. Bit-identical, in all six cells, across two orders of magnitude
of the knob — and identical triangle counts, so it is the same mesh six times.

### The controls, because "it did not move" and "I cannot see movement" look the same

| control | max chord deviation | triangles |
|---|---|---|
| `clmax`/4, curvature off | 6.612081954640234 × 10⁻⁵ m | 9,424 |
| `clmax` as given, curvature 20 | 4.1265553836212707 × 10⁻⁴ m | 1,484 |

Both move the deviation by more than an order of magnitude. The measurement sees
movement; there is none to see in the table above. `cad_probes.py` fails the probe if the
controls ever stop moving, so the null result cannot quietly become a broken instrument.

### Verdict

> **`Mesh.StlLinearDeflection` does not govern OCC tessellation on this path.** It
> governs STL *import*. **C5 may not use the knob.** What moves the facets in curved
> regions is `Mesh.MeshSizeFromCurvature`, which is what `cad_convert.py` already uses
> and already reports.

This closes `#6` in the negative and confirms `cad_convert.py`'s own reservation —
"deliberately not used yet ... whether it governs OCC tessellation on this path rather
than only STL import" — was the right call. The docstring in `cad_convert.py` may now
say so as fact rather than as an open question; that edit belongs to C5, which owns the
file.

*Note on `MeshSizeFromCurvature = 2`.* At r = 0.02 m, two facets per full turn asks for
a 0.063 m edge, which is coarser than the 0.01 m cap, so the cap wins and the column is
identical to curvature-off by arithmetic rather than by coincidence. The plan asked for
the cross at 2 and it is here; the curvature-20 control is what shows the knob that does
work.

---

## Probe 3 — unit declaration on gmsh 4.15.2 (`#14`'s owed re-run)

The behaviour `cad_convert.py`'s refusal text describes was measured on gmsh 4.12.1 /
OCC 7.6.3. Re-measured here on gmsh 4.15.2 / OCC 7.9.3, on the same pair of files the
test fixture builds: a 100 × 40 × 20 mm block with an r = 12 mm boss, written by OCC as
`mm.step`, and the same bytes with the length unit removed as `nounit.step`.

| | |
|---|---|
| `mm.step`, `Geometry.OCCTargetUnit = "M"` | **0.1000002 × 0.0400002 × 0.0450002** — converted to metres |
| `nounit.step`, same option | **100.0000002 × 40.0000002 × 45.0000002** — unchanged, a no-op |
| `declared_unit()` on the stripped file | reads no unit, as it must |

**Unchanged from the original measurement.** `Geometry.OCCTargetUnit` converts *from* a
declared unit and does nothing where there is none, so it cannot carry an override — the
claim `cad_convert.py`'s `NO_UNIT` text makes, still true one gmsh and two OCC minor
versions later. (The 2 × 10⁻⁷ is OCC's absolute bounding-box tolerance, not a scale
error; `cad_convert.py` measures its reported extent off mesh nodes for exactly that
reason.)

### Both refusals, reproduced verbatim

`tests/test_toolbox_cad_convert.py` in full: **29 passed, 0 failed**, with `HAS_GMSH`
true (no test skipped for a missing gmsh). Recorded so a changed count is a finding.

The two refusal texts are pinned by hash, and these are the hashes of the texts on
`main`:

| | SHA-256 |
|---|---|
| `NO_CLMAX` | `a6fac989f746fc0fab6e1302b0bc656e3afe1cee80d2274564bfb82d9e236c15` |
| `NO_UNIT` | `7ef1eb051e9f19fe48412865442a688af449cd954c7792b258fe490254998b58` |

    refused: no --clmax given, and there is no default worth guessing.
    refused: {name} declares no length unit, and none was supplied.

### The finding nobody asked for: a declaration that is simply wrong

The real fixture declares **metres** — `SI_UNIT($,.METRE.)` — and is **53.95 × 46.84 ×
31.35** of them across. A lidar module 54 m wide does not exist; the numbers are
millimetres and the header is wrong by a factor of 1000.

`cad_convert.py` converts it without ceremony, and correctly so by its own rules: the
refusal fires on a **missing** declaration, and a declaration that is present and false
reads exactly like a good one. Everything the `NO_UNIT` text says about a factor of 1000
producing "a perfectly plausible mesh, a converged solve, and completely wrong forces"
applies here, and the refusal cannot catch it.

The cheap defence is the one `cad_convert.py` already half has: it reports the extent it
converted to. **A declared-metre part whose extent is tens of metres is nearly always a
declared-millimetre part.** Whether that becomes a warning is C5's and C2's call, not
this chunk's — but it should not be discovered again by someone meshing a 54-metre lidar.

---

## Probe 3b — `Geometry.OCCTargetUnit` outlives the session that set it

Found by the probes corrupting each other: probe 3 run after probe 2 in one process
measured a factor of 1000 that probe 3 run alone does not.

Minimal reproduction, and what `probe3b_unit_leak` now runs:

1. Fresh gmsh session. Add a 100 × 40 × 20 box, `gmsh.write("before.step")`. The file
   declares millimetres and its largest coordinate is **100**. Finalize.
2. A second session which sets `Geometry.OCCTargetUnit = "M"`, imports that file, and
   does nothing else. Finalize.
3. A third session, no options set, same box, `gmsh.write("after.step")`. The file still
   declares millimetres and its largest coordinate is **100000**.

**Write scale ratio 1000.0, across two `gmsh.finalize()` calls.** The gmsh *option* is
reset by finalize; the OpenCASCADE static it writes through is not, and it is read by the
STEP **writer** as well as the reader. The second file is wrong by a thousand and says
nothing about it — the precise failure `cad_convert.py`'s unit refusal exists to prevent,
arriving from the export end where there is no refusal at all.

**What this costs downstream.** Anything that imports CAD (which sets the option) and
later writes STEP in the same process — the persistent kernel of `#2` is exactly such a
process — will write a file wrong by 1000. C5 writes STLs, which are unitless and
unaffected. The exposure is STEP or IGES *output*, and the kernel's long life is what
makes a leak that would be harmless in a one-shot script into a standing hazard. Anyone
writing STEP after an import should set `Geometry.OCCTargetUnit` explicitly for the
write, or write from a fresh process, and check the number in the file.

`scripts/cad_probes.py` runs **one process per probe** for this reason. `--in-process`
reproduces the contamination if anyone wants to see it.

---

## Probe 4 — `healShapes` on a real file

What it repairs on `real/ldrobot_ld19_lidar.step`. Answer: nothing, and it breaks the
file. Numbers are in file units (see probe 3).

| | faces | solids | volume | free edges | faces OCCT flags |
|---|---|---|---|---|---|
| **before** | 78 | **3** | 41193.70080197623 | **0** | **0** |
| after `healShapes()` (defaults) | 78 | **0** | 41193.72112078132 | **354** | 0 |
| after `healShapes(sewFaces=False)` | 78 | **3** | 41193.72112078132 | **0** | 0 |

gmsh's own entity counts agree: 3 volumes / 78 surfaces / 189 curves before, **0 volumes**
/ 78 surfaces / 189 curves after the defaults.

**`sewFaces=True` is the argument that does it**, and it is on by default. A parameter
sweep during development found every other flag innocent — `fixDegenerated`,
`fixSmallEdges`, `fixSmallFaces`, `makeSolids`, and `tolerance` at 1 × 10⁻⁸, 1 × 10⁻⁶,
1 × 10⁻⁴ and 1 × 10⁻³ — all lose the solids when sewing is on and all keep them when it
is off. It is not a tolerance to tune. Sewing a **multi-solid assembly** merges faces
across bodies that were never meant to be one shell, the shells stop closing, and
`makeSolids` cannot put back what sewing took apart.

The volume barely moves in either case, which is the trap: a downstream check that
compares volumes sees nothing worth reporting while every solid has become a loose bag of
faces. The two numbers above agree to six significant figures and differ by **five parts
in ten million** (4.9 x 10^-7 relative) -- far inside any tolerance a volume check would
be written with. Solid count and free-edge count are what catch it.

*Measure the shape, not the entity list.* Asked of gmsh's remaining 3D entities the
volume after the defaults is **zero**, because there are none left; that reads as an
obvious failure and is not the trap. The near-equal numbers above are the shape's volume
as written to STEP and read back, which is what a downstream check would actually see.

### What this means for the plan

§2 calls `healShapes` "the repair primitive §1 needs". On the one real multi-solid file
in the tree it is a **damage primitive** at its defaults. Two things follow, neither of
them this chunk's to implement:

- Whoever calls `healShapes` on a multi-solid import should pass **`sewFaces=False`**,
  or call it per solid, and must check the solid count across the call.
- Repair should be **measured, not assumed**: faces, solids, volume and free edges before
  and after, with the call refused or reverted if the solid count drops. C2's `closure`
  and C6's check are where that belongs.

And the honest limit on all of it: **this fixture had nothing wrong with it**. Zero faces
flagged before healing means probe 4 records what `healShapes` does to a healthy file and
says nothing about what it does to a broken one. `tests/data/cad/PROVENANCE.md` carries
that admission in as many words, and `cad_probes.py` fails if the sentence is removed.

---

## What C1 leaves owed

- **Every answer above, re-run on the image** at OCC 7.8.1. Nothing here is certified for
  it.
- A real-CAD fixture with a **genuine defect**. Six candidate files were checked and none
  had one. Until then, C5's thin-wall and degenerate-face outcomes are untested on real
  input rather than passing.
- Whether a declared unit that contradicts its own extent should warn (probe 3) — C5/C2.
- Whether `healShapes` should be called with `sewFaces=False` by default, or per solid,
  or at all (probe 4) — C2/C6.
