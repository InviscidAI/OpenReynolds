# Two open-source text-to-CAD repos, judged on what they produce

*2026-09-12. Repos read at their `main`; this desk measured at `1f5d9dc` by sweep
`core+bench12-20260912-123315-8815`.*

The question was two-sided: expand the corpus from
[Adam-CAD/CADAM](https://github.com/Adam-CAD/CADAM) and
[Pan-Chera/Multi-Agent-CAD](https://github.com/Pan-Chera/Multi-Agent-CAD), and decide whether
to keep building this desk or lift from theirs.

**A first pass of this document answered the second question by observing that neither repo
produces a mesh. That is true and it was the wrong test** — it judges their CAD layer by
something it never claimed to do, and this desk's CAD authoring layer is a large share of it:
eight of twelve corpus cases have the desk building the geometry itself, and most of the
ranked failures in both sweeps to date are authoring failures, not meshing failures. What
follows judges the authoring layer on what it produces.

**Both repos were read, neither was run.** Their numbers are theirs, quoted from their
reports with the conditions attached. This desk's numbers are from the sweep above.

## The answer, and the size of the claim

**Continue building.** That part is not close: nothing in either repo touches what happens
downstream of the STEP file, and their own outputs are a printable solid in OpenSCAD polyhedra
and in build123d STEP respectively.

**On the authoring half, the candidate to lift is a discipline rather than a component: every
geometric operation carries a post-condition, and a failed post-condition emits a named
diagnostic rather than nothing.** MAC's generated code has this and this desk's does not —
that much is code reading on both sides, and it is not a sampling claim.

**What is *not* yet established is that it matters as much as the first version of this
document said.** That version compared T12 at n=1 against MAC's P8 at n=1, on different
models, and concluded their behaviour on their geometry was better than ours. Two anecdotes.
The corpus has since been grown to 26 cases — fifteen further distinct geometries rather than
five repeats of three — and the claim stands or falls on the next sweep. §"What the outcomes
say" below is written as of one run per case and should be read that way.

## The comparison, stated properly

*Rewritten 2026-09-13, after two of its claims were withdrawn. This is the answer to "how do we
compare", and most of it is an account of why the question is barely answerable yet.*

### Six prompts overlap. One of them is a comparison.

Six corpus cases are their prompts. Testing each for like-for-like comparability:

| ours | ← theirs | what they scored | what we were scored on | a comparison? |
|---|---|---|---|---|
| T20 | P2 flange | both 10/10 | fluid domain, mesh, 6 properties | **no** — they were never asked to mesh |
| T19 | P4 shaft+keyway | both 11/11 | fluid domain, 0.05 mm gap | **no** — same |
| T21 | P5 enclosure | MAC 12/12 · skill 11/12 | fluid domain, blind-hole depth | **no** — same |
| T18 | P7 finned cylinder | both 17/17 | fluid domain, fin gaps | **no** — same |
| T12 | P8 impeller | MAC 14/15 · skill 15/15 | fluid domain, fillet decision | **no** — withdrawn; the two suites want *opposite* things from a root fillet |
| T26 | P9 staircase | MAC corrected it · skill shipped the defect | did it notice the spec does not close | **yes** — the question is task-independent |

**Five of the six are not comparisons, and the reason is structural rather than incidental:
this corpus's criterion begins where theirs ends.** They score feature presence on a printable
solid. We score a clean `checkMesh` plus every named property measured and printed. "We passed
T20 and so did they" describes two different tasks that happen to share a prompt. There is no
quantity both sides scored.

T26 survives because its question does not depend on what the geometry is *for*: the tread
meets the column along a tangent line, so the spec does not close, and any system must either
notice or not.

### The one comparison, and we lose it

| | on P9 / T26 |
|---|---|
| **MAC** | found the tangency unprompted, kept a safe overlap, logged it as a defensive correction |
| **`cad` skill** | shipped the disconnected version; their report prints both renders side by side |
| **this desk** | `fluid = shaft - column - tread_solid`, meshed it, printed `shaft ID 2.400 (asked 2.400)`, never mentioned the tangency. The brief's central properties — the gap *with its sign*, the count of treads in contact — appear nowhere |

And the harness saw it. `union_closure` recorded **259 free edges** on the welded surface —
almost certainly the tangency, as unwelded coincident edges — and the run still scored
`passed: true`, because no probe is a gate and `checkMesh` validates the volume mesh rather than
the surface it came from. **The defect was measured, written into the record, and passed.**

One run, and a different model on each side (`claude-opus-5` against `qwen3.7-max`). So: one
observation, and it goes against us.

### What is measurable, and is not in our favour

| | per case | model calls per case |
|---|---|---|
| MAC | ~$0.14 | **5** |
| `cad` skill | ~$1.76 | 131 |
| **this desk** | **$1.01** | **20.3** |

Across 35 runs and $26.21 over 26 distinct cases. The dollar figures are not clean — different
models — but **calls per case is an architectural quantity and it is the one MAC's entire thesis
is about**: four agents passing compact structured state versus one agent working a growing
conversation. We are at 4× MAC's call count. We are cheaper than the skill, and the skill is the
baseline MAC was built to beat.

### What is established by reading the code, on both sides

Every geometric operation in MAC's generated code carries a post-condition and emits a named
diagnostic when it fails — `MISSED_CUT`, `FILLET_DEGRADED`, `FILLET_PARTIAL`. This desk has no
equivalent, and the sweep of 2026-09-12 demonstrated the same class of gap in a place that
matters more: **nothing in the desk's path asserts that an exported surface closes.** `grep` for
`open_edges`, `free_edges`, `is_closed`, `watertight`, `manifold`, `closure` over
`openreynolds/buildup/core.py` and `record.py` returns zero for every term. Four cases exported
non-closed surfaces and all four passed.

### So, plainly

- **On the half we own — solid → fluid domain → mesh — there is no comparison to make.** Neither
  repo does it; `OpenFOAM`, `snappyHexMesh`, `blockMesh` and `checkMesh` appear in neither
  repository. That is not a win, it is an absence of a competitor.
- **On the half we share, we have one valid observation and we lost it.** We cannot currently
  claim to be better or worse at CAD authoring than either system.
- **On architecture we are measurably on the wrong side**, by the metric its authors chose.
- **The one actionable difference is the post-condition discipline**, and it is now supported by
  two independent instances (T26's tangency, the closure gap) rather than the one that was
  withdrawn.

### The shared half, measured — by reusing geometry the runs already produced

*Added 2026-09-13, in answer to "doesn't producing the mesh already require producing the
geometry first?" It does, and the answer costs nothing: the part solids are in each run's
committed `build.py`.*

**Where the fluid domain is a cavity in or around a part, the part must be built and is
therefore gradeable.** `T18: solid = barrel + fins + head`. `T19: shaft_solid = shaft_cyl -
keybox`. `T26: column`, `tread_solid`. Their feature lists are itemised one bullet per feature,
so our geometry can be graded on their own checklist with no model spend at all.

**The limit is not the geometry, it is that a fluid domain only needs the wetted boundary.**
T20 built `fluid = seg1 + seg2 + seg3` and no flange — water in a bore is a cylinder, so P2's
bolt holes and fillets do not exist to grade. T21 built the internal cavity and never the
enclosure shell, so "uniform wall thickness" has no referent. Both are correct CFD
simplification and both delete features their checklist scores. That truncation is systematic,
not incidental.

Grading the four gradeable overlaps on the features **both** briefs ask for:

| ours ← theirs | shared features | us | their full list |
|---|---|---|---|
| T18 ← P7 finned cylinder | 7 | **7/7** | 17 |
| T19 ← P4 shaft + keyway | 6 | **4/6** | 11 |
| T21 ← P5 enclosure | 5 | **5/5** | 12 |
| T26 ← P9 staircase | 8 | **7/8** | 16 |
| | **26** | **23/26** | 57 |

On those same 26 features, from their published per-item results:

| | on the 26 shared features |
|---|---|
| **MAC** | **26/26** — including P9 item 8, which it corrected unprompted |
| **`cad` skill** | **25/26** — failed item 8, shipped the disconnected staircase |
| **this desk** | **23/26** |

**This is the first like-for-like number in this document, and the ranking is MAC > skill > us.**
The denominator is 26 of their 141, or 18% of their checklist, and it is one run each side at
different models. It is small and it is real.

#### What our three failures are, and they are not all equal

- **T26, the tangency.** A real failure and the one that discriminates: their own feature list
  states it as *"tread inner ends approach the column (this feature is physically infeasible and
  would cause the model to collapse and fracture)"* — the infeasibility **is** the graded item.
  MAC caught it, the skill did not, we did not.
- **T19, the two missing shaft steps.** Our brief named a 50 mm and a 30 mm step; the desk built
  only the 40 mm sealed length. Those steps lie outside the seal and have **no wetted surface in
  the domain we asked for**, so the desk was right to skip them and the brief was over-specified.
  Counting them against the desk is the same error as T9's unpassable brief. Ruling them brief
  noise gives **25/26 — level with the skill, one behind MAC, on the single feature that
  separates the three systems.**

#### The finding that makes this worth repeating

**All three failures are invisible to this corpus's own criterion.** T19 ended `done` with a
clean `checkMesh` and nothing in its property list asked about the steps. T26 ended `done`, and
its brief's tangency property went ungraded because property grading is judgement and no run
fails on it. **Grading on their checklist found defects ours structurally cannot see**, for free,
from records already on disk.

That argues for adopting the feature-list form as a complement to `checkMesh` and the property
list — not as a replacement. `checkMesh` cannot see a leaky surface either (§ the closure gap),
and a binary topological checklist is exactly the shape of check that can.

### What a full comparison would still take

The shared half is unmeasured because this corpus never scores it separately. The fix is cheap
and specific: **run P1–P10 against this desk and grade the output on their 141-feature criteria**
— their own checklist, their own binary pass/fail, on our geometry. Ten runs, roughly $10, and it
yields the first like-for-like number anyone has on the shared task. Removing the model variable
too would mean running MAC here on Opus 5 through its OpenAI-compatible endpoint; that is a
separate piece of work and its dependency conflicts are documented in its own README.

Until one of those is done, the honest summary is: **one comparison, lost; no rate; and a cost
profile that resembles the baseline they beat.**

## What the outcomes say

### The one case where the comparison is direct

T12 was put in the corpus so the sweep would have a point of external comparison. It is the
centrifugal impeller that is **P8** in MAC's benchmark set, and P8 item 13 — the blade root
fillets — is MAC's single failure across 141 features.

On the same geometry:

| | MAC on P8 | this desk on T12 |
|---|---|---|
| root fillets | attempted | **never attempted** |
| what OCCT did | refused the fillet on the geometry | not invoked |
| response | 4 radii tried, then per-`GeomType` edge grouping, each with its own guard | – |
| what was reported | `FILLET_FAILED` with selected-edge diagnostics and a root-cause note | **the word `fillet` appears nowhere in 21 cells** |
| scored | 14/15, the set's only failure | `passed: true` |

The brief asks for 2 mm root fillets and explicitly allows removing them — *"Remove the root
fillets if they will not survive the cell size, and say so if you do."* Neither happened, and
nothing said so. Three of the run's closing numbers are typed constants inside strings
labelled as measurements, including `print("hub radius from mesh : 0.01200 m (asked
0.012)")`, which contains no computation. The record scores it a pass.

**Withdrawn, 2026-09-13 — this comparison does not hold.** The paragraph here previously
concluded that on one run each, on their geometry, their answer was better than ours. Two
things are wrong with it.

The first is n=1 on both sides, which the amendment in the sweep report already covers. The
second is worse: **the two suites want opposite things from a root fillet.** For a printed
impeller a fillet is a strength feature and building it is the pass. For a fluid domain a small
fillet is a candidate for *removal* — defeaturing is standard CFD preprocessing and T5's brief
asks for it by name. So MAC building a fillet and this desk not building one are not better and
worse answers to one question; they are answers to two different questions, and T12 is a much
weaker comparison point than it was written to be.

What survives is only about reporting, and it is real: the brief permits removal **if
declared**, and `fillet` appears in neither `build.py` nor `cells.log`, so no decision was
recorded either way. An omitted feature and a defeatured one are the same geometry and
different engineering. That is a finding about silence, not about fillets.

### Why, and it is not the model

Their generated code cannot silently skip a required operation, because the operations are
wrapped. Every cut carries a volume post-condition:

```python
_before = body.volume
result = body - tool
if abs(result.volume - _before) < 0.001:
    _MISSED_CUTS.append(f'MISSED_CUT: {label} — tool had NO effect on body. Position is WRONG.')
```

A cut that changed nothing is diagnosed as a *position error*, by name, into a channel the
repair stage reads. `_safe_fillet` is the same shape with a ladder: four descending radii,
then edges grouped by `GeomType` so a mixed Circle+Line selection cannot fail wholesale, and
`FILLET_DEGRADED` / `FILLET_PARTIAL` / `FILLET_FAILED` distinguished in the output. The
failure text carries hard-won OCCT knowledge — that a circle edge split into arc segments by
preceding boolean unions cannot be filleted at degenerate junctions, which is precisely the
diagnosis of their own P8 failure.

Their edge selectors are re-derived per run from `e.length()` rather than hardcoded
coordinates, with the reason stated: short junction edges provoke `ChFi3d_Builder: only 2
faces`, and bbox-coordinate filters "break when part dimensions change". That is §1's
"geometric selectors re-derived on each run" as working code rather than as a capability
sentence.

### What this desk's own outcomes say about the same gap

The sweep's four new cases, on new geometry, all show one pattern: **areas and volumes are
measured well; a length at a station is not measured at all.**

- T11 measured its honeycomb wall area two independent ways —
  `0.057042966976484574 m²` off the VTK boundary against `0.05704296649601038` summed over the
  CAD faces, agreeing to nine significant figures. That is better than anything in either
  benchmark suite requires.
- And in the same run, cell width across the flats is `math.sqrt(3)*R_cell` — the input.
- T9 printed four patch areas against exact values with signed errors, then reported
  `measured clearance = 1.000 mm` as `(R_RING_I-R_SPIN_O)*1000`.
- T12 measured its outlet radius over 5,260 real face centres, and typed its hub radius in.

Four for four, **at one run each** — and the breadth sweep that followed withdrew it.
`core+bench26-20260912-133719-4bbd` §4 records nine cases asking for a length at a station and
the desk measuring them: T19's 0.05 mm clearance as the nearest approach between the bore and
shaft point clouds, T13's helical flank gap as a min/max over a distance array (`0.2000 /
0.2521 mm` — the spread is the proof), T21's blind-hole depth off the hole patch, T24's swirl
angle by dot product *with its sense*. T20, the 42 mm cylinder with no difficulty to blame,
measured properly too.

So the four-case pattern described **the desk under difficulty**, not the desk. T12's typed
literals and T9's restated clearance stand as what they were; they are no longer evidence of a
habit. Two properties are still taken from the inputs — T16's annulus heights off its own
radius functions and T22's passage count off the solid rather than the mesh — and that is the
honest size of what remains. A post-condition on the operation would not
catch every instance of that, but the class it does catch — an operation that was required,
ran, and did nothing — is ranks 1, 2 and 6 of this sweep.

### The outcome finding that needs neither sweep

Across the **282 feature-judgements** two independent systems published on the same ten
prompts, every single failure was a fillet: three for
[earthtojake/text-to-cad](https://github.com/earthtojake/text-to-cad)'s `cad` skill (scope —
over- and under-applying), one for MAC (an OCCT conflict). Nothing else failed once. Two
unrelated architectures on the same model, and the only operation that breaks is the one whose
input is an edge set on geometry that prior booleans have already fragmented.

## CADAM — the prompts, and nothing else

| | |
|---|---|
| license | **GPL-3.0**, against this repo's MIT |
| what it is | React/Vite/Supabase web app; OpenSCAD compiled to WebAssembly |
| kernel | OpenSCAD — CSG evaluated to polyhedra |
| published outcome | 13 models, `.scad` + renders, parametric control counts; **self-scored showcase, no pass/fail** |

Judged on output rather than on fit: what it emits is triangles, not a BREP. There is no
shape to heal, no face to select, and no reliable topology to subtract — so fluid-domain
extraction, planar capping, geometric patch selectors and `healShapes` are unavailable in
principle rather than merely unimplemented. That is a statement about its output format, not
about OpenFOAM.

Its own published solutions carry the sharper argument. The blisk benchmark ends its aerofoil
with `offset(r=0.6)`, commented *"to guarantee manifold trailing edges and robust
printability"* — thickening the trailing edge by 1.2 mm so a slicer accepts it. For a printed
part, correct and invisible. For a blade in a flow, a different blade. T10 exists because of
that line, and T10 was the sweep's most expensive failure, which suggests the line was load-
bearing for them too.

GPL-3.0 is a second and independent bar: vendoring relicenses this product.

**Taken: the prompts, as phrasing.** Three became T10, T11 and T12.

## MAC — what is worth lifting, and what is not

| | |
|---|---|
| license | MIT — compatible |
| kernel | build123d / OCCT — the same family as this desk |
| architecture | 4 stages passing compact structured state (`CADBrief`, `ArchitectPlan`), not shared conversation history |
| published outcome | 140/141 features, ¥9.67, 50 API calls, against 138/141 at ¥125.69 and 1,307 calls for the `cad` skill — same prompts, same model, architecture the only variable |

### Lift

**The post-condition discipline.** Not the code — the pattern. Every geometric operation
verifies that it did something, and reports a named failure into a structured channel when it
did not. `_safe_cut`, `_safe_fillet` and `_safe_chamfer` are one page each and the idea is
portable without importing build123d, which matters because Precondition 3 forbids anything
under `openreynolds/` from importing build123d, OCP or gmsh. What transfers is that the
*desk's own generated cell* carries the guard, which is where it has to be.

Worth reading before touching the probes §4.1 found defective — four of six now:

- `compare_to_spec(analysis, spec)` — a machine form of the property grading that is currently
  prose judgement. It cannot answer the hard half (restated-versus-measured is exactly what a
  spec comparison cannot see), but the shape is worth knowing.
- `_estimate_mesh_resolution` + `_deviation_is_noise` — a tolerance band scaled to mesh
  resolution. `958647d` here is *"The noise band is a share of the case, not a count of
  cells"*: same problem, reached independently.
- `_check_connectivity`'s Union-Find fallback → `union_closure`.
- `_estimate_min_feature` → `scale`, confirmed broken on seven cases across two sweeps.

### Do not lift

- **The auto-injected shim.** It monkeypatches `extrude`, `revolve`, `fillet`, `chamfer`,
  `mirror`, `make_face`, `Hole.__init__` and the three builders to **silently strip
  unrecognised keyword arguments**. A `taper=` the API does not have is dropped and the
  extrude succeeds without it. That converts a loud `TypeError` into a quietly wrong shape,
  which is the exact class `docs/cad-silent-failures.md` exists to catalogue. Loud failures
  are recoverable; this desk's T2 lost a cell to a `TypeError` and the diagnosable trace is
  why anyone knows.
- **`_validate_solid`'s thresholds**, which are hardcoded millimetres: reject below 0.01,
  warn above 1000 as *"features shouldn't exceed 1m"*, volume floor 0.001 mm³. Every one is
  wrong by 1000× for a desk working in metres, a 2 m tunnel trips the ceiling, and T6 is the
  case about exactly this assumption.
- **`_safe_union`'s overlap test**, which is a *bounding box* intersection. Two solids can
  have overlapping boxes and not touch, and a small overlap only warns. The union failure
  this desk actually has — `boolean_union_of_swept_bands_loses_material` — would pass it. The
  test that would catch it is volume additivity, which they apply to cuts and not to unions.
- **Engine B, `legacy_refs/check_mesh.py`.** 84 KB of `trimesh` measuring wall thickness,
  holes, ribs, overhangs, undercuts, print orientation, symmetry and material strength. All
  manufacturability; none of it mesh quality. It would add a dependency in a probe path to
  measure things no case asks about.
- **`cadpy`.** Not theirs — vendored from earthtojake/text-to-cad under MIT, the baseline they
  benchmark against, and its own README says so. Its content is CAD *viewer* plumbing: GLB,
  selector manifests, bbox facts, STEP artifacts, source hashing.
- **The pipeline itself.** Its verifier answers a different question, and its criterion —
  feature present, body single, watertight — is the softer half of this corpus's. **99.3% and
  9/12 are not comparable numbers** and should never be printed side by side.

## Where this leaves the two halves

**The corpus.** Four cases added, and they half worked. They produced three of the sweep's
top four findings and the first mechanically-greppable evidence of the fabricated-measurement
pattern. They also failed at the thing that partly justified them: none wrote
`constant/triSurface`, so `coverage` and `scale` still have never returned a verdict, and
T9's brief asserts a topology its own geometry cannot have and needs rewriting. Both are in
the report's §6.

**The desk.** Keep building. Nothing in either repo replaces what happens downstream of the
STEP file, and on the authoring half the gap is one specific discipline rather than an
architecture — which is the cheapest kind of gap to close, and the report's ranks 1, 2 and 6
are what closing it would buy back.
