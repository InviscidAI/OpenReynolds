# Sweep `core+declare_gate-20260913-124524-aa21` — the failure report

*This is the `/cad-sweep` report: what failed, ranked by what it cost. It is a decision
input, not a decision. It names no tools and proposes no remedies.*

*The `/cad-addition` report on the same sweep sits beside it in `report.md`, arguing the case
for the addition under test. This one reads the corpus as a corpus, and it corrects that
report in two places — §3.1 and §6.*

## 1. The sweep

| | |
|---|---|
| label | `core+declare_gate` |
| model / effort | `claude-opus-5` at `medium` |
| core sha | `8ba92f9`, clean tree |
| cases | T1–T26, one run each |
| baseline | `core+bench26-20260912-133719-4bbd` (`claude-opus-5` at `medium`, core `af131f1`) |
| ended `done` | 20/26 |
| total cells | 518 |
| total spend | $24.29 |

Paired against the baseline on the 15 cases that pair; 11 excluded as unpaired.

```
1 case used fewer cells, 8 more, the rest within 30% of their own cell count.
sign test over the cases that moved: p = 0.039
```

**Read the sign test, not any row.** The paired figure is **13/15 → 11/15**; T19 and T22 flipped.
Throughout this report "cells" means notebook cells — steps — not mesh cells.

### 1.1 How the six failures ended

| case | stopped | expects | mesh | checkMesh | steps | $ |
|---|---|---|---|---|---|---|
| T5 | `time` | done | **no** | – | 19 | 0.67 |
| T12 | `steps` | done | yes | ok | 27 | 1.30 |
| T15 | `time` | done | yes | **failed 1 check** | 23 | 1.61 |
| T19 | `steps` | done | yes | ok | 28 | 1.46 |
| T22 | `steps` | done | yes | ok | 28 | 1.00 |
| T25 | `steps` | **refused** | yes | ok | 28 | 0.99 |

Four of six ran out of budget with a mesh on disk that `checkMesh` accepted. **Four never
reached a declare at all** — T5, T15, T19, T25. The gate never saw them.

## 2. Contamination

**0 of 26.** `contamination.hits` is empty on every record. The numbers below measure what
they claim to. This is the first sweep carrying gate text in every conversation, so a zero
here also says the channel added to the desk carried nothing of ours into it.

## 3. Failures, ranked by cost

Cost is cases hit × cells burned × spend, because cost is what an addition buys back.

### 3.1 `exported_surface_duplicated_in_trisurface` — 2 cases, ~10 cells, $1.98 — **and it cost a pass**

T22 and T25 each left a stale whole-surface STL in `constant/triSurface` beside the live
per-patch files, from a meshing route they abandoned. The probes weld every file in the
directory, so they read the dead file too.

- **T22**: `brakeDisc.stl` is a byte-for-byte duplicate of the five patch STLs concatenated —
  verified by exact vertex-tuple comparison, 20,736/20,736 identical, 0 unique to either side.
  Every triangle present twice leaves `union_closure` at 0 free edges (each edge's reverse is
  always present) while `normals` fires on all **62,208** undirected edges, because each is now
  walked twice *in the same direction* by its own duplicate. `system/snappyHexMeshDict`
  references only the five patch files. **The delivered mesh was built without the duplicate.**
- **T25**: `all.stl`, an OCC triangulation written as a closure diagnostic at cell 11, was
  superseded ten steps later by a finer gmsh re-export and never deleted. Two non-conforming
  triangulations of the same surface, welded together, manufacture **85,902** crossing pairs
  and inflate the 591 open-edge and 314 flipped-normal counts.

**Neither extreme reading is a defect in a delivered surface.** `checkMesh` reports both
meshes closed and OK.

**This is the finding that cost the sweep a pass, and the mechanism is not the one `report.md`
gives.** That report says *"T22 is plausibly the gate. Warned once, went after it, and spent
the rest of its budget."* The transcript does not support it. T22 has exactly one
`declare_complete`, and turn/step arithmetic (`steps == turn − 2`, no gap through turn 29)
places it at **turn 30, the run's last action**. The budget went to ordinary geometry
iteration first — a `NameError`, an OCC per-face tessellation the desk's own check caught at
2,432 free edges of 5,080, a hand-built Coons-patch replacement it verified clean
(`free 0 nonmanifold 0 same-dir-dup 0`), an abandoned cfMesh attempt, then
blockMesh+snappyHexMesh with two live bugs fixed en route. It declared once, at the end, and
the gate bounced it on the dead file. `turns >= 30` fired immediately after. **There is no
turn 31: the desk never had a chance to see or answer the warning.**

So the gate did cause T22's regression — by rejecting a correct, complete run on a false
reading at the last possible turn. That is a sharper problem than the one the report claims,
and `openreynolds/buildup/probes.py:58` already documents the mechanism against T4:
*"T4's 132 flipped edges were a directory holding one surface twice."* Known, and unfixed.

### 3.2 Properties named and never measured — 24 cases, the whole corpus, ~$24

**60 of 136 named properties were measured and printed: 44%.** Five cases measured **zero** —
T5, T13, T15, T17, T19 — and two of those five (T13, T17) passed.

This is the largest failure in the sweep by cost, because every case pays it. It is not one
id; it is the corpus-level shape the vets kept finding:

- **T13** declared complete on *"Mesh is clean and every stated dimension measures right."* It
  had verified its **construction inputs** — OD, depth, pitch, gap, height — not one of the
  five derived properties the case named. 0 of 5.
- **T17** printed none of its six and passed: no runner count on the mesh, no per-port spread,
  no centreline length, no plenum/runner volume split.
- **T18**, the case the addition exists for, measured about 1.5 of 6 — fin gap, fin count and
  fin thickness all read off **pre-export CAD B-rep faces**, not the meshed fin patch; pitch
  never printed; *"≈3 cells across each fin gap"* is arithmetic (6 mm / 1.875 mm), not a count.
- **A recurring sub-shape, five cases**: the desk slices its own `checkMesh` output past the
  answer. T26's final print uses `['Checking patch topology':'Checking faceZone']`, omitting
  the `Checking topology` block entirely, so `Number of regions:` never appears for the
  delivered mesh. T17's slice starts at `"Checking patch topology"` — one line after it.
  T21's grep pattern never matches it. T20 prints it for the coarse mesh only. The number the
  property asks for was on disk, in output the desk captured from, and was filtered out.

Existing ids that fit pieces of this: `named_requirement_dropped_without_mention`,
`duct_area_schedule_measured_off_generating_functions`,
`printed_number_is_a_literal_not_a_measurement`.

### 3.3 Silent failures on green runs — 3 cases, ~$3.0

Runs that ended `done`, passed `checkMesh`, and are wrong in ways no instrument saw. This is
the section the mesh vet exists for, and it found three.

- **T10** — *the case's own `false_pass`, mechanically confirmed.* The NACA thickness law
  closes to exactly zero at x=1 (`0.2969−0.1260−0.3516+0.2843−0.1036 = 0.0000`), so the CAD
  trailing edge is a zero-thickness knife edge by construction; the blade mesh size is set
  explicitly to `SizeMin = 0.0005` (0.5 mm). A 0.5 mm cell cannot resolve a 0 mm feature. The
  run never printed the as-meshed TE thickness, the local cell size, or how many cells span it
  — `grep -in "trailing|TE thick|resolve"` finds nothing but comment text. Every other
  geometric property is re-evaluated from the same constants that generated the geometry.
  `checkMesh`: `Mesh OK`.
- **T26** — *a degenerate spec resolved silently.* Shaft radius 1.200 − tread radial length
  1.100 = 0.100 m, **exactly the column radius**: every tread's inner edge is in zero-clearance
  tangency, a boolean with no single correct answer. `build.py` resolves it by welding
  (*"tread solid runs from the axis (x=0) outward so it welds into the column"*), yielding
  `Number of regions: 1 (OK)` — 20 supported treads, every dimension exactly as requested. The
  three properties written to expose this (signed gap/overlap, per-tread contact count, region
  count vs expected) are all absent, and the words *gap*, *overlap*, *contact*, *headroom*
  never appear in the replies.
- **T16** — the area schedule is computed by `π(R_out² − R_core²)` from the **generating
  radii** at 2 stations, not measured off the mesh at the 5 the case asks for; annulus height
  absent at every station; the area ratio never printed.
  (`duct_area_schedule_measured_off_generating_functions`, on a passed run.)

### 3.4 `warning_chased_until_the_step_budget_ran_out` — 1 case, 9 cells, $1.30

**T12 only.** Shown 2,177 free edges and 3,425 crossing pairs, it spent cells 20–28 on real
diagnosis: STL byte inspection, per-face and whole-solid `BRepMesh` re-tessellation, pyvista
merge/clean/feature-edge extraction twice, z/r histogramming of the free-edge points, an OCCT
`.clean()` and remesh, and a raw `TopExp` edge→face adjacency walk. It settled on a cause —
*"Binary STL in float32 — the quantization stops shared vertices from welding"* — and ran out
at 27 of 30 without a fix or a waiver.

`report.md` groups T22 with T12 here. §3.1 shows that is wrong: T12 chased a warning for nine
cells; T22 was shown one on its last turn and never acted at all. **T12 is the only case in
the sweep that matches this id.**

**Corrected 2026-09-14: the warning was right and the desk's diagnosis of it was wrong.**
This report first repeated the float32 explanation as if it were established. It does not
survive a tolerance sweep: the open-edge count is flat at **2,177 from a 1 nm weld to a 10 µm
weld**, moves only at 50 µm — where genuinely separate geometry is being merged rather than a
precision gap closed — and still leaves 1,225 open edges at 100 µm. The coordinates on disk are
not even bit-identical to their own float32 cast. **T12's exported union has a real 2,177-edge
topological gap, no tolerance closes it, and nine cells went to a cause that was not there.**

So the warned cases split cleanly: **T12 was warned correctly about a real defect; T22 and T25
were warned about leftover-file artefacts** (§3.1).

### 3.5 `defeature_call_cut_without_printing_anything` — 1 case, 19 cells, $0.67

**T5.** `BRepAlgoAPI_Defeaturing().Build()` ran ~643 s across three polled cells and
**succeeded**, but the same cell died on `df.HasErrors()` — a method that object does not have
— before `body_df = Solid(df.Shape())` could capture the result. The desk diagnosed it
correctly and restarted the whole defeaturing from scratch; that cell was still running,
unprinted, when the clock ran out. **883 s of 1,150 s — 77% of the run — inside or waiting on
one operation that never returned a usable result.** No mesh, 0 of 6 properties.

### 3.6 `near_contact_solids_produce_highly_skew_faces` — 1 case, 2 cells, $1.61

**T15**, the only run to fail a `checkMesh` check. The sole failure is skewness
(`***Max skewness = 8.52667, 28 highly skew faces`); topology, region count, openness, aspect
ratio, non-orthogonality and volumes all pass. The desk's own `BRepExtrema` put the
pinion–wheel flank gap at **0.0960 mm** against a 0.150 mm target and meshed it at background
refinement with no local refinement region. The 28 skew faces sit exactly there.

### 3.7 `dimensionless_request_built_instead_of_refused` — 1 case, 28 cells, $0.99

**The defect is not that it guessed. It is that it never said so.**

The request is *"A complete high-bypass turbofan: a front fan you can see into, a bypass cowl,
an internal core with compressor and turbine stages, outlet guide vanes, and an exhaust plug.
Mesh the air through it."* There is no length, no diameter, no stage count — nothing. Turn 1,
the first cell, before any question:

```
# ---- Turbofan flow-path profile, metres, axis = +X ----
NAC = [(0.00,1.05),(0.30,1.02),(0.60,1.00),(1.60,1.00),(2.20,0.94),(2.60,0.85)]
HUB = [(3.00,0.25),(2.60,0.40), ... ,(0.15,0.00),(0.00,0.00)]
```

A 2.10 m fan, 3.00 m long. The case's `false_pass` names this almost verbatim.

An autonomous desk choosing plausible dimensions rather than stopping to ask is defensible —
arguably correct, for a request that cannot be answered otherwise. **What is not defensible is
choosing them silently.** Across all 2,386 characters of reply text this run produced, there
is not one word marking those numbers as its own: zero occurrences of *assume*, *chose*,
*typical*, *representative*, *plausible*, *unspecified* or *clarify*. The dimensions are
written as though they were given. The result is a correct mesh of a machine nobody specified,
handed back with nothing to audit it against.

**T6 shows the same desk doing it right**, in the same sweep. Given a STEP file whose length
unit is untied, it wrote *"The facts about this file, all measured from it"*, named the defect
exactly (`#426 = ( LENGTH_UNIT() NAMED_UNIT(*) )` — no `SI_UNIT`, no `CONVERSION_BASED_UNIT`),
said what it would need confirmed, and built nothing. That is disclosure, and the corpus scores
it as a pass.

One caveat on the record: T25 was cut off at step 28 mid-inspection and never reached a closing
summary, so what it would finally have claimed is not knowable. But turn 1 — where the numbers
were invented — is where the disclosure belonged, and it is not there.

So `expects: refused` may be the wrong scoring for this case. **Refusal and silent invention are
not the only two options, and the corpus currently cannot express the third**: build it, say
plainly that every dimension is yours, and hand back something auditable.

#### 3.7.1 The repo already has the convention for this

Nothing new needs inventing. `openreynolds/cad/check.py` states the rule in its own header —
*"a sealed cavity driven by sources on cell zones legitimately has one patch, so one patch is
a **warning with its assumption stated**, not a refusal"* — and implements it as a `Finding`
whose three fields separate exactly what T25 conflated:

```python
class Finding(NamedTuple):
    check: str
    status: str      # "ok" | "warn" | "fail" | "skipped"
    measured: str    # evidence, and should survive being quoted on its own
    meaning: str = ""  # the interpretation, and is allowed to be wrong
    repair: str = ""   # a suggestion, and is allowed to be ignored
```

The live exemplar is `_patch_findings`: a one-patch mesh is legitimate, so it emits
`status="warn"` carrying the count as `measured` and the assumption as `meaning`, rather than
failing a correct result or passing a suspect one silently.

**That is the shape a dimensionless request wants.** T25's 2.10 m fan should have produced a
`warn` whose `measured` is the invented profile and whose `meaning` says the request carried no
dimension and these are the desk's own. The desk would still build, still finish, and the
number would arrive attached to the fact that nobody asked for it. The convention exists, is
tested, and is used elsewhere in this repo; the buildup corpus simply scores this case
`refused` instead of reaching for it.

### 3.8 `xpass_conflates_a_misread_check_with_a_repaired_cause` — 2 cases, 0 cells

T21 pre-waived `normals` with an argument about outward-vs-inward normal direction — a
property the probe does not measure; `flipped_edges` was already 0 before the waiver was
written. T18 waived `location_in_mesh` correctly, fixed the surface, and the same waiver read
back `xpass` because the point was now inside the welded union. The state cannot separate *the
desk misread the check* from *the desk removed the reason it fired*.

### 3.9 API-surface failures in a desk given no tools — 15 cases, ~$14

**This is the largest recoverable failure in the sweep, and nothing in the gate can see it.**

`report.md` leaves the attribution open: *"this sweep changed four things and can say which
one did anything only where the gate left a trace."* The vets closed it. Six runs were traced
cell by cell against baseline and **none of the increase is gate-driven**:

| case | Δ | where it actually went |
|---|---|---|
| T15 | +10 | failed `loft` (`RuntimeError`), 2 rebuild cells, 6 verification cells, and `import trimesh_check` — a guard import of a module that does not exist, after a successful export |
| T19 | +8 | 6 cells re-confirming **by picture** a cell count already fixed by its own `setTransfiniteCurve`; 2 of those lost to `pv.read('seal.msh')` (`KeyError: '.msh'`) and `matplotlib.use("Agg")` silently discarding a figure |
| T17 | +6 | 4 cells on a malformed `blockMeshDict` it wrote itself — one extra `(` before `background` — plus a deliberate refinement bump |
| T20 | +4 | binary-STL export bug, a `UnicodeDecodeError` reading it as text, a regex that *"swallowed the first `facet` keyword"*, a voluntary refinement; minus the bolt-hole check the baseline did |
| T23 | +11 | OCP API misuse (`.X` vs `.X()`, twice), STL tolerance tuning, and 13 cells of the desk's own verification — which found 2 degenerate zero-area triangles at the sphere poles and re-meshed |
| T25 | +12 | three triangulation strategies chased in sequence; meshing only starts at step 20 |

#### 3.9.1 The corpus-wide shape

Those six are not special. Across all 26 runs, **38 of 523 cells exited non-zero — 7% — in
19 of 26 runs**. Grouping the 36 that raised a named exception by cause:

| cause | cells | cases | which |
|---|---|---|---|
| **API surface** | **27** | **15** | T1 T4 T5 T6 T10 T11 T12 T15 T16 T17 T18 T19 T22 T23 T24 |
| file format | 4 | 4 | T2 T11 T19 T20 |
| geometry kernel | 4 | 4 | T4 T10 T13 T15 |
| syntax | 1 | 1 | T3 |

**Three quarters of every failed cell in the sweep is the desk calling something that does not
exist**, in 15 of 26 cases. Not a modelling mistake, not a geometry mistake — a method name.

- `AttributeError` ×11 — `'numpy.ndarray' object has no attribute 'ptp'` (T1),
  `'Face' object has no attribute 'geometry_type'` (T5),
  `'BRepAlgoAPI_Defeaturing' object has no attribute 'HasErrors'` (T5),
  `'OpenFOAMReader' ... 'set_all_patch_arrays_on'` (T4).
- `TypeError` ×10 — **`'bool' object is not callable`, in six separate cases** (T5, T6, T10,
  T11, T23, T24). One pattern: a property accessed as a method, `.X()` where `.X` was meant.
- `NameError` ×4 — `name 'Shape' is not defined`, `name 'Scale' is not defined`: an import
  the desk assumed.
- `ModuleNotFoundError` ×1 — `No module named 'trimesh_check'`, invented outright.

#### 3.9.2 The failed cell is not the cost; the recovery is

All 38 failed cells together burn **190 s of wall clock** — under 1% of the sweep. The expense
is entirely in what follows: T20 spent four cells re-exporting after one `UnicodeDecodeError`;
T5 restarted a 643-second defeaturing from scratch and died with it still running; T17 spent
four cells on one stray parenthesis. **A cell that fails in 0.05 s can cost four cells and a
pass.**

This is also why T5's failure (§3.5) belongs here rather than in a category of its own:
`df.HasErrors()` is an `AttributeError` on a method that does not exist, and it discarded a
result that had already succeeded.

#### 3.9.3 Why the gate is structurally blind to all of it

T17's vet states the mechanism plainly: the gate's six checks read the **final** STL union and
the **final** `snappyHexMeshDict` seed point. They do not read the trace of failed attempts,
nor the budget spent recovering from them. **The gate cannot see a wasted cell** — by
construction, not by oversight. A desk that reaches a correct surface after fifteen wrong
turns and a desk that reaches it in three produce identical readings.

So `p = 0.039` belongs to this, not to the addition under test, which is exactly consistent
with it appearing on cases the gate never spoke to.

#### 3.9.4 These failures are the experiment working, not a defect in it

`CoreDesk.__init__` sets `self.toolbox = ""`, and the class says why: *"the same loop, told
less, judged by `checkMesh`, and given **no tools at all**"*, with `isolation.scan_run` proving
afterwards that it never found one. **The core arm is supposed to fail this way.** The 27 cells
are not a bug in the corpus; they are the measurement the corpus exists to take.

So the question is not how to stop them. It is which tool they are evidence for — and by the
addition rule, a tool is earned only once the failure it closes has been observed. This one now
has been, with a cost attached.

#### 3.9.5 Which library, and therefore which tool

| library | cells | cases |
|---|---|---|
| **build123d** | **17** | **12** — T5 T6 T10 T11 T12 T15 T16 T17 T18 T19 T23 T24 |
| raw OCCT | 5 | 3 — T5 T10 T22 |
| numpy | 3 | 3 — T1 T4 T19 |
| pyvista | 2 | 2 — T4 T17 |

**Nearly two thirds are build123d, in 12 of 26 cases**, and the repo already has the tool for
exactly that: `openreynolds/toolbox/b123d_api.py`, which generates a ~17 KB
`b123d_api.md` — *curated selection, introspected content, verified live*, with `--check`
refusing to emit an index containing a dead name. Its own header describes the three tiers:
the brief carries one line pointing at it, the file is grepped on demand, and
`help()` / `inspect.signature()` in the kernel answer the exact call against the installed
version. `openreynolds/cad/brief.py:228` already hands that line to the full CAD desk.

**Across this sweep, `b123d_api.md` was opened by 0 of 26 runs and `help()`/`inspect.signature()`
by 1** — which is exactly right, because the core arm does not have a toolbox. That is the
before-measurement.

Two qualifications, both of which matter to scoping the addition rather than to whether to
make it:

- **`b123d_api.md` deliberately excludes raw OCCT** (`BRep*`, `TopoDS*`, `gp_*`) as "the kernel
  underneath, reachable and occasionally necessary". Those 5 cells — including T5's
  `BRepAlgoAPI_Defeaturing.HasErrors`, which cost that run its mesh — are outside its scope by
  design and would not be closed by it.
- **The 3 numpy failures are not API-knowledge failures at all.** `ndarray.ptp` (T1) and
  `numpy.trapz` (T19) were both *removed in NumPy 2.0*. That is version drift, and the fix for
  it is a stated version, not a reference page.

So the honest scoping: an existing, tested tool addresses **17 of 27** of the largest
recoverable failure in this sweep, and the remaining 10 split into a deliberate scope exclusion
and a version note. That is a candidate for the next `/cad-addition` with its before-number
already measured — and per the rule, it goes in **alone**.

## 4. Probes

**Readings, not verdicts.**

| probe | read on | `n/a` | non-zero readings |
|---|---|---|---|
| `union_closure` | 16/26 | 10 | T12 (2,177), T25 (591 — artefact, §3.1) |
| `normals` | 16/26 | 10 | T22 (62,208 — artefact), T25 (314 — artefact) |
| `self_intersection` | 15/26 | 11 | T12 (3,425), T25 (85,902 — artefact) |
| `location_in_mesh` | 12/26 | 14 | T2, T25 `outside`; 10 `inside` |
| `coverage` | **0/26** | **26** | — |
| `scale` | **0/26** | **26** | — |

**Three of the five non-zero surface readings in this sweep are artefacts of a leftover file.**

### 4.1 Two probes screen nothing

`coverage` and `scale` returned `n/a` on all 26 runs and all 22 declares. `coverage` needs a
patch manifest nothing produces; `scale` needs a `spec.json` nothing passes it. The previous
sweep found this from the probe end; giving the probes a channel to the desk changed nothing,
which is the point — the input is missing, not the delivery.

`scale`'s `n/a` string — *"the case states no dimension"* — is false on at least T8, T9, T14
and T24, which all state dimensions in prose. It needs a spec field, and says something else.

### 4.2 A probe declines the largest surfaces and calls it `n/a`

The caps are not the same and neither is announced:

| case | triangles | `union_closure`/`normals` | `self_intersection` |
|---|---|---|---|
| T23 | 224,858 | read | **n/a** (>200k) |
| T16 | 473,184 | **n/a** (>400k) | **n/a** |
| T15 | 695,308 | **n/a** | **n/a** |

A declined read is reported in the same state, with the same word, as a genuinely absent
surface. Nothing downstream can tell "too big to check" from "nothing to check".

### 4.3 A probe reads one path and calls every miss `n/a`

T3, T10 and T11 all exported STLs — to the **case-dir root**, not `constant/triSurface`,
because their route was gmsh/BREP → `gmshToFoam`, not snappyHexMesh. All three read
`n/a: does not exist` with four to eight STLs sitting on disk.

### 4.4 `self_intersection` reports `measured` having tested nothing

T2: `{"pairs": 0, "triangles": 0, "pairs_tested": 0}` against a sphere `union_closure` read as
20,480 triangles. It is reported identically to T13's genuine 1,448-pair clean reading.
(`self_intersection_probe_returns_measured_on_zero_triangles`, recurring.)

### 4.5 The readings can be stale

T19's three surface readings are of the **coarse** triSurface export, frozen before the
refinement; the delivered 41,250-cell mesh superseded it and no probe read the volume mesh at
all.

## 5. The mesh vet, per case

Every run was vetted against its request with independently computed arithmetic.

| case | delivered mesh is the asked-for volume? | on what number |
|---|---|---|
| T1 | yes | `2.87075e-06` vs `2.87124e-06` computed (0.017%) |
| T2 | yes | `0.127967` vs box−sphere `0.1279665` |
| T3 | yes | `1.47421e-06` vs CAD face × depth `1.4741e-06` |
| T4 | yes | `3.55994e-06` vs boolean union `3.560e-06` |
| T5 | **no mesh** | nothing built; 4 PNGs on disk |
| T6 | **no mesh, correctly** | genuine refusal; STEP length unit truly untied |
| T7 | yes | `0.00024` exact; both zones `8e-06` = 0.02³ |
| T8 | yes | `179999.9946` mm³ vs 180,000 |
| T9 | yes | `1.01082e-05` vs `1.0109645e-05` (0.014%) |
| T10 | yes **but §3.3** | `4.21058e-05` vs `4.21117e-05` |
| T11 | yes | `0.000869793` vs `8.697932e-04` (6 s.f.) |
| T12 | yes | `129,321` mm³ vs CAD `129,330` |
| T13 | yes | `357.78` mm³ vs `≈360.0` independently |
| T14 | yes | `0.628826` vs `0.628826308` (6 s.f.) |
| T15 | yes, skew-failed | `0.00256318` vs `0.002558933` (0.17%) |
| T16 | yes **but §3.3** | `0.27629947` vs `0.276393839` (0.03%) |
| T17 | yes | `0.00400193` vs `0.004007` computed |
| T18 | **yes — repair confirmed** | `0.054071155` vs `0.05407058` computed |
| T19 | yes, **never established by the run** | `3.75746e-06` vs `3.7584483e-06` — vet's own `checkMesh` |
| T20 | yes | `0.000118562` vs `1.187522e-04` |
| T21 | yes | `0.00063461` vs `6.34549e-04`; `holes` bbox z `0.0130–0.0250` |
| T22 | yes | `0.000396121` vs `≈4.00e-04`; 36 regions each 2600–2708 cells |
| T23 | yes | `2.0658999e-05` vs `2.065850809e-05` (0.002%) |
| T24 | yes | `0.002379` vs `0.0023819` |
| T25 | **N/A — no volume was asked for** | `5.6391492` of an invented machine |
| T26 | yes, **of a silently-resolved geometry** | `13.2341` vs CAD `13.25043` |

**T18 is confirmed independently and the addition report is right about it.** Duct box
`0.3 × 0.3 × 0.615 = 0.05535`; solid = barrel `763,407.3` + dome `190,853.1` + 12 fins net of
overlap `325,155.6` = `0.00127942`; air = `0.05407058` against `checkMesh`'s `0.054071155` —
**agreement to the sixth decimal**. Merging any single 6 mm gap would add ~54,000 mm³, an
order of magnitude above the residual. It would show. The pre-export B-rep face count is
exactly **51** = 12 fins × 3 + 13 barrel bands + bottom disk + dome, which only a build of 12
discrete fin solids produces. The two bugs the desk found and fixed were duct walls built on a
default in-plane axis, and per-patch independent tessellation preventing shared edges from
welding — per-patch tessellation being the same root-cause family as T12's real gap (§3.4),
though not the float32 mechanism the T12 desk proposed.

Two runs were graded by a `checkMesh` the vet had to run itself, because the run never did:
**T15** (timed out mid-poll) and **T19** (overwrote `polyMesh` and hit the ceiling one step
into reading it back).

## 6. Properties

**60 of 136 measured and printed: 44%.** Per case, strictly counted (`—` = case names none):

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| T1 3/5 | T2 3/5 | T3 **5/5** | T4 2/5 | T5 **0/6** | T6 — | T7 3/4 | T8 4/5 |
| T9 **6/6** | T10 2/6 | T11 4/6 | T12 2/6 | T13 **0/5** | T14 5/6 | T15 **0/6** | T16 2/6 |
| T17 **0/6** | T18 2/6 | T19 **0/6** | T20 3/6 | T21 3/6 | T22 3/6 | T23 4/6 | T24 3/6 |
| T25 — | T26 1/6 | | | | | | |

**T9 and T3 are the only clean sweeps**, and T9 shows what the bar looks like: clearance and
end gap measured on the mesh, region count quoted from `checkMesh`, cells-across-gap counted
by `compute_cell_sizes()` on the **read-back OpenFOAM mesh**, and exhaustiveness proven by
`total patch area 0.008256105493633975 shape area 0.008256105493633975` — equal to full float
precision.

A property this corpus almost never measures: **"blockage ratio"** is absent on both cases
that name it (T2, T14). **"number of connected regions, and the number expected"** is absent
or partial on nine, usually because the expected number is never stated even when `checkMesh`
printed the actual one.

`passed=true` requires none of this. A run that measured nothing the case asked for still
passes clean — T13 and T17 both did.

## 7. Candidate corpus cases

Failures with no case behind them. A failure without a case is a case, not a tool.

1. **A stale artefact in `constant/triSurface` that is not the meshed surface.** Hit twice
   here (T22, T25) and already on file against T4, and in T22 it cost a pass. No case makes a
   leftover export the thing under test.
2. **A surface too large for its instrument.** 200k and 400k caps are load-bearing at
   T23/T16/T15 and invisible in the output. No case is built at a size that trips them
   deliberately.
3. **An export route that is not `constant/triSurface`.** T3, T10 and T11 are all legitimate
   gmsh→`gmshToFoam` builds that no surface probe can see. No case establishes what should be
   read for a desk that never runs snappyHexMesh.
4. **A measurement attempted, found wrong, and silently replaced.** T24 computed a true mesh
   face normal (41.58°), identified its own tangent-reference bug, then substituted a
   design-anchored hybrid (15.23°) and never reconciled the two. Distinct from a literal —
   the substitute is mesh-sensitive — and no case names it.
5. **A zero-clearance spec.** T26 is the only case whose numbers are exactly degenerate, and
   it is scored `expects: done`, so the silent resolution costs nothing.
6. **A run whose only failed `checkMesh` check is quality, not topology.** T15's skewness
   failure is a different kind of wrong from T5's missing mesh, and the record flattens both
   to `passed: false`.

---

*Produced by `/cad-sweep` from 26 per-run mesh vets. The ranked list is above; the next step
is the reader's.*
