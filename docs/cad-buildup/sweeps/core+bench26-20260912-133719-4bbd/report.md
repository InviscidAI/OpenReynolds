# Sweep `core+bench26-20260912-133719-4bbd` — fifteen geometries, and three claims withdrawn

## 1. The sweep

| | |
|---|---|
| label | `core+bench26` — the same core desk, 15 cases not previously run |
| model / effort | `claude-opus-5` at `medium` — **unchanged for the third sweep running** |
| core sha | `af131f1` (clean tree; the diff from `1f5d9dc` adds fourteen prompt files, corrects T9's, and generalises one test) |
| cases | T9 and T13–T26, **one run each** |
| baseline | `core+bench12-20260912-123315-8815`, recorded in the manifest |
| ended `done` | 13/15 (T15 and T25 ended `time`) |
| passed | **13/15** |
| cells | 262 |
| spend | **$15.26** |

| case | ended | cells | first mesh | checkMesh | passed | $ | probes measured |
|---|---|---|---|---|---|---|---|
| T9 | done | 18 | 6 | ok | yes | 0.84 | 3 |
| T13 | done | 22 | 9 | ok | yes | 2.19 | 3 |
| T14 | done | 14 | 6 | ok | yes | 0.48 | 4 |
| T15 | **time** | 13 | 7 | **not ok** | **no** | 1.45 | 4 |
| T16 | done | 21 | 8 | ok | yes | 0.88 | 3 |
| T17 | done | 18 | 11 | ok | yes | 0.71 | 4 |
| T18 | done | 14 | 3 | ok | yes | 0.52 | 4 |
| T19 | done | 20 | 11 | ok | yes | 1.84 | 3 |
| T20 | done | 11 | 3 | ok | yes | 0.36 | 4 |
| T21 | done | 21 | 15 | ok | yes | 0.77 | 3 |
| T22 | done | 14 | 7 | ok | yes | 0.70 | 3 |
| T23 | done | 15 | 9 | ok | yes | 0.72 | – |
| T24 | done | 24 | 12 | ok | yes | 1.69 | 4 |
| T25 | **time** | 16 | – | **not ok** | **no** (wants `refused`) | 1.13 | – |
| T26 | done | 21 | 11 | ok | yes | 0.98 | 4 |

**The paired half is one case and carries nothing.** Only T9 appears in both sweeps, and its
brief was rewritten between them, so `p = 1.000` over a single row is not a result. This sweep
is a breadth measurement, not a comparison: fourteen of its fifteen cases had never run.

### 1.1 The export requirement worked

Three probes went from **2 verdicts in 12 runs** to **13 in 15**. The last sweep's §4.1
recorded that T9–T12 named patches without requiring them exported and all four returned
`n/a` on all six probes; every case here demands one STL per patch in the request itself.
`union_closure`, `normals` and `self_intersection` now have a surface to read on 13 of 15,
and `location_in_mesh` on 7. **§6 candidate 4 is closed.**

## 2. Contamination

**0/15. Zero.** Three sweeps, 41 runs, no contamination.

## 3. Three findings from the last two sweeps are withdrawn or corrected

Put first because two of them were mine and one changes what the last sweep concluded.

### 3.1 `self_intersection` is **not** a defective probe — withdrawn

The last sweep's §4.2 named it a fourth defective probe for returning `state: measured` with
`triangles: 0` while `union_closure` read 51,282 triangles from the same directory. That was a
misreading of two fields that share a name:

```python
# probes.py, _self_intersection
measured = {"pairs": len(crossings["pairs"]),
            "triangles": len(crossings["triangles"]),      # triangles *involved in crossings*
            "pairs_tested": crossings["tested"]}
```

`union_closure`'s `triangles` is the surface total; `self_intersection`'s is the count
*participating in a crossing*. So `triangles: 0` against `pairs_tested: 204` reads "204 pairs
tested, none crossing" — coherent, and a pass. This sweep makes it unmistakable: seven cases
report 0 and six report non-zero, from 340 on T26 to 544,306 on T23. A probe that was not
reading its input could not do that. **Finding withdrawn; the probe is sound.**

### 3.2 `scale` skips because nothing ever gives it a spec — mechanism corrected

The baseline's §4.1 said *"the probe is not reading the dimensions the case states"*, and the
last sweep repeated it. The conclusion — no verdict in any sweep — stands. The mechanism was
wrong, and this is the actual one:

```python
def _scale(case, spec):
    stated = spec.get("extent_m")
    if not stated:
        return ProbeResult("scale", NOT_APPLICABLE, "the case states no dimension, ...")
```

```python
# cad_sweep.py:123 -- the driver passes --spec only if the file is already there
*(["--spec", str(run_dir / "spec.json")] if (run_dir / "spec.json").is_file() else [])
```

**Nothing in the pipeline writes `spec.json`.** No run directory in any sweep contains one. So
`spec` is `{}`, `extent_m` is absent, and the probe correctly declines to measure against a
dimension it was never handed. The briefs state dimensions in abundance and `load_prompts()`
parses them; the missing step is between the prompt and the probe, in the harness.

### 3.3 The four new cases were the wrong fix for `coverage`, and cannot be the right one

`coverage` refuses without a patch manifest declaring which face was meant to be whose. The
core desk writes no manifest. Thirteen of fifteen runs here exported one STL per patch and
`coverage` still returned `n/a` on all fifteen. **The last sweep's §6 candidate 3 — "a case
that forces `coverage` and `scale` to return a verdict" — is not closable by adding cases**,
and writing four partly for that reason was a misdiagnosis. Both probes need harness work:
`scale` a `spec.json`, `coverage` a manifest.

## 4. And the biggest claim from the last sweep does not survive breadth

### 4.1 "Areas and volumes measured well, never a length at a station" — withdrawn

The last sweep's §5 closed on that pattern, four cases for four, and called it *"the most
reproducible finding in the corpus"*. Nine of this sweep's cases ask for a length at a
station. **The desk measured them**, and mostly in the hardest available form:

- **T19**, the 0.05 mm clearance — `radial clearance [mm] = 0.05000 (target 0.050)`, computed
  as `(np.hypot(bo[:,0],bo[:,1]).min() - np.hypot(sh[:,0],sh[:,1]).max())*1e3`: the nearest
  approach between the bore point cloud and the shaft point cloud. That is exactly "measured
  between the two meshed wall patches", on the smallest gap in the corpus.
- **T13**, the helical flank gap — `gap neck->lid min/max : 0.2000 / 0.2521 mm (asked 0.2)`
  off a distance array. **The spread is the proof**: a restated input prints 0.2 twice.
- **T21**, the blind hole — `hole z-range (mm): 13.000 .. 25.000 -> depth 12.000, asked 12
  from top 25`, off the hole patch. That is the through-hole-versus-blind-hole discriminator
  `checkMesh` cannot see, measured correctly.
- **T24**, the direction — `duct0: axis-vs-tangent 15.00 deg (asked 15)` by dot product per
  duct, all four, **and its sense**: `swirl sense (din x rhat)_z, all same sign: [-0.966 ×4]`.
  The brief called reporting the angle without its sense half a measurement; the desk reported
  both.
- **T22** — vane width from two face centres; **T9** — patch areas against exact values.

### 4.2 The control says the same thing

T20 is a 42 mm cylinder, deliberately the easiest geometry either source suite contains, put
in the corpus so a failure could not be blamed on difficulty. It measured:

```
length z = 0.041999999999999996 want 0.042
diameter = 0.06 want 0.06
gasketInner  faces=  344 area=0.000377 z=[0.0200,0.0220] r=[0.0300,0.0300]
exact areas: inlet/outlet 0.0028274333882308137  bore(20mm) 0.0037699111843077517  gasket 0.00037699111843077514
bolt hole inner radius = 0.063 > bore radius 0.03 -> True
```

Five patches named distinctly — `inlet, outlet, flangeBore1, gasketInner, flangeBore2` — not
collapsed into one `walls.stl`, with the gasket band localised to `z=[0.0200,0.0220]` between
the two flange bores. The bolt holes' absence from the flow domain established by computation
rather than asserted. And `0.041999999999999996` is worth reading twice: **the floating-point
tail is the signature of a computed number**, which is the cheapest available tell against a
typed literal.

**What the earlier claim actually was.** Four cases, one run each, all four hard — a
sub-cell clearance, a sub-cell trailing edge, a hundred-cutter boolean, a filleted impeller.
The behaviour generalised from them was a behaviour of the desk under difficulty, and this
sweep shows it is not a behaviour of the desk. T12's literals and T9's restated clearance
remain what they were; they are no longer evidence of a habit.

### 4.3 Two properties are still taken from the inputs

Named because the withdrawal above is not a clean sheet:

- **T16** evaluated its annulus heights from the radius functions the geometry was built from —
  `np.round(r_out(Z)-r_in(Z),4)` giving `[0.17 0.12 0.16]`. The non-monotonic bulge the case
  exists to catch is reported correctly and is **not evidence the mesh has it**.
- **T22** counted its passages with `len(solid.solids())`, the CAD model's own answer, before
  meshing. The brief asks for the count on the mesh because a boolean can lose a vane after the
  count is taken.

## 5. Failures, ranked by cost

| # | failure_id | cases | cells | $ |
|---|---|---|---|---|
| 1 | `dimensionless_request_built_instead_of_refused` | **T25** | 16 | 1.13 |
| 2 | `exported_surface_has_open_edges` | **T16, T17, T18, T26** | 67 | 3.09 |
| 3 | `exported_surface_winding_inconsistent` | **T13, T14, T15, T16, T18** | 84 | 5.52 |
| 4 | `near_contact_solids_produce_highly_skew_faces` | T15 | 13 | 1.45 |
| 5 | `scale_probe_never_receives_a_spec` | all 15 | 0 | 0.00 |
| 6 | `coverage_probe_unreachable_without_patch_manifest` | all 15 | 0 | 0.00 |
| 7 | `duct_area_schedule_measured_off_generating_functions` | T16 | 21 | 0.88 |
| 8 | `repeated_feature_counted_on_solids_not_mesh` | T22 | 14 | 0.70 |

### Rank 1 — the refusal path fails on both of its cases, for different reasons

T25 states no dimension anywhere. The desk built a turbofan:

```
bypass exit annulus r 0.55..0.78, core exit annulus r 0.14..0.44
volumes: inlet 1.8749  bypass 1.3378  core 0.4726  union 3.6853 m^3
{'fan': (960, 3.4891), 'ogv': (960, 1.1755), 'compressor1': (960, 0.3615), 'turbine1': (960, 0.3903)}
```

Fan and bypass radii, core radii, stage count, 960 faces per blade row, a splitter nose, a
3.69 m³ fluid volume. **Every one of those numbers is invented** — the request contains no
length, no ratio and no count. It then ran out of wall clock polling a mesh that never
finished, its last cell being `import time; time.sleep(120); print("poll")`.

This is what the case was written to find out, and it is the one finding here that breadth
bought and repetition could not. T6's failure is a single missing unit, guessed — and guessed
correctly, as it happens. T25's is a machine specified from nothing. The brief put it as:
*"A desk that refuses T6 and builds T25 has not learned the rule, it has learned to check for
`LENGTH_UNIT`."* **The answer is worse than that: it does neither.** Two cases, two different
triggers, both failed — which is a finding about the principle rather than about one sample.

### Ranks 2 and 3 — there is no closure assertion anywhere in the desk's path

*Rewritten 2026-09-13. The first version of this section reported dirty surfaces. The question
"how can a leaky surface ship silently — is there no assertion?" has an answer, and it is the
finding rather than the symptom.*

**There is no assertion.** `grep -ci` over `openreynolds/buildup/core.py` and `record.py` for
`open_edges`, `free_edges`, `is_closed`, `watertight`, `manifold` and `closure` returns **0 for
every one**, and the core desk's brief never mentions closure, watertightness or leakage. The
only thing that looks at closure is the `union_closure` probe, which by design lives in the
supervisor — a separate process, reading the case off disk, **with no channel into the
conversation** — and no probe is a gate.

So the chain is:

1. the desk exports a surface; nothing checks that it closes;
2. `snappyHexMesh` meshes it — and snappy is a Cartesian cutter, so it emits closed cells
   **whatever the input surface did**;
3. `checkMesh` validates the *volume mesh*, which is therefore closed, and prints `Mesh OK.`;
4. `checkmesh_ok` is the pass criterion, so the run passes;
5. `union_closure` notices, from outside, and gates nothing.

**A leaky surface does not produce a bad mesh. It produces a good mesh of the wrong volume.**
That is the failure `checkMesh` is structurally unable to see, and `checkMesh` is the desk's
sole authority. The corpus now has it four times — T16, T17, T18, T26 — every one
`checkmesh_ok: true, passed: true`.

**One honest qualification, which the probe states itself:** *"Open is not the same as wrong: a
baffle is a zero-thickness wall on purpose, and snappy meshes one deliberately."* True, and it
does not rescue these four: T18's fins are `Cylinder(R_F, 0.003)` solids unioned into the
barrel and T26's treads are 40 mm thick, so a deliberate baffle is not the explanation. It does
mean the gap is not "assert closure" but **"declare what was intended and check against it"** —
which is the same missing declaration `coverage` needs a manifest for. One gap, two probes.

**T26 makes the point twice over.** Its 259 free edges are almost certainly the tread–column
tangency the case exists to catch, surfacing as unwelded coincident edges. So the case's central
defect *was* measured, *was* written into the record, and the run still scored `passed: true`.
The desk never mentioned the tangency and the harness detected it and passed anyway. Its fluid
volume is nonetheless right to 0.04% — 13.25043 m³ against 13.256 analytic — so the boolean did
not lose material. The geometry is sound and its description is not, which is precisely the
distinction nothing in the record can make.

### The numbers behind that

Making the export mandatory got the probes their input, and the input is worse than the two
earlier sweeps suggested, because there was almost nothing to read then.

`union_closure` returned zero open edges on every case that reached it in the baseline. Here:
**T18 13,204** open edges on 79,187 triangles, T17 1,908, T16 776, T26 259. And `normals` now
fires on five cases at once, worst on **T15 at 3,700 flipped edges out of 7,796 triangles —
47% of the surface**, against the 4-on-20,804 that first raised it on T2.

Whether any of it damaged a delivered mesh is exactly the question the baseline's §6 candidate 5
asked and this sweep still cannot answer: T16, T17, T18 and T26 all ended `done` with a clean
`checkMesh`. Three firings with no downstream consequence was already the position; it is now
nine, and the evidence still neither justifies a gate nor refutes one.

### Rank 4 — T15, where `checkMesh` did its job

Two bevel gears meshing across 0.3 mm of backlash:

```
***Max skewness = 4.1299627, 4 highly skew faces detected which may impair the quality of the results
Failed 1 mesh checks.
Finished meshing in = 302.47 s.
```

The case was written to ask what happens when two solids nearly touch, and the answer is four
highly skew faces at the contact — caught by `checkMesh`, correctly failing the run, with the
number and the location available. The run then hit its second budget. **This is the corpus's
first case where a starred `checkMesh` warning became the verdict rather than being discarded**,
which is the other half of the last sweep's rank 3.

## 6. Probes

| probe | measured | n/a | what it found |
|---|---|---|---|
| `union_closure` | 13 | 2 | **4 cases not closed**, worst 13,204 open edges |
| `normals` | 13 | 2 | **5 cases with flipped edges**, worst 47% of the surface |
| `self_intersection` | 13 | 2 | 6 cases with crossing triangles; **probe exonerated, see §3.1** |
| `location_in_mesh` | 7 | 8 | `inside` on all 7 — correct for internal flow, and the first internal-flow evidence the probe works |
| `coverage` | 0 | 15 | **never once in 41 runs**; needs a manifest, not a case (§3.3) |
| `scale` | 0 | 15 | **never once in 41 runs**; needs a `spec.json`, not a case (§3.2) |

`location_in_mesh` is worth one line of credit. The baseline established it as a false positive
on external flow, where `outside` is the correct answer and the probe reports it as a finding.
Seven internal-flow cases here all report `inside`, correctly. The probe's defect is narrower
than "it does not work": it lacks a notion of the background box, and says nothing useful about
external flow specifically.

## 7. Candidate corpus cases

1. **A third refusal case whose trigger is neither a missing unit nor a missing dimension** —
   two triggers, two failures, and the refusal terminal has never once been reached in 41 runs.
   A contradictory spec, or a request for something that is not a fluid domain.
2. **A case where an open edge or a flipped normal demonstrably damages the mesh.** Nine
   firings across three sweeps, every one alongside a clean `checkMesh`. Still the failure with
   evidence and no consequence.
3. **A long-running single cell** — still uncovered after three sweeps. T15 and T25 both died
   on wall clock across many cells, not in one.
4. **A case whose pass is a count taken on the mesh after a boolean.** T22 counted on the
   solid; nothing yet forces the distinction.

**Not corpus cases, and they were miscategorised as such before:** `coverage` and `scale`. Both
need a wire, not a geometry.

---

*Produced by `/cad-sweep`. This is a decision input, not a decision. Three claims from earlier
sweeps are withdrawn or corrected above; the report they appeared in has been amended in place
rather than rewritten.*
