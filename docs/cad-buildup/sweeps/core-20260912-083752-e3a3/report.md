# Sweep `core-20260912-083752-e3a3` — the corpus's first baseline

## 1. The sweep

| | |
|---|---|
| label | `core` — the bare core desk, no added tool |
| model / effort | `claude-opus-5` at `medium` |
| core sha | `0841924` (clean tree) |
| cases | T1–T8, **one run per case**, `--parallel 2` |
| baseline | **none** — first sweep of the corpus, so it takes none and becomes the chain's root |
| ended `done` | 6/8 (T3 `steps`, T5 `no-progress`) — but see T6 |
| passed | **5/8** — T6's criterion is now recorded, see §1.2 |
| cells | 138 |
| spend | **$7.49**, and still an undercount — see §1.1 |

| case | ended | cells | first mesh | checkMesh | $ | probes fired |
|---|---|---|---|---|---|---|
| T1 | done | 13 | 3 | ok | 1.66 | – |
| T2 | done | 16 | 4 | ok | 0.49 | `location_in_mesh`, `normals` |
| T3 | steps | 28 | 19 | no | 3.04 | – |
| T4 | done | 16 | 8 | ok | 0.66 | `normals` |
| T5 | no-progress | 22 | – | no | 0.00 | `normals`, `self_intersection` |
| T6 | done | 19 | 11 | ok | 0.77 | – |
| T7 | done | 10 | 3 | ok | 0.30 | – |
| T8 | done | 14 | 4 | ok | 0.58 | – |

### 1.0 The dollars in the first version of this report were 2.5x under

Corrected in place. `scripts/cad_accept.py` held one untagged price table at Sonnet 5's
rates -- Sonnet 5 being what the default preset runs -- while the sweep sets
`mesher_model` to Opus 5 at $5/$25 per Mtok. Nothing reconciled the two, so every figure
here was multiplied by 0.4: the corpus was reported at **$3.00** and cost **$7.49**.

The *ranking* in §3 is untouched: one sweep runs one model, so the error is a uniform
scalar and a uniform scalar cannot reorder anything. Every absolute number was wrong, and
so would be any comparison against a sweep whose model differed -- which is what the
baseline chain exists to make. Pricing now lives in `openreynolds/llm/presets.py` keyed by
model, and `cad_buildup.py` refuses a model it has no rates for.

### 1.1 Two numbers in this table are wrong, and both flatter the desk

**T5's spend is not recorded, it is missing.** `record.json` carries `usd 0.0`, `seconds 0.0`
and `tokens {}` while `watch.json` independently logged `seconds 535.0` and the heartbeat
spans 25 turns and 17,049 output tokens. Every other run — *including T3, which also ended
abnormally, at `steps`* — is fully accounted. So the correlation is not "did not finish
normally"; it is specifically **the alarm-kill path returning before the accounting runs**.
At the corpus median of $0.0412/cell, T5's 22 cells are about **$0.91**, and by wall clock
its 535 s is the second-longest run in the sweep. Ranking failures by the recorded number
puts T5 last when it belongs near the top. *This is a harness defect, not a result.*

**T6 was counted as a `checkMesh` pass and is the sweep's clearest failure.** See §3,
rank 3. It now scores as a failure in the record too — see §1.2. The headline is 5/8.

### 1.2 T6 is now scoreable, and scores as a failure

When this sweep ran, nothing in the record could express a case whose pass is the desk
*declining*. `checkmesh_ok` scored T6 by whether it finished, and it finished — so the
corpus's clearest failure was its sixth success, and a later core that correctly refused
would have read as a regression in the same column.

Since fixed. The desk has a refusal terminal (`print("CAD_REFUSED: <reason>")` →
`stopped: refused`), `TERMINAL` has a word for it, T6's prompt declares `**Passes as:**
`refused`` in its own file, and the record carries `expects` alongside a derived `passed`.
The records here were backfilled: `expects` is a fact about the case and `passed` is
derived from it, so neither touches what was measured. T6 now reads `passed: false`
against `stopped: done`, which is the whole point — those two disagreeing is the finding.

Totals in this report are on `passed`, so **5/8**, not the 6/8 the first version printed.

## 2. Contamination

**0/8. Zero.** No run's own output contains a house surface.

This is worth one line of history because the number was not zero an hour ago. The first
attempt at this sweep (`core-20260912-082722-9e9c`, discarded) graded **every** run
contaminated: `cad_sweep.py` writes the runner's captured stdout to `runner.log` inside the
run directory, that stdout is operator-facing and names the repo path twice, and `scan_run`
grepped it. `OBSERVER_FILES` already existed for exactly this hazard but listed only
`record.json`. Fixed in `0841924`, with a regression test asserting both halves — the
driver's log is skipped, and a house path in `cells.log` beside it still contaminates. The
void sweep was discarded rather than regraded.

## 3. Failures, ranked by cost

Cost is `cases hit × cells burned × spend`. Attributed spend uses each case's own per-cell
rate; T5 uses the corpus median, because T5's spend was never recorded (§1.1). All
figures are at Opus 5 rates after the correction in §1.0.

| # | failure_id | cases | cells | $ | probe |
|---|---|---|---|---|---|
| 1 | `fabricated_cell_output_after_fence` | **T3, T5** | 4 direct | 0.43 | – |
| 2 | `exported_surface_winding_inconsistent` | **T2, T5** | 1 | 0.05 | `normals` |
| 3 | `undeclared_step_unit_guessed_instead_of_refused` | T6 | 19 | 0.77 | – |
| 4 | `boolean_union_of_swept_bands_loses_material` | T3 | 13 | 1.41 | – |
| 5 | `checkmesh_fatal_on_missing_case_dicts` | T3 | 3 | 0.33 | – |
| 6 | `gmshtofoam_drops_physical_patch_names` | T3 | 3 | 0.33 | – |
| 7 | `gmsh_recombination_fails_on_odd_boundary_division` | T3 | 2 | 0.22 | – |
| 8 | `polymesh_boundary_parsed_by_hand_rolled_regex` | T3 | 2 | 0.22 | – |
| 9 | `defeature_call_cut_without_printing_anything` | T5 | 2 | 0.08 | – |
| 10 | `exported_surface_self_intersecting` | T5 | 1 | 0.04 | `self_intersection` |
| 11 | `meshing_point_outside_exported_surface` | T2 | 0 | 0.00 | `location_in_mesh` |
| 12 | `exported_surface_duplicated_in_trisurface` | T4 | 0 | 0.00 | `normals` |

**Rank 1 is ranked first on 4 directly-attributed cells, and that understates it badly.**
Its cost is almost entirely downstream. In T5 it *is* the ending: the desk stopped emitting
cells and wrote its own results, the step count went flat at 22, and the watcher killed the
run — the whole 22 cells and ~535 s. In T3 the invented transcript broke the fence at turn
26 so the runner extracted no cell at all, silently dropping the cell that would have written
`fvSchemes`/`fvSolution` — which is rank 5, which is why T3 never got a `checkMesh` verdict at
all. A failure that costs two whole runs and manufactures a third failure is the most
expensive thing here by a distance; the cells column simply cannot see it.

The desk invented an exit line, a cell count of **9884** where the real cell printed **8539**,
and a `Mesh OK.` verdict that appears nowhere in `cells.log`. It then made design decisions
on evidence it had authored. In T5 it caught itself once — *"Wait — I need to actually run
it."* — and did it again two turns later.

**Rank 2 recurs across two cases with unrelated geometry** (T2: 4 flipped edges on 20,804
triangles; T5: 1,156 on 72,966), and in both the desk printed a triangle count and a solid
count but never an orientation number.

**Rank 3, T6, is the one to read twice.** Its brief makes *not finishing* the pass: the STEP
declares a bare `LENGTH_UNIT`, and the desk must report up rather than guess. The desk found
the missing unit — it scanned and saw `#426 = ( LENGTH_UNIT() NAMED_UNIT(*) );` — wrote the
guess into a constant, and carried on to a finished, correct-looking mesh:

```
MM_TO_M = 0.001    # STEP declares a bare LENGTH_UNIT; extents imply mm
```

The brief anticipates this verbatim, under the heading *"A pass that is really a failure"*:
**"It passed by luck, and the next file is inches."** It also breaks the *refusal arriving
late* clause — the refusal was to come from the first read of the file, not after a shape had
been built on numbers of unknown scale. **Nothing in the record can express T6's criterion**:
`checkmesh_ok` scores a case whose correct outcome is not finishing by whether it finished.

## 4. Probes

**79% of all probe verdicts are `skipped` — 38 of 48.** Skipped is not a pass.

| probe | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | verdicts |
|---|---|---|---|---|---|---|---|---|---|
| `union_closure` | skip | pass | skip | pass | pass | skip | skip | skip | 3 pass |
| `normals` | skip | **fire** | skip | **fire** | **fire** | skip | skip | skip | 3 fired |
| `self_intersection` | skip | pass | skip | pass | **fire** | skip | skip | skip | 2 pass, 1 fired |
| `location_in_mesh` | skip | **fire** | skip | skip | skip | skip | skip | skip | 1 fired |
| `coverage` | skip | skip | skip | skip | skip | skip | skip | skip | **never once** |
| `scale` | skip | skip | skip | skip | skip | skip | skip | skip | **never once** |

**Every probe skipped in 5 of 8 cases** (T1, T3, T6, T7, T8). Only T2, T4 and T5 exercised
Layer B at all. The cause is structural, not incidental: `union_closure`, `normals` and
`self_intersection` all refuse without `constant/triSurface`, and five of eight cases never
write one — T1/T7/T8 are blockMesh-only, T3 went gmsh→gmshToFoam, T6 exports STEP and meshes
an analytic O-grid. **The corpus is not exercising the probes it has.**

`coverage` and `scale` have never returned a verdict in any case of any sweep. That is worth
saying out loud, per the skill.

### 4.1 Three of the six probes are themselves defective

Every firing this sweep produced was interrogated, and three probes were found measuring
something other than what the desk did. **A gate built on any of them as written would block
correct work.**

- **`scale` — confirmed wrong, three cases.** It skips with *"the case states no dimension,
  and this probe is measured against the request"* on T1, T7 and T8, whose briefs state
  10 mm / 120 mm / 15 mm, 100 × 60 × 40 mm, and 30 mm / 200 mm / 40 × 30 × 10 mm respectively.
  Three graders flagged it independently. The probe is not reading the dimensions the case
  states.
- **`location_in_mesh` — false positive on external flow (T2).** It reads only
  `constant/triSurface` and has no notion of the blockMesh background box, so for *any*
  external-flow case a correct point is necessarily `outside`. The desk's own number refutes
  the probe: `Total volume = 0.127967` against a 0.8 × 0.4 × 0.4 box minus the sphere — the
  mesh is the fluid *outside* the part, which is what the case wants.
- **`normals` — fires on two unrelated conditions under one name (T2 vs T4).** T4's 132
  flipped edges on 88 triangles is *every* edge, because `constant/triSurface` held the same
  44 triangles twice; `surfaceCheck` on the surface actually meshed printed `Number of zones
  (connected area with consistent normal) : 1`. The probe welds the whole directory while
  `meshDict` names one file, so its input is not the surface that was meshed. A
  `flipped_edges > 0` gate would have failed a run whose meshing surface was certifiably
  clean and whose delivered mesh came from blockMesh anyway.

**Recommendations on activation, not decisions.** `normals` and `location_in_mesh`: reported
finding, not gate, until their inputs are fixed. `self_intersection`: the one firing (T5,
3,262 of 6,074 pairs crossing, 54%, from exporting an un-fused three-solid compound as one
STL) is mechanically clear and `union_closure` passed alongside it, so closure alone will not
catch it — but it is one firing in one case, and T6 cannot confirm it, having no STL and no
fuse. I have **not** moved any probe's state in `docs/cad-silent-failures.md`; §7 leaves the
gate-vs-advice question to the first activation, and that call is yours.

## 5. Properties — measured **and printed**?

The rule applied throughout: a number re-printed from a variable the desk set is **not** a
measurement, whatever the closing prose claims.

- **T1** — bend radius ✅ `centreline 15.000 mm`; included angle ✅ but only as `arc R = 20.000
  mm … len = 62.832 mm` (= π·20), never in degrees; one-cell thickness ✅ `Mesh has 2 geometric
  (non-empty/wedge) directions (1 1 0)`. ❌ **passage width at inlet and mid-leg** — crown only;
  the closing table's "legs 10.000 mm" appears nowhere in the output. ❌ **leg length, both
  legs** — the cell that would have measured it died on `ValueError: substring not found` and
  was never retried; the prose claim "x from −120.000 mm to 0 on both legs" was never printed.
- **T2** — diameter ✅ `sphere bbox dx,dy,dz: 0.0399926…`; domain extent ✅ `upstream/D 5.00
  downstream/D 15.00`. ❌ **blockage ratio** — never printed, the word never appears. ❌ **the
  four patches' areas and mean normals** — only `sphere patch area 5.004867e-03`; the per-patch
  loop died on `TypeError: '<' not supported between instances of 'PolyData' and 'int'` and was
  never retried. ⚠️ layer coverage printed as `100` off snappy's log table, not off the mesh.
- **T3** — overall length ✅ `length 0.1200 m`. ❌ **channel width** — `width 0.0060 m` is a
  literal in the format string. ❌ **4 bypass loops** measured on the 2D face only, never on the
  mesh. ❌ **connectivity, no pocket, no dead end** — never established; checkMesh's region check
  is exactly the verdict that never ran.
- **T4** — channel width/depth ✅ at two stations, off the meshed patches; fluid volume ✅ `3400.0
  mm^3 (exact 3400.0)`. ❌ **plate footprint**, ❌ **number of passes**, ❌ **wall thickness between
  passes**, ❌ **minimum local width** — `rib gap (mm): 3.0` is arithmetic on inputs; no
  minimum-distance query over any solid anywhere in the run.
- **T5** — fluid volume ✅ `fluid vol 0.00571881  expected 0.00571881`, but the enclosure is the
  duct box *around* the assembly, not the air path inside it. ❌ **healShapes before/after**, ❌
  **fillets removed and volume added** (the only defeature cell was cut at the 240 s cap printing
  nothing), ❌ **capping faces**, ❌ **patch exhaustiveness**.
- **T6** — geometry and fluid volume ✅ throughout (`fluid volume: 2.513274e-04 m^3   expected
  2.513274e-04`). The brief names no property list; its criterion is the refusal, and that
  failed. See §3 rank 3.
- **T7** — cavity dims ✅; zone sizes ✅ `size=[0.02 0.02 0.02]`; single boundary patch ✅ `ok
  (closed singly connected)`. ❌ **zone centroids** — *the brief names the centroid as the very
  thing that exposes the false pass* (a topoSet box that also caught floor cells) and **no
  centroid was printed anywhere**, only bounding boxes. ❌ **positions relative to floor and
  ceiling** — "5 mm above the floor" is `GAP_Z` restated in prose, never differenced against the
  cavity extent. T7 is the cheapest run in the sweep at 10 cells and $0.12, and the property
  that would have caught a false pass is the one it skipped.
- **T8** — the best-measured case in the corpus. Duct and block dims ✅ with `want:` echoed beside
  the bbox query, which is the honest form; both regions ✅ with counts; **interface conformality
  ✅ properly** — `192 faces, area 0.0012 m^2` on both sides and `max face-centre mismatch across
  interface: 0.0 | all at z=0: True`, computed across regions from `polyMesh` rather than from
  inputs. This is the one property in the corpus backed by a cross-region computation, and it is
  exactly the check the brief says per-region `checkMesh` cannot see.

## 6. Candidate corpus cases

Failures the corpus does not currently cover. **A failure without a case is a case, not a tool.**

1. **A case whose pass is a refusal, that the harness can score.** T6 proves the desk will guess
   rather than refuse, and equally that nothing in the record can express it. Needs a recorded
   field for the refusal criterion before any addition can be measured against it.
2. **A unit-less fixture that is *not* millimetres.** T6 passed by luck. An inch-declared or
   metre-scaled fixture turns the lucky pass into a visible factor-of-1000.
3. **A case that forces `coverage` and `scale` to return a verdict.** Both have never fired,
   passed, or been anything but skipped. Until one case exercises them they are untested code.
4. **An STL-exporting case among the blockMesh five.** Five of eight cases cannot reach three of
   the six probes by construction.
5. **A case where flipped edges actually damage the mesh.** Three firings, zero downstream
   consequence — so the evidence to date cannot justify a gate, and cannot refute one either.
6. **A long-running single cell.** T5's defeature was cut at the 240 s cap having printed nothing,
   and the desk never retried; nothing in the corpus tests a legitimately slow operation.

---

*Produced by `/cad-sweep`. This is a decision input, not a decision: it names failures and what
they cost, and proposes no tool. The next step is yours.*
