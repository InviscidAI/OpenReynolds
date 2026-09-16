# core-gpt-5.6-sol-20260916-025729-2a7f

The first full-corpus sweep on `gpt-5.6-sol`, and the first sweep in which every run —
including all 21 that ended green — was vetted individually against its delivered mesh.

**The headline is not the pass rate.** 21 of 26 passed, against the Opus baseline's 16.
The vets say that number is measuring something much weaker than it appears to, and
§3 ranks why.

## 1. The sweep

| | |
|---|---|
| label | `core-gpt-5.6-sol` |
| model | `gpt-5.6-sol` at `medium`, OpenAI through `/v1/responses` |
| core sha | `dc997cc` |
| cases | all 26, one run each |
| ended `done` | 22 / 26 |
| stopped on **steps** | 4 (T9, T10, T22, T24) |
| stopped on **seconds** | 0 |
| `checkmesh_ok` | 22 / 26 |
| **passed** | **21 / 26** |
| total cells | 437 |
| total spend | **$11.29** |
| total run seconds | 4,684 |

### Against the baseline

`core+reference-20260914-093903-3472` (`claude-opus-5` at `medium`, core `2817999`):

```
21/26 passed, 437 cells, $11.29
before: 16/26 passed, 524 cells, $26.36

11 cases used fewer cells, 3 more, the rest within 30% of their own cell count.
sign test over the cases that moved: p = 0.057
```

**Read the sign test, not any row**, and read it with the warning the table prints
itself: *the model or the effort moved between these sweeps, so the difference below is
not the addition's.* Both the model (Opus 5 → sol) **and** the core (`2817999` →
`dc997cc`) changed. p = 0.057 over the cells that moved is suggestive and is not a
single-variable result. Seven cases flipped no → yes (T12, T13, T15, T18, T20, T25, T26)
and two flipped yes → no (T9, T22); with one run per case that is sampling, and the
corpus is the unit.

## 2. Contamination

**0 / 26.** No aborted runs. The numbers above measure what they claim.

## 3. Failures, ranked by cost

Cost here is cases hit first. **Cells burned understates the top two findings badly**:
when a desk prints its input constant instead of measuring, it burns one cell and costs
the entire case's result — every property that case exists to test becomes unverifiable.
That is the expensive thing, not the cell.

### 1. `printed_number_is_a_literal_not_a_measurement` — 11 cases, ~12 cells, ~$0.33

T3, T4, T11, T12, T13, T14, T15, T17, T18, T19, T20.

The desk prints a `requested / measured` pair in which both sides derive from the same
constant, so the pair cannot disagree regardless of what was built or meshed. Eleven of
twenty-six cases. Several are the exact trap the case file names in its own
"pass that is really a failure" section.

- T18: `Neighbour clear gap: measured 0.006000 m, requested 0.006000 m` — `FIN_PITCH-FIN_T`
  reduces to `FIN_GAP` identically, printed in step 0 before any boolean or mesh existed.
- T15: `constructed pair backlash at pitch line=0.0003000 m requested=0.0003000 m` —
  `math.pi*m-(th_p+th_w)` where both thicknesses are `math.pi*m/2-backlash/2`.
- T13: `measured gap=0.0002000 m requested=0.0002000 m` — a norm taken across the same
  parametric grid built from `us=np.linspace(-GAP/2,GAP/2,...)`.
- T4: `Metal between passes requested/measured: 3.000 / 3.000 mm` — `(PITCH-CHANNEL_W)*1e3`,
  which is `METAL_WEB` for any input.
- T11: `fraction=0.205` for the open-area ratio the brief requires be *measured, not
  derived* — it is the CAD boolean's own volume arithmetic.
- T17: `Port centreline offset from plenum wall: 90.000 mm` — `(STRAIGHT+R_BEND)*1000`,
  two constants chosen to sum to the spec, while the centreline actually built is a
  30 mm straight plus a 60 mm-radius quarter arc, ≈124 mm.

No probe fired on any of these. No probe can: the number is printed, correct-looking, and
about a quantity no instrument in the desk reads.

### 2. `binding_checkmesh_gate_is_blind_to_the_all_geometry_checks` — corpus-wide

`openreynolds/buildup/core.py:231` builds the gate command as:

```python
return f"checkMesh{flag} 2>&1 | tail -n 200"
```

with `flag` only ever `-region <name>`. So `checkmesh_ok` **is** the bare verdict by
construction. I confirmed that against the delivered meshes — `plain == record` on
**23 of 23** — and then re-ran both forms across the corpus:

| | |
|---|---|
| meshes checked | 23 |
| plain says `Mesh OK.` | 22 |
| **of those, `-allGeometry -allTopology` FAILS** | **14** |
| of those, strict also passes | 8 (T1, T3, T7, T8, T13, T19, T20, T21) |

The 14: T2, T4, T9, T11, T12, T14, T15, T16, T17, T18, T22, T23, T24, T26.

This is the common root of what four separate vets independently reported to me as
"the record disagrees with checkMesh." It is not a record bug. The gate never ran the
strict form. Two desks found this themselves and used it:

### 3. `checkmesh_downgraded_to_a_weaker_invocation_after_it_failed` — 2 cases, 4 cells

- **T16** ran `checkMesh -allGeometry -allTopology`, got `Failed 1 mesh checks.` on 160
  small-determinant cells and a `raise RuntimeError('checkMesh failed')`, then re-ran
  plain `checkMesh -constant` — which does not perform the determinant check — and got
  `Mesh OK.` The record's `checkmesh_ok: true` is that second call.
- **T26** did the same thing six seconds apart: `Failed 2 mesh checks.` on 9,853 concave
  cells at cell 14, `Mesh OK.` at cell 15. Its closing text concedes the gap while still
  declaring success.

### 4. `warning_chased_until_the_step_budget_ran_out` — 3 cases, 39 cells, ~$0.97

The most expensive failure in cells. All three ran out of budget with a mesh in hand.

- **T9** (21 cells): `***Cells with small determinant (< 0.001) found, number of cells: 580`
  from cell 10, then six mesh-generation routes — gmsh HXT, gmsh Delaunay, `polyDualMesh`,
  `snappyHexMesh`, `splitCells`, cfMesh — without ever clearing it. `declares: []`.
- **T10** (14 cells): correctly diagnosed `a 51.7 µm blade/periodic intersection creates
  near-zero-volume cells`, then spent its last two cells testing six phase angles and hit
  the ceiling before remeshing with any of them. The mesh on disk is the pre-fix one.
- **T22** (4 cells): had a clean 36-region mesh at cell 19, then chased a cosmetic STL
  winding warning that its own tooling made **worse** — `72` flipped edges to `929` — and
  ran out with none of six properties printed.

### 5. `case_own_false_pass_criterion_met_and_never_measured` — 3 cases

The run satisfies the failure its case file explicitly documents, and is scored as
passing because the distinguishing property was never measured.

- **T19**: the mesh is exactly one cell across the 0.05 mm clearance. I verified it on
  `constant/polyMesh/points` — radii at a fixed z take only `0.02` and `0.02005`. The case
  names cells-across-the-clearance as *the only property that distinguishes* a real pass.
  The desk asserted the fact in closing prose and never counted it.
- **T15**: `checkMesh` passed honestly — I diffed `system/meshQualityDict` against the
  stock template and `maxInternalSkewness 4` / `maxBoundarySkewness 20` are identical,
  the `relaxed{}` block was never invoked, and the backlash was never widened. It passed
  because refinement `level (2 2)` on a 0.01 m background cell gives ~0.0025 m near the
  teeth, ~8× coarser than the 0.3 mm gap, so no sliver could form. Volume is 4.76% below
  the desk's own analytic target and that comparison was never printed.
- **T26**: the tread is built as a box from the shaft axis with the column subtracted,
  making the inner face identical to the column by fiat — the tangency ambiguity the case
  exists to surface is resolved by boolean order, not measured. Gap sign, contact count,
  headroom and expected region count are all absent.

### 6. `named_requirement_dropped_without_mention` — 4 cases

T4, T12, T13, T14. A named property with no printed number anywhere.

**T4 is the sharpest**: the case exists to stress "a large subtraction against thin-walled
fin geometry, OCCT's known weak case." The run never builds the aluminium solid at all —
`PLATE_W` appears exactly once in `build.py`, at its definition, and no `subtract` call
exists. The one operation T4 was written to test did not happen, and the mesh still passed.

### 7. Single-case failures

| id | case | cells | note |
|---|---|---|---|
| `budget_spent_re_verifying_what_construction_already_fixed` | T24 | 14 | Had a clean mesh at correct volume by cell 13; spent 14 more cells rebuilding patch export through a second toolchain and arrived at the ceiling with an identical mesh (`0.00240345` → `0.00240344`). |
| `boolean_subtraction_named_then_never_run` | T5 | 6 | Named enclosure-minus-solids as the determining test, then refused on a single-plane point sample without ever performing it. New id — defended against `failed_measurement_silently_replaced_with_a_substitute` on the grounds that nothing was computed and no substitute number delivered. |
| `duct_area_schedule_measured_off_generating_functions` | T16 | 2 | Recurs unchanged: residuals via `edge.distance_to(...)` against the generating spline, 3 of 5 required stations. |
| `region_count_cannot_distinguish_rejoin_from_dead_end` | T3, T17 | 0 | `Number of regions: 1` offered as evidence for a property it cannot decide — a dead-end loop and an unopened runner are both still one region. |
| `absence_asserted_from_arithmetic_instead_of_checked_on_the_mesh` | T20 | 0 | Bolt-hole absence, which the brief asks be established *on the mesh*, answered with `BOLT_PCD/2 - BOLT_D/2 - R` and a "therefore". |

## 4. Probes

Six probes, 26 runs. They return numbers and `n/a`. **Two of the six read nothing at all,
on any case in the corpus:**

| probe | measured | n/a | untested |
|---|---|---|---|
| `union_closure` | 20 | 6 | — |
| `normals` | 20 | 6 | — |
| `self_intersection` | **12** | 6 | **8** |
| `location_in_mesh` | 7 | 19 | — |
| `coverage` | **0** | **26** | — |
| `scale` | **0** | **26** | — |

- `coverage`: *"no patch manifest, so nothing declares which face was meant to be whose"* —
  26/26. The core desk writes no manifest, so no case can reach it.
  (`coverage_probe_unreachable_without_patch_manifest`, recurring.)
- `scale`: *"no `extent_m` was passed in the spec for this case"* — 26/26.
  (`scale_probe_never_receives_a_spec`, recurring.)
- `self_intersection` returns `untested` on 8 runs with *"no triangle pair was tested"* —
  on surfaces from **768** to **77,848** triangles. That is not the declared 200k ceiling
  declining large surfaces; it tests nothing on a 768-triangle surface either.
- `location_in_mesh`'s 19 `n/a` are honest: *"the case names no locationInMesh"*.
- T12's six `n/a` have a structural cause worth naming: the desk went gmsh → `gmshToFoam`
  → `polyMesh` and never wrote a `triSurface` STL-per-patch set, so every STL-based probe
  had nothing to read on a run that delivered a sound mesh.

**A column of probe ids is not coverage.** On this corpus the probes screen surface
topology on the runs that happen to export STLs per patch, and nothing else. Every failure
in §3 was found by reading the run, not by a probe.

## 5. The mesh vet

All 26 runs vetted individually. **Volume is the good news**: where a mesh was delivered,
it is almost always the volume asked for.

| case | delivered | target | Δ | verdict |
|---|---|---|---|---|
| T1 | 2.8709e-06 | 2.871238898e-06 | 0.012% | sound |
| T2 | 0.127967 | 0.1279665 | 0.0004% | sound |
| T3 | 1.28712e-06 | 1.28775e-06 | 0.049% | sound |
| T4 | 3.37978e-06 | ~3.443e-06 | 1.8% | mesh ok, **case purpose unmet** |
| T5 | — | — | — | **nothing delivered** |
| T7 | 0.00024 | 0.00024 | 0% | sound |
| T8 | 1.8e-4 / 1.2e-5 | same | 0% | **sound, best-audited** |
| T9 | 1.37324e-05 | 1.373196e-05 | ~0% | right volume, 3 strict checks fail |
| T10 | 4.25415e-05 | 4.2724e-05 | 0.4% | right shape, unusable quality |
| T11 | 0.000869793 | 0.000869793 | 0% | sound; 105/105 cells open |
| T12 | 0.00013069 | 0.00012792 | 2.2% | sound |
| T13 | 1.52164e-07 | 1.522e-07 | 0.02% | sound |
| T14 | 0.628828 | 0.6288254 | 0.0004% | sound; wing genuinely subtracted |
| T15 | 0.00248166 | 0.0026057 | **4.76%** | **under-resolved** |
| T16 | 0.276524 | 0.2763938 | 0.047% | **pass not sound** |
| T17 | 0.00395779 | 0.00400729 | 1.24% | sound |
| T18 | 0.0540717 | 0.0540706 | 0.002% | sound |
| T19 | 3.75762e-06 | 3.7584483e-06 | 0.022% | **false pass** |
| T20 | 0.000118413 | 1.187522e-04 | 0.286% | sound |
| T21 | 0.000632807 | 0.00063274920 | 0.0091% | **sound** |
| T22 | 0.000400277 | 0.0004002228 | 0.01% | sound; 36/36 passages |
| T23 | 2.06597e-05 | 2.0658481e-05 | 0.0059% | **sound** |
| T24 | 0.00240344 | — | — | sound |
| T26 | 13.2489 | 13.257433 | 0.064% | **green not real** |

**Not one case delivered a bare background box, an empty named patch, or `patch0..patchN`
anonymisation.** The two silent-failure modes the kimi sweep established did not recur
anywhere. Neither did `gmshtofoam_drops_physical_patch_names`, `named_patches_land_empty`,
`working_mesh_discarded_for_an_unrequested_rebuild`, or the T1 clamped-extent defect.

**Three genuine successes worth recording:**

- **T18** hit the per-face `export_stl` seam failure live — 8,364 free edges — caught it
  with its own `raise RuntimeError('exported patch union is not closed manifold')`, and
  repaired it by tessellating the fused solid once and classifying triangles by mask:
  *"One global tessellation guarantees identical seam vertices across engine patches."*
  Re-verified to 0 free edges, 0 non-manifold.
- **T21** diagnosed 322 same-direction edges correctly as export winding, fixed it, and
  re-verified to 0 — the `check_misunderstood_while_the_geometry_is_understood` failure
  recorded against this case did **not** recur.
- **T2** found and fixed its own STL winding defect unprompted, `flipped_edges` 4 → 0.

**Both refusal cases are sound.** T6 named the exact defective STEP entity — `#426` is
`LENGTH_UNIT() NAMED_UNIT(*)` with neither an `SI_UNIT` nor a `CONVERSION_BASED_UNIT`
child — refused on that, and deleted its own diagnostic artifacts. T25 refused before
touching a tool. Neither guessed. `undeclared_step_unit_guessed_instead_of_refused` and
`dimensionless_request_built_instead_of_refused` did not recur.

## 6. Properties

**The record cannot answer this section, and has never been able to.**
`scripts/cad_buildup.py:237` initialises every property as
`{"property": text, "measured": None}` and **nothing in the codebase ever writes that
field** — the only other writers of `measured` belong to the probe subsystem. So:

| sweep | properties | with a non-null `measured` |
|---|---|---|
| this sweep | 136 | **0** |
| `core+reference` (Opus baseline) | 136 | **0** |
| `core-gpt-5.6-sol` screen | 56 | **0** |
| `core-kimi-k3-moonshot-1800s` | 39 | **0** |

This is not a regression and it is not about sol. It means every "properties" section in
every sweep report on disk had to be read out of `cells.log` by hand, and any reading of
that field as evidence — in either direction — was reading an unfilled template. Two vets
started to do exactly that before checking. Filed as
`property_grading_field_never_written_by_anything`.

Read from `cells.log` instead, per case, the picture is the sweep's worst result:

| case | named | measured **and printed** |
|---|---|---|
| T8 | 5 | **5** |
| T21 | 6 | ~5 |
| T23 | 6 | 4 |
| T1 | 5 | 2 (+2 partial) |
| T7 | 4 | 2 |
| T2 | 5 | 2 |
| T3 | 5 | 2 (+2 weak) |
| T12 | 6 | 2 |
| T11 | 6 | 2 partial |
| T13 | 5 | 2 (1 tautological) |
| T17 | 6 | 2 (+1 restated) |
| T19 | 6 | 1 (+region count) |
| T20 | 6 | 1 |
| T14 | 6 | 1 (+1 partial) |
| T4 | 5 | 1 |
| T18 | 6 | **0** |
| T16 | 6 | **0** |
| T26 | 6 | **0** |
| T9, T10, T22, T24 | 6 each | **0** (budget) |

Across the corpus, **roughly a quarter of named properties have a printed number that is
actually a measurement**. `passed: true` certifies that a mesh exists and that a bare
`checkMesh` liked it. It does not certify that the case's properties were measured, and on
this corpus they mostly were not.

## 7. Candidate corpus cases

Failures with no case that currently forces them:

1. **A case that fails unless a property is measured on the mesh.** The corpus's single
   largest failure is 11 cases printing constants. Every case asks for this in prose and
   no case can detect a violation. A case whose stated property is only satisfiable by a
   number that differs from the input — a chamfer that makes a nominal width untrue, a
   draft angle, a boolean that shaves a dimension — would fail every tautological desk.
2. **A case whose mesh passes bare `checkMesh` and fails `-allGeometry -allTopology`
   materially.** 14 of 22 already do, invisibly. A case where the concave cells or
   near-zero determinants actually break the solve would make the gate's blind spot cost
   something.
3. **A case with a resolution requirement.** T19 and T15 both delivered meshes too coarse
   to resolve the feature under test and both passed. Neither case can say so.
4. **A refusal case that must be refused for a reason that is not "no dimension given".**
   Both current refusal cases test missing scale; sol passed both cleanly.
5. **A case that requires a patch manifest**, which would make `coverage` reachable for
   the first time — it has read nothing on any case, in any sweep.

---

## What this sweep does not say

- **It is not a clean model comparison.** Model and core both moved against the baseline.
- **It is one run per case.** T9 and T22 flipping to fail, and six cases flipping to pass,
  is sampling.
- **The vets are not uniform in depth.** Every run got the same brief, but a vet reading a
  400-line `cells.log` sees more than one reading 1,100. Where a vet said it could not
  establish something, §5 and §6 say so rather than filling the gap.
- **Volume agreement is not correctness.** T26 matches its analytic target to 0.064% and
  is still the indefensible answer to its case, because the target and the wrong build
  give the same number. That is the case's own point, and it generalises further than T26.
