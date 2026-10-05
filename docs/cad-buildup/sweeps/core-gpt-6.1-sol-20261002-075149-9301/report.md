# core-gpt-6.1-sol-20261002-075149-9301

The first corpus on `gpt-6.1-sol`, run through the `reynolds` preset the way a hosted study
reaches it: the agent's Responses adapter, foamd's `/v1/llm` proxy (run locally,
`scripts/local_llm_proxy.py`), Azure Foundry. Every run was vetted individually against its
delivered mesh, the green ones included.

## 1. The sweep

**22 of 26 survive the vet** (20 sound meshes and 2 correct refusals). The harness passed 24 of
26; the vet broke two of those, T9 and T24, both green under `checkMesh`.

| | |
|---|---|
| label | `core-gpt-6.1-sol` |
| model | `gpt-6.1-sol` at `medium` (mesher effort `medium`), Responses API via the `reynolds` proxy to Azure Foundry |
| core sha | `b54022f` (branch `gpt-reynolds-model`) |
| cases | all 26, one run each, `--parallel 2` |
| ended `done` | 22 / 26 (T6, T25 `refused` as expected; T10 `refused`, T15 `time`) |
| passed, by the harness | 24 / 26 (23 / 25 as the table prints it, T5 held out) |
| **passed, and the vet agrees** | **22 / 26** |
| total cells | 246 (T5 excluded) / 262 (all 26) |
| total spend | **$3.05** for all 26 ($2.85 as the table prints it, T5 excluded) |
| wall time | 59 min |

### Against the baseline

`core-gpt-5.6-sol-20260916-025729-2a7f` (`gpt-5.6-sol` at `medium`, core `dc997cc`), which
vetted to 16 / 26:

```
23/25 passed, 246 cells, $2.85      (T5 unpaired, held out by the table)
before: 21/26 passed, 437 cells, $11.29

16 cases used fewer cells, 1 more, the rest within 30% of their own cell count.
sign test over the cases that moved: p = 0.000
```

Vetted against vetted: **22 / 26 now, 16 / 26 before.** Both model and core moved, so this
is not a single-variable result and the table says so. The core change that matters is
`b54022f`: `read_file` returned a render as raw PNG bytes whenever a model passed
`offset: 0` with a large `limit`, which GPT models do and Claude models do not. The baseline
ran without that fix, so gpt-5.6-sol may have looked at some renders blind; how much of the
difference is that, and how much is the model, cannot be separated from these two sweeps.

Spend is a quarter of the baseline's, at half the token price: the cache served most input
(across the integrated runs, under 0.1% of input tokens were uncached), and the desk used
56% of the cells.

## 2. Contamination

**1 / 26, a false positive — and the sweep is reported, not voided, on that reading.** The
harness flagged T5: its contamination grep matched `grep -n fillet /work/.toolbox/b123d_api.md`
in `cells.log` line 55. That string is line 16 of `b123d_api.md`, the reference document the
case hands the desk (`given: ["b123d_api.md", "cad_export.py"]`, in `t5/.reference/`), which
the desk grepped. The supervisor's vet confirmed independently: `build.py` and `replies.jsonl`
contain no `toolbox`, the workspace has no `.toolbox`, and the desk's only reads were the STEP
and the two given files. Isolation held. The skill's rule is that any contamination voids
the sweep; the person running it chose to vet all 26 with this exception on record.

The defect is the instrument's: it greps for house paths that the house's own reference
documents print. No earlier model printed that document in full, which is why it never fired.

## 3. Failures, ranked by cost

Cost here is cells and dollars on the cases hit; a finding that cost nothing but hid a broken
result is marked **hid a failure**.

1. **`valid_cad_solid_refused_after_gmsh_failed_on_own_cap_surfaces`** — T10, 11 cells, $0.13,
   nothing delivered. New. The desk built a valid closed passage solid (4.41e-5 m³, 0 free
   edges), gmsh then failed on the hub and shroud caps it had drawn as BSplines
   (`Warning : 450 elements remain invalid in surface 2`,
   `Error : Invalid boundary mesh (overlapping facets) on surface 11`), and after three
   rebuilds of those caps it declared the case unanswerable. The request is satisfiable; the
   baseline meshed it to 0.4%. No probe fired: three of them were over their triangle ceilings
   at 622,136 triangles.
2. **`single_boolean_cell_consumed_the_time_budget_before_any_mesh`** — T15, 6 cells, $0.08,
   stopped on `time`, nothing delivered. New. One `enclosure.cut(pinion, wheel)` on
   1,200-face gear solids ran ~740 s of the 910 s budget (a 240 s window plus a ~500 s poll),
   leaving a valid surface and a gmsh `.msh` but no OpenFOAM mesh. The baseline passed this
   case with a gear union built in seconds.
3. **`checkmesh_fatal_on_missing_case_dicts`** — T12, T14, T16; 5 cells, $0.32. Recurs.
   `checkMesh` run with only a `controlDict`, died on missing `fvSchemes`/`divSchemes`, fixed
   on the next cell each time.
4. **`defeature_produces_invalid_shape_and_run_continues`** — T5, 3 cells. Recurs. Defeaturing
   returned a shape with `is_valid False`; `fix()` did not clear it; the desk meshed anyway,
   disclosed it, and the mesh is sound.
5. **`patch_mapping_sampled_point_off_trimmed_face`** — T12, 1 cell. New: the patch map sampled
   the parametric midpoint of a trimmed face, which fell in a hole. Recovered next cell.
6. **`assumed_closure_adds_oil_outside_requested_envelope_and_volume_never_compared`** — T9,
   **hid a failure**. New. The desk capped the bearing with 1 mm of oil beyond each ring end
   (`H_ENCLOSURE = 0.012  # assumed`), so the delivered oil is 1.5764e-5 m³ against
   1.0110e-5 m³ for the bore swept minus the spinner (+56%). It printed its CAD volume and
   never compared it to the target the case names. Bare `checkMesh`: OK.
7. **`duct_angle_referenced_to_embedded_inner_point_not_the_wall`** — T24, **hid a failure**.
   New. Each inlet duct was set 15° off the tangent at a point 50.3 mm from the axis, inside
   the vessel, rather than where it meets the wall; on the mesh the inlets come in 36° off the
   tangent at the wall and 52° at the port face. The swirl angle is the case's purpose and was
   not built. The desk printed 15° from the vectors it constructed the ducts with.
8. **The measured-from-construction family** — no cells, but it is how 6 and 7 stayed hidden,
   and how T15's backlash went unnoticed (the desk printed 0.300 mm from its constant; its own
   solid-distance measurement two lines earlier read 0.112 mm):
   - `measured_off_the_generating_construction_not_the_delivered_mesh` — 10 cases
     (T3, T9, T12, T13, T14, T16, T17, T23, T24, T26); T12's vetter filed it as
     `duct_area_schedule_measured_off_generating_functions`, folded in here as the same cause.
   - `printed_number_is_a_literal_not_a_measurement` — 8 cases (T1, T4, T11, T15, T18, T19,
     T20, T21).
   - `named_property_never_measured_though_geometry_holds` — 9 cases (T5, T9, T11, T16, T17,
     T18, T19, T24, T26).
   - `repeated_feature_counted_on_solids_not_mesh` — T11, T22.
   - `passage_width_property_conflated_with_vane_width` — T22.
   - `absence_asserted_from_arithmetic_instead_of_checked_on_the_mesh` — T20.
   - `region_count_cannot_distinguish_rejoin_from_dead_end` — T3.

   The first two ids describe one behaviour — a requested/measured pair whose two sides come
   from the same constant — and have done since the sweeps that coined them. They are kept
   apart here because `findings.jsonl` on earlier sweeps uses both, and merging them is a
   decision for the corpus, not for this report.
9. **`refusal_reason_not_persisted_in_run_record`** — T10, an instrument defect. A declared
   refusal's `result.error` never reaches `record.json` (`declares: []`), and turn 14 carried no
   text, so why T10 stopped can be inferred from the cells but not read back.

## 4. Probes

Readings, not findings. On the 22 cases that delivered a surface, `union_closure`, `normals`,
`coverage`, `scale` and `self_intersection` read clean wherever they could read at all; no
reading anticipated either silent failure (T9's over-volume and T24's angle are both
properties of a closed, consistently wound surface). `location_in_mesh` read on the
snappyHexMesh routes and was `n/a` on the gmsh, blockMesh and hand-written routes, which are
most of this sweep. On T10, `union_closure`, `normals` and `self_intersection` were over their
triangle ceilings (622,136 triangles) and `coverage` skipped. `scale` read on every case that
states a dimension, within 1.05.

## 5. The mesh vet

| case | ended | cells | $ | delivered (m³) | target (m³) | Δ | verdict |
|---|---|---|---|---|---|---|---|
| T1 | done | 5 | 0.08 | 2.87115e-06 | 2.87124e-06 | -0.003% | sound |
| T2 | done | 7 | 0.09 | 0.127967 | 0.1279665 | +0.0004% | sound |
| T3 | done | 19 | 0.20 | 1.66168e-06 | 1.66156e-06 (own closed surface) | +0.007% | sound |
| T4 | done | 11 | 0.11 | 3.47968e-06 | 3.4798e-06 | -0.03% | sound |
| T5 | done (flagged) | 16 | 0.20 | 0.0692473 | 0.0692474 | <0.001% | sound |
| T6 | refused | 3 | 0.01 | — | refusal (no length unit) | — | **correct refusal**; baseline guessed mm |
| T7 | done | 5 | 0.05 | 0.00024 | 0.00024 | 0% | sound |
| T8 | done | 9 | 0.13 | 1.8e-4 / 1.2e-5 | same | 0% | sound |
| T9 | done | 11 | 0.11 | 1.5764e-05 | 1.0110e-05 | **+56%** | **false pass** — oil outside the bearing |
| T10 | refused | 11 | 0.13 | — | ~4.27e-05 | — | **nothing delivered** |
| T11 | done | 10 | 0.13 | 0.000869793 | 0.000869793 | 0% | sound |
| T12 | done | 11 | 0.11 | 1.33262e-4 | 1.27919e-4 | +4.2% (declared bore and ports) | sound |
| T13 | done | 10 | 0.16 | 4.27065e-07 | 4.27141e-07 | -0.018% | sound |
| T14 | done | 8 | 0.12 | 0.628826 | 0.628826 | ~0% | sound |
| T15 | time | 6 | 0.08 | — | ~2.63e-03 | — | **nothing delivered**; backlash built at 0.11 mm against 0.3 |
| T16 | done | 10 | 0.10 | 0.276292 | 0.276391 | -0.036% | sound (baseline: pass not sound) |
| T17 | done | 17 | 0.20 | 0.00399779 | 0.0040073 | -0.24% | sound |
| T18 | done | 10 | 0.11 | 0.0540713 | 0.0540706 | +0.001% | sound; all six patches populated |
| T19 | done | 13 | 0.19 | 3.75845e-06 | 3.7584483e-06 | 0.0001% | sound; ~8 cells across the clearance (baseline: false pass at 1) |
| T20 | done | 12 | 0.09 | 1.18413e-4 | 1.187522e-4 | -0.29% | sound |
| T21 | done | 10 | 0.10 | 6.34639e-4 | 6.3455e-4 (with the vent pocket) | +0.014% | sound |
| T22 | done | 8 | 0.10 | 0.000400168 | 0.000400227 | -0.015% | sound; 36 / 36 passages |
| T23 | done | 14 | 0.18 | 2.06192e-05 | 2.0661e-05 | -0.20% | sound |
| T24 | done | 13 | 0.16 | 0.00233146 | ~2.38e-3 | chamber -0.03% | **case purpose unmet** — swirl 36–52° against 15° |
| T25 | refused | 0 | 0.003 | — | refusal (no dimensions) | — | **correct refusal** |
| T26 | done | 13 | 0.12 | 13.2487 | 13.2552 | -0.049% | sound; 20 / 20 treads welded (baseline: green not real) |

What could not be established is in each run's vet: chiefly the desk's own assumptions where
a request leaves a dimension free (T7 zone placement, T11 edge clipping, T12 bore and ports,
T13 thread profile, T16 interpolant, T17 the 90 mm reading, T21 vent depth, T26 the hidden
overlap that resolves the tangency).

`-allGeometry`, reference only: run on 20 of the 22 delivered meshes (not on T19 or T23;
on T8 it exits FOAM FATAL on the desk's stub per-region dictionaries). Clean on 11; it flags
0.0005% to 4.3% of cells on the other 8 (T12 4.3%, T4 2.6%, T2 2.5%, T17 0.54%, T14 0.25%, T5 0.12%, T26
0.015%, T11 one cell). No case is failed on it.

## 6. Properties

Graded by the supervisor into each `record.json` with `cad_supervise.py grade`: 73 named
properties across the 24 cases that name any.

- **Does the property hold** (the geometry): **65 hold, 5 differ, 3 unmeasurable.** The five
  that differ are T9's volume, T15's backlash, T24's swirl angle and chamber/duct volumes, and
  T5's body count (the three touching solids fused to one surface, with no volume lost).
- **Did the desk measure it** (the run): **21 printed, 31 printed-from-input, 21 absent.**

Most properties hold on a mesh the desk never measured: 52 of 73 are printed-from-input or
absent. Three of the five that differ were printed-from-input, and that is how they passed:
the desk's number agreed with its constant because it was its constant.

## 7. Candidate corpus cases

- **An inlet angle referenced to a curved wall** (T24's failure): the current case catches
  it only because the vet measured it; a case whose pass criterion is the angle at the wall,
  stated that way, would make it a gate failure rather than a vet finding.
- **A boolean on high-face-count solids under a time budget** (T15's failure): no case
  exercises a CAD operation slow enough to eat the budget by itself.
- **A request whose natural enclosure invites extra fluid** (T9's failure): a case with an
  explicit envelope and a named volume target, where the obvious closure overshoots it.
