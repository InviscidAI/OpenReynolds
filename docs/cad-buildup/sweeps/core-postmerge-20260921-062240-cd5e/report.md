# core-postmerge-20260921-062240-cd5e

## 1. The sweep

gpt-5.6-sol at medium, core `185db0d`, 26 cases x 1, `--parallel 2`.
Baseline: `core-shipped-20260919-124001-f33d` (gpt-5.6-sol at medium, core `66df673+dirty`).

**21 of 26 survive the vet; the harness scored 24 of 26.** Lead with the first number.
The gate's `passed` means the run ended `done` (or correctly `refused`) and a bare
`checkMesh` liked the mesh. Three of the runs it passed do not deliver the geometry
their case asked for -- T10, T19 and T9 -- and are broken in section 5 below.

The sha carries `+dirty` because this sweep's own output directory and the preceding
two-case smoke were untracked in the tree when it started. No tracked file differed;
`git status --short` showed only `??` entries for the two sweep directories. The core
under test is exactly the merge commit.

| | this sweep | baseline |
|---|---|---|
| harness pass | 24/26 | 22/26 |
| survives the vet | **21/26** | not vetted to this depth |
| cells | 387 | 398 |
| spend | $9.54 | $9.67 |

Paired comparison: 5 cases used fewer cells, 5 more, 16 within their own 30% noise band.
**Sign test over the cases that moved: p = 1.000.** The verdict is that line, not any row.
Nothing about the desk's behaviour moved across the merge, which is the result a merge
should produce.

Two cases flipped fail -> pass against the baseline (T11, T19). Neither is an improvement:
T19 is the identical one-cell-thick clearance the baseline already recorded, now scored
green because nothing measured the distinguishing property; T11's geometry does hold, but
it held on the vet's measurement, not on anything the run established. On a p = 1.000
corpus a single case's flip is sampling.

## 2. Contamination

**0 of 26.** No run reached a house path. The numbers measure what they claim.

## 3. Failures, ranked by cost

Cost is cases hit x steps burned x spend. Ranked because cost is what an addition buys back.

| rank | failure_id | cases | steps | spend | what it is |
|---|---|---|---|---|---|
| 1 | `printed_number_is_a_literal_not_a_measurement` | T1, T12, T14, T15, T18 | 65 | $2.11 | A `requested/measured` pair whose two sides derive from the same constant. T18 is the clearest specimen: `clear gap measured/requested: 6.000/6.000 mm` is `fin_pitch - fin_thickness` where `fin_pitch = fin_thickness + fin_gap`. It prints `fin_gap` whatever was built. |
| 2 | `named_property_never_measured_though_geometry_holds` | T11, T26 | 36 | $0.82 | The property the case wrote to expose an ambiguity is never printed in any form; the geometry turns out right when someone else measures it. |
| 3 | `region_count_cannot_distinguish_rejoin_from_dead_end` | T3, T17 | 28 | $0.82 | `Number of regions: 1` offered as connectivity evidence for the exact property the case states a region count cannot decide. Third recurrence for T17. |
| 4 | `clearance_spanned_by_one_cell_and_never_counted` | T9, T19 | 17 | $0.57 | **Breaks both runs.** A small clearance meshed with exactly one cell across it, everywhere, and never counted. checkMesh cannot see this. |
| 5 | `named_feature_never_built_as_a_surface` | T10 | 22 | $0.58 | **Breaks the run.** The blunt trailing edge has no wall in `constant/polyMesh/boundary` at all -- the two TE points are never joined by an edge -- and the desk printed a thickness for it anyway. |
| 6 | `gear_flanks_touch_where_a_gap_was_requested` | T15 | 15 | $0.47 | Flanks in contact where 0.3 mm of backlash was requested. The desk checked overlap *volume*, which is ~0 for touching solids too. |
| 7 | `case_dicts_never_written_blocks_mesher_before_checkmesh` | T5 | 13 | $0.25 | `fvSchemes`/`fvSolution` never written; snappyHexMesh died fatally before castellation. |
| 8 | `named_requirement_dropped_without_mention` | T16 | 19 | $0.40 | None of the three-station radii, areas or ratios printed; only the spline residual at its own control points. |
| 9 | `measured_off_the_generating_construction_not_the_delivered_mesh` | T2 | 16 | $0.46 | Sphere diameter read off the CAD primitive and an STL array built from the same literal, not the meshed patch the case names. |
| 10 | `named_flank_patch_merged_with_unrelated_closure_face` | T13 | 16 | $0.35 | **New id.** Each named flank patch is one true flank plus one unrelated closure face, so a literal "gap between the two flank patches" measurement samples a mix. Average gap is right (0.19992 mm vs 0.2 requested). |
| 11 | `identical_meshing_command_rerun_for_reproducibility_not_geometry` | T15 | 1 | $0.47 | **Harness, not desk.** 235 s of meshing re-run verbatim for the same 575,242 cells because the reproducibility checker rejected a cell over a leaked `time` import. 23% of that run's budget, immediately before it timed out. |
| 12 | `defeature_call_cut_without_printing_anything` | T5 | 1 | $0.25 | `BRepAlgoAPI_Defeaturing()` ran 240 s and returned empty output; the fillet removal never happened. |

### A clustering problem in the registry

`case_own_false_pass_criterion_met_and_never_measured` arrived at this sweep carrying two
different causes, and three vets noticed independently:

- the property was not measured **and the geometry is wrong** (T19, T9, T15);
- the property was not measured **and the geometry is fine** (T26, T11).

An addition that closes either closes neither other, so they are split here into
`clearance_spanned_by_one_cell_and_never_counted` / `gear_flanks_touch_where_a_gap_was_requested`
and `named_property_never_measured_though_geometry_holds`. The old id's registry text also
asserts the geometry holds on re-vet across its four prior instances; T15 and T19 are the
first counter-examples and belong in that text as such.

## 4. Probes

Readings, not findings. Six probes over 26 runs:

| probe | measured | n/a | why n/a |
|---|---|---|---|
| `coverage`, `union_closure`, `normals`, `self_intersection` | 22 | 4 | no `constant/triSurface` on disk (T6, T7, T8, T25) |
| `scale` | 22 | 4 | same, plus cases with no stated `extent_m` |
| `location_in_mesh` | 15 | 11 | the case names no seed point |

**Every reading in the sweep is benign.** 0 open edges, 0 flipped edges, coverage ok,
scale ratio 1.0, 0 crossing pairs -- no probe produced a non-zero reading on any case.
Read as a column of ids this looks like coverage; it is not. Five of six probes reported
"nothing wrong" on an axis where nothing was wrong, and none of them fired on any of the
three runs the vet broke. The probes did not catch T10, T19 or T9.

**T7 and T8 are green, `checkMesh`-passing, with a delivered mesh, and all six of their
probes read `n/a`** -- `constant/triSurface` never exists, so the probe layer is
structurally blind to them. T6 and T25 are also all-`n/a`, but correctly: nothing was built.

## 5. The mesh vet

Every run was vetted, including the green ones. Three green runs do not survive.

**Broken:**

- **T10** -- the blunt 0.4 mm trailing edge is not under-resolved, it was never built. `build.py` wires the pressure-curve TE point to the outlet-plane corner at s=0 and the suction-curve TE point to the other corner one pitch away (the neighbouring blade's); the two are never joined. No wall in `constant/polyMesh/boundary` could be the cap. The periodic faces also differ by 2.0% (0.0022107 vs 0.0022559 m2), unchecked.
- **T19** -- one cell across the 0.05 mm clearance. Point radii at six angular positions take only the values 0.02 and 0.02005 m; nothing in between, anywhere. Volume is right to 0.01%. The desk read the single-layer signature as confirmation: *"the 0.05 mm annulus appears as two nearly coincident circular lines, as expected"*.
- **T9** -- one tet edge across the 1.0 mm clearance, everywhere. Radii cluster only at 0.022 and 0.023. Volume right to 0.03%. The baseline managed two cells, so this run is worse on the axis the case names.

**Non-passes, both open rather than silent:**

- **T5** -- delivered mesh is the raw `blockMesh` box, 82,720 hexes, volume 80,237 against the asked-for 39,043. No `device` patch exists. Two causes: a 240 s defeaturing call that returned nothing, and missing `fvSchemes` killing snappyHexMesh.
- **T15** -- has a mesh `checkMesh` likes, but the gear flanks touch where 0.3 mm was requested: 30 STL vertices at float-exact zero distance, corroborated at 1e-17 m in the desk's own trace.

**Correct refusals, both earned:** T6 refused because the STEP genuinely declares no length unit (`#426 = ( LENGTH_UNIT() NAMED_UNIT(*) );`, confirmed on disk), nothing built. T25 refused on turn one naming the missing specification, before any tool call; the workspace holds no `build.py`.

**Survive the vet (21):** T1, T2, T3, T4, T6, T7, T8, T11, T12, T13, T14, T16, T17, T18, T20, T21, T22, T23, T24, T25, T26.

Several were confirmed on evidence the run itself never produced: T8's interface by a
point-cloud comparison (768/768 faces, 825/825 points, max deviation 0.0), T3's four
branch-and-rejoin loops by counting closed wall components, T16's mid-duct bulge by the
core cowl at 0.3097 m where a linear loft gives 0.265, T22's 36 passages by `Number of
regions: 36`, T11's honeycomb by a 0.795 open-area ratio at a mid-core slice.

## 6. Properties

**Not graded. 73 of 73 carry `measured: null`.** The records hold only `property` and
`measured`; no `source`, `verdict` or `desk` field exists on any of them, so
`cad_supervise.py grade` never ran on this sweep. This section therefore reports that the
runs were never graded rather than inferring anything from `cells.log`. The measurements
below come from the vets reading delivered artifacts directly.

Across the cases where a property could be established, the two axes diverge constantly,
and the divergence is the finding:

- **does it hold** (geometry): holds on nearly every green case, including ones the desk never checked.
- **did the desk measure it** (the run): `printed-from-input` or `absent` on the large majority.

`printed-from-input` is the corpus's most common failure and no instrument catches it: a
`requested / measured` pair whose two sides derive from the same constant cannot disagree
with itself, whatever was built. T18, T15, T1, T14, T12 and T11 all print one.

Genuine measurements do exist and are worth naming: T8 ran `checkMesh -region` per region
and asserted on the result; T24 derived the 15 deg tangent offset by comparing built
direction against a tangent computed from position; T14 compared a pre-mesh CAD volume
against an independently-meshed `checkMesh` volume (0.006% apart); T2 captured the real
100% layer line from `log.snappyHexMesh`; T22 quoted checkMesh's own 36-region count.

## 7. Candidate corpus cases

A failure without a case is a case, not a tool.

1. **A run whose probes are structurally blind.** T7 and T8 pass green with a delivered mesh and zero probe readings, because no `constant/triSurface` is written. Nothing in the corpus currently tests a meshing route that bypasses patch STL export, so nothing measures how often the instrument layer is simply absent.
2. **A named patch that is not what its name says.** T13's `neck_flank` and `lid_flank` each merge a true flank with an unrelated closure face. No case currently asks whether a patch contains only the surface its name claims.
3. **A harness cost the corpus cannot see.** T15 lost 23% of its budget re-running an identical mesh because the reproducibility checker rejected a cell over a leaked import. No case measures budget lost to harness-imposed re-work rather than to geometry.
