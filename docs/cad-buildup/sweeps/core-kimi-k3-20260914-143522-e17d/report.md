# core-kimi-k3-20260914-143522-e17d

## 1. The sweep

| | |
|---|---|
| label | `core-kimi-k3` |
| model | `kimi-k3` at `medium` effort, via Aster (`https://api.asterlab.ai/v1`) |
| core sha | `8a25b44` |
| cases | all 26, one run each |
| ended `done` | **2 / 26** (T7, T16) |
| passed | **2 / 26** |
| `checkmesh_ok` recorded | 16 / 26 |
| total cells | 503 |
| total spend | **$7.99** |
| wall clock | 2 h 39 m (14:35:22 → 17:14), 2 cases in parallel |

Stop reasons: `time` 12, `steps` 9, `no-progress` 3, `done` 2.

**This sweep takes no baseline and begins a new chain.** The preceding chain
(`core+reference-20260914-093903-3472`, through `2817999`) ran Opus 5, and a sweep whose
model moved is not comparable to the one before it. The core sha moved only to add an
`aster` preset and its price table; no desk logic changed between `2817999` and `8a25b44`.

For orientation only — **not a paired comparison, and no sign test is computable across
models** — the same 26 cases on Opus 5 gave 16/26 passed, 17 `done`, 2 `time`, 0
`no-progress`, at $26.36. Kimi K3 costs 30% as much and finishes 12% as often.

## 2. Contamination

**0 / 26 contaminated, 0 aborted.** The numbers below measure what they claim.

## 3. Failures, ranked by cost

Cost is cases hit x cells burned x spend. A case hit by two ids contributes its full run
cost to both, so the column ranks, it does not sum.

### 1. `background_box_delivered_as_mesh_under_passing_checkmesh` — 6 cases, 155 cells, $2.29
T14, T17, T19, T21, T24, T6. Every `snappyHexMesh` invocation died before castellation
completed, leaving the `blockMesh` background box in `constant/polyMesh` — and `checkMesh`
passes on it, because a plain hex block is trivially well-formed.

- T21: `cells: 4800`, `boundary patches: 1`, `Total volume = 0.000666701` — exactly the
  0.161x0.101x0.041 box, one `defaultFaces` patch where five named patches were expected.
- T24: `Total volume = 0.0052875` = 0.15x0.15x0.235, `Cells per refinement level: 0 25000`
  — zero cells above level 0, i.e. castellation never ran at all.
- T19: `Total volume = 0.00010547` against the desk's own printed target
  `vol mm3: 403.25` — the delivered mesh is ~261x the requested fluid volume.
- T6: `Unknown searchableSurface type triSurface` (the dict said `triSurface`, not
  `triSurfaceMesh`) on every attempt.

No probe fires on this. The probes read `constant/triSurface`, and in every one of these
runs the exported surface is clean — which is exactly why the failure is invisible to them.

### 2. `dict_completeness_discovered_one_fatal_error_at_a_time` — 5 cases, 118 cells, $1.80
T14, T18, T19, T21, T24. The desk hand-writes an OpenFOAM case dict, runs the mesher, reads
the one `FOAM FATAL IO ERROR` it gets, patches that single key, and runs again — never
validating the dict as a whole. T19 did this ten times in eleven cells:
`format` -> `object` -> `deltaT` -> `divSchemes` -> a `(`/`}` mismatch -> `refinementRegions`
-> `meshQualityControls` twice, 381 s of 996 s. T24's sixth attempt still died on
`Entry 'errorReduction' not found`. This is the engine of failure #1: the budget goes here,
so castellation never happens, so the box is what gets delivered.

### 3. `step_budget_exhausted_before_first_mesh` — 3 cases, 63 cells, $1.03
T3, T5, T15. The budget goes to build123d API friction before any mesher is called. T5 lost
ten of 29 cells to ten *different* wrong API guesses (`'bool' object is not callable`,
`cannot import name 'to_vtkpolydata'`, `unsupported operand type(s) for |`,
`Add must have an active builder context`). T3 the same (`name 'fuse' is not defined`,
`cannot import name 'Ring'`, `'Edge' object has no attribute 'point_at'`). Not one loop on
one error — a chain of distinct guesses, each costing a cell.

### 4. `named_patches_land_empty_under_a_passing_checkmesh` — 3 cases, 66 cells, $0.87
T18, T8, T9. The volume is right; the patch identity is not.
- T18: volume 0.054074408 vs the geometry's own 0.05426143814553113 (0.35% apart), but
  `inlet`, `outlet` and `walls` each carry **0 faces** — all 16320 boundary faces landed in
  `defaultFaces`.
- T9: volume 1.00945e-05 vs expected 0.00001010964516, but `gmshToFoam` wrote `patch1` and
  `patch2` where `endGapTop`/`endGapBottom` were intended.
- T8: `splitMeshRegions` printed `Only one region. Doing nothing.` on all five attempts, so
  the conjugate fluid/solid pair the case asks for is one undivided 39900-cell volume.

`gmshtofoam_drops_physical_patch_names` (T11, $0.44) is the same class by a third route:
the `.msh` carries `Surface 1 inlet ... Surface 4 honeycombWalls` and `gmshToFoam` still
wrote `patch0..patch7`, three of them empty.

### 5. `code_fence_emitted_as_text_never_invoked_as_a_tool_call` — 4 cases, 18 cells, $0.50
T10, T23, T4, T8. Corpus-wide 26 of 552 turns (5%) emitted no `tool_use` block, almost all
carrying a fenced ```python block in the text instead. Two distinct severities:
- **Fatal** (T4, T10, T23): never recovers. T23 burned all three turns this way and named
  the problem itself — *"I keep failing to actually invoke the tool — doing so now."* — then
  emitted the identical bare fence a third time. Run died at `no-progress`, 0 steps.
- **Doubling** (T8): 12 of 30 turns, but each is immediately followed by a turn executing
  the same code for real. Not fatal, but it halved the run's effective budget — 30 turns
  bought 18 steps, and the run ended one cell after `topoSet` finally succeeded.

This is **not** `fabricated_cell_output_after_fence`. Four classifiers checked that id
independently and all four rejected it: nothing is fabricated, because nothing is ever
executed to fabricate output for.

### Single-case failures
`working_mesh_discarded_for_an_unrequested_rebuild_that_never_recovers` (T26, $0.79) —
`Mesh OK.` on 24,527 cells at cell 17, then an unrequested tread-profile rebuild that
crashed snappy and settled into a mesh `checkMesh` fails.
`recorded_checkmesh_ok_disagrees_with_checkmesh_on_the_delivered_mesh` (T26) — see §5.
`diagnosed_volume_defect_repaired_but_fix_never_reverified` (T20, $0.35) — the desk
correctly diagnosed that its mesh covered only the gasket band, rebuilt, and the final
`checkMesh` it ran reports the identical defect *worse* (4.25039e-06 vs 5.43714e-06, against
an analytic 1.19e-4 it computed itself) and it never read the number.
`verified_mesh_overwritten_by_a_route_that_never_converged` (T2, $0.35) — a clean
`Total volume = 0.127974` mesh *with* a sphere patch at cell 11, discarded for a snappy route
that never converged; what ships is `Total volume = 0.128` exactly, three patches, no sphere.
`single_seed_keeps_one_of_many_disconnected_regions` (T22, $0.33) — 36 vanes partition the
annulus into 36 volumes and one `locationInMesh` seed was written, so snappy kept one
passage: 1.1465029e-05 against the desk's own 4.0028e-4, i.e. 2.9%.
`undeclared_step_unit_guessed_instead_of_refused` (T6, $0.27) — a refusal case. The desk
searched three times for a length unit, got nothing, wrote `SCALE = 0.001  # mm -> m` and
built on the guess.
`budget_spent_re_verifying_what_construction_already_fixed` (T9, $0.27),
`controldict_left_broken_across_three_blockmesh_reruns` (T12, $0.26),
`case_dicts_never_written_blocks_mesher_before_checkmesh` (T13, $0.23),
`checkmesh_output_sliced_past_the_line_the_property_asks_for` (T7, $0.17),
`checkmesh_fatal_on_missing_case_dicts` (T1, $0.28),
`duct_area_schedule_measured_off_generating_functions` (T16, $0.35),
`wedged_kernel_returns_polls_at_step_ceiling_with_no_output` (T25, $0.11).

## 4. Probes — what each measured, and on how many cases it could read anything

| probe | measured | n/a | why it could not read |
|---|---|---|---|
| `union_closure` | 15 | 11 | `constant/triSurface` does not exist |
| `normals` | 15 | 11 | same |
| `self_intersection` | 14 | 12 | same |
| `location_in_mesh` | 12 | 14 | the case names no `locationInMesh` |
| `coverage` | **0** | 26 | no patch manifest, on every single case |
| `scale` | **0** | 26 | no `extent_m` in the spec, on every single case |

**Two of six probes read nothing on any case in the corpus.** `coverage` and `scale` are
not screening anything — they are not instruments with a null result, they are instruments
that were never given an input, 26 times out of 26. A column of six probe ids in the table
reads as coverage; it is four at best.

And the four that do read, read the **exported surface**, never the delivered mesh. On
T21, T24, T6, T14 the surface probes are all clean while the delivered mesh is a bare box.
On T2 and T9 they read STL files from a route the desk had already abandoned. Not one of
the 35 findings in this sweep was caught by a probe. Every one came from the mesh vet.

Known probe defects seen again: `self_intersection` reports `triangles: 0` alongside a
nonzero `pairs_tested` on T9, T19, T20, T22, T24
(`self_intersection_probe_returns_measured_on_zero_triangles`), and returns suspiciously
small test counts on large surfaces (135 pairs on T16's 12,560 triangles).

## 5. The mesh vet — is the delivered mesh the volume the case asked for?

| verdict | n | cases |
|---|---|---|
| **No mesh delivered** | 8 | T3, T4, T5, T10, T12, T15, T23, T25 |
| **Delivered mesh is not the asked-for volume** | 11 | T2, T6, T8, T13, T14, T17, T19, T20, T21, T22, T24 |
| **Right volume, wrong patch identity** | 3 | T9, T11, T18 |
| **`checkMesh` fails on recheck** | 1 | T26 |
| **Validity unknown** | 1 | T1 |
| **Sound** | **2** | T7, T16 |

**This is the section that matters.** 16 runs recorded `checkmesh_ok: true`. Of those, the
two that passed (T7, T16) are sound; **all 14 of the others deliver a mesh that is wrong in
volume, wrong in patch identity, or fails on a fresh check.** `checkMesh` green is not
evidence the right volume was built, and on this corpus it was wrong about that 14 times.

On T26 specifically, the record's `checkmesh_ok: true` does **not** describe the mesh on
disk. I re-ran the harness's own bare `checkMesh` against every case directory in the
sweep: 15 of the 16 still return `Mesh OK.`, and T26 returns
`Failed 1 mesh checks.` on `***Max skewness = 12.6324, 7 highly skew faces`. So the flag is
sound as an instrument in 15 cases and stale in one — worth knowing, but it is not the
general defect; the general defect is that a true `checkMesh` verdict does not answer the
question the corpus is asking.

T1 is the one case where validity is genuinely unknown rather than wrong: `blockMesh` wrote
480 cells, but `checkMesh` died on a missing `system/fvSchemes` and the desk printed only
`r.stdout[-1400:]` — the fatal error was on stderr. No verdict was ever readable.

## 6. Properties — measured *and printed*, per case

Across the corpus this is close to total. `record.json` carries every case's named
properties with `measured: null`, and the driver never fills them in, so the only evidence
is a printed number in `cells.log`.

The corpus names **136 properties across 24 cases** (the records total only 119, because
three of them lost theirs — see below). T6 and T25 name none, correctly: they are the two
refusal cases. Of those 136, **four were measured and printed**, both runs that passed.

- **0 printed** on every case except T7 and T16 — including all 16 runs that recorded
  `checkmesh_ok: true`.
- **T16** (passed): 1 of 6 — `Number of regions: 1 (OK).` — plus one partial, the volume
  check `Total volume = 0.274741` against `analytic volume 0.275573000997305`. The three
  headline quantities the case names (hub/tip radius at five stations, annulus height and
  flow area at each, nozzle:inlet area ratio) were **never computed at all**, on a run the
  corpus scores as a clean pass.
- **T7** (passed): 2 of 4 on the delivered mesh — zone sizes and positions
  (`heater 1000 ... 8e-06 (0.008 0.02 0) (0.028 0.04 0.02)`). Cavity dimensions and the
  single-patch claim were printed only against a superseded 3750-cell mesh, and the named
  **centroids were never computed**.

**The record under-reports this, and the report nearly repeated the error.** T4, T10 and
T23 record `properties: []`, which reads as "this case names none". It is not: `load_prompts`
returns 5, 6 and 6 for them. `entry.properties` is assigned only on the normal completion
path (`cad_buildup.py:351`), while `record.save` also fires per-turn (`:263`) and per-step
(`:277`) — so a run the watcher kills never reaches the assignment and its record loses the
case's properties silently. Filed as
`killed_run_record_loses_the_cases_named_properties`. Any sweep with killed runs will
understate this section until that is fixed.

A recurring mechanism: the desk slices its own `checkMesh` output past the line the property
needs. T7 filtered on `re.search(r"heater|cooler|Mesh OK|cells", l)`, which cannot match the
bounding-box or patch-topology lines; T26 used `rc.stdout[-900:]`; T11 used
`out[find("Mesh stats"):+1500]` and cut off before `Total volume`.

## 7. Candidate corpus cases — failures with no case covering them

1. **A model that narrates a cell instead of calling the tool.** Four cases hit it and three
   died of it, but no case *tests* for it. It is a harness-observable property (`block_types`
   without `tool_use`), and nothing in the corpus asserts on it.
2. **A mesher that exits non-zero while leaving a well-formed background box behind.** The
   dominant failure in this sweep, and no case is designed to catch a mesh that is valid and
   wrong. A case whose pass criterion is "delivered volume within X% of a stated analytic
   volume" would have caught 11 runs that currently record `checkmesh_ok: true`.
3. **A named patch that exists in the dict and is empty in the delivered `boundary`.** Three
   routes produce it (snappy pre-snap, `gmshToFoam`, `splitMeshRegions`) and no case asserts
   that a named patch has nonzero faces.
4. **A case supplying `extent_m` and a patch manifest.** `scale` and `coverage` read nothing
   on 26 of 26 cases. Until some case supplies those inputs, two of the six probes cannot be
   said to work or not work — they are untested, not passing.
5. **A wedged kernel.** T25 spent 720 s of its 978 s on three consecutive 240 s step
   ceilings with empty output, and nothing distinguishes that from slow progress.

---

*35 findings over 26 cases, one `cad-supervisor` classification per run. Findings in
`findings.jsonl`. The ranked list above is a decision input; what to add is not decided here.*
