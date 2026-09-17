# core+cad_export-20260917-022129-dd05

The addition is `cad_export.export_patches`, handed to the core desk as its second
reference file. **23 of 26 survive the vet.** Every run was vetted individually, including
all 23 that ended green.

**Revised 2026-09-17, and the revision is the more useful half of this report.** The first
pass broke three runs -- T4, T10, T26 -- and all three findings were wrong. They graded the
case files' `## What this catches` sections, which are provenance notes citing the plan,
against runs whose delivered geometry matched the request. §9 has them. The number above
was 20 before that was caught, and **not one of the six failures in this sweep turns out to
be the desk's**: two are the harness's, one is a case that has never been satisfiable, and
three were my own reading.

**This sweep carries two changes and cannot separate them.** The addition, and the
`-allGeometry` paragraph added to the brief at `115da52` the day before. The chain was
already broken when this sweep started and was run anyway, on the instruction that
serialising costs more than the attribution is worth. What follows says which claims
survive that and which do not.

## 1. The sweep

| | |
|---|---|
| label | `core+cad_export` |
| model | `gpt-5.6-sol` at `medium`, unchanged from the baseline |
| core sha | `d074a46` |
| cases | all 26, one run each, `--parallel 2` |
| ended `done` | 22 / 26 |
| `checkmesh_ok` | 23 / 26 |
| passed, by the gate | 23 / 26 |
| **passed, and the vet agrees** | **23 / 26** |
| total cells | 404 |
| total spend | **$9.76** |

Broken by the vet: **none**. Not passed: **T12** (§3.1, our gate), **T15** (§3.4, our
parameter name), **T5** — a case no run has satisfied in ten attempts across every sweep on
record, which is now a question about the case rather than about any desk.

### Against the baseline

`core-gpt-5.6-sol-20260916-025729-2a7f` (`gpt-5.6-sol` at `medium`, core `dc997cc`):

```
23/26 passed, 404 cells, $9.76
before: 21/26 passed, 437 cells, $11.29

7 cases used fewer cells, 4 more, the rest within 30% of their own cell count.
sign test over the cases that moved: p = 0.549
```

**Read the sign test: p = 0.549 is not a cell-count result.** The corpus is the unit and
four cases moving one way against seven is sampling.

**And the vetted comparison does not hold either, which the first draft of this report got
wrong.** The baseline's 16 was scored through the same over-strict lens this report has
since withdrawn -- two of its five broken runs were T4 and T26, on the same reasoning -- so
16 → 23 is not a measurement, it is two different methods. Re-scoring the baseline against
request text alone is what would make the pair comparable, and until that is done **this
sweep has no pass-rate result at all.** What it has is §4.

Three of the four runs that exhausted the step budget in the baseline — T10, T22, T24 —
finished this time, and T9 with them. T22 went from 27 cells to 11.

## 2. Contamination

**0 / 26.** No aborted runs, and `cad_export.py` was greped, imported and called without
a single `house_names` hit, which is the half of the addition that had to work before
anything else could be measured.

## 3. Failures, ranked by cost

### 3.1 `coverage_gate_warns_regardless_of_the_probes_own_verdict` — 19 declares, 1 run

**Ours, and the most expensive thing in the sweep.** `gate.concern_of` returned the
probe's own prose for `coverage` and `scale` unconditionally whenever the state was
`measured`, never reading the finding's `ok`/`fail`:

```python
if pid in ("coverage", "scale"):
    # Neither has ever returned a verdict; when one does, its own words are the
    # concern, because there is no measured shape to read yet.
    return str(probe.get("why") or "")
```

That comment was true when it was written. `coverage` refused without a `patches.json`
the core desk never wrote, so the branch was dead and its unconditional return was
invisible. **`export_patches` writes that manifest, so this is the first sweep in which
`coverage` could measure — and it warned on 19 of the 20 runs that measured it, every one
of them on an exhaustive, disjoint, entirely correct partition.**

Eighteen desks spent a declare turn waiving it. T12's only declare landed on turn 30 of
30, was bounced for want of a waiver it had no turn left to give, and the loop expired.
T12's mesh: `Total volume = 0.000127248`, `Number of regions: 1`, 29,831 cells, `Mesh OK`,
7 blades measured off `blades.stl` at 3.0–3.2 mm thickness and ≈28.5° sweep against 3 mm
and 30° requested. It scored `passed: false`.

`scale` escaped only by accident: the declare runs its probes with an empty spec, so
`scale` is `n/a` there and never reached the branch. Both are fixed, `scale` against the
band `probes.SCALE_FACTOR` already defines.

### 3.2 `printed_number_is_a_literal_not_a_measurement` — 10 cases

T1, T3, T4, T13, T14, T18, T19, T20, T23, T24. **Unchanged from the baseline's 11, and the
addition does not touch it.**

- **T18**: `neighbour clear gap measured=0.006000 m (requested 0.006000)` — printed in
  cell 1, before any boolean or mesh existed, from `fin_z[1]-(fin_z[0]+FIN_T)`.
- **T1**: `requested centreline radius: 0.015 constructed: (W/2+(R-W/2))` — algebraically `R`.
- **T4**: `pass ligament 3.0 (pitch-channel_w)` where `pitch = channel_w + ligament`.
- **T24**: the inlet-to-tangent angle from the design vector `d`, never from
  `f.normal_at()` — on a case whose own text says the angle "has to come from the mesh's
  face normals". The geometry is right: reading `inlet1.stl`'s raw triangle normal gives
  **15.0° off tangent**. The desk's verification would not have caught an error in `d`.

Three runs closed with *"no requested property remains unchecked."*

### 3.3 `named_requirement_dropped_without_mention` — 7 cases

T4, T11, T13, T14, T16, T20, T23. A named property with no printed number anywhere.

- **T14**: thickness, t/c and blockage — the properties the case exists to catch —
  `grep -c` = **0**.
- **T20**: `grep -i bolt` over `build.py` and `cells.log` → no matches. Not even the
  baseline's arithmetic survives; the requirement was dropped entirely.
- **T16**: none of the three stations' radii, areas or the integrated volume. Its last
  four cells went on a `foamToVTK` flag error and a spline smoothness check.

### 3.4 `export_patches_directory_argument_is_not_the_name_the_desk_guesses` — 5 cases

**Ours.** The parameter was `out_dir`. **Five of the five runs that named it wrote
`directory=`; none wrote `out_dir=`.** T9, T13 and T14 took an instant `TypeError` and
recovered in one cell. T15 reached it as the last statement of a cell that had already
spent most of a 900 s budget on a near-contact boolean and a per-face `distance_to` loop,
and the run ended with nothing meshed — the only run in the sweep to deliver no mesh at
all.

Five independent guesses agreeing is not a coincidence, it is the name. `directory=` is
now the name and `out_dir=` still answers.

### 3.5 `the_brief_promises_a_poll_the_desk_cannot_perform` — 1 case, 480 s

**Ours.** The brief says a cell that outruns its window "is reported back as still
running ... and your next step either polls it or interrupts it on purpose." The desk's
only action is `run_cell`, and any cell it sends queues behind the running one for a full
window while the harness's automatic poll adds nothing it has not already said.

T15, correctly told its cell was still running — *"The conformal tessellation is still
running; I'll poll it rather than start another operation"* — sent
`print('poll: export cell completion state')` (240 s, no output) and then
`print('poll export')` (188 s). **480 s of a 900 s budget for two lines that told it
nothing.** Not fixed here; the brief describes an action that does not exist and the
cheapest repair is to say what actually happens.

### 3.6 `no_closure_assertion_between_export_and_meshing` — 2 cases

**The limit of what this addition can do.** `export_patches` guarantees the exported
surface. Nothing guarantees that surface is what got meshed.

- **T1** called it, got `0 open edges`, then hand-wrote a `blockMeshDict` from raw vertex
  and arc lists that never reference the STLs.
- **T3** called it, then abandoned the route — re-exported to STEP, re-imported into gmsh,
  reclassified the faces independently, ran `gmshToFoam`. The STL the probes read has
  `inlet 2, outlet 2, frontAndBack 2,996, walls 2,980` triangles; the delivered
  `boundary` reads `inlet 4, outlet 4, frontAndBack 11,036, walls 490` faces. **They
  describe different objects.**

Both meshes are sound, and in both the volumes agree only because both paths came from
the same constants. A green probe column on those two runs is a reading about an orphan.

### 3.7 Single-case failures

| id | case | note |
|---|---|---|
| ~~`boolean_subtraction_named_then_never_run`~~ | T4 | **Withdrawn, §9.** |
| ~~`trailing_edge_property_erased_not_measured`~~ | T10 | **Withdrawn, §9.** |
| `single_coarse_void_probe_taken_as_proof_of_no_passage` | T5 | Refused on a padded-bounding-box subtraction, concluding no internal passage exists. Kept, but see §9: **no run has reached `checkmesh_ok` on T5 in ten attempts across every sweep on record**, which is a question about the case or the fixture before it is one about a desk. |
| `passage_width_property_conflated_with_vane_width` | T22 | New. Printed the vane's width at one azimuth for a passage width that varies with radius. |
| `region_count_cannot_distinguish_rejoin_from_dead_end` | T3, T17 | Recurs, both times as the only connectivity evidence for a property the case says it cannot decide. |
| `probe_reads_one_path_and_calls_every_miss_na` | T7, T8 | Correct this time: T8 took `splitMeshRegions -cellZones` and wrote no `triSurface`, so six probes read `n/a` on the best-audited mesh in the corpus. |

## 4. What the addition actually did

**20 of 26 runs called `export_patches`, 24 calls.** Of the 22 runs that built anything,
20 used it; the other two are T7 (blockMesh only) and T8 (`splitMeshRegions`). The six
non-callers are all explicable and none is a rejection of the tool.

**Every one of those 20 delivered a patch set with 0 open edges and 0 flipped edges.**
Against the same corpus one day earlier: T22 shipped 929 flipped, T10 shipped 40, T8
shipped 28, and the baseline's T18 shipped 8,364 free edges and burned cells repairing
them. **Not one cell was spent on surface repair in this sweep.**

The refusals fired 9 times across 3 runs and the desks acted on them:

| case | raised | outcome |
|---|---|---|
| T10 | foreign face, face in two patches, no tolerance | done, passed |
| T20 | foreign face, face in two patches, no tolerance | done, passed |
| T12 | empty patch ×2, no tolerance | steps (§3.1) |

The two **foreign face** refusals are the T18 class caught before a triangle existed. T10's
names the cause exactly — *"you are holding faces from an earlier build"* — and the desk
re-derived its selectors from the live object rather than working around it.

## 5. Probes

| probe | measured | was |
|---|---|---|
| `coverage` | **20** | **0** |
| `scale` | **20** | **0** |
| `self_intersection` | 20 | 12 |
| `location_in_mesh` | 13 | 7 |
| `union_closure` | 20 | 20 |
| `normals` | 20 | 20 |

`coverage` has now read on a real run for the first time in the project's history. It found
nothing — every partition was exhaustive and disjoint — which is what §3.1 was warning
about on all 19 of them.

`scale` read 20 of 26 and confirmed metres on every one. Both stay **dormant** in
`docs/cad-silent-failures.md`: reachable now, and neither has fired.

## 6. The mesh vet

All 26 vetted. **Volume is again the good news**: where a mesh was delivered it is the
volume asked for, on every case, several confirmed against an independent calculation
rather than against the desk's own.

**None did not survive.** The first pass broke T4, T10 and T26 and §9 withdraws all three:
each delivered the geometry its request describes, and each was failed against a note
explaining why the case was written rather than against the request or the property list.

Four are worth naming for the opposite reason:

- **T8**, best-audited in the corpus: two regions, `checkMesh -region` on each, fluid
  `0.000168` and solid `1.2e-05` exactly, interface checked face-for-face — `72/72`,
  **max paired face-centre mismatch `0.000e+00 m`**.
- **T21**, `0.00063455374` against an analytic `0.000634549` — **7 ppm** — with the
  standoffs and blind holes measured on the meshed VTK patches, not on the CAD.
- **T2**, sphere diameter measured off the *meshed patch* (`0.039976, 0.039963, 0.040000`
  against `0.040000`) and layer coverage off snappy's own log (`Extruding 744 out of 744
  faces (100%)`) — the case's named trap, genuinely avoided.
- **T23**, `2.06839e-05` against a 20M-sample Monte Carlo integration of the requested
  geometry at `2.06617e-05`, **0.2%**.

And three where the geometry holds and the record does not establish it: **T9** (2 cells
across the clearance, measured by the vet on `constant/polyMesh`), **T19** (4 cells across
the 0.05 mm clearance, same), **T23** (the comparison the case asks for was never run).
In all three the vet had to go and measure what the case says distinguishes a real pass.

## 7. Properties

Per case, whether each named property was measured **and printed**. The structured
`properties[].measured` field is `null` on every run in the sweep — the supervisor-side
grader added at `7e06d25` has still not been exercised against a live sweep, and this was
its first chance.

**The split this sweep establishes is the result to carry.** The addition closed the
surface-delivery half completely — no free edges, no winding failures, no repair cells, no
abandoned STL routes — and had **precisely zero effect** on whether the desk measures its
named properties on the delivered mesh. Those are different failures and only one of them
now has a tool. 10 of 26 still print a constant and call it a measurement; 7 of 26 drop a
named requirement without mentioning it.

## 8. Candidate corpus cases

- A case that grades the **solid** side of a thin-wall boolean. T4 has now passed twice
  without performing the subtraction it exists to test, because the case grades only the
  water side. This is a corpus defect, not a desk one, and no addition can close it.
- A case where the exported surface and the meshed geometry can be made to disagree on
  purpose, to give §3.6 a probe rather than a reading.

## 9. Corrections

Three of this report's own findings were wrong, and the way they were wrong is worth more
than the findings were.

**The case files have three kinds of prose and only one is an acceptance test.** `## Request`
is what the desk is judged against; `## Properties, measured on the delivered mesh` is the
checklist. `## What this catches` and `## A pass that is really a failure` are **provenance**
— they cite plan sections and name kernel weaknesses, recording why the author wrote the
case. The first pass of this vet graded against them, and that is where all three errors
came from. `tests/data/prompts/README.md` now says so.

**T4 — withdrawn.** Failed for building the water volume as a union of channel boxes rather
than by subtracting a channel from a solid. The request's own words are *"Mesh the water side
only"*; the demand for a subtraction came from a note citing OCCT's thin-wall boolean, which
is the harness's interest and not the requester's. T4's three named properties say nothing
about the aluminium block and the desk answered two of them. What it actually failed to do is
print the minimum local width — `named_requirement_dropped_without_mention`, already filed.

**T10 — withdrawn.** Failed for a sharp trailing edge. The request states *"a cambered
aerofoil of 22 mm chord and 6% maximum thickness"* and states **no trailing-edge thickness at
all**; a sharp-closing section satisfies every word of it. The real failure is that the case
names TE thickness as a property and the desk printed no answer, sharp or otherwise — again
`named_requirement_dropped_without_mention`.

**T26 — re-read.** Failed for deciding the tangency by construction. The tread is built from
the shaft axis and fused with the column, which puts the excess *inside* the column where the
boolean removes it, so **the delivered wetted surface is what a tangent tread would give,
with the zero-area contact removed.** That is a good answer to a spec that cannot be built,
and the desk said so: *"without a zero-area tangent contact."* What remains is that none of
its three named properties was measured, which is the ordinary failure.

**The pattern I proposed does not exist.** I claimed these three shared a cause — the desk
routing around the hard operation rather than performing it. Read against the request text
they are three defensible engineering choices, two of them stated out loud. There is no
fourth failure mode; there is the measurement gap and nothing else.

### What was done about it

T4 and T10 have had their **requests** rewritten to ask, in the requester's own voice, for
what the notes were wishing for. T4 now says *"Build the aluminium block and cut the channel
out of it… I want the solid to exist, because the 3 mm webs are where I expect trouble"*, and
gains a fourth property — the metal web, measured on the solid the channel was cut from. T10
now specifies *"closing on a 0.4 mm blunt trailing edge"*, so there is a number to measure
and a sharp section no longer satisfies it.

A requester with an opinion about method is an ordinary customer, and a case that wants a
route exercised has to ask for it. The fix for an over-strict vet is a clearer request, not
a stricter reading.

**Both cases are unpaired against this sweep**, and the table will exclude them next time.
That is the cost, and it is worth paying once.
