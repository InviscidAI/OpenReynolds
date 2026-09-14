# Sweep `core+reference-20260914-093903-3472` — the tool works and the corpus got worse

## 1. The sweep

| | |
|---|---|
| label | `core+reference` — `b123d_api.md` handed to the core desk, plus three other changes |
| model / effort | `claude-opus-5` at `medium`, unchanged for the sixth sweep |
| core sha | `2817999`, clean tree |
| cases | T1–T26, one run each |
| baseline | `core+declare_gate-20260913-124524-aa21` (`claude-opus-5` at `medium`, core `8ba92f9`) |
| passed | **16/26**, against 20/26 before |
| cells | 524, against 518 |
| spend | **$26.36**, against $24.29 |
| contamination | **0/26** |

```
1 case used fewer cells, 6 more, the rest within 30% of their own cell count.
sign test over the cases that moved: p = 0.125
```

**This sweep carries four changes and was launched knowing it could not separate them.**
The `/cad-addition` rule is one at a time; it was broken deliberately, on the instruction
that serialising the sweeps costs more than the attribution is worth. What follows says
which claims survive that and which do not.

The four: the reference file; the repaired probes; the `normals` explanation in the brief;
the assumption-recording line in the brief.

**A void predecessor sits beside this one.** `core+reference-20260914-025525-6a2b` reached
10 of 26 and cost $11.14. It is kept because it found that the addition it was measuring
was never delivered — `prepare()` copied the reference to the workspace root while the
desk's working directory is the case directory one level down, so the path the brief gives
resolved to nothing. Its `VOID.md` has the evidence.

## 2. Contamination

**0 of 26**, with `b123d_api.md` in every workspace and greped by 11 desks.

This is the half of the addition that had to work before anything else could be measured:
`isolation.house_names` reads every toolbox `.py` and `.md` off disk, so without the
`given` mechanism the first desk to grep the reference would have graded contaminated and
voided the sweep. It held — 0 hits across 26 runs, while `cad_convert.py` and every other
toolbox name stayed covered.

## 3. The tool was delivered, and it was used

**11 of 26 runs opened it** — T2, T3, T4, T6, T7, T8, T11, T16, T21, T25, T26 — against
**0 of 26** in the baseline, where the reference did not exist, and 1 of 10 in the void
sweep, where it existed and was unreachable.

## 4. It did the specific job it was built for

| | before | after |
|---|---|---|
| failed cells, all causes | 36 in 19 cases | **30 in 16 cases** |
| **api-surface failures** | **27 in 15 cases** | **13 in 9 cases** |
| `AttributeError` | 11 | **2** |
| `NameError` | 4 | **0** |
| `ModuleNotFoundError` | 1 | **0** |
| `TypeError` (`not callable`) | 10 in 6 cases | 11 in **8 cases** |

**API-surface failures halved.** The three categories that are pure discovery failures —
a method that does not exist, a name never imported, a module invented outright — went
11 → 2, 4 → 0 and 1 → 0. `ModuleNotFoundError: No module named 'trimesh_check'` did not
recur.

**And the one category the OCCT preamble was aimed at did not move.** `'bool' object is
not callable` — a property called as a method — went from 6 cases to 8, against a line
that says in as many words *"A wrapper listed in this file with no `()` takes no `()`"*.
Reading a list does not stop a hand typing `()`. That line has now been measured and did
not earn its place; it is a candidate for removal rather than for rewording.

Raw-OCCT-implicated failed cells went 7 in 4 cases → 6 in 3 cases, which at those counts
is not a change.

## 5. And the corpus got worse

**20/26 → 16/26.** Six cases flipped `passed → failed` and two flipped back:

| case | before | after | |
|---|---|---|---|
| T10 | done, 27c | **steps, 28c** | pass → fail |
| T13 | done, 17c | **steps, 28c** | pass → fail |
| T18 | done, 27c | **steps, 27c** | pass → fail |
| T20 | done, 15c | **steps, 28c** | pass → fail |
| T24 | done, 22c | **steps, 28c** | pass → fail |
| T26 | done, 20c | **steps, 28c** | pass → fail |
| T19 | steps, 28c | done, 20c | fail → **pass** |
| T22 | steps, 28c | done, 11c | fail → **pass** |

**Every one of the six ran out of the step budget.** Not one failed a check, produced a
bad mesh, or errored out — `stopped=steps`, five of them at exactly 28 cells.

**And the gate spoke to only one of them.** Of T10, T13, T20, T24 and T26, no declare
carried a single non-clean state. Whatever consumed their budget, it was not a warning
being chased — the same conclusion the previous sweep reached about T15, T17, T19, T20,
T23 and T25, on independent evidence.

What they were doing at the ceiling is legitimate work. T20, turn 25: *"The gasket band is
only 2 mm wide and snappy never resolved it, so it got no faces. I'll align the background
mesh so z = 20 mm and 22 mm are cell-faces…"*, and at turn 29 it was snapping rim edges
with feature edges. T13, turn 27: *"Gap measures 0.1999–0.2040 mm over the 25th–100th
percentile…"*. These are desks refining and measuring, and running out of room.

### 5.1 The budget is the binding constraint, and this sweep is the evidence

| | before | after |
|---|---|---|
| ended `done` | 20 | 17 |
| ended `steps` | **4** | **7** |
| ended `time` | 2 | 2 |
| median cells | 21 | 21 |
| max cells | 28 | 28 |

The distribution did not move — same median, same maximum, 518 → 524 cells in total, and
`p = 0.125` says the per-case changes are sampling. **What moved is how many runs tipped
over a ceiling they were already sitting against.** Seven of 26 now die at it.

This was named in the previous sweep's report as the third thing to do, with T12 and T22
stopping at 27–28 of 30 as its evidence. That evidence is now stronger and is the clearest
single finding here: **a corpus whose median run is 21 cells and whose ceiling is 28 cannot
absorb any change that costs a desk two or three cells**, whatever that change is worth.

### 5.2 The two that flipped back are the probe repair

**T22** is the case the repair was written for. Its baseline run was bounced at turn 30 on
62,208 phantom flipped edges welded out of a leftover `brakeDisc.stl`. This run:
`normals` reads **0 flipped edges on 2,736 triangles**, it declared, and it finished in
**11 cells** — the fewest of any run in the sweep. **T19** likewise went 28 cells and
`steps` to 20 cells and `done`.

Both are consistent with the repair and neither proves it alone: T22 also built its
geometry differently this time (five patch files, different names, no combined export at
all), so the duplicate it tripped over last time was never created.

## 6. The verdict

**The tool works and does not pay for itself here.**

What is established: the reference is reachable, 11 desks used it, and the failures it
targets halved — `AttributeError` 11 → 2 is not noise. The isolation mechanism that had to
exist for it to be measurable at all held at 0/26.

What is not established: any corpus-level benefit. The pass rate fell by four, the cell
count is flat, and the sign test is `p = 0.125`. And this sweep cannot attribute either
direction, because it moved four things at once — the six step-exhaustion failures are as
consistent with the assumption-recording line asking for more reporting work as they are
with anything else, and nothing here separates them.

**An addition that cannot show a difference is a candidate for removal at the next audit.**
This one shows a difference in the failures it was built to close and a worse corpus around
it. On the evidence it should be kept and re-measured, not removed — but the next iteration
has to be one change, and the step budget is the change with the strongest measured case
behind it.

**What the next iteration should do:**

1. **Raise the step budget, alone.** Seven of 26 runs now end at the ceiling, five at
   exactly 28, and at least two of those were doing refinement work they had chosen. This
   is the one change with a failure count already measured against it, and until it is made
   every other measurement is taken through a ceiling.
2. **Re-measure the reference with nothing else moving**, to find out whether the halved
   API failures buy anything once the budget is not the constraint.
3. **Drop the `no ()` line from the OCCT preamble** or replace it. It is measured, it did
   not work, and `'bool' object is not callable` went 6 cases → 8 against it.

## 7. The tests that demonstrate these failures in their absence

- `test_the_reference_the_brief_names_is_reachable_from_the_desks_own_directory` — resolves
  the brief's own path against the directory the desk is handed. Verified to fail against
  the code that produced the void sweep.
- `test_a_file_the_arm_was_handed_is_not_a_house_surface` and
  `test_greping_the_handed_reference_does_not_contaminate_but_the_toolbox_does` — the
  isolation rule, both halves.
- `test_the_core_brief_names_exactly_what_it_hands_over_and_nothing_else` — the tightened
  invariant.
- `test_a_file_the_mesh_dict_does_not_name_is_not_welded_in`,
  `test_the_dict_restriction_does_not_drop_a_patch_the_surface_needs`,
  `test_open_edges_are_located_rather_than_only_counted`, and the three other probe tests.

## 8. One intervention to disclose

This sweep ran with a **kernel janitor** alongside it, killing each run's IPython kernel
once its record showed `completed` and carried `observe`'s probe results. It reaped 26
kernels, 430 MB–1.3 GB each.

It was not part of the addition and it changes nothing a desk saw — it touches no
workspace and signals nothing until the run is measured and on disk. It is disclosed
because it is an intervention during a measurement, and because without it this sweep
would have died the way the one before it did: kernels outlive their runs, twelve runs'
worth plus an 18.6 GB IDE backend exhausted 62 GB of RAM with no swap, `sshd` could no
longer fork at 03:41, and the machine was unreachable until it was restarted at 09:22.

**The real fix is that `cad_buildup.py` should shut its kernel down when a run ends.** The
janitor is a prop holding a measurement up, and it is the next infrastructure change
whether or not it is the next addition.

---

*Produced by `/cad-addition`, against the rule it names: this sweep changed four things and
can attribute none of them at the corpus level.*
