# Sweep `core+declare_gate-20260913-124524-aa21` — the addition works, and this sweep cannot prove it

## 1. The sweep

| | |
|---|---|
| label | `core+declare_gate` — the declared finish and the advisory gates |
| model / effort | `claude-opus-5` at `medium` — unchanged for the fifth sweep running |
| core sha | `8ba92f9` (clean tree) |
| cases | T1–T26, one run each |
| baseline | `core+bench26-20260912-133719-4bbd` — **15 of 26 pair**, 11 unpaired and excluded |
| passed | **20/26** (paired: **13/15 → 11/15**) |
| cells | 518 |
| spend | **$24.29** |
| contamination | **0/26** |

```
1 case used fewer cells, 8 more, the rest within 30% of their own cell count.
sign test over the cases that moved: p = 0.039
```

**Two earlier attempts at this sweep are kept beside it and are void.** `090051` died when
the API account ran out of credits (21 of 26 cases never made a model call); `091133` died
with its session at 10 of 26 and, in doing so, found that the addition it was measuring did
not work. Both carry a `VOID.md`. Together they cost $11.76 and the second one paid for
itself.

## 2. What the gate did

| state | count |
|---|---|
| `clean` | 45 |
| `n/a` | 77 |
| `warned` | 6 |
| `xfail` | 2 |
| `xpass` | 2 |
| `waived` | 0 |

22 declares across 26 runs. **Five cases saw the gate say anything at all**: T2, T12, T18,
T21, T22.

### 2.1 T18 is the case the addition exists for, and it worked

| | declare 1 | declare 2 |
|---|---|---|
| `location_in_mesh` | **xfail** — predicted, external flow, point deliberately outside | xpass |
| `union_closure` | **warned** — 9,364 free edges | gone |
| `normals` | **warned** — 3 flipped edges | gone |
| `self_intersection` | **warned** — 15,131 crossing pairs | gone |

Told about a surface that does not close, the desk went and closed it, declared again, and
passed. **In `core+bench26` the same case shipped 13,204 open edges, scored `passed: true`,
and the number was never delivered to it.** That is the failure this addition was built to
close, closed, on the case it was built from.

### 2.2 The pre-waiver incentive works

T2 predicted `location_in_mesh` would fire — *"external aerodynamics: the only STL is the
closed sphere and the fluid is outside it"* — it fired, the reason stood, and the run
finished in **one declare**. Understanding the geometry costs nothing; not understanding it
costs a round trip. `xfail` appeared twice here against zero in the broken version, which is
the mechanism coming alive rather than a number moving.

## 3. And the sweep cannot attribute its own headline

**The cell count rose significantly — and six of the eight increases are on cases the gate
never spoke to.**

| cases that used ≥30% more cells | gate said something? |
|---|---|
| T18, T22 | yes |
| **T15, T17, T19, T20, T23, T25** | **no — clean or `n/a` throughout** |

So `p = 0.039` is real and is mostly not the warnings. What else changed for every case is
the rest of the addition: the whole `# Finishing` section rewritten, the finish moved off
`print("CAD_DONE")` onto a tool call, and the `union_closure` explanation added. **This
sweep changed four things and can say which one did anything only where the gate left a
trace**, which is five cases.

That is a self-inflicted wound and the skill names it: *two changes in one sweep is a sweep
that cannot say which one did anything.* The declare tool and the gates are arguably one
thing — the gates need a boundary to run at — but the brief rewrite is not, and the
`union_closure` line certainly is not.

**I also predicted this would not happen.** Before the first launch: *"the cell counts
should barely move… this changes what the desk is told at the finish, not how it builds."*
Eight cases up, one down, p = 0.039. The prediction was wrong and is recorded as wrong.

## 4. The two regressions, which are not the same

`13/15 → 11/15` on the paired set. T19 and T22 flipped.

- **T22 is plausibly the gate.** Warned once, went after it, and spent the rest of its
  budget: `steps` at 28 cells of 30.
- **T19 is not.** Zero declares, the gate never said a word, `steps` at 28 cells against 20
  before. Whatever moved it is the same unattributable thing as §3.

T12 did the same as T22 and was already failing, so it is not a regression but is the same
behaviour: shown 2,177 free edges and 3,425 crossing pairs, it merged the patch STLs in
pyvista, pulled the feature edges, histogrammed their z positions and walked OCCT's
edge-to-face map — real diagnosis of a real defect — and ran out of room at 27 cells.

**This is the rule working as designed and costing what it costs.** A desk that neither
fixes nor explains a warning no longer finishes, and T12 and T22 are what that looks like
when the desk chooses to fix. Both sat at 27–28 of a 30-turn budget: the budget was set for
a desk that finishes at its first declare, and it no longer always does.

## 5. Probes

| check | warned | xfail | xpass | n/a |
|---|---|---|---|---|
| `union_closure` | 2 | – | – | 7 |
| `normals` | 2 | – | 1 | 7 |
| `self_intersection` | 2 | – | – | 8 |
| `location_in_mesh` | – | 2 | 1 | 11 |
| `coverage` | – | – | – | **22** |
| `scale` | – | – | – | **22** |

`coverage` and `scale` returned `n/a` on **every declare**, which confirms from the other
side what the last sweep established: they need a manifest and a `spec.json`, and a channel
to the desk changes nothing about that.

### 5.1 `xpass` is doing its job, and on a new probe

T21 predicted `normals` would fire because *"STLs are the boundary faces of the fluid solid,
so their normals point out of the fluid"* — correct about the geometry, and not what
`normals` measures, which is winding consistency. **This is the same class the
`union_closure` explanation closed**: `xpass` on `union_closure` went 2 → 0 once the brief
said what it welds. `normals` has no such line. That is the next one-line change, and it
should go in alone.

### 5.2 And `xpass` has a defect of its own

T18 waived `location_in_mesh` correctly on declare 1, fixed the surface, and on declare 2
the same waiver reads `xpass` — because the point is now inside the welded union. So
`xpass` cannot separate *the desk misread the check* from *the desk removed the reason it
fired*, and those are opposite verdicts on the desk.

## 6. The verdict on the addition

**Keep it.** T18 alone is the case for it: a surface with 9,364 free edges that the previous
sweep shipped as a pass, found, reported, and repaired. Nothing else in the desk can do
that, and `checkMesh` structurally cannot — it validates the volume mesh, which
`snappyHexMesh` emits closed whatever the input surface did.

**But this sweep does not establish it**, and the report should not pretend otherwise. Five
of 26 cases exercised the gate. The corpus-level numbers moved for reasons the sweep cannot
assign. The honest summary is: the mechanism is demonstrated on one case, the cost is
demonstrated on two, and the aggregate is confounded.

**What the next iteration should do, one at a time:**

1. **The `normals` explanation**, alone. It is one line, it has an `xpass` behind it, and
   the `union_closure` precedent says it will work.
2. **Re-measure with nothing else moving**, to get a clean attribution of the cell cost.
3. **Look at the step budget.** T12 and T22 both stopped at 27–28 of 30 doing legitimate
   work the gate asked for. Raising it is a change with a measured failure behind it —
   but it is its own change, and it must not ride along with anything else.

---

*Produced by `/cad-addition`. An addition that cannot show a difference is a candidate for
removal at the next audit; this one shows a difference on one case and cannot show it on the
corpus, which is a weaker claim than it looked like it would be when it was written.*
