# core-kimi-k3-moonshot-1800s-20260915-073451-a25f

**The budget test.** Seven cases at `--seconds 1800`, double the desk's default, to ask
whether the kimi-k3 runs that built a correct mesh and ran out of clock can close when
given the clock. Five were the cases that did exactly that (T11, T14, T16, T18, T22); T1
and T7 already passed and are controls.

This run was only possible from `43bc706`: before it the supervisor's deadline was a
constant 1500 s, so an 1800 s desk would have been killed at 1500 s and filed `wedged`.

## 1. The result

| case | 900 s | | | 1800 s | | |
|---|---|---|---|---|---|---|
| | stop | steps | secs | stop | steps | secs |
| T1 | done | 11 | 544 | **done** | 16 | 656 |
| T7 | done | 9 | 225 | **done** | 15 | 352 |
| T11 | time | 15 | 907 | time | 24 | 1896 |
| T14 | time | 26 | 931 | **done** | 23 | 911 |
| T16 | time | 20 | 942 | steps | 28 | 1144 |
| T18 | time | 26 | 1040 | steps | 28 | 1579 |
| T22 | time | 25 | 922 | steps | 29 | 1663 |

**2/7 -> 3/7.** Contamination 0/7. $4.21.

**The hypothesis mostly fails.** Doubling the clock bought one conversion, T14, which on
n=7 is one case and not a result. What it did instead is move the binding constraint: three
runs that died on the clock now die on the 30-step cap. Given twice the time they take more
steps and still do not close.

## 2. And the mesh was never the problem

Every run here -- including all four that failed -- carries `why: ""`, meaning the finish
check found nothing missing, and a `checkMesh`-clean mesh of the right shape:

| case | delivered | target | |
|---|---|---|---|
| T22 | 0.00039851424, **36 regions** | CAD 4.0028e-4 | 0.4% |
| T16 | 0.27634697 | analytic 0.275573 | 0.3% |
| T11 | 0.000869793 | CAD 8.5627e-4 | 1.6% |
| T18 | 0.0499974, 6 named patches | 0.054261 | 7.9% |

Four failing runs, four correct meshes, all with their named patches populated. Whatever
kimi-k3 is failing at, it is not the geometry and it is not the mesh.

## 3. What it is failing at: closing the run

`declares` splits the four failures in two:

| case | declares | what happened |
|---|---|---|
| T16, T18 | 1 | declared complete, the declare was **refused**, and the rest of the budget went to not resolving it |
| T11, T22 | 0 | built the right mesh and **never called `declare_complete` at all** |

And the refusal is legible. Both declares carry `checkmesh_ok: true` -- OpenFOAM's binding
verdict passed -- and an advisory warning the desk left unanswered:

- **T16**: `union_closure` warned, *"the welded surface has 1,096 free edges, so it does not
  close"*, with `because: ""`.
- **T18**: `union_closure` (24,039 free edges) **and** `normals` (*"2 edges are walked twice
  the same way"*), both `because: ""`.

The brief states the rule plainly: *"A warning you neither fix nor waive means the declare
is not accepted, exactly as a failing `checkMesh` is not accepted, and it comes back to
you."* The waiver exists for exactly this case -- a correct volume mesh built from a
surface that does not weld closed -- and it costs one sentence. kimi-k3 neither closed the
surface nor said why it did not matter, on either attempt, at either budget.

Opus 5 shows the working shape on the same corpus: T14 has `declares: 2` -- refused once,
resolved, passed.

### What the desk is doing instead, in its own cells

The two that never declared were not idle and were not lost. They were repairing the
surface.

**T11** (died on the clock at 1,896 s) spent its whole endgame on the free edges:

    cell 22  print("free-edge vertices (mm):")  -> [[-33.486 -4. 0.] [-35.796 0. 0.] ...
    cell 23  # inspect individual free-edge segments
    cell 24  nearest STL point to a free-edge vertex -> dist mm: 0.000282
    cell 25  inspect.signature(bd.Shape.tessellate)

**T22** (died on the step cap) said at turn 27, in plain text, *"All 36 passages are
meshed"* -- it knew -- and then spent its last three cells writing a
`surfaceFeatureExtractDict`, hunting for `*.eMesh` files, and reading
`surfaceFeatureExtract -help`. The run ended mid-lookup.

So all four failures are one failure. The exported STL has free edges, `union_closure`
warns, and the desk treats an advisory warning as a defect it must repair: two declare and
cannot answer it, two never declare because they are still fixing it. The waiver is the
sentence that ends either, on a mesh that was correct to within 0.4% the whole time.

**This failure is already in the corpus**, from the Opus chain:
`warning_chased_until_the_step_budget_ran_out`. It is not new, and it is not
kimi-k3-specific. What is kimi-k3-specific is the rate -- Opus hits it once in these
seven, kimi-k3 hits it four times in four.

It also explains why time does not help. More clock buys more free-edge chasing: T11 spent
1,896 s of it.

One detail beside it: T22's cell 27, the patch-area comparison that would have answered a
named property, **exited 1**. The property measurement was failing too, which is consistent
with no arm on this corpus printing its named properties.

## 4. Where that leaves the investigation

Across four arms on these cases the explanation has narrowed each time and ended somewhere
none of the early hypotheses pointed:

- not the adapter -- 2/10 on Chat Completions at Aster and on the Messages API at Moonshot,
  the adapter Opus passes 7/10 on;
- not the brief -- fixed in `88a393f`, and the Moonshot arms ran on it;
- not the provider -- Moonshot is first-party and 1.7x faster;
- not throughput or the clock -- doubling it converts one case of seven;
- not the geometry -- the meshes are right to a fraction of a percent.

**It is the closing protocol.** kimi-k3 builds the mesh and cannot finish the run: it does
not declare, or it declares and cannot answer an advisory warning that a single waiver
would clear.

That is a claim about one model on one corpus at `medium` effort, from n=7. What would test
it directly is cheap and is not a sweep: give the desk a mesh that is already correct and a
surface that does not weld, and see whether it waives.
