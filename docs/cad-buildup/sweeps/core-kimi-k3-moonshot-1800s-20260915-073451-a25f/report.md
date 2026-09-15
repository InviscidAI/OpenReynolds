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
