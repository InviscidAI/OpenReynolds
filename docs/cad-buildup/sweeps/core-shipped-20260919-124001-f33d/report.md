# core-shipped-20260919-124001-f33d

The first sweep of the desk that ships. `cli.py` builds a `CoreDesk`, and this is that
object: the lean brief, no toolbox, two reference files, `checkMesh` per region binding,
and the advisory gate reading `cad_audit.py` and `domain_probe.py` over the backend.

**22 of 26 by the gate, against 23 of 26.** Flat, and the headline is not the number.
Three things have to travel with it and two of them are about this harness rather than
about the desk.

## 1. The sweep

| | |
|---|---|
| label | `core-shipped` |
| model | `gpt-5.6-sol` at `medium`, unchanged from the baseline |
| core sha | `66df673` (`+dirty` is two untracked sweep output directories, no source) |
| cases | all 26, one run each, `--parallel 2` |
| ended `done` | 20 / 26 (4 `refused`, 1 `time`, 1 `no-progress`) |
| `checkmesh_ok` | 21 / 26 |
| passed, by the gate | **22 / 26** |
| contaminated | **0 / 26** |
| unobserved | **0 / 26** |
| total cells | 398 |
| total spend | **$9.67** |

`scripts/cad_smoke.py` passes against the artifact set learned from
`core+cad_export-20260917-022129-dd05`: all 26 runs carry `heartbeat.jsonl`,
`replies.jsonl`, `record.json`, `run.pgid`, `runner.log`, `spec.json`, `watch.json`.
That check exists because the previous attempt at this sweep did not, and eight of its
runs were killed by a heartbeat that was never written.

### Against the baseline

```
22/26 passed, 398 cells, $9.67
before: 23/26 passed, 404 cells, $9.76
1 case used fewer cells, 5 more, the rest within 30% of their own cell count
sign test over the cases that moved: p = 0.219
```

**T4, T5 and T10 are paired in that table and should not be.** Their prompt files differ
from the baseline's `d074a46` -- T4 and T10 had their requests rewritten after the
`core+cad_export` vet withdrew three of its own findings, and T5 was reposed to external
flow. The tooling pairs on the case's *name*, so it cannot see this. None of the three
changed verdict, so the headline is unaffected, but the comparison is over 23 cases and
not 26.

## 2. What is not a desk failure

**T11 ran the machine out of disk.** `snappyHexMesh` on a ~100-cutter honeycomb filled the
filesystem and the desk refused rather than report a partial mesh: *"the machine now
rejects even an in-case cleanup cell with `Errno 28: No space left on device` ... I am
therefore reporting the case as incomplete rather than misrepresenting the existing
background/partial mesh."* That is the desk doing the right thing with a broken
environment, and it scores `passed: false`.

It was the third run, at +122 s, and the **only** one affected -- all 23 runs after it
finished clean, including the two longest. So it is a transient and not a sweep-wide
caveat. The root is upstream: `/` is at **98%** with 5.7 GB free, and
`~/.openreynolds-buildup/work` holds **302 workspaces from previous runs, 10 GB**, because
isolation makes a fresh root per case per run and nothing removes them.

**Discounting it, 22 of 25.** T11 needs re-running after the stale workspaces are cleaned;
until then the corpus has no verdict on that case from this codebase.

## 3. The three real failures

**T19 -- declared complete over nothing.** `mesh_exists: false`, `checkmesh_ok: false`. It
declared at turn 11, was bounced, produced an empty turn, declared again, and
`no-progress` killed it at K=3. The alarm is correct and the diagnosis is the desk's.

**T15 -- a passing mesh at the buzzer.** 924.9 s against a 900 s budget, ended `time` on
its own clock with no alarm fired, and `checkmesh_ok: true`, `mesh_exists: true`. It built
an acceptable mesh and ran out of seconds before it could declare. Scored a non-pass,
which is defensible -- an undeclared finish is not a finish -- but it is the second case
in two sweeps where the declare cost more budget than the geometry did. `core+cad_export`
§3.1 has the first: T12 landed its only declare on turn 30 of 30.

**T5 -- the kernel hung.** *"The OCCT defeaturing operation hung for the remainder of the
execution budget before any mesh could be created."* A time-out expressed as a refusal.
T5 was reposed to external flow on 2026-09-17 and has now been attempted once under that
request; this is its first real result rather than the unsatisfiable-fixture story the
old T5 had.

The other two refusals are **correct answers**: T6 (a STEP with `LENGTH_UNIT()` and no SI
prefix -- refusing beats guessing a factor of a thousand) and T25 (no dimensions stated at
all). Both score `passed: true`.

## 4. The gate is live, and conditional

This is the first sweep in which the desk-facing gate worked at all. Before this it ran
`cad_audit.py` from `<workspace>/.toolbox`, which this desk is measured on not having, so
every case reported `n/a`. The scripts are now staged over the backend for the length of
the gate's call and removed after it.

Across the 26 runs:

| | |
|---|---|
| gate measured something | **15** |
| gate entirely `n/a` | 5 (T1, T7, T8, T12, T13) |
| no declare reached | 6 (the four refusals, T15, T19) |

Of the 15 live runs: **147 `clean`, 2 `xfail`, 1 `waived`, and zero `warned`.** No warning
fired anywhere in the corpus.

**The five `n/a` runs are the finding.** All five say "no patch set", and `_cad_command`
guards on `constant/triSurface/patches.json` -- so the gate is blind on exactly the cases
where the desk exported patches without using `export_patches`. T12 shows the asymmetry
plainly: its gate said "no patch set" while the supervisor's own probes measured 4,368
triangles on the same directory, clean on closure, normals, scale and self-intersection.
The supervisor reads the STLs; the gate requires the manifest.

**So T12's `no -> yes` cannot be attributed.** In the baseline it was failed by
`coverage` warning on a correct partition -- a bug in `gate.concern_of`, since fixed. Here
`coverage` never measured at all. Two different reasons it would not warn, and this sweep
cannot say which one recovered the case.

**And zero warnings is not yet evidence of clean geometry.** One sweep cannot separate "the
desks are producing sound surfaces" from "the gate under-reports". The two `xfail`s are
the only sign the channel carries anything: a desk predicted a flag and got one.

## 5. What the pass rate still does not mean

**73 named properties across the sweep, zero with a measured value.** `properties[].measured`
is written by nothing; the runner sets it to `None` and the grading that would fill it is
the `cad-supervisor` subagent's judgement, which did not run here.

So `passed` means `stopped == "done" and checkmesh_ok`, and nothing else. It is a
statement that OpenFOAM accepted the mesh, not that the desk answered the request. Every
case's `## Properties, measured on the delivered mesh` checklist is unexercised, and the
one recurring finding of the previous sweep -- `named_requirement_dropped_without_mention`
-- cannot be detected by this table at all.

## 6. What to do next, ranked

1. **Clean the 302 stale workspaces** (10 GB, `/` at 98%) and re-run T11. One case in this
   corpus currently has no verdict.
2. **Decide what the gate does without a manifest.** Either `export_patches` becomes
   something the brief insists on, or the gate reads the STLs the way the supervisor's
   probes do. As it stands the advisory layer is off for a fifth of the corpus and says
   so only in the record.
3. **Pair sweeps on the request text, not the case name.** A sweep records `git_sha` but
   not what each case asked; a hash of each `## Request` in the manifest would make
   unpaired cases mechanical instead of something a reader has to remember. Three cases
   were silently mispaired here.
4. **The declare is costing runs their budget.** Two cases in two sweeps had acceptable
   geometry and no turn left to declare it.
5. **Property grading, or drop the claim.** Either the supervisor pass fills those 73
   fields or the corpus stops describing itself as having per-case acceptance criteria.
