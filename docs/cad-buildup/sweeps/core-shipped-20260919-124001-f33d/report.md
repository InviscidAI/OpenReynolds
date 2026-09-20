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
an acceptable mesh and ran out of seconds before declaring. Scored a non-pass, which is
what a budget is for: a desk that cannot finish inside one has not finished.

**An earlier draft of this report called that "the declare costing a run its budget" and
that was wrong.** T15 made **zero** declares -- it never reached one. The claim was
imported from `core+cad_export` §3.1, where T12 genuinely did land its only declare on
turn 30 of 30 and was bounced by the `coverage` bug since fixed. One case, one cause,
already repaired. There is no pattern here and the recommendation that rested on it has
been withdrawn.

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

1. **Done.** 274 stale workspaces removed, 9.5 GB reclaimed, `/` from 98% to 94%. T11
   re-run separately; until it lands the corpus has no verdict on that case.
2. **Done, and it was not what §4 said it was.** Three things blinded the gate, only one
   of them the manifest: the shell guard silenced both scripts entirely; `cad_audit`'s
   `derived_manifest` was reachable only by importing the module, so a caller running it
   as a script got a refusal; and the gate hardcoded `constant/triSurface` while the
   probes search three candidates. T12 hit the first and the third -- one
   `fluid_preview.stl` at the case root. With all three fixed its gate reports eleven
   findings on the same 4,368 triangles the probes measured.
3. **Done.** A sweep now records `asked`, a digest of each case's `## Request` and its
   properties, and the table excludes cases whose question moved. The provenance sections
   are deliberately not in the digest: editing a note is not a reposing. A baseline
   without the field says so rather than reporting agreement it cannot check.
4. **Withdrawn.** See T15 above -- the case it rested on made no declares.
5. **Left as is, deliberately.** The 73 fields stay unfilled and `passed` stays a
   `checkMesh` statement. Recorded here so the next reader knows it is a decision and not
   an oversight.
