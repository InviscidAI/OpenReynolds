---
name: cad-sweep
description: Run the CAD build-up corpus through the core desk once per case and produce the failure report the next decision is made from. Use when asked to sweep the corpus, measure the desk against the cases, establish a baseline, or find out what is failing and what a tool would have to close. Produces a ranked failure report; it never decides what to add.
allowed-tools: [Bash, Read, Grep, Glob, Write, Edit, Task]
---

# Sweeping the corpus

One pass of the corpus through the desk as it currently stands, and a report of what
failed, ranked by what it cost. `docs/cad-build-up-handoff.md` is the argument behind every
rule here; `docs/cad-buildup/README.md` is the machinery.

**What this produces is a decision input, not a decision.** You name failures and what they
cost. The person reads the report and decides which addition to make. Do not propose tools,
do not add one, and do not rank by how fixable something looks.

## Before you start

```bash
python3 scripts/cad_supervise.py preflight /work        # the well-known path, not the workspace
git status --short                                      # a dirty tree makes the sweep uncitable
```

A sweep records the git sha of the core it ran. If the tree is dirty, say so and ask
whether to commit first — a comparison against this sweep later will not be able to say
what "the same core" meant.

## Running it

```bash
python3 scripts/cad_sweep.py run --label core --cases all --parallel 2
python3 scripts/cad_sweep.py run --label core+audit --cases all --baseline latest
```

**`--baseline` is how an iteration joins the chain.** Past the first sweep, every sweep
follows one: iteration N's sweep is iteration N+1's baseline, because the "after" of an
addition is the "before" of the next. Pass `latest` for the previous iteration's, or a
sweep id for a particular one, and it is recorded in the manifest — so `table` compares
against it without being told again, `list` shows the chain, and nobody has to remember an
id a week later. `python3 scripts/cad_sweep.py list` prints what is on disk, newest last,
each with the sweep it follows.

The first sweep of a corpus has no baseline and takes none.

- **One run per case.** `--repeat` is for two narrow jobs only: calibrating variance, and
  asking whether an addition closed one specific intermittent failure. Neither is this.
- `--label` says what is under test (`core`, `core+cad_convert`) and becomes how the sweep
  is cited later. Never leave it as `core` for a sweep that is not the core.
- Each case takes roughly 6–12 minutes and a dollar or several depending on the model. Say
  the expected cost and duration up front, before starting a full corpus.
- **A run that aborts on a dirty environment voids the sweep.** The driver stops and exits
  3. Do not clean up and resume — one contaminated run can be discarded, but an environment
  that was dirty at run 6 was dirty at run 1. Report it and start again.

The driver launches `cad_buildup.py run` and `cad_supervise.py watch` as two processes per
case and then calls `observe`. **The supervisor is that `watch` process, not the subagent** —
so do not start your own watch on a swept run. It already has one, a second would evaluate
the same alarms and overwrite the same `watch.json`, and `watch` refuses with exit 2 anyway.
The subagent's job here begins when a run has finished.

You are not an observer of a live run either: do not read a case directory while it is
working, and do not call a probe yourself.

## Classifying what failed

For every run that did not end `done` with `checkmesh_ok`, get one structured finding —
and get a mesh vet for **every** run, including the ones that ended green, because a run
that finishes clean is exactly where a silent failure hides. A probe reading is an input to
that vet, never a trigger for it: the probes returned three readings that looked like
failures and were not, and stayed silent on the two runs that were. Use the `cad-supervisor` subagent, one run at a time,
so each classification reads one record rather than the whole sweep:

```json
{"failure_id": "block_topology_not_closed",
 "statement": "one sentence: what went wrong, in the desk's own evidence",
 "evidence": ["***Open cells found, max cell openness: 1, number of open cells 8"],
 "steps_burned": 9, "case": "T1", "run": "<run-dir>"}
```

Four rules on `failure_id`, all of which decide whether the report is readable:

- **An id names a cause, not a symptom.** This is the one that decides whether the report
  is actionable, and it is the easy one to get wrong, because the symptom is what the
  evidence shows you. Three runs that each ended on the step budget while chasing a
  warning look identical in the record and are not one failure: one had a mesh it could
  never clean, one diagnosed a 51.7 µm tangency correctly and ran out applying the fix,
  one chased a cosmetic winding warning it should have waived and made it worse. Those
  need three ids, because an addition that closes any one of them closes neither other.
  **Before reusing an id, check the cause matches, not just the symptom** — and before
  coining one, check you are not splitting a single cause across two names.
- **Reuse an existing id wherever the cause is the same.** Read `findings.jsonl` from
  previous sweeps first. The same cause under three names never clusters, and a report
  that does not cluster cannot be ranked.
- **The id names the failure, never the remedy.** `block_topology_not_closed`, not
  `needs_blockmesh_helper`.
- **A new id is a claim that no known cause matches.** Say why in the statement, in terms
  of the cause — "this is X rather than Y because ..." — not in terms of the case.

When one run's symptom has two causes behind it, file two findings against that run. A run
is not limited to one.

Write them to `<sweep-dir>/findings.jsonl`, one per line.

## The report

`<sweep-dir>/report.md`, in this order:

1. **The sweep**: label, model, effort, core sha, cases, how many ended `done`, how many
   passed `checkMesh`, total cells, total spend. From `python3 scripts/cad_sweep.py table
   <sweep-id>`. If this sweep has a baseline, that same command also prints the paired
   comparison against it — put it here, and read the sign test rather than any row.

   **The harness's pass count is not this section's headline, and must never be stated
   without the vetted count beside it.** `passed` means the run ended `done` and a bare
   `checkMesh` liked the mesh. It does not mean the case was satisfied: on the sol corpus
   21 of 26 passed and the vets in §5 broke three of those outright. Write §1 last, after
   §5 and §6 exist, and give the number as a pair — *"21 of 26 by the harness; N of 26 survive
   the vet"* — with the vetted figure carrying the emphasis. A reader who stops after §1
   must not come away with a number the rest of the report takes back. The same rule
   governs how the sweep is reported to the person in conversation: lead with what
   survived, not with what the gate said.
**`checkMesh -allGeometry` is a reference reading, not a pass criterion.** The binding gate
is a bare `checkMesh`. `-allGeometry` runs checks the bare form does not run at all -- cell
determinant, face interpolation weight, concave cells, face tets -- and fails 14 of the 22
meshes the gate passes on this corpus, at 1.7-6% of cells. Nine of those were put through
`simpleFoam`: four converged, four were still converging at the iteration cap, none
diverged. snappyHexMesh made 5 of the 14 and gmsh 6. Report the **fraction of cells** it
flags where that is informative -- 12 faces of 678,227 and 6.14% of cells both print
`Failed N mesh checks.` -- and never rank a case as failed on it, or count it against the
pass rate.

2. **Contamination**: the rate. If it is not zero, stop here — the numbers are not
   measuring what they claim, and the rest of the report is void.
3. **Failures, ranked by cost** — cases hit × cells burned × spend. Each one: the id, the
   statement, which cases, the evidence lines, and whether a probe fired alongside it. Cost
   is the ranking because cost is what an addition buys back.
4. **Probes**: what each one *measured*, and on how many cases it could read anything at
   all. They return numbers and `n/a` now, never `fired` or `pass` — so this section
   reports readings, and a reading is not a finding. A probe that returns `n/a` across the
   corpus is not screening anything, and a corpus where most do is one that is not
   exercising Layer B; say that out loud rather than letting a column of ids read as
   coverage.
5. **The mesh vet**, per case: is the delivered mesh the volume the case asked for, on
   which number, and what could not be established. This is the section that catches
   silent failures — the probes did not, in the sweep that established this corpus, and
   two of the eight runs were wrong in ways no instrument could see.
6. **Properties**: read this out of `record.json`, not out of `cells.log`. The supervisor
   measures each named property off the delivered artifacts and writes its own answer with
   `cad_supervise.py grade`, so the record carries `measured`, `source`, `verdict` and
   `desk` per property. Report two numbers per case and keep them apart:

   - **does the property hold** — `holds` / `differs` / `unmeasurable`, which is about the
     geometry;
   - **did the desk measure it** — `printed` / `absent` / `printed-from-input`, which is
     about the run.

   They diverge constantly and that divergence is the finding: a property can hold on a
   mesh the desk never measured. `printed-from-input` is the corpus's most common failure
   and no instrument catches it — a `requested / measured` pair whose two sides derive
   from the same constant cannot disagree with itself, whatever was built.

   Before this existed the field was inert on every sweep on disk — 136 of 136 here —
   so if a record you are reading still carries `measured: null` throughout, the run was
   never graded and the section says that rather than inferring from `cells.log`.
7. **Candidate corpus cases**: failures the corpus does not currently cover. A failure
   without a case is a case, not a tool.

Then hand the person the ranked list and stop. The next step is theirs.

## What not to do

- Do not retry a case to get a better number. Every attempt is a data point; re-running
  until green is how a corpus becomes a story.
- Do not average a contaminated run back in, or quietly drop it — name it.
- Do not read a single case's result as a result. With one run per case, a flip is
  sampling; the corpus is the unit.
- Do not change two things between sweeps. If the model or the effort moved, the sweep is
  not comparable to the last one and the table says so.
