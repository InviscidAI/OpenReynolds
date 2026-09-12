---
name: cad-addition
description: Make one addition to the CAD core desk — a tool, a brief line, a check — carrying the failure it closes and a test that demonstrates that failure in its absence, then re-sweep the corpus and report the paired comparison against the baseline. Use when a decision has been made about what to add after a cad-sweep report, or when asked whether an addition earned its place.
allowed-tools: [Bash, Read, Grep, Glob, Write, Edit, Task]
---

# Adding one thing, and measuring whether it moved

The build-up phase runs the other way round from the last one: **start from the smallest
thing that works and add only what a measured failure demands**. So an addition arrives
with three things or it does not arrive.

1. **The failure it closes**, cited by `failure_id` and the runs it came from in a sweep's
   `findings.jsonl`. An addition with no run behind it is the top-down catalogue this phase
   exists to undo.
2. **A test that demonstrates that failure in its absence.** Not a test that the new code
   works — a test that without it, the thing that went wrong goes wrong. Anything that
   cannot produce its own failure is not earning its place, and this test is what makes the
   next audit cheap.
3. **A measurement**, which is the rest of this skill.

**One addition at a time.** Two changes in one sweep is a sweep that cannot say which one
did anything.

## Making it

Read the finding and the runs it names before writing anything — the evidence lines, the
cells around them, what the desk was trying. Then the smallest change that closes it:

- a line in the core brief (`openreynolds/buildup/core.py`) costs nothing at run time and
  was the largest single measured effect anything had last round;
- a criterion in the core check, if the failure is one `checkMesh` cannot see;
- a tool in the workspace, which is the most expensive option and is what the last round
  showed costs 85% of the discovery spend to learn. Reach for it last.

If the failure is a silent one, its probe is in `docs/cad-silent-failures.md` and this is
the activation the registry is for: record the run id and date that fired it, move `state`
to `active`, and say in `activated` what the agent was given. §7 leaves it open whether an
activated check is a gate or a reported finding — decide it on the evidence of that case
and write the decision down.

Run the suite before you measure anything. A red suite makes the sweep uninterpretable.

## Re-sweeping

```bash
python3 scripts/cad_sweep.py run --label core+<what-you-added> --cases all --baseline latest
python3 scripts/cad_sweep.py table <new-sweep-id>
```

`--baseline latest` names the sweep this one follows and records it, so the `table` above
needs no `--against`: the chain is in the manifest rather than in somebody's memory. Pass a
sweep id instead of `latest` when the previous sweep is not the right baseline — a re-run
baseline after the corpus grew, or a comparison two additions back. `--against` on `table`
overrides whatever was recorded, for the comparison nobody planned.

**The baseline is the previous sweep, and past the first iteration that is the previous
addition's own sweep** — the "after" of iteration N is the "before" of iteration N+1. That
chaining is valid only while nothing else moved, so check all four before you use it:

- the same **corpus** (a case added since is unpaired, and the table excludes it — if many
  are, re-run the baseline instead);
- the same **model and effort** (the table warns, because model capability dominated every
  tooling effect measured last round);
- the same **machine and desk budgets**;
- the baseline's core sha is the one this addition was made on top of.

If any of those moved, the chain is broken and the baseline has to be re-run. Say so rather
than comparing anyway.

## Reading it

The table gives per-case cells before and after, a delta marked `fewer`, `more` or `noise`,
and a sign test over the cases that moved.

- **The verdict is the sign test, never a row.** With one run per case a single flip is
  sampling. Measured on T1 with Opus at medium, the same prompt lands within three to eight
  cells of itself, which is where the noise band comes from; it is one number from one case
  and it is worth re-measuring when the model or the brief changes.
- **Check the cases the addition was not aimed at.** If a tool for STEP unit traps moves
  the 2D authoring cases, that is a regression, and with one run across many cases it shows
  as a pattern even though no single case proves it. This check is free here and is most of
  what the breadth is for.
- **`checkMesh` outcomes are the other axis.** Fewer cells and fewer meshes is not an
  improvement.

## Reporting

Append to `docs/cad-buildup/README.md` and write `<new-sweep-dir>/report.md`:

- what was added, and the failure and runs it cites;
- steps and spend before and after on the **affected** cases, and on the rest;
- the sign test over the corpus, and whether anything regressed;
- the test that demonstrates the failure in its absence, by name.

**An addition that cannot show a difference is a candidate for removal at the next audit.**
Say that plainly when it happens rather than keeping it because it seems sensible — that
sentence is the whole discipline, and the last round's desk is what it looks like when
nobody applies it.
