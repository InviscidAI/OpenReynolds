# The build-up phase, as it runs

`docs/cad-build-up-handoff.md` is the argument. This is what exists and how to drive it.

## Two processes, and the boundary between them

```
python3 scripts/cad_buildup.py run T1                     # prints its run directory
python3 scripts/cad_supervise.py watch <run-dir> --case <case-dir> --deadline 1200
python3 scripts/cad_supervise.py observe <run-dir> --case <case-dir> [--spec spec.json]
```

The **runner** enforces §3 and then gets out of the way. Before a model is called it makes
a workspace root no other run has used, asserts that nothing of ours is reachable beneath
it and that no well-known house path resolves anywhere, and **aborts (exit `3`) rather
than running** if either is false. It does not sync the toolbox — not to a hidden path,
not read-only, not at all — and the desk it drives is briefed without one. It writes
`run.pid` and a heartbeat line per model turn, and that is the whole of what it offers an
observer.

The **supervisor** is the only thing that observes: liveness and the three alarms, the
silent-failure probes, the contamination grep, and the grading. It runs in its own
process, reads the case off disk, and has no channel into the conversation. The
`cad-supervisor` subagent (`.claude/agents/`) drives it and does the two parts that are
actually judgement — whether each named property was measured and printed, and what a
fired probe should activate.

The split is the measurement, not tidiness. A check that lives in the harness is one
refactor away from living in the desk's path, and then the thing being measured has
quietly become the thing doing the measuring.

## The desk under test

`openreynolds/buildup/core.py` — a kernel, a cell log, a brief, and `checkMesh` per
region. No instrument catalogue, no recipe folders, no per-patch manifest, no rebuild or
render gates. `checkMesh` is the sole authority on mesh quality and nothing re-decides its
verdict. Additions arrive one at a time, each carrying the measured failure that asked for
it and a test that demonstrates that failure in its absence.

It is the same loop as the shipped desk (`openreynolds/cad/agent.py`), differing in
exactly five seams: the brief, the nudge, what verifies the finish, **the tools it is
offered** and **what happens when it declares itself complete**. The last two arrived on
2026-09-13 with the declare gate below; before that there were three, and the shipped desk
is still handed exactly what it was handed then.

## Where things are

| | |
|---|---|
| run records | `docs/cad-buildup/runs/<case>-<run-id>/` — `record.json`, `heartbeat.jsonl`, `replies.jsonl`, `cells.log`, `build.py` |
| workspaces | outside the tree (`~/.openreynolds-buildup/work` by default), one per case per run, never reused |
| the probe registry | `docs/cad-silent-failures.md` — every row `dormant` until its probe fires |
| the corpus | `tests/data/prompts/T*.md`, which §4 grows rather than freezes |

A record's scalars are flat at the top level — `n_steps`, `n_turns`, `first_mesh_step`,
`mesh_exists`, `checkmesh_ok`, `seconds`, `usd`, `stopped`, `contaminated` — and the step
trace is under `trace`, because a trace stored under `steps` gets dumped whole by any naive
reader.

`stopped` is always one of `done`, `steps`, `time`, `provider`, `wedged`, `no-progress`,
`starved`, `contaminated`. A run that ends outside that set is a bug in the supervisor,
not a result.

## What the alarms cover, and what they do not

`wedged`, `no-progress` and `starved` are about the conversation: a process that has stopped
saying anything, a desk replying and running nothing, a desk reasoning past its reply budget.
They are deliberately blind to what the compute is doing. A twenty-minute mesh and a
twenty-minute mistake look identical from outside, and the thing that tells them apart is
`checkMesh` at the end, not a timer.

So a run meshing uselessly is bounded by its own step and second budget and ends `steps` or
`time` with no mesh — not by an alarm. "No alarm fired" means the run behaved, not that it
worked.

Two asymmetries are handled rather than left to chance:

- **A long legitimate silence is declared.** The loop marks the stretches with no model turn
  in them — `checkMesh` per region (600 s each), the recovery replay (900 s) — with how long
  each may take, and the watcher allows exactly that plus a margin. Without it the 420 s
  staleness threshold would kill a run for being checked, and a conjugate case would trip it
  twice. Marks are liveness, not progress: they do not advance the turn and do not count
  toward `no-progress`.
- **A kill reaches what the run started.** A mesher is a grandchild — the desk runs cells in
  a kernel process and a cell starts `snappyHexMesh` from there — so a sweep's runs take
  their own process group (`--own-group`) and the supervisor signals the group. Otherwise a
  runaway mesh outlives the run that started it and is still going when the next case begins.
  A run that declared no group is signalled on its own pid only; a group is never derived,
  because under a sweep the derived one would be the sweep's.

## Reading a result

`stopped` is how the run ended; `checkmesh_ok` is what it produced; they are different
questions and a run can end `done` with a fired probe — that pairing is the whole reason
the probes exist. A **contaminated** run is discarded from the baseline: not averaged in,
not silently retried. If the contamination rate is not zero, isolation is broken and the
numbers are not measuring what they claim.

## The first run, and what it settles

`runs/T1-20260912-030109-450c` is the first case through the enforced path: the core
desk, a workspace with nothing of ours in it, and the supervisor watching from its own
process. It ended `done` — 18 cells, 370 s, $0.59, `checkMesh` clean, first mesh at step
13 — with **no contamination and no probe fired**, on a machine whose `/work` had been
carrying the last round's toolbox until that morning.

Two things it establishes and one it does not:

- **The core desk works with nothing.** No instrument catalogue, no recipe folders, no
  `cad_convert.py`. It wrote its own `blockMeshDict`, typed `frontAndBack` `empty`, and
  `checkMesh` reported two geometric directions — the 2D trap this case exists to catch,
  avoided without being told about it.
- **The isolation holds under measurement, not just under assertion.** The whole thread
  greps clean of every house filename, the toolbox directory name and the repo path.
- **It does not establish that the probes work on a real case.** All six came back
  `skipped`: a `blockMesh` case exports no surface, so there was nothing for them to read.
  `skipped` is recorded distinctly for exactly this reason, and a corpus that keeps
  returning it is a corpus that is not exercising Layer B. T5 and T6 are the cases that
  will.

The property grading is the supervisor subagent's and is not in the record's scalars. Worth
noting from this run: the desk printed passage width, leg length and bend angle against the
request, but derived the width from its own parameters rather than measuring it at the
inlet, mid-leg and crown as the case asks. That is the kind of gap `checkMesh` cannot see
and the finish check was never going to catch.

## The declared finish, and the advisory gates — added 2026-09-13

**The failure it closes** is `no_closure_assertion_between_export_and_meshing`, from
`core+bench26-20260912-133719-4bbd`. Nothing in the desk's path asserted that an exported
surface closes: `grep` for `open_edges`, `free_edges`, `is_closed`, `watertight`,
`manifold` and `closure` over `core.py` and `record.py` returned zero for every term.
`snappyHexMesh` is a Cartesian cutter, so it emits closed cells whatever the input surface
did; `checkMesh` then validates the volume mesh and passes. **A leaky surface does not
produce a bad mesh, it produces a good mesh of the wrong volume** — which is the one thing
the desk's sole authority is structurally unable to see. Four cases exported non-closed
surfaces and all four scored `passed: true`.

T26 is the sharpest instance and is the test: `union_closure` measured 259 free edges —
the tread-column tangency the case was written around — the record kept the number, and
nothing told the desk, because a probe is the supervisor's and the supervisor has no
channel into the run.

**What was added.** A second tool, `declare_complete`, and with it the two new seams:

| | |
|---|---|
| `_tools()` | what the desk is offered. `CadDesk` returns `[CELL_TOOL]` exactly as before; `CoreDesk` adds `DECLARE_TOOL` |
| `_declare()` | runs `checkMesh` **and** the six probes, binds on `checkMesh` alone, returns the rest as advice |

`checkMesh` is the only binding check and is absent from the `waive` enum, so waiving it
cannot be expressed. Everything else is reported and recorded and blocks nothing — which
is the answer §7 left open, decided on the evidence: four of the six probes have been
wrong at least once, so a gate built on them would block correct work, while a warning
built on them costs nothing when wrong.

**Waivers, and why the order matters.** The desk may name a check in `waive` with a
reason. Named *before* that check has ever fired, it is a prediction — falsifiable, made
without the result in hand — and records as `xfail`. Named *after*, it is a reaction by a
desk that now has an interest in dismissing the finding, and records as `waived`. The desk
states the waiver; `gate.evaluate` picks the label from `self._warned`, so the stronger
claim cannot be claimed. A named check that then does not fire records as `xpass`, which
is what makes blanket-waiving self-punishing: name all six, and five come back `xpass` in
the record.

**What it is not.** It does not make a desk want to refuse. T6 and T25 are the corpus's two
refusal failures and neither reached for refusal at all, so moving the terminal off
`print("CAD_REFUSED: ...")` onto a schema arm is an affordance argument, not a measured
one. It is recorded as such.

**The test that demonstrates the failure in its absence** is
`tests/test_buildup_gate.py::test_t26_measured_its_own_defect_and_nothing_told_the_desk`.
It reads T26's committed record, asserts `passed: true` beside 259 free edges, and shows
that `gate.render` on those same recorded probes produces the sentence nobody said. Twelve
more pin the labelling, the enum, and the path scrubbing — which is the one thing here
that voids a sweep if it is wrong, because gate text reaching the desk puts a house path
into the conversation that `scan_run` greps the whole thread for.

### What the sweep said — `core+declare_gate-20260913-124524-aa21`

**20/26 passed, 518 cells, $24.29, contamination 0/26.** Paired against `core+bench26` on
the 15 cases that pair: **13/15 → 11/15**, and cells rose — 8 up, 1 down, **p = 0.039**.

**It works.** T18 was told its surface had 9,364 free edges, closed it, declared again and
passed; in `core+bench26` the same case shipped 13,204 open edges as a pass with the number
never delivered. T2 pre-waived correctly and finished in one declare. `xfail` appeared twice
against zero in the version that dropped its advisory.

**And the sweep cannot attribute its own headline.** Six of the eight cell increases are on
cases the gate never spoke to, because the addition bundled four changes — the tool, the
gates, the rewritten `# Finishing` section, and the `union_closure` explanation. One at a
time is the rule and this broke it. The prediction on record beforehand was that cells would
not move; they moved, significantly.

The cost is real too: T12 and T22 were each shown an open-surface warning, went after it
rather than waiving it, and spent the rest of a 30-turn budget doing so. That is the rule
working — a warning nobody fixes or explains is no longer a finish — and it says the budget
was set for a desk that finishes at its first declare.

Full report and findings in the sweep directory. Next, one at a time: the `normals`
explanation (an `xpass` in T21 is behind it, and the `union_closure` precedent took that
probe's xpass from 2 to 0), then a clean re-measure, then the step budget.

## `core+reference-20260914-093903-3472` — the tool works and the corpus got worse

`b123d_api.md` handed to the core desk, with the repaired probes, the `normals`
explanation and the assumption-recording line riding along — four changes in one sweep,
deliberately, and it can attribute none of them at the corpus level.

16/26 passed against 20/26, 524 cells against 518, $26.36, contamination 0/26 with the
reference in every workspace and greped by 11 desks. Sign test `p = 0.125`.

The tool did its specific job: api-surface failures 27 → 13, `AttributeError` 11 → 2,
`NameError` 4 → 0, `ModuleNotFoundError` 1 → 0. The one line aimed at `'bool' object is
not callable` failed — 6 cases → 8 — and is a candidate for removal rather than rewording.

All six pass → fail flips ended `stopped=steps`, five at exactly 28 cells, and the gate
spoke to only one of them. Seven of 26 runs now die at the ceiling against four before,
on an unchanged median of 21 cells. **The step budget is the binding constraint and is the
next change with a measured failure count behind it.** T19 and T22 flipped back and are
consistent with the probe repair — T22 declared clean at 11 cells where it had been bounced
at turn 30 on 62,208 phantom flipped edges.

Preceded by the void `core+reference-20260914-025525-6a2b`, which found that the reference
was copied where no desk could reach it. Run with a kernel janitor, disclosed in §8 of the
report: kernels outlive their runs and exhausted the machine during the void sweep.

## Four model arms, and what the corpus can say about a model — 2026-09-15

Started from a question about the harness — *is our tool calling set up wrongly?* — because
kimi-k3 scored 2/26 where Opus 5 scored 16/26, and kimi-k3 benchmarks at or above Opus 5.
It ends somewhere else, and the route is most of the value.

### The arms

Ten cases (T1, T3, T5, T7, T11, T14, T16, T18, T22, T24), one run each, `medium` effort.

| arm | model | provider | adapter | passed | $ | gen tok/s |
|---|---|---|---|---|---|---|
| `core+reference-20260914-093903-3472` | Opus 5 | Anthropic | `anthropic` | 7/10 | 9.54 | 51.6 |
| `core-kimi-k3-20260914-143522-e17d` | kimi-k3 | Aster | `openai` | 2/10 | 3.33 | 14.8 |
| `core-kimi-k3-moonshot-20260915-060615-0057` | kimi-k3 | Moonshot | `anthropic` | 2/10 | 5.26 | 25.5 |
| `core-gpt-5.6-sol-20260915-044749-5e92` | gpt-5.6-sol | OpenAI | `openai-responses` | 8/10 | 5.12 | 39.3 |
| `core-kimi-k3-moonshot-1800s-...-a25f` | kimi-k3 | Moonshot | `anthropic` | 3/7 at 2x clock | 4.21 | — |

Contamination 0 in all four.

### What was eliminated, and how

- **The adapter.** kimi-k3 scores 2/10 on Chat Completions at Aster *and* on the Messages
  API at Moonshot — the same adapter Opus 5 passes 7/10 on.
- **The shared layer.** gpt-5.6-sol passes 8/10 through a third adapter, so the loop, the
  block helpers, the brief and the corpus all work for a non-Claude model.
- **The brief.** `88a393f`; both Moonshot arms ran on it.
- **The provider.** Moonshot is first-party and 1.7x faster than Aster.
- **Throughput and the clock.** Doubling the budget converted one case of seven.
- **The geometry.** Failing kimi-k3 runs delivered meshes correct to 0.05-0.4%.

### The conclusion, with its scope

**On this corpus, kimi-k3 is materially weaker than Opus 5 or gpt-5.6-sol, principally
through fluency in the library the corpus is built on.** It exits non-zero on 22% of cells
against Opus 5's 6%, about two thirds of those `AttributeError`/`TypeError`/`ImportError`
on build123d. Per-run classification of the doubled-budget arm found **four failures with
four proximate causes** — a truncated `checkMesh` stdout, a zero-tolerance weld producing a
false positive, and two instances of per-face `export_stl` leaving seams unwelded. That
diversity is itself the evidence: one mechanism would be a bug, four unrelated ones is a
capability profile.

**It does not support "kimi-k3 is a worse model."** This corpus scores build123d, OpenFOAM
dict syntax and a declare protocol, none of which a benchmark measures. Both things can be
true at once.

### Three limits, stated because they are not small

1. **n is 10, then 7.** The 2-to-3 conversion at double budget is one case.
2. **The cores differ.** Opus 5's 7/10 ran the old brief; the Moonshot and sol arms ran the
   fixed one. No arm pair is a clean single-variable comparison.
3. **The arms are unevenly audited, in the direction of the conclusion.** The adversarial
   mesh vet — *try to break the green result* — was run on kimi-k3's passes and found T1
   printing `min(legx.max(), 0.12)*1000`, clamping a wall extent that genuinely reaches
   0.14 m down to the requested 120 mm, plus a hardcoded `0.030` dressed as a measured bend
   radius. **That vet was never run on Opus 5's seven passes or sol's eight.** If they carry
   the same thing the gap narrows. Closing this costs subagents and no model spend, and was
   left undone deliberately rather than overlooked.

### Harness defects found on the way, all fixed

Every one was a control that was accepted and then silently did nothing:

- `88a393f` — the core brief still described the fenced-block protocol `CELL_TOOL` replaced,
  and named `run_cell` zero times. Cost 3.5-9% of turns on every model.
- `47012e6` — a killed run's record dropped the case's named properties, so a sweep report
  read "this case named none" instead of "none measured".
- `4fc5fb7` — the IPython history db reported its own failures on the desk's stdout; off, the
  suite went 22:04 to 6:14.
- `43bc706` — `--seconds` moved the desk's budget and left the supervisor's deadline behind,
  so a longer run would have been killed early and filed `wedged`.
- `fb39d89` — the price table was keyed on model id alone; kimi-k3 is 20% dearer at Moonshot
  than at Aster.
- `4d9bb56` — `openai_api.py` cannot run any OpenAI model above gpt-5.2 with tools and
  reasoning; the Responses adapter exists because of that.

And two measured facts about endpoints worth keeping: **Aster accepts `reasoning_effort` and
ignores it** (low/medium/high returned 114/73/78 reasoning tokens), which is why the Aster
arm's meshes were wrong where Moonshot's were merely late; and **Anthropic's
OpenAI-compatible endpoint refuses adaptive thinking outright**, which is why Opus 5 could
not be used as the adapter control.
