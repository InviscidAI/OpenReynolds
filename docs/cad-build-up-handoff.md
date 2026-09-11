# CAD agent — the build-up phase (hand-off, 2026-09-11)

The plan this replaces built the desk top-down: a toolbox, a brief naming every instrument,
and a finish check carrying every criterion the old desk had. Measured against a bare
kernel with none of it, that desk used **three times the steps on eight of eight prompts**
(p = 0.008) and produced **no more meshes** (p = 0.375), with **85% of its discovery spend**
going to learn its own surfaces rather than the domain. The overhead is close to fixed, so
it dominates wherever the task is easy, and most tasks are.

So the direction reverses. **Start from the smallest thing that works and add only what a
measured failure demands.** Every addition carries the failure it closes and a test that
demonstrates that failure in its absence; anything that cannot produce its own failure is
not earning its place.

The numbers above are in `docs/cad-acceptance.md` and the probe records under
`docs/cad-probe2/`. Read them before changing the design, not after.

---

## 1. The core

The starting desk is a kernel, a cell log, a brief, and **`checkMesh`**. Nothing else.

`checkMesh` is in the core because it is OpenFOAM's own verdict on an OpenFOAM mesh: it
generalises to any case on this stack, it needs no house format, and it costs no discovery
— the agent already knows it. Run it **per region** (`checkMesh -region <name>`, discovered
from `constant/*/polyMesh` with `constant/polyMesh` as the fallback), because a conjugate
case has no singular mesh and the old desk called such a case "nothing has been meshed yet".

It stays the **sole authority on mesh quality**. Do not re-decide its verdict against
non-orthogonality or skewness thresholds. Two authorities on one number is a desk being
told contradictory things about a mesh OpenFOAM has already judged.

The brief carries the inheritance rules and **one line of per-turn scope discipline** —
send one short runnable cell, look at what it prints, build on it next turn. That sentence
was the single largest measured effect of anything tested: on the hardest prompt it turned
three consecutive total failures into a completed mesh. It costs nothing and it is not a
tool.

**What is deliberately not in the core:** the instrument catalogue, the recipe folders, the
per-patch manifest, the rebuild-script and render gates. Those are candidates, not floor.

---

## 2. Layer B — the silent failures, watched but not wired in

Some defects never present as failures. Mesh the outside of the part and `checkMesh` passes
a perfectly valid mesh of the wrong volume. A leak in the exported surface gives the wrong
fluid volume and still checks clean. These cannot be discovered by waiting for a failure,
so they cannot wait for the build-up loop to surface them.

The resolution is to **separate detection from delivery**.

**On every evaluation run, the harness probes for every known silent failure.** The agent is
not told, the checks are not in its brief, and nothing is mounted in its workspace. The probe
result is graded into the run record only.

**A check is activated for the agent only once its probe has actually fired** — that is, once
some run has produced a silently-wrong result of that kind. On that trigger, and not before:
the criterion joins the finish check, and the measuring script becomes reachable.

A probe that never fires across the whole corpus stays dormant forever, and that is the point.
It is the tool rule made mechanical: the agent pays for a check only after the failure it
catches has been observed to happen.

### The probe registry

Keep one file, `docs/cad-silent-failures.md`, with a row per probe:

| field | meaning |
|---|---|
| `id` | stable name, cited from run records |
| `catches` | the silent failure, stated as what goes wrong and why nothing errors |
| `detect` | how the harness measures it, on the case as left on disk |
| `state` | `dormant` or `active` |
| `triggered` | first run id and date that fired it, empty while dormant |
| `activated` | what was given to the agent in response |

Start the registry with at least these, all `dormant`:

- **`location_in_mesh`** — the meshing point sits outside the fluid, so the mesh is of the
  volume around the part. Detect by testing the point against the exported surface after the
  run.
- **`union_closure`** — the exported patch set has free edges, so the meshed volume is not
  the one intended. Detect on the union, never per file: individual patch files are open
  surfaces by construction and a per-file check passes nothing real.
- **`normals`** — inconsistent outward orientation turns a solid into a void for snappy.
- **`coverage`** — a face assigned to two patches, or to none. An unassigned face lands
  silently in a default patch and takes whatever boundary condition it carries.
- **`scale`** — the case is a factor of a thousand from the dimensions the request stated.
  Detect against the request, which means the request must state them.
- **`self_intersection`** — the exported surface crosses itself.

The implementations already exist and are tested — `cad_audit.py`, `domain_probe.py` and
`surfaces.py` from the previous round. **Use them harness-side.** They do not go near the
agent until their probe fires.

---

## 3. Tool isolation, and why not naming them is insufficient

In the last round the bare arm was given no toolbox and found one anyway. It located
`/work/.toolbox` through a stale symlink, read `cad_convert.py` with `sed` and `grep` to
recover a function signature, and `cat`-ed a copy-modify-run template that the brief no
longer mentions. A `find /` for a house filename turned up leftovers from unrelated test
runs.

**Absence has to be enforced and then verified, not assumed.**

*Before the run:*
- Do not sync the toolbox into the workspace. Not to a hidden path, not read-only.
- Give the run a fresh workspace root per case. Never reuse a directory another arm has used.
- Resolve the workspace root and assert no house file is reachable beneath it.
- Check `/work` and any well-known path the brief has ever mentioned: if it exists and
  resolves anywhere, the environment is dirty — abort rather than run.

*After the run, and this is the part that was missing:*
- Grep the whole cell log and every captured output for house filenames, the toolbox
  directory name, and the repo path.
- A run that touched any of them is **contaminated**. Record it as such and **discard it from
  the baseline** — do not average it in, and do not silently retry it.
- Report the contamination rate. If it is not zero, isolation is broken and the numbers are
  not measuring what they claim.

The same discipline applies in reverse for the toolbox arm: a run that failed to reach a
tool it was supposed to have is equally invalid, and for the same reason.

---

## 4. The corpus, and promoting what works

The eight prompts are a starting set, not a specification. Under build-up the corpus *is* the
design input, so a fixed corpus means a design overfitted to it — and the current one has
known holes: one case fails in every configuration, and the refusal path is exercised once.

**Grow it.** Each case carries the request verbatim, the properties the desk must measure and
print, the fixture if any, and the expected outcome — including the cases whose correct
outcome is **not finishing**, like a STEP with no declared unit, where a mesh is the failure
and reporting up is the pass. Judge every run on its named properties, not only on the finish
check: a property you did not measure is a property you did not build.

**Promote successes into the library.** When a case passes, its accepted cell log is a worked
example of a recipe that actually ran. Save it with its provenance — model, core version,
date, run id — and the properties it satisfied.

Two constraints on promotion, both learned the hard way:

- **Replay before promoting.** Re-run the saved script from empty and compare the geometry it
  produces against what was accepted. A log that only works because a checkpoint file survived
  is not a recipe. One prompt in the last round produced a correct mesh and failed exactly
  this check.
- **Store worked examples, not scripts to copy and edit.** The copy-modify-run template is the
  pattern the kernel decision removed, and the desk went looking for one anyway when it got
  stuck. A promoted case is reference material the agent may read, not a parameterised file it
  edits and runs.

And a promoted template is a tool like any other, so it lives under the same rule: it earns
its place if having it measurably reduces steps on related cases. If the library grows without
that check, it becomes the discovery cost this whole phase exists to remove.

---

## 5. Babysitting, done properly

The supervision used in the last round was unreliable and should not be copied. Its two
failures are worth stating because both are easy to repeat:

- `until ! pgrep -f '<pattern>'` **matched the watcher's own command line**, so `pgrep` always
  found at least itself, the condition was never true, and the loop could never exit. Watchers
  spun until timeout regardless of what they were watching.
- The pathology that actually mattered — a run producing replies but **zero executed cells** —
  went undetected for three consecutive runs. Each burned its full clock and reported an
  ordinary budget exhaustion.

**Never match a process by its command line.** The child writes a pidfile; the supervisor uses
`kill -0` on that pid. That is the whole mechanism and it cannot self-match.

**Heartbeat on turns, not on steps.** The run appends a line after **every model turn**,
whether or not a cell executed, carrying: turn index, executed-step count so far, `stop_reason`,
output tokens, and whether a fenced block was found. A step-based heartbeat cannot see the
failure above, because there are no steps.

The supervisor raises three distinct alarms, each a named terminal state:

| alarm | condition | means |
|---|---|---|
| `wedged` | heartbeat age > `HEARTBEAT_STALE_S` | the process is stuck or dead |
| `no-progress` | turns advancing, executed-step count flat for `K` turns | replying but never running anything |
| `starved` | `K` consecutive turns with `stop_reason=max_tokens` and no text block | reasoning is consuming the whole reply budget |

`no-progress` and `starved` are the ones that were missed. Set `K` small — three is enough;
the failure was 8 for 8 — and abort with the reason in the record rather than letting the clock
run out and reporting `time`.

**Every terminal state is classified and recorded**: `done`, `steps`, `time`, `provider`,
`wedged`, `no-progress`, `starved`, `contaminated`. A run that ends without a classification is
a bug in the harness, not a result.

### Run records

Two things the last round got wrong in ways that cost real time:

- **Flat scalars at the top level** — `first_mesh_step`, `mesh_exists`, `checkmesh_ok`,
  `n_steps`, `n_turns`, `seconds`, `usd`, `stopped`, `contaminated`. Put the step trace under a
  separate key. A trace stored under `steps` gets dumped whole by any naive reader.
- **Persist every raw reply** — full text, thinking length, `stop_reason`, per-turn tokens,
  block types, whether a fence closed. Without these the `starved` failure is undiagnosable;
  with them it took one look. Write them incrementally so a killed run keeps what it had.

---

## 6. What this phase is measuring

The question is not whether the desk can be made to pass the corpus. It is **which additions
actually move it**, measured one at a time against the core.

Report, per addition: the failure it closes, the run that triggered it, steps and spend before
and after on the affected cases, and whether anything else regressed. An addition that cannot
show a difference is a candidate for removal at the next audit, and the audit is cheap because
every tool ships with the demonstration of its own failure.

Two things to hold onto while doing it:

**The arms do not produce the same deliverable.** A bare kernel produces a mesh. The desk is
supposed to produce a mesh *and* the evidence that it was examined. Where the step gap is
partly paying for verification, say so rather than reporting it as waste — 34% of the toolbox
arm's budget went to inspecting its own work against 21% for bare.

**Model capability dominated every tooling effect measured.** At matched tooling the stronger
model reached a mesher at step 1 where the weaker reached it at step 8, and the weaker model
with the full toolbox was the worst of four configurations — worse than the weaker model with
nothing. Fix the model before concluding anything about the tools.

---

## 7. Open, and deliberately not settled here

- **Where Layer B's checks live once activated.** Folding a probe into the finish check makes
  it a gate; leaving it as a reported finding makes it advice. The first activation should
  decide it on the evidence of that case, and the decision recorded in the registry.
- **Whether the rebuild-script and render criteria are gates or defaults.** One prompt in the
  last round produced a correct mesh and failed on replay alone, and another failed partly on
  no picture having been drawn. Both are defensible product requirements and neither is a mesh
  verdict.
- **What a promoted template costs.** The library has to be readable on demand without being
  paid for on every prompt, and the last round showed the catalogue being front-loaded is where
  the fixed overhead comes from.
- **The corpus's own coverage.** One case currently fails in every configuration; whether that
  is a task worth closing or a task to record as out of reach is not yet decided.
