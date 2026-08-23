# OpenFOAM Agent — Build Architecture

A specification complete enough to build from. The system takes underdetermined natural-language CFD requests ("simulate airflow through an L-shaped junction", "check this drone for stability", "verify this floorplan against ASHRAE"), turns them into approved specifications with executable success criteria, executes them as real OpenFOAM studies, and delivers evidence-linked results with stated confidence bounds — across sessions, over hours-long runs, without an LLM ever holding gigabytes of raw data in context.

---

## 1. Design principles → architectural commitments

| # | Principle | Commitment it forces |
|---|---|---|
| P1 | The question underdetermines the spec | Spec-with-provenance contract; blocking-question triage; user approval gate G0 |
| P2 | Ground truth is available for some cases and absent for most | Benchmark tier anchors the system to published experiment/DNS where it exists (§9); for everything else verification is a process claim — hand-calc expectations, conservation, QoI plateau, mesh independence, executable spec tests |
| P3 | Wrong decisions cost far more than deliberation | Gates with hard stops at spec and cost-commit; watchdog kills doomed runs early; cheap checks run before every expensive operation, including meshing |
| P4 | Any layer is an abstraction; OpenFOAM is the truth | The LLM reads and writes real OpenFOAM files directly; the grammar oracle is the installed binary itself, not a hand-written schema; digests are read-side projections only |
| P5 | State must outlive context | Filesystem is the memory; `state.json` + orient ritual; sessions are stateless and rehydrate from disk |
| P6 | Failure modes recur across studies | Failure taxonomy for classification; remediation = curated advice ∪ retrieved precedent; attempt graph with duplicate-launch guard |
| P7 | LLMs can't parse GBs, and unmediated reads have no audit surface | One reader over a projector registry (§6): *every* artifact type has an entry, large ones projected to md + PNG, small ones served as identity views. Uniformity is the point — identity checks, provenance and fidelity marks apply everywhere rather than only where someone wrote a digest. Raw access via shell with output caps |
| P8 | A provenance tag is unauditable if the utterance behind it was discarded | Context bank (§6.5): verbatim append-only transcript of user turns, system replies, approvals and amendments; `[user]` spec entries resolve to exact spans in it; the spec stays authoritative and the bank is evidence, not a second source of truth |

**The cost asymmetry, stated precisely.** Mistakes are cheap to fix in file authoring and expensive to fix after a build or a solve. The architecture therefore places checking immediately *before* each expensive operation and immediately *after* each authoring step:

| Operation | Cost of a mistake | Checks that must run first |
|---|---|---|
| Dictionary authoring | seconds | oracles 0–4 (§8.2), all sub-second |
| `snappyHexMesh` build | 20 min – hours | mesh pre-flight (§7.3) — deterministic, seconds |
| Production solve | hours – days | G3 oracle stack, G4 smoke or soft-smoke, cost gate |
| Reporting a wrong number | the whole study | G6 suite, benchmark regression, expectation gate |

Freedom on the write side is a capability decision (write-time schemas are lossy and rot); it is not a claim that all authoring errors are cheap. Meshing is the counterexample and gets its own pre-flight layer.

---

## 2. Actors

| Actor | What it is | Lifetime | Writes what |
|---|---|---|---|
| **Deliberator** | The LLM agent. One role, one system prompt. Interprets, decides, authors, judges, composes. | Short sessions, spawned per event, stateless between | everything inside the study sandbox except protected paths (§14) |
| **Subagents** | Fresh-context LLM calls as pure functions (files in → files out): standards→tests extraction, log forensics, adversarial spec review | One task | only their `task/outputs/` |
| **Executor scripts** | Deterministic tools: checkers, oracles, digest/render generators, launcher, verify instruments | Per invocation | digests, verdicts, events |
| **Watchdog** | Dumb daemon per run: tails logs/monitors, applies threshold rules, kills doomed runs, emits events | Run lifetime | `events.jsonl` only |
| **Supervisor** | Event loop: watches events + user inbox + timers, manages the study lock, spawns deliberator sessions, owns `state.json` writes | Persistent | `state.json` via the state service (§14.2) |

Rules that keep this safe:

1. **Single writer.** One deliberator session holds the study lock. Watchdog and executor scripts never edit case files or the spec.
2. **Subagents never touch state.** Task-folder contract; the deliberator validates and integrates outputs.
3. **Parallelism at the case level only** (e.g., six alpha-sweep cases under one watchdog), never at the deliberation level.
4. **No role-fragmented agent pipeline.** Phases live in the state machine; the same deliberator role runs every phase, rehydrating from disk.

---

## 3. Pipeline: the OpenFOAM-native gate DAG

Gates map onto the standard OpenFOAM workflow — a CFD engineer should read this as "the normal process, instrumented." The deliberator authors freely between gates; gates verify.

| Gate | OpenFOAM step | Deterministic verification | Deliberator does | User stop? |
|---|---|---|---|---|
| **G0 Spec** | — | analytical-first triage (§3.1); spec schema; expectation calcs exist with declared uncertainty class; regime classification recorded; `spec_frozen/tests/` parse & dry-run against fixture data; QoI instrument block present with declared dimensions and conversions; quantity-definition review verdict for every QoI | drafts spec + executable success tests + QoI instrument definitions; triages blocking questions; retrieves seed cases | ✅ approve spec, tests, QoI instruments, and simulate-vs-correlate decision (frozen after) |
| **G1 Geometry** | `surfaceCheck`, `surfaceFeatures` | watertight/manifold/self-intersections; bbox vs. spec scale (mm-vs-m); normals; region count and region *names* | authors/edits geometry code or canonicalizes upload; inspects renders | only if units/shape ambiguous |
| **G2 Mesh** | pre-flight → `blockMesh` → `snappyHexMesh` (staged) → `checkMesh` | mesh pre-flight (§7.3) before any build; then full staged checker battery; leak detector; layer coverage; y⁺ pre-estimate; patch contract; regime re-classification against realized geometry | authors mesh code/configs freely (seeded by retrieval); judges renders; authors fixes | — |
| **G3 Case probe & lint** | `foamDictionary -expand`, dict-probe, (`potentialFoam`) | oracle stack §8.2: parse → cross-file lint → keyword lint → percentile lint → pre-mesh probe (advisory) → **post-mesh 1-iteration probe on the real mesh (blocking)** | authors all dictionaries freely; queries selection oracle when unsure; resolves advisories or waives with note | — |
| **G4 Smoke** *(conditional — §10.3)* | short coarse solver run | residual trend; mass-closure order; bounding count; measured s/cell/iter → cost recalibration; scheme-sensitivity probe (§11.4) | go/no-go; recalibrates cost | ✅ approve production cost |
| **G5 Production** | solver + function objects, detached | watchdog rules §10.2; effective-config echo diff (written vs. announced) | wakes only on events: continue/kill/remediate/extend | escalation only |
| **G6 Verify** | `postProcess`, sampling, `yPlus` | hermetic pytest over frozen `spec_frozen/tests/` (§11.2) + system instruments (conservation, GCI, y⁺ coverage) + tiered expectation gate | pass / waive-with-justification / physics-revisit | only for waiver or physics change |
| **G7 Report** | — | case fingerprint from disk; results.json schema; confidence statement matches evidence actually present | composes report, limitations, postmortem | ✅ deliver |

Auto-advance G1→G4 on green. Hard stops: G0 and pre-G5 cost commit. Gate statuses change only through the `run_gate` tool, with evidence attached.

### 3.1 The analytical-first off-ramp

Before drafting a full spec, G0 asks whether simulation is the right instrument. For a bounded set of case classes there are published correlations that are cheaper, better-characterized, and often more accurate than a RANS solve: pipe and duct pressure drop (Idelchik, Crane TP-410, Darcy–Weisbach + minor-loss coefficients), fan and system curves, simple heat-exchanger duties, ACH and mixing-time estimates from supply flow, hover thrust and disc loading.

If the request matches such a class, G0 produces the correlation answer with its citation and uncertainty, and asks the user directly: *this is the correlation result; do you still want the simulation, and if so, what does it need to tell you that this doesn't?* Legitimate answers exist — spatial field information, off-correlation geometry, compliance evidence requiring distributions rather than a bulk number, a design sweep where correlation coverage runs out. But the question is asked, the answer is recorded in `spec.md`, and the study can terminate here with a one-page deliverable. A system that cannot reach the correlation is a worse engineer than the correlation.

When the study proceeds, the correlation value becomes the primary expectation for the Tier-3 gate (§11.5) and its uncertainty class sets that gate's tolerance.

---

## 4. Study bundle filesystem spec

The bundle is the memory (P5) and the deliverable. A fresh session with zero chat history must fully rehydrate from it.

```
study-001-elbow/
  context/                 # the context bank (§6.5) — the one non-regenerable artifact
    transcript.jsonl       #   verbatim user turns + system replies, O_APPEND, never rewritten
    approvals.jsonl        #   {gate, artifact, artifact_sha256, rendered_sha256, by, at, turn_id}
    amendments.jsonl       #   spec amendments with turn_id and affected sections
    request.txt            #   generated view: turn 1, kept as a stable path
    rollup/                #   generated per-read summaries of aged turns; disposable
  intake/attachments/      # STL/STEP, standards PDFs, datasheets
  context.yaml             # budget caps, hardware, deadline (optional)
  spec.md                  # the contract (§17), provenance-tagged
  spec_frozen/             # everything frozen at G0 approval — hermetic, hash-manifested
    tests/                 #   test_qoi.py test_compliance.py conftest.py
    qoi_instruments/       #   the function-object subdicts that produce spec QoIs (§11.1)
    conftest_root.py       #   pinned pytest root config
    freeze.manifest.json   #   sha256 of every file in the import + config closure
  predicates.md            # generated human-readable view of the frozen tests
  decisions.md             # ADR log: what/why/alternatives/revisit-if/seeded-from
  attempts.jsonl           # attempt graph
  attempts.md              # rendered tree view (generated)
  events.jsonl             # watchdog + system events (O_APPEND — see §10.4)
  state/                   # split state (§14.2)
    gates.json             #   written only by the state service
    budget.json            #   written only by the state service
    lock.json              #   written only by the supervisor; carries a lease
    workspace.json         #   deliberator scratch: notes, frontier, handoff draft
  calcs/                   # expectation calculations (python + outputs), with uncertainty class
  geometry/
    source/                # as-received upload or authored generation code
    canonical/             # unit-normalized, repaired STL + geometry.md
  runs/
    mesh-01/               # meshing treated as a run: code, logs, digests
    run-01-coarse/
      case/                # REAL OpenFOAM case: 0/ constant/ system/
      logs/                # raw logs (reachable via capped shell; digested by default)
      digests/             # *.md + *.png (§6)
      verdict.md
    run-02-medium/ ...
  report.md
  results.json
  postmortem.md            # taxonomy gaps, lint-rule candidates, playbook candidates
  manifest.json            # hashes, OF version + fork, script versions, decomposition, timestamps
```

**Orient ritual** (a literal tool, first call of every session): reconciled state pack → `spec.md` → recent `decisions.md` entries → `attempts.md` frontier → projection index with staleness flags → context-bank tail (verbatim recent turns, unprocessed user messages first). Bounded to a few KB; everything else is pull. Reconciliation is described in §16.2 — the orient pack never ships a handoff that predates unprocessed events.

---

## 5. Repository layout (the system itself)

```
ofagent/
  deliberator/system-prompt.md
  sandbox/                 # shell/python policy: output caps, protected paths, writable roots
  tools/                   # boundary-crossing tools only (§14)
  state_service/           # the only writer of gates.json / budget.json (§14.2)
  executors/
    mesh/
      preflight.py         # pre-build deterministic checks (§7.3)
      check_surface.py  check_castellation.py  check_snap.py
      check_layers.py   check_final_mesh.py
    lint/
      harvest_keywords.py  # greps $FOAM_SRC lookup patterns → keywords-<ver>.jsonl (ADVISORY)
      xfile_lint.py        # patch↔field matrix, refs, solver↔model compat
      percentile_lint.py   # corpus stats: flags statistically unusual values (ADVISORY)
      advisory_budget.py   # precision tracking, suppression list, per-gate volume cap
    oracle/
      selection_oracle.py  # foamToC/foamInfo primary; invalid-token error probe as fallback
      dictprobe.py         # pre-mesh trivial probe (advisory) + post-mesh real-mesh probe (blocking)
    regime/
      classify.py          # case-class + shedding-risk classifier (§11.4)
      scheme_sensitivity.py
    run/                   # launch.py, watchdog.py, decompose policy
    projection/            # the reader (§6)
      registry.yaml        #   one entry per artifact type: view_kind, round_trippable,
                           #   fidelity, identity, window
      read_artifact.py     #   the only read path; stamps provenance, fidelity, staleness
      generators/          #   one per view type, incl. effective-config echo parser
      context_view.py      #   context-bank windowing + rollup (§6.5)
    render/                # pvpython headless scenes, fixed cameras per case class
    verify/
      hermetic_pytest.py   # pinned-root, closure-hashed G6 runner (§11.2)
      qoi_recompute.py     # Tier-2 independent QoI extraction from raw fields (§11.1)
      gci.py               # Richardson + non-monotone handling (§11.6)
      conservation.py  yplus_coverage.py
      result_interfaces.py # returns dimensioned quantities; dimensions read from
                           #   field headers, carried through every operation
      dimensions.py        # unit algebra + declared/sourced conversions (§11.1)
  corpus/
    benchmark/             # published experimental/DNS cases + reference values (§9)
    curated/               # reviewed, CI-validated seed cases + manifests
    tutorials.index.jsonl  # harvested from $FOAM_TUTORIALS at install — version-matched
    studies.index.jsonl    # one row per completed study bundle
    search.py              # one retrieval interface over all tiers
  taxonomy/                # failure classes (labels) + playbooks-as-advice (§18)
  config/
    thresholds/<class>.yaml   watchdog.yaml   smoke_policy.yaml
    loops.yaml                advisory.yaml   fingerprint_exclusions.yaml
  supervisor/
  ci/                      # golden meshes, oracle mutation tests, fixture bundles,
                           #   benchmark regression suite
```

---

## 6. The LLM-visible layer

### 6.1 Access rule

**One reader, no bypass.** Every artifact type in the bundle has an entry in the projector registry (`executors/projection/registry.yaml`), and the deliberator reads through `read_artifact` rather than opening paths directly. Small text is still served as itself — an identity view whose registry entry is nearly empty — but it is served *through the registry*, which is the change from the earlier design. Previously dictionaries, indices and logs were read raw, meaning the files the deliberator edits most often were the ones arriving with no identity check, no provenance header and no staleness refusal. Uniformity buys three things: coverage becomes structural (a missing registry entry is an error, not a silent default), staleness applies to every read, and a projection's fidelity can be stated at point of use.

Raw bytes remain reachable through the capped shell: `grep`, `head`, `tail` and windowed reads always work, subject to per-command truncation configured in `sandbox/`. Projection saves tokens; it does not restrict access.

Each registry entry declares:

| Field | Values | Why it exists |
|---|---|---|
| `view_kind` | `identity` · `transformed` · `summary` · `summary+renders` | What the reader serves |
| `round_trippable` | bool | Whether the served form is editable back to disk |
| `fidelity` | `full` · measured 0–1 · `degraded` | A summary that drops what mattered is worse than none, because it looks like a check that ran |
| `identity` | `hash` · `hash-with-fast-path` | §6.3 |
| `window` | null · rule | Required for any source that grows without bound |

**`round_trippable` is load-bearing given P4.** The deliberator authors real OpenFOAM files forever, so the read and write paths are deliberately asymmetric and the asymmetry is announced rather than assumed. `foamDictionary -expand` output is a `transformed` view: better to reason over, catastrophic to write back, because includes are resolved and variables flattened. The reader stamps `NOT EDITABLE AS SERVED — edit <raw path>` on every non-round-trippable view.

**Unbounded text needs a window even though it is text.** `attempts.jsonl` and `events.jsonl` are small by type and large by the time they matter. Their entries declare recent-in-full plus rolled-up-older, and the reader states the window it applied.

### 6.2 Projector registry (inventory)

| Raw artifact | Typical size | View kind | Round-trip | Served as | Contents |
|---|---|---|---|---|---|
| `system/*`, `constant/*`, `0/*` dictionaries | KB | identity | ✅ | the file | provenance header + bytes |
| dictionaries with `#include`/`$var` | KB | transformed (on request) | ❌ | `expand:<path>` | `foamDictionary -expand` output; identity view remains the default and the editable one (§22.10) |
| `context/transcript.jsonl` | KB–MB | summary + windowed identity | ❌ | `context.md` | §6.5 |
| standards PDF in `intake/attachments/` | 1–50 MB | summary, **fidelity: degraded** | ❌ | `standard.md` | subagent-extracted clause→test list with source page ranges; marked degraded because nothing yet validates the extraction against the whole document (§15) |
| retrieved corpus case | KB | summary | ❌ | `precedent.md` | **tier and provenance in the header before any content**, so a convention inherited from the system's own earned tier can never read as benchmark-grade (§9) |
| `geometry/canonical/*.stl` | 1–500 MB | summary + renders | ❌ | `geometry.md` | bbox, units decision, area, triangle count/quality, watertight verdict, region count **and region names**, 4-view renders |
| `log.blockMesh`, `log.snappyHexMesh` | 10–200 MB | summary | ❌ | `mesh_build.md` | per-stage outcome, warnings classified to taxonomy IDs, per-patch layer coverage table, timings, predicted-vs-realized cell count |
| `checkMesh` output | small, noisy | summary + renders | ❌ | `mesh.md` | metric table vs. thresholds (fail/warn tiers), worst-cell locations, verdict, axis-cut renders + closeups at QoI zones |
| `constant/polyMesh/` | 100 MB–5 GB | summary + renders | ❌ | `mesh.md` | no separate view; represented by the `checkMesh` projection + renders |
| solver log | 10 MB–GBs | summary + renders | ❌ | `run.md` | residual plot PNG, last-iteration table, event classification, continuity errors, s/cell/iter, effective-config echo table (every "Selecting …" line + printCoeffs blocks, diffed against written dicts) |
| `postProcessing/` monitors | KB–MB | summary + renders **+ decimated series** | ❌ | `qoi.md` | QoI time-series plots, last-window mean/slope/CI, per-test acceptance preview. The decimated series is served alongside the plot because an `inconclusive` URANS growth-rate fit (§11.4.1) wakes the deliberator, and it needs the trace rather than only the verdict |
| field data per write time | 0.1–5 GB | summary + renders | ❌ | `fields.md` + `renders/*.png` | min/max/mean/σ per field, extrema cell locations, fixed-camera slices |
| `events.jsonl`, `attempts.jsonl`, corpus indices | KB–MB | identity, windowed | ✅ (append-only) | the file, recent-first | recent entries verbatim + rolled-up older; reader states the window applied |

### 6.3 Staleness

Every view records the identity of its source, and identity is a content hash in all cases. The earlier design demoted large artifacts to `(mtime, size, inode)`, which put the mesh and the geometry on a weaker check than the dictionaries while the rest of the system treats hashes as the primitive — an inversion of the risk ordering, and one that a CAD re-export at the same size can defeat.

The cost objection is answered by caching rather than by weakening the check. The hash is computed once at projection time and stored in the view header. `(mtime, size, inode)` is retained as a **fast path only**: unchanged triple ⇒ hash assumed current; changed or ambiguous triple ⇒ rehash before serving. This is `identity: hash-with-fast-path` in the registry, and it costs one full read per actual change rather than one per access.

The reader refuses a view whose source identity has changed and offers regeneration. The orient pack flags staleness. Reasoning from a stale view is blocked at the reader, not by convention.

### 6.4 Images are first-class

Fixed cameras per case class make run-to-run comparison a visual diff: mesh cuts, |U| and p slices, residual/QoI plots, layer-collapse closeups. Multimodal judgment ("the jet separates oddly", "refinement artifact at that patch") is cheap perception that would otherwise require brittle heuristics. The mesh loop is exactly: author code → pre-flight → build → checker verdicts + renders → judge → re-author.

### 6.5 The context bank

**The gap it closes.** §17 tags spec entries `[user]`, but the bundle held only `request.txt` — turn 1. Every subsequent thing the user said during clarification, triage, G0 review and cost approval was discarded when the session ended, so `[user]` on anything past the opening request pointed at nothing and was unauditable. §16.2 has the same problem from the other side: it promises "every spec amendment later than the handoff" against a store that was never specified.

Three things are lost without it, and none reconstruct from the spec:

1. the exact wording behind every `[user]` provenance tag;
2. remarks that never entered the spec at all — the offhand *"it usually runs at part load"* that turns out to decide a boundary condition at G6;
3. the approval record proper, which is a commitment about a specific rendered artifact at a specific time and needs the hash of **what the user was actually shown**, not merely of the file that now exists at that path.

**What it is.** `context/transcript.jsonl` is a verbatim, append-only record: every user turn and every system reply in full, exactly as exchanged, ordered and timestamped, interleaved with approvals and amendments. Nothing in it is ever summarized in place, edited or deleted. `request.txt` survives as a generated view of turn 1 so existing paths keep resolving.

It is also **the only artifact in the bundle that cannot be regenerated.** Meshes, projections, run logs, `results.json` and the report all reconstruct from the spec plus the code plus enough compute — that is what P5 and the G7 disk-reconstruction check assert. The conversation does not. The smallest file in the bundle is the one that most needs to survive, and it is the one to replicate first under any retention or backup policy.

**Four rules.**

| Rule | Mechanism |
|---|---|
| **Append-only, enforced** | Written only through the `append_context` boundary tool with `O_APPEND`, same discipline as `events.jsonl` (§10.4). `context/transcript.jsonl` and `approvals.jsonl` are protected paths (§14.1): the deliberator can append, never rewrite. A session that could edit the transcript could retroactively manufacture a `[user]` provenance for its own assumption |
| **A source, not an authority** | `spec.md` is authoritative. The bank is evidence about how the spec came to say what it says. On disagreement the spec wins **and** the discrepancy is surfaced in the orient pack as a candidate missed amendment. Without this, sessions relitigate settled physics out of raw chat, which is the failure P5 exists to prevent |
| **Summarization is a view, never a write** | Registry window rule: spans referenced by spec provenance links always served in full; the last *N* turns verbatim; older material rolled up. The rollup is generated per read into `context/rollup/` and is disposable. The stored stream stays complete underneath and is one shell call away |
| **Deliberator internals live elsewhere** | The bank holds the conversation as the *user* would recognize it, plus approvals and amendments. Session reasoning, subagent tasks and tool calls stay in `attempts.jsonl` and `events.jsonl`. Keeping them separate is what lets the bank stay short enough to read and clean enough to audit an approval against |

**Provenance links resolve into it.** A `[user]` tag in `spec.md` carries `turn_id` and a character span, and the quantity-definition review (§11.1) and G0 approval both display the resolved span next to the spec line. A `[user]` entry that fails to resolve is a G0 blocker, not a warning — an unresolvable tag means either the transcript was truncated or the entry was mislabelled, and both are conditions under which the spec should not be approved.

---

## 7. Geometry & meshing

### 7.1 Acquisition

| Route | When | Mechanism |
|---|---|---|
| **Uploaded** | user supplies STL/OBJ/STEP | canonicalize (STEP→STL via gmsh headless, controlled tessellation tolerance; units→m; normals; bounded repair loop) |
| **Authored** | everything else with definable geometry | the LLM writes code: a blockMeshDict directly, Python that emits one, or cadquery/gmsh scripts — seeded by corpus retrieval (§9), run sandboxed, verified by G1/G2 checkers + renders |
| **Refused** | "this drone design" with no file | blocking question at spec time — never invent geometry |

Curated parametric generators (`corpus/curated/generators/elbow90.py` …) are high-trust seeds the agent may copy and modify, not a required route.

### 7.2 Staged mesh verification

Meshing is the largest empirical failure source and carries the densest checker coverage. Each stage is a deterministic script producing a JSON verdict plus an md digest.

| Sub-gate | Step | Key checks | Failure classes |
|---|---|---|---|
| **G1a** | canonicalize + `surfaceCheck` | closed/manifold/self-intersections/degenerates; bbox vs. spec characteristic length (large ratio → mm suspected → auto-scale if spec permits, else ask); normals via signed volume; solid count and names | GEO-01..05 |
| **G1b** | `surfaceFeatures` (ESI dict-driven; `surfaceFeatureExtract` on older builds) | feature-edge count sane for class; render overlay | |
| **G2-pre** | **pre-flight (§7.3)** | runs before any build; all deterministic, seconds | M-08..11 |
| **G2a** | background `blockMesh` | domain extent per class rule; base Δ₀ = target refined size × 2^levels | |
| **G2b** | snappy castellate + snap (layers off) | leak detector: realized nCells vs. Δ³-volume estimate outside the configured band → M-05; region count as expected; refinement realized; post-snap checkMesh light; surface-deviation sample | M-05..07 |
| **G2c** | snappy layers | per-patch coverage %, thickness fraction, collapse renders | M-04 |
| **G2** | full `checkMesh` | maxNonOrtho, skewness, zero negative volumes (thresholds per class in `config/thresholds/`); y⁺ pre-estimate; patch names/types/areas vs. case expectations; renders | M-01..03 |

Warn-tier mesh metrics couple to numerics as advice the agent applies by editing dictionaries (high non-orthogonality → non-orthogonal correctors and limited gradients; high skewness → limited divergence schemes); accepting a warn-tier mesh without the coupled settings raises an advisory at G3. Checkers are unit-tested offline against CI golden meshes and threshold-configured per case class.

### 7.3 Mesh pre-flight — cheap checks before an expensive build

A snappyHexMesh build costs twenty minutes to several hours. Every check below is deterministic, runs in seconds against the STL and the dictionaries alone, and catches a failure mode that would otherwise be discovered only after the build. This layer exists because the "cheap to fix" property of free authoring does not hold for meshing unless it is engineered to hold.

| Check | Mechanism | Catches |
|---|---|---|
| `locationInMesh` validity | point-in-solid ray cast against the canonical STL; also distance to nearest surface vs. base cell size | M-08: point outside the fluid region, or inside a wall, or so close to a surface it lands in a cell that gets removed — the classic cause of an empty or inverted mesh after an hour |
| Region-name resolution | every name referenced in `refinementSurfaces`, `refinementRegions`, `layers`, and `geometry` cross-checked against the actual solid/patch names in the STL | M-09: silently unrefined surfaces, missing layers, snappy ignoring an entry it cannot resolve |
| Predicted cell count | surface area × 2^level / Δ₀² for surface refinement, plus volume-region estimates, plus layer cell multiplier | M-10: a refinement level that implies hundreds of millions of cells, or a mesh too coarse to resolve the QoI zone — before the build, and before the cost gate |
| Layer feasibility | total layer thickness (first-layer × expansion^n) vs. local base cell size after refinement | M-11: a layer specification geometrically incompatible with the background mesh, which produces collapse rather than an error |
| Refinement-vs-feature check | smallest geometric feature size vs. finest realized Δ | under-refinement at features that will fail to snap |
| Dry-run setup check | `snappyHexMesh -dry-run` where the build provides it | dictionary-level setup errors |

Predicted cell count also feeds the cost estimate at G4, and the prediction-versus-realized comparison lands in `mesh_build.md` — a large divergence is itself a signal (usually a leak, M-05).

---

## 8. Dictionary authoring: write freely, verify against the installation

### 8.1 The structural fact

There is no complete static grammar for OpenFOAM dictionaries. The grammar is distributed across the C++ classes that consume each dict (`lookup`, `getOrDefault`, `readIfPresent`, …), differs by fork and version, and is extended by every compiled library. A hand-maintained write-time schema is a lossy copy that decays with every release, and it constrains authoring in exchange for incomplete coverage. But the grammar does exist in one authoritative place — the installed binaries and source — and it is queryable. All checking therefore queries the installation.

### 8.2 The oracle stack (all post-write; nothing blocks authoring)

| # | Oracle | Mechanism | Cost | Catches | Misses |
|---|---|---|---|---|---|
| 0 | Parse | `foamDictionary -expand` per file | ms | syntax, includes, macros | everything semantic |
| 1 | Cross-file lint | `xfile_lint.py` | ms | patch↔field coverage matrix; function objects referencing real patches/fields; solver↔turbulence↔thermo compat; init sanity | value quality |
| 2 | Keyword lint (advisory) | registry harvested by grepping `$FOAM_SRC`/`$FOAM_APP` lookup-call patterns, regenerated per installed version | ms | the silent-typo class: keys no class ever reads → warning + nearest match | dynamically built keys (`name_ + "Coeffs"`), templates, macro expansion, per-patch subdicts, `#codeStream`. The harvest is incomplete by construction and its false-alarm rate equals its incompleteness rate — hence advisory, and hence subject to the precision budget in §8.5 |
| 3 | Percentile lint (advisory) | corpus value stats (§9) | ms | valid-but-unusual values against the tutorial distribution | tutorials are a demonstrative, biased sample; novel-but-correct choices fire falsely. Advisory, budgeted |
| 4 | Selection oracle | `foamToC` / `foamInfo` where the build provides them; invalid-token error probe as fallback (§8.3) | ms–sec | exact enum vocabulary for schemes, BC types, models, function objects, as compiled into *this* build with *this* `libs` list | non-selection keys; options in libraries not loaded by the querying executable |
| 5a | Pre-mesh dict probe (advisory) | clone `system/ constant/ 0/` onto a trivial blockMesh whose patches match the case's names and base types; run the solver one iteration | sec | required-key absence, invalid enums, dimension mismatches, BC construction, model instantiation — before meshing | self-skips (with a recorded reason) when the case references cellZones/faceZones created by snappy, cyclicAMI or mapped patches needing geometric pairing, nonuniform internal field lists sized to the real mesh, or anything downstream of `setFields`/`topoSet`. On those cases it is silent, not wrong |
| 5b | **Post-mesh dict probe (blocking)** | one solver iteration on the **real mesh**, after G2, before the smoke or production launch | sec–min | the same class of errors, with no false-block class, because the mesh is the real one | mesh-quality-dependent behaviour; silent optional keys |
| 6 | Effective-config echo | parse startup "Selecting …" lines + `printCoeffs on;` blocks from the run log into `run.md`; diff written vs. announced | free | drift between what was written and what the solver used | settings with no echo |
| 7 | Outcome verification | G6 (§11) | run cost | semantics | — |

The split at 5a/5b is deliberate. The pre-mesh probe is where the cost saving lives — catching a missing key before an hour of snappy — but it has a real false-block class on exotic cases, so it is advisory and skips rather than guesses. The post-mesh probe has no false-block class at all and is therefore blocking; it still runs before the expensive solve, which is where most of the money is.

### 8.3 Selection oracle: mechanism and bound

Where the build provides `foamToC` (ESI v2306 and later) or `foamInfo`, selection tables are enumerated directly: one call, no error required, no ordering constraint. This is the primary path.

The fallback is the error probe — place a deliberately invalid token in a selection slot and read the valid options out of the resulting fatal error. Three properties must be stated because they shape how it is used:

- It enumerates only what is compiled into the querying executable **with the `libs` entries present in that case**. A model in an unloaded library will not appear. The answer is build-specific, which is the point, but it is also `libs`-specific, which is a trap.
- It is fatal by construction, so each invocation yields exactly one enumeration.
- It is serially dependent: the error only reaches the slot in question if every object constructed earlier in the sequence is already valid. Resolving N uncertain slots is N probes with N−1 prerequisite fixes.

Consequently, on the fallback path the oracle is a loop, not a lookup, and it is governed as one (§13). On a novel case with several uncertain selections, prefer one `foamToC` enumeration pass over a probe chain.

### 8.4 Drafting protocol

Retrieval-first (§9): search corpus → copy nearest case(s) → edit → oracle stack → judge → iterate. Provenance recorded in `decisions.md` (`seeded-from: $FOAM_TUTORIALS/.../pitzDaily + study-014`). From-scratch authoring is always permitted; retrieval is economics, not law.

### 8.5 Advisory precision budget

Oracles 2, 3, and 5a are advisory, and each has a structurally non-zero false-alarm rate. Requiring a response to every advisory without bounding their number converts due diligence into reflexive waiving — the worst outcome, because it looks like diligence in the record.

The advisory channel is therefore governed by `config/advisory.yaml`:

- **Volume cap per gate.** Advisories are ranked by estimated precision and only the top N surface for mandatory response. The remainder are written to the digest as an unranked appendix, visible but not demanding a per-item note.
- **Precision target and tracking.** `advisory_budget.py` records the disposition of every surfaced advisory (applied / waived / waived-and-later-implicated). Per-rule precision is computed over the study history. A rule whose measured precision falls below the configured floor is automatically demoted to appendix-only until it is re-tuned.
- **Suppression list.** Per-case-class and per-corpus-family suppressions, with a required reason and an expiry, so that a known-good exotic pattern stops generating the same advisory in every study of that class.
- **Acceptance measures both directions.** The oracle-stack acceptance test (§21, M3) measures recall on injected errors *and* precision on valid cases, and the precision figure is a release criterion, not an aspiration.

### 8.6 Attribution and the duplicate-launch guard

`change` in the attempt graph is computed, not enforced: `foamDictionary -expand` both versions → canonical form → structured diff → attempt node. Whole-file rewrites, surgical edits, and script-generated dicts all attribute identically.

The configuration **fingerprint** is a hash over the geometry hash, the canonical expansion of all dictionaries, and the mesh-generation code hash, **with run-control keys excluded** — `endTime`, `startTime`, `writeControl`, `writeInterval`, `purgeWrite`, `numberOfSubdomains`, the decomposition method, and the rest of the list in `config/fingerprint_exclusions.yaml`. Excluding them is not a write-time schema: it does not constrain what may be authored, and being wrong about an entry costs at most one redundant run rather than a lost capability.

What this guard is, stated honestly: **a duplicate-launch check, not a thrash detector.** It reliably refuses relaunching a configuration byte-identical (modulo run control) to one already attempted, with a pointer to the prior attempt node. It does not catch A → B → A′ where A′ differs from A in some physics-irrelevant but hashed way, and no fingerprint short of a semantic model of every key would. Thrash prevention rests primarily on the progress requirement in §13 and on escalation, with the fingerprint as a cheap backstop. `launch_run(..., force=true, reason=...)` overrides the guard for legitimate re-runs after node failure or a wall-time change; the override is an event and appears in the report.

---

## 9. Corpus & retrieval

Four tiers, one interface (`corpus/search.py`), all provenance-tracked:

| Tier | Source | Trust properties |
|---|---|---|
| **Benchmark** | published experimental and DNS cases with reference values and stated experimental uncertainty: backward-facing step, periodic hills, Ahmed body, flat-plate boundary layers, standard elbow/fitting loss coefficients (Idelchik, Crane), room-ventilation benchmark datasets | the only tier anchored outside the system's own judgment; carries reference values, not just configurations |
| **Curated** | reviewed seed cases + generators, CI-validated on canonical geometries | high; version-pinned |
| **Vendor** | `$FOAM_TUTORIALS`, indexed at install | version-matched by construction (ships with the binaries → zero keyword drift); every case known-running, but tutorials demonstrate features, not accuracy — never treat a tutorial value as a validated value |
| **Earned** | completed study bundles | regime-tagged, verdict-weighted, includes failure knowledge via attempt graphs |

**Index row:** `{path, tier, solver: {executable, module}, turbulence, regime: {class, Re, Ma, shedding_risk}, mesh_type, bc_map, of_version, of_fork, verdict, reference_value?}`. The `solver` field is a pair rather than a string so that Foundation-fork cases (`foamRun -solver incompressibleFluid`) and ESI cases (`simpleFoam`) index and retrieve under one schema.

**Retrieval calls:** by regime/class ("internal incompressible steady, Re ~10⁴") → seed cases; by keyword ("what does `div(phi,U)` take in RAS incompressible tutorials") → value distribution; by failure signature ("M-04, trailing-edge collapse") → past attempt-graph nodes whose branch resolved it, ranked by outcome.

**Why the benchmark tier is load-bearing.** Without it the corpus is a closed loop: curated is reviewed by us, vendor demonstrates features, earned is weighted by G6 verdicts produced by tests the system itself wrote. A systematically wrong convention that passes G6 enters the earned tier, gets retrieved into the next study, becomes the percentile norm, and then causes the *correct* value to look statistically unusual and raise an advisory. The benchmark tier breaks the loop by anchoring to values the system did not produce.

**Benchmark regression** runs in CI on every OpenFOAM version bump, every executor change that touches meshing, numerics, or the verify instruments, and every corpus re-index. A benchmark case whose result moves outside its recorded band blocks the change. This is also what licenses the word "verified" in a report: a study's confidence statement may cite benchmark agreement for its case class where one exists, and must say so explicitly where none does.

**Rules:** provenance in `decisions.md`; cross-version retrieval flags drift risk and re-runs the keyword lint; earned-tier configs are weighted by their G6 verdicts, and diverged configs remain retrievable but labelled.

**Every retrieval is logged, including the hits not taken.** `search.py` appends a `retrieval` record to `attempts.jsonl` with the query, the ranked hits and their tiers, and which hit seeded the decision. Adopted hits become `[retrieved]` pointers in the spec (§17). The rejected ones matter for the same reason the adopted ones do: when a benchmark hit and an earned hit disagree and the earned one was taken, that is the signature of the closed loop tightening, and it is invisible if only the winner is recorded.

**Accumulation:** each study drafts from precedent rather than priors; every postmortem emits lint-rule candidates and playbook candidates; raw drill-downs shrink as digest classifiers grow. Institutional memory is a filesystem artifact, not a fine-tune.

---

## 10. Execution

### 10.1 Launcher

`launch_run(run, mode, force?, reason?)`: fingerprint duplicate check → budget decrement → `decomposePar` per plan → `mpirun` detached (nohup + pidfile; SLURM adapter later) → returns run-id immediately. No blocking-wait tool exists, so the deliberator cannot sit in a polling loop.

Function objects always installed: `solverInfo` residuals, the spec QoI monitors re-injected from `spec_frozen/qoi_instruments/` at launch time, `fieldMinMax`, `CourantNo` for transient, and `printCoeffs on;` for the echo oracle. Re-injection happens at launch rather than at G3 because the case directory remains writable in between; the launcher hashes the case's complete function-object set against the frozen definitions immediately before starting the solver and aborts on mismatch (§11.1).

Write policy: sparse intervals plus purge to latestTime except a keep-list. The keep-list is determined by the **recomputation mode declared in the spec** (§11.1), not negotiated at launch: `field-recompute` requires one field write, `subsample-recompute` requires the declared subsample and its disk cost is budgeted at G0, `parallel-instrument` and `none` require none. The launcher refuses a write policy inconsistent with the declared mode, which is a check rather than an open-ended retention obligation — the earlier formulation, where retention had to satisfy whatever recomputation might later want, is unbounded on transient cases.

Runs execute under the same confinement as the sandbox. `#codeStream` and `codedFixedValue` compile and execute arbitrary C++ at solver runtime; this is permitted, because forbidding it would remove real capability and the deliberator already holds a shell at the same privilege. But it has two consequences the design must respect: a detached production run is an arbitrary-code-execution path and must not be launched into a less-confined context than the sandbox itself, and the dict probes (5a/5b) are not guaranteed side-effect-free, so they run in a throwaway copy of the case directory.

### 10.2 Watchdog (dumb by design)

Polls on a fixed interval with incremental log parsing. Rules are configuration, not judgment; all thresholds live in `config/watchdog.yaml`.

| Rule | Default action |
|---|---|
| `FOAM FATAL` | kill + event |
| residual ratio vs. N iterations ago above threshold | kill + event |
| Courant number above cap for consecutive steps | kill + event |
| timestep below floor (transient collapse) | kill + event |
| bounding messages above rate threshold | warn event |
| disk below threshold | pause + event |
| walltime beyond budget share | graceful `writeNow` stop + event |

Judgment calls (continue vs. remediate vs. physics-revisit) wake the deliberator. **Soft-smoke mode**: when G4 is skipped, kill thresholds tighten by the configured factor for an initial window of the production run.

### 10.3 Smoke policy (G4 conditional)

```
smoke_required =
     (case_class, solver, model) not in validated_set     # novelty
  OR est_production_cost > threshold                      # expense
  OR first_case_of_study                                  # family calibration
  OR physics_amended_since_last_run
  OR shedding_risk != low                                 # §11.4 — smoke carries the scheme probe
```

Otherwise skip to soft-smoke. Free-smoke cases: in a planned mesh ladder the coarse rung is the smoke; in a sweep the first case is the family's smoke.

### 10.4 Events

`events.jsonl` is written with `O_APPEND` by the watchdog daemon, which is a separate process that may be killed mid-write; append-only single-line writes are what make concurrent producers and a partially-written tail safe to handle. This is a concurrency requirement, not a general convention — other bundle files are ordinary files and may be rewritten.

The supervisor spawns deliberator sessions on terminal and anomaly events.

```json
{"id":"e-041","run":"run-02-medium","type":"anomaly.residual_divergence",
 "t_sim":0.42,"wall_min":38,"evidence":"digests/run.md#e-041","severity":"warn",
 "action_taken":"none — within kill-threshold grace"}
```

---

## 11. Verification (G6)

G6 is a test run: pytest over the frozen suite plus the system instruments, against standard result interfaces (`monitors`, `polars`, `planes`, `fields_summary`).

### 11.1 What is frozen: the tests, the instruments, and the definition of the quantity

Freezing the tests alone does not work. `test_pressure_drop` calls `monitors.qoi("dp")`, and `dp` is produced by a `surfaceFieldValue` function object. If that function object stays writable, the deliberator can change the patch it measures on, the operation (`areaAverage` → `areaIntegrate` → `max`), the sampling interval that determines the averaging window, or which fields get written at all — and the test passes without a character of the test changing. The same applies to `planes.occupied_zone`, whose sampling planes are case configuration. A function-object diff looks exactly like ordinary case authoring in the attempt graph, so this is *harder* to see than editing a test would be.

Two mechanisms close it:

1. **The QoI instruments are part of the spec.** The function-object subdicts that produce spec QoIs live in `spec_frozen/qoi_instruments/`, are approved at G0 alongside the tests, hash into `freeze.manifest.json`, and are injected into every case by the launcher. The case's own `system/functionObjects` may add monitors, but a definition colliding with a frozen QoI name is refused at G3. Changing one is a spec amendment.
2. **Tier-2 recomputes independently.** `qoi_recompute.py` derives each primary QoI from raw field and boundary data using system-owned code that reads nothing the deliberator configured — patch identification comes from the spec's patch contract, not the case's function objects. A disagreement between the frozen instrument and the independent recomputation beyond tolerance is a G6 failure in its own right (V-06).
3. **The launcher re-verifies, because G3 is too early.** The case directory stays writable after G3, so a collision check performed only at G3 can be defeated by editing `system/functionObjects` between the gate and the launch. `launch_run` therefore re-injects the QoI instruments from `spec_frozen/qoi_instruments/` and hashes the case's complete function-object set immediately before starting the solver, recording the hash in the run record. A mismatch against the frozen definitions aborts the launch.

**Recomputation modes, and what independence actually means per mode.** Independent recomputation from field data is straightforward for steady cases — one field write, one recomputation, agreement to rounding. It does not generalize to transient QoIs, where a time-averaged force computed from field history would require retaining every write time at 0.1–5 GB each, on exactly the high-shedding-risk cases the architecture most wants to police. The launcher's retention rule and the purge policy would otherwise deadlock against each other, so the mode is declared in the spec and constrains both:

| Mode | Mechanism | Independence achieved | Retention implied |
|---|---|---|---|
| `field-recompute` (steady default) | recomputation from the final field write | code-independent and run-independent | one field write |
| `parallel-instrument` (transient default) | a second, system-owned function object written from `spec_frozen/`, computing the same QoI by a different route (e.g. surface integration vs. the monitor's patch operation), running alongside the frozen instrument | code-independent, **not** run-independent — a run-time fault affects both | none beyond monitors |
| `subsample-recompute` (transient, when retention allows) | recomputation over a retained subsample of write times, declared at spec time | code- and run-independent over the subsample | subsample × field size, budgeted explicitly at G0 |
| `none` | no Tier-2 recomputation available | none | — |

The mode is named in `spec.md`, the launcher enforces the retention it requires, and `report.md` states which mode was used. `none` is permitted and must be visible: a report whose Tier-2 column says `none` is making a weaker claim, and should say so rather than imply a check that did not run.

**Comparison tolerance is computed, not configured.** For `field-recompute` the two paths should agree to numerical rounding, and the tolerance is tight and fixed. For the transient modes they will *always* disagree by the difference between a monitor-window average and a field-sample average, and a fixed tolerance cannot distinguish that expected disagreement from a wrong patch. The tolerance is therefore derived from the sample itself — the standard error of the subsample or parallel-instrument mean, times a configured coverage factor — so that sampling noise is inside the band by construction and a definitional or plumbing error is outside it. The computed tolerance and the observed difference both appear in `results.json`.

**The dimensional contract.** A tolerance band cannot catch a units error whose factor is smaller than the band. In incompressible OpenFOAM, `p` is kinematic pressure with dimensions m²/s²; for air, comparing it to a dynamic-pressure expectation is an error of ρ = 1.204, invisible inside any band wide enough to be useful. For water the same error is a factor of 998 and any band catches it. Without a dimensional check the system is least protected against its named catastrophic error class on precisely the fluid in its motivating examples.

Therefore:

- `result_interfaces.py` and `qoi_recompute.py` return **dimensioned quantities**. OpenFOAM states each field's dimensions in the field header; the interface reads them rather than assuming, and carries them through every operation.
- Every expectation in `calcs/` **declares its units** alongside its value and uncertainty class.
- The comparison in Tier 3 **fails on dimension mismatch before comparing magnitudes** (V-09). This also catches the sibling errors that share the shape: force against force coefficient, mass flow against volumetric flow, gauge against absolute.
- **Conversions are named and sourced, not written inline.** The obvious response to a dimension failure is to multiply by ρ until it passes, which is the reflex the freeze exists to prevent. A conversion is therefore a declared entry in the frozen instrument definition, naming the conversion and the constant's source (`rho` from `constant/transportProperties`, not a literal), and it is reviewed at G0 with everything else. A conversion added after freeze is a spec amendment.

**The definitional contract.** Dimensional agreement is necessary and not sufficient. Two instruments can have identical dimensions, agree with each other to a fraction of a percent, and measure a different physical quantity than the expectation they are graded against. The elbow example is the canonical case: an Idelchik K-factor is defined on **total** pressure loss, while the frozen instrument differences area-averaged **static** pressure across the elbow. Those coincide only if the velocity profiles at the two stations are identical, and after a sharp elbow the outlet profile is still recovering at 20D, so the kinetic-energy correction differs between stations. V-06 is structurally blind to this — independent recomputation reads the same field with the same convention and agrees beautifully, because the two instruments are independent in plumbing and identical in definition.

The spec therefore carries, for each QoI, a **quantity definition**: what physical quantity is being measured, on what surfaces, with what convention (static vs. total, gauge vs. absolute, instantaneous vs. averaged over what window), and the same statement for the expectation it will be compared against. The adversarial review of that pairing is a required pre-G0 check rather than an optional one (§15), and a mismatch is V-10. Where the mismatch is real but accepted — measuring static difference because total-pressure sampling is impractical — it is recorded in the spec with its estimated magnitude, and that magnitude widens the Tier-3 band rather than being ignored.

### 11.2 The freeze is a hermetic runner, not a mount

A read-only mount on a test directory does not freeze a test suite. pytest will still collect a `conftest.py` from a writable parent directory, honour a writable `pytest.ini` or `pyproject.toml`, and import a shadowed module from a writable `PYTHONPATH` entry. A monkeypatched `plateaued()` passes every test with the frozen files byte-identical.

`hermetic_pytest.py` therefore runs G6 as follows:

- fresh process, `rootdir` pinned to `spec_frozen/`, `-p no:cacheprovider`, no `conftest` collection above the pinned root;
- `PYTHONPATH` constructed explicitly from the frozen directory and the system executor package only; no writable directory appears on `sys.path`, and the current working directory is not on it;
- plugin autoloading disabled; the plugin set is enumerated in the frozen config;
- before execution, the full import and configuration closure is hashed and compared against `freeze.manifest.json` — every file in `spec_frozen/`, every system module imported by the suite, and the pytest/plugin versions. Any mismatch aborts G6 rather than failing it, and raises an escalation;
- the case directory is mounted read-only into the runner, so a test cannot mutate the thing it is grading.

Editing a frozen artifact requires `amend_spec`, which unfreezes, re-approves, and re-manifests.

### 11.3 Which run is graded

Run selection is a governed choice, not a free one. The spec names the grading run by role — the finest converged rung of the ladder, or the named production run — and `run_gate("G6")` refuses to grade a run that does not match the declared role, or that post-dates a spec amendment it was not re-checked against. If a later run is to be graded instead, that substitution is recorded with a reason in `decisions.md` and appears in the report. Without this, a self-grading system has an ungoverned degree of freedom: fail on run-03, pass on run-04, report run-04.

### 11.4 Regime classification and the silently-wrong steady solve

The most dangerous failure is a steady solver applied to genuinely unsteady flow. The tempting detector — residuals plateau high while the QoI oscillates — catches only the *visible* version, where the solver is struggling in public. The dangerous version is the opposite: first-order upwinding, heavy under-relaxation, and steady SIMPLE together damp the physical instability; residuals fall several orders, the QoI sits flat, and the result is a well-converged wrong number with neither oscillation nor elevated residuals. A detector keyed on oscillation is blind to exactly this case.

The architecture handles it upstream and at three points:

- **G0 — classification with evidence.** `regime/classify.py` produces a `shedding_risk` label from geometry and flow descriptors: bluffness (frontal area vs. streamwise extent, presence of sharp separation edges), expected adverse pressure gradient, Reynolds number against known regime boundaries for the class, aspect ratio, and any recognized benchmark family. The label, its inputs, and the resulting solver decision are recorded in `spec.md` and are part of what the user approves. A steady solver on a case labelled medium or high risk requires an explicit justification in the spec.
- **G2 — re-classification against realized geometry.** The G0 label is produced from a text request before any geometry exists, which is the weakest link. After the mesh is built, the classifier re-runs against the actual surface — measured bluffness, real feature angles, realized blockage. An upgraded risk label at G2 is a spec-amendment trigger, not a note.
- **G4/G6 — active probes rather than passive detection.** For any case not labelled low risk: a URANS spot-check (§11.4.1), a scheme-sensitivity probe (§11.4.2), and a separation-extent check comparing the steady field's separation region against the classifier's expectation. Passive signals (R-05 residual plateau, R-06 QoI oscillation) remain as detectors for the visible case, but they are not the primary defence.

#### 11.4.1 URANS spot-check: measure growth, not amplitude

The naive version of this probe is a strong false-negative generator on exactly the case it exists for. A converged steady solution on a shedding case is an unstable equilibrium; started from it, the physical instability must grow out of numerical noise, and for a bluff body that typically takes tens of shedding cycles before the QoI amplitude is visible. A short window started from the steady field shows a flat QoI, the probe reports no shedding, and the confident wrong number survives with a certificate attached.

The probe is therefore specified as:

- **Perturb deliberately.** The initial field is the steady solution plus a small, reproducible, seeded perturbation — an asymmetric velocity perturbation in the expected separation region, at an amplitude recorded in the run record. Growth from a known seed is measurable in far fewer cycles than growth from roundoff.
- **Fit a growth rate; do not threshold amplitude.** The verdict is the sign and magnitude of an exponential growth rate fitted to the QoI envelope, not whether the amplitude crossed a level. A perturbation that decays is positive evidence the steady solution is stable; a perturbation that grows is a positive detection while the amplitude is still small. Both are informative; an amplitude threshold turns the first case and the too-short-window case into the same "no shedding" answer.
- **Scale the window to a physical period.** Window length is set from a Strouhal estimate for the case class and characteristic length (a small number of estimated shedding periods), not a fixed step count. Where no Strouhal estimate exists, the window extends until the growth-rate fit reaches a stated confidence or the probe budget is exhausted, and an exhausted budget without a confident fit is reported as **inconclusive** — which routes to escalation, not to pass.

`inconclusive` is a distinct verdict from `stable` throughout, and a report may not claim a steady result on a medium- or high-risk case whose spot-check was inconclusive without an explicit waiver.

#### 11.4.2 Scheme sensitivity: vary something that is actually varied

Two failure modes in the obvious implementation. First, if the baseline already uses second-order divergence schemes — which it should — then "rerun on second-order schemes" varies nothing and the probe silently becomes a no-op that reports zero sensitivity, the most reassuring possible output. The probe therefore varies along whichever axis has headroom relative to the baseline, and **records which axis it varied**: divergence scheme (first- ↔ second-order, limited ↔ unlimited), gradient limiting, under-relaxation, and where available a coarser/finer time or iteration budget. A probe that finds no axis to vary reports `not-applicable`, never zero sensitivity.

Second, relaxing under-relaxation on a marginally stable steady solve frequently diverges. Divergence here is not a numerics failure to be remediated — it is among the strongest available evidence that the steady formulation is holding together an unsteady flow by numerical damping. Divergence of a scheme-sensitivity probe is classified as **V-07 evidence**, not R-01, and routes to physics revisit rather than to the numerics playbook. The taxonomy classifier is given the probe's run id explicitly so it cannot mistake the two.

`config/thresholds/<class>.yaml` carries the risk thresholds and probe tolerances per case class.

### 11.5 The three tiers

**Tier 1 — study-specific, LLM-authored, user-approved at G0, frozen after** (`spec_frozen/tests/`):

```python
def test_pressure_drop(monitors):
    dp = monitors.qoi("dp")
    assert plateaued(dp, slope_pct_per_1k=1.0)
    assert within(dp.last_window_mean, expected=29.2, band=EXPECT_BAND)  # band from calcs/ class

def test_static_stability(polars):                     # drone study
    assert slope(polars.Cm, polars.alpha, window=(-4, 8)) < -MARGIN

def test_ashrae_draft_limit(planes):                   # compliance study
    occ = planes.occupied_zone(heights=[0.1, 0.6, 1.1, 1.7])
    assert occ.speed.p99 < 0.8                          # clause-cited in docstring
```

**Tier 2 — invariant instruments, system-owned**, identical across studies so results are comparable: QoI plateau machinery, independent QoI recomputation (§11.1), conservation (mass in vs. out within tolerance), y⁺ coverage (§11.7), GCI over the mesh ladder (§11.6), and benchmark agreement where the case class has a benchmark entry.

**Tier 3 — the outside check, with tiered tolerance.** The hand-calc or correlation expectation is authored at spec time, before any test code exists, and lives outside the suite the agent wrote. A flat factor-of-ten band is not a check: it catches unit errors and gross BC inversions and nothing else, and it lets a wrong expectation and a wrong simulation agree comfortably. Every expectation therefore declares an **uncertainty class**, which sets its band:

| Class | Example | Band |
|---|---|---|
| Benchmark / published reference | Idelchik K for a standard elbow; DNS reference for periodic hills | reference uncertainty, typically ±10–25 % |
| Correlation-backed | Darcy–Weisbach + tabulated minor losses; fan curve | ±25 % |
| Composed correlation | several correlation terms summed with interaction neglected | ±50 % |
| Order-of-magnitude estimate | scaling argument with an assumed coefficient | ×/÷ 3 |
| Dimensional analysis only | no coefficient available | ×/÷ 10 |

Exceeding the band blocks the report until explained. A user-visible consequence: a study whose only expectation is dimensional-analysis class cannot claim a tight result, and the report says so.

*Worked example, corrected.* For the elbow study: ½ρU² = 15.05 Pa. Minor loss at K ≈ 1.1 gives 16.6 Pa — but the spec's own geometry has 30 diameters of pipe, and Blasius at Re = 1.65 × 10⁴ gives f ≈ 0.028, so friction contributes f·(L/D)·½ρU² ≈ 12.6 Pa. The expectation is ≈ 29 Pa, not 16.6. This is exactly the error class the tiered band exists to catch: the earlier figure omitted a term the geometry spec creates, and a ten-times band would never have noticed. The expectation calculation in `calcs/` must therefore account for every term the spec's geometry implies, and its uncertainty class must reflect that it is a composed correlation.

### 11.6 Mesh independence and non-monotone convergence

GCI with a factor of safety of 1.25 over three grids is the default, but Richardson extrapolation presumes monotone convergence, and real triplets frequently do not deliver it. `gci.py` handles the cases explicitly rather than reporting a number that presumes them away:

- **Monotone, observed order near formal order** → standard GCI, reported normally.
- **Monotone, observed order implausible** (negative, complex, or far above formal order) → the observed order is clamped to the formal order of the discretization, the GCI is reported as an estimate with the clamp declared, and the verdict is flagged.
- **Oscillatory convergence** (the sign of the change reverses across the triplet) → Richardson is not applicable; the instrument reports a bounded estimate — the QoI spread across the ladder as an interval — labels the convergence as oscillatory, and does not emit a GCI percentage.
- **Insufficient ladder** → adding a rung is an available action, and the ladder is bounded by the budget and by the requirement that each rung reduce the estimate spread, not by a fixed rung count. A ladder that stops improving escalates.

**Cross-study transfer.** Mesh independence may be established as its own study, but a sweep does not silently inherit its verdict. A drone polar is far more mesh-sensitive at α = 14° than at α = 2°. A GCI verdict carries a validity condition — the regime envelope over which it was established (angle range, Reynolds range, absence of large-scale separation) — and retrieval refuses to apply it outside that envelope; a sweep crossing the boundary must re-establish independence at the worst-case point, and the report states which cases inherit and which were verified directly.

### 11.7 y⁺

A hard 30–100 band tested over a histogram fails on every real geometry. Two reasons: modern `kOmegaSST` in OpenFOAM uses continuous/blended wall treatments that are deliberately y⁺-insensitive across the buffer layer, so the band is legacy advice that rejects good meshes; and any real geometry has stagnation points and separation lines where y⁺ → 0 regardless of mesh quality. A check that fires on every case and is waived on every case corrodes the meaning of a waiver.

The instrument is therefore a coverage criterion: the *fraction of wetted area* within the wall-treatment's valid range, computed with stagnation and separation zones excluded (identified from the skin-friction field, not from geometry), and evaluated against a per-class coverage floor in `config/thresholds/`. Where the wall treatment is blended, the check is reported as informational for the QoI-relevant surfaces only. V-03 fires when coverage on QoI-relevant surfaces falls below the floor — a condition that should be rare and therefore means something.

### 11.8 Failure routes

Extend run (§13) · refine mesh (ladder) · **physics revisit** — consult `revisit-if` clauses; a physics change is a spec amendment and re-triggers approval. Waivers are allowed but explicit: justification in `verdict.md`, surfaced in the report, listed in `results.json.waivers`, and counted — a study accumulating waivers past the configured count escalates rather than delivering.

---

## 12. Attempt graph

A DAG in `attempts.jsonl` with a rendered tree in `attempts.md`. Nodes are attempts; edges are derivations with computed diffs (§8.6). Nodes are written as they occur and are not rewritten in the normal path; a correction is a new node referencing the one it corrects, because the record's value is in showing what was tried and in what order.

```json
{"id":"a-007","parent":"a-005","gate":"G2c",
 "fingerprint":"sha256:…",
 "change":{"computed_diff":"snappy.addLayersControls: nSurfaceLayers 5→3, expansionRatio 1.3→1.2",
           "advice_used":["M-04a","study-014#a-011"]},
 "hypothesis":"layer collapse at trailing edge from aggressive expansion",
 "result":"fail","evidence":"runs/mesh-07/digests/mesh.md",
 "delta":{"layerCoverage":"0.42→0.61"},"cost_min":14,"cost_tokens":38200}
```

What it buys: duplicate-launch refusal with a pointer to the prior node (§8.6); causal attribution, where the `delta` field carries gradient information — a failure that moved coverage 0.42 → 0.61 says "right direction, not far enough", and this is the primary input to the progress requirement in §13; cross-session memory, since the orient pack ships the frontier plus one-line summaries of dead branches; and cross-study memory, since earned-tier retrieval searches these nodes by failure signature.

Distinct from `decisions.md`: decisions are deliberate choices with rationale and revisit-if; attempts are experiment history. Cross-referenced by ID.

---

## 13. Looping semantics

Every loop terminates on two conditions that are properties of the problem rather than arbitrary counts:

1. **Budget.** All iterations decrement shared budgets (§13.1). A depleted budget is a hard stop and an escalation, regardless of loop state.
2. **Progress.** Each loop declares a progress metric, and each iteration must improve it. An iteration that does not improve its metric is not automatically fatal — one flat step can be informative — but a loop whose metric has not improved over its recent history terminates and escalates. "Recent history" and the improvement threshold are per-loop settings in `config/loops.yaml`; they are tunable operational defaults, not architectural facts.

| # | Loop | Driver | Progress metric | Escape |
|---|---|---|---|---|
| 1 | Gate retry | failed check | the `delta` fields in successive attempt nodes must move the failing metric toward its threshold | escalate with the structured report |
| 2 | Mesh ladder | GCI plan | estimate spread across the ladder must narrow (§11.6) | accept with a bounded-estimate waiver, or escalate |
| 3 | Run extension | QoI not plateaued | the QoI slope must decrease across successive extensions — a flat slope means physics, not patience | physics revisit |
| 4 | Physics revisit | revisit-if triggered at G6 | — | user; each revisit requires re-approval, so it cannot spin silently |
| 5 | Selection-oracle probe chain | uncertain selection slots on the fallback path (§8.3) | number of unresolved slots must fall each probe | switch to a `foamToC` enumeration pass, or escalate |
| 6 | Study sweep | plan | cases complete | — (not error-driven) |
| 7 | Watchdog poll | timer | — | run lifetime |
| 8 | Checkpoint review | timer on long runs | QoI trend vs. remaining budget | kill / extend |

Anti-thrash rests on three things: the progress requirement above, monotone escalation (a loop that escapes never re-enters at the same level without a spec amendment or user instruction), and the duplicate-launch fingerprint as a cheap backstop (§8.6). Small diffs per retry are a convention because they make `delta` interpretable, not because a rule caps diff size.

### 13.1 Budgets

Compute is not the only cost, and on a render-heavy multimodal mesh loop it is frequently not the dominant one. Three budgets are tracked in `state/budget.json` and decremented by the state service:

| Budget | Decremented by | Enforced at |
|---|---|---|
| **CPU·h** | `launch_run`, mesh builds, probes | launcher refuses a launch that would exceed the cap |
| **Tokens / session count** | every deliberator session and subagent spawn, recorded per attempt node (`cost_tokens`) | supervisor refuses to spawn past the cap; escalates to the user instead |
| **Wall-clock** | elapsed study time against any declared deadline | checkpoint review |

Each budget has a soft threshold that raises a warning event and a hard cap that stops and escalates. The report includes actual spend against all three, since a study that produced a good answer for ten times its token estimate is a calibration input.

---

## 14. Deliberator: sandbox + boundary tools

### 14.1 Inside the sandbox

The deliberator gets bash and Python rooted in the study bundle, with per-command output truncation configured in `sandbox/`. Structured reads go through `read_artifact` (§6.1), which is the only path that carries provenance, fidelity and staleness; raw bytes stay fully reachable via `grep`/`head`/`tail`/windowed reads for anything the view does not answer. Dictionaries are edited with any tool the model likes; digest, render, oracle, and checker scripts are ordinary executables it invokes; `postProcess`, `foamDictionary`, and pvpython are on PATH.

**Protected paths** (enforced by filesystem permissions, with the deliberator running as a uid without write access to them):

- `spec_frozen/` after G0 approval — and, independently, the G6 runner verifies the closure hash, so a permission failure alone does not compromise grading (§11.2);
- corpus indices and system executors;
- `state/gates.json`, `state/budget.json`, `state/lock.json`;
- `context/transcript.jsonl`, `context/approvals.jsonl`, `context/amendments.jsonl` — **append-only, not read-only**: the deliberator appends via `append_context` and can never rewrite history. A session able to edit the transcript could retroactively manufacture `[user]` provenance for its own assumption, which would make the G0 audit trail worthless.

Everything else in the bundle is writable, including `state/workspace.json`.

### 14.2 State is split, because a JSON field cannot be mounted read-only

Gate status and budget integrity are load-bearing, and the enforcement mechanism has to be one that exists. A single `state.json` with some fields writable and some not is not implementable with filesystem permissions.

State is therefore split into separate files with separate owners. `gates.json`, `budget.json`, and `lock.json` are owned by the **state service** — a small local process that is the only writer. It exposes exactly the transitions the design allows: `run_gate` submits a gate transition with evidence and the service validates and records it; the launcher submits a budget decrement; the supervisor acquires and releases the lock. The deliberator can read all of them and write none of them. `workspace.json` is the deliberator's own scratch and is freely writable.

**Evidence is typed, produced, and bound.** Validating that an evidence path exists is a typo check, not an evidence check: under that rule `run_gate("G2", evidence="report.md")` passes. Each gate declares the **type** of evidence it requires, and the service accepts only an evidence record of that type, emitted by the corresponding checker, carrying a hash of the artifact it graded:

```json
{"gate":"G2","type":"mesh_checker_verdict","produced_by":"executors/mesh/check_final_mesh.py@v3",
 "artifact":"runs/mesh-07/digests/mesh.md","artifact_sha256":"…",
 "graded_config_sha256":"…","verdict":"pass","at":"…"}
```

The service verifies that the type matches the gate, that the producer is the expected checker, that the artifact hash matches the artifact on disk, and that `graded_config_sha256` matches the study's current configuration fingerprint. A gate graded against a configuration that has since changed is stale, not passed — the same binding the digest reader already applies (§6.3), applied here too.

**Transition legality is a declared graph.** Predecessor requirements are part of the schema, not left implicit: G6 cannot pass while G2 or G3 is failed or stale, G5 cannot start without a G4 verdict or an approved soft-smoke waiver, G7 cannot pass without G6. The full allowed-transition graph, including which gates a spec amendment marks stale, is in Appendix A. The service rejects any transition not in the graph.

The service also enforces the arithmetic — a budget cannot be decremented below zero, a lock cannot be acquired while a live lease is held.

**Failure semantics.** The state service is a single point of failure for gates, budget, and lock, so its unavailable behaviour is specified rather than emergent. It **fails closed**: no transition, no launch, no budget spend, no lock acquisition. A deliberator that cannot reach it may continue to read cached state and to author files in the writable bundle, but any boundary tool requiring a transition returns a service-unavailable error, and the session's correct response is to write its workspace and escalate rather than to proceed on stale reads. The supervisor does not spawn new sessions while the service is down.

Durability is write-ahead plus atomic rename per file. Recovery is a rebuild: `gates.json` and `budget.json` are reconstructible from the evidence records, `events.jsonl`, and the attempt graph, all of which are separately durable, so a corrupted state file is a rebuild-and-verify operation rather than a lost study. The rebuild is a maintenance command, not something a deliberator session can invoke.

### 14.3 Boundary-crossing tools

```
orient(study) → reconciled pack          ask_user(questions[], blocking)
run_gate(gate) → verdict                 # the only path to a gate transition; attaches evidence
launch_run(run, mode, force?, reason?) → run_id
extend_run(run, endTime, reason)         kill_run(run, reason)
amend_spec(diff, reason)                 # logged event; physics-, test-, or instrument-level
                                         #   diffs re-trigger G0 and re-manifest the freeze
spawn_subagent(task_dir) → validated outputs
escalate(report)                         finalize(report)
```

### 14.4 Enforcement

| Rule | Mechanism |
|---|---|
| No unbounded raw reads | shell output caps |
| No launching and polling in its own loop | `launch_run` returns instantly; no wait primitive exists |
| No gate flips by editing state | `gates.json` writable only by the state service, which accepts only typed evidence records from the declared checker, hash-bound to the artifact and to the configuration graded, and only along the declared transition graph |
| No test or instrument edits after approval | filesystem permissions on `spec_frozen/`, **plus** closure-hash verification in the hermetic G6 runner, **plus** launch-time re-injection and hash check of the function-object set, plus `amend_spec` as the only legitimate path |
| No comparing a quantity to an expectation in different units | dimensioned result interfaces; Tier-3 fails on dimension mismatch before magnitude (V-09); conversions declared and sourced inside the frozen instrument |
| No proceeding when state integrity is unavailable | state service fails closed; boundary tools error; supervisor stops spawning |
| No grading a run of convenience | `run_gate("G6")` checks the run against the role declared in the spec (§11.3) |
| No paying twice for an identical configuration | fingerprint refusal at `launch_run`, overridable only with a logged reason |
| No silent spec drift | `amend_spec` is a logged event and re-manifests the freeze |
| No reasoning from stale views | the reader refuses on source-identity mismatch, on every read of every artifact type (§6.1, §6.3) |
| No manufactured user provenance | `context/` is append-only; a `[user]` tag whose transcript span fails to resolve blocks G0 (§6.5) |
| No orphaned study | lock lease with TTL and recovery (§16.1) |
| No acting on a stale handoff | orient reconciliation (§16.2) |

---

## 15. Subagents

Task-folder contract: `task/instructions.md` + `task/inputs/` in, `task/outputs/` out, schema-validated on return; one bounce on schema failure, then escalate. Subagent spawns decrement the token budget.

| Subagent | Input | Output | Validation |
|---|---|---|---|
| Standards → tests | standards PDFs | draft `test_compliance.py` + `predicates.md` view: {clause citation, executable predicate, measurement protocol, derived probes} | every test cites a clause in its docstring; suite dry-runs against fixture data; goes through G0 approval with the spec |
| Log forensics | chunked log windows | structured findings (event-class candidates, line refs) | refs must exist |
| **Quantity-definition review (required pre-G0)** | for each QoI: the frozen-candidate instrument, the test that consumes it, and the expectation's own definition and units from `calcs/` | a verdict per QoI — `match`, `mismatch`, or `accepted-mismatch with estimated magnitude` — covering physical quantity, surfaces, convention (static/total, gauge/absolute, instantaneous/averaged), and dimensions | every QoI must carry a verdict; a `mismatch` blocks G0; an `accepted-mismatch` must state a magnitude, which widens the Tier-3 band (§11.1). This is the only check that catches a definitional error, since V-06 cannot |
| Adversarial spec review (optional pre-G0) | `spec.md`, frozen-candidate tests, QoI instruments | objections list ("test passes trivially if…", "spec silent on…", "this test is satisfied by a degenerate solution") | — |

---

## 16. Supervisor & session lifecycle

Wake conditions: run terminal event · anomaly event · user message · checkpoint timer · gate awaiting deliberation. On wake: acquire the study lock → build the reconciled orient pack → spawn the deliberator → session acts → session writes `handoff` to `workspace.json` as its last act → release lock.

### 16.1 Lock leases

`lock.json` carries `{holder, since, lease_expires, heartbeat}`. A live session heartbeats; a lease that expires without a heartbeat is reclaimable. On reclaim the supervisor emits a `system.lock_reclaimed` event, marks the interrupted session's work as unconfirmed in the orient pack, and re-derives the next action from events and gate state rather than trusting the dead session's handoff. Without a lease, a session that dies holding the lock hangs the study permanently.

### 16.2 Handoff reconciliation

A handoff is a hypothesis written at a moment in time, and events keep arriving. The watchdog can kill a run and the user can amend the spec between the handoff being written and the next session starting. The orient ritual therefore never ships a raw handoff. It ships a reconciliation:

- every event with an id later than the handoff's `events_seen_through` watermark, in order;
- every spec amendment later than the handoff, with the affected sections, read from `context/amendments.jsonl` (§6.5) — which is what gives this clause a concrete store rather than an implied one;
- any user turn appended since the watermark that has not yet been reflected in the spec, flagged as a candidate missed amendment;
- an explicit verdict: *handoff still valid*, *handoff superseded by <events>*, or *handoff unverifiable* — the last requiring the session to re-derive its next action from gate state and events before acting.

The handoff records the watermark precisely so this comparison is cheap and total.

```json
{"study":"study-001-elbow","phase":"P5",
 "lock":{"holder":null,"lease_expires":null},
 "gates":{"G0":{"status":"pass","by":"user","at":"…"},
          "G4":{"status":"pass","evidence":"runs/run-01/digests/run.md"}},
 "runs":{"run-02-medium":{"status":"running","pid":41233,"watchdog":"active","last_event":"e-041"}},
 "budget":{"cpu_h":{"cap":40,"spent":12.4},"tokens":{"cap":8e6,"spent":2.1e6},
           "wall_h":{"cap":72,"spent":19}},
 "attempts_frontier":["a-007"],
 "handoff":{"next_action":"on run-02 terminal: regenerate digests, run G6 suite",
            "events_seen_through":"e-041",
            "wake_on":["run-02 terminal","run-02 anomaly","user message"],
            "watch":["k bounded 3× early in smoke — if it recurs at scale, retrieve M-12 precedent"]}}
```

**User touchpoints:** the simulate-vs-correlate decision and G0 approval (spec, tests, QoI instruments) · pre-G5 cost approval · G7 delivery · escalations, always structured: gate, error class, attempts as change→result pairs, hypothesis, costed options. Questions are batched at gates rather than dribbled mid-pipeline.

---

## 17. Spec format

Provenance tags: `[user]` · `[inferred]` (standard default) · `[assumed]` (flagged; proceeds unless corrected) · `[derived]` (computed, calc-linked) · `[retrieved]` (taken from a corpus precedent, tier named) · `[BLOCKING]`.

`[retrieved]` is its own tag rather than folded into `[inferred]` because the corpus is the one source that can feed the system its own past output (§9), and G0 approval is where that has to be visible.

**Both non-inferential tags carry resolvable pointers, and both are checked at G0.** `[user]` carries `turn_id` plus a character span into `context/transcript.jsonl` (§6.5). `[retrieved]` carries `{tier, index_path, of_version, verdict?, reference_value?}` into the corpus index row, and the approval view renders the tier and the resolved precedent beside the spec line — for the earned tier, with the originating study's G6 verdict and whether that study had a benchmark anchor for its case class. A pointer that fails to resolve blocks G0 in either case: an unresolvable `[user]` span means the transcript was truncated or the entry mislabelled, and an unresolvable `[retrieved]` row means the spec is carrying a convention whose provenance has been lost, which is exactly the state the earned tier is most likely to produce and least likely to advertise.

`[inferred]`, `[assumed]` and `[derived]` need no pointer — the first two are the system's own defaults, and the third links to its calc.

An ambiguity blocks only if plausible answers change (a) solver/physics, (b) the QoI, (c) cost by more than a configured factor, or (d) a pass/fail outcome.

```markdown
# spec.md — study-001-elbow                    status: awaiting-approval

## Question
Pressure drop of air through a 90° L-junction, D=50mm circular, 5 m/s.   [user + assumed]

## Analytical-first triage
Correlation answer available: Idelchik elbow K ≈ 1.1 + Darcy friction over 30D
→ Δp ≈ 29 Pa ± 25 %.  [derived: calcs/exp-dp.py]
Simulation requested anyway for: velocity field in the downstream leg and
separation extent, which the correlation does not provide.        [user-confirmed]

## Case class
internal / incompressible / steady / single-phase                         [derived]

## Regime classification
shedding_risk: low — internal confined flow, no bluff body, attached except at
the elbow inner radius; Re 1.65e4.  Inputs and thresholds: calcs/regime.json
Steady solver justified on this basis.  Re-checked at G2 against realized geometry.

## Geometry
source: authored (blockMeshDict; seeded from corpus/curated/elbow90)      [inferred]
D = 0.05 m            [assumed — "junction" gave no size; correct if wrong]
legs: 10D up, 20D down                                                    [inferred]
elbow: sharp (r/D = 0) [assumed — affects K strongly]

## Fluid & flow
air 20°C: ρ=1.204, ν=1.516e-5                                             [inferred]
inlet U = 5 m/s uniform [user "airflow" + assumed speed]
outlet p = 0 gauge                                                        [inferred]
Re = 1.65e4 → turbulent · Ma ≈ 0.015 → incompressible     [derived: calcs/re.py]

## Physics
simpleFoam · kOmegaSST · blended wall treatment
[decisions.md D-003 · seeded-from: pitzDaily, tutorials-index hits n=23]

## QoI instruments  →  spec_frozen/qoi_instruments/  (frozen on approval)
dp: surfaceFieldValue, areaAverage of p, patches (inlet, outlet), difference,
    written every 10 iterations
    field dimensions: [0 2 -2 0 0 0 0]  (kinematic — simpleFoam solves p/ρ)
    conversion: multiply_by_density
      source: constant/transportProperties:rho    # named + sourced, not a literal
      → reported dimensions [1 -1 -2 0 0 0 0] (Pa)
recomputation mode: field-recompute
    qoi_recompute.patch_pressure_difference, using the spec patch contract

## Quantity definition  (reviewed pre-G0 — §15)
instrument measures: difference in area-averaged STATIC pressure, inlet↔outlet,
    steady, gauge
expectation measures: TOTAL pressure loss (Idelchik K is defined on total)
verdict: accepted-mismatch
    the profiles differ between a developed inlet and a recovering outlet at 20D;
    the kinetic-energy correction difference is estimated at ≤5 % of Δp
    [calcs/ke-correction.py]
    → Tier-3 band widened from ±50 % to ±55 %

## Success criteria  →  spec_frozen/tests/  (frozen on approval)
test_pressure_drop: dp plateaued (<1 %/1k it) AND within ±55 % of 29.2 Pa
[expectation authored first, calcs/exp-dp.py
 value: 29.2 Pa   units: Pa   ← declared; dimension-checked against the
   instrument's converted output before any magnitude comparison (V-09)
 uncertainty class: composed correlation (minor loss + friction, interaction
   neglected) → base band ±50 %, widened to ±55 % by the accepted definitional
   mismatch above]

# Note on why the conversion is explicit: without it the instrument returns
# 24.3 m²/s², the expectation is 29.2 Pa, and 24.3 sits inside a ±50 % band
# around 29.2.  For air the kinematic/dynamic error is a factor of 1.204 and no
# usable band catches it; for water it is 998 and any band does.  The band is
# not the mechanism — the dimension check is.

## Grading run
role: finest converged rung of the mesh ladder

## Mesh plan
authored blockMesh O-grid · ladder 0.8M / 2.4M / 7M · refine elbow + 5D downstream
independence: Richardson on dp, with non-monotone handling (§11.6)
validity envelope for the GCI verdict: this geometry, Re 1e4–3e4

## Budget
smoke: coarse rung of the ladder (free)
production estimate: fine rung ≈ 8 CPU·h, full ladder ≈ 15–20 CPU·h
  [derived: calcs/cost.py — 7M cells × ~2000 iters × ~2 µs/cell/iter/core;
   recalibrated at G4 from measured s/cell/iter]
caps: 40 CPU·h · 8M tokens · 72 h wall

## Open questions (blocking)
none — assumptions above used unless corrected

## Amendments
(none)
```

A drone study adds `## Study plan` (alpha/beta sweep table; `test_static_stability` over the polar), a `props: actuator-disk vs resolved [BLOCKING]` fork, a `shedding_risk: high` classification at post-stall angles that forces URANS spot-checks, and a per-angle mesh-independence validity note. A compliance study has the standards subagent draft `test_compliance.py`, with `predicates.md` rendering it human-readable; both ride the same G0 approval, since a user can verify a predicate table against clause numbers in minutes and an LLM misreading a standard is a real risk.

**Expectations are non-optional, pre-test, typed, dimensioned, and defined.** Hand-calcs and correlations are authored before any test code exists, must account for every term the geometry implies, must declare units and the physical quantity they describe, and must declare an uncertainty class that sets the Tier-3 band (§11.5).

The band is not what catches a units error. A ρ-factor on air is 1.204 and fits inside any band wide enough to be useful; the dimension check (V-09) is the mechanism, and the band's job is only to catch magnitude errors within a correctly-dimensioned, correctly-defined comparison. Wrong BCs remain the case the band does catch, since they typically move the answer by much more than ρ.

---

## 18. Failure taxonomy & playbooks-as-advice

The taxonomy is a classification vocabulary. It keys digest classifiers, watchdog events, attempt records, and corpus retrieval, and it grows from postmortems rather than upfront enumeration.

**Geometry:** GEO-01 not watertight · GEO-02 units/scale · GEO-03 inverted normals · GEO-04 self-intersections · GEO-05 multi-solid / region-name mismatch

**Mesh:** M-01 nonOrtho high · M-02 skewness · M-03 negative volumes · M-04 layer collapse · M-05 snappy leak · M-06 refinement not realized · M-07 snap failure at sharp features · M-08 locationInMesh invalid · M-09 unresolved region reference · M-10 cell-count prediction out of envelope · M-11 layer spec infeasible for base cell size

**Case:** C-01 patch↔field mismatch · C-02 parse error · C-03 solver/model incompat · C-04 unstable BC pairing · C-05 bad initialization · C-06 QoI instrument collides with a frozen name

**Run:** R-01 residual divergence · R-02 Courant explosion · R-03 bounding spam · R-04 timestep collapse · R-05 residuals plateau high · R-06 QoI oscillation · R-07 continuity error growth · R-08 crash/FPE · R-09 walltime/disk

**Verify:** V-01 QoI not plateaued · V-02 GCI too high or non-monotone · V-03 y⁺ coverage below floor on QoI surfaces · V-04 mass imbalance · V-05 expectation outside its declared band · V-06 frozen instrument disagrees with independent recomputation (plumbing) · V-07 scheme-sensitivity probe moves the QoI beyond tolerance, or diverges · V-08 benchmark regression outside band · V-09 dimension mismatch between QoI and expectation · V-10 definitional mismatch — instrument and expectation measure different quantities at matching dimensions · V-11 URANS spot-check inconclusive on a non-low-risk case

Note the relationship between V-06, V-09, and V-10: they are three different independence failures, and each is invisible to the others. V-06 catches wrong plumbing at a correct definition. V-09 catches a correct quantity in the wrong units. V-10 catches correct units and correct plumbing measuring the wrong thing — two instruments agreeing perfectly on a quantity that is not the one the expectation describes.

**Playbooks are advice, not menus.** On a classified failure the remediation context is curated playbook notes ∪ retrieved precedent — past attempt-graph branches with this signature, ranked by outcome. The deliberator authors the actual fix; thrash is prevented by the progress requirement and escalation (§13), not by constraining the fix.

```yaml
M-04-layer-collapse:            # advice, not an op menu
  symptoms: [coverage below threshold, thickness fraction low, collapse renders]
  notes:
    - widespread collapse → fewer layers / gentler expansion usually dominates
    - feature-localized → minThickness down, featureAngle up
    - curvature-driven → local surface refinement level +1
    - check M-11 pre-flight: total layer thickness may exceed the local base cell
    - partial coverage away from QoI zones → acceptable with waiver
  retrieve: attempts where signature=M-04, ranked by resolution
```

---

## 19. Studies & compliance

Hierarchy: **study → cases → runs**, with a shared mesh strategy and an aggregate digest. Sweeps are supervisor for-loops with per-case state; parallel cases run under one watchdog, capped by budget. Mesh independence is itself a study, and its verdict transfers only within its recorded validity envelope (§11.6).

Compliance adds the standards→tests subagent (§15) and a report section mapping test → measured value → verdict → evidence artifact, with the honest framing that CFD is design-stage evidence where standards specify physical measurement protocols.

---

## 20. Outputs

`report.md`: question → analytical-first triage and why simulation was chosen → method (reconstructed from the disk fingerprint, not from memory) → results against expectations with their declared bands → G6 suite results table → benchmark agreement for the case class, or an explicit statement that no benchmark exists → limitations → confidence statement → compliance mapping.

The confidence statement is generated from evidence actually present, not asserted. Its strongest form requires a benchmark-anchored case class, a converged ladder with a valid GCI, an expectation in a tight uncertainty class, and no waivers. Its weakest form says so plainly.

```json
{"qois":{"dp":{"value":31.4,"units":"Pa","dimensions":"[1 -1 -2 0 0 0 0]",
               "conversion":"multiply_by_density(constant/transportProperties:rho)",
               "quantity":"static pressure difference, inlet-outlet, gauge",
               "definition_verdict":"accepted-mismatch (static vs total, ≤5 %)",
               "last_window_slope":0.001,
               "gci_pct":3.1,"gci_mode":"monotone",
               "expectation":29.2,"expectation_units":"Pa",
               "expectation_band":"±55 % (composed correlation + accepted mismatch)",
               "dimension_check":"pass",
               "recompute_mode":"field-recompute","recompute_value":31.6,
               "recompute_tolerance":0.4,"recompute_diff":0.2,
               "verdict":"pass"}},
 "gates":{"G0":"user-approved","G6":"pass"},
 "regime":{"shedding_risk":"low","reclassified_at_G2":"low",
           "urans_spotcheck":"not-required",
           "scheme_sensitivity":{"axis_varied":"divergence scheme (2nd→1st order)",
                                 "qoi_movement_pct":1.8,"outcome":"stable"}},
 "grading_run":"run-03-fine",
 "benchmark":{"class":"internal-elbow","reference":"Idelchik 6-1","agreement_pct":7.6},
 "tests":[{"id":"test_pressure_drop","verdict":"pass",
           "evidence":"runs/run-03/digests/qoi.md"},
          {"id":"test_ashrae_draft_limit","clause":"ASHRAE 55 §5.3.3","measured":0.34,
           "verdict":"pass","evidence":"runs/run-03/digests/occupied_zone.md"}],
 "waivers":[],
 "spend":{"cpu_h":17.2,"tokens":3.4e6,"wall_h":26}}
```

Plus `postmortem.md` (taxonomy gaps → new digest-classifier rules; playbook candidates; lint-rule candidates; advisory precision figures for this study) and `manifest.json` (hashes, OpenFOAM version and fork, script versions, `numberOfSubdomains`, decomposition method, MPI ranks).

**Reproducibility claim.** The bundle reproduces the study to solver tolerance, not bit-for-bit. Parallel OpenFOAM results depend on decomposition through reduction ordering, so a different `numberOfSubdomains` or decomposition method gives a slightly different answer; the manifest records both so that an exact-decomposition rerun is possible, and the report states the claim as reproducible-to-tolerance.

---

## 21. Build order — milestones with acceptance tests

| M | Deliverable | Acceptance test |
|---|---|---|
| M0 | bundle layout + split state + state service + `orient` + sandbox policy + lock leases | fresh session rehydrates a fixture bundle and states the correct next action; protected paths enforced; a killed session's lock is reclaimed and its handoff marked unverifiable; a direct write to `gates.json` fails; **untyped or wrong-producer evidence is rejected, evidence whose `graded_config_sha256` is stale is rejected, an out-of-graph transition (G6 pass with G2 failed) is rejected, and a killed state service leaves boundary tools failing closed rather than proceeding on cached reads; `gates.json` rebuilt from evidence records + events matches the original; the deliberator can append to `context/transcript.jsonl` but a rewrite or truncation attempt fails and emits an event** |
| M1 | projector registry + reader + generators + renders + echo parser + context bank | 2 GB fixture case → full view set; deliberator answers 10 questions correctly from views alone; written-vs-announced diff produced; **every artifact type present in the fixture bundle resolves to a registry entry and a read with no entry fails rather than falling back to raw**; a non-round-trippable view is stamped and a write-back attempt through it is refused; staleness correct on a 5 GB field directory via the fast path, and a same-size same-mtime content change is still caught on rehash; **a spec `[user]` tag resolves to its exact transcript span, an unresolvable span blocks G0, and a rollup regenerated from a truncated transcript fails rather than serving a shorter history silently** |
| M2 | corpus: benchmark tier + tutorial indexer + study indexer + `search.py` + percentile stats | elbow spec → retrieval returns sane seeds; benchmark case reproduces its published value within band; keyword value distributions produced for common keys; index schema round-trips both ESI and Foundation solver identities; **a `precedent.md` view states tier and provenance before any content, a `[retrieved]` spec pointer resolves to its index row and an unresolvable one blocks G0, and a retrieval where a benchmark hit was outranked by an earned hit is recorded with both** |
| M3 | oracle stack: keyword harvest, xfile lint, selection oracle, dict probes 5a/5b, advisory budget | **recall**: injected dictionary errors including silent-optional-keyword typos are flagged at the configured rate; **precision**: measured on a corpus of valid exotic cases, meeting the configured floor, with zero blocking false positives from 5b; 5a self-skips correctly on AMI/zone/nonuniform cases |
| M4 | mesh: pre-flight + freeform authoring path + staged checkers | agent-authored blockMeshDict for elbow passes G2; each injected defect caught at the right sub-gate; **every pre-flight defect (bad `locationInMesh`, unresolvable region name, cell-count blowup, infeasible layer spec) caught before any build starts**, verified by asserting no snappy invocation occurred |
| M5 | run infra: launcher + watchdog + events + fingerprint guard + confinement | seeded divergent case killed early; duplicate-config launch refused and `force` override logged; session spawned on terminal event; a `#codeStream` case runs no less confined than the sandbox |
| M6 | spec flow + freeze + hermetic G6 runner + launch-time instrument verification + smoke policy | elbow end-to-end autonomous: request → triage → approved spec/tests/instruments → report with correct Δp; **freeze adversarial suite**: writable-parent `conftest.py`, shadowed module on path, edited `pytest.ini`, and monkeypatched helper each cause G6 to abort on closure-hash mismatch; **an edit to `system/functionObjects` after a G3 pass aborts the launch**; post-freeze test edit routed to `amend_spec` |
| M7 | attempt graph + retrieval-driven remediation + progress-based loop control | seeded snappy failure fixed within budget; a loop with a stalled progress metric escalates rather than continuing; the same signature in a second study retrieves the first study's resolution |
| M8 | G6 instruments: dimensioned interfaces, recomputation modes, tiered expectations, GCI with non-monotone handling, y⁺ coverage | **the kinematic-pressure case is blocked on dimensions, on air, where the magnitude sits inside the band** — this is the primary test, because a band-only system passes it; a seeded definitional swap (static differencing graded against a total-pressure expectation, both instruments agreeing) is caught by quantity-definition review and **not** by V-06, verifying the two are distinct; independent recomputation catches a seeded QoI-instrument patch swap; expectation gate blocks a seeded missing-friction-term expectation; transient comparison tolerance computed from the sample separates a sampling difference from a wrong patch; oscillatory triplet produces a bounded estimate rather than a GCI number; y⁺ coverage passes a real geometry with stagnation points |
| M9 | regime classification + scheme sensitivity + URANS spot-check | a seeded steady-solve-on-shedding-flow case is caught on a case where **residuals converge cleanly and the QoI does not oscillate**; **the URANS probe is validated on growth rate, not amplitude — a window deliberately too short to show visible amplitude must still return a positive growth-rate fit, and an unperturbed control on the same case must be shown to return a false negative**, so the probe's own mechanism is what passes the test rather than a conveniently detectable seed; the scheme probe reports `not-applicable` rather than zero sensitivity when the baseline has no headroom on the varied axis; a diverging scheme probe is classified V-07, not R-01 |
| M10 | studies (alpha sweep) + parallel watchdog | 5-case sweep, first case as smoke, aggregate digest, `test_static_stability` over the polar; GCI inheritance refused at angles outside the validity envelope |
| M11 | compliance path | fixture standard PDF → drafted tests → approval → freeze → verdict table with clause-cited evidence |

Highest-leverage components: **M1** (perception quality bounds every judgment downstream — and the single reader raises the stakes rather than lowering them, since one door means one failure point and a projector that quietly drops what mattered is now the whole view), **M3** (the oracle stack makes freeform authoring safe on the expensive end), **M4** (the mesh pre-flight is the same argument at the other end of the cost scale), and **M9** (the only defence against the failure mode that produces a confident wrong number with clean residuals).

---

## 22. Open decisions

1. **OpenFOAM fork/version**: ESI v2412+ recommended for snappy maturity and `foamToC`. The selection oracle adapts to either fork automatically, but the corpus schema, the per-class defaults in Appendix D, and any retrieval keyed on solver identity do not adapt for free — hence the `{executable, module}` pair in the index row. Foundation v11+ replaced per-physics solvers with `foamRun -solver <module>`; supporting it is a schema and defaults exercise, not just an oracle one.
2. **Harness**: Claude Code / custom loop / MCP — immaterial to the contracts; the sandbox plus boundary tools are the interface.
3. **Queue**: local nohup first; SLURM adapter later (touches launcher and watchdog only).
4. **Autonomy defaults**: auto-advance G1→G4; always stop pre-G5 (recommended; confirm).
5. **Render stack**: pvpython headless (recommended); whether it runs inside the sandbox or as a boundary service is open.
6. **Retention**: field-data purge policy per budget tier. The circular dependency with recomputation is resolved by declaring the recomputation mode at G0 (§11.1), so retention is a consequence of a declared choice rather than an open-ended obligation. What remains open is the default subsample size for `subsample-recompute`, and whether `parallel-instrument` is acceptable as the standing default for transient QoIs given that it is code-independent but not run-independent.
7. **Sandbox caps**: exact truncation limits.
8. **Benchmark tier scope**: which case families to seed first, and whether benchmark regression gates releases or only warns.
9. **Advisory precision floor**: the numeric target, and whether demotion is automatic or requires review.
10. **Default view for templated dictionaries**: identity or `expand`. Identity is recommended because it is the editable form and P4 makes authoring the load-bearing path, but a heavily `#include`-d case is materially harder to reason about un-expanded. Serving both by default doubles the token cost of the most-read artifact class.
11. **Fidelity measurement**: what `fidelity` actually means for a summary with no ground truth, and the floor below which a projector is served as `degraded` rather than plainly. The standards-PDF extractor is the forcing case — it is currently marked degraded by assertion, not by measurement.
12. **Context-bank verbatim window**: how far back the *view* serves turns before rolling up. Storage is lossless either way, so this is a token-budget decision, not a retention one.
13. **Transcript redaction**: whether a bundle can be shared, archived or published with the transcript redacted, and what an `approvals.jsonl` record still certifies once the turns it references are gone. Bears directly on §19 compliance deliverables, where the approval chain is part of what is being attested.

---

## Appendix A — schemas (abridged)

**`state/gates.json`**: `{study, phase, gates{G*:{status, by, at, evidence}}}` — written only by the state service, which accepts transitions only from `run_gate` and validates that the evidence path resolves.

**`state/budget.json`**: `{cpu_h{cap,spent}, tokens{cap,spent}, wall_h{cap,spent}, soft_thresholds{}}` — written only by the state service.

**`state/lock.json`**: `{holder, since, lease_expires, heartbeat}` — written only by the supervisor; expiry triggers reclaim (§16.1).

**`state/workspace.json`**: `{runs{}, attempts_frontier[], handoff{next_action, events_seen_through, wake_on[], watch[]}}` — deliberator-writable.

**Evidence record**: `{gate, type, produced_by, artifact, artifact_sha256, graded_config_sha256, verdict, at}` — the only input the state service accepts for a gate transition. `type` must be the type declared for that gate; `produced_by` must be the checker registered for that type; `artifact_sha256` must match disk; `graded_config_sha256` must match the current configuration fingerprint or the evidence is stale.

**Gate evidence types**: G0 → `spec_approval` (user) + `quantity_definition_review`; G1 → `surface_checker_verdict`; G2 → `mesh_preflight_verdict` + `mesh_checker_verdict`; G3 → `oracle_stack_verdict` (must include a post-mesh probe result); G4 → `smoke_verdict`; G5 → `run_terminal_record`; G6 → `hermetic_suite_report`; G7 → `manifest`.

**Gate transition graph**:

```
G0 → G1 → G2 → G3 → G4 → G5 → G6 → G7

predecessors (all must be pass, none stale):
  G1: G0        G2: G1        G3: G2        G4: G3
  G5: G3 and (G4 pass or approved soft-smoke waiver)
  G6: G2, G3, G5      G7: G6

stale propagation:
  spec amendment, physics level      → G1..G7 stale
  spec amendment, test or instrument → G6, G7 stale
  geometry change                    → G1..G7 stale
  mesh change                        → G2..G7 stale
  dictionary change                  → G3..G7 stale
  executor version change on a checker → gates whose evidence that checker produced go stale

legal statuses: pending · pass · fail · stale · waived(reason, by)
```

Any transition not in this graph is rejected by the state service.

**Attempt**: §12. `change.computed_diff` from canonical-expand diff; `fingerprint` = sha256 over (geometry hash, canonical expansion of all dicts minus `config/fingerprint_exclusions.yaml`, mesh-generation code hash).

**Event**: `{id, run, type: terminal.*|anomaly.*|budget.*|user.*|system.*, t_sim, wall_min, severity, evidence, action_taken}`.

**Corpus index row**: `{path, tier, solver{executable, module}, turbulence, regime{class, Re, Ma, shedding_risk}, mesh_type, bc_map, of_version, of_fork, verdict, reference_value?}`.

**Projector registry entry** (`executors/projection/registry.yaml`): `{match: glob|type, view_kind: identity|transformed|summary|summary+renders, round_trippable: bool, fidelity: full|<0–1>|degraded, identity: hash|hash-with-fast-path, generator: <path>|null, window: null|{verbatim_tail, rollup}}`. A read whose artifact matches no entry is an error; there is no raw fallback through the reader.

**Served view header** (prepended to every read, all view kinds including identity): `{artifact, view_kind, fidelity, source_sha256, identity_checked_at, window_applied, round_trippable, generator_version}`.

**Transcript record** (`context/transcript.jsonl`, O_APPEND): `{turn_id, at, role: user|system, text, attachments[]?, gate?}` — `text` verbatim, never rewritten.

**Approval record** (`context/approvals.jsonl`): `{turn_id, gate, artifact, artifact_sha256, rendered_sha256, by, at}` — `rendered_sha256` hashes what the user was actually shown, which is not necessarily the file now on disk at `artifact`.

**Amendment record** (`context/amendments.jsonl`): `{turn_id, at, sections[], level: physics|test|instrument|other, stale_gates[]}` — `level` drives the stale-propagation rules below.

**Freeze manifest**: `{files{path: sha256}, python_modules{name: sha256}, pytest_version, plugin_versions{}, approved_at, approved_by}`.

## Appendix B — watchdog defaults

Configured in `config/watchdog.yaml`; values below are starting defaults, not architecture.

Poll 30 s · `FOAM FATAL` → kill · residual ratio > 10³ over 100 iterations → kill · Co > 2 for 20 consecutive transient steps → kill · Δt < 1e-9 → kill · bounding > 50 per 100 iterations → warn · disk < 10 % → pause · soft-smoke: first ~300 iterations with kill thresholds divided by 10.

## Appendix C — oracle stack quickref (G3 default order)

parse → xfile lint → keyword lint (advisory, budgeted) → percentile lint (advisory, budgeted) → pre-mesh dict probe (advisory, self-skipping) → [on demand: selection oracle, `foamToC` first] → **post-mesh dict probe on the real mesh (blocking)** → smoke/production → echo diff → G6 suite.

## Appendix D — per-case-class defaults (starter)

Solver names given for ESI v2412+; the Foundation equivalents are `foamRun -solver <module>` and are carried in the corpus index as the `module` half of the solver pair.

| Class | Solver | Turb | Domain rule | Wall treatment | Default shedding risk | Smoke |
|---|---|---|---|---|---|---|
| internal-duct-steady | simpleFoam | kOmegaSST | fit + development lengths (10D/20D) | blended | low | ladder-coarse |
| external-aero-steady | simpleFoam | kOmegaSST | 10D/20D/10D box | blended | **medium — scheme probe required; high if bluff or post-stall** | required, first of study |
| indoor-buoyant | buoyantSimpleFoam | kEpsilon | room + plenum | blended | low–medium | required |
| transient-shedding | pimpleFoam | kOmegaSST (LES later) | per class | per treatment | high | short-window smoke required |

Class membership is assigned by `regime/classify.py` at G0 and re-checked at G2 against the realized geometry (§11.4), not fixed by the initial text request.
