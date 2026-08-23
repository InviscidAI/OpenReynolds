# Plan 2 — openreynolds: The Agent

Open source. A UI-less CFD agent: an Anthropic tool-use loop, a small tool surface, a backend abstraction (hosted OpenFOAM service first, local OpenFOAM later), and an optional toolbox. Works entirely from a terminal.

**Relationship to the other repos:** it talks to the hosted service (Plan 1, "foamd") over HTTPS with an API key and never imports its code — the published `/v1` spec is the only coupling. A UI comes later and must not be designed for now; the only obligation toward it is already satisfied by capture plumbing (§11).

**Relationship to the architecture document:** it is a reference library, not a spec. Its durable ideas — cheap checks before expensive operations, digests and renders for multi-GB artifacts, retrieval from precedent — survive here **demoted to optional capabilities**. Its control structure (gates G0–G7, state service, frozen specs, hermetic runners, locks, provenance enforcement, budgets, supervisor/deliberator split) does not survive at all, and none of its checker inventory is on the implementer to port. Cherry-pick later only if the agent's actual behavior shows something is missing.

---

## 1. The free-will contract (binding design constraints)

The agent decides everything about how to work. The harness is plumbing. Concretely:

**The harness MAY:**
- cap tool output sizes, with a marker that says exactly where the full data lives and how to window into it;
- sync the toolbox directory into the workspace at session start;
- poll running jobs in pure code and wake the model with *facts* (job exited, rc, log tail);
- post capture records (messages, results, artifacts) to the platform;
- assemble a factual situation blurb on resume (instance id, running jobs and their statuses — no interpretation).

**The harness MUST NOT:**
- enforce any ordering of actions, phases, or "check X before Y";
- block or rewrite a tool call on policy grounds;
- require approvals, verdicts, or sign-offs before anything runs;
- inject step-by-step instructions, checklists, or mandated workflows — in the system prompt, in wake messages, or in any file the agent is required to obey;
- grade, veto, or amend the model's outputs.

There is no gate DAG, no state machine, no lock, no watchdog with authority, no budget the model must reason about. If the agent wants to write itself a spec, tests, or a checklist, it can — and nothing verifies that it did.

---

## 2. Stack & repo layout

Python. Anthropic SDK for the loop; `httpx2` for the hosted backend (the Anthropic SDK is built on it, so using plain `httpx` would mean two HTTP stacks in one process); `rich` for terminal output; `click` for the CLI.

```
openreynolds/
  openreynolds/
    cli.py           # entry: interactive session, -p one-shot, --study resume, watch mode
    loop.py          # Anthropic messages loop: streaming, tool dispatch, context strategy
    tools.py         # tool schemas + thin handlers (everything delegates to Backend)
    prompt.py        # builds the short environmental system prompt
    backend/
      base.py        # the Backend protocol
      hosted.py      # HTTP client for the foamd /v1 contract
      local.py       # stub — deferred (§12)
    capture.py       # optional platform capture (messages/results/artifacts)
    store.py         # local mirror ./studies/<id>/ + session metadata
    watch.py         # pure-code job polling + wake facts
    config.py        # env, then a user config file outside the repo
    toolbox/         # inside the package so a pip install ships it
      log_digest.py    mesh_digest.py    render.py    cells_estimate.py
      notes/
        openfoam-field-notes.md    # distilled domain wisdom (reference, optional)
        bundle-layout.md           # one suggested /work layout (explicitly optional)
        openfoam-agent-architecture.md   # the full arch doc, optional reading (§14.2)
  tests/
  pyproject.toml
```

---

## 3. The Backend protocol

The whole independence story. Nothing above this interface may know whether it's talking to Docker-over-HTTP or a local install.

```python
class Backend(Protocol):
    def exec(self, cmd: str, cwd: str | None = None, timeout_s: int = 120) -> ExecResult
        # → exit_code, output (combined, capped), truncated, log_path
    def put_file(self, path: str, data: bytes) -> None
    def get_file(self, path: str, offset: int = 0, limit: int | None = None) -> bytes
    def stat(self, path: str) -> Stat           # type, size, mtime, entries for dirs
    def put_tree(self, local_dir: Path, remote_dir: str) -> None    # toolbox sync
    def get_tree(self, remote_paths: list[str], local_dir: Path) -> None  # artifact pull
    def job_start(self, cmd: str, cwd: str | None = None,
                  name: str | None = None, kill_on: list[str] | None = None) -> str
    def job_status(self, job_id: str) -> JobStatus   # running·exited·killed, rc, times
    def job_tail(self, job_id: str, offset: int) -> tuple[bytes, int, bool]
    def job_kill(self, job_id: str) -> None
```

**HostedBackend** maps this 1:1 onto the foamd contract (authoritative spec: foamd's published `spec/openapi.yaml`, pinned by version):

| Protocol method | foamd endpoint |
|---|---|
| `exec` | `POST /v1/instances/{id}/exec` |
| `put_file` / `get_file` / `stat` | `PUT·GET /v1/instances/{id}/files?path=` |
| `put_tree` / `get_tree` | `POST /v1/instances/{id}/tar` |
| `job_*` | `POST /v1/instances/{id}/jobs`, `GET /v1/jobs/{id}`, `GET /v1/jobs/{id}/log`, `POST /v1/jobs/{id}/kill` |

Instance acquisition: the CLI creates (or reuses, `--instance`) an instance at session start; lazy-start on a stopped instance is the backend's problem, invisible above the protocol.

---

## 4. Tool surface (deliberately few)

| Tool | Signature | Semantics |
|---|---|---|
| `bash` | `(cmd, timeout_s?)` | run in the instance; combined output up to ~48 KB, then a marker: `[truncated — full output at <log_path>; use read_file with offset]` |
| `write_file` | `(path, content)` | plain write under `/work` |
| `read_file` | `(path, offset?, limit?)` | windowed reads so the model handles big files **its own way** — no mediation layer |
| `job_start` | `(cmd, name?, kill_on?)` | detached long command → job_id; `kill_on` regexes are the model's opt-in compute-saver, nothing more |
| `job_check` | `(job_id, log_offset?)` | status + incremental log tail in one call (cheap to use repeatedly) |
| `job_kill` | `(job_id)` | |
| `fetch` | `(paths[])` | pull files to the local mirror `./studies/<id>/`, print local paths, register as platform artifacts |

That is the entire surface. No `run_gate`, no `amend_spec`, no `ask_user` tool — asking is just talking; this is a chat. Meshing, checking, rendering, post-processing are all `bash`.

---

## 5. System prompt (content spec)

Short and environmental — roughly one page:

- who you are (a CFD engineer-agent with full control of a Linux workspace with OpenFOAM ESI v2512 — the version pinned in Plan 1);
- what exists: the tools above and their semantics; `/work` persists across sessions (so notes-to-self on disk are useful — stated as information, not instruction); `/work/.toolbox/` holds optional scripts and reference notes, usable, editable, replaceable, or ignorable;
- long-running things go in jobs; while a job runs you can end your turn and you'll be woken with the outcome;
- it's a conversation — ask the user whenever *you* want their input;
- you decide how to work: what to check, when, and whether. Honesty about what was and wasn't verified is the only standing expectation.

**Explicitly absent:** phases, required checks, mandated file formats, "always do X before Y". Verify in review that the prompt contains no imperative workflow language.

---

## 6. The loop

Standard Anthropic tool-use loop: stream text to the terminal, dispatch `tool_use` blocks through `tools.py` → `Backend`, return `tool_result`s, repeat until the model ends its turn.

**LLM transport.** The Anthropic client is constructed from config (`base_url` + key), so proxy-vs-direct is a flag, not a code path. Default: **direct, bring-your-own-key** (`ANTHROPIC_API_KEY`). The loop sets an `X-Study-Id` header either way, so usage ties to the study. LLM transport is deliberately **not** part of the Backend protocol — it belongs to the loop, so a future LocalBackend user inherits whatever transport is configured.

> **Amended.** This section originally defaulted to a platform LLM proxy at `<FOAMD_URL>/v1/llm` billed against a master key. Plan 1 subsequently dropped that proxy — its F5 shipped as "BYOK + attribution ledger", the published `openapi.yaml` states there is no `/v1/llm`, and there is no `llm_usage` table. **A hosted proxy is future work, tracked as a TODO in `openreynolds/config.py`**; nothing proxy-specific is built. If one ever ships, `llm_base_url` points at it and the key changes — a config change, not a code change.

**Context strategy — resume doubles as compaction.** A resumed session (`--study X`) is a fresh thread: system prompt + a machine-assembled factual blurb (instance id, running jobs + statuses, study id) + the user's message. The model reorients itself from `/work` — supported, never scripted. When a live session approaches the context window, the harness uses the same move: tells the model the thread is being refreshed (so it can jot anything to disk it wants to keep), then rebuilds as a resume. The model's memory is the filesystem plus whatever notes it chose to keep — which is the free-will version of the architecture doc's "state must outlive context."

---

## 7. Toolbox v1 — four scripts and some notes, nothing more

Synced to `/work/.toolbox/` at session start. Small on purpose; not a port of the architecture doc's executor tree.

| Script | Does |
|---|---|
| `log_digest.py` | solver log → residual plot PNG + last-iteration table + continuity summary |
| `mesh_digest.py` | `checkMesh` output → compact metric table (no verdicts — the model judges) |
| `render.py` | a few fixed pyvista scenes: mesh cuts, \|U\| and p slices, saved as PNGs |
| `cells_estimate.py` | rough snappy cell-count prediction from STL + dicts — the one place cheap foresight reliably saves an hour-long doomed build |

`notes/openfoam-field-notes.md`: a few distilled pages — the cost asymmetry (checks are cheap before meshes and solves), common failure signatures and what usually fixes them, per-class solver/turbulence defaults, classic traps (kinematic vs dynamic pressure, mm-vs-m geometry, steady solvers quietly damping shedding flows), and the long-solve pattern (sane `writeControl` intervals + `startFrom latestTime`, so a solve interrupted at the hosted 24 h container lifetime resumes with one relaunch). Written as an engineer's field notes, not procedures. `notes/bundle-layout.md`: one suggested `/work` layout, labeled as a suggestion.

---

## 8. Long runs & watch mode

The model launches a job and ends its turn. The CLI enters **watch mode**: pure-code polling of `job_check` every 15–30 s. Wake triggers: job exited (wake message = job name, rc, `end_reason`, last ~2 KB of log), a `kill_on` fired (include the matched line), or the user typed something (immediate wake with their message). Wake messages are facts, never suggestions. One `end_reason` worth knowing: `sandbox_expired` means the hosted instance's 24 h container lifetime ran out mid-solve — the case, its write times, and the log all survive on the persistent volume, and OpenFOAM restarts natively from `latestTime`, so whether and how to relaunch is the model's call like everything else.

Closing the laptop is fine — jobs live on the instance. `openreynolds --study X` later resumes via §6. The CLI process is the entire "supervisor"; there is no daemon, no lock, no lease, no handoff schema.

---

## 9. Sessions & local mirror

`./studies/<id>/` on the user's machine holds session metadata (instance id, study id, message log) and everything `fetch`ed (renders, reports). Working state lives in `/work` on the instance in whatever layout the agent chose. `openreynolds studies` lists local sessions.

---

## 10. Capture plumbing (invisible to the model)

On by default, `--no-capture` to disable, **never blocks the study** if the platform is unreachable — buffer and retry, then drop with a warning.

- every user message, assistant message, and tool call/result summary → `POST /studies/{id}/messages` (content capped sanely);
- every `fetch`ed file → `POST /studies/{id}/artifacts`;
- at session end, if the agent happened to leave a `results.json` (or similar, per its own habits) in `/work`, upload it via `POST /studies/{id}/results`. The system prompt may *mention* this pickup exists; it never requires it.

This is the platform-value capture that makes the closed pieces worth building, and it's pure harness — the model's decision-making never touches it. It also fully pre-satisfies the future UI's data needs.

---

## 11. CLI UX

| Command | Behavior |
|---|---|
| `openreynolds` | new study, interactive session |
| `openreynolds -p "…" ` | non-interactive: run until the model is done and no jobs remain, print, exit |
| `openreynolds --study <id>` | resume (fresh-thread reorientation) |
| `openreynolds --instance <id>` | reuse an existing instance |
| `openreynolds studies` | list local sessions |
| `openreynolds config` | set `FOAMD_API_KEY`, `ANTHROPIC_API_KEY`, base URL, model |

Fetched PNGs print their local paths; inline terminal image display (iTerm2/kitty protocols) is an A5 nicety.

**LLM auth — bring your own key.** `FOAMD_API_KEY` covers compute and capture; `ANTHROPIC_API_KEY` covers the LLM, billed to the user directly. Per-user/per-key request attribution is still logged server-side by the platform's audit ledger (Plan 1, F5). Since the agent is open source, direct mode could not have been prevented anyway; it is now simply the only mode. A hosted proxy stays possible later as a config change (§6).

---

## 12. LocalBackend (deferred, one negative obligation now)

Later: the same protocol over `subprocess` with a sourced OpenFOAM environment, jobs via the same `jobd`-style wrapper run locally, capture still pointing at the platform (that's the "local users still send data our way" motion) with `--no-capture` for fully offline use. The only v1 obligation is negative: **nothing above the Backend protocol may assume HTTP, Docker, or foamd specifics.** Enforce in review.

---

## 13. Milestones

Dependencies on Plan 1: A1 needs F2 (exec+files), A3 needs F3 (jobs), capture in any milestone degrades gracefully until F4 exists. LLM: all A-milestones can run `--direct-llm` until F5 lands; flipping to the proxy is a config change, not a milestone.

| # | Deliverable | Acceptance — "you can now…" |
|---|---|---|
| A1 | loop + HostedBackend + tools | watch the model run `foamToC -functionObjects`, read and write files on a live instance; transcript rows appear in Supabase (if F4 up, else buffered/skipped cleanly) |
| A2 | toolbox sync | the model reproduces pitzDaily and renders fields; PNGs land in `./studies/<id>/` |
| A3 | jobs + watch mode | a long solve launched, laptop closed mid-run, resumed later, completed; wake-on-exit works; a `kill_on` the *model chose* fires correctly |
| A4 | the elbow, free-form | "simulate airflow through an L-shaped junction" → the agent clarifies as *it* sees fit, authors geometry and case, meshes, solves, sanity-checks however it chooses, delivers numbers + renders + a write-up. **Success includes the negative check: zero forced stops, zero harness-imposed ordering occurred; it asked the user only when it actually wanted to** |
| A5 | polish | streamed output everywhere, resume UX smoothed, `-p` solid for scripting, optional inline images |

A4 is also the joint acceptance with the hosted service.

---

## 14. Decisions (previously open — now resolved)

1. **Model:** default `claude-opus-5` — 1M context, and it supports mid-conversation `role: "system"` messages, which is the right channel for job-wake facts (operator authority rather than impersonating the user, and it leaves the cached prefix intact); `--model` overrides per study. The context refresh (compaction-as-resume, §6) triggers at ~80 % of the window.

   > **Amended.** This originally read `claude-sonnet-4-6`, justified as the sweet spot "on the platform's master key". That justification went away with the master key (§6, §11) — under BYOK the user pays their own, so the choice is capability, not platform cost. `--model claude-sonnet-5` remains a one-flag downgrade; note it does not support the mid-conversation system role, so wake facts fall back to a marked user message.
2. **Reference material:** ship both — the distilled `openfoam-field-notes.md` **and** the full architecture document as one more optional file under `notes/`. Disk is free, context is not, and the agent chooses what to read.
3. **Toolbox sync:** always, at session start. It's under a megabyte; conditional sync isn't worth a code path.
4. **Tool-output caps:** 48 KB default, env-tunable, never a hard wall — the truncation marker always says where the rest lives and how to window into it.
5. **Inline images:** auto-detect iTerm2/kitty graphics protocols and show renders inline; otherwise print local paths. A5 scope.
6. **`fetch`:** auto-registers pulled files as platform artifacts unless `--no-capture` is set.
