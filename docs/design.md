# The design of openreynolds

A UI-less CFD agent: a tool-use loop, a small tool surface, a backend abstraction (a
hosted OpenFOAM service first, local OpenFOAM later), and an optional toolbox. Works
entirely from a terminal.

This document is why the code is shaped the way it is. It is kept true to the code
rather than to the plan it started as: `tests/test_briefing.py` reads this file, on the
grounds that a plan contradicting the code is worse than no plan, because it is the
document someone reads first and it will be believed.

**Relationship to the compute service.** The agent talks to the hosted workspace service
over HTTPS with an API key and does not import its code. The published `/v1` OpenAPI spec
is the coupling, which is what lets either side be replaced. Today the service runs what
it is told and the decisions about how to work are made here; that describes the current
code, and it can change if a different split measures better.

**Relationship to the earlier architecture study.** An earlier internal design proposed a
controlled pipeline: enforced gates, frozen specs, a state service, provenance
enforcement, a supervisor and deliberator split. None of that control structure survives,
deliberately (§1). What did survive is demoted to **optional capabilities** the agent may
reach for and may ignore: cheap checks before expensive operations, digests and renders
for multi-gigabyte artifacts, retrieval from precedent.

---

## 1. The free-will contract (the current design constraints, held by tests)

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

These are the constraints the code has today and three tests hold it to them (`test_prompt.py`, `test_briefing.py`, `test_negative_obligation.py`). They are a thesis with evidence behind it (`found-by-using-it.md`), not a law: if a measured result argues for a different division of labour, change the tests together with the code rather than working around them.

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

- who you are (a CFD engineer-agent with full control of a Linux workspace with OpenFOAM ESI v2512 — the version the workspace image pins);
- what exists: the tools above and their semantics; `/work` persists across sessions (so notes-to-self on disk are useful — stated as information, not instruction); each study works in `/work/<study-id>`, named in the briefing, which commands default to and which a new study finds empty, with the rest of the volume holding other studies' work; `/work/.toolbox/` holds optional scripts and reference notes, usable, editable, replaceable, or ignorable;
- long-running things go in jobs; while a job runs you can end your turn and you'll be woken with the outcome;
- it's a conversation — ask the user whenever *you* want their input;
- you decide how to work: what to check, when, and whether. Honesty about what was and wasn't verified is the only standing expectation.

**Explicitly absent:** phases, required checks, mandated file formats, "always do X before Y". Verify in review that the prompt contains no imperative workflow language.

---

## 6. The loop

Standard Anthropic tool-use loop: stream text to the terminal, dispatch `tool_use` blocks through `tools.py` → `Backend`, return `tool_result`s, repeat until the model ends its turn.

**LLM transport.** The Anthropic client is constructed from config (`base_url` + key), so proxy-vs-direct is a flag, not a code path. Default: **direct, bring-your-own-key** (`ANTHROPIC_API_KEY`). The loop sets an `X-Study-Id` header either way, so usage ties to the study. LLM transport is deliberately **not** part of the Backend protocol — it belongs to the loop, so a future LocalBackend user inherits whatever transport is configured.

**Two transports, and the flag was enough.** This section originally assumed a platform
proxy billed against a master key; that was then dropped in favour of bring-your-own-key
only; and a metered proxy has since shipped after all. All three of those turned out to
be the same code with a different `llm_base_url` and a different key, which is the point
of putting transport in config rather than in the loop.

What exists today:

- **Bring your own key** (the default, and the only mode for a self-hosted agent). The
  agent calls your provider directly with your key. The workspace service never sees it.
- **The metered provider** (`OPENREYNOLDS_PROVIDER=reynolds`). The agent calls
  `{service}/v1/llm` with its *workspace* key as the model key; the service swaps in the
  platform's own credential, relays the request including streaming, and meters the
  tokens into the same monthly budget as compute. No model key of your own.

Neither is privileged in the code. `llm/presets.py` carries both, and `make_provider`
cannot tell which it has.

**Context strategy — resume doubles as compaction.** A resumed session (`--study X`) is a fresh thread: system prompt + a machine-assembled factual blurb (instance id, running jobs + statuses, study id, its own directory and what is in it, and whether anyone is at the terminal to answer) + the user's message. The model reorients itself from `/work` — supported, never scripted. When a live session approaches the context window, the harness uses the same move: tells the model the thread is being refreshed (so it can jot anything to disk it wants to keep), then rebuilds as a resume. The model's memory is the filesystem plus whatever notes it chose to keep — which is the free-will version of the architecture doc's "state must outlive context."

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

**Two keys, two bills.** The workspace key covers compute and capture and is billed to
the workspace account. The model key covers the model and is billed to you by your
provider, or, under the metered provider, to the same workspace account (§6). The service
logs per-key request attribution either way, so usage ties to a study without the service
ever holding a bring-your-own key.

---

## 12. LocalBackend (deferred, one negative obligation now)

Later: the same protocol over `subprocess` with a sourced OpenFOAM environment, and jobs
via the same `jobd`-style wrapper run locally. Capture would keep its current default of
pointing at the workspace service, with `--no-capture` (or `OPENREYNOLDS_CAPTURE=0`) for
fully offline use, on the same terms as today: it is opt-out, `doctor` reports whether it
is on, and it fails quietly rather than delaying a study.

The only obligation this places on the code now is a negative one: **nothing above the
Backend protocol may assume HTTP, Docker, or anything specific to the hosted service.**
That is what makes a local backend an addition rather than a rewrite, and it is enforced
in review and by `tests/test_negative_obligation.py`.

---

## 13. How it was built

Kept because the acceptance column is the useful part: each milestone was defined by what
a person could newly do, not by what had been written. A4 in particular is the one that
tests the contract rather than the code.

Each depended on the corresponding capability existing in the workspace service: the loop
needed exec and files, jobs needed the job API, and capture degraded gracefully until the
capture plane existed.

| # | Deliverable | Acceptance — "you can now…" |
|---|---|---|
| A1 | loop + HostedBackend + tools | watch the model run `foamToC -functionObjects`, read and write files on a live instance; transcript rows reach the capture plane, or are skipped cleanly if it is not up |
| A2 | toolbox sync | the model reproduces pitzDaily and renders fields; PNGs land in `./studies/<id>/` |
| A3 | jobs + watch mode | a long solve launched, laptop closed mid-run, resumed later, completed; wake-on-exit works; a `kill_on` the *model chose* fires correctly |
| A4 | the elbow, free-form | "simulate airflow through an L-shaped junction" → the agent clarifies as *it* sees fit, authors geometry and case, meshes, solves, sanity-checks however it chooses, delivers numbers + renders + a write-up. **Success includes the negative check: zero forced stops, zero harness-imposed ordering occurred; it asked the user only when it actually wanted to** |
| A5 | polish | streamed output everywhere, resume UX smoothed, `-p` solid for scripting, optional inline images |

A4 is also the joint acceptance with the hosted service.

---

## 14. Decisions (previously open — now resolved)

1. **Model:** default `claude-opus-5` — 1M context, and it supports mid-conversation `role: "system"` messages, which is the right channel for job-wake facts (operator authority rather than impersonating the user, and it leaves the cached prefix intact); `--model` overrides per study. The context refresh (compaction-as-resume, §6) triggers at ~80 % of the window.

   > **Amended.** This originally named a cheaper model, justified by what the platform
   > would pay for it. Under bring-your-own-key the user pays their own bill, so the
   > choice is capability rather than platform cost. `--model` remains a one-flag change;
   > note that not every model supports the mid-conversation system role, and where it is
   > missing the job-wake facts fall back to a marked user message.
2. **Reference material:** ship both — the distilled `openfoam-field-notes.md` **and** the full architecture document as one more optional file under `notes/`. Disk is free, context is not, and the agent chooses what to read.
3. **Toolbox sync:** always, at session start. It's under a megabyte; conditional sync isn't worth a code path.
4. **Tool-output caps:** 48 KB default, env-tunable, never a hard wall — the truncation marker always says where the rest lives and how to window into it.
5. **Inline images:** auto-detect iTerm2/kitty graphics protocols and show renders inline; otherwise print local paths. A5 scope.
6. **`fetch`:** auto-registers pulled files as platform artifacts unless `--no-capture` is set.
