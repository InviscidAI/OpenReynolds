# OpenReynolds

A UI-less CFD agent. It is an Anthropic tool-use loop with seven tools pointed at a real
Linux workspace that has OpenFOAM in it. The agent decides how to work; this repository is
the plumbing that lets it.

```bash
pip install -e .
openreynolds config          # service URL + keys, stored outside the repo
openreynolds                 # start a study
```

## What it is

The design is in [`plan-2-agent.md`](plan-2-agent.md). The short version is a contract
about who decides what:

**The harness may** cap tool output (always saying where the rest lives), sync a toolbox
into the workspace, poll running jobs in plain Python and wake the model with *facts*, and
capture the transcript to a platform.

**The harness may not** enforce an ordering of actions, block or rewrite a tool call on
policy grounds, require an approval before anything runs, inject checklists or mandated
workflows, or grade the model's output. There is no gate DAG, no state machine, no
watchdog with authority, and no budget the model has to reason about.

That is not a style preference — it is the thing being tested. `tests/test_prompt.py`
fails the build if imperative workflow language appears in the system prompt.

## The tools

| Tool | What it does |
|---|---|
| `bash` | Run a command in the workspace and wait. Capped at 300 s. |
| `write_file` / `read_file` | Write, and read a byte window — so a multi-gigabyte file is the model's problem to approach however it likes, not something a digest layer decides for it. |
| `job_start` | Detach a long command. It outlives the turn, and the session. |
| `job_check` | Status plus the log since an offset, in one call. |
| `job_kill` | Stop one. |
| `fetch` | Copy files out to the user's machine. |

Meshing, checking, rendering and post-processing are all `bash`. There is no `run_gate`,
no `amend_spec`, and no `ask_user` — asking is just talking.

## Long runs

The model starts a job and ends its turn. The CLI polls in plain code and wakes it when
something happens: the job exited, a `kill_on` regex matched, or the user typed. Wake
messages carry a name, an exit code, an end reason and a tail of log — never a suggestion
about what to do next.

Closing the laptop is fine; jobs live on the instance. `openreynolds --study <id>` picks it
back up. A resumed session is a fresh thread: the frozen system prompt, a factual blurb
(which instance, which study, which jobs are still running), and the user's message. The
model reorients itself from `/work`. When a live session approaches the context window, the
same move happens mid-session — the model is told the thread is being refreshed so it can
put anything it wants to keep on disk first.

Its memory is the filesystem plus whatever notes it chose to write. Nothing summarizes its
reasoning on its behalf.

## The toolbox

`openreynolds/toolbox/` is synced to `/work/.toolbox/` at session start. It is offered,
never imposed — the model can use, edit, replace or ignore any of it.

| Script | What it produces |
|---|---|
| `log_digest.py` | Residual plot, last-iteration table, continuity and bounding summary |
| `mesh_digest.py` | `checkMesh` output as a metric table — numbers, no verdicts |
| `render.py` | Fixed pyvista scenes: mesh cuts and field slices, as PNGs |
| `cells_estimate.py` | Cell-count prediction from the STL and dicts, before a snappy build |

`notes/openfoam-field-notes.md` is distilled OpenFOAM practice written as field notes
rather than procedure — the cost asymmetry, failure signatures, the traps that produce a
confident wrong number, restart-from-`latestTime`. The full architecture document it was
distilled from sits beside it as optional reading.

## Configuration

Environment first, then `~/.config/openreynolds/config.json` (`%APPDATA%` on Windows),
written with restrictive permissions.

| Setting | Meaning |
|---|---|
| `FOAMD_URL`, `FOAMD_API_KEY` | The workspace service |
| `ANTHROPIC_API_KEY` | Bring your own key — the service is BYOK and does not proxy LLM calls |
| `OPENREYNOLDS_MODEL` | Default `claude-opus-5` |
| `OPENREYNOLDS_MAX_TOOL_OUTPUT` | Inline tool-output budget, default 48000 bytes |

The Anthropic client is built from `base_url` plus a key, so if the service ever grows an
LLM proxy, pointing at it is a config change rather than a code change.

## Layout

```
openreynolds/
  cli.py       loop.py      tools.py     prompt.py
  watch.py     store.py     capture.py   config.py
  backend/
    base.py    # the protocol — no transport, no vendor
    hosted.py  # the only module that knows the service contract
    local.py   # deferred
  toolbox/
```

`tests/test_negative_obligation.py` enforces the one obligation v1 carries toward a future
local backend, and it is negative: nothing above `backend/base.py` may name a transport, a
URL, or a hosting service.

## Tests

```bash
python -m pytest
```

The suite runs entirely against an in-memory workspace — no network, no credentials.
It covers the parts that are easy to get quietly wrong: that the truncation marker
points at the *tail* of a log whose head was kept, that a wake message carries an end
reason and no advice, that job records survive a restart, that capture never blocks or
raises, that a `503` cold start is retried while a `404` is not, and that the toolbox
parsers still read a real `checkMesh` table. CI runs it on 3.10 and 3.12 and checks the
toolbox actually ships inside the wheel.

### Against a live service

```bash
FOAMD_URL=... FOAMD_API_KEY=... python scripts/smoke.py
```

Drives the real `Backend` with no model in the loop, so it checks this client against
the actual contract rather than a fake of it: command execution with the OpenFOAM
environment sourced, truncation and its pointer, windowed reads, toolbox sync, a job
streaming its log, a `kill_on` regex firing and reporting the line that matched, a
headless matplotlib render fetched back to disk. It reuses a workspace and never
deletes one, so the persistent volume is safe.

## Inline images

On iTerm2, kitty or WezTerm, a fetched PNG is drawn in the terminal as well as saved.
Anywhere else you get the path, which is what the model prints anyway. Set
`OPENREYNOLDS_INLINE_IMAGES=off` to disable.

## License

MIT. See [LICENSE](LICENSE).
