# OpenReynolds

A UI-less CFD agent. It is an Anthropic tool-use loop with seven tools pointed at a real
Linux workspace that has OpenFOAM in it. The agent decides how to work; this repository is
the plumbing that lets it.

```bash
pip install -e .
openreynolds config          # service URL + keys, stored outside the repo
openreynolds doctor          # check it can all be reached, before spending anything
openreynolds                 # start a study
```

## The interface

```
 study 20260823-213712-babc   instance 974f4406   model claude-opus-5
 412,355 tokens  (41% of the window)
+------------------------------------------------------+ running
| I'll build a mitred 90 degree elbow, 100mm square.    |   * miter_medium
| Starting with the coarse rung so the ladder is cheap. |
|                                                       | jobs keep running
+------------------------------------------------------+ if you leave
+------------------------------------------------------+
| job_start   ./run_case.sh miter_medium                |
| watching 1 job(s) - type to interrupt                 |
+------------------------------------------------------+
  Ask for something, or say what looks wrong...
```

A session has four things worth watching at once, and a single scrolling log shows one
of them while burying the rest. So what the agent *says* gets the main pane, what it
*does* to the workspace goes to its own activity pane, what is still **running out on
the instance** — the part that outlives the session — gets a panel that says so, and
what is actually **in the workspace** gets a file tree you can open things from. The bar
keeps the study, the instance, the model, and how full the thread is, because context
filling up is the thing that changes what happens next.

Reasoning is one summary line on the stage indicator rather than hundreds of grey lines
in the transcript; ctrl+T puts it back in full.

`--plain` gives the old streaming terminal, and `-p` still implies it. If the interface
cannot start — no terminal, missing library — the session falls back rather than
failing: an interface problem should cost the look of the thing, not the work.

Presentation is a seam (`openreynolds/view.py`). Both interfaces implement the same
`View`, and the loop cannot tell which one it has, so **adding an interface can change
what the user reads and never what the model does.**

`doctor` checks each piece separately and says which one is wrong: whether the settings
are present, whether the workspace service answers and what instances you have, whether
the model API accepts your key and knows the model id (one free token count, nothing
generated), whether the toolbox is where it will be synced from, and whether this
terminal can draw a render inline.

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

## Stopping

```bash
openreynolds stop --study <id>          # stop this study's jobs
openreynolds stop --study <id> --force  # and anything that outlived them
```

Jobs outliving the session is the design — closing the laptop on a long solve is the
point. Leaving *without being told* is not. On exit the session says what is still
running, that it is still costing, and how to stop it.

`stop` does not trust the acknowledgement. The service marks a job killed whether or not
the signal reached anything, and a solver launched through `mpirun` puts its ranks
outside the job's process group — so the wrapper dies, the record says killed, and eight
cores keep going. That happened here, and the first sign of it was the bill. So stopping
signals, looks, escalates to a signal that cannot be ignored, looks again, and reports
what is *still there* rather than what was requested.

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
| `geometry_view.py` | Surfaces drawn from four views with every facet edge, plus open edges, bodies and extents |
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
  view.py      tui.py       images.py    stopping.py
  browse.py    # read-only window onto the workspace
  commands.py  # /btw, /status - what the user types that is not a message
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

## Acceptance

[`docs/a4-acceptance.md`](docs/a4-acceptance.md) is the record of the A4 milestone: an
open-ended "simulate airflow through an L-shaped junction", run free-form against the
live service. The negative check passed twice — two harness-authored messages in 212, no
forced ordering, no refused tool call, no grading. The CFD came back as a time-averaged
loss coefficient over a limit cycle, with the agent stating before it was asked that it
had not converged. Seven harness defects surfaced there that no unit test had caught.

## Testing as a user

Everything above tests the agent from the inside, by someone who knows what a
`tool_result` is. `scripts/user_test.py` drives it from the outside instead:

```bash
python scripts/user_test.py --persona all --turns 8 --budget 12 --speak-after 150
```

A persona holds up the user end of a real interactive session. It cannot code: it never
types a command, never names a piece of software, never suggests a fix, and sees nothing
but what appears in the terminal. That constraint is enforced rather than requested — a
line that looks like a command is dropped before it is sent, because a persona that
quietly turns into an engineer stops testing what it claims to. `/status` and `/btw` are
allowed through, since typing those is using the product, not writing code.

Four of them, because one persona finds one class of problem:

| Persona | Who | What it catches |
|---|---|---|
| `engineer` | Good physical intuition, in a hurry | A wrong number, and a number given without a confidence statement |
| `controller` | Wants to be told before anything long starts | Whether `/status` works, and whether it can be stopped mid-flight |
| `shifting` | Requirements arrive late — hot air, rough duct | Whether it says, unprompted, that an earlier answer no longer holds |
| `novice` | No physical intuition at all | Whether it volunteers uncertainty when nobody is checking it |

Each starts on a clean workspace (earlier work is moved to `/work/.attic/<stamp>`, never
deleted), each has its own geometry so the second cannot answer from the first's files,
and whatever is still running gets stopped before the next one starts.

A run is bounded three ways — turns, seconds per reply, total minutes — because a test
nobody will sit through is a test nobody runs. It ends on its own with `[SATISFIED]` or
`[STUCK]`, and a turn that ends any other way is *labelled*: timed out, exited, said
nothing, went quiet without asking for input.

Two details that took a while to get right, and both matter more than they sound:

- **Waiting for the prompt, not for silence.** Silence is a guess, and it guesses wrong
  every time a command runs long: the reply gets cut mid-sentence and the next message
  lands while the agent is still working. Seeing the prompt means reading bytes rather
  than lines — it has no trailing newline, because nothing follows it until somebody
  answers.
- **Speaking over work in progress.** A turn can legitimately run ten minutes; the agent
  polls its own solve with a blocking sleep, so there is neither a prompt nor a pause. A
  real user would not sit through that, and anything they typed would be heard between
  tool calls. `--speak-after` is that user, and it is the only thing that exercises
  interrupting at all.

Every defect in [docs/found-by-using-it.md](docs/found-by-using-it.md) was found this
way, and none of them by the 392 unit tests. Most were invisible until something above
them was fixed first, which is the argument for driving the whole product from outside
rather than testing its parts.

## Seeing

Two different eyes, and they need separate plumbing.

**The agent's.** `read_file` on a `.png`, `.jpg`, `.gif` or `.webp` returns the picture
itself as an image block, not bytes. Without that, an agent can render a mesh and then
only ever read its own description of what it meant to draw — geometry with the units
wrong, a surface with a hole in it and a perfectly good one are indistinguishable from
a directory listing. Anything over 3.5 MB is reported as a size instead of vanishing.
Base64 never reaches `messages.jsonl`; the log keeps a description.

**Yours.** On iTerm2, kitty or WezTerm a fetched PNG is drawn in the terminal as well as
saved (`OPENREYNOLDS_INLINE_IMAGES=off` disables it). In the interface a full-screen app
cannot use those protocols, so opening an image in the files pane copies it to the study
directory and tells you where — the file browser can draw what the terminal cannot.

## Looking at the workspace

The agent decides what it copies out. You should not have to negotiate with it to find
out what is there, so the workspace is readable directly, and none of it involves the
model or costs a token:

```
openreynolds files                       # the tree under /work
openreynolds files /work/case            # a subtree
openreynolds files --cat /work/case/log.simpleFoam
openreynolds files --pull /work/case/renders   # copy it to this machine
openreynolds files --open                # the study folder in your file browser
```

In the interface the **files** tab is the same thing live: ctrl+F to open it, ctrl+R to
refresh, enter on a file to read it, ctrl+P to copy it out.

## Saying something without stopping the work

Anything typed while the agent is working used to sit unread until the whole turn
ended, so "just run the coarse one" arrived minutes late and read as a contradiction.
It now rides along with the next tool result and lands at the agent's very next step.

```
/btw <anything>   say it without asking the agent to stop
/btw   or /status what is happening right now - answered locally, no turn, no interruption
/files [path]     the workspace
/open             the study folder
/help             all of it
/exit             leave (jobs keep running on the instance)
```

`/status` is the one that matters most: a question that costs a turn and derails the
work is a question people stop asking, and then they have no idea what is going on. It
is answered from what the harness already knows — running jobs, current stage, thread
size, files pulled — and the model is never told it was asked.

Anything else beginning with `/` is treated as a message, because `/work/case/log looks
wrong` is a sentence, not a typo.

## License

MIT. See [LICENSE](LICENSE).
