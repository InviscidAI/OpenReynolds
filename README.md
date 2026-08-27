# OpenReynolds

A UI-less CFD agent. It is a tool-use loop with seven tools pointed at a real
Linux workspace that has OpenFOAM in it. The agent decides how to work; this repository is
the plumbing that lets it.

```bash
pip install openreynolds     # or: uvx openreynolds · pipx install openreynolds · npm i -g openreynolds
openreynolds login           # approve this terminal in a browser; the service key is saved
openreynolds config          # which model, and its key -- bring your own, any vendor
openreynolds doctor          # check it can all be reached, before spending anything
openreynolds                 # start a study
```

`login` asks for your email and password -- the same account as app.tryreynolds.com,
and it offers to create one if there is none -- and hands this machine its own service
key, stored outside the repository. `--browser` approves a short code in a browser
instead, for a terminal with no keyboard of its own. `--service` points at another
deployment. Everything the agent does on the hosted workspace is billed to that key;
everything the model does is billed to yours.

By default the transcript of every study is uploaded to the workspace service as it
runs, so a study is kept somewhere other than one laptop; `--no-capture` for a session,
or `OPENREYNOLDS_CAPTURE=0` in the environment, keeps it on this machine only.

## The interface

```
 study 20260823-213712-babc   instance 974f4406   model claude-opus-5
 412,355 tokens  (41% of the window)
 ██████████░░░░░░░░░░░░░░  41%  solving miter_medium · Time 820 / 2000 s · 14 min · ~19 min left
                                Ux 3.1e-04  Uy 2.0e-04  p 8.7e-03 · Co max 0.50 · continuity 2.1e-06
 thinking 12s: the residual plateau at 1e-4 is the mesh, not the solver
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

Above all of that sits a three-line status stack, each line earning its place only when
it has something to say:

- **The bar** appears when real compute is running — a solve, a mesh, a decomposition,
  a sync — and takes no room otherwise (a pulsing strip that only says "busy" is worse
  than nothing). A solve shows its solver time against the case's `endTime`, a
  percentage, how long it has run, a `~` estimate of what is left, and the last
  residuals; a `snappyHexMesh` shows which phase and iteration it is in; a `bash`
  command that redirects to a log is read while it runs.
- **The now line** is one plain-language sentence on what the agent is doing *right
  now* — "reworking the near-wake mesh before spending solver time" — written by the
  front desk (below) from the transcript and the live facts, not a mechanical phase
  label.
- **The stage line** is the model's own reasoning, one summary line at a time rather
  than hundreds of grey lines; ctrl+T puts it back in full.

`/status` reads from the same picture as the bar, so the two cannot disagree.

**The front desk.** The agent runs one thing at a time on one thread: a `bash` call can
hold it for minutes, and until the current step returns nothing on that thread can read
what you typed, let alone answer. So a second, cheap agent (Claude Haiku, same BYOK key)
runs on its own thread, read-only. When you type while the agent is mid-turn, the desk
answers within seconds from what it can see — the transcript and the live job facts —
shown as `desk` so it is never mistaken for the agent itself; your message still reaches
the agent unchanged and it replies in its own time. The desk also writes the now line
above. It cannot act, steer, or speak for the agent — it is a narrator with a phone, and
it says so when a real answer needs the agent. `OPENREYNOLDS_DESK=0` turns it off;
`OPENREYNOLDS_DESK_MODEL` picks the model.

`--plain` gives the old streaming terminal, and `-p` still implies it. If the interface
cannot start — no terminal, missing library — the session falls back rather than
failing: an interface problem should cost the look of the thing, not the work.

Presentation is a seam (`openreynolds/view.py`). Both interfaces implement the same
`View`, and the loop cannot tell which one it has, so **adding an interface can change
what the user reads and never what the model does.**

`doctor` checks each piece separately and says which one is wrong: whether the settings
are present, whether the workspace service answers and what instances you have, whether
the model API accepts your key and knows the model id (one free token count, nothing
generated), whether transcripts are still reaching the platform, whether the toolbox is
where it will be synced from, and whether this terminal can draw a render inline.

Capture is the one worth having in there: it is on by default and fails quietly by
design, because it must never delay or break a study. The cost of that is there being
no moment at which anyone finds out it stopped. This is that moment.

## What it is

The design is in [`docs/design.md`](docs/design.md). The short version is a contract
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

## What the instance costs you

The instance starts when you run `openreynolds` and stops when the session ends --
`/exit`, end of input, ctrl+C, or a `-p` run finishing. Jobs are stopped first, the
study is mirrored down first of all, and stopping an instance leaves its volume
untouched, so nothing is lost by it.

A standing note travels with you. If `preferences.md` exists next to `config.json`
(the path `openreynolds config` prints), its contents are relayed at the start of
every session's briefing, verbatim, in your own words — the place for durable
preferences like "when meshing, render the geometry and the mesh and look at the
images before running anything expensive." The harness adds nothing to it and
enforces nothing about it: it is you talking, and what to do about it stays the
agent's call.

The study also comes home *while* it runs. A background mirror syncs the study's
directory on the instance down to `./studies/<id>/files/` on a short cycle — meshes,
renders, postProcessing data, logs, everything under the caps — for the whole
session, including the hours a solve spends writing while no turn is in flight. The
files pane draws from the same sync, so it shows the workspace as of the last cycle
and refreshes itself; opening a file that is already here costs no round trip, and
`/status` says when the last sync ran. `OPENREYNOLDS_MIRROR_INTERVAL_S` tunes the
cadence; `0` turns the background sync off (turn-end syncs still run).

A long solve is not a silent one. While jobs run, the bar shows where each one is —
solver time against `endTime`, residuals, elapsed and estimated remaining, read
from the log every twenty seconds on the bar's own thread — and, by default once a
minute, the agent itself is woken with the same facts (never the estimate) so it
can say where things stand (or say nothing; that stays its call). `OPENREYNOLDS_NARRATE_EVERY_S` adjusts the cadence; `0` turns
narration off. Each narration is a model turn and is priced like one. The agent
also has a quieter way to pace itself than `sleep` in `bash`: `job_check` takes
`wait_s` and holds its answer until the job ends or the user speaks.

Video assembly is local by design. Stills render on the instance, next to the
data — moving gigabytes of fields to make a 100 KB PNG would be the wrong
direction — but a video needs only the frames, which are already mirrored home,
and the instance image deliberately ships no encoder. `openreynolds video
[frames-dir]` builds an `.mp4` from a mirrored frame directory using the ffmpeg
this machine has (imageio as the fallback); with no argument it picks the
study's biggest frame set, and `openreynolds doctor` says which encoder it
would use. Purely local — no instance starts and no token is spent.

`--keep-alive` leaves it up, which is the right choice when you are stepping away from
a long solve and mean to come back. It says what it is choosing when you use it.

Everything else -- `files`, `pull`, `stop` -- borrows the instance to answer a question
and puts it back down afterwards. If it was already running, it belongs to whoever
started it and is left alone. `doctor` never starts one at all.

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

### Running unattended

`openreynolds -p "..."` runs one study with nobody at the keyboard: the plain terminal,
the model's turn, then the wait on whatever jobs it started — for hours, if that is how
long the solve takes. `--max-wait <minutes>` bounds the wait; the job carries on out on
the instance either way, and `--study <id>` picks it up. The exit code says how it went,
because a script has nothing else to go on:

| Code | Meaning |
|---|---|
| `0` | the model finished and no jobs remain |
| `1` | the model API would not complete a turn — a rate limit, a bad key, a dropped connection; the study is intact and resumes with `--study` |
| `2` | `--max-wait` ran out with a job still running |

Interactive sessions exit `0`; whatever happened in them was said on screen.

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
| `OPENREYNOLDS_PROVIDER` | Whose model: `anthropic` (default), `openai`, `zai`, `deepseek`, `moonshot`, `minimax`, `openrouter`, `ollama` — or a bare API family with `OPENREYNOLDS_LLM_BASE_URL` |
| `OPENREYNOLDS_LLM_API_KEY` | Bring your own key — the service never proxies a model call. The vendor's own name works too: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `ZAI_API_KEY`, … |
| `OPENREYNOLDS_LLM_BASE_URL` | Any endpoint that speaks the Messages API or Chat Completions; a gateway, a local server |
| `OPENREYNOLDS_MODEL` | Default from the provider preset (`claude-opus-5` for Anthropic) |
| `OPENREYNOLDS_CONTEXT_WINDOW` | Tokens the model holds in one thread; the preset's number unless set |
| `OPENREYNOLDS_MAX_TOOL_OUTPUT` | Inline tool-output budget, default 48000 bytes |
| `OPENREYNOLDS_CAPTURE` | `0`, `false`, `no` or `off` keeps transcripts on this machine only; the per-session form is `--no-capture` |

**Bring your own model.** The loop is written against two API dialects, not one
vendor: the Messages API (Anthropic, and the compatible endpoints Z.ai, DeepSeek,
Moonshot and MiniMax expose) and Chat Completions (OpenAI, OpenRouter, Ollama, vLLM,
most gateways). `openreynolds config --provider zai` points the settings at a vendor —
endpoint, model ids, context window — and asks only for the key; anything not in the
preset table is a family plus `OPENREYNOLDS_LLM_BASE_URL`. Claude-only request fields
(adaptive thinking, effort, prompt caching) are dropped for the session the first time
an endpoint refuses them, so a compatible vendor costs one failed request, not a flag.
A vendor that streams its reasoning shows it in the stage line like Claude's; one that
does not simply thinks in silence. The front desk uses the same provider and its own
cheaper model from the preset. Model ids in the presets are the ones that existed when
the table was written; `doctor` says whether yours still answers.

## Layout

```
openreynolds/
  cli.py       loop.py      tools.py     prompt.py
  watch.py     store.py     capture.py   config.py
  view.py      tui.py       images.py    stopping.py
  browse.py    # read-only window onto the workspace
  mirror.py    # the local copy of the study, kept without being asked
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

The repository carries a pre-commit hook that runs the suite and refuses a red
commit. Turn it on once per clone:

```bash
git config core.hooksPath .githooks
```

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
way, and none of them by the unit suite. Most were invisible until something above
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

Each study works in `/work/<study-id>`, made when the session starts. A new project
starts in an empty directory -- there is nothing to clear and no flag to remember -- and
commands run there unless the model says otherwise. The volume still persists, so the
rest of it holds other studies' work: readable if wanted, and written for somebody
else's question.

The agent decides what it copies out. You should not have to negotiate with it to find
out what is there, so the workspace is readable directly, and none of it involves the
model or costs a token:

```
openreynolds studies                     # every study on this machine
openreynolds files                       # your study's own directory
openreynolds files /work                 # or the whole volume
openreynolds files --cat /work/case/log.simpleFoam
openreynolds files --pull /work/case/renders   # copy it to this machine
openreynolds files --open                # the study folder in your file browser
openreynolds renders                     # every picture this study made, newest first
openreynolds renders --open              # and open the folder
```

In the interface the **files** tab is the same thing live: ctrl+F to open it, ctrl+R to
refresh, enter on a file to read it, ctrl+P to copy it out.

### The pictures come to you

A render is the point of the run, and for five sessions the agent made thirty of them
and the answer to "where is the image?" was still "buried three directories down, if
you go looking." So delivery stopped being the agent's job. As the mirror brings files
home, every render is copied into one flat `studies/<id>/renders/` folder and a
directory of animation frames is assembled into a gif **on this machine** — the
instance renders the frames next to the data, the encoder lives here. New pictures are
announced as they land, the **renders** tab (ctrl+G, or `/renders`) always holds the
newest, and `openreynolds renders` shows them after the fact. None of it asks the agent
to run `fetch`; the agent produces, the harness delivers.

### It comes down on its own

Reading the workspace over the wire is not the same as having the study. For a while
that was all there was: a run that took half an hour left two files on the machine that
commissioned it, its transcript and its session record, and everything it had actually
made stayed out on the volume. So the session mirrors `/work/<study-id>` into
`studies/<id>/files/` at the end of every turn and once more on the way out.

Selectively, because a case is not a document. One can be tens of gigabytes of
reconstructed time directories, `processor*/` decompositions and VTK, and a tool that
copied all of it by default would be a tool nobody could leave running. What comes down
is the part a person reads — images, reports, notes, logs, `postProcessing/`, and the
case *dictionaries* under `system/`, `constant/` and `0/`, which are the setup itself
and weigh kilobytes. What stays: `processor*/`, every written time directory after the
initial one, `polyMesh/`, VTK, `__pycache__`, anything over 25 MB, and anything past a
200 MB budget for one sync.

Every file left behind is named, with the reason it was left, because a silent filter
and an empty workspace look identical from this end — and telling those two apart is
the entire reason this exists.

```bash
openreynolds pull                            # again, now
openreynolds pull /work/<study-id>/case      # or one directory of it
openreynolds push ./mycase                   # the other direction: local -> instance
openreynolds push geom.stl --to /work/<id>/constant/triSurface
openreynolds pull --all                      # when that judgement is wrong for this study
```

Only what changed is asked for — remote size and mtime against what is already here —
so once a study settles the per-turn sync costs one directory listing and nothing else.
A failure warns and the session carries on; a mirror that ends a study would be worse
than one that misses a file.

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

## Security

The agent runs an unsandboxed shell on your hosted instance with no approval gate; that
is the design, not an oversight. [SECURITY.md](SECURITY.md) says what follows from it —
what the model can see on the instance, what leaves your machine and when, and where to
report a problem (security@inviscidai.com).

## License

MIT. See [LICENSE](LICENSE).
