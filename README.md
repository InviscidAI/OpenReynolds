# OpenReynolds

**A CFD agent with a real OpenFOAM workspace.**

OpenReynolds is a tool-use loop with nine tools pointed at a Linux machine that has
OpenFOAM v2512 on it. You describe the flow; it writes the case, meshes it, launches
the solver, reads the residuals while they come in, looks at its own renders, and
hands back the figures and the files that made them.

There is no pipeline underneath choosing the order of those things. The agent decides
how to work. This repository is the plumbing that lets it, and the plumbing is
deliberately not allowed to tell it what to do.

**[tryreynolds.com](https://tryreynolds.com)** &middot;
[Documentation](https://tryreynolds.com/docs) &middot;
[Worked studies](https://tryreynolds.com/studies) &middot;
[Hosted app](https://app.tryreynolds.com) &middot;
[PyPI](https://pypi.org/project/openreynolds/)

You do not need to know OpenFOAM to use it. You bring the engineering question: the
geometry, the fluid, the speed, and what you want measured. Choosing the solver,
writing the dictionaries, sizing the mesh and reading the residuals is the agent's job.

[![A study in progress: the conversation and the agent's tool calls on the left, the mesh it built on the right, and the solver's own residuals across the top.](https://tryreynolds.com/figures/ui-app-study.jpg?v=2)](https://tryreynolds.com)

<sub>A study in progress. What the agent says is on the left, what it did to the
workspace is underneath, and what it built is on the right. The bar across the top is
the solver's own numbers, read out of the log while the job runs.</sub>

## What comes out

Every one of these is from a real run, with its transcript published beside it at
[tryreynolds.com/studies](https://tryreynolds.com/studies).

| | |
|---|---|
| ![The lambda shock on the ONERA M6 wing, upper-surface pressure coefficient.](https://tryreynolds.com/figures/studies/m6-cp-upper-surface.png?v=2) | ![Vortex shedding past a cylinder at Re = 100.](https://tryreynolds.com/figures/studies/cyl-shedding.gif?v=2) |
| **[The lambda-shock, at one seventh the mesh](https://tryreynolds.com/studies/onera-m6-transonic)** <br> <sub>ONERA M6 at Mach 0.8395. Shock positions within 0.06 chord of the 1979 wind-tunnel data, on 1.79 million cells.</sub> | **[Four numbers, four published bands, one animation](https://tryreynolds.com/studies/vortex-shedding-cylinder)** <br> <sub>Vortex shedding at Re = 100, with the Strouhal number checked against the published band.</sub> |
| ![Conjugate heat transfer on a finned heat sink.](https://tryreynolds.com/figures/studies/hs-temp.png?v=2) | ![Mach number through a converging-diverging nozzle.](https://tryreynolds.com/figures/studies/nozzle-mach.png?v=2) |
| **[A heat sink, and an energy balance closed to 99.7%](https://tryreynolds.com/studies/finned-heat-sink)** <br> <sub>Conjugate heat transfer, with the energy balance closed as its own check.</sub> | **[The solver said it had converged. The agent checked.](https://tryreynolds.com/studies/converging-diverging-nozzle)** <br> <sub>A nozzle designed for Mach 2, run overexpanded, and the disagreement caught.</sub> |

```bash
pip install openreynolds     # or: uvx openreynolds · pipx install openreynolds · npm i -g openreynolds
openreynolds login           # email and password, or --browser for Google; the app.tryreynolds.com account
openreynolds config          # which model, and its key — bring your own, any vendor
openreynolds doctor          # check it can all be reached, before spending anything
openreynolds                 # start a study
```

`login` hands this machine its own service key, stored outside the repository, and
offers to create an account if there is none. `--browser` approves a short code in a
browser instead: that is the way in for a terminal with no keyboard of its own, and the
only way in for **an account created with Google**, which has no password to type. A
password sign-in for one of those fails exactly like a wrong password, and the service
cannot tell the two apart, so the terminal names the browser flow whenever a sign-in
fails rather than offering to create a second account for an address that already has
one.

**A company email address starts the account with $10 of credit, once**, covering the
hosted workspace and Reynolds' own metered model together; a personal address starts at
zero. Which addresses count is the service's rule and is never guessed at here, because
a domain list copied into a client goes stale and starts telling people the opposite of
what they are about to get.

**Run `doctor` first.** It checks settings, the workspace service, the model API,
capture, the toolbox, the terminal and the video encoder, and it writes nothing.
Almost every confusing first session is one of those seven being wrong, and `doctor`
says which in a sentence.

---

## What a session looks like

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
*does* goes to its own activity pane, what is still **running out on the instance** —
the part that outlives the session — gets a panel that says so, and what is **in the
workspace** gets a file tree you can open things from.

`--plain` gives a plain streaming terminal instead, which is what you want in CI or
over a poor connection.

## The nine tools

That is the whole surface. Anything the agent does to a case, it does through one of
these.

| Tool | What it does |
| --- | --- |
| `bash` | Run a command with the OpenFOAM environment already sourced. Capped in output and in time, so a runaway command cannot fill the context. |
| `write_file` | Write a file. Every `blockMeshDict`, `fvSchemes` and boundary condition arrives this way. |
| `read_file` | Read a file back. Reading a PNG is how the agent looks at its own geometry and mesh renders, which is why the model has to be one that can see. |
| `job_start` | Launch a detached job: a mesh, a solve, a post-process. It outlives the call that started it, so an hours-long run is not held open by a request. |
| `job_check` | Ask how a job is doing. It can hold the answer until the job ends, and returns early the moment you type. |
| `job_kill` | Stop a job, and confirm it actually stopped. |
| `fetch` | Copy files or directories out of the workspace onto your own machine, and say where they landed. Renders and reports come home this way. |
| `cad` | Describe a shape in words — or point at a `.step`/`.iges` file already on the workspace — and get back an OpenFOAM mesh of it there. A separate agent does the work on the same machine, one python cell at a time in a kernel: it builds or imports the shape, repairs and tags it, picks the mesher (`snappyHexMesh`, cfMesh, gmsh body-fitted, `blockMesh`), renders the mesh, measures it, and revises until `checkMesh` passes and the shape measures up to what was asked for. Saying it is done is not what ends it — the mesh has to be there, pass, carry patch names somebody chose, and have a script that rebuilds it. You get the picture, the patch table, `checkMesh`'s verdict, the script and where the case is. A mesh only: fields, boundary conditions and the solve stay with the agent. |
| `mesh_review` | Have an independent reviewer look at a mesh that already exists — one built with `bash`, one the `cad` tool returned, one you uploaded — from several views (2D: overview, cells, zoomed quadrants; 3D: isometric corners, orthographic faces, mid-plane cuts) and say whether it is the shape that was asked for. A fresh model that did not build it and has not seen the conversation, judging the pictures against the request: it fails only for what would change the CFD answer, does not run `checkMesh` and changes nothing. The `cad` desk consults the same reviewer at its own finish. |

A tenth, `checkpoint`, exists only when you choose structured mode (see *Modes*):
it puts a summary of where the study stands, and what comes next, in front of you and
waits for your answer. In full auto it is not in the tool list at all.

## The rule this repository keeps

The harness may cap tool output, keep the toolbox in sync, poll a running job in plain
code and wake the model with the facts, and capture a transcript.

It may **not** enforce an order of work, block or rewrite a tool call, require an
approval, inject a checklist or a workflow, or grade the output.

That holds in the default mode, full auto, exactly as written. The one exception is
yours to make: you can *choose* to be asked before compute is spent, or to run a study
in stages you approve (see *Modes*). The harness then gates exactly what you asked to
have gated and nothing else, for the same reason `/btw` exists: it is your say in how
you want to be heard, not the harness's opinion about how the model should work.

That is not a convention anyone has to remember. `tests/test_prompt.py` fails the build
if imperative language appears in the system prompt, and `tests/test_briefing.py`
applies the same rule to the briefing, because a harness that starts telling the model
how to do CFD caps the work at whatever its author knew about CFD.

What the model gets instead is **your standing preferences**, relayed verbatim in your
own voice from `preferences.md` beside your config. Say you want the mesh rendered and
checked before any solver time is spent, and that is what the agent is told you want.
What it does about it is its call.

## Modes

How much the agent does before it asks you. The mode never changes what the agent is
told about CFD; it decides only whether compute waits for you.

| Mode | Label | What happens |
| --- | --- | --- |
| `auto` | Full auto | The default. Nothing is held and nothing waits for you. The briefing is the same bytes it was before modes existed. |
| `partial` | Ask before compute | Every `job_start` call is put to you before it runs, since that is what spends compute. `bash`, `read_file`, `write_file`, `fetch`, `job_check`, `job_kill` and `cad` run freely. |
| `structured` | Structured | The agent gets a `checkpoint` tool that shows you a summary and what comes next and waits for your answer. `job_start` is held until you have approved a checkpoint, the plan. The stages are the guided pipeline's own: geometry, preview, mesh, checkMesh, probe, solve, reconstruct, render, animate, report. |

```bash
openreynolds --mode partial              # or OPENREYNOLDS_MODE=partial, or "mode" in the config file
```

Answer a question with `/yes`, `/no <reason>` or `/all`. Typing `y` or `yes` approves;
any other words decline and go back to the agent as your reason, verbatim. `/all`
approves and switches the session to full auto for the rest of it. `/mode <name>`
switches mid-session from the next tool call, and the agent is told. A resumed study
keeps its mode unless you give one. `-p` with a mode other than `auto` is refused with
exit code `2`, because nobody is there to answer.
[docs/modes.md](docs/modes.md) has the whole of it.

## In a session: commands, help and completion

Anything you type reaches the model at its next step, so you can steer a run without
stopping it. These are answered by the harness instead:

| Command | What it does |
| --- | --- |
| `/btw <something>` | Say it without asking the agent to stop. |
| `/status` | What is happening right now, with no model turn. |
| `/files [path]`, `/renders`, `/open` | Look at the workspace, the pictures, or the local folder. |
| `/mode [name]` | Show or switch the mode. |
| `/model [model]` | Show the provider, model and effort, or switch model. |
| `/effort [level]` | `low`, `medium` or `high`, from the next request. |
| `/yes`, `/no [reason]`, `/all` | Answer an open question. |
| `/help [topic]` | Everything here; `/help commands`, `modes`, `model`, `tools` or `keys` goes deeper. |
| `/exit` | Leave. Jobs keep running. |

In the interface, typing `/` opens a list of matching commands above the prompt, with
the best match as grey text. Up and Down move, Tab takes the suggestion, Right takes the
grey text, Esc closes the list. After `/mode `, `/model `, `/effort ` or `/help ` the
list offers that command's choices. The hosted app's composer does the same from the
same registry. [docs/session-commands.md](docs/session-commands.md) has every alias and
key.

## Changing the model mid-study

```
/model claude-opus-5            another model, same provider
/model anthropic:claude-opus-5  another provider, if its key is in the environment
/effort medium
```

A new model is probed, with an image, before it is accepted, so a typo or a text-only
model is refused when you type it. It is applied between turns: typed while the agent
is idle it answers your next message, typed mid-turn it takes over when the turn ends.
Earlier thinking blocks are dropped from the thread, a thread too large for the new
model's window (the one known for that model, otherwise the provider's) is refreshed
first, and the prompt cache is written once more, so the first
request after a switch costs more. Effort is read on every request and applies at once.

A switch outlives the session: `session.json` records the model, the provider that
served it and the endpoint it was served from, and `--study <id>` carries on on that
record, so a study moved to Opus for the hard part is still on Opus tomorrow. `--model`
or `OPENREYNOLDS_MODEL` still wins. Here the record is restored whenever the provider
that served it can be reached from this machine, switching provider if that is what the
record says and bringing that provider's key, endpoint, window and desk model with it;
on the provider this machine is already configured for, this run must also point at the
endpoint the model was served from, because a vendor's own key and a gateway in front
of it answer to different model ids. When the record cannot be honoured the session
says so in a line and runs on the configured model, and the study keeps what it
recorded, so a resume where the key is set carries on there.

Two limits. The record is the study's own `session.json` on this machine, so a study
resumed where it has never run (opened in the browser, or on another laptop) starts on
the configured model; nothing is fetched to stand in for it. And naming a provider is
not naming a model: only `--model` and `OPENREYNOLDS_MODEL` count as asking for one, so
`OPENREYNOLDS_PROVIDER` on its own does not hold a resume to that provider. A study
recorded before the provider was kept names a model and no provider, and starts on the
configured model as it always did: an id does not name a provider on its own, and
`claude-opus-5` is valid on `anthropic` and on `reynolds`.
[docs/switching-models.md](docs/switching-models.md) has the detail.

## Bring your own model

The agent calls your provider directly with your key; the workspace service never sees
it. Each preset carries an endpoint, a default model and a context window, so
connecting is a key and nothing else. Anything speaking the Messages API or Chat
Completions works even without a preset.

| Preset | Default model |
| --- | --- |
| `reynolds` | `claude-sonnet-5`, metered through the workspace service, no key of your own |
| `anthropic` | `claude-opus-5` |
| `openai` | `gpt-5` |
| `zai` | `glm-4.6` |
| `deepseek` | `deepseek-chat` |
| `moonshot` | `kimi-k2-thinking` |
| `minimax` | `MiniMax-M2` |
| `openrouter` | `anthropic/claude-sonnet-4.5` |
| `ollama` | `qwen3`, on your own machine |

```bash
openreynolds config --provider anthropic    # a preset, then paste the key
openreynolds config --provider ollama       # local, no key
```

**The model has to be able to see.** The agent renders its geometry and its mesh and
looks at the images before spending solver time on them, so a text-only model works
blind. A key is refused at the point it is connected rather than halfway through a
paid study, and `doctor` makes the same check with a one-pixel image.

## What it costs

Two meters run at once: the model, charged per token by your provider, and the hosted
instance, charged per second it is up. Three things move the bill, and each was found
the expensive way:

- **Reasoning effort.** The one knob that changes cost without changing what the agent
  may do.
- **How often a long run wakes the model.** Every wake re-sends the context, so a
  chatty harness on a two-hour solve pays for the whole thread repeatedly.
  `OPENREYNOLDS_NARRATE_EVERY_S` is the control, and its default is generous.
- **Cores.** Billed whether the solver scales to them or not. A small 2D case on four
  cores can be slower and dearer than the same case on one.

An instance left running is an instance being billed. It stops on its own when idle,
and `openreynolds stop` ends it now, verifying the jobs are actually dead rather than
assuming.

## The workspace

The durable thing is the volume, not the container: an instance can be stopped and
started again with the case still on it. Stopping keeps the data.

A background mirror copies the study's directory down to your machine while the session
runs, and fires immediately whenever the agent looks at an image, so a render you are
about to be shown is already local. Remote deletions are not propagated home, which is
deliberate: a local copy of something the agent cleaned up is a recovered file, not a
bug. A session ending on a workspace the service has already stopped syncs from the
service's own copy of it rather than starting a machine to be listed.

Rendering happens on the instance, beside the data, because moving gigabytes home to
make a hundred-kilobyte picture is the wrong way round. Frames come home and
`openreynolds video` assembles them here, where a real encoder lives. The instance
image has no encoder on purpose.

## Editing the mesh by hand — the hosted app only

Asking in words is right for "make the duct wider" and wrong for "that face is the
inlet": the second is quicker to point at than to describe, and a description is exactly
what the agent can misread. So the mesh tab in the
[hosted app](https://app.tryreynolds.com) is an editor. Click the mesh and the smooth
surface under the pointer is picked, or one face, a patch, or a box; name it Inlet,
Outlet, Wall or Symmetry, with a flow direction and a speed per inlet. Sketch a 2D domain
in metres with bodies cut out and name its edges by clicking, and gmsh meshes it on the
instance with one curve per drawn edge, so a name lands on exactly the faces swept from
it. Move, scale and turn a mesh with the result shown before it applies. Every edit runs
`createPatch` and `checkMesh` on the same instance the session is using, keeps the mesh
it replaced so it can be undone, and refuses a page whose mesh changed underneath rather
than editing the wrong faces; handing it back tells the session what the patches now are.
**None of this is in this package.** It needs a pointer and a 3D view, so it lives in the
web app, and it is mentioned here because it changes the same workspace a CLI session is
working in.

## The toolbox

Small scripts kept on the instance and **offered, never imposed**: geometry and mesh
rendering, a mesh digest, a solver log digest, a cell-count estimate, animation frames,
a first look at a finished case, a preflight check. Beside them in
[`openreynolds/toolbox/notes/`](openreynolds/toolbox/notes/) are field notes on things
that have gone wrong before, such as deciding between a steady and a transient run, or
the recipe for a genuinely 2D case and how to verify it worked.

The agent reaches for these or does not. They exist because a hint that is available
costs nothing and a rule that is enforced costs everything.

## Benchmarks

[`benchmarks/moving_mesh/`](benchmarks/moving_mesh/) holds the two cases the moving-mesh
field note rests on, as the verbatim prompts they were run from and a grader: a
`cyclicAMI` sliding interface cut through a circular-Couette annulus, where the torque is
closed form, and a cylinder on a spring free to move across the stream at Re = 100, where
no published amplitude for exactly that case has been found.

```bash
openreynolds -p "$(cat benchmarks/moving_mesh/couette_ami.txt)"
openreynolds pull --study <study-id>
python benchmarks/moving_mesh/grade.py couette <case> --control <the case with no interface>
python benchmarks/moving_mesh/grade.py viv <fixed case> <released case> [<continuation> ...]
```

The grader reads only what the solver wrote — `postProcessing/*/*.dat`, `log.*`,
`constant/` and `system/` — and never the run's own analysis, because the first time
these were graded by hand four of the runs' own summary numbers disagreed with their raw
files. Every physical constant comes off a case file, and a missing one stops the grade
and names the flag that supplies it rather than being assumed. `--json` for a
machine-readable verdict; exit `0` when every check passes, `1` when one fails, `2` when
the case could not be read.

Two of its checks are findings rather than formalities. The sliding interface held its
**area** to 43 ppm of unity while the two wall torques, which conservation makes equal
and opposite exactly, disagreed by **0.91%** — against 0.011% on the same annulus meshed
conformally — so the check every rotating-zone case runs passed one whose torque was a
percent wrong. And the freely moving cylinder settled at A/D = 0.640, stationary over 15
cycles and spectrally clean, which an energy audit rejected: with zero structural damping
the fluid was doing −10.4% of work per cycle, and tight coupling brought the amplitude
down to **0.567**.

## Commands

| Command | What it does |
| --- | --- |
| `openreynolds` | Start a study. `--study <id>` resumes one, `--instance <id>` attaches to a particular workspace. |
| `-p "..."` | Run non-interactively and exit. Exit code `0` done, `1` the model API failed or the session crashed, `2` hit `--max-wait` with work still running. With a mode other than `auto` it is refused as a usage error, also exit `2`. |
| `--mode auto\|partial\|structured` | How much the agent does before asking you. See *Modes*. |
| `--model <id>` / `--effort low\|medium\|high` | The model and reasoning effort for this session; the model is recorded on the study, so a resume carries on on it unless one is named again. `/model` and `/effort` change either mid-study. |
| `--output-format stream-json` | One JSON object per line on stdout and nothing else. See *Driving it from a program*. In front of `studies` or `doctor` it means their `--json`; in front of any other subcommand it is refused rather than ignored. |
| `openreynolds login` | Sign in; this machine gets its own service key. `--browser` approves a short code in a browser, which is the device-code flow and the only way in for a Google account. |
| `openreynolds config` | Provider, key, model, context window. `--key-file` and `--from-env` keep keys out of shell history. |
| `openreynolds doctor` | Check all seven surfaces. Read-only. `--json` answers in one object. |
| `openreynolds studies` | List the studies on this machine. `--json` answers in one object. |
| `openreynolds files` | What is in the workspace, and what has been copied down. |
| `openreynolds pull` | Bring this study's files down to this machine. |
| `openreynolds push` | Send a local file (a geometry, say) up to the instance. |
| `openreynolds renders` | Every picture and assembled animation. |
| `openreynolds video` | Assemble mirrored frames into a video, here. |
| `openreynolds stop` | Stop this study's jobs and confirm they stopped. `--force` skips the prompt. |

In a session, `/status` answers locally with no model turn, `/btw` says something
without interrupting the work, `/mode`, `/model` and `/effort` change how the session
runs, and anything else you type reaches the model at its next step, so you can steer a
run without stopping it. `/help` has the rest; see *In a session: commands, help and
completion*.

## Driving it from a program

Two questions come before a session, and both answer as data:

    openreynolds doctor --json     # {"ok": …, "failed": […], "checks": [{"check", "ok", "detail"}, …]}
    openreynolds studies --json    # {"dir": …, "studies": [{"study_id", "title", "instance_id", "model", …}, …]}

Straight to stdout, one object, no markup, and `doctor`'s exit code is unchanged for a
caller that only reads that. These two existed only as styled terminal lines, so the
`study_id` that `--study` takes had to be recovered by parsing a coloured row.
`--output-format stream-json` in front of either means the same thing as its own
`--json`; in front of any other subcommand it is refused rather than ignored.

`--output-format stream-json` puts one JSON object per line on stdout and **nothing
else**: every notice, warning and error the terminal would have shown goes to stderr,
and inline images are not drawn at all in this mode -- not onto a pipe and not onto a
pseudo-terminal, which is how an agent harness usually runs a child process. Each
object carries `v` (the schema), `type`, `at` (seconds since the session started) and
`study`, and `cost` rows are measured from the same origin as everything else.

    openreynolds -p "mesh and solve the elbow" --output-format stream-json

The first object is always `session_start`, with `study_id`, `instance_id`, `model` and
the local study directory -- `study_id` is what `--study <id>` takes to resume. The
stream always ends with exactly one `session_end`, carrying the same outcome the exit
code means: `ok`, `failed`, `timeout`, or `crashed` for an exception that escaped the
session. A failure before the session could start is that one object and nothing else,
with an `error` and an outcome of `config` (something is missing from the
configuration) or `unreachable` (the workspace service could not be reached) -- the
three cases exit code `1` alone cannot tell apart. In between:

| `type` | What it says |
| --- | --- |
| `workspace` | The study's own directory on the instance. The second object of every session. |
| `workspace_ready` | The workspace is up and set up, so tool calls run at once from here: `instance_id` and `seconds` since the session began. `session_start` names the instance while it may still be coming up; until this arrives a tool call waits for it rather than failing, and the model has been told so. |
| `thinking_begin` | The model started thinking; `thinking` carries what it thought. |
| `text` / `thinking` | Model output as it arrives, coalesced to a line rather than a token. |
| `message` | The whole assistant message once the turn ends, `text` and `thinking` in full. |
| `tool` / `tool_error` | A tool call, and a tool call that went wrong. |
| `step` | One round of think-then-act finished: which round, how long, how many calls. |
| `jobs` | Every job and its state, whenever any of it changes. |
| `progress` | What is running and how far along, when the picture changes. |
| `stage` / `narration` / `desk` / `status` | What is happening now, in words. |
| `mirrored` / `delivered` / `files` / `renders` | Files coming home, and what is in the workspace. |
| `notice` / `warn` / `info` / `usage` / `watching` / `interjection` / `prompt` | The rest of the terminal's own reporting. |
| `model` | The session's `model`, `effort` and `provider`: once at the start, and again whenever `/model` or `/effort` changes one. |
| `mode` | The session's `mode` (`auto`, `partial` or `structured`) and its `label`: once at the start, and again at every `/mode` switch. |
| `approval` | A question for the person, in ask-before-compute or structured mode: `id`, `kind` (`job` or `checkpoint`), `title`, `detail` and `choices`. The session waits for the answer. |
| `approval_done` | The question `id` was answered: `outcome` is `approved`, `declined` or `approved_all`, and `note` carries what the person said. |
| `error` | An exception escaped the session. `session_end` follows with `crashed`. |
| `cost` | A trace event (see `OPENREYNOLDS_TRACE`), on the same stream. |

Without `-p` the same flag makes a **conversation**: it reads newline-delimited JSON
from stdin, one message per line, `{"type": "user", "text": "..."}`. A `prompt` event
says when it is your turn. Anything on stdin that is not an object this understands is
ignored rather than guessed at. Commands go the same way: `{"type": "user", "text":
"/status"}`. An `approval` is answered with a `user` line saying `/yes`, `/no <reason>`
or `/all`, and until it is answered the session waits.

## Configuration

Credentials live in a config file outside any repository, and beside it is
`preferences.md`, your standing note. Studies are written under `./studies/` in the
directory you started from.

Everything `config` sets can be set in the environment instead, which is what CI and
containers want:

| Variable | Meaning |
| --- | --- |
| `OPENREYNOLDS_PROVIDER` | A preset name, or `reynolds` for the metered model. It does not hold a resume to that provider: a study restores the provider it recorded, so name a model too if you mean to move one. |
| `OPENREYNOLDS_LLM_API_KEY` | The model key. The vendor's own name (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) is read too. |
| `OPENREYNOLDS_MODEL` / `OPENREYNOLDS_EFFORT` | Which model, and how hard it is asked to think (`low`, `medium` or `high`). A resumed study keeps the model it was last running unless `OPENREYNOLDS_MODEL` names one. |
| `OPENREYNOLDS_MODE` | `auto`, `partial` or `structured`, or an alias. A value that is not a mode is ignored with a warning: the config file's `mode` applies, or full auto, and a resumed study keeps its stored mode. |
| `FOAMD_URL` / `FOAMD_API_KEY` | The workspace service and this machine's key. |
| `OPENREYNOLDS_MIRROR_INTERVAL_S` | How often files come home. `0` turns it off. |
| `OPENREYNOLDS_NARRATE_EVERY_S` | How often a long run wakes the model. `0` turns it off. |
| `OPENREYNOLDS_WORKSPACE_ETA_S` | How long a workspace usually takes to come up (default 30). A session starts talking before its workspace is up; this is the "usually about N seconds" the model is told while it waits. |
| `OPENREYNOLDS_CAPTURE` | `0` sends nothing to the platform. |
| `OPENREYNOLDS_TRACE` | A file to append cost events to: one JSON object per turn, tool call and mirror cycle, with the token counts split apart rather than summed. Unset writes nothing. |

By default the transcript of every study is uploaded to the workspace service as it
runs, so a study is kept somewhere other than one laptop. `--no-capture` for a session,
or `OPENREYNOLDS_CAPTURE=0`, keeps it on this machine only.

## Layout

| Module | What lives there |
| --- | --- |
| `cli.py` | Entry point, session assembly, subcommands. |
| `loop.py` | The tool-use loop: streaming, interjections, thread refresh. |
| `tools.py` | The eight tool schemas, `checkpoint` for structured mode, and their handlers. |
| `modes.py` / `approval.py` | The three modes and what each gates; putting a question to the person and reading the answer. |
| `commands.py` | The one registry of typed commands, read by the parser, `/help` and both completions. |
| `switch.py` | Changing the model, provider or effort mid-study. |
| `cad/` | The CAD desk behind the `cad` tool: its own brief, its own model client, and one python cell a step in a kernel on the workspace — build or import the shape, tag it, mesh it, look at it, revise. `check.py` is the finish line it does not declare for itself, and `cells.py` decides which cells become the script the run leaves behind. Nothing here imports a CAD kernel into this process. Geometry and meshing used to be a stack of generators and a spec language here; an agent replaced all of it on 2026-09-07, and the kernel replaced that agent's bash. |
| `watch.py` | Job polling, wake facts, progress, narration. |
| `mirror.py` / `store.py` | Files home, and the local `./studies/<id>/` record. |
| `backend/` | The `Backend` protocol. `hosted.py` is the only module that knows the service exists; `pending.py` stands in for a workspace that is still coming up, so a session talks before its machine is there. |
| `llm/` | Provider adapters: Messages API, Chat Completions, and the preset table. |
| `tui.py` / `view.py` | The interface, behind a presentation-only `View` seam. |
| `toolbox/` | The optional scripts and the field notes. |

The `View` seam is why the same session runs in a terminal, in CI, and behind the
hosted web app without the loop knowing which it has.

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests -q          # no network, no account, no workspace needed
```

The suite is hermetic: no model vendor is contacted and no workspace is booted.
`tests/test_wiring.py` requires that every CLI option, `Config` field, `ToolContext`
field, `View` method and `Backend` method is actually read somewhere, so adding a field
means wiring it end to end or failing the build.

## Security

The agent runs an unsandboxed shell on your hosted instance. In the default mode, full
auto, there is no approval gate at all; that is the design, not an oversight. If you
choose ask before compute or structured, the harness holds `job_start` for
you and nothing else: `bash` still runs without asking in every mode, so a mode decides
when compute is spent and is not a sandbox. [SECURITY.md](SECURITY.md) says what follows from it:
what the model can see, what leaves your machine and when, and where to report a
problem (security@inviscidai.com).

## License

MIT. See [LICENSE](LICENSE).
