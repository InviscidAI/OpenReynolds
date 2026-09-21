# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **The workspace is listed by the service, running or stopped; the `find` over the
  exec channel is the fallback, and a cut output can no longer invent a file.**
  `Browser.tree` listed a workspace with `find ... | sort | cut | head -n 4001` over
  the exec channel, and the hosted workspace's daemon caps a command's output at 64 KB
  and cuts it mid-line (`OpenFoam_Instance/daemon/settings.py` `EXEC_OUTPUT_CAP_BYTES`,
  `daemon/runner.py` `out[:cap]`; the answer carries `truncated: true`, which
  `HostedBackend.exec` mapped onto `ExecResult.truncated` and `Browser.tree` ignored).
  Measured in production on 2026-09-21 (study 20260921-033356-076b, workspace
  35c9f018): the listing is ~1,532 rows at ~83 bytes each, ~127 KB, against the 64 KB
  cap -- so the 4,000-entry cap never bound anything, the byte cap did, at ~850 rows.
  The partial last row, `f\t1528155\t<mtime>\t/work/20260921-033356-076b` -- the row
  for `mesh/zoom.png`, 1,528,155 bytes, cut right after the study id -- has four
  well-formed fields, and `_parse` accepted it: an `Entry` with that path,
  `is_dir: False` and `size: 1.5 MB` -- a file at the study root. Everything that
  followed was measured too: the page's tree collapsed to one unexpandable "remote
  1.5M" leaf (ui #31 now hardens the page); the mirror asked the service to archive
  that "file" every cycle (`tar?mode=pack&paths=/work/20260921-033356-076b` -> 502/504
  each 20 s from 04:07 to 04:16), plus 404s for the other paths a cut produced (`/r`,
  `/run/processor`, `/run/p`); and the live tree during a run held ~850 of the
  study's 1,532 rows -- since #36 the order is breadth-first, so the cut lands in
  `run/processors4/<time>/`, but everything past it was invisible to the live pane and
  to the mirror until the workspace was stopped. Meanwhile #38/#39 had given the
  hosted backend `list_stored` -- `GET /v1/instances/{id}/files?list=1`, recursive,
  5,000 entries -- which lists any path under `/work` from the live daemon while the
  workspace is up and from the service's copy once it is stopped, breadth-first,
  capped at 5,000 with its own `truncated`, never starting a machine and never
  counting as use of one (foamd #53) -- and `Browser.tree` used it only when a poll
  said the workspace was idle. Now `tree()` asks `list_stored` first, whatever state
  the workspace is in and whether or not the call is a background poll; the answer is
  the listing, cut to the depth and to 4,000 entries breadth-first, the service's
  `truncated` carried. No `find`, no exec channel, no byte cap, no partial line, and
  one path for both states. The walk runs only when `list_stored` answers None -- a
  local backend, a workspace still coming up (`PendingBackend.list_stored` is None
  until the machine is there, and the walk then waits for it as every call did), a
  service without the route, a failed request -- and then exactly as before: a poll,
  then work only when nothing is running and somebody is asking; a background walk
  that finds nothing running still raises `workspace_idle` and starts nothing. The
  walk itself is made safe for the local backend and for the fallback: when the
  backend says the output was cut, the tail after the last newline is dropped
  unparsed, and the listing is marked truncated with a notice that names the output
  cap (a listing of 850 rows told it was "capped at 4,000 entries" would contradict
  what is in front of the reader); and `_parse` refuses a row whose path is the
  listed root or is not under it -- the walk is `-mindepth 1`, so such a row can only
  be a cut or noise. The listing request is bounded at 60 s, the minute the `find`
  had (`FoamdClient.request` with no timeout is no timeout at all, and the mirror
  asks this every twenty seconds). The route's `mtime` are whole seconds where
  `find`'s carried a fraction; `Entry.mtime` stays a float and nothing that reads it
  is finer than two machines' clocks. Behaviour changed for the background cycles of
  a session whose workspace is stopped: they now read the service's copy each cycle
  -- one bounded request, nothing started, nothing pulled when nothing has changed --
  where before they waited out the idle workspace. Unchanged: a backend without
  `list_stored` (a test's, an embedder's) lists as it always did; the 4,000-entry cap
  and its notice; the poll-first order of the walk.
- **A steady solver's residuals levelling off on an unsteady flow is no longer reported
  to the person as a run that "did not converge".** The owner, after five solving
  studies in three days: "In each solve I just wanted a quick look, not a mesh
  independence study, why is it that every time it kept telling me it couldn't
  converge??? ... Nothing converges is it?" -- and, once the transcripts were read
  back to him: "if it's never gonna converge it doesn't mean it's physically
  inaccurate! But the way it's phrased it's always so negative." Read with the
  operator tool (`OpenFoam_Instance/scripts/study_log.py`), not one of the five had
  diverged, and every mesh was checkMesh OK (max non-orthogonality 30-38, skewness
  0.6-1.9). What happened, and what was said: (a) 20260919-041432-e0b6, a laminar
  backward-facing step at Re_S = 800 under `simpleFoam` -- the Ux residual sat at
  2.2e-2..3.2e-2 from iteration 250 to 30,000, on two meshes (9k, 23k cells), at
  relaxation down to p 0.2 / U 0.4, first- and second-order, eight solver runs in all;
  the flow is past the 2D Hopf bifurcation, so the plateau is the physics. Headline:
  "**The headline, honestly: at Re_S = 800 there is no steady solution to converge
  to.**"; README column "converged?" answered "no (res. floor 1.5e-2)" three times.
  (b) 20260920-155504-4379, the "Inviscid AI" text under `simpleFoam` k-omega SST,
  106,716 cells, Re 6.7e4, 2000 iterations in 157 s -- Ux 1.2e-2..2.0e-2, Uy
  3.6e-2..5.1e-2, p 3.0e-2..3.9e-2 from iteration 200 to 2000, flat, no bounding; the
  pictures were made from that field. Told: "**Honesty:** the run did **not**
  converge -- residuals plateau at ~1e-2 ... The fields are a frozen pseudo-transient
  snapshot ... treat them as +/-25%." (c) 20260920-161908-c7ef, the same text under
  `pimpleFoam`, 0 -> 0.8 s in 22 min, per-step residuals Ux 5e-5 / p 4e-2 (a
  transient's normal shape): "take Cd as +/-20-30%, not converged. One mesh, no
  independence study", with "Mesh independence -- 3 meshes ... ~1.5 h" first among the
  offered follow-ups; nobody had asked for one. (d) 20260921-033356-076b, `pimpleFoam`
  at Re 6.7e3, stopped by the person at t = 1.48 s ("just plot whats there already"),
  no bounding, Co max 2.4: the gif was delivered, then "Cd ... **Not converged and not
  trustworthy.**", "Strouhal number: **I will not quote one.**", a figure titled "NOT
  statistically converged". (e) 20260921-033019-e1b4, `pisoFoam` laminar at Re 200:
  clean, one line about "no grid-refinement check". Three senses of "converge" -- a
  steady residual, a grid, a time average -- reached the person as one message, four
  times in five. Every diagnosis was accurate; the framing turned the physics into a
  confession. Where it came from: the system prompt's one sentence on the subject
  named "an unconverged solve" as the first thing to be honest about (the "Honesty:"
  headers in (a), (b) and (c) are that sentence answered); nothing told the model what
  a stall is, that a bluff body under a steady solver stalls by nature, or what a
  quick look may take from a stalled field; no residual tolerance in the harness was
  involved (none of the five cases used `case_gen.py`'s `residualControl`) and no
  mesh-quality gate sent anything back to remeshing. The change is guidance, in the
  places the model reads at the right moment, plus the one summariser that had the
  negative wording built in: `openreynolds/convergence.py` holds the line -- a stall on
  an unsteady flow or a plateau around 1e-4..1e-3 on a first look is a physically
  meaningful field, reported as what the flow is doing with the residual in one neutral
  clause; failure words are for a residual that climbs, a floating point exception, a
  field the solver keeps bounding or a mesh checkMesh rejects, with what failed and what
  would fix it; a transient window or a mesh study is offered, not started unasked --
  with an example sentence of each kind, and the briefing carries it in every session
  (`cli._situation_brief`). The system prompt's honesty sentence now names "a run still
  moving when its number was read" as "a fact about it and not a verdict on the run".
  `job_start` on a steady solver (`simpleFoam` and its kin, told by name) adds one
  clause to the launch note: its residuals level off rather than fall on an unsteady
  flow, and the levelled-off field is a snapshot worth showing, not a failed run.
  `toolbox/log_digest.py` read every run that reached its end without the solver's own
  "converged" message as "ran to the end of controlDict without reporting convergence"
  -- the same words for a plateau on a shedding wake as for a residual climbing towards
  an exception; it now reads the residual series (`residual_shape`: diverging, still
  falling, levelled, or too short to say) and says which, e.g. "residuals: levelled off
  (Ux ~1.5e-02, p ~3.0e-02) from about step 200 and stayed there -- a plateau, not a
  divergence", against "residuals: climbing (Ux best 1.6e-05, last 1.8e+01) -- this run
  is diverging, not converging slowly, and the last field is not one to show"; the
  climbing/blow-up thresholds are the ones `preflight.py`'s residual gate already used,
  now shared. No solver default changed: the runs were right. Tests pin the guidance
  text (the failure words appear once, as the words ruled out; no imperative workflow
  language), the briefing and launch note carrying it, and the digest's reading of a
  steady stall, a divergence, a still-falling series and a run sitting at its floor.
- **The workspace listing is breadth-first, so the cap falls in the solver's bulk, not
  on the pictures.** `Browser.tree` ran `find | head -n 4001`, and `find` walks
  depth-first in directory order. In a transient study (20260920-161908-c7ef, in
  production) the walk went down `run/processors4/` first and spent the whole cap on
  per-time field files; `renders/shedding.gif`, `README.md`, `make_gif.py` and the 201
  animation frames -- all written, looked at and described to the person -- were past
  the cap in every listing, so the live mirror never saw them and never brought them
  home, and the finished study's page showed `run/` and nothing else. The listing now
  prints each entry's depth, sorts on it (stably) and strips it again before the cap:
  every shallow entry precedes any deep one, and the cut, when there is one, lands
  among the deepest. The cap and its notice are unchanged.
- **The close-down of a session whose workspace has already been stopped no longer
  starts a machine to list it.** The final sync begins with a listing (`Browser.tree`,
  a `find` over the exec channel), and a foreground exec on a hosted workspace the
  service had stopped lazy-starts a new one to run it. In production on 2026-09-21
  (issue #37, item 1) an idle-timed-out session's close-down started a c7i.2xlarge at
  02:24:32 -- the workspace had been reaped at 01:31 -- listed the files, and the
  machine then sat until the reaper took it down again at 02:42: eighteen minutes of
  instance for one listing. The service keeps a copy of the workspace, written at every
  checkpoint and stop, and once the workspace is stopped it already serves `get_file`
  and `get_tree` from it; `GET /v1/instances/{id}/files?list=1` lists it. So a
  foreground listing is now asked as a poll first (`background=True`, which never
  starts anything and does not count as use of the workspace); when the poll finds
  nothing running, the listing is read from the copy (`Backend.list_stored`, which
  `HostedBackend` answers through that route, cut to the depth asked, capped and
  ordered as the walk's is); and the machine is started only when there is no copy
  to read -- which is what a foreground listing always did, and is still right when
  somebody is working. The first cut asked the study-scoped route
  (`GET /v1/studies/{id}/workspace`) and still started a machine for a study whose
  home is the workspace root itself -- that route refuses `/work` -- measured on
  production the same morning (09:58 SGT, a pool adoption for one `find`); the
  instance-scoped route has no such refusal and needs no study id. The session still
  tells the workspace which study it serves (`Backend.study_id`, set when the study's
  row is opened or resumed, and carried through the `PendingBackend` stand-in).
  Unchanged: a running workspace's
  own listing is used and the copy is never asked; the background cycles still wait
  out a stopped workspace rather than read it every twenty seconds; a local backend,
  and a study with no row on the platform, list exactly as before.

### Changed

- **The mesh desk builds in the background.** A `mesh` call used to run the desk on
  the loop's own thread and hold it for the whole build. Measured in production
  (study 20260920-161908-c7ef): one call held the agent for 402 s and some twenty-five
  model calls, during which the main agent answered nothing, every line the person
  typed was drained into the desk by its `interject` -- the same inbox the loop reads
  between its own tool calls -- and on the web page each typed line sat marked as
  pending until the desk consumed it, which read as "sending a message takes minutes".
  Now `mesh` starts the desk on a daemon thread (`openreynolds/mesher/background.py`,
  `DeskRun`) and returns at once with where it is working; the conversation carries
  on; the desk's steps show in the transcript as before; and its end wakes the model
  through the same watch loop a job's end does (`watch.Wake("desk")`), the person's
  typed lines still winning. Two tools join it: `mesh_note` passes a remark to the
  running desk, which reads it at its next command -- the desk's ears are its own now,
  and the session's inbox is the main agent's again -- and `mesh_wait` holds for the
  result, bounded like `job_check`'s wait and ending early when the person types. One
  desk runs at a time; a second `mesh` call answers with where the first has got to.
  `mesh` with `wait: true` is the old shape, whole. A `-p` run waits for a live desk
  as it waits for a job. `session.json` records a live run, so a session resumed
  after its process ended mid-build is told so once, with where the desk's work is.
  All three tools are absent when the desk is not configured, as `mesh` was.

## [0.3.1] - 2026-09-17

### Fixed

- **A session that joins a running workspace no longer leaves it up for nobody.**
  Joining used to decide the close-down on its own: a joined session always left the
  workspace running, so the next `openreynolds` found it still up, printed "joining
  the workspace already running", and left it up again -- relaunching the CLI kept an
  idle container alive (and billing) indefinitely, with no session on the web to
  explain it. A joined session now asks the same questions as one that started the
  workspace -- running job rows, and the probe for another session's processes -- and
  stops it when nobody is working on it. A sibling session that is live but idle at
  that moment starts a fresh container on its next call; the volume is untouched.
  Studies with no directory of their own still leave a joined workspace up.

## [0.3.0] - 2026-09-17

### Added

- **Three modes: full auto, ask before compute, structured.** `--mode`,
  `OPENREYNOLDS_MODE`, the config file's `mode`, or `/mode <name>` mid-session
  (`openreynolds/modes.py`). `auto` is the default and is today's behaviour byte for
  byte: the briefing is unchanged and no tool call is ever held. `partial` puts every
  `job_start` and `mesh` call to the person before it runs (`/yes`, `/no [reason]`,
  `/all`), and a declined call goes back to the model as an error result carrying the
  person's reason verbatim. `structured` offers a `checkpoint` tool that shows the
  person a summary and what comes next and waits for their answer, and holds
  `job_start` and `mesh` until a checkpoint has been approved since the mode began.
  Its stage names are the guided pipeline's own phases (`toolbox/study_state.py`).
  The question and its answer live in `openreynolds/approval.py`; the one place a call
  is held is `Loop._consult`. A resumed study keeps its mode unless one is given.
  `-p` with a non-auto mode is a usage error (exit `2`), because nobody is there to
  answer.
- **Change the model or effort mid-study.** `/model` shows the provider, model, effort,
  mode and the models this provider is known to have; `/model <model>`,
  `/model <provider>:<model>` and `/model <provider>` switch
  (`openreynolds/switch.py`). A candidate is probed, image included, before it is
  accepted, and applied only between turns. Earlier thinking blocks are dropped when
  the model changes, and a thread too big for the new model's window (known per model
  where it differs from the preset's, as for `claude-haiku-4-5`) is refreshed on the
  current model first. `/effort low|medium|high` applies from the next request, and
  `--effort` sets it at the start. The mesh desk and the front desk follow the switch.
  A resumed study carries on on the model it was last running: `session.json` now
  records the provider and the endpoint beside the model, and `--study <id>` restores
  the record unless `--model` or `OPENREYNOLDS_MODEL` names a model, or this machine
  cannot serve it -- in which case the configured model runs and the session says why.
  Serving it means a key for the recorded provider here, switching provider if the
  record says another one (it brings that provider's key, endpoint, window and desk
  model with it), and on the configured provider the same endpoint as well, because a
  gateway or router in front of one family answers to other vendors' ids and a pair
  restored across endpoints fails a turn later with the vendor's 400. A study that
  recorded no provider or no endpoint is refused rather than guessed at, and starts as
  it did before. A refused restore does not rewrite the study's record: the run that
  cannot serve the pair is the last one that should forget it, and the next resume
  where the key is set carries on there. The record is the local `session.json`, so a
  study resumed where it has never run starts on the configured model, and
  `OPENREYNOLDS_PROVIDER` on its own does not hold a resume to that provider.
- **`/help` with topics.** `/help commands`, `/help modes`, `/help model`,
  `/help tools` and `/help keys`. Every command lives in one registry,
  `commands.COMMANDS`, which the parser, the help text, the terminal's completion and
  the hosted app's suggestion list are all read from.
- **Completion in the interface.** Typing `/` in the prompt opens a list of matching
  commands above it with a grey completion in the prompt itself; Up and Down move,
  Tab takes the suggestion, Esc closes the list. After `/mode `, `/model `, `/effort `
  and `/help ` the list offers that command's choices. The session bar shows model,
  effort, provider and mode.
- **New stream events.** `approval`, `approval_done`, `model` and `mode` on
  `--output-format stream-json`. A question is answered by sending `/yes`, `/no ...` or
  `/all` as an ordinary `user` line.
- **The moving-mesh benchmarks are in the repository.** `benchmarks/moving_mesh/`
  carries the two prompts run end to end on 2026-09-12 -- a `cyclicAMI` sliding
  interface through a circular-Couette annulus, and a cylinder on a spring free to move
  at Re = 100 -- and `grade.py`, which grades a rerun the same way. What was graded
  lived outside this repository, so nobody could reproduce it. The grader reads only
  raw solver output and never the run's own analysis, because grading those two by hand
  found four summary numbers that disagreed with their own files and one analysis
  script that read a torque at t = 13 and called it t = 20, having never opened the
  restart directory. Restart directories are stitched and each cut at the start of the
  next, columns are found by header name, and every physical constant comes off a case
  file or the grade stops and names the flag that supplies it. Two checks are the
  findings themselves: the interface's torque imbalance, which conservation fixes at
  exactly zero and which the solver's own `sum(weights)` cannot see, and the VIV energy
  audit. The known result now carries its grid convergence (three meshes per case,
  observed order 1.93 conformal and 0.79 with the interface) and the separation that
  matters for a rerun: at a fixed time step, four times the cells moves the interface's
  torque error from 0.2174% to 0.2205%, so the quantity to converge is `omega*dt` and
  not the cell count. The two nearest published amplitudes are named with the
  parameters that differ, and A/D is still reported without a verdict.
- **The moving-mesh field note carries those two runs.** "When the mesh moves" had been
  written from the Wigley free-surface hull alone. It now also records the cell
  diffusion number `nu*dt/dr^2` as what a diffusion-dominated refinement is limited by
  (stable at 3.1 on two correctors, divergent at 6.25 and 12.5, stable to 11.5 on
  four), that holding Courant fixed under refinement doubles it every level, that a
  sliding interface's error is first order in the angular slide per step, that the two
  wall torques' imbalance is a free check the `sum(weights)` health check misses
  (0.91% against a conformal control's 0.011% on a case whose weights were perfect to
  43 ppm), and the energy audit that moved a VIV amplitude from 0.640 to 0.567 after
  every conventional convergence test had passed it.
- **The mesh desk's finish check asks whether anything changed.** Asked to make a
  channel 30 mm tall instead of 20, the desk rebuilt it, ran its own look, reported a
  clean rebuild -- and the channel was still 20 mm, with identical bounds and an
  identical cell count. Every clause the check had passed, because after a no-op there
  is a valid mesh of the right rough size and nothing compared what was asked against
  what changed.
- **One tag publishes both registries.** `publish.yml` sends the Python package to PyPI
  and the launcher to npm from the same tag, by trusted publishing on both sides, and
  refuses before publishing anything if the tag, `__version__` and
  `launcher/package.json` disagree or if the tag falls outside the launcher's
  `PYTHON_PACKAGE_SPEC`. 0.2.0 reached PyPI while npm still shipped a launcher pinned
  `>=0.1,<0.2`, which does not half-ship a release, it mis-ships one: every
  `npm install -g openreynolds` keeps installing the old version and `upgrade` keeps
  agreeing it is current. `workflow_dispatch` catches one registry up on its own.

### Fixed

- **A Sandbox that cycled under a read is waited out instead of failing the tool
  call.** The hosted Sandbox goes away under load -- 9 of 68 tool calls in one live
  study and 56 of 244 in another came back `409 sandbox_gone`, in bursts, three of the
  four starting within 90 seconds of an 8-rank solve being launched (F-58). Nothing was
  lost, because `/work` is a persistent Volume and a job restarts from the latest time,
  but every one of them reached the model as a failed call and one run spent about 16
  minutes improvising recovery and relaunched the same solve stage five times. The
  service now answers that 409 with `Retry-After: 5`, and `FoamdClient.request` asks
  again once for a method that only reads (a stat, a file, a listing, a job status),
  honouring the header and capping what it can add at 15 seconds. 409 stays out of
  `_RETRY_STATUSES`: that set is keyed on the status alone and would cover
  `POST .../jobs` too, and a retried job start once produced five duplicate running
  jobs. `job_start` needs an idempotency key before it can be sent twice.
- **A file read back mid-write is no longer handed over as though it were whole.**
  `read_file` stats the path, asks for exactly that many bytes and is given exactly
  that many, so the short-read guard could never fire: a `postProcessing` forces file,
  a `.dat`, a `.csv` or a log read while the solver is still appending came back
  looking complete, and a file cut short is a number the model will happily average
  (F-64). Pictures were covered when a half-written PNG ended two sessions; the text
  the conclusions are drawn from was not. The path is stated a second time after the read -- one
  round trip on a path that already makes several -- read once more if the size moved,
  and if it is still moving the answer says so and names the three sizes rather than
  presenting a snapshot as the file.
- **A named model is no longer swapped for a preset's default.** A provider named on
  its own still arrives at its preset's model, but `OPENREYNOLDS_PROVIDER=reynolds`
  with `OPENREYNOLDS_MODEL=claude-opus-5` -- one of the two models that service meters,
  and what the hosted app sends when someone picks Opus -- loaded as Sonnet, on new
  sessions as well as resumes, and `/model claude-opus-5` on `reynolds` was undone the
  same way. What was asked for is re-asserted after the preset fills its blanks
  (`Config.load`, `switch.candidate`), for the desk model too.
- **Piped output stops folding long lines.** `rich` folds at the width it is given,
  however wide, so a message longer than `PIPED_WIDTH` arrived at a pipe with a newline
  in the middle of it -- a broken path for any program or agent reading the CLI through
  one, and the last of Windows CI's red. `soft_wrap` on the non-terminal console.
- **`job_kill`'s docstring stops promising a `killed` status for a kill that did
  nothing.** The service no longer marks a job killed whether or not the signal reached
  anything (F-63): it records a real ending for a job that had already ended, and
  refuses with `409 kill_not_delivered` for one that may still be running. Callers
  already treated a `BackendError` as a failed kill, so only the promise changed.
- **The dev extra installs `pyvista`.** Six toolbox tests read measurements through
  `mesh_look.py` and got empty values back instead of a loud `ModuleNotFoundError`.
  The symlink test also measured the fixture rather than the link, and failed on a
  `disk.py` that was already correct (#28, #29).
- **The README's images load, and the eighth tool is the one that exists.** All five
  figures had been 404 since tryreynolds.com moved them into a hashed asset bundle, so
  the repository front page, the PyPI page and the npm page had shown broken images
  through a release of each. The eighth tool has been `mesh`, not `geometry`, since
  2026-09-07, and `fetch` copies files out of the workspace rather than reading the
  open web.
- **The README says how to get in and what it costs.** Sign-in was described as email
  and password only; an account created with Google has no password at all, and
  `login --browser` is the way in for one. A company email address starts the account
  with $10 of credit, once, which the CLI already printed and the README did not
  mention. The hosted app's mesh editor and `benchmarks/moving_mesh/` are named, and
  `doctor --json` and `studies --json` are documented where a program looks for them
  rather than in the last clause of a table row.

### Changed

- **The free-will contract is amended, not dropped.** `docs/design.md` section 1, the
  system prompt and the README's Security section now say what is true: no approvals
  and no enforced ordering in the default mode, and exactly what the person chose to
  have gated in the other two.
- **`fetch` is described correctly in the README.** It copies files from the workspace
  to your machine; the README had said it read the open web.

## [0.2.0] - 2026-09-12

### Added

- **A fourth interface: `--output-format stream-json`.** A program can now drive a
  session and read it as data. `openreynolds/jsonview.py` implements the whole `View`
  protocol as NDJSON on stdout, one JSON object a line, locked and flushed per line so a
  reader never sees half a record; `JsonReader` is the other half for anything that
  wants to consume it in Python. `studies --json` and `doctor --json` answer the same
  way. The seam already carried three interfaces (the terminal, the hosted app, the
  quiet mode), so this is a fourth implementation rather than a new pathway, and
  stdout purity is held by a test: nothing but NDJSON may reach it, which is why
  `images.suppress()` exists (a graphics escape sequence in the middle of the stream
  leaves a strict reader resynchronising inside a base64 blob, permanently).
- **Trace rows carry identity.** Every tool call now records its `tool_use_id` and a
  `result` of `{ok, bytes}`, so a transcript can be joined call-to-answer by a program
  instead of by eye.
- **The mesh is allowed to move.** `mesh_look.py` reports cell and face zones, the
  motion ladder in the field notes covers `solidBodyMotion`, AMI and morphing meshes,
  preflight carries motion rules, and a "When the mesh moves" note is drawn entirely
  from what the free-surface Wigley run actually paid for.

### Fixed

- **A picture the model will not accept no longer costs the session.** Two long runs
  died on `400 invalid_request_error: Could not process image`, both immediately after
  the agent redrew a figure and read it back, one 27 minutes in and one 2 h 23 m in.
  Two defects had to line up. `attachment` base64-encoded whatever it was handed, and
  base64 of half a PNG is well-formed base64, so nothing between the disk and the API
  could tell: `images.incomplete` now checks that a file carries the marker saying it
  ENDED (PNG's IEND, JPEG's EOI, GIF's trailer, WebP's RIFF length) and a partial
  render comes back as a sentence saying it is not ready. And a 400 ended the run,
  which was right about waiting and wrong about repair, because the bad bytes sit in a
  thread this process owns: `Loop.drop_images` replaces every image with a note and the
  turn is retried once.
- **A resumed session says which workspace it joined.** `acquire()` took `existing[0]`
  from an unordered listing. With one instance per account that was not a choice; with
  several it attaches a resume to whichever row the service returned first, and the
  failure does not look like a wrong choice, it looks like a workspace that lost its
  files. The listing is ordered most-recently-active first, by the same key the
  service's own repair step uses, and the join notice says how many workspaces the
  account holds and that `--instance` picks another.
- **`/work` is resolved before it is compared.** It is a symlink to
  `/__modal/volumes/vo-<id>`, and `/proc/<pid>/cwd` yields the physical path, so any
  probe that string-matched a cwd against the literal `/work` matched nothing in
  production. That had silently disabled the escaped-solver probe for its whole life.
  `stopping.py` and `toolbox/disk.py` resolve the root first.
- **A leaving session stops killing its neighbour's work.** A session that joins a
  workspace somebody else started no longer sweeps or stops it on the way out.


### Added

- The mesh desk (`openreynolds/mesher/`): geometry and meshing are now one small agent
  with one tool. It gets a shape in words and a case directory, and works the way a
  person at a terminal does -- one fenced ```bash block a message, run on the instance
  that already has OpenFOAM, gmsh, build123d and pyvista on it, with the output pasted
  back. Two things the harness owns rather than the model: any PNG a command writes
  comes back attached to that command's output, so seeing is not something to remember
  to arrange; and `echo MESH_DONE` does not end the run -- `mesher/check.py` runs
  `mesh_look.py` on the machine and hands the run back with the reasons unless
  `constant/polyMesh` exists, `checkMesh` passes, at least two patches carry names
  somebody chose (not `patch0` out of gmshToFoam), an `Allmesh` or build script is
  there to rebuild it, and a picture was drawn. Budgets are a turn count and a wall
  clock, and a run that hits one is still checked, because an unexamined mesh is the
  failure this desk exists to end. Three files and a brief, in place of a spec
  language, a compiler, a linter, a claims engine and a shape library.
- `toolbox/mesh_look.py`: any meshed case drawn and measured in one call, straight from
  `constant/polyMesh` -- no solve, no fields, no time directory needed. One PNG with
  **one colour per patch** (2D: the patches and the cells, with the `empty` front and
  back left out so the others are visible; 3D: the patches with the flow box drawn faint
  and a cut on each axis), and under it cells, faces, points, the bounding box, whether
  the mesh is one cell thick, `checkMesh`'s verdict with maximum non-orthogonality,
  skewness and aspect ratio, and for every patch its type, face count, area, centre and
  mean unit normal. The normal is the point: it says which end is the inlet without
  anybody guessing from where the patch sits, which is the classification mistake that
  silently mislabelled every L, U and elbow.
- The hosted app's **Mesh** panel (`ui`): pick a case, press Look, and see the same
  picture and the same table the desk works from, with a box to ask the session for a
  change in words. It replaces the CAD-spec editor, whose spec language went with the
  toolbox that read it.

### Added

- `toolbox/disk.py` -- what is filling the workspace and what of it is safe to delete
  (F-56). There is one disk: an account is capped at one instance, a new study joins the
  existing one, and every study ever run is a sibling directory under `/work` sharing a
  single 20 GB quota, with no retention policy and no owner. Measured 2026-09-08 at
  30.8 GB used against that quota, with a live 24 MB study unable to write; 24.5 GB of it
  was regenerable scratch. `report` ranks by size and says how much of each directory
  could be produced again; `prune` removes only that and never an input, a log, a result,
  a figure or an `Allrun`. Dry run unless `--apply`, refuses to guess which study, and
  never deletes a whole study directory.
- `casebundle.py` -- the case itself, captured at session end. The capture plane has
  always carried the conversation and never the work: measured over 215 production
  studies, the transcript is there for essentially all of them, an artifact for 40%, a
  results payload for 12%, and the case definition, mesh and solver logs for **none**. So
  every mesh and every `Allrun` this product has made lived in exactly one place, on that
  shared disk. The bundle is built from the local mirror (no extra network) in tiers,
  irreplaceable first -- definition, record, notes, geometry, then mesh and latest fields,
  which a resume can rebuild and re-solve. `definition + record` is about 4 MB a study, so
  the part that cannot be reproduced always fits. Nothing is dropped silently: the report
  and the `MANIFEST.json` inside the archive both name what was left out and why.
- `toolbox/cfmesh.py` -- cfMesh reachable from the toolbox, with the four traps that each
  cost a round encoded so they cannot be walked into (F-53). snappy's layer coverage
  erodes 88.6% -> 66.7% over fifty iterations on a thin hull while cfMesh covers 100% of
  the same wall at its defaults, and cfMesh has been prebuilt in the image all along.
  What it costs is stated rather than glossed: no y+ control.
- `toolbox/reattach.py` and `toolbox/claims.py` -- one blessed reattachment implementation,
  and the number a figure was drawn from stamped into the figure (F-36). A study once
  delivered a figure annotated 7.57 beside an answer that said 6.4 and reconciled neither.

### Fixed

- A redirect on a request carrying a body is no longer followed blind (F-47). httpx
  re-issues a 301/302/303 on a POST as a **GET with no body**, so the blanket
  `follow_redirects=True` added for F-45 made any redirect on an upload a body-losing
  event. 307/308 and bodyless requests are followed; a 303 is followed and *remembered*,
  so a 4xx collected through a hop the body did not travel on is treated as ambiguous
  rather than as a fact about the request; 301/302 on a body-carrying request are refused
  and named. And `put_tree` still holds the archive it built, so on a `400 not a valid
  tar.gz archive` it opens those bytes and, if they are a readable gzipped tar, says so:
  the bytes that arrived are not the bytes that were sent. The message used to point at
  the wrong party, and the wrong party is the only one that can prove it.
- `describe_job` no longer labels every `killed_by` "matched kill_on line". That was right
  while a `kill_on` regex was the only writer; foamd now writes the field on every
  terminal state that is a kill, and only one of six producers is a log line. The old
  label would have told the model a vanished container was a regex match.
- A session no longer stops a workspace with somebody else's work still running on it
  (F-46). Teardown asked only whether another session had the instance up when this one
  joined it, which is the wrong question about a detached job: jobs outlive sessions by
  design, and a job started outside any session leaves no flag anywhere, so the next
  session to start the instance also owned it and stopped it on the way out. A detached
  rendering job died that way with six of its eight steps done and its two animation
  passes holding empty output directories. `Backend.active_jobs()` asks the service what
  is actually running -- one read of the job rows, which starts no container -- and the
  workspace is left up, and named, when anything is. A listing that fails answers
  "nothing" and says so, because a workspace left up on an unanswerable question bills
  until a reaper notices.
- The capture warning says what it lost (F-44). `capture dropped 1 item(s)` reported a
  count and nothing else, and the only question a person has on reading it is *what*: a
  dropped message leaves a gap in the transcript on the web, a dropped artifact leaves a
  picture that exists on the laptop and nowhere else. Up to six items are now named
  (`artifact mesh_z.png`, `message seq 41 (assistant)`); past that the kinds are counted
  and the first is named, because the middle of a long outage is not readable.
- A word in a comment is no longer a finish. The mesh desk's `MESH_DONE` was matched
  anywhere in a bash block, and the brief hands the desk that token -- so a step whose
  comment said `# will echo MESH_DONE once checkMesh passes` had its actual command
  never run, and got the finish check's refusal as the answer to something it had not
  asked. Found by reading the new code adversarially rather than by a run failing.
- A picture is sent to the desk once per version of the file. A command that merely
  mentioned a `.png` -- a `cat` of the build script, an `ls` -- used to have that file
  attached to its output as though it had just been drawn.
- A model call that fails with an overloaded or gateway status is retried once. A desk
  five minutes into a mesh cannot resume; the next call starts a clean thread.
- A user turn carrying a picture reaches an OpenAI-family model. The provider's renderer
  handled an image inside a tool result and dropped a bare `image` block in a user
  message, which is exactly the shape the mesh desk sends every render in: a model was
  shown nothing and told it had been shown a picture.
- The workspace mirror no longer stalls a session for minutes (F-54). The desk's uploaded
  `Allrun`, a file nothing on the instance ever opens, made foamd's path probe time out
  and answer 400, and the mirror read the 400 as one unresolvable file, split every batch
  into single requests, and repeated that every cycle; a 27-second finish step waited five
  minutes behind it. The mirror now remembers a path the service refused and names it,
  splits a failed batch once to find the culprit, and stands aside while a tool call is in
  flight (`mirror.Gate`, held by the loop around each call); each cycle reports its round
  trips and seconds. Uploaded files keep their source mtime instead of 1970, so a file the
  instance never opened is not mistaken for one never written. foamd's side (a timed-out
  probe answered 504 with its paths) shipped as foamd `9b52d97`.

### Changed

- The previous geometry and meshing stack is **gone**, not extended: the
  `openreynolds/geometry/` package (a sketch API, a compiler, a linter, a claims engine,
  a measurement kernel, a fitness table, a preview renderer, a shape library -- 12,400
  lines), `toolbox/mesh2d.py`, `toolbox/cad_gen.py` and `toolbox/snappy_gen.py`, their
  tests and fixtures, the `geometry` tool, the `geometry` install extra, and the gmsh +
  matplotlib + X/GL layer the hosted runner carried so the old desk could draw in its own
  process. Every mesh generator went with it: the mesh desk writes the geometry itself,
  in whatever gmsh, build123d, blockMesh, snappyHexMesh or cfMesh script the shape
  actually calls for, and leaves that script in the case so a person can change a number
  and re-run it. What is kept from all of it is the one lesson that paid: render and
  measure, never assert.
- `toolbox/case_gen.py` dresses a mesh instead of building one. Its template half went
  with the rest -- a dozen parameterised shapes, an O-grid tiler and a blockMeshDict
  writer, and with them the habit of adding a thirteenth shape whenever a new case did
  not fit. What is left reads the mesh in front of it (patch names and types from
  `constant/polyMesh/boundary`, cells, bounding box and smallest cell from `checkMesh`,
  each patch's area, centre and normal from `mesh_look.py`) and writes `0/`, `system/`
  and `constant/` onto it. The three things that used to come from the template are now
  measurements: the inlet's direction is its own normal with the side the fluid is on
  probed off the mesh (the bounding-box rule it replaces is exactly backwards on a
  U-duct, and worse on anything concave); the characteristic length is the inlet's
  hydraulic diameter, 2w for a plane passage; and a wall patch that reaches at most one
  of the domain's axes is a body in the flow rather than the flow's boundary, which is a
  hundredfold in the free-stream turbulence. Every patch gets a role from what you name,
  then from the mesh's own constraint types, then from its name -- and a name that says
  nothing is a wall, which is the safe reading. Measured: the L-duct the mesh desk built,
  dressed by this, solved 20 iterations of simpleFoam with residuals falling.

### Added

- Each generated case carries an `Allmesh` script: blockMesh, feature extraction,
  snappyHexMesh, `topoSet` where there is a rotating zone, and checkMesh in one
  call, with one log per stage and a digest at the end -- replacing the fifty-odd
  round trips a 3D setup was taking. It stops on the first failing stage with that
  stage's log, and treats checkMesh as the diagnostic it is rather than as a build
  step, since a layered snappy mesh routinely reports concave cells.
- `Allmesh` fails loudly when snappyHexMesh refined **nothing**. A background cell
  wider than the body's cross-section leaves the surface crossing no cell edges, so
  snappy meshes an empty box and exits 0 saying "Finished meshing without any
  errors" -- and the empty box then solves cleanly. The generator now sizes the
  background cell so it cannot happen, and the script checks anyway.

### Fixed

- An unfollowed redirect says so. Modal's edge answers any request open past 150 s
  with a bodyless `303` to the real result; `FoamdClient` has followed those since
  `a228cbf`, but a `3xx` is under 400, so a client built without redirect-following
  handed one back as a success and it died three frames later in the JSON decoder as
  `bad_response (303): the body was empty` -- the most frequent tool error in the
  transcripts, and a sentence that named the symptom while hiding the cause. Both the
  request path and the decoder now raise `redirect_not_followed`, which says which
  switch is off. The four sign-in helpers, which built their own clients, were that
  exact mistake still in the codebase; they follow redirects too. `EXEC_MAX_TIMEOUT_S`
  stays at 300: the cap governs the command, the 150 s governs one HTTP request, and a
  redirect-following client runs a 250 s exec to completion.
- `locationInMesh` survives a body that overhangs the domain. A box cut
  deliberately INSIDE the geometry -- tubes trimmed by their own end planes so they
  span the bank the way a correlation assumes -- puts the body's bound below the
  domain's, and stepping a fraction of the way towards it walked out through the
  floor. snappy then rejects the point while printing a bounding box that does not
  contain it, which reads as a domain problem and is not one.
- `forceCoeffs` states `rhoInf` on compressible cases too. It is the reference
  density the coefficients are normalised by whichever solver is running, and
  without it a thermal run aborts on its first write -- after the mesh is built and
  the case decomposed.
- A symmetry plane gets its own patch. `symmetryPlane` is a constraint requiring
  coplanar faces, so a centreline and a waterline in one patch is rejected by
  blockMesh; and `--far symmetry` now types the mesh patches to match the fields,
  which otherwise fails at `decomposePar` with "attempt to cast type patch to type
  symmetryPlane".
- `locationInMesh` no longer lands on a cell face. Half of an even number of cells
  is exactly a face, and snappy rejects such a point while printing a bounding box
  that plainly contains it.
- Thermal cases use plain SIMPLE rather than SIMPLEC. Pairing SIMPLEC with a 0.3
  pressure factor is conservative on both counts and converges too slowly for a
  fixed iteration budget, which on a heat-transfer case means the wall heat flux --
  the entire answer -- has not settled.
- `case_gen.py` had no `symmetry` role. Every 0/ writer now answers one, `p`
  included -- it was the omission that mattered, because a `symmetryPlane` patch
  carrying a `zeroGradient` pressure is rejected by OpenFOAM outright. The
  constraint is written as `symmetry` rather than `symmetryPlane`: the latter
  requires its faces to be coplanar and so cannot carry the two opposite walls of a
  tunnel as one patch.
- A long `exec` returning a bare 303 with an empty body (F-45). `FoamdClient` now
  follows redirects, so a command that outlives the edge's patience returns its
  result instead of raising. Measured: `sleep 200` now returns rc=0 in 203.9 s
  where it previously raised `bad_response (303)`. This was expensive -- re-reading
  round 2's timings, **27 of 114 minutes (24%) went to backend errors, more wall
  clock than all the solving (18 min, 16%)**, concentrated in one study that lost
  45% of its runtime to it.

- Every field picture came out a quarter turn round. `view_vector` was given the slice
  normal and no up vector, so VTK chose one, and for a z-normal slice it chose +x: a
  cylinder wake in a left-to-right flow rendered as though the flow went upward. Found
  by looking at a render from a live run rather than by reading the code -- the physics
  was right and the orientation was not, which is the worse of the two, because a wrong
  number gets checked and a wrong orientation gets believed. `results.py` and
  `animate.py` now name the up vector; `animate.py` reuses the first frame's camera for
  the rest, so one wrong frame was a wrong animation.
- The colour bar sat across the bottom of the picture, its tick labels over the flow.
  It is under the picture now, in `results.py` and `animate.py`.
- `first_look.py`'s patch panel was a coloured slab on any 2D case: the `empty` patches
  are most of the boundary by area (8,476 faces against 384 for everything else in the
  live run), and from an iso camera they covered every patch the panel exists to name.
  They are left out and marked `[hidden]` in the legend.
- `preflight.py` failed a healthy case. Its cell-count check ran the snappyHexMesh
  estimator -- a uniform background mesh refined around an STL -- against a case with no
  `snappyHexMeshDict`, read a 12-block graded O-grid as 105 cells against 4,238 real
  ones, and advised checking `refinementRegions` the case does not have. Where blockMesh
  builds the mesh alone the count is the product of the block divisions, which is exact,
  so that is what it compares now.
- `render.py` pointed its camera with `plotter.camera_position = "z"`, whose string
  form takes a view plane ("xy", "xz") and not an axis letter, so every field and mesh
  render came out of a camera that was not aimed where the caption said. A study found
  this, worked out the fix and applied it to its own copy under `/work/.toolbox/` --
  the copy that is overwritten from the distribution at the start of the next session.
  Fixed in the distribution, where it survives the sync.

### Changed

- A refusal from the model service is no longer retried. A 402, a 401 or a bad model
  id is the API answering a question about the account or the request, and it answers
  the same way a minute later; 408/409/425/429 and every 5xx are still retried as
  before. When the answer is a refusal the session says so plainly, on the page as well
  as the console, and stops sending until you say something -- job endings are still
  recorded in the thread for whenever it resumes, progress chatter is not. Found in a
  live study: the account budget ran out thirty-five minutes in, and the harness spent
  the next twenty-six minutes making ninety refused calls while the person typed
  "whats going on?" and got nothing.
- The system prompt no longer says the container has 8 cores (the default shape is 4)
  or that `mpirun` fails without `OMPI_ALLOW_RUN_AS_ROOT`. The workspace service now
  starts every instance with the environment OpenMPI needs, so `mpirun` works without
  arranging anything; `nproc` is what reports the core count.

### Added

- Eight toolbox scripts, and one file they all agree on. The toolbox was six small
  readers; the studies kept rewriting the same four hundred lines around them --
  a blockMeshDict generator, a "render every field" script, a plotting script, a
  frames-to-gif script -- from scratch, per study, with the same mistakes in each.
  What is offered now (still offered, never imposed):
  - `study_state.py` -- the contract the rest share. A manifest under
    `<study>/.reynolds/` saying what every artifact is *for* (`mesh-full`,
    `vorticity`, `residuals`, ...) and a phase table saying how far the study got.
    Plain text on the Volume, so a session that ends mid-solve leaves an answer
    behind rather than a directory to re-derive.
  - `case_gen.py` -- runnable cases from a template and a few numbers: 2D external
    flow (circle, square, L, a vehicle, an imported profile), Y/T/Z/F/M ducts, sharp,
    mitred and rounded bends; mesh-only, steady or transient; a moving ground and
    rotating wheels. It solves `nu = U*L/Re` and says so, and every 2D case it writes
    is one cell thick with a single `empty` patch declared in the mesh *and* in every
    field -- the pairing that otherwise stops the solver on the first time step.
  - `first_look.py` -- geometry, the whole mesh, a close-up on whatever the mesh was
    refined around, the named patches and the counts, composed into one contact sheet.
    One `read_file` instead of five round trips, before a core-hour is spent.
  - `preflight.py` -- the cheap questions asked before the expensive run: patch names
    against the fields, `empty` on a 2D case, STL scale, Re against nu, predicted
    cells, checkMesh, a one-iteration probe, Courant, residual divergence, force units,
    free disk. Each answered as a finding with what was measured, what it means and a
    repair -- a suggestion, not a gate.
  - `results.py` -- a finished case as the standard set of pictures and plots in one
    call, by preset, rather than a render command per view.
  - `animate.py` -- extended: velocity, pressure, vorticity, turbulence fields and
    streamlines; colour limits and camera fixed across the sequence; labels burned in;
    a sidecar naming the intended container and fps for the machine that has an
    encoder; and resume, so re-running during a long solve renders only what is new.
  - `progress_report.py` -- one answer to "what is happening?": the phase, the time
    reached against the target, the residual trend, the Courant number, frame progress,
    an ETA that says what it was estimated from, and the paths of the pictures that
    already exist.
  - `study_run.py` -- the pipeline geometry -> preview -> mesh -> checkMesh -> probe ->
    solve -> reconstruct -> render -> animate -> report, resumable: it reconciles the
    recorded phase table against what is actually on disk and picks up at the first
    thing not done. A failing phase stops it and is recorded; what to do about the
    failure stays the agent's call.
  - `gallery.py` -- the manifest as a self-contained `gallery.html` (images embedded,
    no network of any kind in it), a contact sheet, and `--final`, the paths of the
    latest artifact of each kind.
- The frames an animation is rendered from now say what they were rendered *for*, and
  the harness reads it. `animate.py` writes a `frames.json` beside them naming the
  container and the frame rate; `delivery.py` (the automatic assembly during a session)
  and `openreynolds video` both take the output name and the fps from it, falling back
  to a 10 fps gif only for a directory that never declared one. Before this,
  `--format webp --fps 24` rendered fifty correct frames and the harness silently made a
  10 fps gif -- the pictures crossed the gap between the instance and the laptop and the
  intent did not, and `webp` was unreachable by any path.
- Field notes: "When `mpirun` will not start" -- the PMIx signature, why an instance
  with no outbound network produces it, and `PMIX_MCA_gds=hash`. Two studies lost an
  afternoon between them to this; one ran eight cores serially for an hour rather than
  read to the end of the error that named the fix.
- Field notes: a pointer to `log_digest.py` and `mesh_digest.py` where the failure
  signatures are read. Grepping a long log several times is three round trips to the
  instance for what one call answers -- a live session spent a minute on three passes
  over the same 1,700-line `log.snappy`.

## [0.1.0] - 2026-08-27

The first release: the tool-use loop over a hosted OpenFOAM workspace, seven tools, the
terminal interface, watch mode with factual wakes, the local study mirror, capture,
stopping that verifies, the toolbox and the field notes -- and, since the first cut
on 2026-08-24:

### Added

- An npm launcher (`launcher/`, published as `openreynolds`): `npm install -g
  openreynolds` finds or installs `uv`, installs the Python package from PyPI once,
  and runs it. No Python source ships in the npm tarball.
- `openreynolds login`: sign in from the terminal by approving a short code in the
  browser; the service key lands in the config file. The service has a default address
  now (`https://api.tryreynolds.com`), so a fresh install needs no URL.
- Bring your own model, not just your own key: `OPENREYNOLDS_PROVIDER` /
  `openreynolds config --provider` pick Anthropic, OpenAI, Z.ai, DeepSeek, Moonshot,
  MiniMax, OpenRouter or a local Ollama, or any endpoint speaking the Messages API or
  Chat Completions (`openreynolds/llm/`). `ANTHROPIC_API_KEY` still works; the general
  name is `OPENREYNOLDS_LLM_API_KEY`, and older config files are read as before.
- A live mirror: the study's files are synced to `./studies/<id>/files/` in the
  background for the whole session, and a render the model just looked at arrives at
  once. The interface's files pane reads from it.
- A progress bar for real compute -- solver time against `endTime`, residuals, Courant
  number, snappyHexMesh phase -- shown only while a solve, mesh, decomposition or sync
  is running.
- A front desk: a second, cheap agent that answers the user within seconds while the
  main agent is mid-turn, and writes the plain-language "now" line. It is read-only
  and cannot steer the agent. `OPENREYNOLDS_DESK=0` turns it off.
- Render delivery: every image the instance writes lands in a flat
  `./studies/<id>/renders/` folder, frame directories are assembled into gifs locally,
  and the interface has a renders tab. `openreynolds renders` lists them.
- `openreynolds video` encodes a frame set on the user's machine (ffmpeg, or imageio as
  a fallback); the instance never carries an encoder.
- `openreynolds push` carries a local file or directory up to the study's workspace.
- `job_check` takes `wait_s`, holding the answer until the job ends or the user types.
- `python -m openreynolds` as an entry point beside the console script.
- `cli.session(..., interface=)`: a seam through which another interface (a web page,
  a test harness) can drive a session using the same `View` the terminal uses.
- `-p` runs exit `1` when the model API would not complete a turn and `2` when
  `--max-wait` ran out with a job still running, so a script can tell the two apart
  from a run that finished.
- `OPENREYNOLDS_CAPTURE=0` switches transcript upload off from the environment; the
  README says up front that transcripts go to the workspace service by default.
- Packaging metadata: project URLs, authorship, keywords, classifiers, a PEP 639
  licence expression, the version read from the package, `video` and `toolbox`
  extras, and an sdist that leaves study mirrors and engineering scaffolding behind.
  CI runs on Windows and Python 3.11 as well.
- `SECURITY.md`, `CONTRIBUTING.md` and this changelog.

### Changed

- Turn-end syncs no longer block the session thread; the mirror is poked instead, so a
  message typed during a long solve is read at once.
- After two model-API failures in a row the session says plainly what is happening and
  how to resume, instead of repeating that the thread is intact.
- The thread sheds the pixels of images the model has already looked at, keeping the
  path and description, so long render-and-look sessions stay a size the API accepts.
- Field notes: steady-versus-transient, the 2D recipe, `job_check wait_s` in place of
  `sleep`, rendering next to the data, and what a steady solve that will not converge
  usually means.
- The design document moved to `docs/design.md`.
- `--rotate y:90` turns an uploaded surface before meshing, because the inlet is
  always -x and geometry does not arrive aligned to anybody's wind tunnel. The case
  is written with the surface that was actually meshed, not the original file --
  copying the original would leave the body somewhere the domain is not, which
  snappy reports as an empty mesh rather than as a mismatch.
- Every case prints its projected area down each axis, and an `--mrf` case objects
  when the axis it was given is not the axis of largest projection: looking down a
  rotor's shaft you see its blade planform, so that projection IS the disc.
  Spinning a propeller about a line lying in its own disc gives a plausible thrust
  beside a torque an order of magnitude too large, which reads as a mesh problem
  and is not one.
- Thermal cases carry an `outletTemperature` function object. Every heat-transfer
  coefficient worth comparing with a correlation is defined on the **log-mean**
  temperature difference, and without the outlet bulk temperature the obvious thing
  to reach for is `T_wall - T_inlet` -- 9% of the answer on a real bundle, in the
  pessimistic direction, with every term in it right except the one nobody writes
  down.
- The MRF zone is sized to **contain** the rotor by default. A cylinder taken from
  the disc radius but a quarter of it thick is thinner than the blades are long
  axially, so the tips sit outside the rotating frame and feel no rotation.
- `--medial-ratio`, `--medial-angle` and `--layer-iter` expose the three controls
  that decide whether a layer stack is BUILT rather than how thick it is, and
  `maxThicknessToMedialRatio` defaults to 0.6 rather than the tutorial 0.3, which
  assumes a chunky body. Worth recording what they cannot fix: the binding
  constraint on a thin hull turned out to be the ratio of first layer to surface
  cell. snappy builds 1:26 and 1:32 happily and refuses 1:118, and no combination
  of these controls changed that.

### Fixed

- `openreynolds doctor` opened a real study on the platform every run to check that
  capture worked. It now makes a read-only call and changes nothing.
- A duplicate copy of the architecture notes at the repository root is gone; the one
  under the toolbox notes is the one that ships.

[Unreleased]: https://github.com/InviscidAI/OpenReynolds/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/InviscidAI/OpenReynolds/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/InviscidAI/OpenReynolds/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/InviscidAI/OpenReynolds/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/InviscidAI/OpenReynolds/releases/tag/v0.1.0
