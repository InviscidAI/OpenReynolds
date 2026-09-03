# Open issues

Things found while running the product that are real, reproducible enough to write
down, and *not* what was being worked on at the time. The point of this file is to stop
each discovery from becoming the new task: note it, keep the evidence, carry on.

`found-by-using-it.md` is the companion narrative — what running it found and what was
then fixed. This is the queue of what has not been.

An entry that has been fixed is marked **RESOLVED** with the version that carries it and
kept in place for one cycle, rather than deleted: the evidence is most useful to the next
person while the fix is still fresh enough to be doubted.

**A note on evidence.** `studies/` is gitignored, so the transcripts referenced below
live only on the machine that ran them. Anything that matters long-term should be copied
out before it is lost. Session logs under `/tmp` do not survive a reboot at all.

---

## 1. No shedding develops, and the agent runs longer instead of diagnosing

**What happened.** A seeded 2D cylinder at Re=100 reached `t=6000` — roughly 107
shedding periods at the agent's own estimate of a 56 s period — with no shedding in the
lift signal. The agent read that as "not long enough" and extended `endTime` five-fold
to 30,000, spending 24 minutes of solver time on the extension. It never questioned
whether the case *could* shed.

**Why it matters.** `endTime` has dominated the wall clock of every study measured. If
"no oscillation yet" reliably produces "run longer", every such case costs multiples of
what it should, and the run that finishes may still be a well-converged wrong answer.
The field notes already warn about exactly this shape (`openfoam-field-notes.md`, the
steady-solver-on-unsteady-flow trap) — the agent does not read them.

**Root cause: undetermined.** The obvious hypothesis was numerical dissipation, and the
evidence does not support it: `divSchemes default Gauss linear` is second-order central,
the least dissipative choice available. Remaining candidates, unmeasured:

- `ddtSchemes default Euler` (first-order in time) combined with `adjustTimeStep` at
  `maxCo 0.8` — large steps plus first-order time may damp the growth.
- Wake mesh too coarse to sustain the instability.
- The `setFields` perturbation decaying before it can grow; the seed was applied in the
  right order (`0.orig` restored first, then `setFields`) but its magnitude and location
  were never checked against the shear layer.
- The shedding did develop and the agent misread its own lift signal.

**Evidence.** Study `20260902-073526-be36`.
- Reasoning: session log lines 96, 103 (the `endTime` budget), 419 (the decision to
  extend), 574/583 (still waiting).
- `files/*/case/log.pimpleFoam` — 12,012 steps to t=6000, 356 s.
- `files/*/case/log.pimpleFoam2` — 48,000 steps to t=30,000, 1,419 s.
- `files/*/case/log.setFields`, `system/fvSchemes`, `system/controlDict`.

**The control this issue proposed cannot be run.** The plan was to run the stock
`incompressible/pimpleFoam/laminar/cylinder2D` tutorial unmodified and see whether *it*
sheds, separating "the agent builds a case that cannot shed" from "something about this
OpenFOAM build". Attempted 2026-09-02 (`scripts/gate_live.py`, phase 3); it does not
reach a solve, for three reasons that are nothing to do with shedding:

- The 2512 tutorial of that name is a **12-subdomain reduced-order-model case**
  (`mirrorMesh`, `redistributePar`, `createROMfields`, two `pimpleFoam` passes), not the
  small laminar case the name suggests. On a 4-core workspace it fails on width alone.
- Narrowing `numberOfSubdomains` is not sufficient: `simpleCoeffs (4 3 1)` remains, and
  `decomposePar` exits on `Wrong number of domain divisions in geomDecomp`.
- Its `system/blockMeshDict.main` uses `#codeStream`, which is refused outright while the
  sandbox runs as root (issue 12).

So the question this issue exists to answer — does a cylinder shed on this image — is
still open, and now has three obstacles in front of any stock-tutorial control.

---

## 2. `/work/.foamd/exec` grows without bound

One log file per `bash` call, forever; nothing prunes it. Reached 2,993 files in about a
day of heavy use. Every `exec` writes here *before* returning anything
(`sandboxes.run_exec`), so if this directory ever becomes unwritable, every command
returns exit 1 with empty output and the cause is invisible from the client.

Related: nothing distinguishes "checkpoints a study might restart from" from
"checkpoints of a study that finished weeks ago". `processor*` directories accumulate on
the volume and the only tool for removing them is deleting the instance, which destroys
the volume with them.

---

## 3. An unclean client shutdown can corrupt the workspace mount

`sandboxes.terminate` passes `wait=True` specifically so the Volume commit completes.
A client killed with `SIGKILL` mid-write bypasses that. After roughly a dozen such kills
the container could read `/work` and not write to it — `PUT /v1/instances/{id}/files`
returned 500, every `exec` returned exit 1 with no output, and `jobd` 503'd. The Volume
itself was fine throughout: `modal volume put` wrote to it happily from outside, and a
fresh sandbox inherited the same fault. Deleting and recreating the instance cleared it.

Worth confirming whether this is reachable without `SIGKILL` — if an ordinary crash or a
Modal preemption can do it, it is a data-availability bug rather than an own-goal.

---

## 4. `303` with an empty body instead of an error

**TRACKED — `InviscidAI/OpenFoam_Instance#2`, 2026-09-03.** Merged into that issue,
which is now the single home for it. The evidence below widened its scope: that issue
framed the 303 as a synchronous request outliving the web-endpoint window, which cannot
explain `job_kill` or `list_instances`, neither of which is long-running.

Seen on `job_kill`, on `list_instances`, and on an `exec` whose `timeout_s` exceeded the
300 s the protocol advertises. The client surfaces
`bad_response (303): the service answered 303 with a body that is not JSON`, which is
accurate and useless. A request over the advertised ceiling is answered with neither the
ceiling nor an error a client can act on.

---

## 5. The agent writes its own busy-wait jobs

Observed: `job_start "while ps -p 5619 > /dev/null; do sleep 5; done"`, then polling
*that* job — 23.6 minutes of wall clock in one study. `job_check`'s `wait_s` does the
same thing for free and is described in the tool. Study: the `h2` run of the hosted
series, 2026-08-31.

---

## 6. The agent will not abandon a stalled job

A `snappyHexMesh` that had made no progress for 40 minutes was left running on an
explicit sunk-cost argument: *"rather than killing it and restarting with a fixed
parallel setup, since that would lose progress already made."* Its own diagnosis of the
stall was wrong — it blamed smoothing, which the log timed at 0.52 s.

---

## 7. Deployed service is behind the checkout

**Closed 2026-09-02, except the env layer, which remains undiffable.** The image-only
commit that was outstanding (`c17558c`, poppler-utils) deployed as v17 at 14:51 UTC, and
that deploy is the first to carry a `--tag`, so `modal app history` now records the
commit it came from. What is written below is kept because the env-layer half is still
open and because the method is worth having on record.

The original report: the deployed foamd treated `monthly_budget_cents = 0` as a hard
zero while the checkout treated it as unlimited (`config.budget_is_unlimited`), so
setting the documented staff value locked the account out rather than freeing it.
Adjacent: `budget_for()` did not give an `inviscidai.com` account the staff budget at
signup, so the founder's account was created with the free-plan cap.

**Both are resolved in the deployed code.** `28bc7c2` ("A new account starts with no
credit, and zero stops meaning unlimited") inverted the semantics deliberately —
`budget_is_unlimited` is now `cents is None or int(cents) < 0`, and `0` means zero
allowance and is refused. `config.budget_for()` returns `staff_budget_cents()` (default
`-1`, unlimited) for any domain in `FOAMD_STAFF_DOMAINS` (default `inviscidai.com`).
That commit landed 2026-08-30 17:30 UTC and deployed as v10 at 18:21 UTC.

**How the diff was done, since `modal app history` records no commit.** Deploy times
were matched against commit times (both UTC), then confirmed against the running
service: the generated OpenAPI document from the checkout is **byte-identical** to
`https://api.tryreynolds.com/openapi.json` — same 37 paths, 47 operations, all 14
component schemas equal. The API contract has not drifted.

**What is still out of sync.** Deploy v16 is 2026-09-01 18:51 UTC; the checkout is
`c17558c` at 19:05 UTC. The single undeployed commit is `poppler-utils in the workspace
image`, which touches `image/image.py` only — so the *control plane* is current and the
*workspace image* is not. `modal_app.py` builds the image at deploy time, so
`pdftoppm`/`pdftotext` are absent from every sandbox until the next `modal deploy`. Any
session that re-renders a region of an uploaded blueprint at higher DPI fails today.

**What could not be diffed.** Everything the running container reads from the
`foamd-secrets` Secret — `FOAMD_STAFF_DOMAINS`, the `Caps` values
(`FOAMD_DEFAULT_CPU`/`_MEM_GB`, `FOAMD_MAX_CPU`/`_MEM_GB`), the rate limits, the tariff.
`modal secret list` shows the Secret's name and nothing of its contents, so code parity
does not prove behaviour parity. `/health` already exposes `signups` and `llm` for
exactly this reason ("how an operator confirms a secret edit actually reached the running
container"), but the resolved caps and tariff are not among what it reports, so that part
of the configured surface cannot be checked from outside at all.

**Noted while looking.** `/health` reports `"signups":"open"` on production today, while
`OpenFoam_Instance#5` item 1 — the fleet-wide 10/min key-minting throttle — is filed as a
blocker *before* open signup.

**What remains open.** Only the env layer: nothing outside the running container can say
what `FOAMD_STAFF_DOMAINS`, the `Caps` values, the rate limits or the tariff resolve to,
so code parity still does not prove behaviour parity.

---

## 8. `imageio` is not in the image, so every study pays to install it

**RESOLVED — foamd v18 (`bec67af`), 2026-09-03.** Verified in a sandbox booted from
the deployed image with egress blocked: `imageio` 2.37.4, its ffmpeg plugin 0.6.0,
and `ffprobe` 6.1.1 on PATH.

The container does not have it, and `pyproject.toml` lists it only as an optional `video`
extra. The agent therefore runs `pip install imageio` from the network in study after
study, and retries when it is slow — **2 to 5 minutes per study, 17.7 minutes across the
five traced runs that hit it**, sometimes 4 calls in one study.

**Correction to the original report, 2026-09-02.** "Animations need it" named the wrong
consumer. `toolbox/animate.py` does not import `imageio` and does not shell out to
anything: it renders PNG frames plus `frames.json` and states that the encode happens on
the user's machine. What was being paid for is the *ad-hoc* encode the agent does on the
instance anyway, having decided it wants a video. The measured cost is unchanged.

**Two ways this bites that a bare `pip install imageio` would not have settled.** The
sandbox has no egress, so anything fetched at *first use* rather than install time fails
just as hard; and `.mp4` resolves to imageio's `FFMPEG` plugin, which bare `imageio`
cannot even import — so an image carrying only `imageio` passes an import check and
still cannot write a video.

---

## 9. A busy volume makes trivial commands cost minutes

`sandboxes.run_exec` writes each command's output to `/work/.foamd/exec/<id>.log` before
returning any of it. So when the workspace is under load, the cost lands on *every*
command, including the ones used to check on the load.

Measured in one study: `tail -5 log.reconstruct3` and `ls | grep` took **150 s each**
while repeated `reconstructPar` jobs were saturating the volume. That study spent 31.9
minutes in `bash` against 310 s of actual solver time. `reconstructPar` is the worst
case for this filesystem — 11x slower there than on container-local disk, 87% of it
system time — so the shell is slowest exactly when someone is trying to find out why.

Shell responsiveness is therefore coupled to solver I/O, on the same volume, with no
separation between the two. Related to issue 2, which is the same directory considered
from the other end.

---

## 10. `jobd start` reuses a job directory without clearing its terminal state

**RESOLVED — foamd v18 (`bec67af`), 2026-09-03.** Verified against the deployed
image: a first run exits 3; one second into a second run started in the same
directory the exit code is absent rather than a stale 3; the second run then
finishes with its own 0.

`cmd_start` does `mkdir -p "$dir"` and truncates `log`, and leaves `rc` and `killed_by`
alone. So starting a job into a directory that has been used before leaves the *previous*
run's exit code sitting next to a log that was just emptied. A caller that polls for
completion reads the old `rc` within milliseconds of starting and concludes the job is
already over, with whatever status the last one ended in.

The API path does not reach this: `app/jobs.py:176` mints a fresh `uuid4()` per job, so
no two API-created jobs share a directory. The exposure is the free-will path — the agent
has a shell, `jobd` is described to it, and `jobd start --dir /work/.foamd/jobs/mesh`
twice is an ordinary thing to type.

**Evidence.** Found 2026-09-02 by `scripts/gate_live.py` reusing
`/work/.foamd/jobs/gate-motorbike` across runs on a persistent volume: run 3 read run 2's
`rc=1` and reported a failed solve while `snappyHexMesh` was in fact still running and
went on to finish. The two pieces of evidence actively contradicted each other — a fresh,
growing log beside a stale terminal exit code — which is what made it convincing.

**A near neighbour, still open.** `jobd` builds its command with `$*`, so a quoted
fragment does not survive: `jobd start --dir D -- sh -c "exit 3"` arrives as
`sh -c exit 3` and returns 0. Found while verifying the fix above — the first attempt
reported success on evidence that showed nothing. Same shape as issue 11, and it will
mislead an agent the same way.

---

## 11. Tutorial shell functions do not survive a detached `jobd` job

`restore0Dir`, `runApplication`, `runParallel` and `getApplication` are **shell functions**
defined in `$WM_PROJECT_DIR/bin/tools/RunFunctions`, which every OpenFOAM tutorial's
`Allrun` sources at the top. Exported environment variables reach a job started by `jobd`
(which is why `snappyHexMesh`, `topoSet` and `potentialFoam` resolve normally), but shell
functions do not survive its `setsid sh -c`.

So a case built by copying a tutorial and following its idiom fails with `127` the moment
the same commands are run as a detached job rather than interactively. `127` reads as "the
binary is not installed", which points a diagnosis at the image rather than at the shell.

This sits directly across the path the product steers agents toward: long work is supposed
to go through `job_start` rather than a hand-rolled busy wait (issue 5), and the tutorials
the agent is pointed at are written in an idiom that breaks there.

**Evidence.** `scripts/gate_live.py` run 5, 2026-09-02:
`/work/gate/solve.sh: line 29: restore0Dir: command not found`, `rc=127`, after
`decomposePar` and a parallel `snappyHexMesh` in the same script had both run fine.

---

## 12. The sandbox runs as root, and OpenFOAM refuses two things because of it

`euid` is 0 inside every workspace. Two separate refusals follow, with different
mechanisms and different blast radii.

**`#codeStream` is refused outright.** `dynamicCode::checkSecurity` holds two independent
guards: the `allowSystemOperations` switch, and a separate, unconditional
`isAdministrator()` test. Setting `FOAM_ALLOW_SYSTEM_OPERATIONS=1` satisfies the first and
does nothing for the second (measured 2026-09-02, not inferred). So any dictionary using
`#codeStream` fails with *"This code should not be executed by someone with administrator
rights for security reasons. It generates a shared library which is loaded using dlopen"*
— at `blockMesh`, before any solver is reached. Nothing in the environment changes this
while the process is root.

**OpenMPI refuses to launch as root.** `mpirun has detected an attempt to run as root`,
unless `--allow-run-as-root` is passed or `OMPI_ALLOW_RUN_AS_ROOT` and
`OMPI_ALLOW_RUN_AS_ROOT_CONFIRM` are set. The tutorials' own `runParallel` passes neither,
so **every stock `Allrun` that runs in parallel fails on this image**. Confirmed
2026-09-02 that setting both variables clears this one.

**Why it matters beyond the individual case.** `OpenReynolds#1` treats `$FOAM_TUTORIALS`
as a corpus of known-running seed cases the agent can start from and adapt. A slice of
that corpus does not run here at all, and the messages it fails with name permissions and
administrator rights rather than anything the agent chose — so the natural reading is that
the agent did something wrong, or that the image is broken.

**Evidence.** `scripts/gate_live.py` phase 3 and
`studies/`-external diagnostics, 2026-09-02: `cylinder2D` reaches neither a mesh nor a
solve; `log.blockMesh.main` carries the administrator-rights fatal, `log.pimpleFoam.*` the
mpirun one.

---

## 13. The `request_log` middleware blocks the event loop

**RESOLVED — foamd v18 (`bec67af`), 2026-09-03.**

`app/audit.py`'s `request_log_middleware` is `async def` and calls `supa.insert(...)`
synchronously, on every authenticated request, after the handler returns. That is a
network round trip taken *on the event loop*, so it serializes every other request behind
it — `/health` included — no matter how the thread pool is sized. It never reaches a
thread, so pool sizing and per-route concurrency limits cannot help it.

It is invisible to the test suite because `dependency_overrides[require_key]` skips the
`request.state.principal` assignment that arms the write.


