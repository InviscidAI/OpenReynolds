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

**RESOLVED in part — foamd `1ed92ef`, 2026-09-04.** The capture buffer is now
container-local (`/tmp/foamd-exec`); only a log the response could not carry whole is
copied to `/work/.foamd/exec`, and that directory is pruned inline to a count cap
(`FOAMD_EXEC_LOG_RETENTION`, 200). A sub-cap exec no longer touches the Volume at all,
which is also the fix for issue 9. Verified on the deployed service by gate phase 6.

The `processor*` half is **not** resolved and is now issue 16.

One claim here was wrong and is worth correcting rather than deleting: "every command
returns exit 1 with empty output and the cause is invisible from the client". The exit 1
is right; the invisibility was not this directory's doing. `run_exec` passed
`stderr=StreamType.STDOUT` to `sb.exec`, which does **not** merge stderr into stdout —
Modal routes it to the control-plane container's own log. So the wrapper's own diagnosis
of its own failure went to foamd's logs, unattributed, and never reached the caller. That
is issue 17, and it is what made issue 3 undiagnosable too.

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

**The premise above is wrong, and the real mechanism is fixed — foamd `1ed92ef`,
2026-09-04.** The client never terminates the Sandbox; it calls `/stop`. A client killed
with `SIGKILL` simply never sends it, and the reaper collects the sandbox fifteen minutes
later *with* `wait=True`. A `SIGKILL` therefore **defers** the commit rather than bypassing
it, and the signal handler this entry implies would have changed nothing. There is no
signal handling anywhere in any of the three repos.

What actually put two containers on one Volume, read-write, needing no client
misbehaviour at all:

- `get_or_start` wrapped `Sandbox.from_id(id).poll()` — a live gRPC call — in a bare
  `except: pass` and fell through to creating a second Sandbox. A transient Modal error
  was indistinguishable from "the sandbox is gone". `modal_sandbox_id` was then
  overwritten, so the first Sandbox became an orphan: invisible to the reaper, billing to
  the 24 h ceiling.
- No per-instance lock around lazy start, with `max_inputs=200` admitted concurrently.

**Measured, not argued.** Against real Modal on a throwaway Volume: the old
`get_or_start` turned 4 concurrent first-requests into **4 Sandboxes on one Volume**; the
fix returns 1. Gate phase 5 now asserts this against the deployed service (6 concurrent
requests, one new Sandbox).

Fixed by narrowing the `except` so an ambiguous error fails the request, a per-instance
lock, and an orphan sweep in the reaper. The sweep collected a real orphan on its first
production run on 2026-09-04 while another user's claimed Sandbox ran alongside it,
untouched.

**Still not established:** that this is what corrupted the workspace originally. The
mechanism fits every symptom — reads working, writes failing through the mount, a fresh
Sandbox inheriting it, `modal volume put` still working from outside because it never
touches the mount — but it was not caught in the act. Note also that "deleting the
instance cleared it" is not evidence about the row: `delete_instance` destroys the Volume.

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

**RESOLVED — `InviscidAI/OpenFoam_Instance#2`, closed 2026-09-04.** Modal's edge returns
a bodyless 303 on any web request open past **150 s**; there is no 303 anywhere in the
service. `job_kill` reaches it by cold-booting a Sandbox through `get_or_start`;
`list_instances` reached it by queueing behind the 40-thread starvation. One mechanism,
two causes of slowness — so the note below, that a timeout "cannot explain" the short
requests, was wrong. Both causes were already fixed before this was re-examined.

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

**RESOLVED — foamd `1ed92ef`, 2026-09-04.** See issue 2: the capture buffer left the
Volume, so a sub-cap exec no longer writes there or pays a `mkdir -p` on it before the
command starts.

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

**Split by measurement, 2026-09-04. The parallel half is closed; the `#codeStream` half
is confirmed and is now the only blocker on issue 1.**

**OpenMPI: closed, and it was never an image bug.** `OMPI_ALLOW_RUN_AS_ROOT`,
`OMPI_ALLOW_RUN_AS_ROOT_CONFIRM` and `PMIX_MCA_gds` have been in
`config.SANDBOX_ENV_DEFAULTS` and applied at `Sandbox.create` since `28bc7c2`
(2026-08-30, deployed v10). The refusal was measured through `scripts/gate_live.py`,
which built its **own** Sandbox with no `env=` and therefore inherited none of them —
a container production never ships. The gate now goes through `config.sandbox_env()` and
asserts the env key for key, and `mpirun -np 4 hostname` succeeds **with
`--allow-run-as-root` removed**. `bec67af` had added an `export` of the two vars inside
the gate script the day after the measurement, which patched the symptom and hid exactly
the drift the gate exists to catch; that export is gone.

**`#codeStream`: confirmed, wider than recorded, and unreachable as posed.** Gate phase 4:
`stock cylinder2D: rc=1, cannot complete as root`. The refusal covers **every `coded*`
feature** — `codedFixedValue`, `codedMixed`, `codedFunctionObject` — not just
`#codeStream`, since all route through `dynamicCode::checkSecurity`. Modal offers no uid
knob at all (`Sandbox.create` and `.exec` have no `user`/`uid` in 1.5.4), so "run the
sandbox non-root" is not a configuration that exists. What is reachable is a per-command
privilege drop (`setpriv --reuid`), because `isAdministrator()` tests the calling
process's euid at the moment of the check. The cost is not in the code — two insertion
points and a `useradd` — but in the Volume: `/work` is root-owned on every existing
instance, so it needs a recursive chown over multi-GB case trees on the filesystem issue 9
measured at 150 s for a `tail`, plus mixed ownership in `.foamd` between root
control-plane execs and dropped-privilege user processes.

`plan-1-hosted-openfoam.md` promises "`#codeStream` and user-compiled bits just work" as
an in-scope design property, which argues against closing this as won't-fix.

A side effect worth knowing: **a directory cannot be made unwritable to root**, so
`chmod a-w` is useless as a test fixture here. Gate phase 6 was written that way, passed
for the wrong reason, and had to be rewritten to replace the path with a regular file so
`mkdir -p` fails for everyone.

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



---

## 14. The LLM rates in the README and in the code disagree by 3x

`README.md:208` documents the Opus 5 rates as `1500 / 7500 / 150 / 1875` cents per million
tokens. `config.LLM_DEFAULT_RATES` has `500 / 2500 / 50 / 625`. Sonnet 5 disagrees too:
`300/1500` documented against `200/1000` in code.

These price directly into `monthly_budget_cents`, so one of the two is wrong about real
money. The code comment dates itself "Anthropic list price as of 2026-06"; the README row
still carries a "verify before the first real bill" TODO. Found while auditing the
configured surface, 2026-09-03. Nothing has been changed — the point is that nobody knows
which is right.

---

## 15. `ui`'s README documents a deploy command that unclaims the production domain

`ui/README.md:143` says `modal deploy modal_app.py`. `modal_app.py:39` reads
`APP_PUBLIC_DOMAINS` from the environment with an **empty default**, so that command
deploys with no custom domain claimed and `app.tryreynolds.com` stops resolving to the
app. `deploy.sh` exists precisely to set it — and its own comment records this having
happened to the apex, which served `modal-http: invalid function call` until it was
redeployed correctly.

Nearly repeated on 2026-09-04, following the README line. What stopped it was an unrelated
error: `reynolds_app/static` was not built, so the deploy aborted before it could claim
nothing.

The fix is documentation, not code: the README should point at `deploy.sh`, which also
builds the frontend, typechecks, runs the suite, verifies the built bundle is the one that
ships, and curls the site afterwards.

---

## 16. Nothing distinguishes a finished study's checkpoints from a running one's

Split out of issue 2, which resolved the exec-log half. This is the half that actually
hits the 20 GB quota.

`processor*` directories accumulate on the Volume — a decomposed case writes a full field
copy per subdomain per time directory — and the only tool for removing them is deleting
the instance, which destroys the Volume with them. The quota (`du -sm /work` against
`VOLUME_QUOTA_GB`, checked only at job start in `app/quota.py`) answers with a 507 and no
way to recover.

The missing primitive is a server-side notion of a study being **finished**: `workspace.py`
has a `home` per study and nothing marks one done, so nothing can distinguish restartable
checkpoints from dead ones. The client already knows — `mirror.py` classifies
`processor\d+` as data "nothing in either repo will ever open" — but it is in the wrong
repo to act on it.

---

## 17. An exec's own stderr never reached the caller

**RESOLVED — foamd `1ed92ef`, 2026-09-04.** Recorded because it is the reason two other
issues looked the way they did.

`run_exec` passed `stderr=StreamType.STDOUT` to `sb.exec`. Despite the name that does not
merge stderr into `p.stdout`: Modal routes it to the **control-plane container's own log**,
unattributed to any instance or request, and `run_exec` read only `p.stdout`. `raw_exec`
used `StreamType.DEVNULL` and discarded it outright. So a failure in the wrapper — as
opposed to in the user's command — reached the client as an `rc` and an empty string.

This is why issue 3 presented as an undiagnosable blank, why `jobd` 503s carried only a
return code, and why `PUT /files` 500s were bare. The user's own stderr was never affected:
it is redirected into the log with `2>&1` and comes back in `output`.

Fixed by capturing both channels and attaching stderr to the exec, job-start and
file-write failure paths, uncapped — truncation is how the important half of an error goes
missing.

**Found on the way:** the user command ran in a `{ }` group rather than a subshell, so a
command ending in `exit N` — ordinary in a pasted script — exited the wrapper's own shell,
`head` never ran, and `output` came back empty. Survivable only because the log was on the
Volume to read back, which is exactly what issue 2's fix stops doing for a sub-cap exec.
Now a subshell.

---

## 18. `browse.py` silently truncates the study tree

`Browser`'s default home is `WORKSPACE_ROOT = "/work"` and `browse.py:88` lists it with
`find -H <path> -maxdepth 12 … | head -n 4000`. With 2,993 exec logs under
`/work/.foamd/exec`, three quarters of that budget went to files nobody asked for and the
study tree was cut short — with no indication that anything was missing.

Issue 2's fix removes today's cause. The truncation itself is still silent, and any other
directory that grows will do the same thing. A listing that hits its cap should say so.

---

## 19. `api.version` identifies nothing, so a deploy cannot be matched to a commit

`/health` reports `"version":"0.1.0"` — the string in `FastAPI(title="foamd",
version="0.1.0")`, unchanged since the file was written. Nothing in the deployed service
says which commit it came from.

This is the whole reason issue 7 had to diff the deployed service by matching Modal deploy
timestamps against commit timestamps and comparing generated OpenAPI documents. A
`FOAMD_GIT_SHA` injected at deploy time and reported here would make that a single
request. Deploys carry a `--tag` since v17, which records the commit in `modal app
history` but is not visible from the service itself.

---

## 20. `migrate.py --sql` is not read-only

`scripts/migrate.py`'s docstring lists `--sql` alongside `--status` and `--dry-run`, but
`run()` applies every pending migration first (line 143), clears `pending` (150), and only
then executes the ad-hoc statement (152). So any `--sql` invocation also migrates.

Hit on 2026-09-04: two `--sql` calls made to inspect which tables existed applied
`schema_f13.sql` as a side effect, and the tables they then reported were ones the same
command had just created — evidence that briefly looked like someone had applied the
migration by hand.

No data was harmed and the outcome was wanted a few minutes later, but there is currently
no way to query through this tool without migrating. A `--no-apply` flag, or refusing
`--sql` while anything is pending, would fix it.

---

## 21. A captured transcript can hold the same message more than once — RESOLVED (client side)

**What happened.** In one 3D transient study the captured transcript held a single
`job_check` reply three times, at 04:58:41, 05:00:39 and 05:09:47, all three under
message `seq` 197; five other messages of the same study appeared twice. Read back, it
looked like an agent being handed a stale "the job is still running" answer for
twenty-six minutes after the job had in fact been killed — a correctness failure worth
half an hour of a run.

**It was not one.** The agent produced that reply exactly once, and two facts prove it.
`Store.append_message` increments a counter before it returns, so no two records can
share a seq — three rows at seq 197 are three copies of one record, not three records.
And the reply's own text is a timestamp: `running_for=` and `[waited …s]` are computed
from the clock at the moment the string is built, and all three copies carry the same
values, down to the second the sandbox died.

**Mechanism.** `POST /v1/studies/{id}/messages` was retried at two layers — five
attempts in `FoamdClient.request`, three more around them in `Capture._drain`, so up to
fifteen posts of one message. The service takes `seq` from the client and its `messages`
table indexes `(study_id, seq)` without a unique constraint, so a post that had been
carried out and whose answer was lost (a read timeout, a 502 from the edge) wrote its row
and every repeat wrote it again. The capture worker is also serial, so a repeat storm
holds every later message behind it — and `messages.created_at` defaults to `now()` at
insert, so the transcript is stamped with the uploader's clock, not the agent's. That is
the whole of the apparent stall.

**Fixed here** (`hosted.py`, `capture.py`): a request is repeated only when it is safe
to repeat. Reads keep every retry they had; a write is retried only on the failures that
prove the service did nothing (429, a cold-start 503, a connection never made) and is
handed to the caller on an ambiguous one. `Capture` no longer retries at all, since the
transport below it now knows what this module cannot. `POST .../tar` says
`repeatable=True` for itself, because unpacking the same archive twice is unpacking it
once.

**Still open, on the service side.** Two changes would make the transcript
self-defending rather than merely no longer attacked:

- a unique key on `messages(study_id, seq)`, so a repeat is idempotent instead of
  additive and a client may retry freely again;
- a client-supplied timestamp column, so a row records when the agent said the thing
  rather than when the row happened to be inserted. Without it no transcript timeline
  can be trusted to within the upload backlog, and the backlog is unbounded.

Nothing here reached the model: `Loop._record` posts to capture after the message is
already in the thread, and `_job_check` reads a live status on every call with no cache
anywhere in the path.

## 22. Sign-in cannot be throttled by anything in this project

Found while scoping `ui`'s missing rate limiting, 2026-09-06. Filed here rather than on
GitHub because the fix is a setting in somebody's console, not a change to any repo.

`OpenFoam_Instance#5` lists, among the `ui` items, "no rate limiting in the app at all,
and Turnstile is off by default and never verified server-side". The second half implies
the app could verify it. **It cannot.** There is no sign-in route in `ui`: the browser
talks to Supabase directly, and `web/src/turnstile.ts` holds a *site* key only. Sign-in,
sign-up and password reset never reach this server, so nothing in `ui` — and nothing in
foamd — sees them.

Throttling them, and checking the Turnstile token, are settings on the Supabase project
(Authentication -> Attack Protection), where the matching secret also lives.

**Why it matters.** Rate limiting in `ui` protects spend and guessable codes: session
starts, redemption codes, CLI approvals, checkout. It does not protect the front door.
Anything that reads "the app is rate limited now" as covering sign-in abuse is wrong, and
the deployment story does not say so anywhere.

**What to do.** Two things, neither of them code:

- Set the Turnstile secret and enable Attack Protection on the Supabase project, so the
  token the sign-in card already sends is actually checked. The widget renders and the
  token travels today; nothing verifies it.
- Say so where a deployer will see it. `ui/README.md` is the place, and it is already
  being edited for issue 15.

**Evidence.** `reynolds_app/app.py` route list (no auth route among them);
`reynolds_app/settings.py:55` (`turnstile_site_key`, public, "never this server");
`web/src/turnstile.ts`.

---

## 23. `animate.py` draws its scalar bar on top of the flow, and a comment says it does not

**What happened.** Every non-`--bare` frame from `animate.py` puts the colour bar's title
and tick labels over the bottom of the picture. On the `cavity` tutorial at
`--field velocity` the numbers `0.00 0.250 0.500 0.750 1.00` and the title `U_mag` sit on
the dark fluid, not below it.

**The code believes this was already fixed.** `animate.py:715` carries a comment — "Under
the picture, not across it: with a parallel projection filling the frame, pyvista's
default put the tick labels on top of the flow" — above an `add_scalar_bar` call moved to
`position_y=0.02, height=0.06`. That moved the bar out of the *middle* of the frame. It
reserved no space, so the picture still reaches the bottom edge and the two occupy the
same band.

**Root cause, confirmed by reading both ends.** `camera_setup` (`animate.py:502`) fits the
slice to the full viewport with `padding=0.05`, and the bar is drawn into that same
viewport at y = 0.02–0.08. Nothing subtracts one from the other. The fix is to shrink the
renderer's viewport to the top ~88% and let the bar own the strip beneath it, rather than
repositioning the bar inside a full-bleed picture.

**Why it hid.** The collision needs a slice that fills the frame vertically. `cavity` is
square in a 4:3 window, so the fit is height-limited and the flow runs edge to edge. A
wide slice — the cylinder wake most of the corpus renders — is width-limited, leaves
letterbox at the bottom, and the bar lands in it looking correct.

**Why it matters.** This is the engineering path. `--bare` is an opt-out taken by the
content pipeline, not a fix: anyone rendering to read a number off the picture wants the
bar, and gets it printed over the data. The stale comment is the part worth keeping in
mind: the next person reads line 715, believes the problem is handled, and never opens a
frame.

**Also worth a pass, but cosmetic and not a defect.** The burned label is VTK's default
bitmap font at `font_size=10` with no outline, pinned to `position="upper_left"`. Legible,
ugly, and carrying no correctness claim.

**What to do.** Shrink the viewport, then look at a cavity render — the only check that
would have caught this. A unit test cannot see the overlap; the guard that fits is an
assertion that the bar's band and the camera's band do not intersect.

**Evidence.** Reproducible in three commands, which is cheaper than storing the frames
(the ones this was found from were under `/tmp` and are gone):

    blockMesh -case <case> && icoFoam -case <case>   # tutorials/incompressible/icoFoam/cavity/cavity
    python openreynolds/toolbox/animate.py <case> --field velocity --clim-from all --out /tmp/f
    # then look at /tmp/f/frame_0005.png

Found while adding `--bare` (commit `e8df53f`), which is why the flag's own frames are
clean and this was not.

---

## 24. `LocalBackend` jobs die with the session, and the interface says they do not

**What happened.** Stopping a `-p` study on the local backend killed both running
`pimpleFoam` jobs. Their logs stop mid-timestep — no `End`, no fatal error, just a
truncated residual block — and `pgrep pimpleFoam` returns nothing.

**Why it matters.** The interface promises the opposite, in two places. The session panel
is headed "jobs keep running if you leave", and the refusal path prints "Your work is safe
-- every job is still running on the instance and every file is mirrored here"
(`cli.py:2250`). Both are true of `HostedBackend`, where the job lives in a container that
outlives the client. Neither is true locally, where the job is a subprocess of the session
that started it.

It cost four minutes here because the solve was young. The same stop against a
three-hour run loses three hours, and the message on screen while it happens says the work
is safe.

**Where to look.** `backend/local.py:254` (`job_start`) launches into the session's own
process group. The docstring at the top of the file is careful about the two things local
deliberately does not reproduce — no sandbox, and a real directory instead of `/work` —
and job lifetime is a third that is not named.

**What to do.** Either detach local jobs (new session/process group, so a dying client does
not take them), or make the "jobs keep running" copy backend-aware. The second is smaller
and honest; the first is what makes `--study` resume mean the same thing on both backends.

**Evidence.** Study `20260915-073602-3917`, the third and fourth `-p` runs against it.
`room_fan_in_large/log.pimpleFoam` and `room_fan_out_large/log.pimpleFoam` end mid-iteration
at the moment the client was stopped.

---

## 25. The reasoning-effort degrade path retries with a variant the API also rejects

**What happened.** `gpt-5.6-sol` cannot be run at all. Every turn comes back as

    Function tools with reasoning_effort are not supported for gpt-5.6-sol in
    /v1/chat/completions. To use function tools, use /v1/responses or set
    reasoning_effort to 'none'.

and the study stops with "That is a refusal, not a hiccup".

**Why the existing fallback does not save it.** `openai_api.py:177-189` already handles this
shape: on a `BadRequestError` it sets `self.lean = True` and retries, and `lean` means *omit
`reasoning_effort`*. But omission is itself one of the rejected cases. Measured against the
live API, for both `gpt-5.6-sol` and `gpt-5.6-terra`:

| `reasoning_effort` | result |
|---|---|
| omitted | 400 |
| `"none"` | **accepted, tool call returned** |
| `low` / `medium` / `high` | 400 |

So the retry spends its attempts on a variant that fails the same way, and the turn is
reported as a refusal. The one accepted value is the one the error message names, and the
adapter never sends it.

**How far it reaches.** Every OpenAI model above `gpt-5.2` refuses tools beside a reasoning
effort on Chat Completions — measured: `5.6-sol`, `5.6-terra`, `5.6-luna`, `5.5`, `5.4` all
refuse; `5.2`, `5.1` and `gpt-5` accept. Commit `4e5a058` adds the Responses API, which is
the real fix for OpenAI's own models. **This issue is the remaining half:** the Chat
Completions degrade path is still wrong for any other endpoint that returns this error
shape, and a vendor imitating the API will hit it.

**What to do.** In that `elif`, distinguish "this endpoint does not know the field" (omit,
what `lean` does today) from "this endpoint wants the field set to `none`". One extra flag
beside `lean`.

**Evidence.** Study `20260915-073602-3917`, first `-p` run. Reproducible with a three-line
`chat.completions.create` carrying one function tool, against any model above `gpt-5.2`.

---

## 26. The agent says a case is settled without looking, and says so in the same sentence

**What happened.** Asked for two transient cases run "far enough for the room-scale flow
pattern to establish", the agent solved both to a fixed `endTime 30` and reported:

> The 30 s duration is roughly three room-volume exchanges for this setup. **Room-scale flow
> is established**, but no formal statistical-stationarity or mesh-convergence assessment was
> performed.

The two clauses contradict each other. The first is a claim about the flow; the second says
the work that would support it was not done. Across 51 tool calls it never sampled a field
at two times.

It was wrong. Measured afterwards on the z-midplane, the exhaust case's median speed ran
1.88 → 4.04 → 5.51 → 5.94 m/s at t = 2, 10, 20, 30 and was still climbing at the end, from a
3 m/s fan that moves ~3.1 m³/s through a 7.5 m² room section — a flow that cannot be. The
inflow case was genuinely steady from t ≈ 5. So a settled case and an unsettled one were
handed over as a matched pair, which is the one thing a side-by-side must not be.

**Why it matters.** The check that falsified it is fifteen lines and reads the solver's own
`postProcessing/` output — no reconstruction, no re-solve, on data already on disk. This is
the same shape as issue **1**, where the agent read "no shedding yet" as "not long enough"
and extended `endTime` fivefold without asking whether the case *could* shed. Both are
convergence claims made from a plausibility argument rather than from data in the workspace.

**The large confound, stated up front.** This run was at `reasoning_effort=none`, forced by
issue **25** — the model could not be given tools and reasoning at the same time. With
reasoning on, the same model on the same subject later resolved a **0.3%** secular drift
between t = 40 and 45, distinguished a plateau from a pause before a rise, and fitted
windowed slopes unprompted. That is two orders of magnitude finer than the error it missed
here.

**So this is not yet filed against the agent.** It is filed against a configuration, and the
control that separates the two has not been run: the original brief, unchanged, on
`gpt-5.6-sol` through the Responses API with effort high. Until that exists, issue 1's root
cause should also be treated as open in the same way — it may be substantially cheaper to
fix than its text assumes.

**Evidence.** Study `20260915-073602-3917`, `seq62`. The falsifying measurement and the
settling script are in `ContentPipeline/cases/01-window-fan-in-vs-out/scripts/settling.py`.

---

## 27. An irreversible delete gated on a job that had not finished

**What happened.** Planning to continue a pair of solves from t = 180 to 300 and keep only
the settled tail, the agent freed disk first:

    python3 - <<'PY'
    for case in ['room_fan_in_large','room_fan_out_large']:
      ... if 0 < t < 180: shutil.rmtree(d)
    PY
    room_fan_in_large removed 359 old write directories
    room_fan_out_large removed 359 old write directories

718 write directories, ~14 GB, gone. It then started the follow-on job. That job did not
complete — it was stopped nine seconds in, for reasons unrelated to the purge — and the
result is that roughly 180 s of solved flow, about an hour of compute, exists nowhere.

**Why it matters, and why it is not simply a mistake.** The reasoning was sound and was
stated: the startup transient is not wanted, the settled window will come from the new run,
the disk is needed. Every step of that is true *if the new run finishes*. The defect is
ordering, not judgement — the data was destroyed before the thing that justified destroying
it existed.

Nothing in the loop distinguishes "disk I no longer need" from "disk I will need if the next
step fails". A job that has been started is treated as a job that has succeeded.

**This one had reasoning on**, at effort high, through the Responses API. It is not a
symptom of issue 26's configuration.

**What to do.** This cannot be a harness rule — the harness may not enforce an order of work
(`README.md`, and `tests/test_prompt.py` fails the build on imperative prompt language). The
available levers are a toolbox affordance that makes the safe order the easy one (prune
*after* a job reports success, given its id), or a note in the field notes. Worth saying
plainly that the cheap version is: delete old writes once the new ones exist, never before.

**Evidence.** Study `20260915-073602-3917`, `seq231` (the `shutil.rmtree` call and its
output).

---

## 28. Run length expands without anything pricing it

**What happened.** Asked for two cases run long enough to be statistically steady, with "at
least 60 write directories" each, the agent chose `endTime 180` with `writeInterval 0.5`:
**360 writes per case**, six times the ask, ~17 GB for the pair. It then proposed extending
to `endTime 300`.

The justification was real — it had found a drift in the exhaust case between t = 40 and 45
and wanted it flat. But the drift was a slow ~1–2% oscillation whose sign flips (−0.81% per
10 s over t = 60–120, +0.90% over t = 120–180, mean moving 0.802 → 0.806 across the whole
120 s). A longer run samples more of that oscillation; it does not make it stop. The
extension would have bought ~1% of precision for ~50 minutes of solver time and ~8 GB.

**Why it matters.** This is issue **1** in its second form. There it was `endTime` extended
fivefold on a case that could not shed; here it is `endTime` at 6× the request plus a
proposal to double it again. In both, the decision to spend is made entirely on physical
grounds, with nothing on the other side of the scale. `endTime` has dominated the wall clock
of every study measured, and the agent has no notion that a run has a budget.

**Reasoning was on** for this one, at effort high. Reasoning improved the *diagnosis* — the
drift it was chasing was real and finely resolved — and did not supply a cost side at all.

**What to do.** Not a checklist. The honest options are to put the budget in the user's own
standing preferences (`preferences.md`, relayed verbatim, which is the sanctioned channel
for "I care about cost"), or to make the cost visible in the situation the model already
sees — elapsed solver time and disk written, beside the residuals it is already reading.

**Evidence.** Study `20260915-073602-3917`, the fourth `-p` run: `controlDict` at
`endTime 180`, `seq231`–`seq236`, and the settling history in the case's `postProcessing/`.

---

## 29. The disk walker counts a symlink's target twice

**What happened.** `tests/test_toolbox_disk.py::test_a_symlink_is_not_followed_and_not_counted_twice`
fails on a clean checkout:

    assert usages[0].bytes < 10 * plain, "the link's target was counted again"
    AssertionError: assert 31524 < (10 * 2000)

**Why it matters.** It is a red test on `main`, so it erodes the signal from the suite — the
rest of which passes. And the behaviour it guards is a real one: a study whose
tree contains a symlink into another study has its size reported as the sum of both, which
is the number a cleanup decision would be made from.

**Not investigated.** Found while running the full suite to check something unrelated
(`4e5a058`, which touches only `openreynolds/llm/`). Recorded rather than chased, per the
note at the top of this file.

**Evidence.** `python -m pytest tests/test_toolbox_disk.py -q` on `worktree-render-seam` at
`9dec4d4`; the walker is `openreynolds/toolbox/disk.py`, last touched by `df8507c`.


---

## 30. `doctor` crashes in its own cleanup, on the machine that needs it most

`openreynolds doctor` is the command someone runs before their first session, and when the
hosted service is unreachable it does not report that — it raises:

    File "openreynolds/cli.py", line 791, in _check_service
        client.close()
    File "openreynolds/backend/hosted.py", line 531, in close
        self.kernel_stop()
    AttributeError: 'FoamdClient' object has no attribute 'kernel_stop'

`HostedBackend.close()` calls `self.kernel_stop()`, which `FoamdClient` does not define. So
the failure is not in the check, it is in tearing the check down — and it fires regardless
of what the check found.

**Why it matters.** Three of the things `doctor` exists to distinguish — a missing key, an
unreachable service, a working setup — all arrive as the same `AttributeError` traceback,
and a traceback out of a diagnostic reads as "this tool is broken" rather than "your
service is unreachable". It also lands on the first-run path, where the person has the
least context for interpreting it. The same `close()` is on the ordinary session teardown
path, so it is probably not only `doctor` that hits it.

**Not investigated.** Found while checking whether the local backend was configured, on a
machine deliberately running `OPENREYNOLDS_LOCAL=1`, where the hosted service is not
expected to answer at all. Whether `kernel_stop` was renamed or never existed on this
client is not established.

**Evidence.** `openreynolds doctor` on `worktree-render-seam` at `208689a`, with
`OPENREYNOLDS_LOCAL=1` set.

---

## 31. The workspace path the prompt advertises does not work inside a shell command

`LocalBackend._resolve` treats `/work` as an alias for the real workspace root, so every
*tool argument* — a `cwd`, a path handed to `read_file`, the toolbox destination — lands in
the right place. A path written inside a `bash` command string gets no such treatment: the
shell resolves it against the machine's actual filesystem.

So the path the frozen prompt tells the model to use is correct everywhere except the tool
the model uses most.

    pwd                  ->  /home/.../ContentPipeline/work/20260915-073602-3917
    ls /work             ->  /work            (a dangling symlink)
    ls .                 ->  the case files

**What it cost.** An agent resumed onto a study whose data was entirely intact opened with
`find /work -maxdepth 3 -type d`, got nothing, concluded the workspace was empty, and gave
up after two steps. Nothing was wrong with the study; 4 GB of solved cases sat one
directory away.

**Why it matters more than the lost run.** On this machine `/work` happened to be a
*dangling* symlink left months earlier by an unrelated session, so the failure was loud and
empty. Had it pointed anywhere live, the same command would have listed — and a write would
have written — a directory belonging to something else, while every tool-level path
continued to resolve correctly. A silent wrong-directory write is the bad version of this
bug, and nothing in the current design prevents it.

**Options, none chosen.** Creating `/work` needs root, which the module's own header note
says is why the alias exists. Rewriting `/work` inside arbitrary command text is worse than
the disease. The honest fix is probably for a local backend to tell the model its real
root rather than advertise one it cannot provide, which means the workspace path stops
being a constant the prompt can hard-code.

**Evidence.** `worktree-render-seam` at `95228fd`, `OPENREYNOLDS_LOCAL=1` with
`OPENREYNOLDS_LOCAL_WORK` set; reproduced directly through `LocalBackend.exec`. The alias
is `openreynolds/backend/local.py:128-151`.
