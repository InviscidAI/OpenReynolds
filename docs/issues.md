# Open issues

Things found while running the product that are real, reproducible enough to write
down, and *not* what was being worked on at the time. The point of this file is to stop
each discovery from becoming the new task: note it, keep the evidence, carry on.

`found-by-using-it.md` is the companion narrative — what running it found and what was
then fixed. This is the queue of what has not been.

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

**Cheapest next step.** Run the stock `incompressible/pimpleFoam/laminar/cylinder2D`
tutorial unmodified and see whether *it* sheds on this image. That separates "the agent
builds a case that cannot shed" from "something about this OpenFOAM build or mesh".

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

Seen on `job_kill`, on `list_instances`, and on an `exec` whose `timeout_s` exceeded the
300 s the protocol advertises. The client surfaces
`bad_response (303): the service answered 303 with a body that is not JSON`, which is
accurate and useless. A request over the advertised ceiling should be a clean 400.

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

**Diffed 2026-09-02. The control-plane half is closed; one image-only commit is still
undeployed, and the env layer remains undiffable.**

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
container"); extending it with the resolved caps and tariff would make the whole
configured surface diffable from outside.

**Noted while looking.** `/health` reports `"signups":"open"` on production today, while
`OpenFoam_Instance#5` item 1 — the fleet-wide 10/min key-minting throttle — is filed as a
blocker *before* open signup.

**Cheapest fix for the recurrence.** `modal deploy` takes `--tag`; deploying with
`--tag $(git rev-parse --short HEAD)` fills the empty `Commit` column in
`modal app history` and turns this whole exercise into one command.

---

## 8. `imageio` is not in the image, so every study pays to install it

Animations need it; the container does not have it. `pyproject.toml` lists it only as an
optional `video` extra. The agent therefore runs `pip install imageio` from the network
in study after study, and retries when it is slow — **2 to 5 minutes per study, 17.7
minutes across the five traced runs that hit it**, sometimes 4 calls in one study.

The cheapest fix in this file: put `imageio` (and a real `ffmpeg`) in the image. No agent
behaviour has to change.

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

Writing the log to container-local disk, keeping the reported `log_path` shape, would
decouple shell responsiveness from solver I/O. Related to issue 2, which is the same
directory considered from the other end.
