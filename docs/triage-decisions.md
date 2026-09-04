# Triage decisions, wave of 2026-09-04

Issue-by-issue decisions taken after the six read-only probes of 2026-09-03/04.
Each entry: what the issue actually is (where that differs from how it was filed),
the agreed fix, and the acceptance test. Nothing here has been implemented.

A note on the probes: four of the six items were mispriced in the original triage,
and two filed issues rest on a premise the code does not support. Where that is the
case it is said explicitly, because the filed text is still what a reader finds first.

---

## 1. Exec stderr is discarded

**What it is.** `sandboxes.py` passes `stderr=StreamType.STDOUT` to `sb.exec`. Despite
the name that does not merge stderr into `p.stdout`: Modal routes it to the foamd
control-plane container's own log, unattributed to any instance or request, and
`run_exec` reads only `p.stdout`. `raw_exec` uses `StreamType.DEVNULL` and discards it
outright.

So a failure in the *wrapper* — as opposed to in the user's command — reaches the client
as an `rc` and an empty string. This is why local issue 3 presented as an undiagnosable
blank, why `jobd` 503s carry only a return code, and why `PUT /files` 500s bare.

**Not affected:** the agent's own stderr. The user command is wrapped in a group
redirected into the log with `2>&1`, so it is already captured in-band and returned in
the 64 KB output. The channel being discarded is normally silent.

**Fix.** Capture stderr separately (`StreamType.PIPE` on both channels) and attach it to
the failure paths: `ExecResult` when `rc != 0`, `start_job`'s `unavailable(...)`, and
`write_file_chunks`'s `RuntimeError`. Restore it for `raw_exec`'s callers, which
currently fail mute.

**No cap on the captured stderr.** Truncation is how the important part of an error goes
missing. Noted while deciding this: `run_exec` already does `head -c 65537` on the log,
so a failing command whose diagnosis is at the *end* of a long output gets exactly the
wrong 64 KB. Different code path, same mistake — worth its own look.

**Acceptance test.** Unit: fake `sb.exec` returning distinct stdout/stderr; assert the
stderr text reaches the failure. Live, on a throwaway instance: `chmod a-w
/work/.foamd/exec`, run an exec, assert the failure names the path. Recoverable by
construction — `start_job` shells `jobd` directly and never touches that dir, so
`chmod u+w` always works.

**Not in the permanent suite:** the "successful output is byte-identical to today" check.
It is a null test; run it once during implementation and discard it.

---

## 2. Fleet spend ceiling (`OpenFoam_Instance#5` item 7)

**What it is.** Per-user budgets do not compose into a fleet ceiling. Three multipliers:
the LLM is resold at cost, a live Sandbox is never cut off once started (~2.5x overrun by
design), and `ui`'s `run_session` has no `max_containers` and is metered nowhere.

**The filed fix cannot work.** The issue proposes the reaper flip
`FOAMD_SIGNUPS_ENABLED=0`. The reaper is a separate Modal function in its own container;
it cannot change an env var in the API container's process, and `config.py`'s own
docstring notes that a redeploy with no code change does not restart it.

**Fix.** Move the gate into the database: a `fleet_state` row that `signups_enabled()`
consults, with the env var retained as a manual override. The reaper sums the month's
`sandbox_sessions` + `llm_usage` on its existing 5-minute pass and sets it.
`create_instance` checks the same flag, so the ceiling refuses **new Sandboxes**, not only
signups — signups alone do not stop spend from existing accounts. Plus `max_containers`
on `run_session`.

**Acceptance test.** Seed usage above the ceiling, run one reaper pass, assert the flag
flips and `POST /v1/instances` refuses with a distinct error code. Seed below the ceiling,
assert no flip. Assert a manual env override wins in both directions.

---

## 3. Workspace mount corruption (local issue 3)

**The filed premise is wrong.** The issue blames an unclean client shutdown bypassing
`terminate(wait=True)`. The client never terminates the Sandbox — it calls `/stop`. A
killed client simply never sends it, and the reaper collects the sandbox 15 minutes later
*with* `wait=True`. A SIGKILL therefore **defers** the commit rather than bypassing it,
and no signal handler would have changed anything. There is no signal handling anywhere
in any of the three repos.

**What it actually is.** Two server-side paths put two containers on one Volume
read-write simultaneously, neither needing any client misbehaviour:

- `get_or_start` wraps `Sandbox.from_id(id).poll()` — a live gRPC call — in a bare
  `except: pass` and falls through to creating a second Sandbox. A transient Modal error
  is indistinguishable from "the sandbox is gone". `modal_sandbox_id` is then overwritten,
  so the first Sandbox becomes an orphan: invisible to the reaper, billing to the 24 h
  ceiling.
- No per-instance lock around lazy start, with `max_inputs=200` admitted concurrently.

Consistent with every observed symptom, including `modal volume put` still working from
outside (it goes through the Volume API, never the mount). Also: "deleting the instance
cleared it" is not evidence about the row — `delete_instance` destroys the Volume too.

**Confidence.** The mechanism fits the evidence but was not caught in the act. The tests
below prove the double-start happens; they do not prove it is what corrupted that
particular workspace.

**Fix.** Narrow the `except` to distinguish "gone" from "could not tell", and on an
ambiguous error fail the request rather than creating a second Sandbox. Add a per-instance
lock around lazy start. Add an orphan sweep to the reaper — list Sandboxes in the app,
terminate any whose id is not a live instance's `modal_sandbox_id` — which also plugs a
silent billing leak.

**Acceptance test — two of these go in `gate_live.py`, because a fake `Sandbox.create`
proves the code path but only the live gate proves Modal ends up with one Sandbox:**

- *Gate:* stop a throwaway instance, fire N concurrent first-requests, assert exactly one
  Sandbox in `SANDBOX_APP_NAME`. Terminate extras in teardown so a failing run does not
  leak billing.
- *Gate:* `chmod a-w /work/.foamd/exec` via `job_start`, run an exec, assert the failure
  names the path; `chmod u+w` in teardown.
- *Unit:* monkeypatch `Sandbox.from_id` to raise a transient error; assert `get_or_start`
  raises rather than creating a second Sandbox.
- *Unit:* seed a stale `modal_sandbox_id`; assert the reaper terminates the orphan.

Left out: the commit-bypass repro. It wants a Volume you have written off, which is not
something a repeatable gate should assume.

**Prerequisite, same change.** `gate_live.py` builds its own Sandbox with no `env=`,
bypassing `app/sandboxes.py` and `SANDBOX_ENV_DEFAULTS` — this is what made local issue 12
look like an image bug. Both new checks must go through the real API path or they measure
the harness instead of the product. Fix `env=config.sandbox_env()` at both
`Sandbox.create` sites and delete the masking `export` at line 402.

---

## 4. Exec logs on the shared Volume (local issues 2 + 9)

**What it is.** `run_exec` writes every command's output to
`/work/.foamd/exec/<uuid>.log` on the Modal Volume before returning any of it. The
directory grows without bound (2,993 files in a day) and shell latency is coupled to
solver I/O — `tail -5` measured at 150 s while `reconstructPar` saturated the volume; one
study spent 31.9 minutes in `bash` against 310 s of solver time. Every exec pays
`mkdir -p` on that volume before the command starts.

The file is structural, not audit: it is the capture buffer that applies the 64 KB cap
while keeping the user command's own exit code.

**Why it is not a path change.** `log_path` is a public field in `spec/openapi.yaml`,
`prompt.py` promises the agent that `read_file` will window into it, and the only way to
fetch it is `GET /v1/instances/{id}/files`, which hard-rejects anything outside `/work`.

**Fix — sync on truncation, synchronously, before responding.** Rejected: a dedicated
`execs/{id}/log` endpoint (grows the failure surface) and a periodic sync (the truncation
marker hands the agent a `log_path` it reads on the very next turn; if the sync has not
landed that is an intermittent 400 from the jail, which is worse than a deterministic
404).

The wrapper already computes the cap — `head -c 65537` exists precisely to detect
truncation — so it knows whether anyone could ever want the file.

- Sub-cap execs (the large majority) never touch the Volume. That is issue 9's latency
  win.
- Truncated execs copy to `/work/.foamd/exec/<uuid>.log` before returning, so `log_path`
  is valid the moment the client sees it. Existing files endpoint, no new route, no race.
- Growth collapses by the truncated-to-total ratio, so retention gets much cheaper —
  still needed, since a study that truncates repeatedly still accumulates. Retention is a
  **count cap** pruned inline in the exec script, not age (a burst of 3,000 in an hour is
  the failure case) and not size (the cost is per-file metadata, not bytes).

Cost accepted: the rare large-output exec still pays a Volume write, and those are the big
ones. That is the case where durability was actually promised.

**Acceptance test.** Assert a sub-cap exec leaves no file under `/work/.foamd/exec`.
Assert an over-cap exec's `log_path` is fetchable immediately on the first attempt and
returns the full output. Assert that past the retention count the oldest is gone and the
newest still resolves.

---

## 5. The 303 empty body (`OpenFoam_Instance#2`)

**What it is.** Modal's edge returns a bodyless 303 redirect on any web request exceeding
**150 seconds** — documented platform behaviour. There is no `303` or `RedirectResponse`
anywhere in the service. It hit `job_kill` and `list_instances`, which look short, because
`job_kill` goes through `get_or_start` and cold-boots a Sandbox if the instance was
reaped, and `list_instances` was queued behind the 40-thread starvation. One mechanism,
two causes of slowness.

Both fixes already shipped: client `follow_redirects=True` (2026-08-31), threadpool and
bulkhead (2026-09-03). Both repro studies are 2026-08-24, predating both.

**`EXEC_MAX_TIMEOUT_S` stays at 300.** An earlier recommendation to lower it to match the
edge window was wrong and is withdrawn. The two numbers are decoupled: the cap governs the
command's runtime, the 150 s is Modal's edge window, and a redirect-following client
completes a 250 s exec normally — measured, `sleep 200` returns rc=0 in 203.9 s. Lowering
the cap would delete working capability to satisfy a number that is not a limit.

**Fix — close the issue, with three tidies:**

1. `FoamdClient.request` treats a 3xx reaching the JSON decoder as an explicit named
   error, so a client built without redirect-following fails loudly instead of as
   `bad_response (303)`.
2. The four auth/device helpers in `hosted.py` build bare `httpx.Client`s with no
   `follow_redirects` — exactly that mistake, already in the codebase.
3. State the requirement in the API docs and `spec/openapi.yaml`. "Clients must follow
   redirects" is a contract term here, not an implementation detail.

Close the "keepalive / streaming to match `EXEC_MAX_TIMEOUT_S`" checkbox as **not
possible** — 150 s is enforced by the platform.

Split out: the `500 modal-http: ... Server has lost track of input` observation. That is
control-plane liveness — the opposite shape, since the 303 requires the container alive
and answering. Its own issue.

**Acceptance test.** Assert a 250 s exec completes with the real result through a
redirect-following client. Assert a non-following client gets a named redirect error
rather than a JSON-decode failure.

---

## Still to go through

- `OpenReynolds#19` — STEP/IGES intake. gmsh has OpenCASCADE (apt build, hard-depends on
  `libocct-data-exchange`), CLI only; `import gmsh` fails on the shipped image. #19
  collapses to a conversion shim; #6 narrows to dirty-CAD repair.
- Local issue 12 — `#codeStream` and every `coded*` feature. Modal has no uid knob;
  reachable only via `setpriv` privilege drop per command. Cost is a recursive chown over
  existing multi-GB Volumes plus mixed ownership in `.foamd`.
- `ui#1` — the split event pump. Still the one large piece of real work.
- Local issues 1, 5, 6 — agent behaviour.

## To file, found by the probes and not tracked anywhere

1. **LLM rates disagree 3x.** `README.md:208` documents Opus 5 at `1500/7500/150/1875`
   cents per million tokens; `config.LLM_DEFAULT_RATES` has `500/2500/50/625`. Sonnet 5
   disagrees too. These price into `monthly_budget_cents`. Real money.
2. **Orphaned Sandboxes bill silently** to the 24 h ceiling, invisible to the reaper.
   (Fixed by the item-3 sweep; worth its own record for the billing angle.)
3. **The gate measures a hand-rolled sandbox, not the product** — and `bec67af` added an
   `export` masking exactly the drift the gate exists to catch.
4. **`processor*` lifecycle** — the half of local issue 2 that actually hits the 20 GB
   quota. Needs a server-side notion of "study finished", which does not exist. The client
   already classifies this data as worthless (`mirror.py`), in the wrong repo to act on it.
5. **`browse.py` silently truncates the study tree** at 4,000 entries; 2,993 exec logs ate
   75% of the budget. Issue 2 causing wrong answers, not just slow ones.
6. **Control-plane liveness** — the 500 / "lost track of input" split out of `OFI#2`.
7. **`api.version` is the frozen string `"0.1.0"`** and identifies nothing. A
   `FOAMD_GIT_SHA` injected at deploy would have made local issue 7's whole
   timestamp-matching exercise a single request.

---

# Wave outcome, 2026-09-04

All five items implemented, merged to `main` in each repo, nothing pushed and nothing
deployed. Suites: foamd **426 passed**, OpenReynolds **1730 passed, 3 skipped**,
`check_openapi.py` ok. `gate_live.py` phases 0-4: **21 checks passed**, 2m49s, ~$0.04.

## What the wave proved that the triage only argued

- **The double-start bug is real, caught live.** Against real Modal on a throwaway
  Volume: old `get_or_start` turned 4 concurrent first-requests into **4 Sandboxes on one
  Volume**; the fix returns **1**. Section 3's confidence note ("fits the evidence but was
  not caught in the act") can be retired for the mechanism, though still not for the claim
  that this is what corrupted that particular workspace.
- **Local issue 12's parallel half is closed.** With `env=config.sandbox_env()` a gate
  sandbox carries the OMPI/PMIX vars and `mpirun -np 4` succeeds **with
  `--allow-run-as-root` removed**. Production was never blocked; the gate was measuring a
  container production never ships. Phase 0 now asserts the env key for key so the drift
  cannot recur silently.
- **The `#codeStream` half is confirmed and is now the only blocker on local issue 1.**
  Gate phase: `stock cylinder2D: rc=1, cannot complete as root`. The shedding control
  experiment needs the non-root work and nothing else.

## What the wave found that nobody had filed

- **A pre-existing wrapper bug.** The user command ran in a `{ }` group, not a subshell,
  so a command ending in `exit N` — ordinary in a pasted script — exited the wrapper's own
  shell: `head` never ran and `output` came back empty. Survivable only because the log
  was already on the Volume to read back, which is exactly what section 4 stops writing
  for a sub-cap exec. Now a subshell. **Found by the byte-identical check that was
  deliberately kept out of the permanent suite and run once** — the null test earned its
  keep and was then discarded, as agreed.

## Corrections to this document, made by implementation

- **§5** said "`FoamdClient.request` treats a 3xx reaching the JSON decoder as a named
  error". Those are two different places: `request` never reaches the decoder, and the
  decoder is also reached from the sign-in helpers, which bypass `request`. Guards in
  both; either alone leaves a real path uncovered.
- **§1/§3's live check is invalid as written.** `chmod a-w /work/.foamd/exec` no longer
  makes an exec fail, because the capture buffer is container-local now. The check must
  run an **over-cap** exec and assert `log_path == ""` plus the copy failure in `stderr` —
  **not** a non-zero rc, which is deliberately preserved through a failed sync.
- **§3's "assert exactly one Sandbox in `SANDBOX_APP_NAME`"** is wrong for a shared
  production app; the check diffs the listing before and after and asserts one *new*
  Sandbox.
- **§3 missed a race in the orphan sweep**: `get_or_start` returns a Sandbox a moment
  before its caller writes the id, so a listing taken in that window shows a container no
  row claims yet. Handled by a per-candidate re-read immediately before terminating.
- **§2 had a trap it did not state.** "Env override winning in both directions" means
  `FOAMD_SIGNUPS_ENABLED` can no longer default to `1` — unset and `=1` must differ. If
  the deployed Secret already carries `=1`, **deploying ships the ceiling switched off.**
- **§2 said "configurable" and named no ceiling.** $500/month was invented to fill the
  gap. An operator should set it deliberately.

## Blocked, and why

- **Gate phases 5-6 cannot pass yet.** They drive the *deployed* service, which still runs
  `bec67af`. They assert the new `stderr` field, `log_path == ""` on a failed sync, and
  one-Sandbox-per-race — none of which exist in production until this is deployed.
- **Phase 5 additionally needs a free instance slot.** The account caps at one concurrent
  instance and `7602ac4c...` holds it. Not deleted deliberately: `delete_instance` destroys
  the Volume, and that is the row local issue 3 cites as evidence.
- **The migration `sql/schema_f13.sql` is unapplied.** A missing `fleet_state` table fails
  open by design, so until it is applied there is no ceiling — today's state, not a
  regression. Deploy order is migration first.
