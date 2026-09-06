# Triage decisions, wave of 2026-09-05

Issue-by-issue decisions for wave 2, taken after the five read-only probes of
2026-09-04/05. Each entry: what the issue actually is (where that differs from how it
was filed), the agreed fix, and the acceptance test. Nothing here has been implemented.

The previous wave's file is `triage-decisions.md`. This one covers the issues that
wave did not reach, plus four the probes re-scoped.

A note on the probes: two of the five overturned the plan they were sent to cost.
`#codeStream` does not need a privilege drop, and PR #17 cannot work as submitted.
Where a filed issue rests on a premise the code does not support it is said explicitly,
because the filed text is still what a reader finds first.

---

## 1. A bare 500 when the sandbox goes away underneath an in-flight request

`OpenFoam_Instance#6`.

**What it is.** `app/main.py:177-191` registers exactly two exception handlers:
`supa.httpx.HTTPError` and `ApiError`. A Modal SDK exception raised when the Sandbox is
gone — from `sb.exec`, from `Sandbox.from_id(...)`, from the file helpers — matches
neither, so it propagates to FastAPI's default handler, which answers
`500 Internal Server Error` with no body at all. The client message is uninformative
because there is nothing in the response to be informative with.

**What triggers it, measured.** Not load. Against the deployed service, 224 requests on
one ephemeral instance: 4 cores pinned for 150 s with 104 concurrent requests throughout
produced **0** 500s; `POST /stop` issued 1.5 s into a burst produced **9 of 120**. A
read-only `GET /files?stat=1` is affected as readily as an `exec` (5/75 vs 4/75). In the
original trail the long render was what made requests slow enough to still be in flight
when the sandbox was reaped — coincidence with the load, not causation by it.

**Fix — A+C.** One app-level handler for Modal SDK exceptions returning a coded
`sandbox_unavailable` **503** with `Retry-After`, uniformly. Plus attribution: log
Modal's own message (`Server has lost track of input`, `App is not running`) against the
request id, which today lands in the control-plane container's log belonging to nothing.

**Why not distinguish 409 from 503.** Considered and rejected: answering "stopped for
good" vs "try again" means classifying which Modal exceptions mean which, and local
issue 3 is the record of what guessing wrong there costs — a bare `except` that could
not tell a transient error from a dead sandbox is what put four Sandboxes on one Volume.
A `/stop` racing an in-flight request is genuinely "try again". Uniform 503 states only
what is known.

**503 is already on the client's retry path.** `backend/hosted.py` has `500` in
`_RETRY_STATUSES` because of this bug; 503 needs no client change to be handled, and the
500 entry can stay until the deploy is confirmed.

**Acceptance test.** Unit: a fake sandbox helper raising the Modal exception; assert the
response carries `error: sandbox_unavailable`, status 503, and a non-empty message.
Live, on a throwaway instance: the probe that found it — 24 concurrent requests across a
`POST /stop` — and assert **zero bodyless 500s**, every failure carrying a JSON `error`.
That live check is the one worth keeping permanently; the unit test cannot prove the
handler catches the exception Modal actually raises.

---

## 2. `api.version` identifies nothing, and the env layer cannot be diffed

Local issues 19 and 7 (the residual half).

**What it is.** `/health` reports `"version":"0.1.0"` — the string in
`FastAPI(title="foamd", version="0.1.0")`, unchanged since the file was written. Nothing
in the deployed service says which commit it came from. Separately, nothing outside the
running container can say what `FOAMD_STAFF_DOMAINS`, the `Caps` values, the rate limits
or the tariff resolve to, so code parity does not prove behaviour parity.

**Decision: won't fix, for now.** Deploys have carried a `--tag` since v17, so
`modal app history` already records the commit for anyone who needs it, and that
audience is developers. Users have no reason to know the version. The remaining gap does
not justify the surface.

**What that accepts, stated so it is not rediscovered as a surprise.** The `--tag` covers
v17 onward only, and it is readable from `modal app history` rather than from the
service — so a repeat of issue 7 (diffing a running service against a checkout) is still
timestamp-matching and OpenAPI comparison, not one request. Issue 7's env layer stays
open and undiffable by exactly the amount it is today.

**Revisit if** a second incident needs the deployed commit identified from outside, or if
an operator edit to `foamd-secrets` is ever suspected of not having reached the running
container. The fix is small whenever it is wanted: `FOAMD_GIT_SHA` injected at deploy,
reported publicly; resolved caps and tariff added to the existing `FOAMD_HEALTH_TOKEN`
block, which already gates the fleet-spend figures with `hmac.compare_digest`.

---

## 3. LLM rates: a stale README, two stale comments, and two zero-margin holes

Local issue 14, widened by probe A2.

**The filed issue is a documentation fix, not a money bug.** `config.LLM_DEFAULT_RATES`
is correct — Opus 5 at `500 / 2500 / 50 / 625` cents per million tokens matches list
price ($5/$25 in, cache read 0.1x, cache write 1.25x), and Sonnet 5 at
`200 / 1000 / 20 / 250` likewise. Commit `18ed011` (2026-08-27) set those and updated
`tests/test_llm_usage.py`; `README.md:211` was written earlier and never followed. Haiku
4.5 agrees in both places.

**Nothing was ever mischarged.** `llm_usage` rows store the four token counts only; the
price is computed at read time by `cents_for`, and the credits ledger is prepaid top-ups
rather than metered invoices. Rows written while the README figures were live are
already repriced at today's rates on every read. The one non-recoverable effect would be
a `402 budget_exhausted` **refusal** issued during that window — an operator query
against `request_log`, not a code change.

**Fix, in three parts.**

1. `README.md:211` — replace the Opus and Sonnet rows with the code's values and drop
   the "verify before the first real bill" TODO, which has now been verified.
2. `config.py:597-601` — delete the outer "TODO verify against anthropic.com/pricing"
   comment, which contradicts the correct inner comment below it. And correct
   `config.py:605-606`, which says the rates are cost and "a margin is a billing
   decision, applied when invoices exist": that was true when written and is now false,
   because `cents_for` applies `price_multiplier()` (default 2.0) on every read. A
   reader of that comment alone would conclude no margin is applied.
3. The margin holes, below.

**The two zero-margin holes — decision: A, strip and refuse. Do this part first.**

Both have the same shape: a model that costs more than its row, billed at its row, with
`price_multiplier()` of 2.0 cancelling exactly against a 2x cost. Margin nil.

- **Fast mode.** `app/llm.py` forwards the request body verbatim (`vet_body` checks only
  the model allowlist and clamps `max_tokens`) and forwards `anthropic-beta` via
  `_FORWARD_REQUEST_HEADERS:44`. Opus 5 fast mode bills at $10/$50 per MTok, 2x the row.
  **Fix:** strip the fast-mode beta header in `vet_body`, next to the allowlist check.
- **Unpriced models.** `LLM_FALLBACK_RATE_MODEL = "claude-opus-5"` means widening
  `FOAMD_LLM_MODELS` to any $10/$50 model reproduces the hole silently. **Fix:** refuse
  an unpriced model rather than guessing a row.

**Why not a `speed` dimension now.** It is the right answer when fast mode is something
to sell; it is schema work in service of a product decision nobody has taken. Stripping
the header cannot be wrong later and does not block adding the dimension.

**Not modelled, and now a known undercharge rather than an unknown one.** `cache_write`
at 1.25x input is the 5-minute-TTL rate; `cache_control: {"ttl": "1h"}` is dearer, and
both TTL classes collapse into one `llm_usage` column. Unfixable without a schema change.

**Acceptance test.** Unit: `vet_body` strips the fast-mode beta header and leaves other
`anthropic-beta` values intact; an unpriced model is refused rather than priced at the
Opus row. Assert `cents_for` against the corrected README table so the doc and the code
cannot drift apart again silently.

**Operator check, not code.** The README row documents the `FOAMD_LLM_RATE_*` env
overrides, and those are read at runtime. If the `foamd-secrets` Secret was populated
from that table, production is charging 3x on Opus today. Not verifiable from source.

---

## 4. `#codeStream` and every `coded*` feature is refused, because the sandbox is root

Local issue 12, the half left open when the OpenMPI half closed. Re-scoped by probe A5.

**What it is, plainly.** Everything in the workspace runs as root. OpenFOAM refuses to
compile-and-load code at runtime when it is root — a guard against a class of attack that
does not apply in a sandbox that already runs whatever the agent asks for. It fails at
`blockMesh`, before any solver, with a message about administrator rights, so it reads as
a broken image rather than as a policy.

**Two corrections to the filed issue.** It records two guards; there is one.
`FOAM_ALLOW_SYSTEM_OPERATIONS` appears **nowhere in the v2512 source** (zero grep hits
against the release tarball the image pins), and `etc/controlDict:75` already ships
`allowSystemOperations 1`. The only guard is the euid test in `dynamicCode.C:71`, which
calls `Foam::isAdministrator()` — `POSIX.C:417`, `return (::geteuid() == 0)` — and that
is the **only call site in the entire OpenFOAM tree**. It is a single `if`, not a
load-bearing utility.

**Fix: interpose the symbol, do not drop privilege.** A 13 KB shared library defining
`_ZN4Foam15isAdministratorEv` to return 0, loaded ahead of OpenFOAM. Two changes: one
`.run_commands` layer appended to the **end** of `image/image.py` (after the imageio
block, so the ~4-minute HiSA layer stays cached), and one `LD_PRELOAD` entry in
`config.SANDBOX_ENV_DEFAULTS`. This is the same mechanism as the already-closed OpenMPI
half — applied at `Sandbox.create`, operator-overridable through `FOAMD_SANDBOX_ENV`,
removable without a code change — and it reaches every existing instance at its next
lazy start, so there is no migration.

**Why interposition works here, verified against the pinned source.** OpenFOAM's own
build rules leave the symbol replaceable: `wmake/rules/linux64Gcc/c++Opt` is `-O3` with
no `-fno-semantic-interposition`, `General/Gcc/c++` sets no `-fvisibility=hidden`, and
`link-c++` uses plain `-shared` with no `-Bsymbolic`. A local replica built with those
same flags was measured both ways: overridden at `-O3`, correctly defeated when
`-fno-semantic-interposition` was added.

**Decision: B — ship it globally, and gate the build on it.** `LD_PRELOAD` applies to
every process in the sandbox, and the image build fails if the library is missing or does
not load.

- **Why global rather than only on OpenFOAM commands.** Narrowing it means finding every
  place a command can start — `run_exec`, `raw_exec`, detached `jobd` jobs, and tutorial
  `Allrun` scripts that spawn their own — and missing one leaves the bug in place for
  exactly the cases the fix exists for. That hunt is what made the privilege-drop route
  expensive; not repeating it is the point.
- **Why the build gate is not optional.** A missing or corrupt library makes `ld.so`
  print a warning before **every command in the sandbox**, contaminating every exec's
  captured output and presenting as a hundred unrelated bugs. Per this repo's own rule, a
  success marker that cannot fail is not a success marker: the layer must assert the file
  exists and loads.

**What was rejected, and what it would have cost.** The privilege drop (`setpriv`) needs
four insertion points in `sandboxes.py`, not the two estimated; `--reuid` alone leaves
egid 0 so every solver file lands group-root; `HOME=/root` survives `setpriv`, putting
`FOAM_USER_APPBIN` somewhere the dropped user cannot write and breaking the other half of
plan-1's promise while fixing the first. And it needs a recursive ownership change over
every existing workspace — measured at 0.133 s for 20,809 inodes on local ext4, but that
is a lower bound on NVMe, and the only Volume datapoint this codebase has is a `tail -5`
at 150 s. Extrapolated 10 s to 83 minutes per instance, which exceeds
`EXEC_MAX_TIMEOUT_S` and would have to become a job the user never started.

**On security, so it is not re-litigated.** The shim disables OpenFOAM's guard against
compiling and `dlopen`ing code as root. That widens no blast radius here: `run_exec`
already runs arbitrary `bash -lc` as root by design, and the agent has a shell. The shim
stops OpenFOAM refusing something the sandbox already permits. Say this in the commit
message.

**Ship-gate, five minutes on a live instance, before the change is written.** Confirm the
ESI-packaged binary exports the symbol —
`nm -D $FOAM_LIBBIN/libOpenFOAM.so | grep isAdministrator` must show
`T _ZN4Foam15isAdministratorEv` — then run a stock `#codeStream` case with the library
preloaded and see it mesh. The source-tree rules are not proof that ESI built with them;
this is the only thing that is. **Fallback if ESI shipped `-fno-semantic-interposition`:**
interpose `geteuid` itself, which is coarser and would also silence OpenMPI's own root
check, so the OpenMPI variables stay as belt-and-braces.

**Acceptance test.** Live, permanently: a stock tutorial using `#codeStream` reaches a
mesh, and a `codedFixedValue` case reaches a solve. Unit: assert the `LD_PRELOAD` entry
survives `config.sandbox_env()` key-for-key, the way the gate now asserts the OpenMPI
variables — the drift that check exists to catch is precisely what hid this bug's
neighbour for a day.

---

## 5. Tutorial helper functions do not survive a background job

Local issue 11, and the filed framing is half right.

**What it is.** Every OpenFOAM tutorial's `Allrun` begins by sourcing
`$WM_PROJECT_DIR/bin/tools/RunFunctions`, which defines `restore0Dir`,
`runApplication`, `runParallel` and `getApplication` as **shell functions**. A case built
by copying a tutorial and following its idiom dies with `127` — "command not found" —
which reads as *the binary is not installed* and points diagnosis at the image rather
than at the shell. It sits directly across the path the product steers agents toward:
long work is supposed to go through `job_start` rather than a hand-rolled busy wait
(issue 5), and the tutorials the agent is pointed at are written in an idiom that breaks
there.

**Correction to the filed issue.** It reads as though jobs differ from execs in their
environment. They do not: both `run_exec` (`sandboxes.py:432`) and
`_wrapped_job_command` (`:483-489`) source the OpenFOAM bashrc through the same
`_SOURCE_OF` constant, so both have `blockMesh` and friends on `PATH`. The real property
is that **every tool call gets a fresh shell** — exec-to-exec as much as exec-to-job.
Nothing the agent exports, sources, aliases or defines survives to the next call. What
`RunFunctions` runs into is not a job-versus-exec asymmetry; it is that no tool call
inherits anything from any other. `cwd` is the one piece of state that does carry, and
only because both paths take it as an explicit parameter.

**A second asymmetry, unfiled, same family.** Execs run `bash -lc` — a *login* shell,
which reads `/etc/profile` and `/etc/profile.d/*.sh`. Jobs run
`printf %s <b64> | base64 -d | bash`, a non-login non-interactive shell that reads
neither. Nothing in the image depends on `/etc/profile.d` today, which is why this has
not bitten; it would fail exactly as issue 11 does, with a `127` that accuses the image.

**Fix: A, plus align the two shells.** Source `RunFunctions` in the job wrapper, so every
background job has the tutorial idiom available and no tutorial needs editing. And make
the job wrapper a **login shell**, so the two paths agree by construction rather than
being reconciled one missing symbol at a time.

**Why not the alternatives.** Loading the helpers for *every* command in the sandbox pays
the cost thousands of times for the handful of scripts that need it. Telling the agent to
source `RunFunctions` itself makes it responsible for a quirk of our plumbing that it has
no way to discover except by hitting it — and hitting it produces a message that blames
the image.

**Statelessness is deliberate and stays.** See §5a.

**Acceptance test.** Live: a stock tutorial `Allrun` copied unmodified, started through
`job_start`, reaches a solve — the gate's own failure (`run 5, 2026-09-02`,
`solve.sh: line 29: restore0Dir: command not found`, `rc=127`) replayed and passing.
Unit: assert the job wrapper and the exec wrapper resolve the same `PATH`, the same
OpenFOAM variables, and the same helper functions — one test that fails if either path
drifts from the other again.

### 5a. The shell stays stateless, and the agent is told so

Decided while settling issue 11, recorded here so it is not reopened as an oversight.

**Statelessness is the design, not an accident.** Every tool call gets a fresh shell.
The argument for keeping it, in this product specifically:

- **Context is compacted.** A stateful shell makes a command's meaning depend on state
  set in a turn the model can no longer see. `runApplication simpleFoam` would behave
  differently according to something that has scrolled out of the transcript — an
  unreproducible bug by construction.
- **The evidence culture assumes commands are self-describing.** `docs/issues.md` works
  because a transcript line is a complete fact. Hidden shell state breaks that: the log
  stops being sufficient to explain what happened.
- **Sandboxes get preempted.** Modal restarts containers underneath in-flight work —
  that is `OpenFoam_Instance#6`'s whole subject. A stateful shell would lose its state at
  exactly those moments, silently, with nothing for the agent to detect.

**It also matches the harness the model is most practised at.** Claude Code's own Bash
tool states its contract as: *"Working directory persists between calls… Shell state
(env vars, functions) does not persist; the shell is initialized from the user's
profile."* That is very nearly what foamd already does — `cwd` is the piece that carries,
`_SOURCE_OF` plus `bash -lc` is the profile initialization. The one deviation is the job
wrapper not being a login shell, which §5 closes. Worth stating the limit of this
argument: what is known is the contract these models are trained and evaluated against,
not the training. The contract is the thing to match, and stateless-plus-cwd is the
well-trodden path.

**The cost, and why it is paid elsewhere.** Statelessness breaks idioms that assume a
persistent shell — which is most tutorial shell scripting, and is exactly how issue 11
presents. The answer is to make the environment **complete on every call** (§5), not to
add state.

**Say it to the agent, as a fact.** Today nothing tells it. It discovers statelessness by
hitting it, and the error it gets (`127`) accuses the image. Add one sentence to the
`bash` and `job_start` descriptions in `openreynolds/tools.py`, in the register the
contract requires — a fact about the environment, no instruction about what to do with
it, phrased after Claude Code's own:

> Each call runs in a fresh shell. `cwd` is what carries between calls; nothing else
> does — variables you export, files you source and shell options are gone by the next
> one. The OpenFOAM environment and the tutorials' `RunFunctions` helpers are loaded for
> you in every call, `bash` and `job_start` alike.

The second sentence is only true once §5 has shipped, so the wording and the wrapper
change land together or not at all.

**Acceptance test.** The existing prompt/tool-description rules apply
(`tests/test_prompt.py`, `tests/test_briefing.py`): assert the sentence states the
environment and issues no instruction, and that it names no toolbox script. Pair it with
§5's parity test, so a description claiming the helpers are loaded fails if they are not.

---

## 6. `migrate.py --sql` also migrates, and nothing says so

Local issue 20.

**What it is.** `--sql "<statement>"` is an operator escape hatch that runs one ad-hoc
statement against the project's Postgres — not a schema dump, not a file (`--file` is
that), and deliberately not recorded in `schema_migrations`. But `run()` applies every
pending migration first (line 143), clears `pending` (150), and only then executes the
statement (152). The docstring lists `--sql` alongside `--status` and `--dry-run`, which
reads as inspection, and says nothing about the ordering.

**What it cost.** 2026-09-04: two `--sql` calls made to inspect which tables existed
applied `schema_f13.sql` as a side effect, and the tables they then reported were ones
the same command had just created — evidence that briefly looked like someone had applied
the migration by hand.

**The ordering is not the bug.** Worth stating, because the first reading of this issue
was that `--sql` should never migrate. It should be able to: a one-off statement that
fixes up data in a table a pending migration creates needs that migration to land first,
and decoupling them breaks a real flow. The defect is that the behaviour is undocumented
and has no opt-out, so somebody reaching for `--sql` to *look* at something changes the
schema instead.

**Fix: A — add `--no-apply`, and document the ordering loudly.** The flag skips the
migration pass for that invocation, which is the read-only path the tool never had.
Default behaviour is unchanged.

Documentation is half the fix here and should not be treated as the trimming:

- The docstring's usage block must say, on the `--sql` and `--file` lines themselves,
  that pending migrations are applied first unless `--no-apply` is passed. Not in a
  paragraph below — on the line a reader copies.
- The tool should **print what it applied before the ad-hoc result**, so a run that
  migrated says so in its own output rather than only in the returned dict. The 09-04
  confusion was possible because the answer arrived with no account of what produced it.
- `--no-apply` with migrations pending should name them in the output — read-only, but
  the reader is told the schema they are querying is not the schema in `sql/`.

**Why not the alternatives.** Refusing `--sql` while anything is pending blocks the
migrate-then-fix-up flow the ordering exists for. A confirmation prompt does not fit: this
runs through `modal run`, often non-interactively.

**Acceptance test.** Unit: `--no-apply` with a pending file runs the statement and leaves
`applied` empty and `pending` intact; without it, the same call applies first and reports
both. Assert the docstring's `--sql` usage line mentions the ordering — a doc test,
because the doc being wrong is what this issue is.

---

## 7. `browse.py` truncates the listing without saying so

Local issue 18.

**What it is.** `Browser.list` runs `find -H … -maxdepth N | head -n MAX_ENTRIES`
(`browse.py:88`, `MAX_ENTRIES = 4_000`) and returns the survivors. When the cap is hit,
nothing in the result says so. With 2,993 exec logs under `/work/.foamd/exec`, three
quarters of the budget went to files nobody was looking for and the study tree was cut
short, silently. Issue 2's fix removed that particular cause — the capture buffer is
container-local now — but the truncation is still mute, and `processor*` directories
(local issue 16) are the next thing to fill the budget the same way.

**Who this is for, which decides the fix.** `browse.py` is **not agent-facing**. Its own
docstring: "Looking at the workspace, without having to ask the model for it… It never
appears in the conversation, and nothing the model does depends on whether anyone is
looking." The audience is the person, and the contract is *show me the tree*.

**Fix: A — say when the listing was capped, and at what.** Nothing else.

**Explicitly rejected: excluding `.foamd` from the walk.** It was proposed to stop the
control plane's bookkeeping eating the budget, and it is the wrong move for a tool with
this contract — a file browser that silently omits a directory that exists on the
filesystem. The person most likely to browse `.foamd` is somebody debugging exactly the
thing that fills it. `MAX_ENTRIES`' own docstring already carries the right intent — "a
listing past this is scrolling, not information" — so the cap is not the defect; its
invisibility is. Told the listing was capped, the user narrows `path` or `depth`, both of
which they already control (`DEFAULT_DEPTH = 4`).

Pagination was considered and dropped as more machinery than the problem needs.

**Acceptance test.** Unit: a fake backend returning `MAX_ENTRIES` lines yields a result
flagged truncated and a message naming the count; one line fewer does not. The message is
the assertion — a truncation flag nothing renders is the same bug again.

---

## 8. The web tier has no rate limiting at all

A bullet in `OpenFoam_Instance#5`'s "What is in `ui`" section, which proposed a companion
issue in the `ui` repo. That issue was never filed, so none of those four items is
discoverable from the repo they belong to. **Filing it is part of this work.**

**Correction to the filed item, found while scoping.** It pairs rate limiting with
"Turnstile is off by default and never verified server-side", implying the app could
verify it. It cannot: there is **no sign-in route in `ui` at all** — the browser talks to
Supabase directly, and `web/src/turnstile.ts` holds a *site* key only. Throttling sign-in,
sign-up and password reset is a Supabase Auth setting (Attack Protection), not app code.
Tracked as an operator task, not part of this change.

So the honest scope: **this protects spend and guessable codes, not the front door.**

**Fix: A — port foamd's `ratelimit.py`, retuned for this tier.** The bucket arithmetic is
proven in production since F6 and was copied rather than rewritten. What differs is who
gets a bucket and which routes are worth defending. Rejected: extracting a shared package
(the first release coupling between the repos, for 103 lines that will diverge rather than
converge — the two tiers guard different things), and limiting at the edge (no per-subject
identity there).

**Who gets a bucket, which is the part that can be got wrong.** An authenticated request
is charged to the **verified** Supabase subject, from inside `auth.require_user`, after
the signature is checked. Anything else — no `Authorization` header, or a token that
failed to verify — is charged to the client address. A forged `sub` therefore never
reaches the user bucket; a thousand invented subjects share one address bucket instead of
minting a thousand allowances. This is foamd's own lesson (`#5` item 1) applied before it
could bite here.

Charged **once**, not twice: a signed-in request does not also spend the address bucket,
so colleagues behind one office address do not spend each other's allowance. The failure
path closes the hole that opens — `require_user` charges the address when a token does not
verify, so a flood of junk tokens cannot pass both halves untouched.

**Hooked into `require_user` rather than added to each route.** Every route in `app.py`
already depends on `require_user` or `require_member` (which calls it), so one insertion
point covers all of them with no ordering risk and no route edits.

**The limits, per minute, per subject.**

| class | routes | allowance |
|---|---|---|
| `guess` | `POST /api/account/redeem`, `POST /api/cli/approve` | 5 |
| `create` | `POST /api/session`, `/billing/checkout`, `/account/service-key`, `/account/providers`, `/account/providers/{id}/default` | 10 |
| `render` | `GET /api/studies/{id}/(mesh|mesh/field|slice|geometry|renders|cases|case-info)` | 60 |
| `general` | everything else, including `say` and the transcript reads | 120 |

All four are env-tunable (`APP_RATE_LIMIT_*`); an unparseable value falls back to the
default rather than stopping the container from booting, and `0` switches a limit off,
which is the only way to disable one without a deploy.

Two of the numbers are not copied from foamd and the reasoning belongs with them:

- **`POST /api/session` is `create`, not `guess`.** Session *concurrency* is already
  closed by the unique partial index on `app_sessions(user_id)` — a second live session is
  a 409, not a second runner — so this guards churn, not spend. 10/min is ample.
- **`guess` is 5, not foamd's strict 10.** 10/min is 14,400 guesses a day per identity. 5
  halves that and no more: a token bucket has no daily ceiling, so this **slows** a
  guessing run rather than ending it. If redemption codes are short, code entropy is the
  actual fix and this is not a substitute for it.

**What this does not achieve, written down rather than discovered later.** Bucket state is
per-process. foamd can rely on that because `modal_app.py` pins `max_containers=1`; the
web tier runs 2 today and `ui#1` wants more permanently. Each container enforces its own
copy, so the effective allowance is **N x** the number configured — a degradation, not a
bypass. Whatever shared bus `ui#1` settles on is where this should move.

**Acceptance test.** `tests/test_ratelimit.py`, 25 assertions. The ones that matter: one
user exhausting an allowance does not touch another's; distinct forged subjects do not
each get a fresh allowance; an anonymous flood is charged to its address; a signed-in user
is **not** squeezed by the address they share; and `say` and the transcript are never in a
tight class — a limiter that lands on typing is a broken app, not a defended one. An
autouse fixture resets the buckets between tests, and one test asserts that fixture works,
because leaked module state would surface as a 429 in an unrelated test.

**Status: decided, not implemented** — it lands with the rest of the wave.

A working implementation was built while scoping and then reverted, so `ui` is clean.
It is kept as a patch (`ui-ratelimit.patch`, 496 lines: `reynolds_app/ratelimit.py`,
`tests/test_ratelimit.py`, and edits to `app.py`, `auth.py`, `settings.py`,
`tests/conftest.py`) and measured green — full `ui` suite 186 passed, 25 of them new.
Two choices in it are the ones to re-examine before it lands: charging a signed-in
request **once** (subject only, never also the address, so an office does not share an
allowance) with the failure path charging the address so junk tokens cannot pass both
halves; and hooking `require_user` rather than decorating each route, which is one
insertion point with no dependency-ordering risk but is less visible at the call site.

---

## 9. STEP and IGES are refused at intake, though the image can already read them

`OpenReynolds#19`. Probe A1 settled the feasibility question the issue's last checkbox
asked.

**What is true, measured.** gmsh **4.12.1** on the workspace image has **OpenCASCADE
7.6.3** compiled in (`gmsh -info` build options, corroborated by packaging:
`libgmsh4.12t64` -> `libocct-data-exchange-7.6t64`, which *is* the STEP/IGES reader). It
round-tripped a real STEP and IGES box in a container rebuilt from the same pinned base
digest and the same apt line, recognising the cylindrical face as a `Cylinder` / `Surface
of Revolution` — B-rep parsing, not a tessellated fallback. **This is issue #19, not #6:
a conversion step into the existing path, no CAD kernel project.**

Not tested: STEP from a real CAD system — assemblies, multiple solids, degenerate faces.
The probe's file was written by OCC itself, the friendly case. A cheap follow-up, and a
Docker image was left locally to run it in.

**The Python module goes in the image.** `pip install gmsh` — the PyPI wheel is also
OCC-enabled. Only the CLI is present today. The shipping order (this touches
`image/image.py`, which §4's shim also touches, so that file takes one owner in the wave)
is a scheduling detail and was explicitly not allowed to decide the design.

**There is no single tolerance knob.** gmsh 4.12 does not surface OCC's
`StlLinearDeflection` / `StlAngularDeflection`. Facet density comes from `-clmax` /
`-clscale` / `Mesh.MeshSizeFromCurvature`; geometry fidelity from `Geometry.Tolerance`;
and `Geometry.OCCTargetUnit` rescales a file that declares no unit. Measured on one box:
`-clscale 4` -> 64 facets, default -> 692, `-clmax 0.1` -> 5238. Same geometry.

### The tolerance follows the mesh, not the file

The parameter that matters is tessellation edge length relative to the **finest surface
cell size** at the wall. Tessellate much finer and you pay for triangles snappyHexMesh
cannot resolve. Tessellate coarser and the faceting *becomes* the geometry: the mesh
faithfully reproduces a polygonal approximation and the flat spots reach the pressure
field looking like physics. A defensible default is `clmax ~ 0.5 x dx_surface` with
`MeshSizeFromCurvature ~ 20`, so curvature drives facets in curved regions rather than the
global cap; the sagitta for a chord L on radius R is ~ L^2/8R, which that ratio keeps
comfortably subgrid — and which can be *reported* rather than assumed.

**This does not require converting twice.** An earlier draft proposed a coarse preview
pass to size the geometry, then a matched pass before meshing. That was wrong, and the
correction is the point: **inspection needs no triangles at all.** A person opens the file
in CAD, reads the smallest feature off the B-rep, and picks a resolution. gmsh's OCC API
answers the same questions without meshing — entity list, per-entity bounding boxes, curve
lengths and areas via `getMass`, curvature, declared units. So: interrogate, choose
`dx_surface`, tessellate **once**.

`geometry_view.py`'s four views do need triangles to draw. A **render** tessellation is
not a geometry decision — nothing downstream consumes it, it never reaches a solver, and
it can be as coarse as looks right. Say so in the code, so nobody later unifies it with
the simulation tessellation and reintroduces the accident.

### What is first-class, and what is left to the agent

**No a-priori stats report.** With the module importable the agent has a shell and an API,
and interrogating a STEP file is a script it can write. A fixed report is not merely
incomplete — it implies *these* are the numbers that matter, and its likeliest effect is
the agent reading the report instead of the geometry. Whatever set is chosen in advance,
the case that matters is the one with a feature nobody thought to measure.

**Deterministic conversion, agent-supplied parameters.** The split is: the agent does the
analysis and chooses; a deterministic script performs the conversion and **logs what it
actually did** — resolved `clmax`, implied max sagitta, unit interpretation, entity and
facet counts, and the exact gmsh calls. The script decides nothing. That closes the
issue's real complaint, which was never that agents choose badly but that nothing records
what was chosen. Offered, not imposed, with the standing of `study_state.py` — the agent
may write its own, and what it cannot do is convert **through ours** without the numbers
being visible.

**So first-class means four things, and no more:** the capability in the image; a fact in
the prompt that the build reads STEP and IGES and the module is importable (today it says
only "gmsh is installed"); the five toolbox readers no longer refusing `.step` `.stp`
`.iges` `.igs` as unreadable surfaces; and the reporting converter.

### Two loud failures, deliberately

The converter **refuses** rather than guessing, in both cases where a wrong answer looks
like a right one:

- **No tolerance given** — no silent default. `clmax` is the decision this whole issue
  exists to make explicit; a default would be the accident wearing a different hat. The
  refusal names what to pass and states the `0.5 x dx_surface` relationship as a fact.
- **No unit declared and none supplied** — a STEP that declares no unit is silently
  interpreted, and a 1000x scale error produces a perfectly plausible mesh and completely
  wrong forces. The refusal says the file declares no unit and names the override.

A file that *does* declare a unit converts without ceremony, with the interpretation
reported.

### No auto-triggered workflow on CAD detection

Considered and rejected. `docs/design.md` §1 forbids the harness injecting a checklist or
workflow, and a subagent firing on detection is exactly that. Three reasons beyond the
contract:

- **The right tessellation depends on the question, not only the geometry.** `dx_surface`
  for an external drag study and for internal flow through the same part are different
  numbers. A triggered subagent sees the file and not the study.
- **Detection fires on the wrong things** — any `.step` in the workspace, including a
  reference part or something uploaded and abandoned.
- **It moves the analysis out of the transcript**, when the judgement about smallest
  feature size is exactly the reasoning that belongs beside the mesh choice it produced.

What replaces it: at intake, state the facts and stop — this is a STEP file, it has not
been converted, the module is importable, the declared unit is X or there is none.

**Acceptance test.** Live: a STEP with declared units converts and the log carries
`clmax`, sagitta, unit, entity and facet counts. Unit: no `clmax` refuses and the message
names the parameter; a unitless STEP refuses and the message names the override; the same
file with a unit supplied converts. Assert the five readers accept the four suffixes.
Assert the render tessellation is never read by anything that feeds a solver — the one
regression that would be invisible in a passing suite.

---

## 10. Tutorial and past-case retrieval: PR #22, and the harvest question it asks

`OpenReynolds#1`, implemented by an outside contributor in PR #22. Reviewed by probe A3:
**merge with changes**. The suite is green on Linux (1915 passed, 3 skipped — pre-existing
`blockMesh`-missing skips), and both symlink tests the contributor could not run pass here.

**What the corpus is**, since it decides the rest: a searchable index of cases in two tiers
that are never merged. The **vendor tier** is `$FOAM_TUTORIALS` — 556 cases, fixed,
identical on every instance. The **earned tier** is this instance's own past studies, the
ones that left a `.reynolds/` bundle. The separation is §9's warning made structural: a
vendor case is one somebody else validated, an earned row is our own prior output, and
merging them lets the system cite itself as evidence.

### Decision: build the index eagerly, once per session

In `_sync_toolbox`, after `put_tree`, run `corpus.py build` on the instance inside the
existing `try`, so a failure is the yellow line that is already there. The lazy
rebuild-if-stale path in `search.py` **stays** as the fallback for an instance that missed
a sync.

**The contract does not decide this, and the contributor asked the wrong question.**
Building an index is none of the forbidden five — it gates nothing, orders nothing, and is
invisible until the model runs a query. It is also strictly less intrusive than what
`_sync_toolbox` already does unasked: `cli.py:1870` pushes 22 scripts and 3 notes of
hand-written CFD know-how onto the volume. A derived index over files already there adds
no know-how at all.

**What decides it is that under lazy, the earned tier never refreshes.** `staleness()`
rebuilds on a missing file, a schema bump, a changed tutorial tree, or a changed
`$WM_PROJECT_VERSION`. **None of those fires when a study finishes.** On a long-lived
instance the earned index freezes at whatever it was when first built, and `--rebuild` is
the only way out — so issue #1's fourth bullet, "the earned-tier feedback loop stays
visible", does not close under lazy. Eager closes it once per session, at the one moment
it is correct to: the current study has produced nothing yet.

**Cost, and the bound on it.** The 0.75 s the contributor measured is the vendor tier
only. Measured against 42 real study directories (~89k files): earned harvest **~21 s
cold**, 0.44 s warm; `foam_fork(work)` alone 3.5 s cold. Both halves are bounded — the
vendor tier is static, the earned tier lives under a 20 GB quota — which is why this is
affordable at session start and is not affordable inside a model tool call, which is
output- and time-capped. One caveat kept deliberately: **the quota bounds bytes, not
inodes.** A decomposed case writes a field set per rank per write time, so a workspace can
hold far more paths than its size implies (the same lesson A5 hit costing a recursive
chown). Best-effort placement is what makes that safe.

Ideally `corpus.py build` grows an `--if-stale` flag so the sync-time call skips the vendor
re-harvest when nothing changed and always refreshes the earned tier.

### Must change before merge

1. **Land the eager decision** — two lines plus a `try`.
2. **`stamp["work"]` is written and never read.** `corpus.build` records it; `staleness()`
   checks only `stamp["tutorials"]`. So `--work` pointing at a different tree silently
   reuses the old index — the exact bug the contributor found and fixed for `--tutorials`.
   The `SCHEMA_VERSION = 3` docstring claims both are detectable, which is half true.
   Check it or stop claiming it.
3. **CHANGELOG entry.** Every prior toolbox script got one under `[Unreleased]`; the
   prompt change belongs there too, given CONTRIBUTING's "a change to what the model sees
   gets a closer look".
4. **Qualify `search.py::_append`'s docstring with "on Windows"** — see below.
5. **Tie-break vendor ahead of earned in `regime`'s ranking** — see below. Decided as a
   fifth must-change rather than a follow-up issue: it is two lines, and an issue filed
   against unmerged code would outlive its own fix. This entry is the record.

### The append-corruption report: real, and not a bug here

The contributor reported that `study_state.record`'s docstring claim — concurrent appends
"interleave lines, never halves of a line" — is false, measuring 763/900 lines with 23 cut
mid-JSON, and that an `os.write`/`O_APPEND` rewrite did not help (796/900). They flagged
that they measured on **Windows**.

**It does not reproduce on Linux.** Re-run against the real `study_state.record`: 900/900
at 6x150, and 4800/4800 at 16x300 across record sizes from ~146 B to 64 KB — zero loss,
zero tearing, in every configuration including records far larger than the io buffer.
Linux holds the inode lock across `generic_file_write_iter` and computes the `O_APPEND`
offset under it, so one `write()` to a regular file is atomic against other writers
regardless of size. Windows implements append as an unjoined lseek-to-end followed by a
write, which is exactly the numbers they saw and exactly why their rewrite did not help.

**Disposition: not a separate issue.** The harness targets a Linux sandbox; the manifest is
only ever written from there, and the docstring is true where the code executes. The
contributor was right to report it and right to flag the platform; the platform turns out
to be the whole story. Their lock file stays — cheap, correct, theirs to make portable.
The unqualified "That reasoning is wrong, and it was inherited rather than measured" in
`search.py::_append` needs the qualifier; `docs/found-by-using-it.md` already carries one.

### Known and accepted: the ranking favours the earned tier

The two-tier separation is genuinely §9-compliant at the level of the corpus — separate
files, `tier` on every row, no `reference_value` in the vendor tier, verdicts derived from
`phases.json`, unread values null, `distributions()` per tier and never merged.

But `regime` returns a single merged **ranking**, and the scoring function systematically
favours earned rows: `text_of()` scores a row's notes, artifact kinds and rung names, and
only earned rows have any of those. In a four-row corpus (3 tutorials, 1 study, identical
solver and regime) the study won 10-9 on a single extra `text~kEpsilon`. That is §9's
closed loop in miniature, arising from the **scoring function** rather than any operator
choice. The tier is printed and the whole ranked list is logged, so it is auditable — but
auditable is not the same as right, and §9's warning is precisely about the earned tier
outweighing the validated one. **Fixed in the PR** (must-change 5), not accepted: tie-break
vendor ahead of earned, or drop `notes`/`artifacts` from `regime` scoring and keep them for
`failure`, where the tier is already pinned to earned.

### Scope of the earned tier, which nothing currently states

`study_roots(work)` walks one level under `/work` — the instance's own Volume. So the
earned tier is **per instance**, not per user and not global, with two consequences that
belong in a separate issue rather than in this PR:

- **`delete_instance` destroys it**, because it destroys the Volume. And local issue 16 is
  precisely about instances whose only remedy for a full 20 GB quota *is* deleting the
  instance — so the recovery path for one issue silently wipes the corpus of another.
- **`FOAMD_MAX_CONCURRENT_INSTANCES` is 1**, so per-instance is effectively per-user
  today, but not reliably: a user who deletes and recreates starts empty with nothing
  saying what was lost.

Durability of the earned tier is a foamd capture-plane question, not a toolbox one.

**Also worth knowing, and contract-consistent:** the earned tier only ever sees studies
where the agent chose to use `study_state.py`, and `toolbox/notes/bundle-layout.md` opens
by calling that "a suggestion, not a convention you are expected to follow". Coverage is
gated on adoption of an optional script. Do not expect it to be complete.

**Nice to have, not blocking:** print `query.tier` in `search.py log` (a keyword query
currently prints two identical-looking lines); remove or make reachable the dead `skipped`
counter in `harvest_studies`; guard or document `foam_fork(work)`'s unbounded `rglob`;
fix earned-tier family naming, which falls back to `commonpath` and produces "The largest,
case, holds 1 of the 1 (100%)".

---

# Collation decisions, after the wave ran

The eight agents landed. What follows is decided during collation, mostly on findings the
wave itself produced.

## C1. One gmsh, from the wheel, at 4.15.2

**What was found.** `pip install gmsh` is unpinned and gives **4.15.2 / OCC 7.8.1** while
the apt CLI is **4.12.1 / OCC 7.6.3** — two versions of the same library on one image, and
the agent has a shell and will mix them. It also breaks the premise §9 was written on:
4.15 **does** expose `Mesh.StlLinearDeflection` and `StlAngularDeflection`, which 4.12 does
not. Separately, the wheel drops a console script at `/usr/local/bin/gmsh` that shadows the
apt binary with the shebang `#!/usr/bin/env python`, a name Ubuntu does not provide — so a
bare `pip install gmsh` leaves every `gmsh` command in every workspace failing.

**Measured, not assumed:** that console script is a **real, complete gmsh CLI**, six lines
calling `gmsh.initialize(sys.argv, run=True)`. It reports 4.15.2 / OCC 7.8.1 and carries
strictly more than apt's build (Netgen, Mmg, PETSc, Med, CGNS besides).

**Decision.** Drop the apt package, **pin the wheel at 4.15.2**, repair the shebang to
`/usr/bin/python3`. One gmsh — module and CLI, one build, one OCC — so no script can get
different tessellation behaviour depending on which it reached for. Unpinned was never an
option: a rebuild touching nothing else would change the image's CAD behaviour.

Safe rather than a gamble because the CAD work was **already built and gated against
4.15.2 / OCC 7.8** — it is A1's 4.12 measurements that are now the stale ones — and the
shim agent's gate already asserts `command -v gmsh` resolves and runs, so a broken entry
point fails the build. Re-point that assertion at the new path. Cost: the wheel bundles its
own OCC, so the layer is larger. Licence unchanged; gmsh is GPL either way.

**Re-run on the final image, not the probe container:** the unit-declaration behaviour the
refusal text describes. It was measured on 7.8.1, so it should hold — but it is exactly the
class of thing that moves between OCC versions.

## C2. Use 4.15's deflection control — as an addition, and only after measuring

Now that the version is pinned above 4.12, `Mesh.StlLinearDeflection` is available. It
controls chord deviation directly, which is the sagitta `cad_convert.py` currently computes
and reports as a *consequence* of `clmax`.

**It does not replace `clmax`,** because the two bound different things. Deflection bounds
how far a facet departs from the surface — on a flat face that is satisfied by arbitrarily
large triangles, which is geometrically exact and still a bad input: snappyHexMesh uses the
STL triangles for surface refinement and proximity detection, and `surfaceFeatureExtract`
works off them. `clmax` bounds edge length, which is what ties tessellation to `dx_surface`.

**Decision.** Keep `clmax` primary; add `--deflection` as a second, optional, **reported**
control for curvature fidelity. Both in the log line, neither with a silent default, and the
no-tolerance refusal unchanged — something must still be said.

**Two measurements first, because the option's name is suggestive rather than definitive:**
whether `Mesh.StlLinearDeflection` governs OCC tessellation on the path we use (it may only
affect STL *import*), and how it interacts with `MeshSizeFromCurvature`, which does the
curvature job today. One measurement each on the new image; `cad_convert`'s fixture already
has a curved boss to measure against. **Do not build on it before then** — a knob that
silently does nothing is worse than one that is not there.

## C3. Prune the corpus harvest walk

**What was found.** `corpus`'s harvest is linear in **paths under `/work`**, not in studies:
`case_roots` `rglob`s each study, so every `processor*/<time>/` directory is walked. A
realistic decomposed shape — 42 studies x 16 ranks x 40 write times x 8 fields — is 243,264
paths. Measured at 0.47 s, but on a tmpfs `/tmp`, i.e. memory speed. Against the only
Volume datapoint this codebase has (~236 us/path, from the 21 s over ~89k files), the same
tree projects to **~57 s at session start**, and 606k paths to ~2.4 minutes.

**Decision: prune `processor*` and numeric time directories from the walk.** What
`case_roots` looks for is `system/controlDict`, which cannot be inside either, so this is a
safe cut of orders of magnitude. Land it with PR #22.

**Deliberately not added:** a reported harvest duration. It was proposed so a slow harvest
would be visible rather than merely survived; declined to keep the change small. The 300 s
timeout and the best-effort yellow line remain the failure mode.

## C4. Rejected Modal credentials get their own error — implemented

**What was found.** §1's handler catches `modal.exception.Error` as a base class, which is
right: it covers all 43 current subclasses and any Modal adds later, where a hand-written
list would let a new one silently restore the bodyless 500. But that base includes
`AuthError`. So rejected or expired Modal credentials answered `503 sandbox_unavailable`
with a `Retry-After` — and `backend/hosted.py` has 503 in `_RETRY_STATUSES`, so a caller
spent four backoffs on a fault no amount of waiting fixes, and was then told the sandbox
was busy.

**Decision: carve it out.** `500 modal_auth_failed`, no `Retry-After`, Modal's own sentence
carried into the body, and a message that says explicitly it is a deployment fault, not a
problem with the request, and that retrying will not clear it.

**The principle behind overruling the alternative.** Leaving it uniform was argued on the
grounds that a credentials failure is total and loud in the logs within seconds. That is
true and not the point: **a tool should lean towards verbosity, and most of all an
agent-facing one.** An agent reading "unavailable, retry shortly" burns four backoffs and
then reasons about the wrong thing, when one accurate sentence would have stopped it. The
cost of saying it plainly is nothing.

**This is not the taxonomy §1 declined to build.** One exception whose cause is known
exactly, named rather than classified. The refusal to sort Modal's errors into transient
and permanent stands — guessing wrong there is what put four Sandboxes on one Volume.
500 rather than 503 because it is an internal fault of this service; no `Retry-After`
because there is no time at which retrying helps.

**Status: implemented** in `app/main.py`, with three tests. The existing whole-module
subclass sweep now asserts the carve-out is exactly one class and every other subclass --
named or not yet written -- still reaches the 503 handler. Suite: 504 passed, 1 skipped.

**Noticed while running it:** `tests/test_metering.py::test_reaper_isolates_a_bad_row`
fails only sometimes. `_sweep_orphans` is not stubbed there, so it reaches the real
`modal.Sandbox.list` when the box permits and then dies on a missing `SUPABASE_URL`.
Environment-dependent and pre-existing (it fails identically on a pristine `git archive
HEAD`); worth filing against that test, not against this wave.

## C5. `X-Forwarded-For` is safe as deployed — measured, no code change

**The concern.** `client_ip` reads `X-Forwarded-For` and takes the **first** entry, in both
`ui`'s new limiter and foamd's `app/ratelimit.py:58`. The header is a chain that each proxy
**appends** to, so the leftmost entry is whatever the *client* wrote — attacker-controlled.
If it is trusted, a caller sending a fresh value per request gets a fresh bucket per
request and the address half of the limiter stops existing. That half is the entire
backstop for anything not pinned to a verified subject: anonymous traffic, junk-token
traffic, and the unclaimed-route bucket.

Worth being clear about what the value is *for*: not the true origin, which is unknowable
from an untrusted header and meaningless for a caller behind a carrier NAT. A limiter needs
a **stable key the caller cannot cheaply rotate**, and "the address our own edge observed"
is exactly that.

**Measured against the deployed service, 2026-09-06.** No route echoes what the container
sees, so the limiter was asked instead: 200 bursted `GET /health`, each carrying a
**distinct forged `X-Forwarded-For`**. Result: **128 served, 72 refused with 429.** Had the
forged entry been the bucket key, all 200 would have had their own allowance and none would
have been refused; instead they shared one bucket, and 128 is the 120 allowance plus refill
during the burst. **Modal's edge normalises the header.** `ui` sits behind the same edge.

**Decision: no code change.** Taking the last entry defensively was rejected — if the edge
replaces the header, last and first are the same value, and if an unknown intermediate hop
exists, either choice can be wrong in a way only measurement distinguishes. Guessing the
hop count too high collapses every caller into one global bucket: a self-inflicted outage,
strictly worse than the exposure.

**What is recorded instead**, because this is correct by property of the deployment and not
by construction — an added proxy or a changed edge would silently end the address half with
no error and no failure, just a limiter that no longer limits:

- Both `client_ip` docstrings state that the first-entry read is safe **only** because the
  edge overwrites, with this measurement and its date.
- `xff_probe.py` is kept as a runnable check rather than a one-off. It is ~10 lines and
  re-answers the question after any infrastructure change.
- `ui`'s `test_a_spoofed_forwarded_for_still_mints_a_fresh_bucket` stays. It documents the
  code's behaviour in isolation honestly, and it is what keeps the deployment dependency
  visible rather than latent.

**Noticed while probing, and nobody chose it:** `/health` is subject to the general
allowance like every other route, and it is what an uptime monitor polls. A monitor
exceeding ~2 requests/second from one address would start receiving 429s. Not a problem
today; worth knowing before one is configured.

## C6. The unowned collation work — done

Five small items no wave agent could take, because each needed a file another agent owned.

1. **§7 reaches the surfaces a person actually looks at.** `view.show_files` and
   `tui.FilesTree.load` rendered entries and nothing else, so a capped listing was still
   mute in the interactive `/files` view -- "a truncation flag nothing displays is the
   same bug again". Both now print the notice. Read off the `Listing`, not the slice:
   slicing a list subclass returns a plain list and drops it.
2. **`mirror.py:377`'s duplicate heuristic is gone.** It counted `len(entries) >=
   MAX_ENTRIES` from the outside and phrased its own warning; it now asks
   `entries.truncated` and reports `entries.notice`, so the wording lives in one place.
   Its boundary was also wrong: a workspace holding **exactly** the cap is a complete
   listing, and it called that truncated. `browse` asks `find` for one line more than it
   keeps, so the flag is measured rather than inferred.
   `tests/test_mirror.py` encoded the old boundary and was corrected, with a companion
   test pinning the other side of it.
3. **`gate_live.py`'s `MOTORBIKE_SOLVE` fallback is deleted.** It sourced `RunFunctions`
   itself and hand-rolled a `restore0Dir` replacement if the helper was missing -- which
   is why this gate stayed green for the entire life of local issue 11: the product could
   not run a tutorial idiom in a job, and the gate could, because the gate arranged the
   environment it was supposed to be measuring. `restore0Dir -processor` is now called
   bare, and that *is* the assertion.
4. **`model_not_priced` moved to `errors.py`**, where every other constructor lives. It
   takes the config key as an argument rather than deriving it, so `errors.py` goes on
   knowing nothing about `config`.
5. **foamd's `_MAX_BUCKETS` -- reverted, and the "fix" was the bug.** See C7.

Suites after: foamd **504 passed, 1 skipped**; OpenReynolds **1805 passed, 13 skipped**;
ui **199 passed**.

## C7. `_MAX_BUCKETS` is a prompt-to-prune mark on purpose, in both repos

The `ui` agent reported that `_MAX_BUCKETS` "was a wish, not a bound" -- `_prune` drops
only buckets idle over 60 s, so traffic from more than 20,000 distinct keys inside one
minute frees nothing and the map grows for as long as the traffic lasts. It changed `ui`
to evict the oldest tenth when nothing is stale, and it was carried into foamd during
collation on the same reasoning.

**Both are reverted.** foamd's `tests/test_ratelimit.py::test_a_burst_of_new_subjects_
cannot_evict_a_live_bucket` failed immediately, and it is guarding a documented decision:
`_prune`'s own docstring already said `_MAX_BUCKETS` is "a prompt-to-prune mark rather
than a hard ceiling… Overshooting a dict of small lists is the cheaper failure."

The test is right and the reasoning is worth restating, because it will tempt somebody
again. Evicting active buckets means **a caller who can mint keys can flush their own
spent bucket and start over with a full allowance.** A limiter that can be flushed does
not limit. Weighed against a map that grows under a flood of a million distinct keys
inside one minute and shrinks a minute later, the memory is the cheaper failure by a wide
margin -- and it is a failure anyone would notice, where a silently defeatable limiter is
not.

`ui`'s `_prune` now carries the reasoning in full, names the date it was "fixed" and the
test that caught it, and says not to re-fix it without answering that test. `ui` also
gains a mirror of foamd's test, which it did not have -- which is why the port could be
changed without anything objecting.

**The general lesson, for the next port:** the ui limiter was copied from foamd's *code*
and not its *tests*, so a documented trade-off arrived looking like an oversight. When a
module is ported, its invariant tests are part of it.

## C8. The two branch conflicts were not conflicts

Both files that PR #22 and the CAD work touch turned out to touch different parts of them,
verified by an actual three-way merge (`git merge-file`, base `HEAD`, ours = the wave's
working tree, theirs = `pr22-corpus`) rather than by reading:

- **`openreynolds/prompt.py`** — PR #22 rewrites the sentence about what the rest of the
  volume holds ("searchable alongside the tutorials"); the CAD work rewrites the paragraph
  about what `python3` and gmsh provide. Different paragraphs, clean merge.
- **`openreynolds/toolbox/README.md`** — PR #22 adds the `corpus.py` and `search.py` rows;
  the CAD work edits the `geometry_view.py` row and adds `cad_convert.py`. Different table
  rows, clean merge.

Both merged results are now in the working tree, so the eventual branch merge has nothing
left to resolve.

**The thing that could have bitten and did not.** `tests/test_prompt.py` caps
`SYSTEM_PROMPT` at 6000 bytes, and the CAD change had left **6 bytes** of headroom by
compressing the paragraph around its new fact. Two independent edits to a byte-capped
prompt is exactly the shape that fails at merge time rather than in either branch. It
came out at **5987** — PR #22's rewrite is shorter than what it replaced, so it handed 6
bytes back. Worth noticing that the margin is now the binding constraint on that file: the
next fact added to the prompt has to displace one.

## C9. What the end-to-end gate found, and the two fixes it forced

The final gate built one scenario crossing the whole wave, in a container whose layer
chain is **generated from `image/image.py`'s own constants** rather than paraphrased.
**37 of 40 checks held**; the three reds were two real defects, not gate noise.

**What the chain proved.** A STEP block -> `cad_convert --clmax 0.0015` -> 4,732 facets,
sagitta 4.69e-5 m, rescaled to 0.04 x 0.024 x 0.025 m -> `geometry_view` and
`cells_estimate` both read it with **extents agreeing exactly** -> a `blockMeshDict` that
computes its vertices in C++ at read time (so the mesh cannot exist without runtime
compilation) meshed -> a tutorial-idiom script ran as a detached `jobd` job to **rc=0**,
`simpleFoam` under `mpirun` as root on the environment alone -> `corpus.py build` indexed
556 vendor cases **and the study that same run had just finished**, returned by a `regime`
query at the top score.

**Every negative control fired.** The same `blockMesh` with `LD_PRELOAD` unset: `rc=1`,
administrator-rights fatal, in the detached job as well as the exec. The same job through
the pre-wave wrapper sliced out of `git show HEAD:`: `rc=127`, `restore0Dir resolves to
[]`. So steps 3 and 4 are green **because of** the changes.

**Seams answered.** `LD_PRELOAD` reaches a detached job's `setsid` child (the inheritance
assumption the `setpriv` rejection partly rested on -- verified, not assumed);
`OMPI_ALLOW_RUN_AS_ROOT` arrives the same way and `mpirun` ran as root with no flag in the
script; both new image layers sit behind HiSA and imageio; and `cad_convert`'s unit
refusals behave identically on OCC 7.8.1.

### Defect 1 — C1 was decided and never implemented. Fixed.

C1 was recorded *after* the wave's agents had finished and was never given an owner, so
the tree still had the apt package, a bare unpinned `pip install gmsh`, and the wheel's
console script **deleted** -- which is what kept the CLI at 4.12. Worse, `_GMSH_CLI_GATE`
asserted `command -v gmsh` = `/usr/bin/gmsh`, so **a gate was holding the defect steady**,
which is worse than no gate.

Implemented as C1 specifies: gmsh out of the `apt_install` line, wheel pinned at
`gmsh==4.15.2`, console script's shebang repaired to `/usr/bin/python3` rather than the
script removed. The gate now asserts the property that matters -- the CLI and the module
report the **same** version, no apt gmsh remains, and the CLI has OCC -- and was checked
to fail when they disagree. Two tests were rewritten (one of them the gate that pinned the
skew) and two added, on the pin and on the apt removal. `cad_convert.py`'s docstring
claiming "gmsh 4.12 surfaces no single tessellation-tolerance one at all" is corrected: it
is false of the build now installed, and C2's premise is recorded there instead.

### Defect 2 — helpers die in a child process. Fixed without enumerating them.

Measured: the wrapper shell has `restore0Dir` as a `function`; `bash -c 'type -t
restore0Dir'` in a child answers **MISSING-IN-CHILD**. Exported *variables* are inherited
by a child process; shell *functions* are not. So a job whose command is the script
**text** works -- which is what §5's gate and the primary run both do -- but

    job_start "bash ./solve.sh"   ->   rc=127, restore0Dir: command not found

the obvious way to run a case written to a file. That is §5's own symptom one layer out,
in exactly the case §5 exists for: a case copied from a tutorial idiom into a file. A
stock `Allrun` is unaffected, because it sources RunFunctions on its own line 3.

**`export -f` was rejected.** It takes a list of names: it covers the four anyone thinks
of and silently misses the other eleven and anything OpenFOAM adds later.

**Fix: one init file, reached two ways, naming no function.** `/etc/foamd/shellinit.sh`
sources the bashrc and then `RunFunctions` -- whatever that file defines is defined. Bash
reads it as `BASH_ENV` (every non-interactive bash, at any depth, which is the child case)
or through `/etc/profile.d/foamd.sh` (login shells, which is foamd's own `bash -lc`); it
reads one or the other, never both. The only literals are the RunFunctions path and
`WM_PROJECT_DIR`, both of which `_SOURCE_OF` already contained and both of which are
OpenFOAM's own contract -- every stock `Allrun` names that same path.

**The guard is not optional and is measured.** `BASH_ENV` fires on every non-interactive
bash, so an unguarded init would make a script spawning `bash -c` in a loop pay the
bashrc every iteration. When `WM_PROJECT_DIR` is already set it was inherited, so only
RunFunctions is sourced. Five child shells: **35 ms cold vs 12 ms warm**.

**Measured in a real container, with its own negative control:** without `BASH_ENV`,
`MISSING-IN-CHILD`; with it, `function` -- in a child bash, in a script file, and through
`profile.d` in a login shell. The image gate asserts all four, the negative control
first, so it cannot pass because the parent happened to have the helpers.

**`_SOURCE_OF` is kept, not deleted as redundant.** Bash ignores a `BASH_ENV` naming a
file that is not there, so an instance still running an older image keeps working exactly
as before, and `_SOURCE_OF` is the only thing that works at all until it is restarted onto
the new image.

Suites after both fixes: foamd **506 passed, 1 skipped**; OpenReynolds **1806 passed, 13
skipped**.

### Not fixed, reported

- **`corpus_gate.py` needs `WM_PROJECT_VERSION`** to pass on a box with no OpenFOAM (1 of
  16 checks); `WM_PROJECT_VERSION=v2512 python3 scripts/corpus_gate.py` -> 16/16. A CI
  note, not a defect in the wave.
- **`regime` returns the earned study tied at the top score but at rank 148 of 355**, with
  147 vendor rows ahead of it. That is §10 must-change 5 working as designed; a caller
  reading only the top N will not see the earned tier, and `--tier earned` is the way to
  reach it. Worth knowing before anyone tunes the ranking again.
