# Felix mode — acceptance tests

Felix mode ([felix-mode-plan.md](felix-mode-plan.md)) is done when every goal below
has a passing test and both capstones pass. Each goal names the repo and the test that
proves it. The tests are written before the code they test.

Repos: **OR** = OpenReynolds, **foamd** = `../OpenFoam_Instance`, **ui** = `../ui`.
felixd (`../felixd`) is deployed and is not changed here.

Test layers:
- **unit / api**: each repo's in-process suite with its existing fakes (foamd:
  `fake_postgrest`, `fake_ec2`; OR: the fake backend). felixd is a fake that serves
  felixd's real public API shapes (`tests/fake_felixd.py` in foamd, shared by OR).
- **image**: the Felix workspace container, built locally (`pytest -m docker`).
- **capstone-local**: everything real on this machine except EC2 and Modal (marker `e2e`).
- **capstone-staging**: the real thing in staging, run by hand, result recorded below.

## Contract between the repos

- **Study and workspace**: `POST /v1/studies {solver: "openfoam"|"felix"}`;
  `POST /v1/instances {kind: "openfoam"|"felix"}`; a study binds only to an instance of
  its own kind.
- **Felix routes on foamd** mirror felixd's public paths under `/felix`, so
  `felixclient.Client(f"{foamd_url}/felix", token)` works unchanged: `GET /felix/v1/gpus`,
  `POST /felix/v1/solves`, `POST /felix/v1/solves/{id}/start`, `GET /felix/v1/solves/{id}`,
  `GET /felix/v1/solves/{id}/log?after&limit` (pages; no SSE), `POST /felix/v1/solves/{id}/kill`,
  `GET /felix/v1/solves/{id}/outputs`. Bodies and answers are felixd's; foamd sets
  `end_user` itself and ignores the caller's.
- **Workspace token**: `fxw.<workspace_id>.<hex>`, hex = HMAC-SHA256 of
  `felix:<workspace_id>` under `FOAMD_EC2_DAEMON_HMAC_KEY`. It goes in a Felix
  instance's user data as `felix_token`. first-boot writes `/etc/reynolds/felix.json`
  `{"foamd_url": ..., "token": ...}`, mode 0640, owner root, group `reynolds`.
- **`felix` command** (in the Felix image): reads `/etc/reynolds/felix.json` (or
  `$REYNOLDS_FELIX_CONFIG`). Subcommands: `felix gpus`, `felix run <case> --gpu G [--wall S]` and
  `felix continue <case> --gpu G [--wall S]`. Log lines go to stderr as they arrive, and
  the final solve status goes to stdout as JSON. Exit 0 = succeeded, 1 = the solve failed or
  was killed or refused, 2 = refused locally before submitting.
- **GPU billing**: `felix_solves.cents` = felixd `cost_usd` × 100, with no multiplier.
- **foamd config**: `FOAMD_FELIXD_API` (plain env), `FOAMD_FELIXD_KEY` (secret),
  `FOAMD_EC2_FELIX_IMAGE`.

## F. foamd — studies and workspaces

| # | Goal | Test |
|---|---|---|
| F1 | `POST /v1/studies` takes `solver: openfoam\|felix` (default openfoam, anything else 422); `GET` shows it; a row from before f29 reads as openfoam. | foamd api: `test_felix_kind.py` |
| F2 | A study binds only to an instance of its own kind; a mismatch is a 400 that names both kinds and writes nothing. | foamd api: `test_felix_kind.py` |
| F3 | `POST /v1/instances` takes `kind`; a Felix instance boots `FOAMD_EC2_FELIX_IMAGE`, never adopts a warm-pool instance, and is refused with a 503 (nothing launched) when no Felix image is configured, or a 400 on a non-EC2 backend. | foamd api: `test_felix_kind.py` |
| F4 | `sql/schema_f29.sql` applies after f28 on a real Postgres, twice (idempotent), and the web app's insert without `solver` still succeeds. | foamd: `test_migrate_pg.py::test_f29_*` |

## G. foamd — Felix solves

| # | Goal | Test |
|---|---|---|
| G1 | A Felix workspace gets a workspace token at boot (`/etc/reynolds/felix.json`, readable by the `reynolds` group, not world). An OpenFOAM workspace gets none. | foamd api: `test_felix_token.py`; image: `test_felix_image.py` |
| G2 | The token authenticates only the `/felix/v1/*` routes, only for its own workspace's owner, and stops working when the workspace is deleted or is not a Felix workspace. A forged or other-workspace token is a 401. A user API key also works on these routes. | foamd api: `test_felix_routes.py::test_auth_*` |
| G3 | `GET /felix/v1/gpus` returns felixd's GPU list (type, memory, $/h) unchanged. | foamd api: `test_felix_routes.py` |
| G4 | `POST /felix/v1/solves` refuses (402, nothing sent to felixd) when the user's budget is spent; otherwise it creates the solve on felixd with `end_user` = the user id and records a `felix_solves` row. | foamd api: `test_felix_routes.py::test_create_*` |
| G5 | Start, status, log page, kill and outputs pass through to felixd, for the owner only (another user's solve is a 404). | foamd api: `test_felix_routes.py` |
| G6 | GPU spend: a solve's `cents` follows felixd's `cost_usd` while running and is final when it ends; it counts in the month and lifetime figures, `GET /v1/account`, and `budget_ledgers`. | foamd api: `test_felix_billing.py`; `test_budget_rpc.py::test_gpu_*` |
| G7 | A running solve whose owner goes over budget is killed by the reaper's Felix pass, which also finalises solves nobody polled to the end. | foamd api: `test_felix_billing.py::test_reaper_*` |
| G8 | felixd unreachable or 5xx is a 503 with felixd's message; a felixd 4xx comes back with its status and message. Nothing is recorded for a create felixd refused. | foamd api: `test_felix_routes.py::test_felixd_errors` |
| G9 | `FOAMD_FELIXD_API` and the `FOAMD_FELIXD_KEY` secret are wired in Terraform for staging and prod; the key never appears in a response, a log line or user data. | foamd: `test_env_parity.py`, `test_felix_routes.py::test_key_never_leaks` |

## W. Felix workspace image (foamd `image/`)

| # | Goal | Test |
|---|---|---|
| W1 | The image runs the workspace daemon, has `felix-tag-mesh`, `felix-check-mesh`, `/opt/felix/docs` (user docs and example cases), Python with the usual geometry/mesh/plot packages, and the `felix` command. It has no OpenFOAM and no solver binary. | image: `test_felix_image.py` |
| W2 | `felix gpus` prints the GPU list as JSON. | image + fake foamd |
| W3 | `felix run <case> --gpu G [--wall S]` checks the case, refuses `output.every: 0` (exit 2, one `ERROR:` line, nothing submitted), packs and uploads the case, starts, prints felixd's log lines as they come, downloads the outputs into `<case>/output/`, prints the final status as JSON, and exits 0 only if the solve succeeded. | image + fake foamd: `test_felix_cli.py` |
| W4 | SIGTERM to `felix run` kills the solve on felixd, still downloads the partial outputs, and exits non-zero. | `test_felix_cli.py::test_term_kills` |
| W5 | `felix continue <case>` starts a new solve from the case's latest snapshot (`io.yaml warm_start:` + remaining steps). | `test_felix_cli.py::test_continue` |

## A. OpenReynolds — the agent

| # | Goal | Test |
|---|---|---|
| A1 | `--solver auto` (the default): no workspace or study until the agent calls `choose_solver` (offered only until then); the choice is recorded at once, then the workspace of that kind is reserved and started and the study created with that solver; from the next turn the solver's own prompt and tools apply. `--solver openfoam\|felix` forces it at the start, as before. Resuming keeps the stored solver and ignores a different flag; a study from before solvers is OpenFOAM; the solver can't change during a study. OpenFOAM studies behave exactly as before. | OR: `test_felix_mode.py::test_solver_*`, existing suite unchanged |
| A2 | A Felix study uses the Felix system prompt; a study still choosing uses the solver-choice prompt (what each solver can and cannot do); the OpenFOAM prompt is byte-identical to today's. Both new prompts pass the same style checks. | OR: `test_prompt.py` |
| A3 | Felix mode doesn't offer the OpenFOAM-only tools (`cad`, `mesh_review`, OpenFOAM case generators); `job_start` runs `felix run` with no OpenFOAM restart guard or trapFpe check. | OR: `test_felix_mode.py::test_tools_*` |
| A4 | The progress parser reads Felix logs: step, time, residuals/health from `step=` lines, and surfaces `ERROR:` (including out-of-memory with its needed GB) and `WARNING:` lines. Recorded logs from real runs are the fixtures. | OR: `test_felix_progress.py` |
| A5 | The results tools read Felix `output/`: `summary.json`, `solution.vtu` and snapshots (fields, slices), probes and forces CSVs (plots). | OR: `test_felix_results.py` |
| A6 | Felix studies run in the web app the same way (study created with its solver, jobs shown live). | ui: `test_felix_study.py` |

## B. Benchmark

| # | Goal | Test |
|---|---|---|
| B1 | `benchmarks/felix/` has prompts with known answers and a `grade.py`, run as `openreynolds --solver felix -p ...`. Cases: lid-driven cavity Re 1000 (Ghia centerline velocities), and one external-flow case with a published drag coefficient. | OR: `test_benchmark_felix.py` (grader unit tests) |

## Capstone 1 — local, end to end (`pytest -m e2e`, foamd repo)

Real: OpenReynolds as a user runs it (CLI, real LLM), foamd under uvicorn against a
real Postgres, the Felix workspace container from W1 under docker with the real
daemon, felixd running locally against this machine's A100 (felixd's own local
capstone setup). Replaced: EC2 (a local docker backend) and Modal (felixd's local
runner).

The agent is given one prompt: *"Lid-driven cavity at Re 1000. Report the
u-velocity along the vertical centerline and compare with Ghia et al."* Expect:

1. The study is created `solver: felix` on a `felix` workspace; the OpenFOAM prompt
   and tools never appear in the transcript.
2. The agent builds the case from the docs, and `felix-check-mesh` passes.
3. It calls `felix gpus` and picks a GPU, saying why (size and price).
4. It starts the solve with `job_start`, follows it with the job tools and reads
   progress.
5. **Stop and continue**: once two snapshots exist, the harness kills the job (as a
   user would). The agent notices, continues from the latest snapshot, and the run
   finishes.
6. Results: centerline u within 2% RMS of Ghia; a plot and the comparison in the
   final report.
7. Billing: foamd's `felix_solves` rows sum to felixd's `cost_usd` for both solves,
   and `GET /v1/account` shows the spend.
8. Refusals: the same user with no budget left is refused at `felix run` with
   foamd's 402 message, and the agent reports it rather than retrying.
9. A deliberately oversized case on the smallest GPU fails with Felix's
   out-of-memory `ERROR:`; the agent picks a larger GPU and the rerun succeeds.

## Capstone 2 — staging (by hand, result recorded here)

`scripts/felix_capstone_staging.py` (foamd repo) against staging foamd, a real Felix
EC2 workspace, staging felixd and Modal:

1. Capstone 1's steps 1–7 through the CLI on H100.
2. The same prompt started from the web app, run to the same result.
3. The workspace token can't reach felixd directly (no felixd key on the instance:
   `grep -r` the volume and the image), and the solver uid's egress reaches only
   foamd and S3.
4. GPU billing: foamd's recorded cents match felixd's usage for the staging service
   account over the run window.

| Run | Date | Result |
|---|---|---|
| | | |
