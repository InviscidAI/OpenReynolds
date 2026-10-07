# Felix mode — plan (2026-10-07)

The agent gains a second solver: **Felix**, our GPU incompressible Navier-Stokes
solver, next to OpenFOAM. This file says what the agent needs to be able to do. How
to build it is left to whoever builds it.

The service side (`felixd`, and foamd's `felix` job kind and Felix workspace type)
is planned in `OpenFoam_Instance/plan-4-felix-gpu-solver.md`.

## What Felix is, as the agent sees it

- **A Felix study has its own workspace**: no OpenFOAM, no solver binary. It has
  the CPU mesh tools `felix-tag-mesh` and `felix-check-mesh`, the Felix user
  documentation at `/opt/felix/docs` (including worked example cases and
  `gpu_sizing.md`), and the usual Python geometry/mesh/plotting tools.
- **A case is a directory** of YAML files plus a tagged `.vtu` mesh. Every setting
  lives in the case files; the solver takes no other input. An illegal case fails
  before any GPU time with one `ERROR:` line naming the file and key.
- **A solve runs on a cloud GPU** the agent picks per solve (A100, H100, H200, B200),
  billed per second. The GPU exists only while the solve runs. Results come back
  into the case's `output/` directory: `summary.json`, `solution.vtu`, snapshots,
  CSVs.
- **A stopped solve can continue exactly** from its latest snapshot.
- **Output contract**: stdout is JSON only; stderr lines start with `ERROR:`,
  `WARNING:` or `NOTE:`, plus one `step=` record per step. Exit 0 means the run
  finished; exit 1 means an error or a non-finite stop.

## What the agent needs to do

1. **Work in Felix mode when a study is a Felix study.** The solver is chosen when
   the study is created and does not change during it. This is a separate choice
   from the existing approval modes.
2. **Know Felix the way it knows OpenFOAM.** It needs its own system prompt, with
   the documentation at `/opt/felix/docs` as its reference. The OpenFOAM prompt
   stays unchanged.
3. **Build cases** from the documentation and the example cases, including getting
   a mesh into tagged `.vtu` form and checking it with `felix-check-mesh`.
4. **Choose a GPU** for each solve from the case's size and the speed and price of
   each GPU type. The current GPU types and prices come from the service, not from
   the prompt.
5. **Start, follow and stop solves** through the service, with the same job
   experience it has for OpenFOAM jobs: live log, status, kill.
6. **Understand a Felix run**: follow its progress, read its health and
   convergence, recognise its errors (including out-of-memory, which says how much
   memory is needed), and continue a stopped run from a snapshot.
7. **Make snapshots happen on every solve,** so a preempted or stopped run can
   continue. How this is guaranteed is open (below), because the harness must not
   block or rewrite the agent's tool calls.
8. **Show results** from Felix output: fields, plots, forces, probes.
9. **Be measured**: a Felix benchmark with known reference answers, so prompt
   changes are judged by results.

Not needed for launch: the `cad` tool producing Felix meshes, and the web app
offering Felix (it can start CLI-only).

## Open questions

1. **How are snapshots guaranteed?** Options: the service refuses a solve without
   snapshots; Felix writes its own resume file on the cloud runner regardless of the
   case settings; or the prompt states the fact and the agent chooses.
2. **The default wall-time limit** for a solve when the agent sets none (the
   service cap is 24 h).
3. **Does the web app offer Felix at launch,** or only the CLI?
