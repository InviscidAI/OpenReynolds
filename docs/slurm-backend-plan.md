# A Slurm backend: the plan, and the one thing it needs that this repository has not got

The 120-item roadmap (an internal document, kept outside this repository) lists "HPC
clusters (Slurm, SSH)" as **out**, and the reason it gives is not cost:

> a `SlurmBackend` over `sbatch`/`squeue`/`scancel` is a client like `LocalBackend` and is
> implementable in a few hundred lines. It is **out because nothing here can verify it**:
> there is no Slurm cluster to run it against, and a backend whose only test is a fake
> `sbatch` on `PATH` would ship subtly wrong and look tested. The honest version is a plan
> plus one real cluster to measure against.

This is that plan. It exists so that whoever has a cluster spends their time measuring
rather than deciding, and so the next person reading that "out" finds a design
instead of a shrug.

## What a backend has to be

`openreynolds/backend/base.py` defines the surface. `LocalBackend` (`local.py:322`) is the
closest relative: it runs commands on a filesystem it can see, and a Slurm login node is
the same shape with a queue in the middle. The methods that matter, and what each becomes:

| `Backend` method | `LocalBackend` | `SlurmBackend` |
|---|---|---|
| `exec(cmd, cwd, timeout_s)` | `subprocess` under a login shell | the same on the **login node**: this is for `ls`, `checkMesh`, `foamDictionary`, not for solves |
| `job_start(...)` | spawn, record a PID, tail a log file | `sbatch --parsable` a script; the job id is Slurm's |
| `job_status(job_id)` | `waitpid` / `/proc` | `squeue -j <id> -h -o %T` while queued or running, then `sacct -j <id> -o State,ExitCode` once it leaves the queue |
| `job_tail(job_id, offset)` | read the log from `offset` | the same read, of the `--output` path the batch script was given |
| `job_kill(job_id, signal)` | `kill -SIG` | `scancel --signal=SIG <id>` |
| `put_file` / `get_file` / `put_tree` / `get_tree` | direct filesystem | direct if the login node shares the scratch filesystem, else `scp`/`rsync` |
| `stat` / `list_stored` / `real_path` / `advertised` | filesystem walk | unchanged, on the shared scratch path |

So the work is `job_start`, `job_status` and `job_kill`. Everything else is `LocalBackend`
with a different root, which is why subclassing it rather than `Backend` is the shorter
road -- the same relationship `DockerBackend(LocalBackend)` already has.

## The four things that will actually be wrong

These are the reasons a fake `sbatch` test would pass while the real thing failed. Each
one is a measurement to take on the first cluster, not a decision to make here.

**1. A queued job is not a started job.** `LocalBackend.job_start` returns once the
process exists. `sbatch` returns once the job is *accepted*, which may be hours before it
runs. Every caller that treats "started" as "running" is wrong against Slurm:
`Loop`'s poll, `progress.py`'s elapsed time, and the `job_status` contract's `started_at`.
`JobStatus` already has the vocabulary -- `status`, `started_at`, `end_reason` -- so the
mapping is `PENDING -> queued`, `RUNNING -> running`, `COMPLETED -> exited`,
`FAILED|NODE_FAIL|OUT_OF_MEMORY -> exited` with `end_reason`, `TIMEOUT -> exited` with
`end_reason="walltime"`, `CANCELLED -> killed`. **Measure:** how long a trivial job sits
`PENDING` on the target cluster, and whether anything upstream times out waiting.

**2. `squeue` forgets.** A finished job leaves the queue, and `squeue -j` then exits
non-zero or prints nothing -- which is indistinguishable from "no such job" and from a
transient `squeue` failure. `sacct` remembers, but only if accounting is enabled, and on
some clusters it is not. **Measure:** whether `sacct` answers on the target cluster, how
long after the job ends, and what it says for a job that was cancelled while `PENDING`.
The fallback when accounting is absent is the batch script writing its own exit code to a
sentinel file, which is worth implementing from the start because it also survives a
`sacct` outage.

**3. The exit code is two numbers.** `sacct`'s `ExitCode` is `code:signal`. A job killed
by the out-of-memory handler reports `0:9` with `State=OUT_OF_MEMORY`, not a non-zero
code, so reading the left half alone calls an OOM kill a success. OpenFOAM's own
`exit -8` SIGFPE -- which this repository has already met twice in `mhdFoam` -- arrives as
`0:8`. **Measure:** what `sacct` reports for a deliberately OOM-killed job and for one
that raises SIGFPE.

**4. `/work` is not `/work`.** `WORKSPACE_ROOT` is `/work`, and `LocalBackend` already
rewrites that to a real root (`local.py`'s `_WORK` regex, which needed a lookbehind
because the default root itself ends in `/work`). A cluster's scratch is per-user and
often per-job (`$SLURM_TMPDIR`), and a path that exists on the login node may not exist on
the compute node. **Measure:** whether the target cluster's scratch is shared between
login and compute, and whether `$SLURM_TMPDIR` is purged at job end -- because if it is,
`get_tree` after a run returns nothing and the study loses its case.

## What the batch script has to carry

`job_start` writes a script rather than passing a command, because Slurm needs the
resource request alongside it and because a script is what a person can re-run by hand:

```bash
#!/bin/bash -l
#SBATCH --job-name=<study>-<phase>
#SBATCH --output=<log path on shared scratch>
#SBATCH --ntasks=<ranks>          # from advise.py, which already recommends this
#SBATCH --time=<walltime>
#SBATCH --partition=<from config>
set -euo pipefail
source <OPENREYNOLDS_FOAM_BASHRC>
cd <case>
<the command>
echo "$?" > <sentinel>            # survives sacct being unavailable (see 2 above)
```

`advise.py` already reads cores three ways and recommends a rank count with a measured
speed-up beside it, so the `--ntasks` value is not a new decision. The walltime is, and it
should come from the config rather than be guessed: a job killed at the walltime looks
exactly like a diverged solve unless `end_reason` says otherwise, which is point 3.

## How to test it honestly

The thing the roadmap refuses is a test suite that passes against a stub. What *can* be
tested without a cluster, and should be:

- the **script writer**, as text. Given a rank count, a walltime and a command, the
  `#SBATCH` lines are exact. This is `tests/test_toolbox_templates.py`'s shape and needs
  no Slurm.
- the **state mapping**, as a table. Every `squeue`/`sacct` state string this repository
  knows maps to exactly one `JobStatus.status`, and an unknown string is **refused by
  name** rather than defaulted to `running` -- which is the failure that would hide a
  node failure as a long solve.
- the **`ExitCode` parser**, including `0:9` and `0:8`, against the `State` beside it.

What cannot be faked, and is the deliverable the roadmap asks for:

- one real `sbatch` of a trivial job, measuring the four items above;
- one real OpenFOAM solve, with `get_tree` afterwards, confirming the case comes home;
- one deliberate `scancel` while `PENDING` and one while `RUNNING`, confirming both arrive
  as `killed` and not as `exited`.

Until those three have been run on a cluster somebody actually has, this stays **out**,
and the roadmap should keep saying so. A backend that is 90% right about a queue is worse
than none: it reports a job as running that is waiting, or as succeeded that was killed,
and the study's transcript then records a conclusion drawn from a lie.

## Why not SSH as a separate backend

The roadmap's item is "HPC clusters (Slurm, SSH)". SSH is not a second backend: it is a
transport under the same one. `exec`, `put_file` and `get_file` over `paramiko` or a
plain `ssh`/`rsync` subprocess give `LocalBackend`'s behaviour on a remote filesystem, and
`SlurmBackend` then sits on top of whichever transport reaches the login node. Building
SSH first and Slurm on it is the shorter path and the one that gets a usable intermediate:
a remote machine with OpenFOAM and no queue at all, which is what most people calling
something an "HPC cluster" actually have.
