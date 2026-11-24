# `tests/data/live/` -- the live backend's graded GPU runs, as the runner wrote them

One directory a recorded run, each holding the `result.json`
`benchmarks/live/modal_run.py` returned, byte for byte. `tests/test_benchmark_live.py`
asserts that every graded check in every one of them passed, at the tolerance the check
itself states, on a device it names; `benchmarks/live/summarise.py` turns the same files
into `PROGRESS_LIVE.md`'s summary table.

## Why the data is here rather than the runs

The brief for this extension asks that the M1 to M3 validations "pass at the stated
tolerances and live in the test suite", and the two halves of that pull against each
other. The runs cost minutes of H100 each -- the cylinder sweep is three resolutions of
170,000 to 310,000 lattice steps, the Re 1000 cavity spins up over 14 million, the sphere
at Re 200 is 63 million nodes -- and on Warp's single-threaded CPU backend, which is what
CI has, each is hours. This repository's whole suite is budgeted at about 24 minutes.

So they live here the way this repository's other expensive validations already do.
`tests/data/moving_mesh/` holds a finished `pimpleFoam` run trimmed to what its grader
reads, and `tests/test_benchmark_moving_mesh.py` asserts the numbers from it;
`tests/data/felix/` does the same for the Felix solver. The pattern and its limits are
the same here: **this is a regression test on a recorded result, not a re-derivation of
the physics.** A tolerance quietly loosened, a check that stops being graded, a case that
disappears from the catalogue, or a number that drifts from the one
`benchmarks/README.md` quotes will all fail. Nothing here re-solves anything.

What *is* re-derived, every test run, is at the bottom of
`tests/test_benchmark_live.py`: the grader's own exit codes and refusals, and the two
cases small enough to solve on a CPU in seconds. And `tests/test_live_cases.py` solves
the exact Poiseuille case and the 32-cell cavity from scratch in the ordinary suite.

## What was kept

Only `result.json`. Not the figures: `modal_run.py` writes a PNG and sometimes a GIF
beside each result, and the whole set is 3.1 MB of pictures that no test opens and that
are reproducible from the grader in one command. The JSON is about 6 kB a run.

Each file holds the case name, every check with its measured value, its target as a
sentence and its verdict, the report the CLI would have printed, the device name as
`nvidia-smi` reported it, the array layout, the elapsed seconds, and the case's own
extra fields. **Nothing in it is an identifier**: no path, no account, no workspace, no
job id, no cost. The only proper nouns are the hardware's (`NVIDIA H100 80GB HBM3`) and
the case's own.

## The device every number came from

Every graded run here is an H100 80GB HBM3 on Modal, through
`benchmarks/live/modal_run.py`, with `warp-lang==1.19.0`, `newton==1.8.0` and
`mujoco-warp==4.0.1` pinned exactly so that the numbers are comparable with each other.
The `smoke` run is an A10, because it exists to prove the path works rather than to
measure anything.

A run's own `gpu_name` and `device` fields are what `test_every_recorded_run_names_the_device_that_produced_it`
checks, and `PROGRESS_LIVE.md` reprints beside every figure. A rate without a device is
not a measurement; an agreement without one cannot be reproduced.

## Regenerating it

With a Modal workspace configured:

```
MODAL_PROFILE=inviscid python -m modal run benchmarks/live/modal_run.py::m1
MODAL_PROFILE=inviscid python -m modal run benchmarks/live/modal_run.py::m2
MODAL_PROFILE=inviscid python -m modal run benchmarks/live/modal_run.py::m3
MODAL_PROFILE=inviscid python -m modal run benchmarks/live/modal_run.py::main \
    --case quadrotor --gpu h100
```

which writes `benchmarks/live/results/<label>/{result.json,report.txt,*.png}`. Then

```
python benchmarks/live/summarise.py --write        # PROGRESS_LIVE.md's table
cp benchmarks/live/results/<label>/result.json tests/data/live/<label>/result.json
```

`benchmarks/live/results/` is not committed (`.gitignore`): it is a working directory
that fills with pictures, and the two things worth keeping out of it are the summary
table and these JSON files.

On a machine with a CUDA GPU of its own, `benchmarks/live/grade.py <case> --json` writes
the same structure without Modal.

## When these go stale

They go stale the moment a case's setup changes, and the test will not notice -- it
asserts that the recorded run passed, not that the recorded run is of today's code. That
is the honest limit of the pattern and it is the same limit `tests/data/moving_mesh/`
has. The guard against it is the commit discipline: a change to a case in
`openreynolds/live/cases.py` that moves a graded number is not finished until the run has
been redone and the file here replaced, and the commit that does the first should do the
second.
