#!/usr/bin/env python3
"""Run the live backend's cases on a real GPU, through Modal, one job at a time.

    MODAL_PROFILE=inviscid python -m modal run benchmarks/live/modal_run.py --case smoke
    MODAL_PROFILE=inviscid python -m modal run benchmarks/live/modal_run.py::layouts
    MODAL_PROFILE=inviscid python -m modal run benchmarks/live/modal_run.py::m1
    MODAL_PROFILE=inviscid python -m modal run benchmarks/live/modal_run.py::everything

Not part of the package. `pyproject.toml` excludes `benchmarks/` from the sdist, and
this file is engineering scaffolding: it exists because the machine this extension was
written on has no CUDA device, and every number in `PROGRESS_LIVE.md` that is about a
GPU has to come from a GPU.

## Why this is a file and not a shell loop

Three things have to be true of every measurement it produces and none of them survive
being done by hand.

**The code under test has to be this checkout, not a release.** The image adds the
working tree -- `openreynolds/`, `benchmarks/` and `tests/data/` -- rather than
`pip install openreynolds`, so a change made thirty seconds ago is what runs. The
ignore list keeps `studies/`, `.git/` and the transcripts out, which is most of the
directory by size and all of the part that names accounts.

**`tests/data/moving_mesh/` has to come with it.** The cross-check reads this
repository's own committed OpenFOAM run through `benchmarks/moving_mesh/grade.py`, and
that run is the OpenFOAM half of the comparison. Without those two directories in the
image the cross-check has nothing to check against.

**Nothing may be left running.** Every function here is a one-shot `@app.function` with
an explicit `timeout`; there is no `@app.cls`, no warm pool, no `min_containers` and no
volume. A job starts, runs one case, returns its result and its figures as bytes, and
the container exits. `everything` runs the cases in `.map`, which Modal fans out and
tears down on its own. The cost of a run is therefore the GPU-seconds of the cases
themselves, and `--gpu` is a flag so that a cheap case does not have to take an H100.

## What it returns

Each function returns the case's ordinary result dictionary -- the same one
`openreynolds live run` prints -- with the figures as `{name: bytes}` beside it. The
local entrypoint writes the figures into `benchmarks/live/results/<case>/` and the
result as JSON next to them, so the numbers that go into `PROGRESS_LIVE.md` are on disk
and re-readable rather than scraped out of a terminal.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

def _local_repo() -> Path:
    """The checkout, as seen from the machine `modal run` was typed on.

    Computed in a function and not at module scope, because **this module is imported
    twice**: once locally, where it sits at `benchmarks/live/modal_run.py`, and once
    inside the container, where Modal has placed the entrypoint at `/root/modal_run.py`
    and `parents[2]` is an `IndexError` -- which is exactly how the first run failed,
    during container startup and before a single kernel compiled.

    Nothing remote needs it: the repository is at `/repo` there, which `_run_remote`
    puts on `sys.path` directly.
    """
    here = Path(__file__).resolve()
    return here.parents[2] if len(here.parents) > 2 else Path("/repo")


def _results_dir() -> Path:
    return Path(__file__).resolve().parent / "results"


REPO = _local_repo()
RESULTS = _results_dir()

# Pinned exactly, and to the versions the extension was written against. A benchmark
# whose dependency floats is a benchmark that cannot be compared with last week's:
# `docs/live-version-matrix.md` records what the extension does across the range it
# supports, and this is the one point in it that the published numbers come from.
WARP = "warp-lang==1.19.0"
NEWTON = "newton==1.8.0"
MUJOCO = "mujoco-warp==4.0.1"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        WARP, NEWTON, MUJOCO,
        "numpy>=1.24", "matplotlib>=3.7", "imageio>=2.31", "pillow>=10",
    )
    .env({
        # Warp's kernel cache inside the container, so a cold start does not try to
        # write to a home directory that may not be writable.
        "WARP_CACHE_PATH": "/tmp/warp-cache",
        "MPLCONFIGDIR": "/tmp/mpl",
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir(
        str(REPO), remote_path="/repo",
        ignore=[
            "studies/**", ".git/**", "**/__pycache__/**", "*.log", "launcher/**",
            ".pytest_cache/**", "benchmarks/live/results/**", "docker/**",
        ],
        copy=True,
    )
)

app = modal.App("openreynolds-live")

GPUS = {"h100": "H100", "a100": "A100-80GB", "a10g": "A10G", "l40s": "L40S"}
"""The shapes worth asking for, by the name a flag uses.

H100 is what the brief's 1000+ MLUPS target names, so it is the default and is what the
published rates are measured on. A100-80GB is the fallback when H100 capacity is short.
The two small ones are for a case that only has to *run* on CUDA rather than be timed
on it, which is most of them: the physics is device-independent and only the rates are
not.
"""


def _sanitise(value):
    """Make a result JSON-serialisable without losing anything that matters.

    numpy scalars and arrays come back from the cases in a few places (a check's
    measured value, a row of a resolution sweep), and Modal has to pickle the return
    value while the local side has to write JSON. Converting here rather than at either
    end means one implementation and no silent `TypeError` thirty minutes into a run.
    """
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _sanitise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitise(v) for v in value]
    if isinstance(value, np.bool_):
        # Before `bool`: numpy's bool is not a Python bool and is not an integer
        # either, so it used to fall all the way through to `str(value)` and arrive
        # as the string "True". Every consumer then read it as truthy.
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return _sanitise(value.tolist())
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _run_remote(case: str, settings: dict, layout: str | None = None) -> dict:
    """The body every remote function shares: run one case, collect its figures.

    Imports happen in here and not at module scope because this module is also imported
    locally, where Warp is not installed and the repository is not at `/repo`.
    """
    import tempfile
    import time
    import traceback

    sys.path.insert(0, "/repo")
    if layout:
        os.environ["OPENREYNOLDS_LIVE_LAYOUT"] = layout

    import warp as wp

    wp.init()
    device = "cuda:0" if wp.get_cuda_device_count() else "cpu"
    started = time.time()
    with tempfile.TemporaryDirectory() as out:
        try:
            from openreynolds.live import cases as case_module

            if case == "benchmark":
                result = case_module.benchmark(device=device, progress=print,
                                               **settings)
            elif case == "crosscheck":
                result = case_module.crosscheck(device=device, out_dir=out,
                                                progress=print, **settings)
            else:
                result = case_module.run_case(case, out_dir=out, device=device,
                                              progress=print, **settings)
            result["report_text"] = case_module.report(result)
        except Exception:
            return {
                "case": case, "passed": False, "checks": [],
                "report": [f"the remote run raised:\n{traceback.format_exc()}"],
                "figures": [], "error": traceback.format_exc(),
                "device": device, "seconds": time.time() - started,
            }
        payload = _sanitise(result)
        payload["device"] = device
        payload["layout"] = os.environ.get("OPENREYNOLDS_LIVE_LAYOUT", "")
        payload["gpu_name"] = _gpu_name()
        payload["seconds"] = time.time() - started
        blobs = {}
        for path in sorted(Path(out).iterdir()):
            if path.is_file() and path.stat().st_size < 40 * 2 ** 20:
                blobs[path.name] = path.read_bytes()
        payload["figures"] = sorted(blobs)
        return {"result": payload, "blobs": blobs}


def _gpu_name() -> str:
    import subprocess

    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return done.stdout.strip().splitlines()[0]
    except Exception:
        return "no nvidia-smi"


# -- the remote functions ---------------------------------------------------------------
#
# One per GPU shape rather than one parameterised function, because Modal resolves `gpu`
# at decoration time. Four decorators is less machinery than a factory.


@app.function(image=image, gpu="H100", timeout=72000)
def on_h100(case: str, settings: dict, layout: str | None = None) -> dict:
    return _run_remote(case, settings, layout)


@app.function(image=image, gpu="A100-80GB", timeout=72000)
def on_a100(case: str, settings: dict, layout: str | None = None) -> dict:
    return _run_remote(case, settings, layout)


@app.function(image=image, gpu="L40S", timeout=72000)
def on_l40s(case: str, settings: dict, layout: str | None = None) -> dict:
    return _run_remote(case, settings, layout)


@app.function(image=image, gpu="A10G", timeout=72000)
def on_a10g(case: str, settings: dict, layout: str | None = None) -> dict:
    return _run_remote(case, settings, layout)


RUNNERS = {"h100": on_h100, "a100": on_a100, "l40s": on_l40s, "a10g": on_a10g}


# -- the local side ---------------------------------------------------------------------


def _provenance() -> dict:
    """The commit this run's code came from, and whether the tree was dirty.

    **Stamped on every result because its absence cost real confusion.** Modal mounts
    the working tree, not a commit (`add_local_dir(REPO)` above), so a job picks up
    whatever is on disk the moment it starts. Editing while runs are in flight is
    therefore editing the experiment: one job in this repository failed on a `NameError`
    that a later edit had already fixed, and two runs an hour apart were not runs of the
    same code -- which is indistinguishable, from the result file alone, from a flaky
    solver.

    `dirty` is the part that matters. A result stamped with a commit and `dirty: true`
    says only "something like that commit", and the honest thing is for the file to say
    so rather than for a reader to assume the commit is exact. To get a result whose
    provenance is exact, run from a frozen checkout:

        git worktree add /tmp/frozen <commit>
        cd /tmp/frozen && MODAL_PROFILE=inviscid python -m modal run             benchmarks/live/modal_run.py::m1 --gpu h100

    `REPO` is resolved relative to this file, so a worktree mounts itself and nothing in
    the main checkout can race it.
    """
    import subprocess

    def git(*args: str) -> str:
        try:
            return subprocess.run(("git", *args), cwd=str(REPO), capture_output=True,
                                  text=True, timeout=15).stdout.strip()
        except Exception:
            return ""

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD") or "unknown",
        "dirty": bool(status),
        "dirty_files": len([line for line in status.splitlines() if line.strip()]),
    }


def _save(payload: dict, blobs: dict, label: str) -> Path:
    """Write one result and its figures under `benchmarks/live/results/<label>/`."""
    payload = {**payload, "provenance": _provenance()}
    out = RESULTS / label
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(payload, indent=2) + "\n",
                                     encoding="utf-8")
    text = payload.get("report_text") or "\n".join(payload.get("report", []))
    (out / "report.txt").write_text(text + "\n", encoding="utf-8")
    for name, blob in blobs.items():
        (out / name).write_bytes(blob)
    return out


def _announce(payload: dict, label: str, where: Path) -> bool:
    verdict = "PASSED" if payload.get("passed") else "FAILED"
    print(f"\n{'=' * 78}\n{label}: {verdict} on {payload.get('gpu_name', '?')} "
          f"in {payload.get('seconds', 0):.1f} s\n{'=' * 78}")
    print(payload.get("report_text") or "\n".join(payload.get("report", [])))
    print(f"  saved: {where}")
    return bool(payload.get("passed"))


MILESTONES = {
    "M1": [
        ("cavity-re100", "cavity", {"cells": 128, "reynolds": 100.0}),
        ("cavity-re1000", "cavity", {"cells": 192, "reynolds": 1000.0,
                                     "u_lattice": 0.055}),
        ("cylinder", "cylinder", {"resolutions": (24, 32, 44)}),
        ("sphere", "sphere", {}),
        ("crosscheck", "crosscheck", {"cells_per_d": 20}),
        ("poiseuille", "poiseuille", {"cells": 64}),
    ],
    "M2": [
        ("translating-sphere", "translating_sphere", {}),
        ("falling-sphere", "falling_sphere", {}),
        # The same case with the moving window on, in a box a third the height, so that
        # what the window costs is a measurement in this repository rather than a
        # sentence in a docstring. It is expected to read high; `run_falling_sphere` has
        # the first number it produced and why.
        ("falling-sphere-windowed", "falling_sphere", {"window": True}),
        ("momentum", "momentum", {}),
    ],
    "M3": [
        ("ground-effect", "ground_effect", {}),
        ("quadrotor", "quadrotor", {}),
        ("underwater", "underwater", {}),
    ],
}
"""The runs behind each milestone, with the settings the published numbers come from.

`cavity-re1000` is **192** cells, which is a measured compromise rather than a round
number. At Re 1000 the Mach ceiling and the stiff-tau floor bite together and need about
115 cells at the absolute minimum, so 128 sits almost exactly on both (tau_plus 0.5211
against a floor of 0.52). Going up buys margin and costs steps twice over, because a
cavity spins up over `cells^2/nu_lattice` steps and `nu_lattice` rises with `cells`:

    cells   tau_plus   one diffusive time   four of them, on an H100
      128     0.5211       2,327,273 steps              5.3 minutes
      192     0.5317       3,490,909 steps             17.8 minutes
      256     0.5422       4,654,545 steps             42.1 minutes

256 was the first choice and it is where the 42 minutes went: the run hit a 4,000,000
step cap less than one diffusive time in, and came out 8 to 10% short of Ghia on every
quantity because the primary vortex was still forming. 192 has margin on both bounds and
a budget it can actually finish.
"""


@app.local_entrypoint()
def main(case: str = "smoke", gpu: str = "h100", resolution: int = 0,
         steps: int = 0, collision: str = "", layout: str = "",
         window: bool = False, label: str = "", reynolds: float = 0.0,
         u_lattice: float = 0.0, extra: str = "") -> None:
    """Run one case on one GPU and write the result under `benchmarks/live/results/`.

    `--label` names the directory, so a variant of a case can be recorded beside it
    rather than over it: `--case falling_sphere --window --label
    falling-sphere-windowed` is how the windowed run gets its own row without the
    graded one losing its.
    """
    settings: dict = {}
    if extra:
        # A JSON passthrough for a case's own knobs, so that probing one of them does
        # not need a new flag. `--extra '{"amplitude": 0.0005}'` is how the sloshing
        # tank's wave steepness was swept to separate a finite-amplitude period shift
        # from a solver error -- linear theory is an expansion in steepness, and the
        # measured period sitting ABOVE it is what a finite amplitude does.
        settings.update(json.loads(extra))
    if window:
        settings["window"] = True
    if reynolds:
        settings["reynolds"] = float(reynolds)
    if u_lattice:
        # The lattice velocity, which is the Mach number: `Ma = u_lattice / c_s`. A flag
        # because it is the one knob that separates a compressibility error from a
        # resolution error -- the same case at the same cells and a lower `u_lattice` is
        # the same discretisation at a lower Mach number, and if the disagreement moves
        # it was `O(Ma^2)` all along.
        settings["u_lattice"] = float(u_lattice)
    if resolution:
        settings["resolution"] = int(resolution)
    if steps:
        settings["steps"] = int(steps)
    if collision:
        settings["collision"] = collision
    runner = RUNNERS[gpu.lower()]
    got = runner.remote(case, settings, layout or None)
    payload = got.get("result", got)
    where = label or case
    ok = _announce(payload, where, _save(payload, got.get("blobs", {}), where))
    if not ok:
        sys.exit(1)


carry_image = image.pip_install("torch==2.15.0")
"""`image` plus PyTorch, which `CarryFluidEnv` cannot run without.

Its own image and not an addition to `image`, because torch's Linux wheel carries the
CUDA runtime and is about 2.5 GB: every other function here is warp-and-newton only and
should not pay for that. One `device` string configures both the Warp allocation and
the torch tensors (`CarryFluidEnvCfg.device` says why two would be worse), so a
`cuda:0` run needs the CUDA build rather than the CPU one.
"""


def _run_onset(settings: dict) -> dict:
    """`carry.onset_convergence` on whatever device this container has.

    Separate from `_run_remote` because the carry task's measurements are not `cases.py`
    cases: they are functions on `openreynolds/live/carry.py` itself, deliberately, so
    the environment and the numbers quoted about it cannot drift apart. It **prints as
    it goes**, and the print goes to Modal's log rather than only into the return value,
    so a run started with `--detach` whose local side loses DNS is still readable.
    """
    import time
    import traceback

    sys.path.insert(0, "/repo")
    import warp as wp

    wp.init()
    device = "cuda:0" if wp.get_cuda_device_count() else "cpu"
    started = time.time()
    try:
        from openreynolds.live import carry

        result = carry.onset_convergence(device=device, **settings)
    except Exception:
        return {"error": traceback.format_exc(), "device": device,
                "gpu_name": _gpu_name(), "seconds": time.time() - started}
    payload = _sanitise(result)
    payload.update(device=device, gpu_name=_gpu_name(),
                   warp=wp.config.version, seconds=time.time() - started)
    return payload


@app.function(image=carry_image, gpu="L40S", timeout=14400)
def onset_l40s(settings: dict) -> dict:
    return _run_onset(settings)


@app.function(image=carry_image, gpu="A10G", timeout=14400)
def onset_a10g(settings: dict) -> dict:
    return _run_onset(settings)


ONSET_RUNNERS = {"l40s": onset_l40s, "a10g": onset_a10g}


@app.local_entrypoint()
def onset(gpu: str = "l40s", radii: str = "0.01,0.005,0.0025,0.002",
          rate: float = 0.1, rates: str = "", settle: int = 100,
          stop_fraction: float = 0.05, label: str = "carry-onset") -> None:
    """The spill-angle convergence on a GPU.

    `python -m modal run --detach benchmarks/live/modal_run.py::onset --gpu l40s`

    `--detach` is not optional in practice: a local DNS failure has killed whole
    batches here, and a detached run keeps going and keeps logging.
    """
    settings = {
        "radii": tuple(float(r) for r in radii.split(",") if r.strip()),
        "rate": float(rate),
        "rates": tuple(float(r) for r in rates.split(",") if r.strip()),
        "settle": int(settle),
        "stop_fraction": float(stop_fraction),
    }
    payload = ONSET_RUNNERS[gpu.lower()].remote(settings)
    out = RESULTS / label
    out.mkdir(parents=True, exist_ok=True)
    payload["provenance"] = _provenance()
    (out / "result.json").write_text(json.dumps(payload, indent=2) + "\n",
                                     encoding="utf-8")
    print(f"\n  {payload.get('gpu_name', '?')}, warp {payload.get('warp', '?')}, "
          f"{payload.get('seconds', 0):.1f} s -- saved {out / 'result.json'}")
    if payload.get("error"):
        print(payload["error"])
        sys.exit(1)


@app.function(image=image, timeout=600)
def _noop() -> str:
    return "ok"


@app.local_entrypoint()
def layouts(gpu: str = "h100", resolutions: str = "128,256", steps: int = 100) -> None:
    """Measure both array layouts on the GPU, which is the only thing that may pick the
    GPU default.

    The CPU measurement in `lattice._preferred_layout` says node-major is three times
    faster on one core. That is a fact about a CPU and it must not be allowed to choose
    anything on a GPU, where the argument runs the other way: a warp's reads of one
    direction are one coalesced transaction under the direction-major layout and
    thirty-two scattered ones 76 bytes apart under the other. Both are measured here, on
    the device, at two resolutions, and whichever wins is what the default should be.
    """
    grid = tuple(int(v) for v in resolutions.split(","))
    runner = RUNNERS[gpu.lower()]
    rows = []
    for layout in ("soa", "aos"):
        got = runner.remote("benchmark",
                            {"resolutions": grid, "steps": int(steps)}, layout)
        payload = got.get("result", got)
        _announce(payload, f"benchmark-{layout}", _save(payload, got.get("blobs", {}),
                                                        f"benchmark-{layout}"))
        for row in payload.get("rows", []):
            rows.append(dict(row, layout=layout))
    print(f"\n{'=' * 78}\nboth layouts on {gpu}\n{'=' * 78}")
    best: dict[str, float] = {}
    for row in sorted(rows, key=lambda r: (r["layout"], r["cells"])):
        print(f"  {row['layout']}  {row['cells']:>5} cells  {row['nodes']:>14,} nodes  "
              f"{row['mlups']:10.1f} MLUPS")
        best[row["layout"]] = max(best.get(row["layout"], 0.0), row["mlups"])
    for layout, rate in sorted(best.items()):
        print(f"  best {layout}: {rate:.1f} MLUPS")
    if best:
        winner = max(best, key=best.get)
        print(f"  fastest on {gpu}: {winner} "
              f"({best[winner] / min(best.values()):.2f}x the slower one)")
        summary = RESULTS / "layouts.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps({"gpu": gpu, "rows": rows, "best": best,
                                       "fastest": winner}, indent=2) + "\n",
                           encoding="utf-8")
        print(f"  saved: {summary}")


def _milestone(name: str, gpu: str) -> None:
    runner = RUNNERS[gpu.lower()]
    plan = MILESTONES[name]
    print(f"running {name}: {', '.join(label for label, _, _ in plan)} on {gpu}")
    # `.starmap` so Modal fans them out: they are independent jobs on independent
    # containers, which is the whole reason this is worth doing remotely.
    outcomes = list(runner.starmap([(case, settings, None)
                                    for _, case, settings in plan]))
    failed = []
    for (label, _, _), got in zip(plan, outcomes):
        payload = got.get("result", got)
        if not _announce(payload, label, _save(payload, got.get("blobs", {}), label)):
            failed.append(label)
    print(f"\n{name}: {len(plan) - len(failed)}/{len(plan)} runs passed"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    if failed:
        sys.exit(1)


@app.local_entrypoint()
def m1(gpu: str = "h100") -> None:
    """M1: the validations with published answers, plus the two-backend cross-check."""
    _milestone("M1", gpu)


@app.local_entrypoint()
def m2(gpu: str = "h100") -> None:
    """M2: moving boundaries, the two-way coupling, and the momentum balance."""
    _milestone("M2", gpu)


@app.local_entrypoint()
def m3(gpu: str = "h100") -> None:
    """M3: the robot demos."""
    _milestone("M3", gpu)


@app.local_entrypoint()
def everything(gpu: str = "h100") -> None:
    """Every milestone, in order, stopping at the first that fails."""
    for name in ("M1", "M2", "M3"):
        _milestone(name, gpu)
