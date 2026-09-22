#!/usr/bin/env python3
"""One candidate of a parametric study scored into `metrics.json` -- the only writer of it.

    python3 score.py CASE_DIR --lock goal.lock.json --out metrics.json
    python3 score.py CASE_DIR --lock goal.lock.json --out metrics.json --mesh-only
    python3 score.py CASE_DIR --lock goal.lock.json --out metrics.json --label timeout --detail "killed at 3x median"

`parametric.py` runs this last, after the build, the mesh, the dressing and the solve,
and nothing else writes the file: a `hooks.py` cannot leave a number behind that ends up
in the leaderboard, because this reads the case -- `postProcessing/`, the solver log, the
mesh -- and the lock, and nothing a hook wrote. What a candidate is worth is decided in
one place, on measurements, the same way for every point of the study.

what it measures, and how

    Force coefficients are the mean over the last `fidelity.window` fraction of the
    `forceCoeffs.dat` rows (default the last 20%), never the last row: on a steady
    solver a bluff section's coefficients oscillate about their mean, and a point read
    is whichever phase the final write landed on. `window` in the output is the number
    of rows that mean was taken over, so a ten-row window on a 1000-iteration run is
    visible as one. `converged` is `log_digest.residual_shape` reading `levelled` or
    `falling`, or the solver's own "solution converged" line. y+ comes from the `yPlus`
    function object's output when the case has one (`case_gen.py --yplus`), the body
    patch's own row at the last write, min and max. The cell count is read off
    `constant/polyMesh/owner`'s header, which is the mesh's own statement of it.

the label

    One word the optimizer reads, in this order of precedence, the first that applies:
    `mesh` (no polyMesh, or `checkMesh` failed), `diverged` (a FOAM FATAL in the log, a
    residual climbing off its own best, or a field held at its bound for most of the
    run), `unconverged` (a residual shape that is neither levelled nor falling, or no
    residuals at all), `yplus` (outside `fidelity.yplus`), `cells` (outside
    `fidelity.cells`), else `ok`. `timeout` is never decided here -- this cannot know how
    long a run was allowed -- so the caller that killed the run passes `--label timeout`,
    and a stated label wins over everything measured.

The schema is a contract with whatever reads the leaderboard; the keys stay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import log_digest  # noqa: E402
import results  # noqa: E402

LABELS = ("ok", "mesh", "diverged", "timeout", "unconverged", "yplus", "cells")
"""Every label this can write. `timeout` only ever arrives through `--label`."""

DEFAULT_WINDOW = 0.2
"""The fraction of the force record the coefficients are averaged over when the lock
does not say. `results.py` quotes the last quarter; a design study wants the flatter
tail of a run that was sized to converge, so a fifth."""

COEFFICIENTS = ("Cd", "Cl", "CmPitch")
"""The whole-body coefficients reported under their forceCoeffs names. `Cm` is what
the same moment was called before v2006 and is reported as `CmPitch` when that is the
column the file has."""

BOUND_FRACTION = 0.5
"""A field whose `bounding` message appears on at least this fraction of the parsed
time steps is being held at its bound rather than solved for, and that is a run that
is failing slowly. Early bounding on k or omega is ordinary and is not this."""


# -- the pieces --------------------------------------------------------------------


def load_lock(path: Path) -> dict:
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(lock, dict):
        raise SystemExit(f"{path} is not a JSON object")
    return lock


def fidelity_of(lock: dict) -> dict:
    fidelity = lock.get("fidelity") or {}
    return dict(fidelity) if isinstance(fidelity, dict) else {}


def band(spec) -> tuple[float | None, float | None]:
    """`{"min": a, "max": b}` as a pair, either side optional and None when absent."""
    if not isinstance(spec, dict):
        return (None, None)
    low, high = spec.get("min"), spec.get("max")
    return (float(low) if low is not None else None, float(high) if high is not None else None)


def outside(value, limits: tuple[float | None, float | None]) -> bool:
    low, high = limits
    if value is None:
        return False
    return (low is not None and value < low) or (high is not None and value > high)


def cell_count(case: Path) -> int:
    """Cells, from the `note` line in `constant/polyMesh/owner`'s header.

    OpenFOAM writes `nCells:` there in every format, ascii or binary, so the count is
    read without a reader and without `checkMesh`. `look.json` from `mesh_look.py` is
    the fallback for a mesh whose owner file has no note.
    """
    owner = case / "constant" / "polyMesh" / "owner"
    if owner.is_file():
        try:
            with owner.open("rb") as handle:
                head = handle.read(4000).decode("utf-8", errors="replace")
        except OSError:
            head = ""
        match = re.search(r"nCells:\s*(\d+)", head)
        if match:
            return int(match.group(1))
    look = case / "look.json"
    if look.is_file():
        try:
            return int(json.loads(look.read_text(encoding="utf-8")).get("cells") or 0)
        except (OSError, ValueError):
            return 0
    return 0


def mesh_verdict(case: Path) -> tuple[bool, str]:
    """(mesh present and checkMesh passed, why not).

    `look.json` is what `parametric.py` leaves from its `mesh_look.py` call and carries
    `checkmesh_ok`; without it the presence of `constant/polyMesh/points` is all this can
    say, and it says so.
    """
    if not (case / "constant" / "polyMesh" / "points").exists():
        return False, "no constant/polyMesh in the case"
    look = case / "look.json"
    if look.is_file():
        try:
            payload = json.loads(look.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False, "look.json could not be read"
        if not payload.get("checkmesh_ok"):
            return False, f"checkMesh: {payload.get('checkmesh') or 'no verdict'}"
    return True, ""


def force_metrics(case: Path, fraction: float = DEFAULT_WINDOW) -> tuple[dict, int]:
    """The whole-body coefficients averaged over the tail of the record, and the number
    of rows the average was taken over. Empty and zero when the case logged no forces."""
    grouped = results.find_force_files(case)
    if not grouped:
        return {}, 0
    series = results.choose_force_series(grouped)
    history = results.read_history(grouped[series])
    if not history["rows"]:
        return {}, 0
    columns = list(history["columns"])
    keep = max(1, int(round(len(history["rows"]) * fraction)))
    out: dict = {}
    for name in COEFFICIENTS:
        column = name
        if name not in columns and name == "CmPitch" and "Cm" in columns:
            column = "Cm"
        values = results.column(history, column)
        if values.size:
            out[name] = results.tail_mean(values, fraction)
    return out, keep


def solver_log(case: Path, solver: str) -> Path | None:
    """`log.<solver>` when it is there -- that is the file `parametric.py` writes -- else
    whatever `results.find_solver_log` picks out of the case."""
    named = case / f"log.{solver}"
    if named.is_file():
        return named
    return results.find_solver_log(case)


def read_log(log: Path | None) -> dict:
    """What the solver log says about the run: shape, convergence, divergence."""
    out = {"shape": "", "converged": False, "diverged": "", "steps": 0}
    if log is None or not log.is_file():
        return out
    data = log_digest.digest(log)
    shape = log_digest.residual_shape(data["residuals"])["shape"]
    out["shape"] = shape
    out["steps"] = len(data["times"])
    out["converged"] = data["converged_at"] is not None or shape in ("levelled", "falling")
    if data["fatal"]:
        out["diverged"] = data["fatal"]
    elif shape == "diverging":
        out["diverged"] = "a residual is climbing off its own best"
    else:
        steps = max(1, out["steps"])
        held = [field for field, count in sorted(data["bounding"].items())
                if count >= BOUND_FRACTION * steps and count >= log_digest.MIN_STEPS]
        if held:
            out["diverged"] = (f"{', '.join(held)} held at its bound on "
                               f"{', '.join(str(data['bounding'][f]) for f in held)} of "
                               f"{out['steps']} steps")
    if not data["residuals"] and not data["fatal"]:
        out["shape"] = "short"
    return out


def yplus_range(case: Path, patch: str = "") -> dict | None:
    """`{"min": .., "max": ..}` off the yPlus function object's last write, or None.

    The file is `postProcessing/<name>/<time>/yPlus.dat`, one row per patch per write:
    `time patch min max average`. The body patch's row is the one that matters for a
    wall-function band; without a named patch every wall row at the last write counts.
    """
    base = case / "postProcessing"
    if not base.is_dir():
        return None
    rows: list[tuple[float, str, float, float]] = []
    for dat in sorted(base.rglob("yPlus*.dat")):
        try:
            text = dat.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 4 or parts[0].startswith("#"):
                continue
            try:
                rows.append((float(parts[0]), parts[1], float(parts[2]), float(parts[3])))
            except ValueError:
                continue
    if patch:
        rows = [row for row in rows if row[1] == patch]
    if not rows:
        return None
    last = max(row[0] for row in rows)
    final = [row for row in rows if row[0] == last]
    return {"min": min(row[2] for row in final), "max": max(row[3] for row in final)}


def build_sha256(case: Path) -> str:
    """The hash of the `build.py` that produced this mesh, so a leaderboard entry names
    its parameterization. Empty when the case has no build script."""
    script = case / "build.py"
    if not script.is_file():
        return ""
    return hashlib.sha256(script.read_bytes()).hexdigest()


def versions() -> dict:
    """What produced the numbers, best effort: the environment's OpenFOAM, and the two
    Python libraries a build script is likely to have used. `unknown` where absent."""
    out = {"foam": os.environ.get("WM_PROJECT_VERSION") or "unknown"}
    for name in ("gmsh", "build123d"):
        try:
            module = __import__(name)
            out[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # noqa: BLE001 - absent, broken, either way not here
            out[name] = "unknown"
    return out


# -- the verdict ---------------------------------------------------------------------


def score(case: Path, lock: dict, run: dict | None = None, *, label: str = "",
          detail: str = "", mesh_only: bool = False) -> dict:
    """Everything `metrics.json` carries, from the case and the lock.

    `run` is what `parametric.py` knows and the case does not: the `case_gen.py`
    arguments it used, the wall time, the trim record. `label` when given is the
    caller's verdict and wins -- that is how a mesh failure keeps its reason and how a
    killed run is marked `timeout`.
    """
    case = Path(case)
    run = dict(run or {})
    fidelity = fidelity_of(lock)
    fraction = float(fidelity.get("window") or DEFAULT_WINDOW)
    solver = str(lock.get("solver") or "simpleFoam")
    body = str(lock.get("body_patch") or "")

    mesh_ok, mesh_why = mesh_verdict(case)
    cells = cell_count(case)
    metrics: dict = {}
    window = 0
    converged = False
    shape = ""
    yplus = None
    diverged = ""
    if not mesh_only and mesh_ok:
        metrics, window = force_metrics(case, fraction)
        log = read_log(solver_log(case, solver))
        converged, shape, diverged = log["converged"], log["shape"], log["diverged"]
        yplus = yplus_range(case, body)

    yplus_band = band(fidelity.get("yplus"))
    yplus_bad = yplus is not None and (outside(yplus["min"], yplus_band)
                                       or outside(yplus["max"], yplus_band))
    # The cell band applies to a mesh probe too: a mesh that lands outside it is known
    # before a solve is paid for on it, under the label the solve would have earned.
    cells_bad = bool(cells) and outside(cells, band(fidelity.get("cells")))

    decided, why = label, detail
    if not decided:
        if not mesh_ok:
            decided, why = "mesh", mesh_why
        elif not mesh_only and diverged:
            decided, why = "diverged", diverged
        elif not mesh_only and not converged:
            decided, why = "unconverged", (f"residuals {shape}" if shape else "no residuals in the log")
        elif not mesh_only and yplus_bad:
            decided, why = "yplus", f"y+ {yplus['min']:.3g}..{yplus['max']:.3g} against {fidelity.get('yplus')}"
        elif cells_bad:
            decided, why = "cells", f"{cells} cells against {fidelity.get('cells')}"
        else:
            decided, why = "ok", ""

    return {
        "label": decided,
        "detail": why,
        "fidelity": fidelity,
        "template_version": lock.get("template_version"),
        "build_sha256": build_sha256(case),
        "case_args": list(run.get("case_args") or []),
        "trim": run.get("trim"),
        "metrics": metrics,
        "window": window,
        "converged": bool(converged),
        "residual_shape": shape,
        "yplus": yplus,
        "cells": int(cells),
        "wall_seconds": float(run.get("wall_seconds") or 0.0),
        "versions": versions(),
    }


def write(metrics: dict, out: Path) -> Path:
    """The one place `metrics.json` is written. Atomic, so a reader polling for the
    file never sees half of it."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(metrics, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    return out


def read_run(case: Path, path: Path | None) -> dict:
    """`run.json`, which `parametric.py` leaves beside the candidate as it works:
    case_args, wall_seconds, trim. Absent is fine -- the fields come back empty."""
    target = Path(path) if path else case / "run.json"
    if not target.is_file():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", type=Path, help="the case directory that was built and solved")
    ap.add_argument("--lock", type=Path, required=True, help="goal.lock.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="where metrics.json goes (default <case>/metrics.json)")
    ap.add_argument("--mesh-only", action="store_true", dest="mesh_only",
                    help="a mesh probe: label mesh, cells or ok, no coefficients")
    ap.add_argument("--label", default="", choices=["", *LABELS],
                    help="the caller's verdict, which wins over what is measured")
    ap.add_argument("--detail", default="", help="why, in a sentence, beside --label")
    ap.add_argument("--run", type=Path, default=None,
                    help="parametric.py's run.json (default <case>/run.json)")
    args = ap.parse_args(argv)

    case = args.case.resolve()
    lock = load_lock(args.lock)
    run = read_run(case, args.run)
    if not run.get("wall_seconds") and run.get("started"):
        run["wall_seconds"] = max(0.0, time.time() - float(run["started"]))
    metrics = score(case, lock, run, label=args.label, detail=args.detail,
                    mesh_only=args.mesh_only)
    out = write(metrics, args.out or case / "metrics.json")
    print(f"{metrics['label']}" + (f"  ({metrics['detail']})" if metrics["detail"] else ""))
    for name, value in sorted(metrics["metrics"].items()):
        print(f"  {name} = {value:.5g}   (mean of the last {metrics['window']} rows)")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
