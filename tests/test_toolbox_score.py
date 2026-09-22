"""`score.py`: the sole writer of a candidate's `metrics.json`.

No OpenFOAM here. A case is a directory with the files a solved case leaves behind --
a `constant/polyMesh/owner` header, a `look.json` from `mesh_look.py`, a
`postProcessing/forces/0/forceCoeffs.dat`, a `postProcessing/yPlus/0/yPlus.dat`, a
`log.simpleFoam` -- written by the fixtures below in the shapes the real files have, and
the residual series are the ones `test_toolbox.py` already reads shapes off.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


@pytest.fixture(scope="module")
def score():
    spec = importlib.util.spec_from_file_location("toolbox_score", TOOLBOX / "score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- fixtures shaped like the real files ---------------------------------------


OWNER_HEADER = """\
FoamFile
{
    version     2.0;
    format      ascii;
    arch        "LSB;label=32;scalar=64";
    note        "nPoints:162  nCells:{cells}  nFaces:272  nInternalFaces:112";
    class       labelList;
    location    "constant/polyMesh";
    object      owner;
}
"""


def mesh(case: Path, cells: int = 30000, ok: bool = True) -> None:
    poly = case / "constant" / "polyMesh"
    poly.mkdir(parents=True, exist_ok=True)
    (poly / "points").write_text("0()\n")
    (poly / "owner").write_text(OWNER_HEADER.replace("{cells}", str(cells)))
    (case / "look.json").write_text(json.dumps({
        "polymesh": True, "cells": cells, "checkmesh_ok": ok,
        "checkmesh": "Mesh OK." if ok else "failed 2 mesh checks",
    }))


def forces(case: Path, rows: int = 100, tail_cl: float = 0.8, tail_cd: float = 0.03,
           wobble: float = 0.0, header: str = "Cd Cl CmPitch") -> None:
    """A `forceCoeffs.dat`: the first 80% of the rows still settling, the last 20% at
    the tail value, optionally oscillating about it so the last row is not the mean."""
    lines = ["# Force coefficients", "# magUInf           : 1.0000e+01",
             "# lRef              : 1.0000e+00", "# Aref              : 5.0000e-02",
             "# Time              " + "              ".join(header.split())]
    tail = int(round(rows * 0.8))
    for i in range(1, rows + 1):
        if i <= tail:
            cl, cd = tail_cl + 0.5 * (1 - i / tail), tail_cd + 0.1 * (1 - i / tail)
        else:
            k = i - tail
            cl = tail_cl + wobble * (1 if k % 2 else -1)
            cd = tail_cd + wobble * (1 if k % 2 else -1) * 0.1
        lines.append(f"{i}  {cd:.6e}  {cl:.6e}  {-0.05:.6e}")
    target = case / "postProcessing" / "forces" / "0"
    target.mkdir(parents=True, exist_ok=True)
    (target / "forceCoeffs.dat").write_text("\n".join(lines) + "\n")


def yplus(case: Path, body=(0.8, 120.0), ground=(2.0, 50.0), time: int = 100) -> None:
    lines = ["# y+", "# Time patch min max average"]
    for t in (time // 2, time):
        lines.append(f"{t} body {body[0]} {body[1]} 30")
        lines.append(f"{t} ground {ground[0]} {ground[1]} 20")
    target = case / "postProcessing" / "yPlus" / "0"
    target.mkdir(parents=True, exist_ok=True)
    (target / "yPlus.dat").write_text("\n".join(lines) + "\n")


def levelled(steps: int = 400, level: float = 0.015) -> list[float]:
    """Study 4379's shape: down to ~1.5e-2 by step 80 and oscillating there."""
    out = []
    for s in range(1, steps + 1):
        out.append(max(level * (1 + 0.1 * math.sin(s / 7.0)), 1.0 * 0.95 ** s))
    return out


def falling(steps: int = 400) -> list[float]:
    return [10 ** (-s / 100) for s in range(1, steps + 1)]


def climbing() -> list[float]:
    return [1e-3 * 0.9 ** s for s in range(1, 40)] + [1e-4 * 3 ** (s - 40) for s in range(40, 52)]


def log(case: Path, series: list[float], *, name: str = "log.simpleFoam",
        fatal: str = "", bounding_every: int = 0, end: bool = True,
        bounding_line: str = "bounding omega, min: -1.2e+02 max: 3.4e+05 average: 4.5\n") -> Path:
    """`bounding_line` defaults to a field genuinely held at its bound -- the clip
    27x the field's mean, as `locate.py`'s cascade fixture has it."""
    body = []
    for step, value in enumerate(series, start=1):
        body.append(f"Time = {step}\n\n")
        body.append(f"smoothSolver:  Solving for Ux, Initial residual = {value:.6e}, "
                    f"Final residual = {value / 10:.6e}, No Iterations 3\n")
        body.append(f"GAMG:  Solving for p, Initial residual = {2 * value:.6e}, "
                    f"Final residual = {value / 5:.6e}, No Iterations 5\n")
        if bounding_every and step % bounding_every == 0:
            body.append(bounding_line)
        body.append(f"ExecutionTime = {step * 0.5:.1f} s  ClockTime = {step} s\n\n")
    if fatal:
        body.append(f"--> FOAM FATAL ERROR: {fatal}\n")
    elif end:
        body.append("End\n")
    path = case / name
    path.write_text("".join(body), encoding="utf-8")
    return path


LOCK = {
    "template_version": "t-1",
    "solver": "simpleFoam",
    "case_gen_args": ["--speed", "10", "--reynolds", "1e6"],
    "body_patch": "body",
    "fidelity": {"ranks": 1, "iters": 400, "window": 0.2,
                 "cells": {"min": 10000, "max": 60000}, "yplus": {"max": 300},
                 "reference": True},
}


@pytest.fixture
def good_case(tmp_path):
    case = tmp_path / "cand"
    mesh(case)
    forces(case, wobble=0.05)
    yplus(case)
    log(case, levelled())
    (case / "build.py").write_text("from design_constants import *\nprint(1)\n")
    return case


# -- the schema ----------------------------------------------------------------


SCHEMA = {"label", "detail", "fidelity", "build_sha256", "case_args", "trim", "metrics",
          "window", "converged", "residual_shape", "yplus", "cells", "wall_seconds",
          "versions"}


def test_every_key_of_the_contract_is_present_and_typed(score, good_case):
    out = score.score(good_case, LOCK, {"case_args": ["--speed", "10"], "wall_seconds": 12.5})
    assert SCHEMA <= set(out)
    assert out["label"] == "ok" and out["detail"] == ""
    assert out["fidelity"] == LOCK["fidelity"]
    assert out["template_version"] == "t-1"
    assert len(out["build_sha256"]) == 64
    assert out["case_args"] == ["--speed", "10"]
    assert out["trim"] is None
    assert set(out["metrics"]) == {"Cd", "Cl", "CmPitch"}
    assert out["converged"] is True and out["residual_shape"] == "levelled"
    assert out["yplus"] == {"min": 0.8, "max": 120.0}
    assert out["cells"] == 30000
    assert out["wall_seconds"] == 12.5
    assert set(out["versions"]) == {"foam", "gmsh", "build123d"}
    json.dumps(out)


def test_labels_are_the_closed_set(score):
    assert set(score.LABELS) == {"ok", "mesh", "diverged", "timeout", "unconverged",
                                 "yplus", "cells"}


# -- the windowed mean ---------------------------------------------------------


def test_coefficients_are_the_mean_of_the_last_window_never_the_last_row(score, good_case):
    """The tail oscillates +-0.05 about 0.8; the last row is 0.75 or 0.85 and the
    window's mean is 0.8. A point read would quote whichever phase the run ended on."""
    out = score.score(good_case, LOCK)
    assert out["window"] == 20, "20% of 100 rows"
    assert out["metrics"]["Cl"] == pytest.approx(0.8, abs=1e-9)
    assert out["metrics"]["Cd"] == pytest.approx(0.03, abs=1e-9)
    assert out["metrics"]["CmPitch"] == pytest.approx(-0.05)


def test_the_window_fraction_comes_from_the_lock(score, good_case):
    lock = dict(LOCK, fidelity=dict(LOCK["fidelity"], window=0.5))
    out = score.score(good_case, lock)
    assert out["window"] == 50
    assert out["metrics"]["Cl"] > 0.8, "half the record reaches back into the settling part"


def test_the_default_window_is_a_fifth(score, good_case):
    lock = dict(LOCK, fidelity={k: v for k, v in LOCK["fidelity"].items() if k != "window"})
    assert score.score(good_case, lock)["window"] == 20


def test_the_old_moment_column_name_is_reported_as_cm_pitch(score, tmp_path):
    case = tmp_path / "old"
    mesh(case)
    forces(case, header="Cd Cl Cm")
    log(case, levelled())
    out = score.score(case, LOCK)
    assert "CmPitch" in out["metrics"] and "Cm" not in out["metrics"]


# -- the label, in order of precedence ------------------------------------------


def test_no_polymesh_is_mesh_whatever_else_is_there(score, tmp_path):
    case = tmp_path / "nomesh"
    case.mkdir()
    forces(case)
    log(case, climbing(), fatal="Floating point exception")
    out = score.score(case, LOCK)
    assert out["label"] == "mesh"
    assert "polyMesh" in out["detail"]
    assert out["metrics"] == {} and out["cells"] == 0


def test_a_failed_checkmesh_is_mesh_with_the_verdict(score, tmp_path):
    case = tmp_path / "bad"
    mesh(case, ok=False)
    forces(case)
    log(case, levelled())
    out = score.score(case, LOCK)
    assert out["label"] == "mesh"
    assert "failed 2 mesh checks" in out["detail"]


def test_a_fatal_error_is_diverged_before_anything_about_convergence(score, tmp_path):
    case = tmp_path / "fpe"
    mesh(case, cells=999)  # outside the band too
    forces(case)
    yplus(case, body=(0.8, 9000.0))  # outside the band too
    log(case, levelled(), fatal="Floating point exception")
    out = score.score(case, LOCK)
    assert out["label"] == "diverged"
    assert "Floating point exception" in out["detail"]


def test_a_climbing_residual_is_diverged(score, tmp_path):
    case = tmp_path / "climb"
    mesh(case)
    forces(case)
    log(case, climbing(), end=False)
    out = score.score(case, LOCK)
    assert out["label"] == "diverged"
    assert out["residual_shape"] == "diverging" and out["converged"] is False


def test_a_field_held_at_its_bound_for_most_of_the_run_is_diverged(score, tmp_path):
    case = tmp_path / "bound"
    mesh(case)
    forces(case)
    log(case, levelled(), bounding_every=1)
    out = score.score(case, LOCK)
    assert out["label"] == "diverged"
    assert "omega held at its bound" in out["detail"]
    assert "27 x the mean" in out["detail"]


def test_a_shallow_clip_on_most_steps_of_a_converged_run_is_not_diverged(score, tmp_path):
    """The ground-effect section at its baseline, 2026-09-22: residuals levelled at
    1e-5, forces steady, and `bounding k` on 1355 of 1500 steps with the minimum half
    a percent of the mean below zero. k-omega SST does that on a healthy run, and the
    count-only rule scored the campaign's first candidate `diverged` on it."""
    case = tmp_path / "shallow"
    mesh(case)
    forces(case)
    log(case, levelled(), bounding_every=1,
        bounding_line="bounding k, min: -0.00092155 max: 1.97875 average: 0.0943649\n")
    out = score.score(case, LOCK)
    assert out["label"] == "ok", out["detail"]
    assert out["converged"] is True


def test_the_depth_gate_reads_the_typical_step_not_the_first(score, tmp_path):
    """A first-iteration `bounding omega` at a hundred times the mean and a shallow
    clip after it is the healthy kOmegaSST start; the median is what decides."""
    case = tmp_path / "median"
    mesh(case)
    forces(case)
    path = log(case, levelled(), bounding_every=1,
               bounding_line="bounding omega, min: -0.01 max: 3e3 average: 30\n")
    text = path.read_text(encoding="utf-8")
    text = text.replace("bounding omega, min: -0.01 max: 3e3 average: 30\n",
                        "bounding omega, min: -3000 max: 3e3 average: 30\n", 3)
    path.write_text(text, encoding="utf-8")
    assert score.score(case, LOCK)["label"] == "ok"


def test_early_bounding_on_a_run_that_then_settles_is_not_diverged(score, tmp_path):
    """The first iterations of a kOmegaSST case print `bounding omega` and then stop;
    that is ordinary, and a label that fired on it would fail every healthy run."""
    case = tmp_path / "early"
    mesh(case)
    forces(case)
    log(case, levelled(), bounding_every=50)  # 8 of 400 steps
    assert score.score(case, LOCK)["label"] == "ok"


def test_a_short_or_missing_residual_record_is_unconverged(score, tmp_path):
    case = tmp_path / "short"
    mesh(case)
    forces(case)
    yplus(case, body=(0.8, 9000.0))
    log(case, levelled()[:3])
    out = score.score(case, LOCK)
    assert out["label"] == "unconverged" and out["residual_shape"] == "short"

    nolog = tmp_path / "nolog"
    mesh(nolog)
    forces(nolog)
    out = score.score(nolog, LOCK)
    assert out["label"] == "unconverged"
    assert "no residuals" in out["detail"]


def test_falling_counts_as_converged_and_levelled_does_too(score, tmp_path):
    for name, series in (("fall", falling()), ("level", levelled())):
        case = tmp_path / name
        mesh(case)
        forces(case)
        log(case, series)
        out = score.score(case, LOCK)
        assert out["converged"] is True, name
        assert out["label"] == "ok", name


def test_the_solvers_own_convergence_line_counts(score, tmp_path):
    case = tmp_path / "said"
    mesh(case)
    forces(case)
    path = log(case, levelled()[:5], end=False)
    path.write_text(path.read_text() + "\nSIMPLE solution converged in 5 iterations\n\nEnd\n")
    assert score.score(case, LOCK)["converged"] is True


def test_yplus_outside_the_band_is_yplus_and_comes_before_cells(score, tmp_path):
    case = tmp_path / "yp"
    mesh(case, cells=999)
    forces(case)
    yplus(case, body=(0.8, 900.0))
    log(case, levelled())
    out = score.score(case, LOCK)
    assert out["label"] == "yplus"
    assert out["yplus"] == {"min": 0.8, "max": 900.0}
    assert "900" in out["detail"]


def test_the_yplus_band_can_have_a_lower_side(score, good_case):
    lock = dict(LOCK, fidelity=dict(LOCK["fidelity"], yplus={"min": 1.0, "max": 300}))
    out = score.score(good_case, lock)
    assert out["label"] == "yplus", "the body's minimum of 0.8 is under 1.0"


def test_yplus_is_read_off_the_body_patch_at_the_last_write(score, tmp_path):
    """The ground's row is 2..50 and the body's 0.8..120; the band is on the body."""
    case = tmp_path / "patch"
    mesh(case)
    forces(case)
    yplus(case, body=(0.8, 120.0), ground=(2.0, 5000.0))
    log(case, levelled())
    out = score.score(case, LOCK)
    assert out["yplus"] == {"min": 0.8, "max": 120.0}
    assert out["label"] == "ok"

    every = dict(LOCK)
    del every["body_patch"]
    out = score.score(case, every)
    assert out["yplus"] == {"min": 0.8, "max": 5000.0}, "no body named: every patch counts"


def test_no_yplus_output_is_null_and_no_verdict(score, tmp_path):
    case = tmp_path / "noyp"
    mesh(case)
    forces(case)
    log(case, levelled())
    out = score.score(case, LOCK)
    assert out["yplus"] is None and out["label"] == "ok"


def test_cells_outside_the_band_is_cells(score, tmp_path):
    for cells in (999, 999999):
        case = tmp_path / f"c{cells}"
        mesh(case, cells=cells)
        forces(case)
        yplus(case)
        log(case, levelled())
        out = score.score(case, LOCK)
        assert out["label"] == "cells", cells
        assert out["cells"] == cells and str(cells) in out["detail"]


def test_a_band_with_one_side_missing_binds_on_that_side_only(score, good_case):
    lock = dict(LOCK, fidelity=dict(LOCK["fidelity"], cells={"max": 1000}))
    assert score.score(good_case, lock)["label"] == "cells"
    lock = dict(LOCK, fidelity=dict(LOCK["fidelity"], cells={"min": 1000}))
    assert score.score(good_case, lock)["label"] == "ok"
    lock = dict(LOCK, fidelity={k: v for k, v in LOCK["fidelity"].items() if k != "cells"})
    assert score.score(good_case, lock)["label"] == "ok"


def test_the_callers_label_wins_over_what_is_measured(score, good_case):
    out = score.score(good_case, LOCK, label="timeout", detail="killed at 3x the median")
    assert out["label"] == "timeout" and out["detail"] == "killed at 3x the median"
    assert out["metrics"]["Cl"] == pytest.approx(0.8), "what was measured still travels"


# -- mesh only -------------------------------------------------------------------


def test_mesh_only_is_ok_or_mesh_with_no_coefficients(score, tmp_path):
    case = tmp_path / "probe"
    mesh(case)
    forces(case)  # left over from something; a probe does not read it
    log(case, climbing(), fatal="Floating point exception")
    out = score.score(case, LOCK, mesh_only=True)
    assert out["label"] == "ok"
    assert out["metrics"] == {} and out["window"] == 0
    assert out["converged"] is False and out["residual_shape"] == ""
    assert out["yplus"] is None
    assert out["cells"] == 30000

    mesh(case, ok=False)
    assert score.score(case, LOCK, mesh_only=True)["label"] == "mesh"


def test_a_mesh_probe_outside_the_cell_band_says_cells(score, tmp_path):
    case = tmp_path / "big"
    mesh(case, cells=10 ** 6)
    out = score.score(case, LOCK, mesh_only=True)
    assert out["label"] == "cells"


# -- the pieces on their own ---------------------------------------------------


def test_cell_count_comes_off_the_owner_header_then_look_json(score, tmp_path):
    case = tmp_path / "c"
    mesh(case, cells=4242)
    assert score.cell_count(case) == 4242
    (case / "constant" / "polyMesh" / "owner").write_text("FoamFile { }\n")
    assert score.cell_count(case) == 4242, "look.json stands in"
    (case / "look.json").unlink()
    assert score.cell_count(case) == 0


def test_the_named_solver_log_is_preferred(score, tmp_path):
    case = tmp_path / "logs"
    case.mkdir()
    log(case, climbing(), name="log.potentialFoam", end=False)
    log(case, levelled(), name="log.simpleFoam")
    assert score.solver_log(case, "simpleFoam").name == "log.simpleFoam"
    assert score.read_log(score.solver_log(case, "simpleFoam"))["shape"] == "levelled"


def test_build_sha256_is_of_the_script_and_empty_without_one(score, tmp_path):
    case = tmp_path / "sha"
    case.mkdir()
    assert score.build_sha256(case) == ""
    (case / "build.py").write_bytes(b"from design_constants import *\n")
    import hashlib
    assert score.build_sha256(case) == hashlib.sha256(b"from design_constants import *\n").hexdigest()


def test_versions_are_strings_and_unknown_where_absent(score, monkeypatch):
    monkeypatch.delenv("WM_PROJECT_VERSION", raising=False)
    out = score.versions()
    assert out["foam"] == "unknown"
    assert all(isinstance(v, str) for v in out.values())
    monkeypatch.setenv("WM_PROJECT_VERSION", "v2512")
    assert score.versions()["foam"] == "v2512"


# -- the command line ------------------------------------------------------------


def test_the_cli_writes_the_file_and_reads_run_json(score, good_case, tmp_path, capsys):
    lock = tmp_path / "goal.lock.json"
    lock.write_text(json.dumps(LOCK))
    (good_case / "run.json").write_text(json.dumps(
        {"case_args": ["--speed", "10", "--body", "body"], "wall_seconds": 33.0,
         "trim": {"variable": "ALPHA_DEG", "target": 0.8, "achieved": 0.801,
                  "value": 5.9, "solves": 3}}))
    out = tmp_path / "metrics.json"

    assert score.main([str(good_case), "--lock", str(lock), "--out", str(out)]) == 0

    written = json.loads(out.read_text())
    assert written["label"] == "ok"
    assert written["case_args"] == ["--speed", "10", "--body", "body"]
    assert written["wall_seconds"] == 33.0
    assert written["trim"]["solves"] == 3
    assert not list(tmp_path.glob("*.tmp")), "written atomically, nothing half-left"
    text = capsys.readouterr().out
    assert "ok" in text and "Cl = 0.8" in text


def test_the_cli_takes_the_callers_label(score, good_case, tmp_path):
    lock = tmp_path / "goal.lock.json"
    lock.write_text(json.dumps(LOCK))
    score.main([str(good_case), "--lock", str(lock), "--label", "timeout",
                "--detail", "3x median"])
    written = json.loads((good_case / "metrics.json").read_text())
    assert written["label"] == "timeout" and written["detail"] == "3x median"


def test_wall_seconds_falls_back_to_the_start_time_of_a_killed_run(score, good_case, tmp_path):
    """A run killed mid-solve never wrote its final wall time; `started` is there."""
    import time

    lock = tmp_path / "goal.lock.json"
    lock.write_text(json.dumps(LOCK))
    (good_case / "run.json").write_text(json.dumps({"started": time.time() - 100}))
    score.main([str(good_case), "--lock", str(lock), "--label", "timeout"])
    written = json.loads((good_case / "metrics.json").read_text())
    assert 99 < written["wall_seconds"] < 200
