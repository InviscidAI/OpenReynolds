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
import random
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
    assert set(score.SEVERITY) == set(score.LABELS), "one order over the same labels"


# -- the pressure drop ---------------------------------------------------------


def pressures(case: Path, inlet: list[float], outlet: list[float], time_dir: str = "0",
              function=("pInlet", "pOutlet")) -> None:
    """The two `surfaceFieldValue` records `case_gen.py --dp` leaves, one row per
    iteration, in the shape OpenFOAM writes them."""
    for name, series in zip(function, (inlet, outlet)):
        target = case / "postProcessing" / name / time_dir
        target.mkdir(parents=True, exist_ok=True)
        lines = [f"# Region type : patch {name}", "# Faces  : 40",
                 "# Area    : 1.6e-05", "# Time                areaAverage(p)"]
        for step, value in enumerate(series, start=1):
            lines.append(f"{step}\t{value:.6e}")
        (target / "surfaceFieldValue.dat").write_text("\n".join(lines) + "\n")


def test_dp_is_the_windowed_mean_of_the_difference(score, good_case):
    """Ten rows, the last two the window: inlet 5.0/5.4 against outlet 1.0/1.4, so the
    drop is 4.0 in both of them and 4.0 is the mean. The last row alone would say 4.0
    too here; the point of the window is the settling part before it, which is 9.0."""
    pressures(good_case, inlet=[13.0] * 8 + [5.0, 5.4], outlet=[4.0] * 8 + [1.0, 1.4])
    out = score.score(good_case, LOCK)
    assert out["metrics"]["dp"] == pytest.approx(4.0, abs=1e-9)


def test_dp_is_formed_per_time_step_and_not_as_two_tail_means(score, good_case):
    """The two objects wrote different iterations: the inlet has one row the outlet does
    not. Aligned by time the drop is 2.0 on the shared rows; two separate tail means
    would mix an inlet row with an outlet row from somewhere else."""
    pressures(good_case, inlet=[3.0] * 10, outlet=[1.0] * 10)
    extra = good_case / "postProcessing" / "pInlet" / "0" / "surfaceFieldValue.dat"
    extra.write_text(extra.read_text() + "11\t9.900000e+01\n")
    assert score.score(good_case, LOCK)["metrics"]["dp"] == pytest.approx(2.0)


def test_a_restarted_record_is_merged_and_the_later_write_wins(score, good_case):
    pressures(good_case, inlet=[3.0] * 10, outlet=[1.0] * 10)
    pressures(good_case, inlet=[7.0] * 10, outlet=[1.0] * 10, time_dir="10")
    assert score.score(good_case, LOCK)["metrics"]["dp"] == pytest.approx(6.0)


def test_a_case_run_without_the_dp_objects_has_no_dp(score, good_case):
    out = score.score(good_case, LOCK)
    assert "dp" not in out["metrics"], "no record is not a drop of zero"
    assert score.pressure_drop(good_case) == (None, 0)


def test_one_side_of_the_drop_alone_is_not_a_drop(score, good_case):
    pressures(good_case, inlet=[3.0] * 10, outlet=[1.0] * 10)
    import shutil

    shutil.rmtree(good_case / "postProcessing" / "pOutlet")
    assert "dp" not in score.score(good_case, LOCK)["metrics"]


def test_the_function_object_names_are_the_ones_case_gen_writes(score):
    """A rename on either side would leave the other reporting no pressure drop, quietly."""
    spec = importlib.util.spec_from_file_location("toolbox_case_gen", TOOLBOX / "case_gen.py")
    case_gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case_gen)
    assert score.DP_FUNCTIONS == tuple(name for _, name in case_gen.PRESSURE_FUNCTIONS)


# -- derived metrics -----------------------------------------------------------


def test_a_derived_metric_is_arithmetic_over_metric_names(score):
    values = {"forward.dp": 2.0, "reverse.dp": 5.0}
    assert score.evaluate_derived("reverse.dp / forward.dp", values) == pytest.approx(2.5)
    assert score.evaluate_derived("(reverse.dp - forward.dp) * 2 + 1", values) == pytest.approx(7.0)
    assert score.evaluate_derived("-forward.dp", values) == pytest.approx(-2.0)


@pytest.mark.parametrize("expression", [
    "__import__('os').system('rm -rf /')",
    "open('/etc/passwd').read()",
    "forward.dp ** 99999",
    "forward.dp if forward.dp else 1",
    "[forward.dp][0]",
    "max(forward.dp, 1)",
    "forward.dp // 2",
    "'a' + 'b'",
])
def test_anything_that_is_not_arithmetic_is_refused(score, expression):
    with pytest.raises(score.DerivedError):
        score.derived_names(expression)


def test_a_derived_metric_whose_inputs_are_missing_is_left_out(score):
    assert score.evaluate_derived("reverse.dp / forward.dp", {"forward.dp": 1.0}) is None
    assert score.derived_metrics({"forward.dp": 1.0}, {"diodicity": "reverse.dp / forward.dp"}) == {}


def test_a_derived_metric_that_divides_by_zero_is_left_out(score):
    assert score.derived_metrics({"forward.dp": 0.0, "reverse.dp": 1.0},
                                 {"diodicity": "reverse.dp / forward.dp"}) == {}


def test_a_derived_metric_may_read_one_written_before_it(score):
    out = score.derived_metrics({"forward.dp": 2.0, "reverse.dp": 6.0},
                                {"diodicity": "reverse.dp / forward.dp",
                                 "excess": "diodicity - 1"})
    assert out == {"diodicity": pytest.approx(3.0), "excess": pytest.approx(2.0)}


# -- several cases per candidate -----------------------------------------------


CASES_LOCK = dict(
    LOCK,
    body_patch="",
    cases=[{"name": "forward", "case_gen_args": ["--inlet", "west", "--outlet", "east"]},
           {"name": "reverse", "case_gen_args": ["--inlet", "east", "--outlet", "west"]}],
    derived={"diodicity": "reverse.dp / forward.dp"},
)


def solved_case(case: Path, dp: float, *, series=None, ok: bool = True) -> None:
    mesh(case, cells=30000)
    pressures(case, inlet=[dp] * 10, outlet=[0.0] * 10)
    log(case, levelled() if ok else climbing())
    (case / "build.py").write_text("from design_constants import *\nprint(1)\n")


@pytest.fixture
def valve(tmp_path):
    """One candidate, one mesh, two solved cases under it."""
    candidate = tmp_path / "cand"
    mesh(candidate, cells=30000)
    (candidate / "build.py").write_text("from design_constants import *\nprint(1)\n")
    solved_case(candidate / "forward", dp=2.0)
    solved_case(candidate / "reverse", dp=5.0)
    return candidate


def test_the_cases_metrics_are_namespaced_and_the_derived_one_is_computed(score, valve):
    out = score.score_cases(valve, CASES_LOCK, {"wall_seconds": 61.0})
    assert out["label"] == "ok", out["detail"]
    assert out["metrics"]["forward.dp"] == pytest.approx(2.0)
    assert out["metrics"]["reverse.dp"] == pytest.approx(5.0)
    assert out["metrics"]["diodicity"] == pytest.approx(2.5)
    assert set(out["cases"]) == {"forward", "reverse"}
    assert out["cases"]["forward"]["label"] == "ok"
    assert out["cells"] == 30000, "the mesh is the candidate's, measured once"
    assert out["wall_seconds"] == 61.0
    assert out["converged"] is True
    json.dumps(out)


def test_the_candidates_label_is_the_worst_of_its_cases(score, valve):
    log(valve / "reverse", climbing())
    out = score.score_cases(valve, CASES_LOCK)
    assert out["label"] == "diverged"
    assert "reverse:" in out["detail"]
    assert out["cases"]["forward"]["label"] == "ok"
    assert out["metrics"]["forward.dp"] == pytest.approx(2.0), "the good case is still reported"


def test_a_case_that_did_not_run_is_a_mesh_failure_of_the_candidate(score, valve):
    import shutil

    shutil.rmtree(valve / "reverse")
    out = score.score_cases(valve, CASES_LOCK)
    assert out["label"] == "mesh"
    assert "reverse" in out["detail"]
    assert "diodicity" not in out["metrics"]


def test_a_mesh_outside_the_band_is_the_candidates_label_even_when_both_cases_solved(score, valve):
    lock = dict(CASES_LOCK, fidelity=dict(CASES_LOCK["fidelity"], cells={"min": 1, "max": 100}))
    out = score.score_cases(valve, lock)
    assert out["label"] == "cells"
    assert out["metrics"]["diodicity"] == pytest.approx(2.5), "measured anyway, and reported"


def test_the_callers_label_still_wins_over_the_cases(score, valve):
    out = score.score_cases(valve, CASES_LOCK, label="timeout", detail="killed at 3x median")
    assert out["label"] == "timeout" and out["detail"] == "killed at 3x median"


def test_mesh_only_scores_the_candidates_mesh_and_no_case(score, valve):
    out = score.score_cases(valve, CASES_LOCK, mesh_only=True)
    assert out["label"] == "ok" and out["metrics"] == {} and "cases" not in out
    assert out["cells"] == 30000


def test_a_lock_without_cases_goes_the_one_case_way(score, good_case):
    out = score.score_cases(good_case, LOCK)
    assert "cases" not in out and set(out["metrics"]) == {"Cd", "Cl", "CmPitch"}


@pytest.mark.parametrize("cases,message", [
    ([{"name": "forward flow", "case_gen_args": []}], "identifier"),
    ([{"name": "forward", "case_gen_args": []}, {"name": "forward", "case_gen_args": []}], "twice"),
    ([{"name": "forward", "case_gen_args": [1, 2]}], "list of strings"),
    (["forward"], "name and case_gen_args"),
])
def test_a_malformed_cases_block_is_refused(score, cases, message):
    with pytest.raises(SystemExit, match=message):
        score.cases_of(dict(LOCK, cases=cases))


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


# -- a transient case: averaged in time over whole shedding cycles ---------------
#
# The drag benchmark (Archimedes `goals/max_drag.md`): a bluff body at Re 1e5 sheds, so
# the solve is pimpleFoam and the score is Cd averaged over whole cycles after the
# start-up transient. H = 0.1 m at 15 m/s is a convective time of 1/150 s; the windows
# below are in those units, as the lock states them.

H_M, U_MS = 0.1, 15.0
CONVECTIVE_S = H_M / U_MS

TRANSIENT_LOCK = dict(
    LOCK,
    solver="pimpleFoam",
    case_gen_args=["--speed", "15", "--nu", "1.5e-5", "--length", "0.1"],
    fidelity={"ranks": 1, "iters": 400, "window": 0.2,
              "solver_mode": "transient", "end_time": 60, "write_interval": 10,
              "average_from": 20, "cells": {"min": 10000, "max": 60000},
              "yplus": {"max": 300}, "reference": True},
)


def shedding(case: Path, *, st: float = 0.2, end: float = 60.0, cl_amp: float = 0.8,
             cd_mean: float = 1.2, cd_amp: float = 0.05, cd=None, cl=None) -> None:
    """A `forceCoeffs.dat` with a time column in seconds: Cl a sinusoid at Strouhal `st`
    on H, Cd its mean plus a ripple at twice that frequency (the drag sees both vortices
    of a cycle). The step is uneven -- fine for the first half of the record, four times
    coarser after it -- as an adjustable-time-step pimpleFoam writes it. `cd`, when
    given, is Cd as a function of convective time instead, and `cl` Cl."""
    f = st * U_MS / H_M
    lines = ["# Force coefficients", "# magUInf           : 1.5000e+01",
             "# Time              Cd              Cl              CmPitch"]
    t, fine = 0.0, CONVECTIVE_S / 40
    while t <= end * CONVECTIVE_S + 1e-12:
        tc = t / CONVECTIVE_S
        cd_value = cd(tc) if cd else cd_mean + cd_amp * math.sin(4 * math.pi * f * t)
        cl_value = cl(tc) if cl else cl_amp * math.sin(2 * math.pi * f * t)
        lines.append(f"{t:.8e}  {cd_value:.6e}  {cl_value:.6e}  "
                     f"{0.01:.6e}")
        t += fine if tc < end / 2 else 4 * fine
    target = case / "postProcessing" / "forceCoeffs" / "0"
    target.mkdir(parents=True, exist_ok=True)
    (target / "coefficient.dat").write_text("\n".join(lines) + "\n")


@pytest.fixture
def shedding_case(tmp_path):
    case = tmp_path / "plate"
    mesh(case)
    shedding(case)
    yplus(case, body=(0.1, 1.5))
    log(case, levelled(), name="log.pimpleFoam")
    return case


def test_a_transient_case_scores_cd_over_whole_cycles_with_st_and_cd_rms(score, shedding_case):
    out = score.score(shedding_case, TRANSIENT_LOCK)
    assert out["label"] == "ok", out["detail"]
    m = out["metrics"]
    assert m["Cd"] == pytest.approx(1.2, abs=2e-3)
    assert m["Cl"] == pytest.approx(0.0, abs=2e-2)
    assert m["St"] == pytest.approx(0.2, rel=0.02)
    assert m["Cd_rms"] == pytest.approx(0.05 / math.sqrt(2), rel=0.03)
    assert out["converged"] is True
    shed = out["shedding"]
    assert shed["periods"] == pytest.approx(8.0, rel=0.03), "40 convective times at St 0.2"
    assert shed["window_s"][0] == pytest.approx(20 * CONVECTIVE_S, rel=1e-2), "the first row at or after it"
    assert shed["min_periods"] == 5
    json.dumps(out)


def test_the_transient_mean_is_in_time_not_in_rows(score, tmp_path):
    """Cd 1.0 over the first half of the window, 2.0 over the second -- but the first
    half has four times the rows, because the step was finer there. The mean over time
    is 1.5; a mean over rows would say 1.2 and be a statement about the time step."""
    case = tmp_path / "steps"
    mesh(case)
    shedding(case, cd=lambda tc: 1.0 if tc < 40 else 2.0)
    log(case, levelled(), name="log.pimpleFoam")
    assert score.score(case, TRANSIENT_LOCK)["metrics"]["Cd"] == pytest.approx(1.5, abs=0.01)


def test_a_window_shorter_than_five_periods_is_unconverged_and_says_how_many(score, tmp_path):
    case = tmp_path / "short"
    mesh(case)
    shedding(case, end=40.0)   # 20 convective times of window at St 0.2: four periods
    log(case, levelled(), name="log.pimpleFoam")
    out = score.score(case, TRANSIENT_LOCK)
    assert out["label"] == "unconverged"
    assert "4.0 shedding periods" in out["detail"] and "5" in out["detail"]
    assert out["metrics"]["Cd"] == pytest.approx(1.2, abs=5e-3), "measured and reported anyway"


def test_the_period_count_is_the_locks_to_set(score, tmp_path):
    case = tmp_path / "short"
    mesh(case)
    shedding(case, end=40.0)
    log(case, levelled(), name="log.pimpleFoam")
    lock = dict(TRANSIENT_LOCK, fidelity=dict(TRANSIENT_LOCK["fidelity"], min_periods=3))
    assert score.score(case, lock)["label"] == "ok"


def test_a_lift_that_never_oscillates_is_unconverged_not_a_steady_answer(score, tmp_path):
    """A symmetric start can hold a bluff body's wake symmetric for a long while before
    it breaks: no shedding yet, and a Cd that is not the shedding body's. Unconverged."""
    case = tmp_path / "symmetric"
    mesh(case)
    shedding(case, cl_amp=0.0)
    log(case, levelled(), name="log.pimpleFoam")
    out = score.score(case, TRANSIENT_LOCK)
    assert out["label"] == "unconverged"
    assert "0.0 shedding periods" in out["detail"]
    assert "St" not in out["metrics"]


def noise(scale: float, seed: int = 1):
    """Cl as solver noise about a mean: a seeded normal draw per row, no cycle in it."""
    rng = random.Random(seed)
    return lambda tc: rng.gauss(0.0, scale)


@pytest.mark.parametrize("cl,why", [
    (noise(1e-6), "not shedding"),
    (lambda tc, n=noise(1e-6): 1e-3 * tc / 60 + n(tc), "not shedding"),
    (noise(0.05), "St"),
], ids=["solver-noise", "noise-on-a-slow-drift", "loud-noise"])
def test_a_lift_that_is_noise_is_unconverged_with_no_strouhal(score, tmp_path, cl, why):
    """A symmetric wake's lift is never exactly zero: pimpleFoam leaves round-off and
    solver noise on it. Measured relative to its own rms, that noise crosses its mean
    thousands of times and reads as a shedding frequency far past any bluff body's. It
    is not one: under the lift-rms floor the wake is not shedding, and a frequency whose
    Strouhal number no bluff body has is noise, however loud."""
    case = tmp_path / "noisy"
    mesh(case)
    shedding(case, cl=cl)
    log(case, levelled(), name="log.pimpleFoam")
    out = score.score(case, TRANSIENT_LOCK)
    assert out["label"] == "unconverged", out["detail"]
    assert "St" not in out["metrics"]
    assert out["shedding"]["periods"] == 0.0
    assert why in out["detail"]


def test_real_shedding_under_noise_still_counts(score, tmp_path):
    case = tmp_path / "shedding-noisy"
    mesh(case)
    f, n = 0.2, noise(0.02)
    shedding(case, cl=lambda tc: 0.8 * math.sin(2 * math.pi * f * tc) + n(tc))
    log(case, levelled(), name="log.pimpleFoam")
    out = score.score(case, TRANSIENT_LOCK)
    assert out["label"] == "ok", out["detail"]
    assert out["metrics"]["St"] == pytest.approx(0.2, rel=0.02)


def test_the_lift_rms_floor_is_the_locks_to_set(score, tmp_path):
    case = tmp_path / "weak"
    mesh(case)
    shedding(case, cl_amp=0.005)
    log(case, levelled(), name="log.pimpleFoam")
    assert score.score(case, TRANSIENT_LOCK)["label"] == "unconverged"
    lock = dict(TRANSIENT_LOCK, fidelity=dict(TRANSIENT_LOCK["fidelity"], min_lift_rms=1e-3))
    assert score.score(case, lock)["label"] == "ok"


def test_a_low_strouhal_body_holds_five_periods_on_the_drag_goals_window(score, tmp_path):
    """Archimedes' max_drag lock averages from 20 to 80 convective times so that a wide
    wake shedding at St 0.1 (a deep cup, URANS under-predicting it) still spans six
    periods; St 0.11 is 6.6."""
    case = tmp_path / "deep-cup"
    mesh(case)
    shedding(case, st=0.11, end=80.0)
    log(case, levelled(), name="log.pimpleFoam")
    lock = dict(TRANSIENT_LOCK, fidelity=dict(TRANSIENT_LOCK["fidelity"], end_time=80))
    out = score.score(case, lock)
    assert out["label"] == "ok", out["detail"]
    assert out["shedding"]["periods"] == pytest.approx(6.6, rel=0.03)
    assert out["metrics"]["St"] == pytest.approx(0.11, rel=0.02)


def test_a_record_that_ends_before_the_window_opens_has_no_coefficients(score, tmp_path):
    case = tmp_path / "killed"
    mesh(case)
    shedding(case, end=15.0)
    log(case, levelled(), name="log.pimpleFoam")
    out = score.score(case, TRANSIENT_LOCK)
    assert out["label"] == "unconverged" and out["metrics"] == {}
    assert "before the averaging window" in out["detail"]


def test_a_transient_case_that_blew_up_is_still_diverged(score, shedding_case):
    log(shedding_case, levelled(), name="log.pimpleFoam", fatal="Floating point exception")
    assert score.score(shedding_case, TRANSIENT_LOCK)["label"] == "diverged"


def test_the_steady_path_carries_no_transient_keys(score, good_case):
    out = score.score(good_case, LOCK)
    assert "shedding" not in out and "St" not in out["metrics"] and "Cd_rms" not in out["metrics"]


@pytest.mark.parametrize("change,message", [
    ({"case_gen_args": ["--speed", "15"]}, "--length"),
    ({"case_gen_args": ["--length", "0.1"]}, "--speed"),
    ({"fidelity": dict(TRANSIENT_LOCK["fidelity"], average_from=60)}, "average_from"),
    ({"fidelity": dict(TRANSIENT_LOCK["fidelity"], end_time=0)}, "end_time"),
    ({"fidelity": dict(TRANSIENT_LOCK["fidelity"], write_interval=-1)}, "write_interval"),
    ({"fidelity": dict(TRANSIENT_LOCK["fidelity"], min_lift_rms=0)}, "min_lift_rms"),
    ({"solver": "simpleFoam"}, "simpleFoam"),
    ({"case_gen_args": ["--speed", "15", "--length", "0.1", "--study", "steady"]}, "--study"),
])
def test_a_transient_fidelity_that_cannot_be_measured_is_refused(score, change, message):
    with pytest.raises(SystemExit, match=message):
        score.transient_of(dict(TRANSIENT_LOCK, **change))


def test_the_transient_window_is_convective_times_in_seconds(score):
    t = score.transient_of(TRANSIENT_LOCK)
    assert t["convective_s"] == pytest.approx(CONVECTIVE_S)
    assert t["end_s"] == pytest.approx(0.4)
    assert t["average_from_s"] == pytest.approx(20 * CONVECTIVE_S)
    assert t["writes"] == 6
    assert score.transient_of(LOCK) is None, "a steady lock has no transient window"
