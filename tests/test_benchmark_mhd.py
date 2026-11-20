"""The Hartmann grader, held to the closed form it claims to grade against.

`benchmarks/mhd/grade.py` reads a profile an `mhdFoam` run wrote and compares it with
`u/u_c = (cosh Ha - cosh(Ha y/L))/(cosh Ha - 1)`. There is no OpenFOAM on the machine
that runs this suite, so the profile is written here instead -- the exact one, which has
to pass, and five wrong ones, each wrong in a different way, which have to fail. A
grader that only ever sees the right answer is a function nobody has tested.

The solver is held to the same closed form by `tests/test_journey_mhd.py` in the
`journey` job, which runs the template, solves the case in the workspace image and runs
this grader on what came out.

The five wrong profiles are chosen so that each one trips a different check, because a
grader whose checks all fail together has one check:

* the Ha = 0 parabola, graded at Ha = 20 -- `shape_rms`, and the whole reason
  `parabola_contrast` exists;
* the exact Ha = 20 profile graded at Ha = 5 -- the same shape against the wrong
  reference, which is the mistake a grader that inferred Ha would make silently;
* the exact shape at 1.2x the flow rate -- `flow_rate` and `centreline`, with
  `shape_rms` passing, since dividing by its own centre removes the scale;
* a tilted profile -- `symmetry`, which is what a flow sampled before it has developed
  looks like;
* half a channel -- `y_range`, since a half profile reflected about its own end is
  symmetric and smooth and would otherwise be graded as if it were whole.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GRADER = ROOT / "benchmarks" / "mhd" / "grade.py"

HARTMANN = 20.0
HALF_HEIGHT = 1.0
MEAN_VELOCITY = 1.0


def grade_module():
    spec = importlib.util.spec_from_file_location("mhd_grade", GRADER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def grade():
    return grade_module()


def exact_shape(eta: np.ndarray, hartmann: float = HARTMANN) -> np.ndarray:
    """The closed form, written here rather than imported from the grader.

    A test that takes its expected value from the code under test asserts only that the
    code is self-consistent -- the rule `test_journey_thermal_stress.py` states for the
    same reason.
    """
    if hartmann == 0.0:
        return 1.0 - eta * eta
    cosh_ha = math.cosh(hartmann)
    return (cosh_ha - np.cosh(hartmann * eta)) / (cosh_ha - 1.0)


def exact_centreline(hartmann: float = HARTMANN) -> float:
    cosh_ha = math.cosh(hartmann)
    return (cosh_ha - 1.0) / (cosh_ha - math.sinh(hartmann) / hartmann)


def write_profile(case: Path, eta: np.ndarray, u_over_mean: np.ndarray,
                  time: str = "2") -> Path:
    """A `raw` sampled-set file where `postProcess -func sample` would have put one.

    Four columns -- y and the three velocity components -- which is what `axis y` and
    `fields ( U )` produce. The `#` header is written too, since the grader skips it and
    a file without one is a file the writer did not produce.
    """
    out = case / "postProcessing" / "sample" / time
    out.mkdir(parents=True, exist_ok=True)
    path = out / "hartmannProfile_U.xy"
    rows = ["# y  U_x  U_y  U_z"]
    for y, u in zip(eta * HALF_HEIGHT, u_over_mean * MEAN_VELOCITY):
        rows.append(f"{y:.10g}\t{u:.10g}\t0\t0")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def perfect(points: int = 401, hartmann: float = HARTMANN) -> tuple:
    """The exact profile at the flow rate the template's inlet imposes.

    Scaled so that its mean is exactly MEAN_VELOCITY, by equation (3), which is what
    makes `centreline` and `flow_rate` meaningful rather than circular: the shape and
    the scale are set from the two formulae independently.
    """
    eta = np.linspace(-0.999, 0.999, points)
    shape = exact_shape(eta, hartmann)
    return eta, shape * exact_centreline(hartmann)


def run_grader(case: Path, hartmann: float = HARTMANN):
    return grade_module().grade_hartmann(case, hartmann, MEAN_VELOCITY, HALF_HEIGHT)


# -- the exact profile ----------------------------------------------------------


def test_the_exact_profile_passes_every_check(tmp_path, grade):
    eta, u = perfect()
    write_profile(tmp_path, eta, u)
    result = run_grader(tmp_path)

    failed = [check["name"] for check in result["checks"] if check["passed"] is False]
    assert result["passed"], f"the closed form itself failed {failed}"
    by_name = {check["name"]: check for check in result["checks"]}
    assert by_name["shape_rms"]["value"] < 1e-9, "the closed form is not exact to itself"
    assert by_name["symmetry"]["value"] < 1e-9


def test_the_centreline_check_is_absolute_and_not_a_fit(tmp_path, grade):
    """u_c/u_mean at Ha 20 is 1.05263157, and at Ha 0 it is 3/2. The 29% between them is
    the flattening, and it is the quantity a case can get wrong while its shape is
    perfect -- so the check has to be against u_mean and not against the profile's own
    scale."""
    eta, u = perfect()
    write_profile(tmp_path, eta, u)
    result = run_grader(tmp_path)
    centreline = next(c for c in result["checks"] if c["name"] == "centreline")
    assert centreline["reference"] == pytest.approx(1.05263157, rel=1e-6)
    assert centreline["value"] == pytest.approx(1.05263157, rel=1e-4)


def test_the_parabola_contrast_is_reported_rather_than_graded_at_low_hartmann(
        tmp_path, grade):
    """At Ha = 1 the closed form is 0.0128 RMS from the parabola, which is inside the
    shape threshold, so there is no contrast to measure and saying so is the honest
    verdict -- the same call `benchmarks/sph/` makes about the dam break."""
    eta, u = perfect(hartmann=1.0)
    write_profile(tmp_path, eta, u)
    result = run_grader(tmp_path, hartmann=1.0)
    contrast = next(c for c in result["checks"] if c["name"] == "parabola_contrast")
    assert contrast["passed"] is None
    assert "not graded" in contrast["note"]
    assert result["passed"], "a reported check must not fail the run"


# -- the five wrong profiles ----------------------------------------------------


def test_the_parabola_is_rejected_at_hartmann_twenty(tmp_path, grade):
    eta = np.linspace(-0.999, 0.999, 401)
    write_profile(tmp_path, eta, 1.5 * exact_shape(eta, 0.0))
    result = run_grader(tmp_path)

    by_name = {check["name"]: check for check in result["checks"]}
    assert not result["passed"]
    assert by_name["shape_rms"]["passed"] is False
    assert by_name["shape_rms"]["value"] > 0.3, "the parabola is 0.366 RMS away"
    # The flow rate is right, which is the point: a case can conserve mass perfectly
    # and still have the wrong physics in it.
    assert by_name["flow_rate"]["passed"] is True


def test_the_right_profile_graded_at_the_wrong_hartmann_is_rejected(tmp_path, grade):
    """The failure mode a grader that inferred Ha from the geometry would hide."""
    eta, u = perfect()
    write_profile(tmp_path, eta, u)
    result = run_grader(tmp_path, hartmann=5.0)
    by_name = {check["name"]: check for check in result["checks"]}
    assert not result["passed"]
    assert by_name["shape_rms"]["passed"] is False
    assert by_name["centreline"]["passed"] is False


def test_a_shape_that_is_right_at_the_wrong_flow_rate_is_rejected(tmp_path, grade):
    eta, u = perfect()
    write_profile(tmp_path, eta, 1.2 * u)
    result = run_grader(tmp_path)
    by_name = {check["name"]: check for check in result["checks"]}
    assert not result["passed"]
    # Dividing by its own centre removes the scale, so the shape is still exact.
    assert by_name["shape_rms"]["value"] < 1e-9
    assert by_name["flow_rate"]["passed"] is False
    assert by_name["centreline"]["passed"] is False


def test_an_asymmetric_profile_is_rejected(tmp_path, grade):
    """What a profile sampled before the flow has developed looks like: the shape error
    can be small while the two halves disagree."""
    eta, u = perfect()
    write_profile(tmp_path, eta, u * (1.0 + 0.08 * eta))
    result = run_grader(tmp_path)
    by_name = {check["name"]: check for check in result["checks"]}
    assert not result["passed"]
    assert by_name["symmetry"]["passed"] is False


def test_half_a_channel_is_rejected_rather_than_graded(tmp_path, grade):
    eta = np.linspace(0.0, 0.999, 201)
    write_profile(tmp_path, eta, exact_centreline() * exact_shape(eta))
    result = run_grader(tmp_path)
    by_name = {check["name"]: check for check in result["checks"]}
    assert not result["passed"]
    assert by_name["y_range"]["passed"] is False


# -- the refusals ---------------------------------------------------------------


def test_a_case_with_no_sampled_profile_is_a_refusal_and_not_a_failure(tmp_path, grade):
    """Exit 2 and the command that would have written one, rather than FAILED: a case
    that was never sampled has not been measured, which is a different statement from
    having been measured and disagreed."""
    proc = subprocess.run(
        [sys.executable, str(GRADER), "hartmann", str(tmp_path),
         "--hartmann", "20", "--mean-velocity", "1", "--half-height", "1"],
        text=True, capture_output=True)
    assert proc.returncode == 2
    assert "postProcess -func sample" in proc.stderr


def test_the_three_constants_are_required_rather_than_defaulted(tmp_path, grade):
    """Ha needs B0, sigma, rho and nu together; the mean velocity is the inlet
    condition; and L cannot be read off a line that stops short of the walls. A default
    for any of the three would grade a run against a number nobody supplied."""
    for missing in ("--hartmann", "--mean-velocity", "--half-height"):
        argv = ["hartmann", str(tmp_path), "--hartmann", "20",
                "--mean-velocity", "1", "--half-height", "1"]
        index = argv.index(missing)
        del argv[index:index + 2]
        proc = subprocess.run([sys.executable, str(GRADER), *argv],
                              text=True, capture_output=True)
        assert proc.returncode == 2, f"{missing} was defaulted rather than required"
        assert missing in proc.stderr


def test_a_profile_of_the_wrong_width_is_a_refusal(tmp_path, grade):
    """Three columns is `fields ( Ux )`, which is a different file. Guessing at a column
    is how a grader comes to compare the wrong field."""
    out = tmp_path / "postProcessing" / "sample" / "2"
    out.mkdir(parents=True)
    (out / "hartmannProfile_Ux.xy").write_text("# y Ux\n-0.5\t1.0\n0.5\t1.0\n",
                                               encoding="utf-8")
    module = grade_module()
    with pytest.raises(module.GradeError, match="columns"):
        module.grade_hartmann(tmp_path, HARTMANN, MEAN_VELOCITY, HALF_HEIGHT)


def test_the_grader_runs_through_its_own_command_line(tmp_path, grade):
    eta, u = perfect()
    write_profile(tmp_path, eta, u)
    proc = subprocess.run(
        [sys.executable, str(GRADER), "hartmann", str(tmp_path),
         "--hartmann", "20", "--mean-velocity", "1", "--half-height", "1"],
        text=True, capture_output=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASSED" in proc.stdout
    assert "FAIL" not in proc.stdout
