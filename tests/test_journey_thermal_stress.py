"""The thermal-stress journey: template -> mesh -> solve -> sample -> graded.

The companion to `test_journey_duct.py`, and the second test in this repository that
actually solves something. It exists because every other test of this capability can
be satisfied without the solver: `test_toolbox_templates.py` parses the template,
`test_benchmark_thermal_stress.py` feeds the grader a profile written from the closed
form, and both pass on a laptop with no OpenFOAM on it. Neither can tell you that
`solidDisplacementFoam` is in the image, that `thermalStress yes` does anything, or
that the number it produces is the right one.

the case, and why its answer is known exactly

`openreynolds/toolbox/templates/thermal_stress.py` as it ships: a 0.04 x 0.02 m bar of
steel one cell thick, held between two `symmetryPlane` ends, resting on a hot face it
is fixed to, free at the cold face on top, with a 100 K temperature rise across it.
`benchmarks/thermal_stress/README.md` derives its answer in four lines and this file
asserts it independently of the grader as well as through it:

    T(y)        = dT * (1 - y/H)                      exactly linear
    sigma_xx(y) = -E * alpha * T(y) * f                f = 1 (plane stress)
    sigma_yy    = sigma_xy = 0                         f = 1/(1-nu) (plane strain)
    max sigmaEq = E * alpha * dT * f

-220.0000 MPa and -314.2857 MPa on the hot face for the template's steel. No published
value, no reference uncertainty, nothing to misquote: the constants are the case's own
and the algebra is in the README beside the grader.

Both plane states are run, because the plane-stress answer contains E and alpha only
as their product and so cannot distinguish a wrong nu, while the plane-strain answer
carries 1/(1-nu) and the two are 43% apart. Each solve is 800 cells and 100 steps.

what is asserted at each seam, and not at the end

* the template wrote a case the mesh tools accept, with the two ends `symmetryPlane`
  and the z faces `empty` -- a "2D" case that is really a thin 3D one, or an end left
  as a plain `patch`, changes the answer rather than failing;
* `thermalStress` really is `yes` in what the template wrote, which is the switch the
  distribution's own tutorial ships as `no`;
* the solver ran to `End` and its own per-write `Max sigmaEq` line is the closed form,
  which is the one number that needs no file read at all;
* `components(sigma)` and `singleGraph` wrote a profile, in that order, because
  `sigma` is a symmetric tensor and the sampler wants scalars;
* the sampled T is the exact linear profile, so the conduction half ran and converged;
* the sampled sigma_xx is -E*alpha*T(y) pointwise and not merely at its peak, which is
  the part a uniformly heated bar could not test: with T linear, grad(3*K*alpha*T) is
  non-zero in every interior cell and is cancelled by the gradient of div(sigmaD);
* the top face carries no traction, so sigma_yy and sigma_xy are zero;
* and `benchmarks/thermal_stress/grade.py` agrees, through its own command line, with
  the exit status a person would read.

Skipped where there is no OpenFOAM. The place this runs is the `journey` job in
`.github/workflows/tests.yml`, inside the workspace image.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
TEMPLATE = ROOT / "openreynolds" / "toolbox" / "templates" / "thermal_stress.py"
GRADER = ROOT / "benchmarks" / "thermal_stress" / "grade.py"

BINARIES = ("gmshToFoam", "checkMesh", "foamDictionary", "solidDisplacementFoam",
            "postProcess")


def _absent() -> list[str]:
    missing = [name for name in BINARIES if shutil.which(name) is None]
    if importlib.util.find_spec("gmsh") is None:
        missing.append("the gmsh module")
    return missing


pytestmark = pytest.mark.skipif(
    bool(_absent()),
    reason="not an OpenFOAM machine, so no case can be solved here: "
           + ", ".join(_absent() or ["-"]) + " missing",
)


# -- the case, exactly as the template defines it ------------------------------------

BAR_HEIGHT_M = 0.02          # thermal_stress.py  BAR_HEIGHT_M
DELTA_T_K = 100.0            # thermal_stress.py  DELTA_T_K
E_PA = 2.0e11                # thermal_stress.py  E_PA
NU = 0.3                     # thermal_stress.py  NU
ALPHA_PER_K = 1.1e-5         # thermal_stress.py  ALPHA_PER_K

# The closed form, written out here rather than imported from the template or the
# grader: a test that gets its expected value from the code under test asserts only
# that the code is self-consistent.
PEAK_PLANE_STRESS_PA = E_PA * ALPHA_PER_K * DELTA_T_K            # 2.2e8
PEAK_PLANE_STRAIN_PA = PEAK_PLANE_STRESS_PA / (1.0 - NU)         # 3.142857e8


def run(cmd: list[str], cwd: Path, label: str = "") -> str:
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    elapsed = time.monotonic() - started
    print(f"  {label or cmd[0]}: {elapsed:.1f} s, exit {proc.returncode}")
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:], file=sys.stderr)
        pytest.fail(f"{cmd} failed (exit {proc.returncode})")
    return proc.stdout


def build_and_solve(case: Path, plane_stress: bool) -> str:
    """The template, run as a person runs it, then the solver and the two samplers.

    `PLANE_STRESS` is edited in the copy rather than passed in, because editing a
    constant at the top of a copied template and re-running it is the whole interface
    a template has, and a test that bypassed it would not be testing that.
    """
    case.mkdir(parents=True)
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "PLANE_STRESS = True" in text, "the template's switch has been renamed"
    if not plane_stress:
        text = text.replace("PLANE_STRESS = True", "PLANE_STRESS = False", 1)
    (case / "thermal_stress.py").write_text(text, encoding="utf-8")

    print(f"\n[{'plane stress' if plane_stress else 'plane strain'}] {case}")
    run([sys.executable, "thermal_stress.py"], case, "template")
    solve = run(["solidDisplacementFoam"], case, "solidDisplacementFoam")
    run(["postProcess", "-func", "components(sigma)", "-noZero"], case, "components")
    run(["postProcess", "-func", "singleGraph", "-noZero"], case, "singleGraph")
    return solve


# -- the seams ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def plane_stress_run(tmp_path_factory) -> tuple[Path, str]:
    case = tmp_path_factory.mktemp("thermal") / "plane_stress"
    return case, build_and_solve(case, plane_stress=True)


@pytest.fixture(scope="module")
def plane_strain_run(tmp_path_factory) -> tuple[Path, str]:
    case = tmp_path_factory.mktemp("thermal") / "plane_strain"
    return case, build_and_solve(case, plane_stress=False)


def test_the_template_wrote_a_two_dimensional_case_with_the_ends_constrained(
        plane_stress_run):
    """`gmshToFoam` types every patch `patch`; the retypes are what make the case the
    one whose answer is known. An end left a plain `patch` is traction-free, the bar
    expands freely, and the stress is zero rather than wrong by a little."""
    case, _ = plane_stress_run
    boundary = (case / "constant" / "polyMesh" / "boundary").read_text()
    for patch in ("end_left", "end_right"):
        index = boundary.index(patch)
        assert "symmetryPlane" in boundary[index:index + 200], f"{patch} is not constrained"
    index = boundary.index("frontAndBack")
    assert "empty" in boundary[index:index + 200], "the case is a thin 3D one, not 2D"

    mechanical = (case / "constant" / "mechanicalProperties").read_text()
    assert "planeStress     yes;" in mechanical
    thermal = (case / "constant" / "thermalProperties").read_text()
    assert "thermalStress   yes;" in thermal, (
        "the thermal half of the solver is off, which is what the distribution's own "
        "plateHole tutorial ships")

    # T is a rise above the stress-free state, not an absolute temperature: this
    # solver subtracts no reference temperature anywhere.
    temperature = (case / "0" / "T").read_text()
    assert "internalField   uniform 0;" in temperature
    assert f"value           uniform {DELTA_T_K:g};" in temperature


@pytest.mark.parametrize("plane_stress", [True, False])
def test_the_solvers_own_max_sigma_eq_line_is_the_closed_form(
        plane_stress, plane_stress_run, plane_strain_run):
    """One number, out of the solver's own stdout, with no file read: here sigma is
    diag(s, 0, s), whose von Mises equivalent sqrt(1.5*magSqr(dev(sigma))) is |s|.
    This is the assertion `docker/workspace/check.sh` gates the image on."""
    _, log = plane_stress_run if plane_stress else plane_strain_run
    expected = PEAK_PLANE_STRESS_PA if plane_stress else PEAK_PLANE_STRAIN_PA
    lines = [line for line in log.splitlines() if "Max sigmaEq" in line]
    assert lines, "the solver printed no `Max sigmaEq` line, so it wrote no stress"
    assert "End" in log
    got = float(lines[-1].split("=")[1])
    print(f"  max sigmaEq {got / 1e6:.4f} MPa against an exact {expected / 1e6:.4f} MPa "
          f"({abs(got - expected) / expected:.3%})")
    assert got == pytest.approx(expected, rel=5e-3)


def test_the_two_plane_states_differ_by_one_over_one_minus_nu(plane_stress_run,
                                                             plane_strain_run):
    """The check that pins nu. The plane-stress answer contains E and alpha only as
    their product, so it cannot distinguish a wrong nu; these two are 43% apart."""
    stress = float([line for line in plane_stress_run[1].splitlines()
                    if "Max sigmaEq" in line][-1].split("=")[1])
    strain = float([line for line in plane_strain_run[1].splitlines()
                    if "Max sigmaEq" in line][-1].split("=")[1])
    assert strain / stress == pytest.approx(1.0 / (1.0 - NU), rel=1e-2)


def _profile(case: Path) -> dict[str, np.ndarray]:
    spec = importlib.util.spec_from_file_location("thermal_stress_grade", GRADER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_profile(case)


def test_the_sampled_temperature_is_the_exact_linear_profile(plane_stress_run):
    """The conduction half, on its own. A profile that is not linear has not reached
    a steady state, whatever the stress says."""
    case, _ = plane_stress_run
    profile = _profile(case)
    y = profile["y"]
    assert y.size >= 10, f"only {y.size} sample points"
    assert float(np.max(y)) == pytest.approx(BAR_HEIGHT_M, rel=1e-3)
    exact = DELTA_T_K * (1.0 - y / float(np.max(y)))
    error = float(np.max(np.abs(profile["T"] - exact))) / DELTA_T_K
    print(f"  T profile: max error {error:.3e} of dT over {y.size} points")
    assert error < 2e-3


@pytest.mark.parametrize("plane_stress", [True, False])
def test_the_sampled_stress_is_the_closed_form_pointwise(
        plane_stress, plane_stress_run, plane_strain_run):
    """Pointwise and not only at the peak. This is what a uniformly heated bar cannot
    test: with T linear in y, `grad(3*K*alpha*T)` is non-zero in every interior cell
    and the answer is its cancellation against the gradient of `div(sigmaD)`."""
    case, _ = plane_stress_run if plane_stress else plane_strain_run
    peak = PEAK_PLANE_STRESS_PA if plane_stress else PEAK_PLANE_STRAIN_PA
    profile = _profile(case)
    y = profile["y"]
    exact = -peak * (1.0 - y / float(np.max(y)))
    residual = profile["sigmaxx"] - exact
    rms = float(np.sqrt(np.mean(residual**2))) / peak
    worst = float(np.max(np.abs(residual))) / peak
    print(f"  sigma_xx: RMS {rms:.3e}, max {worst:.3e} of the {peak / 1e6:.4f} MPa peak")
    assert rms < 5e-3 and worst < 2e-2
    # The hot face carries the whole of it, and that is the number the README quotes.
    assert float(profile["sigmaxx"][int(np.argmin(y))]) == pytest.approx(-peak, rel=2e-2)


def test_the_free_face_carries_no_traction(plane_stress_run):
    case, _ = plane_stress_run
    profile = _profile(case)
    for component in ("sigmayy", "sigmaxy"):
        assert component in profile, f"{component} was not sampled"
        worst = float(np.max(np.abs(profile[component]))) / PEAK_PLANE_STRESS_PA
        print(f"  {component}: max {worst:.3e} of the peak sigma_xx")
        assert worst < 2e-2


@pytest.mark.parametrize("plane_stress", [True, False])
def test_the_benchmark_grader_agrees_through_its_own_command_line(
        plane_stress, plane_stress_run, plane_strain_run):
    """The grader as a person runs it, exit status and all. The assertions above are
    this test's own reading of the same files; this is the published grader's."""
    case, _ = plane_stress_run if plane_stress else plane_strain_run
    proc = subprocess.run([sys.executable, str(GRADER), "bar", str(case)],
                          text=True, capture_output=True)
    print(proc.stdout)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "FAIL" not in proc.stdout
    assert proc.stdout.count("PASS") == 5
