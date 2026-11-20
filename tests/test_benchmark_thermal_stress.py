"""`benchmarks/thermal_stress/grade.py`: the closed form, and what it can tell apart.

The grader has no fixture of a real run behind it the way the Felix and moving-mesh
graders do, because a thermal-stress case needs OpenFOAM to produce one and this
suite runs on machines that have none. What it has instead is better in one respect
and worse in another. Better: the answer is exact, so a profile can be written here
from the closed form and the grader has to accept it to within the solver tolerance it
claims. Worse: that proves the arithmetic agrees with itself and not that the solver
agrees with it. The second half is the `journey` job in `.github/workflows/tests.yml`,
which runs the template, solves the case in the image, and runs this grader on what
came out; the `image` job's check.sh gate asserts the same closed form independently.

So the tests here are the ones that fixture cannot do: that the grader accepts the
exact answer, and that it rejects the four ways of being wrong that would otherwise
pass unnoticed -- the plane-stress switch read the wrong way round, the thermal term
missing, its sign flipped, and a case where `thermalStress` was never switched on.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GRADE = ROOT / "benchmarks" / "thermal_stress" / "grade.py"

E_PA = 2.0e11
NU = 0.3
ALPHA = 1.1e-5
DELTA_T = 100.0
HEIGHT = 0.02


@pytest.fixture(scope="module")
def grade():
    spec = importlib.util.spec_from_file_location("thermal_stress_grade", GRADE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_case(root: Path, *, plane_stress: bool = True, thermal_stress: bool = True,
               factor: float | None = None, points: int = 41) -> Path:
    """A case directory holding only what the grader reads, with an exact profile.

    `factor` overrides the thermal factor the profile is written with, which is how
    the wrong answers below are made: the grader derives its own from the case's
    `planeStress` switch and nu, so writing the profile with another one is exactly
    the mistake of reading that switch the wrong way round.
    """
    case = root / ("plane_stress" if plane_stress else "plane_strain")
    (case / "constant").mkdir(parents=True, exist_ok=True)
    (case / "0").mkdir(parents=True, exist_ok=True)
    (case / "constant" / "mechanicalProperties").write_text(
        f"rho {{ type uniform; value 7854; }}\n"
        f"nu {{ type uniform; value {NU}; }}\n"
        f"E {{ type uniform; value {E_PA:g}; }}\n"
        f"planeStress     {'yes' if plane_stress else 'no'};\n")
    (case / "constant" / "thermalProperties").write_text(
        f"C {{ type uniform; value 434; }}\n"
        f"k {{ type uniform; value 60.5; }}\n"
        f"alpha {{ type uniform; value {ALPHA:g}; }}\n"
        f"thermalStress   {'yes' if thermal_stress else 'no'};\n")
    (case / "0" / "T").write_text(
        "dimensions [0 0 0 1 0 0 0];\ninternalField uniform 0;\n"
        "boundaryField\n{\n"
        f"    hot {{ type fixedValue; value uniform {DELTA_T:g}; }}\n"
        "    cold { type fixedValue; value uniform 0; }\n"
        "}\n")

    if factor is None:
        factor = 1.0 if plane_stress else 1.0 / (1.0 - NU)
    out = case / "postProcessing" / "singleGraph" / "100"
    out.mkdir(parents=True, exist_ok=True)
    rows = ["# y T sigmaxx sigmayy sigmaxy"]
    for index in range(points):
        y = HEIGHT * index / (points - 1)
        temperature = DELTA_T * (1.0 - y / HEIGHT)
        sigma_xx = -E_PA * ALPHA * temperature * factor
        rows.append(f"{y:.10g} {temperature:.10g} {sigma_xx:.10g} 0 0")
    (out / "line_T_sigmaxx_sigmayy_sigmaxy.xy").write_text("\n".join(rows) + "\n")
    return case


# -- the exact answer is accepted ---------------------------------------------------


@pytest.mark.parametrize("plane_stress", [True, False])
def test_the_closed_form_passes_every_check(tmp_path, grade, plane_stress):
    result = grade.grade_bar(write_case(tmp_path, plane_stress=plane_stress))
    assert all(item["pass"] for item in result["checks"]), [
        item for item in result["checks"] if not item["pass"]]
    assert len(result["checks"]) == 5, "a check was dropped"


def test_the_two_states_differ_by_exactly_one_over_one_minus_nu(tmp_path, grade):
    """The reason grading the plane-strain run is also a check on nu. -220.0 MPa and
    -314.2857 MPa for the steel above: 43% apart, far outside any threshold here."""
    stress = grade.grade_bar(write_case(tmp_path, plane_stress=True))
    strain = grade.grade_bar(write_case(tmp_path, plane_stress=False))
    a = stress["measured"]["sigma_xx_hot_exact_pa"]
    b = strain["measured"]["sigma_xx_hot_exact_pa"]
    assert a == pytest.approx(-E_PA * ALPHA * DELTA_T)
    assert a == pytest.approx(-2.2e8)
    assert b / a == pytest.approx(1.0 / (1.0 - NU))
    assert b == pytest.approx(-2.2e8 / 0.7)


def test_the_hot_face_stress_is_minus_e_alpha_delta_t(grade, tmp_path):
    """No elastic constant survives in the plane-stress answer but E, which is the
    one line of the derivation worth asserting rather than only writing down."""
    y = grade.np.array([0.0, HEIGHT / 2, HEIGHT])
    settings = grade.read_case(write_case(tmp_path, plane_stress=True))
    answer = grade.exact(settings, y)
    assert answer["sigma_xx"][0] == pytest.approx(-E_PA * ALPHA * DELTA_T)
    assert answer["sigma_xx"][1] == pytest.approx(-E_PA * ALPHA * DELTA_T / 2)
    assert answer["sigma_xx"][2] == pytest.approx(0.0)
    # ex_yy carries nu where sigma_xx does not, which is what a displacement check
    # would add over a stress check. 1.3 * 1.1e-3 = 1.43e-3 at the hot face.
    assert answer["strain_yy"][0] == pytest.approx((1 + NU) * ALPHA * DELTA_T)
    assert answer["strain_yy"][0] == pytest.approx(1.43e-3)


# -- and the four wrong answers are rejected ----------------------------------------


def test_the_plane_stress_switch_read_backwards_fails(tmp_path, grade):
    """A run that is plane strain and says plane stress, or the reverse. This is the
    failure a single-state benchmark cannot see at all."""
    case = write_case(tmp_path, plane_stress=True, factor=1.0 / (1.0 - NU))
    result = grade.grade_bar(case)
    assert not all(item["pass"] for item in result["checks"])


def test_the_thermal_term_missing_altogether_fails(tmp_path, grade):
    result = grade.grade_bar(write_case(tmp_path, factor=0.0))
    failed = [item["name"] for item in result["checks"] if not item["pass"]]
    assert any("sigma_xx" in name for name in failed)


def test_the_thermal_term_with_the_wrong_sign_fails(tmp_path, grade):
    """`DEqn += fvc::grad(threeKalpha*T)` puts the term on the matrix side, so the
    equation carries it with a minus. A compression that came out a tension is a
    sign error and not a small one."""
    result = grade.grade_bar(write_case(tmp_path, factor=-1.0))
    assert not all(item["pass"] for item in result["checks"])


def test_a_case_that_never_switched_thermal_stress_on_is_refused(tmp_path, grade):
    """`thermalStress no` is what the one shipped tutorial has, so this is the likely
    first mistake, and it is refused rather than graded: the run solved pure
    elasticity and its zero stress is not a disagreement with the closed form, it is
    a different case."""
    with pytest.raises(grade.GradeError, match="thermalStress no"):
        grade.grade_bar(write_case(tmp_path, thermal_stress=False))


def test_an_absolute_temperature_is_read_as_what_it_is(tmp_path, grade):
    """The trap: the solver subtracts no reference temperature, so a case written with
    an ambient 300 in `0/T` is asking for the stress of a 300 K rise. The grader reads
    300 and says so rather than quietly treating it as a rise of zero."""
    case = write_case(tmp_path)
    (case / "0" / "T").write_text(
        "boundaryField\n{\n"
        "    hot { type fixedValue; value uniform 300; }\n"
        "    cold { type fixedValue; value uniform 300; }\n}\n")
    assert grade.read_case(case)["delta_t_k"] == 300.0


# -- the reading, which is where a grader usually breaks ----------------------------


def test_a_profile_with_no_header_falls_back_to_the_documented_order(tmp_path, grade):
    case = write_case(tmp_path)
    path = next((case / "postProcessing").rglob("*.xy"))
    rows = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    path.write_text("\n".join(rows) + "\n")
    assert all(item["pass"] for item in grade.grade_bar(case)["checks"])


def test_a_profile_with_no_header_and_the_wrong_width_is_refused(tmp_path, grade):
    """Guessing which column is which is how a grader comes to compare the wrong
    field, so an unlabelled file of an unexpected width stops it instead."""
    case = write_case(tmp_path)
    path = next((case / "postProcessing").rglob("*.xy"))
    path.write_text("0 1 2\n0.01 3 4\n")
    with pytest.raises(grade.GradeError, match="columns"):
        grade.grade_bar(case)


def test_the_newest_time_directory_is_the_one_graded(tmp_path, grade):
    """OpenFOAM writes one directory per write interval and `-noZero` leaves the
    earlier ones there. Grading an early write would grade an unconverged field."""
    case = write_case(tmp_path)
    early = case / "postProcessing" / "singleGraph" / "20"
    early.mkdir(parents=True)
    (early / "line_T_sigmaxx_sigmayy_sigmaxy.xy").write_text(
        "# y T sigmaxx sigmayy sigmaxy\n0 50 0 0 0\n0.02 0 0 0 0\n")
    result = grade.grade_bar(case)
    assert "/100/" in result["measured"]["profile"].replace("\\", "/")
    assert all(item["pass"] for item in result["checks"])


def test_a_case_with_no_profile_says_which_commands_write_one(tmp_path, grade):
    case = write_case(tmp_path)
    for path in sorted((case / "postProcessing").rglob("*.xy")):
        path.unlink()
    with pytest.raises(grade.GradeError, match="singleGraph"):
        grade.grade_bar(case)


def test_a_field_valued_property_is_refused_rather_than_averaged(tmp_path, grade):
    """`type field` means E or alpha is a whole volScalarField on disk; the closed
    form has one of each in it and does not describe that case."""
    case = write_case(tmp_path)
    (case / "constant" / "mechanicalProperties").write_text(
        "rho { type uniform; value 7854; }\n"
        f"nu {{ type uniform; value {NU}; }}\n"
        "E { type field; value 0; }\n"
        "planeStress     yes;\n")
    with pytest.raises(grade.GradeError, match="not uniform|type field"):
        grade.grade_bar(case)


# -- the command line --------------------------------------------------------------


def test_the_exit_status_is_zero_on_a_pass_and_one_on_a_fail(tmp_path, grade, capsys):
    good = write_case(tmp_path, plane_stress=True)
    assert grade.main(["bar", str(good)]) == 0
    assert "PASS" in capsys.readouterr().out
    bad = write_case(tmp_path, plane_stress=False, factor=1.0)
    assert grade.main(["bar", str(bad)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_an_ungradeable_case_exits_two(tmp_path, grade, capsys):
    assert grade.main(["bar", str(tmp_path / "nothing")]) == 2
    assert "cannot grade" in capsys.readouterr().err


def test_the_json_output_carries_the_numbers(tmp_path, grade, capsys):
    import json
    assert grade.main(["bar", str(write_case(tmp_path)), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["settings"]["factor"] == 1.0
    assert payload["measured"]["sigma_xx_hot_exact_pa"] == pytest.approx(-2.2e8)
    assert len(payload["checks"]) == 5
