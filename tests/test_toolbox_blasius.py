"""`blasius.py`: the plate constants, and that the typed one is now checked.

The script exists because `reference_area.py` carries `Cf = 1.328/sqrt(Re)` as a literal
-- the friction floor no laminar total drag can sit below -- and nothing in this
repository derived it. These tests hold the derivation to the published `f''(0)`, hold
the literal to the derivation, and pin both convention traps the script reports, because
a reference quoted in two conventions a factor apart is the error this repository keeps
finding and `blasius.py` is the first case whose whole point is one.

They do not re-implement the ODE. `benchmarks/README.md`'s argument about graders applies
here: a second shooting solve would be a second thing to keep right.
"""
from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLBOX = ROOT / "openreynolds" / "toolbox"
SCRIPT = TOOLBOX / "blasius.py"


@pytest.fixture(scope="module")
def blasius():
    spec = importlib.util.spec_from_file_location("toolbox_blasius", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def solved(blasius):
    """One solve, shared: the default is a second of arithmetic but not free."""
    return blasius.report(10.0, 2000)


def test_the_shot_recovers_the_published_second_derivative(solved):
    """`f''(0) = 0.3320573362151963` is the one number here taken rather than derived."""
    assert abs(solved["fpp0_error"]) < 1e-7, solved["fpp0_error"]


def test_the_typed_constant_in_reference_area_is_four_times_it(blasius, solved):
    """The reason the script exists. `reference_area.blasius` returns `1.328/sqrt(Re)`,
    and 1.328 should be `4 f''(0)`. If somebody edits either, this says so."""
    from openreynolds.toolbox import reference_area

    whole_plate = solved["implied"]["Cf sqrt(Re_L), whole plate"][0]
    assert abs(whole_plate - 4.0 * solved["fpp0"]) < 1e-12, "the algebra drifted"
    # reference_area returns Cf at a Reynolds number; multiply back out.
    typed = reference_area.blasius(1.0e6) * math.sqrt(1.0e6)
    assert abs(typed - blasius.TYPED_IN_REFERENCE_AREA) < 1e-12, (
        f"reference_area.blasius no longer carries "
        f"{blasius.TYPED_IN_REFERENCE_AREA}; it gives {typed}")
    assert abs((whole_plate - typed) / typed) < 1e-3, (
        f"the derived {whole_plate:.7f} and the typed {typed} differ by "
        f"{(whole_plate - typed) / typed:+.4%}, which is more than a rounding")


def test_the_other_convention_is_exactly_root_two_away(blasius, solved):
    """0.4696 and 0.332057 are the same solution. A reader who mixes them is wrong by
    41.4%, which is the same class of error as Gartling's 6.1 in the wrong height."""
    other = solved["implied"][blasius.OTHER_KEY][0]
    assert abs(other / solved["fpp0"] - math.sqrt(2.0)) < 1e-12
    assert abs(other - blasius.OTHER_CONVENTION) / blasius.OTHER_CONVENTION < 1e-4, other


def test_the_textbook_thickness_is_a_rounding_and_is_reported_as_one(blasius, solved):
    """`delta_99 = 5.0 x/sqrt(Re_x)` is in every textbook and the computed value is
    4.91. The script has to print the difference rather than adopt the 5.0."""
    assert 4.88 < solved["eta_99"] < 4.94, solved["eta_99"]
    gap = (blasius.TEXTBOOK_DELTA99 - solved["eta_99"]) / solved["eta_99"]
    assert 0.015 < gap < 0.021, f"the textbook gap is {gap:+.2%}, expected about +1.8%"


def test_the_integrator_order_is_measured_against_its_own_refinement(blasius):
    """The check that was wrong first: differenced against the published value, the error
    at 100+ steps is set by the finite `eta_max`, so the "order" came out 0.15. Against a
    refinement of the same solve it is the integrator that is measured."""
    observed, errors = blasius.order(10.0)
    assert observed >= 3.9, f"observed order {observed}"
    steps = [count for count, _ in errors]
    assert steps == sorted(steps), "the step counts must increase"
    values = [value for _, value in errors]
    assert values == sorted(values, reverse=True), f"the error does not fall: {values}"


def test_the_displacement_thickness_carries_no_verdict(blasius):
    """No source for it was read here, so it is printed and not graded. The pattern is
    `A/D` in `benchmarks/moving_mesh/` and the dam break's surge front."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "no verdict" in text, "delta* must be reported as ungraded, in words"
    assert "displacement" in text


def test_it_is_not_claimed_to_be_a_solved_flat_plate():
    """The gap this does NOT close. An OpenFOAM flat-plate run still does not exist in
    this repository, and a reader must not take this case for one."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "not" in text and "OpenFOAM flat-plate" in text, (
        "the script must say it is not a solved flat plate")


def test_check_exits_zero_and_says_it_agrees():
    """`--check` is what `run_tier1.py` reads, so the word and the status both matter."""
    done = subprocess.run([sys.executable, "-X", "utf8", str(SCRIPT), "--check"],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "agrees" in done.stdout
    assert "FAIL" not in done.stdout, done.stdout


def test_a_bracket_that_misses_is_refused_rather_than_returned(blasius):
    """The shot asserts its bracket. A silent wrong root here would be a wrong reference
    for every laminar plate claim downstream."""
    with pytest.raises(AssertionError):
        blasius.shoot(eta_max=10.0, steps=200, low=0.4, high=1.0)
