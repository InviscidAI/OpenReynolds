"""`nozzle.py`: the exact compressible relations, and the two claims they now check.

The tutorial index said it outright -- "No `compressible.py` case is graded against
anything yet; that is a gap and is named as one" -- and the compressible family is the
largest in `advise.py`. These tests hold the relations to identities rather than to
quoted values wherever an identity exists, because an identity cannot be misquoted:
`A/A*` is exactly 1 at the throat, `M1* M2* = 1` across a shock, and total pressure can
only fall through one.

They also pin the two things in this repository the case exists to check: that
`compressible.stagnation` agrees with the isentropic relation written out independently,
and that `advise.COMPRESSIBLE_SWITCH`'s "under 5%" at M 0.3 is true.
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
SCRIPT = TOOLBOX / "nozzle.py"


@pytest.fixture(scope="module")
def nozzle():
    spec = importlib.util.spec_from_file_location("toolbox_nozzle", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_throat_is_exactly_one_and_is_the_minimum(nozzle):
    """`A/A* = 1` at `M = 1` is an identity, so machine precision is the only bar. And it
    must be the MINIMUM: a nozzle that reads the other branch is a different flow."""
    assert abs(nozzle.area_ratio(1.0) - 1.0) < 1e-14
    assert nozzle.area_ratio(0.999) > 1.0
    assert nozzle.area_ratio(1.001) > 1.0


def test_the_standard_table_at_mach_two(nozzle):
    """The one place a quoted value is used, because these are the numbers in every gas
    table and getting them wrong would mean the formulas are wrong, not the precision:
    at M 2, gamma 1.4 -- T0/T 1.8, A/A* 1.6875, M2 1/sqrt(3), p2/p1 4.5."""
    assert abs(nozzle.isentropic(2.0)["T0/T"] - 1.8) < 1e-14
    assert abs(nozzle.area_ratio(2.0) - 1.6875) < 1e-12
    jump = nozzle.shock(2.0)
    assert abs(jump["M2"] - 1.0 / math.sqrt(3.0)) < 1e-12
    assert abs(jump["p2/p1"] - 4.5) < 1e-12
    assert abs(jump["T2/T1"] - 1.6875) < 1e-12


def test_the_prandtl_relation_grades_the_jump_relations(nozzle):
    """`M1* M2* = 1` comes from the energy equation, not from the jump formulas, so it is
    an independent check on them rather than a restatement. This is what makes the case
    evidence instead of arithmetic agreeing with itself."""
    worst = 0.0
    for tenth in range(10, 60):
        m1 = tenth / 10.0
        product = (nozzle.characteristic_mach(m1)
                   * nozzle.characteristic_mach(nozzle.shock(m1)["M2"]))
        worst = max(worst, abs(product - 1.0))
    assert worst < 1e-12, f"worst {worst}"


def test_total_pressure_only_falls_through_a_shock(nozzle):
    """The irreversibility. At exactly M1 = 1 the shock is infinitely weak and the ratio
    is 1; above it the loss is strict."""
    assert abs(nozzle.shock(1.0)["p02/p01"] - 1.0) < 1e-12
    for tenth in range(11, 60):
        assert nozzle.shock(tenth / 10.0)["p02/p01"] < 1.0


def test_an_expansion_shock_is_refused_by_name(nozzle):
    """The second law. Every jump relation returns a finite number for `M1 < 1`, and a
    report carrying one would look like a finding."""
    with pytest.raises(ValueError, match="second law"):
        nozzle.shock(0.8)


def test_the_area_inversion_needs_its_branch_stated(nozzle):
    """Two Mach numbers share an `A/A*`. Which one a section is at is a boundary
    condition, so the branch is an argument -- and both branches round-trip."""
    for tenth in list(range(1, 10)) + list(range(11, 50)):
        m = tenth / 10.0
        back = nozzle.mach_from_area(nozzle.area_ratio(m), supersonic=m > 1.0)
        assert abs(back - m) / m < 1e-9, f"M {m} came back {back}"
    with pytest.raises(ValueError):
        nozzle.mach_from_area(0.5, supersonic=False)


def test_the_incompressible_switch_this_repository_asserts_is_true(nozzle):
    """`advise.COMPRESSIBLE_SWITCH` says "M < 0.3 is where density varies by under 5%".
    Nothing checked it until this case. It is true -- 4.5609% -- and the departure
    reaches 5% at M 0.3139, which the script prints beside the 0.3 rather than leaving
    the reader to assume the line is exact."""
    departure = nozzle.density_departure(nozzle.SWITCH_MACH)
    assert departure < nozzle.SWITCH_CLAIM, departure
    assert 0.045 < departure < 0.046, departure
    crossing = nozzle.switch_mach(nozzle.SWITCH_CLAIM)
    assert 0.313 < crossing < 0.315, crossing
    assert crossing > nozzle.SWITCH_MACH, "the switch must be on the safe side"


def test_the_switch_text_in_advise_still_makes_this_claim(nozzle):
    """If somebody rewords the switch, this test is how they find out a case grades it."""
    text = (TOOLBOX / "advise.py").read_text(encoding="utf-8")
    assert "M < 0.3" in text, "advise.py no longer draws the line at M 0.3"
    assert "5%" in text, "advise.py no longer claims a 5% density band"


def test_the_shipped_stagnation_agrees_with_the_relation(nozzle):
    """Two implementations of one formula, one of them shipped. `compressible.py`
    recovers gamma from the FORMATTED Cp it wrote into the case, so its gamma is the
    case's gamma to the digit; this measures the difference rather than assuming it."""
    found = nozzle.cross_check_the_shipped_stagnation(0.8)
    for quantity in ("T0", "p0"):
        here, shipped = found[f"{quantity} here"], found[f"{quantity} shipped"]
        assert abs(shipped - here) / here < 1e-12, (quantity, shipped, here)


def test_check_exits_zero_and_says_it_agrees():
    """What `run_tier1.py` reads."""
    done = subprocess.run([sys.executable, "-X", "utf8", str(SCRIPT), "--check"],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "agrees" in done.stdout
    assert "FAIL" not in done.stdout, done.stdout


def test_it_is_not_claimed_to_be_a_solved_nozzle():
    """Quasi-1D has no boundary layer, no shock thickness and no second dimension. The
    gap in the index is the solve, and this must not read as closing it."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "not a solved nozzle" in text.lower()
