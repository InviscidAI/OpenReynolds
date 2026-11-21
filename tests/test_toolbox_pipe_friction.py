"""`pipe_friction.py`: the turbulent references, and the line between exact and fitted.

The tutorial index calls "a turbulent case with a reference and a measured agreement"
**the largest single gap in this index**. This is its reference half, and it is the first
case here whose references are **not exact** -- Blasius and Colebrook are fits to
Nikuradse's 1930s pipe measurements, which `ladder.py` already says in those words.

So these tests grade the arithmetic and the agreement BETWEEN correlations, which is
checkable to machine precision, and never a correlation against nature. They also pin the
two claims this repository already makes: `ladder.py`'s range for the Blasius fit, and
`wall.py`'s y+ band edges.
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
SCRIPT = TOOLBOX / "pipe_friction.py"


@pytest.fixture(scope="module")
def friction():
    spec = importlib.util.spec_from_file_location("toolbox_pipe_friction", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_laminar_branch_is_exact(friction):
    """`f = 64/Re` comes out of the Hagen-Poiseuille profile, so it is the one reference
    here with no scatter behind it."""
    assert abs(friction.laminar(2000.0) - 0.032) < 1e-15
    assert abs(friction.laminar(64.0) - 1.0) < 1e-15


def test_colebrook_satisfies_its_own_equation(friction):
    """It is implicit in f and solved by iteration, so the check is the residual of the
    equation itself -- not a quoted table value. A solver that returned its starting
    guess would look like a result."""
    for exponent in (4, 5, 6, 7, 8):
        reynolds = 10.0 ** exponent
        f = friction.colebrook(reynolds)
        residual = abs(1.0 / math.sqrt(f)
                       + 2.0 * math.log10(2.51 / (reynolds * math.sqrt(f))))
        assert residual < 1e-10, (reynolds, residual)


def test_haalands_claim_about_itself_is_measured(friction):
    """Haaland is an explicit stand-in for Colebrook and claims a couple of per cent.
    That is a checkable statement between two formulas, unlike either against a pipe."""
    worst = max(abs(friction.haaland(10.0 ** n) / friction.colebrook(10.0 ** n) - 1.0)
                for n in (4, 5, 6, 7, 8))
    assert worst < 0.02, worst


def test_the_ladder_range_for_the_blasius_fit_holds(friction):
    """`ladder.py` says the fit holds "to about Re 1e5". Measured: it leaves a 5% band of
    smooth Colebrook at Re 2.2e5, so "about" is doing real work and the order of
    magnitude is right."""
    departs = friction.blasius_departs()
    assert 1.0e5 < departs < 1.0e6, departs
    text = (TOOLBOX / "ladder.py").read_text(encoding="utf-8")
    assert "0.316*Re^-0.25" in text, "ladder.py no longer quotes the Blasius fit"
    assert "Colebrook" in text


def test_friction_falls_with_reynolds_and_rises_with_roughness(friction):
    """Two monotonicities that any wrong sign would break, and which need no reference."""
    values = [friction.colebrook(10.0 ** n) for n in (4, 5, 6, 7, 8)]
    assert all(a > b for a, b in zip(values, values[1:])), values
    assert friction.colebrook(1.0e7, 1.0e-3) > friction.colebrook(1.0e7, 0.0)


def test_the_wall_law_crossing_sits_in_the_buffer_band(friction):
    """`wall.py` bands y+ at 5 and 30 and picks wall functions on those edges. Given
    kappa and B the crossing of `u+ = y+` with the log law is a number, not a
    convention, and it has to sit inside that band for the band to mean anything."""
    where = friction.crossing()
    assert 10.0 < where < 12.0, where
    assert friction.SUBLAYER_EDGE < where < friction.LOG_EDGE
    text = (TOOLBOX / "wall.py").read_text(encoding="utf-8")
    assert '("buffer layer", 5.0, 30.0)' in text, (
        "wall.py's band edges moved; this case grades the crossing against them")


def test_the_published_spread_is_a_range_over_four_corners(friction):
    """The bug this test exists for, made while writing the case: the spread was printed
    as `(kappa_low, B_low)` to `(kappa_high, B_high)`, which is two arbitrary corners and
    not a range -- a smaller kappa RAISES the crossing while a smaller B lowers it, so
    both printed corners sat above the nominal and the "range" excluded its own central
    value."""
    low, high = friction.crossing_range()
    nominal = friction.crossing()
    assert low <= nominal <= high, (low, nominal, high)
    assert low < high
    corners = [friction.crossing(k, b)
               for k in friction.KAPPA_SPREAD for b in friction.B_SPREAD]
    assert abs(low - min(corners)) < 1e-12
    assert abs(high - max(corners)) < 1e-12


def test_the_impossible_is_refused(friction):
    for bad in (lambda: friction.laminar(0.0), lambda: friction.colebrook(-1.0),
                lambda: friction.colebrook(1.0e5, -0.1), lambda: friction.haaland(0.0),
                lambda: friction.blasius_fit(0.0)):
        with pytest.raises(ValueError):
            bad()


def test_check_exits_zero_and_says_it_agrees():
    done = subprocess.run([sys.executable, "-I", "-X", "utf8", str(SCRIPT), "--check"],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "agrees" in done.stdout
    assert "FAIL" not in done.stdout, done.stdout


def test_it_does_not_claim_a_graded_turbulent_solve():
    """The gap this does NOT close. No turbulent OpenFOAM run is graded in this
    repository, and the index's largest gap stays open on the solve."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "No turbulent OpenFOAM run" in text
    assert "not exact" in text, "the fitted references must be labelled as fits"
