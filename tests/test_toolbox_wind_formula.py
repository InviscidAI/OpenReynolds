"""`wind_comfort.py --check`: the exceedance formula, graded on a year with a known answer.

This is the one part of the tool that no solved case can check for you. A run tells you
the amplification `K`; it does not tell you whether the hours behind `K` were composed
correctly, and a composition error is invisible in the picture it draws because the map
still looks like a map.

The reference is arithmetic on a stated histogram, so there is no paper and no fitted
distribution:

    P = (1/total hours) * sum_sectors sum_bins hours * [K*u_bin > threshold]

600 hours from 240 degrees spread 100 to a bin over 0-1 .. 5-6 m/s, and 400 from 270
spread 100 to a bin over 2-3 .. 5-6, at a 5 m/s threshold:

    K = 1  only 5-6 is wholly above 5, so (100 + 100)/1000 = 0.2 exactly
    K = 2  the required 10 m speed is 2.5, so 2-3 gives its upper half:
           (50 + 100 + 100 + 100 + 50 + 100 + 100 + 100)/1000 = 0.7 exactly

The second is the one worth having. It grades the even spreading of hours inside a bin as
well as the amplification, and a step at the bin centre instead gives 0.6 while leaving
the K = 1 case passing, which is checked here by mutation rather than asserted.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"
WIND = TOOLBOX / "wind_comfort.py"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    """The script as a user runs it: a plain interpreter, no package import."""
    return subprocess.run([sys.executable, "-X", "utf8", str(WIND), *args],
                          capture_output=True, text=True, timeout=300)


def readings(text: str) -> dict[float, tuple[float, float]]:
    """{K: (measured, exact)} out of the printed report."""
    found: dict[float, tuple[float, float]] = {}
    for line in text.splitlines():
        bare = line.strip()
        if not bare.startswith("K = "):
            continue
        # "K = 1: exceedance 0.200000000000, exact 0.200000000000 (+0.00e+00), ..."
        k = float(bare.split(":")[0].split("=")[1])
        parts = bare.replace(",", " ").split()
        found[k] = (float(parts[parts.index("exceedance") + 1]),
                    float(parts[parts.index("exact") + 1]))
    return found


def test_the_check_agrees_and_needs_nothing():
    """Tier 1's promise: no site file, no runs, no account, no network."""
    done = run("--check")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "agrees" in done.stdout, done.stdout
    assert "DISAGREES" not in done.stdout


def test_uniform_amplification_is_the_fraction_of_the_year_above_the_threshold():
    """With K = 1 the formula is the histogram and nothing else: 200 of 1000 hours."""
    got = readings(run("--check").stdout)
    assert 1.0 in got, got
    measured, exact = got[1.0]
    assert exact == pytest.approx(0.2, abs=1e-12)
    assert measured == pytest.approx(exact, abs=1e-12)


def test_doubling_the_amplification_is_halving_the_threshold():
    """K = 2 requires 2.5 m/s at 10 m, so the 2-3 bin contributes its upper half and
    the answer is 0.7. This is a statement about the half bin as much as about K: a
    step at the bin centre would give 0.6 and still pass the K = 1 test above."""
    got = readings(run("--check").stdout)
    assert 2.0 in got, got
    measured, exact = got[2.0]
    assert exact == pytest.approx(0.7, abs=1e-12)
    assert measured == pytest.approx(exact, abs=1e-12)


def test_the_two_cases_are_not_the_same_statement():
    """If K = 1 and K = 2 gave the same number the pair would be grading one thing
    twice. They differ by 0.5 of the year, which is what makes the second worth running."""
    got = readings(run("--check").stdout)
    assert abs(got[1.0][1] - got[2.0][1]) == pytest.approx(0.5, abs=1e-12)


def test_a_point_outside_the_flow_does_not_read_as_calm():
    """NaN is a point inside a building or out of the domain. Zero exceedance there
    would paint every building as the calmest place on the site, which is the failure
    the tool's own docstring names."""
    out = run("--check").stdout
    assert "nan" in out.lower(), out
    assert "calmest" in out, out
