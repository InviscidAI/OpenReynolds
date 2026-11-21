"""`acoustics.py --tone`: the level it prints is the level the definition gives.

The tutorial index lists this case in tier 1, so it is a thing a reader runs on a laptop
with nothing installed, and these are the assertions that make the row evidence rather
than a claim. The reference is a closed form and needs no paper: a sine of amplitude `A`
has a mean square of exactly `A^2/2`, so `p_rms = A/sqrt(2)` and
`SPL = 20 log10(p_rms / 20e-6)`.

Before `--tone` existed, nothing in this repository asserted that the overall level this
tool prints equals the level that definition gives. The band assertion is independent of
it rather than a restatement: the overall level comes from the record's mean square, and
the band level comes from summing bins between two edges, so they agree only if the
density scaling and the band edges are both right.
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"
ACOUSTICS = TOOLBOX / "acoustics.py"
P_REF = 20e-6


def run(*args: str) -> subprocess.CompletedProcess[str]:
    """The script as a user runs it: a plain interpreter, no package import."""
    return subprocess.run([sys.executable, "-X", "utf8", str(ACOUSTICS), *args],
                          capture_output=True, text=True, timeout=300)


def exact_spl(amplitude: float = 1.0) -> float:
    return 20.0 * math.log10((amplitude / math.sqrt(2.0)) / P_REF)


def levels(text: str) -> dict[str, float]:
    """The numbers out of the printed report, by the label that introduces them."""
    found: dict[str, float] = {}
    for line in text.splitlines():
        bare = line.strip()
        if bare.startswith("SPL exact"):
            found["exact"] = float(bare.split()[2])
        elif bare.startswith("SPL measured"):
            found["measured"] = float(bare.split()[2])
        elif bare.startswith("loudest band"):
            parts = bare.split()
            found["centre"] = float(parts[2])
            found["band"] = float(parts[5])
    return found


def test_the_overall_level_is_the_closed_form():
    """20 log10((A/sqrt(2))/20e-6) with A = 1 Pa is 90.9691 dB, and that is arithmetic
    rather than a measurement, so the tool has to land on it."""
    done = run("--tone")
    assert done.returncode == 0, done.stdout + done.stderr
    got = levels(done.stdout)
    assert got, done.stdout
    assert got["exact"] == pytest.approx(90.9691, abs=5e-4)
    assert got["exact"] == pytest.approx(exact_spl(), abs=5e-4)
    # The tolerance the tool itself enforces is 0.05 dB; it measured 0.0000 when this
    # was written, so a bound of 0.01 dB here is still twenty times the error seen and
    # fails long before the tool's own gate would let something through.
    assert got["measured"] == pytest.approx(got["exact"], abs=0.01)


def test_the_whole_tone_lands_in_one_third_octave_band():
    """A tone at a band centre belongs to that band and no other, so the band level is
    the overall level. This is not the same statement as the test above: the overall
    comes from the record's mean square and the band from summing bins, so they agree
    only if the density scaling and the band edges are both right."""
    got = levels(run("--tone").stdout)
    assert got["centre"] == pytest.approx(1000.0, rel=0.02)
    assert got["band"] == pytest.approx(got["exact"], abs=0.01)


def test_the_band_centre_is_the_exact_one_not_the_nominal_label():
    """`--tone 250` lands in the band labelled 250 Hz, whose exact centre is
    `1000 * 10**(-6/10)` = 251.1886 Hz. Reporting 250 would be the nominal label
    standing in for the centre, which is the error this distinction exists to avoid."""
    got = levels(run("--tone", "250").stdout)
    assert got["centre"] == pytest.approx(1000.0 * 10 ** (-6 / 10), rel=1e-4)
    assert got["centre"] != pytest.approx(250.0, abs=0.5)
    assert got["band"] == pytest.approx(exact_spl(), abs=0.01)


def test_the_level_is_independent_of_the_tone_frequency():
    """The amplitude is 1 Pa whatever the frequency, so the level is the same number.
    A level that moved with frequency would mean the window or the scaling is leaking."""
    seen = [levels(run("--tone", hz).stdout)["measured"] for hz in ("125", "500", "2000")]
    for value in seen:
        assert value == pytest.approx(exact_spl(), abs=0.02), seen


def test_it_needs_no_case_and_no_file():
    """The tier 1 promise: nothing installed, no account, no case directory. `--tone`
    takes no source, and asking for one would make the tutorial row unrunnable."""
    done = run("--tone")
    assert done.returncode == 0
    assert "needed" not in done.stderr.lower(), done.stderr
    assert "agrees" in done.stdout
