"""`conjugate.py`: the reference for the one family with no coverage at all.

`advise.py` offers `chtMultiRegionFoam` and its steady twin, and this repository has no
case for either -- no template, no tutorial row, nothing graded. That made
`conjugate-heat-transfer` the only named family in `advise.FAMILIES` with nothing behind
it, which is what this file's subject exists to start closing.

The tests hold the two continuities a conjugate interface has to carry at once, the
limits where the answer is geometry alone, and the shell/slab factor -- which is graded
against its own algebra rather than against a printed number, because the first draft of
that docstring asserted "+3.5%" from memory and the real figure is +23.3%.
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
SCRIPT = TOOLBOX / "conjugate.py"


@pytest.fixture(scope="module")
def conjugate():
    spec = importlib.util.spec_from_file_location("toolbox_conjugate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_one_flux_through_every_layer(conjugate):
    """Flux continuity, which is half of what a conjugate interface owes. Restated as a
    check: the per-layer drops must sum to the total, which fails if a resistance is
    dropped or counted twice."""
    layers = conjugate.WALL
    q = conjugate.flux(layers, 400.0, 300.0)
    drops = [q * r for r in conjugate.resistances(layers)]
    assert abs(sum(drops) - 100.0) < 1e-9, drops


def test_the_faces_are_the_same_from_either_side(conjugate):
    """Temperature continuity, the other half. Walking the stack from the cold end must
    give the same interface temperatures; a solver satisfying one continuity and not the
    other gives a smooth plausible field with the wrong interface."""
    faces = conjugate.interface_temperatures(conjugate.WALL, 400.0, 300.0)
    backward = conjugate.interface_temperatures(tuple(reversed(conjugate.WALL)),
                                                300.0, 400.0)
    for a, b in zip(faces, reversed(backward)):
        assert abs(a - b) < 1e-9, (faces, backward)


def test_equal_conductivity_puts_the_interface_at_the_geometric_fraction(conjugate):
    """With one conductivity the answer is geometry alone -- no property enters it -- so
    this is the cleanest possible check on the resistance walk."""
    found = conjugate.interface_temperatures(((0.01, 5.0), (0.03, 5.0)), 400.0, 300.0)[1]
    assert abs(found - (400.0 - 100.0 * 0.25)) < 1e-9, found


def test_a_dominant_resistance_takes_the_interface_to_its_far_face(conjugate):
    """The limit a reader can check by eye, which is why it is worth having."""
    found = conjugate.interface_temperatures(((0.01, 1.0e6), (0.05, 0.05)),
                                             400.0, 300.0)[1]
    assert abs(found - 400.0) < 1e-3, found


def test_the_shell_is_not_a_slab_and_the_factor_is_derived(conjugate):
    """`ln(r2/r1)/(2 pi k)`, and the planar form on the inner area is wrong by exactly
    `(r2-r1)/r1/ln(r2/r1) - 1`. Graded against that algebra rather than against +23.3%,
    so the test cannot drift with the printed figure -- and the figure itself is why this
    is here: the docstring first said +3.5%, from memory, and `--check` caught it."""
    exact = conjugate.shell_resistance(0.020, 0.030, 16.0)
    inner = conjugate.planar_resistance_of_a_shell(0.020, 0.030, 16.0, "inner")
    derived = 0.5 / math.log(1.5) - 1.0
    assert abs((inner - exact) / exact - derived) < 1e-12
    assert derived > 0.2, "the trap is large, not a rounding"
    logmean = conjugate.planar_resistance_of_a_shell(0.020, 0.030, 16.0, "logmean")
    assert abs(logmean - exact) / exact < 1e-12, "the log-mean area must be exact"
    outer = conjugate.planar_resistance_of_a_shell(0.020, 0.030, 16.0, "outer")
    assert outer < exact, "the outer area errs the other way; neither is a safe default"


def test_impossible_geometry_and_properties_are_refused(conjugate):
    """A zero conductivity is an infinite resistance and an inverted shell is not a
    shell. Both return finite nonsense if computed, so both are refused."""
    with pytest.raises(ValueError):
        conjugate.flux(((0.01, 0.0),), 400.0, 300.0)
    with pytest.raises(ValueError):
        conjugate.shell_resistance(0.03, 0.02, 16.0)
    with pytest.raises(ValueError):
        conjugate.resistances(((-0.01, 16.0),))


def test_biot_says_whether_a_conjugate_case_is_needed_at_all(conjugate):
    """The judgement `advise.py` makes for every other family. A body with Bi << 1 is
    isothermal and one region with a surface condition answers the same question."""
    assert conjugate.biot(10.0, 0.001, 16.0) < 0.1
    assert conjugate.biot(500.0, 0.05, 0.6) > 1.0


def test_it_says_there_is_no_cht_case_in_this_repository(conjugate):
    """The claim that must not be made. This is the reference half; a template that
    writes a two-region case is the next item, and a reader has to be able to tell."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "chtMultiRegionFoam" in text
    # Lowercase both sides: the needle had capitals in it against a lowered
    # haystack, which is a test that can only fail.
    assert "no `chtmultiregionfoam` run exists here" in text.lower()


def test_check_exits_zero_and_says_it_agrees():
    done = subprocess.run([sys.executable, "-I", "-X", "utf8", str(SCRIPT), "--check"],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "agrees" in done.stdout
    assert "FAIL" not in done.stdout, done.stdout
