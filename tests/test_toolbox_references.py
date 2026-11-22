"""The published reference data the toolbox ships, and the single source it must match.

Why this file exists. Asked to compare a lid-driven cavity against Ghia, Ghia & Shin
(1982), a study on 2026-11-22 (20261122-110157-2985) wrote the table into its own
analysis script from the model's memory, saying so in its reasoning: "I'm recalling the
standard Ghia et al. Re=1000 reference values". It had no alternative. The repository's
copy lives in `benchmarks/felix/grade.py`, which is not part of the toolbox the agent is
given, so there was nothing in `/work/.toolbox` to read.

That run's recalled table turned out to be correct, point for point to five decimals.
That is luck, not a property of the system, and the whole claim this project makes is
that a number was measured against a reference rather than remembered alongside one. So
the table ships in the toolbox, and this test keeps the shipped copy honest: it is
generated from `benchmarks/felix/grade.py` and must stay equal to it, so there is still
one source of truth and not two that can drift.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "openreynolds" / "toolbox" / "references" / "ghia_re1000.json"


def published():
    sys.path.insert(0, str(ROOT / "benchmarks" / "felix"))
    from grade import GHIA_RE1000_U, GHIA_RE1000_V  # noqa: E402
    return GHIA_RE1000_U, GHIA_RE1000_V


def test_the_shipped_table_is_the_repositorys_own_table():
    """Equal to `benchmarks/felix/grade.py`, which the graders already use. If this
    fails, one of the two was edited and the other was not."""
    shipped = json.loads(DATA.read_text(encoding="utf-8"))
    table_u, table_v = published()
    for key, source in (("table_1_u_on_vertical_centreline", table_u),
                        ("table_2_v_on_horizontal_centreline", table_v)):
        mine = [(round(a, 4), b) for a, b in shipped[key]]
        theirs = [(round(a, 4), b) for a, b in sorted(source)]
        assert mine == theirs, (key, mine[:3], theirs[:3])


def test_it_carries_its_citation_and_its_convention():
    """A table of numbers with no paper and no convention attached is not a reference:
    the ordinate direction is exactly what a reader would otherwise have to guess, and
    guessing it inverts the profile."""
    shipped = json.loads(DATA.read_text(encoding="utf-8"))
    assert "Ghia" in shipped["source"] and "1982" in shipped["source"]
    assert "48:387" in shipped["source"], "the volume and page make it findable"
    assert "lid" in shipped["note"]
    # The convention has to be checkable from the data itself, not only asserted.
    u = dict((round(a, 4), b) for a, b in shipped["table_1_u_on_vertical_centreline"])
    assert u[1.0] == 1.0, "u = 1 at the lid"
    assert u[0.0] == 0.0, "u = 0 at the stationary wall"


def test_every_point_is_inside_the_lid_speed():
    """Both profiles are normalised by the lid speed, so nothing may exceed it."""
    shipped = json.loads(DATA.read_text(encoding="utf-8"))
    for key in ("table_1_u_on_vertical_centreline", "table_2_v_on_horizontal_centreline"):
        for abscissa, value in shipped[key]:
            assert 0.0 <= abscissa <= 1.0, (key, abscissa)
            assert -1.0 <= value <= 1.0, (key, abscissa, value)


# -- the aerofoil closed forms -------------------------------------------------------
#
# Same reason as the Ghia table above, found the same way. Study 20261122-113750-c5a7
# reported that it had to write the thin-aerofoil integral itself because the only copy
# "is in a test file that isn't on this machine", and that it built its own elliptic
# wing because `--wing` has no elliptic planform. The toolbox now ships both.
#
# What these tests protect is the arrangement, not the physics. `tests/test_toolbox_panel.py`
# keeps its own independent implementation of these formulas and that is what grades
# `panel.py`; nothing here changes that, because a test importing the thing it checks is
# checking nothing. These hold the shipped copy equal to that independent one, so the
# reference a study reads and the reference the repository grades with cannot drift.

import importlib.util


def toolbox_copy():
    path = ROOT / "openreynolds" / "toolbox" / "references" / "thin_aerofoil.py"
    spec = importlib.util.spec_from_file_location("toolbox_thin_aerofoil", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def independent_copy():
    """The test file's own implementation, which is what actually grades `panel.py`."""
    sys.path.insert(0, str(ROOT / "tests"))
    import test_toolbox_panel  # noqa: E402
    return test_toolbox_panel


def test_the_shipped_integral_matches_the_one_that_grades_the_panel_method():
    mine, theirs = toolbox_copy(), independent_copy()
    for designation in ("0012", "2412", "4412", "6409", "1408", "9512"):
        a = mine.thin_aerofoil_zero_lift_deg(designation)
        b = theirs.thin_aerofoil_zero_lift_deg(designation)
        assert abs(a - b) < 1e-9, (designation, a, b)


def test_the_shipped_camber_slope_matches_too():
    """The integral is built on this, so agreeing on the integral alone could hide two
    errors that cancel across the integration."""
    import numpy as np
    mine, theirs = toolbox_copy(), independent_copy()
    x = np.linspace(0.0, 1.0, 1001)
    for designation in ("0012", "2412", "4412", "6409"):
        assert np.allclose(mine.naca4_camber_slope(designation, x),
                           theirs.naca4_camber_slope(designation, x), atol=1e-12), designation


def test_the_shipped_elliptic_wing_matches():
    mine, theirs = toolbox_copy(), independent_copy()
    for slope, alpha, ar in ((2 * 3.141592653589793, 5.0, 8.0), (6.0, 2.5, 4.0), (5.5, 10.0, 12.0)):
        a = mine.elliptic_wing_cl(slope, alpha, ar)
        b = theirs.elliptic_wing_cl(slope, alpha, ar)
        assert abs(a - b) < 1e-12, (slope, alpha, ar, a, b)


def test_a_symmetric_section_has_no_zero_lift_angle_and_says_why():
    """Zero is forced by symmetry rather than earned, which the docstring has to say,
    because a study that counts it as a successful accuracy check has miscounted."""
    mine = toolbox_copy()
    assert abs(mine.thin_aerofoil_zero_lift_deg("0012")) < 1e-12
    assert "sign conventions" in mine.naca4_camber_slope.__doc__


def test_the_elliptic_result_warns_that_it_cannot_fail():
    """The lifting-line series has only its first mode for an elliptic planform, so a
    lifting-line solver reproduces this identically whatever else it does."""
    mine = toolbox_copy()
    doc = mine.elliptic_wing_cl.__doc__
    assert "by construction" in doc and "does not say" in doc
