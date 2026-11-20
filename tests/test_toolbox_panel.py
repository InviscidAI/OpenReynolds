"""`panel.py` against results that exist outside this repository.

A panel method is arithmetic whose answer is already known, which is the whole reason it
is worth having: thin-aerofoil theory, the NACA 4-digit camber-line integral, and
Prandtl's lifting line all give closed forms this has to reproduce, and published section
data gives numbers it has to land near. So nothing here checks the script against itself.
Where a value is quoted, its source is named in the test that quotes it, and where a
tolerance is wide the reason it is wide is stated rather than left as a free parameter.

The two independent calculations this file does for itself -- the thin-aerofoil zero-lift
integral and the elliptic-wing closed form -- are written out from their published
definitions below rather than imported from the script, because a test that imports the
thing it is checking is checking nothing.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load(name: str):
    """Import a toolbox script by path; the directory is data, not a package."""
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def panel():
    return load("panel")


def section(panel, designation: str, panels: int = 200):
    nodes, meta = panel.naca4(designation, panels)
    return panel.panelise(nodes), meta


# -- the calculations this file does for itself --------------------------------


def naca4_camber_slope(designation: str, x: np.ndarray) -> np.ndarray:
    """dyc/dx of a NACA 4-digit mean line, from the published family definition.

    Source: Jacobs, Ward and Pinkerton, "The characteristics of 78 related airfoil
    sections from tests in the variable-density wind tunnel", NACA Report No. 460
    (1933), which defines the 4-digit mean line as the two parabolic arcs below; the
    same definition is in Abbott and von Doenhoff, "Theory of Wing Sections" (Dover,
    1959), section 6.4.

    Written out here rather than taken from `panel.py` on purpose: this is the
    independent half of the zero-lift-angle check.
    """
    m = int(designation[0]) / 100.0
    p = int(designation[1]) / 10.0
    if m == 0.0 or p == 0.0:
        return np.zeros_like(x)
    return np.where(x <= p,
                    2.0 * m / p**2 * (p - x),
                    2.0 * m / (1.0 - p) ** 2 * (p - x))


def trapezium(y: np.ndarray, x: np.ndarray) -> float:
    """The trapezium rule, by hand.

    `numpy.trapezoid` only exists from numpy 2.0 and `numpy.trapz` is removed in it; this
    package asks for numpy>=1.24, so a test that named either would pass on some of the
    matrix and not the rest. One line is cheaper than a version check.
    """
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def thin_aerofoil_zero_lift_deg(designation: str, points: int = 200001) -> float:
    """The thin-aerofoil zero-lift angle of a camber line, in degrees.

        alpha_L0 = -(1/pi) * integral_0^pi (dyc/dx) * (cos(theta) - 1) dtheta,
        x = (1 - cos(theta)) / 2

    Source: Anderson, "Fundamentals of Aerodynamics", 6th ed., thin airfoil theory for
    the cambered aerofoil (the alpha_L0 result that follows from the Fourier solution of
    the camber-line integral equation); the same expression is Glauert, "The Elements of
    Aerofoil and Airscrew Theory" (1926), chapter VII. Evaluated by the trapezium rule
    on a fine grid, with the endpoints nudged off the singular limits of theta.
    """
    theta = np.linspace(1e-9, np.pi - 1e-9, points)
    x = 0.5 * (1.0 - np.cos(theta))
    slope = naca4_camber_slope(designation, x)
    return float(np.degrees(-trapezium(slope * (np.cos(theta) - 1.0), theta) / np.pi))


def elliptic_wing_cl(slope_per_rad: float, alpha_abs_deg: float, aspect_ratio: float) -> float:
    """The closed-form lift of an untwisted elliptic wing.

        CL = a0 * alpha / (1 + a0 / (pi * AR))

    Source: Prandtl, "Applications of modern hydrodynamics to aeronautics", NACA Report
    No. 116 (1921); the standard textbook form is Anderson, "Fundamentals of
    Aerodynamics", the elliptic-lift-distribution result of finite-wing theory.
    """
    return slope_per_rad * np.radians(alpha_abs_deg) / (1.0 + slope_per_rad
                                                        / (np.pi * aspect_ratio))


# -- thin-aerofoil theory ------------------------------------------------------


def test_a_symmetric_sections_lift_slope_approaches_two_pi(panel):
    """Thin-aerofoil theory: dCl/dalpha = 2*pi per radian for any thin section.

    Source: Anderson, "Fundamentals of Aerodynamics", 6th ed., thin airfoil theory for
    the symmetric aerofoil -- Cl = 2*pi*alpha, so the slope is 6.28319 per radian.
    Glauert (1926) chapter VII is the same result.

    The theory is for a section of zero thickness, and thickness raises the slope, so the
    comparison is made on a thin section and the thick one is only required to sit above
    the theory rather than on it. How far above is the published correlation
    2*pi*(1 + 0.77*t/c), which gets its own test below.
    """
    two_pi = 2.0 * np.pi

    thin, _ = section(panel, "0006", 300)
    thin_slope = panel.lift_slope(thin)["per_radian"]
    assert thin_slope == pytest.approx(two_pi, rel=0.06), (
        f"a 6% thick symmetric section came out at {thin_slope:.4f}/rad against 2*pi = "
        f"{two_pi:.4f}; measured 6.599/rad, which is 1.050 x 2*pi"
    )

    thick, _ = section(panel, "0012", 300)
    thick_slope = panel.lift_slope(thick)["per_radian"]
    assert two_pi < thick_slope < 1.15 * two_pi, (
        f"a 12% section came out at {thick_slope:.4f}/rad; thickness raises the slope "
        f"above 2*pi and measured 6.913/rad, which is 1.100 x 2*pi"
    )
    assert thick_slope > thin_slope, "the thicker section should carry the steeper slope"


def test_a_symmetric_sections_quarter_chord_moment_is_about_zero(panel):
    """Thin-aerofoil theory: Cm about the quarter chord is zero for a symmetric section
    at every incidence, because the quarter chord is the aerodynamic centre and there is
    no camber to give a moment at zero lift.

    Source: Anderson, "Fundamentals of Aerodynamics", 6th ed. -- for a symmetric
    aerofoil Cm_{c/4} = 0 and the aerodynamic centre is at the quarter chord.

    Two separate claims, with very different tolerances, and the difference is the point.
    At zero incidence it is exact and holds to roundoff: the section is symmetric, so the
    pressure field is and the moment is identically zero. Away from zero it is only
    approximately zero, because thickness moves the aerodynamic centre aft of the quarter
    chord -- measured here at 0.261c for the 12% section and 0.254c for the 6% one, which
    is the residual below and is physics rather than discretisation error (it does not
    shrink with panel count; it converges to -0.0067 at 640 panels).
    """
    thick, _ = section(panel, "0012", 300)

    at_zero = panel.solve(thick, 0.0)
    assert abs(at_zero["cm"]) < 1e-9, (
        f"a symmetric section at zero incidence gave Cm {at_zero['cm']:.3e}, and symmetry "
        "makes that exactly zero"
    )
    assert abs(at_zero["cl"]) < 1e-9

    at_five = panel.solve(thick, 5.0)
    assert abs(at_five["cm"]) < 0.01, (
        f"Cm about the quarter chord came out {at_five['cm']:+.5f} at 5 degrees against "
        "the theoretical zero; 0.01 is about 1% of this section's Cl there and covers the "
        "thickness shift of the aerodynamic centre"
    )
    # The residual is an aerodynamic-centre offset and not an offset in Cm, so it scales
    # with Cl. That is what distinguishes the two.
    at_eight = panel.solve(thick, 8.0)
    ratio = at_eight["cm"] / at_five["cm"]
    assert ratio == pytest.approx(at_eight["cl"] / at_five["cl"], rel=0.02)


def test_a_cambered_sections_zero_lift_angle_matches_the_thin_aerofoil_integral(panel):
    """The zero-lift angle of a cambered NACA 4-digit section, against the thin-aerofoil
    integral over that same camber line.

    The integral is computed in this file from the NACA Report 460 mean-line definition
    and the Glauert/Anderson alpha_L0 expression, so the two sides are independent.

    As a third opinion, the figures quoted from section data for these two aerofoils are
    about -2.1 degrees for the NACA 2412 and -4.1 degrees for the NACA 4412 (the Abbott
    and von Doenhoff section plots are where those come from). They are a quarter of a
    degree of the integral, which is the agreement that matters -- the integral is the
    number this test is really against, because this file computes it.

    The tolerance is 5%: thin-aerofoil theory ignores thickness, which moves the zero-lift
    angle slightly further negative on a 12% section, and the panel answers below sit
    about 3% beyond the theory for exactly that reason. The 2% thick sections are held to
    a tighter 3.5%, which is the whole point -- the gap closes as the section thins, so it
    is a thickness effect and not an error.
    """
    for designation in ("2412", "4412"):
        geom, _ = section(panel, designation, 400)
        measured = panel.lift_slope(geom)["zero_lift_alpha_deg"]
        theory = thin_aerofoil_zero_lift_deg(designation)
        assert measured == pytest.approx(theory, rel=0.05), (
            f"NACA {designation}: panel method {measured:+.4f} deg against the "
            f"thin-aerofoil integral {theory:+.4f} deg"
        )
        assert measured < 0.0, "positive camber gives a negative zero-lift angle"

    # The quoted section-data figures, to a quarter of a degree.
    published = {"2412": -2.1, "4412": -4.1}
    for designation, expected in published.items():
        geom, _ = section(panel, designation, 400)
        measured = panel.lift_slope(geom)["zero_lift_alpha_deg"]
        assert measured == pytest.approx(expected, abs=0.25), (
            f"NACA {designation}: {measured:+.4f} deg against the published "
            f"{expected:+.1f} deg"
        )

    # Thin the section and the gap to the theory closes, which is what makes the 5% above
    # a thickness effect rather than slack.
    for designation in ("2402", "4402"):
        geom, _ = section(panel, designation, 400)
        measured = panel.lift_slope(geom)["zero_lift_alpha_deg"]
        theory = thin_aerofoil_zero_lift_deg(designation)
        assert measured == pytest.approx(theory, rel=0.035), (
            f"NACA {designation} at 2% thickness: {measured:+.4f} against {theory:+.4f}"
        )


def test_the_cambered_quarter_chord_moment_matches_the_thin_aerofoil_integral(panel):
    """Cm about the quarter chord, against the thin-aerofoil Fourier result.

        Cm_{c/4} = (pi/4) * (A2 - A1),  An = (2/pi) * integral_0^pi (dyc/dx) cos(n*theta) dtheta

    Source: Glauert, "The Elements of Aerofoil and Airscrew Theory" (1926), chapter VII;
    the same expression is in Anderson, "Fundamentals of Aerodynamics", 6th ed. as the
    cambered thin aerofoil's moment about the quarter chord. Computed here from the camber
    line in this file, so it is independent of the script.

    Checked on the 2% thick sections, where thin-aerofoil theory is close to exact; the
    12% sections sit about 15% off it and so are checked only for sign and order, because
    at that thickness the theory is the thing that is wrong. Abbott and von Doenhoff give
    the NACA 2412's measured moment coefficient as about -0.047, with the boundary layer
    in it, against the -0.053 the camber line alone gives.
    """
    def thin_cm(designation: str, points: int = 200001) -> float:
        theta = np.linspace(1e-9, np.pi - 1e-9, points)
        x = 0.5 * (1.0 - np.cos(theta))
        slope = naca4_camber_slope(designation, x)
        a1 = 2.0 / np.pi * trapezium(slope * np.cos(theta), theta)
        a2 = 2.0 / np.pi * trapezium(slope * np.cos(2.0 * theta), theta)
        return float(np.pi / 4.0 * (a2 - a1))

    for designation in ("2402", "4402"):
        geom, _ = section(panel, designation, 400)
        measured = panel.solve(geom, 0.0)["cm"]
        assert measured == pytest.approx(thin_cm(designation), rel=0.05), (
            f"NACA {designation}: Cm_c/4 {measured:+.5f} against the thin-aerofoil "
            f"integral {thin_cm(designation):+.5f}"
        )

    geom, _ = section(panel, "2412", 400)
    measured = panel.solve(geom, 0.0)["cm"]
    assert -0.08 < measured < -0.03, (
        f"NACA 2412 Cm_c/4 came out {measured:+.5f}; thin-aerofoil theory gives "
        f"{thin_cm('2412'):+.5f} and Abbott and von Doenhoff measured -0.047"
    )


def test_naca_0012_lift_at_five_degrees_against_published_inviscid_values(panel):
    """NACA 0012 Cl at 5 degrees incidence, against three published numbers.

    The primary inviscid datum is a published thickness correction to the thin-aerofoil
    slope, a0 = 2*pi*(1 + 0.77*t/c):

      D. Peri, "BBun: an unsteady vortex-ring panel method for interacting lifting
      bodies, box wings and propellers", arXiv:2610.20629, section 5.3, which reports
      that an independent two-dimensional linear-vortex panel method gives the NACA 0015
      section with a closed trailing edge an inviscid lift slope of a0 = 7.05 /rad,
      converged in the panel count and in agreement with that correlation.

    At t/c = 0.12 the correlation gives 2*pi*1.0924 = 6.8638 /rad, so Cl(5 deg) = 0.599.
    This file checks both ends of it: the NACA 0015 against the quoted 7.05 /rad directly,
    and the NACA 0012 against the correlation.

    The second datum is measurement, which bounds the inviscid answer from below:

      NACA 0012 at M = 0.15, Re = 6e6, Cl = 0.536 at 5 degrees with a lift-curve slope of
      0.107 per degree -- C. L. Ladson, "Effects of Independent Variation of Mach and
      Reynolds Numbers on the Low-Speed Aerodynamic Characteristics of the NACA 0012
      Airfoil Section", NASA TM 4074 (1988), as tabulated by the NASA Langley Turbulence
      Modeling Resource; the same table to three figures (Cl = 0.5365 at 5 degrees,
      slope 0.107 /deg) is worked example 9 of J. G. Leishman, "Introduction to Aerospace
      Flight Vehicles" (Embry-Riddle, Pressbooks).

    The third is thin-aerofoil theory itself, 2*pi /rad = 0.5483 at 5 degrees (Anderson,
    "Fundamentals of Aerodynamics", 6th ed.).

    The ordering is the physics and is asserted as such. The inviscid answer must sit
    above thin-aerofoil theory, because thickness raises the slope, and above the
    measurement, because the boundary layer's displacement thickness decambers the
    section and costs real lift -- about 11% of it here, which is the usual size at this
    Reynolds number. Measured in this script: 0.6028 at 160 panels, 0.6% above the 0.599
    the published correlation predicts.
    """
    geom, _ = section(panel, "0012", 160)
    result = panel.solve(geom, 5.0)

    thin_aerofoil = 2.0 * np.pi * np.radians(5.0)
    assert thin_aerofoil == pytest.approx(0.5483, abs=0.001)
    published_correlation = 2.0 * np.pi * (1.0 + 0.77 * 0.12)
    assert published_correlation == pytest.approx(6.8638, abs=0.001)
    measured_in_tunnel = 0.536

    assert result["cl"] == pytest.approx(published_correlation * np.radians(5.0), rel=0.02), (
        f"Cl came out {result['cl']:.4f} against the {published_correlation * np.radians(5.0):.4f} "
        f"that 2*pi*(1 + 0.77*t/c) predicts for a 12% section"
    )
    assert result["cl"] > thin_aerofoil, (
        f"Cl {result['cl']:.4f} must sit above the thin-aerofoil {thin_aerofoil:.4f}: "
        "thickness raises the lift slope"
    )
    assert result["cl"] > measured_in_tunnel, (
        f"Cl {result['cl']:.4f} must sit above the measured {measured_in_tunnel:.4f}: "
        "an inviscid calculation does not pay the boundary layer's decambering"
    )
    assert result["cl"] / measured_in_tunnel == pytest.approx(1.12, abs=0.04), (
        "the viscous loss at Re 6e6 is about a tenth of the lift, not a half and not none"
    )


def test_the_lift_slope_follows_the_published_thickness_correction(panel):
    """The NACA 0015 inviscid lift slope against a published panel-method value, and the
    whole 4-digit thickness family against the correlation that value was checked with.

    Source: D. Peri, "BBun: an unsteady vortex-ring panel method for interacting lifting
    bodies, box wings and propellers", arXiv:2610.20629, section 5.3 -- an independent
    two-dimensional linear-vortex panel method gives the NACA 0015 with a closed trailing
    edge a0 = 7.05 /rad, converged in the panel count and agreeing with
    2*pi*(1 + 0.77*t/c).

    This is the strongest external check in the file, because it is a different method
    (linear vortex panels against the constant-source-plus-vortex scheme here) applied
    to the same section by someone else. Measured here: 7.076 /rad for the NACA 0015,
    0.4% off the quoted 7.05. Across t/c from 6% to 18% this script sits 0.4% to 1.2%
    above the correlation, rising with thickness, which is the direction a second-order
    correlation is expected to fall behind a full panel solution.
    """
    fifteen, _ = section(panel, "0015", 400)
    assert panel.lift_slope(fifteen)["per_radian"] == pytest.approx(7.05, rel=0.01), (
        "the NACA 0015 inviscid lift slope is published as 7.05 /rad"
    )

    for designation in ("0006", "0009", "0012", "0015", "0018"):
        thickness = int(designation[2:]) / 100.0
        geom, _ = section(panel, designation, 400)
        measured = panel.lift_slope(geom)["per_radian"]
        correlation = 2.0 * np.pi * (1.0 + 0.77 * thickness)
        assert measured == pytest.approx(correlation, rel=0.02), (
            f"NACA {designation}: {measured:.4f} /rad against the correlation's "
            f"{correlation:.4f} /rad"
        )
        assert measured > correlation, "the full solution sits above the correlation"


def test_no_pressure_drag_which_is_what_dalembert_requires(panel):
    """d'Alembert's paradox: a closed body in steady, inviscid, irrotational flow carries
    no drag at all. So the pressure drag that comes out of the panel integration is pure
    discretisation error, and its size is the honest reading of a panel count.

    Source: the result is d'Alembert (1752); the statement used here is in Batchelor,
    "An Introduction to Fluid Dynamics", section 6.4, and in every potential-flow chapter
    since.
    """
    for designation in ("0012", "2412", "4415"):
        geom, _ = section(panel, designation, 240)
        for alpha in (0.0, 4.0, 8.0):
            result = panel.solve(geom, alpha)
            assert abs(result["cd_pressure"]) < 2e-3, (
                f"NACA {designation} at {alpha} deg has pressure drag "
                f"{result['cd_pressure']:.3e}, and d'Alembert says zero"
            )


def test_the_two_routes_to_lift_agree(panel):
    """Cl from integrating the pressure and Cl from Kutta-Joukowski on the total
    circulation are the same number by theory, and the script reports both.

    Source: Kutta-Joukowski, L = rho * V * Gamma, as in Anderson, "Fundamentals of
    Aerodynamics", 6th ed. They are not two ways of writing one line of code -- one walks
    the panel pressures and the other is the vortex strength times the perimeter -- so
    their agreeing is a check on the linear solve rather than on the arithmetic after it.
    """
    geom, _ = section(panel, "2412", 320)
    for alpha in (-2.0, 0.0, 6.0):
        result = panel.solve(geom, alpha)
        assert result["cl"] == pytest.approx(result["cl_from_circulation"], abs=5e-3), (
            f"at {alpha} deg: pressure gives {result['cl']:.5f}, circulation gives "
            f"{result['cl_from_circulation']:.5f}"
        )


# -- convergence ---------------------------------------------------------------


def test_the_answer_settles_as_the_panel_count_rises(panel):
    """The discretisation has to converge, and the count at which it is good enough has
    to be stated rather than assumed.

    Measured, NACA 0012 at 5 degrees, against a 1280-panel reference of Cl = 0.6029562:
    Cl = 0.600906 at 40 panels, 0.601918 at 80, 0.602487 at 160, 0.602763 at 320,
    0.602894 at 640 -- errors of 2.05e-3, 1.04e-3, 4.69e-4, 1.93e-4, 6.20e-5, each
    doubling taking roughly half of what is left, so first order in the panel size, which
    is what constant-strength panels give.

    The stated count: **160 panels puts Cl within 0.08% of the 1280-panel value**, and
    that is why 160 is the script's default. 40 panels is already within 0.34%, which is
    worth knowing -- the lift coefficient is a cheap quantity. The pressure drag is the
    expensive one and is still falling at 640, which is why it is reported as the error
    measure it is rather than as a drag.

    The 640-panel row is left out of the ratio check: it is only one doubling from the
    reference, so its error is contaminated by the reference's own and the ratio comes out
    at 3.11 rather than near 2. Comparing a convergence study against its own finest grid
    always does this at the fine end, and pretending otherwise is how a tolerance gets
    quietly widened.
    """
    reference, _ = section(panel, "0012", 1280)
    settled = panel.solve(reference, 5.0)["cl"]
    assert settled == pytest.approx(0.6029562, abs=1e-6)

    history = {}
    for count in (40, 80, 160, 320, 640):
        geom, _ = section(panel, "0012", count)
        history[geom["n"]] = panel.solve(geom, 5.0)["cl"]

    counts = sorted(history)
    errors = [abs(history[c] - settled) for c in counts]
    assert errors == sorted(errors, reverse=True), (
        f"the error must fall monotonically with the panel count; got {errors}"
    )
    at_160 = next(history[c] for c in counts if c >= 160)
    assert abs(at_160 - settled) / abs(settled) < 8e-4, (
        f"160 panels gave {at_160:.6f} against {settled:.6f} at 1280, which is "
        f"{abs(at_160 - settled) / abs(settled):.2e} and over the stated 0.08%"
    )
    # First order: each doubling should roughly halve the error.
    clean = [c for c in counts if c <= 320]
    for lower, higher in zip(clean, clean[1:]):
        shrink = abs(history[lower] - settled) / abs(history[higher] - settled)
        assert 1.8 < shrink < 2.6, f"{lower} -> {higher} panels shrank the error by {shrink:.2f}"

    # Pressure drag, the quantity that is actually still converging.
    coarse, _ = section(panel, "0012", 40)
    assert abs(panel.solve(coarse, 5.0)["cd_pressure"]) > abs(
        panel.solve(reference, 5.0)["cd_pressure"]) * 4


def test_the_sweep_lift_curve_rises_monotonically_in_the_linear_range(panel):
    """A lift curve with no viscosity in it has no top, so inside the range this claims
    to cover it can only rise. A curve that turned over would mean the Kutta condition or
    the linear solve had gone wrong at incidence, not that the section had stalled."""
    geom, _ = section(panel, "2412", 200)
    alphas = np.arange(-8.0, 12.1, 1.0)
    results = panel.sweep(geom, alphas)

    cls = np.array([r["cl"] for r in results])
    assert np.all(np.diff(cls) > 0), f"the lift curve turns over: {cls}"

    # And it is close to straight: the solution is linear in the free-stream components,
    # so Cl goes as sin(alpha - alpha_L0) and the second difference is small but not zero.
    curvature = np.abs(np.diff(cls, 2)).max()
    assert curvature < 2e-3, f"largest second difference {curvature:.2e}"
    assert cls[0] < 0 < cls[-1], "the curve should cross zero inside this range"


def test_the_stagnation_point_sits_at_the_nose_at_zero_incidence(panel):
    """A symmetric section at zero incidence stagnates at the leading edge, exactly, by
    symmetry; at incidence the stagnation point moves aft along the lower surface, and
    how far depends on the nose radius. Both are reported, and the second is the one that
    would quietly come out on the wrong surface if the contour winding were mishandled.
    """
    geom, _ = section(panel, "0012", 300)

    at_zero = panel.solve(geom, 0.0)["stagnation"]
    assert at_zero is not None
    assert at_zero["x_over_chord"] < 1e-3, (
        f"a symmetric section at zero incidence stagnated at x/c "
        f"{at_zero['x_over_chord']:.5f}"
    )
    assert at_zero["at_the_nose"]

    previous = at_zero["x_over_chord"]
    for alpha in (2.0, 5.0, 10.0):
        moved = panel.solve(geom, alpha)["stagnation"]
        assert moved["surface"] == "lower", (
            f"at +{alpha} deg the stagnation point must be on the lower surface, got "
            f"{moved['surface']}"
        )
        assert moved["x_over_chord"] > previous, "it should keep moving aft with incidence"
        previous = moved["x_over_chord"]
    # NACA 0012's leading-edge radius is 1.58% of chord, so at ten degrees the stagnation
    # point is still inside the nose region and not somewhere down the chord.
    assert previous < 0.05


# -- coordinate files ----------------------------------------------------------


def as_selig(nodes: np.ndarray, name: str = "TEST SECTION") -> str:
    """One continuous list, trailing edge over the upper surface and back along the lower.
    This is the layout the UIUC Airfoil Coordinates Database calls the Selig format.

    Nine decimals rather than the six a real file carries, so that what these tests
    compare is the reassembly of the coordinates and not the rounding of them -- six
    decimals moves Cl in the fifth place, which would make an exact comparison meaningless
    and a loose one uninformative.
    """
    lines = [name]
    lines += [f"  {x:.9f}  {y:.9f}" for x, y in nodes]
    return "\n".join(lines) + "\n"


def as_lednicer(nodes: np.ndarray, name: str = "TEST SECTION") -> str:
    """Two nose-to-tail blocks under a header giving the two point counts, which is the
    layout the UIUC database calls the Lednicer format."""
    nose = int(np.argmin(nodes[:, 0]))
    upper = nodes[: nose + 1][::-1]      # nose to tail
    lower = nodes[nose:]                 # nose to tail already
    lines = [name, f"       {len(upper)}.      {len(lower)}."]
    lines.append("")
    lines += [f"  {x:.9f}  {y:.9f}" for x, y in upper]
    lines.append("")
    lines += [f"  {x:.9f}  {y:.9f}" for x, y in lower]
    return "\n".join(lines) + "\n"


def test_a_selig_file_is_read_as_a_selig_file(panel, tmp_path):
    """The layout is a guess from the numbers, so the guess is reported and this checks
    the report as well as the answer: reading a Selig list as Lednicer would give a
    contour that crosses itself and a Cl with nothing attached to say so."""
    nodes, _ = panel.naca4("2412", 160)
    path = tmp_path / "selig.dat"
    path.write_text(as_selig(nodes))

    read, meta = panel.read_dat(path)
    assert meta["layout"].startswith("Selig"), meta["layout"]
    assert "TEST SECTION" in meta["source"]
    assert read == pytest.approx(nodes, abs=1e-6)

    geom = panel.panelise(read)
    from_file = panel.solve(geom, 5.0)["cl"]
    from_scratch = panel.solve(panel.panelise(nodes), 5.0)["cl"]
    assert from_file == pytest.approx(from_scratch, abs=1e-6)


def test_a_title_line_carrying_numbers_is_a_title_line_and_not_a_point(panel, tmp_path):
    """#140. Aerofoil files are named with numbers in them -- `E63  (4.25%)`,
    `S1223 12%`, `NACA 63-412` -- and the reader kept any line that yielded two
    numbers. `E63  (4.25%)` yields 63 and 4.25, both above 1.5, which is the test for
    a Lednicer point-count header, so a plain Selig list was split, reversed and
    "closed by hand" across a gap it did not have.

    Measured on `data/qa-runs/geometry/src/airfoil_e63.dat`, the real file this was
    found on: Cl at 5 degrees came out +1.4420 against the +1.3951 the same file gives
    with its title line deleted, 3.4% high, under a confident statement that the file
    was Lednicer. The rule is that a coordinate line is numbers and separators and
    nothing else, and a genuine Lednicer header is two bare numbers on a line of its
    own, so this and the test below both have to hold.
    """
    nodes, _ = panel.naca4("2412", 160)
    titled = tmp_path / "titled.dat"
    titled.write_text(as_selig(nodes, name="E63  (4.25%)"))

    read, meta = panel.read_dat(titled)
    assert meta["layout"].startswith("Selig"), meta["layout"]
    assert "E63" in meta["source"], "the title line is the name, and is kept as one"
    assert read == pytest.approx(nodes, abs=1e-6)
    assert not meta["notes"], (
        f"a file that closes has no trailing-edge note to make: {meta['notes']}"
    )

    # The same section with no title at all has to give the same contour, because the
    # title was never data. Identical to the last digit, not merely close.
    plain = tmp_path / "plain.dat"
    plain.write_text("\n".join(as_selig(nodes).splitlines()[1:]) + "\n")
    assert panel.solve(panel.panelise(read), 5.0)["cl"] == pytest.approx(
        panel.solve(panel.panelise(panel.read_dat(plain)[0]), 5.0)["cl"], abs=1e-12)


def test_a_coordinate_file_that_is_not_there_is_refused_rather_than_raised(panel, tmp_path):
    """Every other bad input to this script takes the `Unusable` path and exits 0 with
    the file named. A missing `--dat` raised `FileNotFoundError` out of `read_text` and
    printed a traceback, which is the one failure mode the refusal path exists to
    avoid. The guard is in `read_dat`, where every caller goes through."""
    with pytest.raises(panel.Unusable, match="cannot be read"):
        panel.read_dat(tmp_path / "nowhere" / "absent.dat")
    assert panel.main(["--dat", str(tmp_path / "nowhere" / "absent.dat")]) == 0


def test_a_lednicer_file_is_read_as_a_lednicer_file(panel, tmp_path):
    """The same aerofoil in the other layout is the same aerofoil, so it has to give the
    same lift to the last digit -- which only happens if the two blocks were reassembled
    in the right order and the shared nose point was not doubled into a zero-length
    panel."""
    nodes, _ = panel.naca4("2412", 160)
    path = tmp_path / "lednicer.dat"
    path.write_text(as_lednicer(nodes))

    read, meta = panel.read_dat(path)
    assert meta["layout"].startswith("Lednicer"), meta["layout"]
    assert len(read) == len(nodes), "the nose point should appear once, not twice"

    geom = panel.panelise(read)
    from_file = panel.solve(geom, 5.0)["cl"]
    from_scratch = panel.solve(panel.panelise(nodes), 5.0)["cl"]
    assert from_file == pytest.approx(from_scratch, abs=1e-6), (
        f"Lednicer {from_file:.6f} against Selig {from_scratch:.6f}"
    )


def test_a_lednicer_file_without_the_count_header_is_still_read(panel, tmp_path):
    """Plenty of files in the wild carry the two blocks and no header line. The fallback
    is the shape of the x column -- a list that starts at the nose and turns back on
    itself is two blocks -- and it is reported as the fallback it is."""
    nodes, _ = panel.naca4("0012", 160)
    text = as_lednicer(nodes).splitlines()
    del text[1]
    path = tmp_path / "headerless.dat"
    path.write_text("\n".join(text) + "\n")

    read, meta = panel.read_dat(path)
    assert "Lednicer" in meta["layout"] and "no count header" in meta["layout"]
    assert panel.solve(panel.panelise(read), 5.0)["cl"] == pytest.approx(
        panel.solve(panel.panelise(nodes), 5.0)["cl"], abs=1e-6)


def test_a_contour_that_does_not_close_is_refused_by_name(panel, tmp_path):
    """A truncated file, or one read in the wrong layout, arrives as a contour whose ends
    do not meet. Closing it quietly would turn a broken file into a plausible Cl, which is
    the failure mode worth spending a refusal on -- so the refusal names the file, both
    end points, the gap, and the tolerance it exceeded.
    """
    nodes, _ = panel.naca4("2412", 160)
    truncated = nodes[: int(0.8 * len(nodes))]
    path = tmp_path / "truncated.dat"
    path.write_text(as_selig(truncated))

    with pytest.raises(panel.Unusable) as refusal:
        panel.read_dat(path)
    message = str(refusal.value)
    assert "truncated.dat" in message, f"the refusal should name the file: {message}"
    assert "does not close" in message
    assert "chord" in message

    # A real finite trailing edge is a small gap, and that is closed and reported rather
    # than refused -- the distinction is the tolerance, which is on the command line.
    finite_te = nodes.copy()
    finite_te[0] = [1.0, 0.0015]
    finite_te[-1] = [1.0, -0.0015]
    gapped = tmp_path / "finite_te.dat"
    gapped.write_text(as_selig(finite_te))
    read, meta = panel.read_dat(gapped)
    assert np.allclose(read[0], read[-1]), "a small gap should be closed"
    assert any("trailing edge closed" in note for note in meta["notes"]), meta["notes"]
    assert "0.003" in " ".join(meta["notes"])

    # And the tolerance is what decides, so tightening it below the real gap refuses it.
    with pytest.raises(panel.Unusable, match="does not close"):
        panel.read_dat(gapped, close_tol=1e-6)


def test_a_degenerate_contour_is_refused_rather_than_solved(panel):
    """Two coincident nodes make a zero-length panel, whose influence coefficients are
    infinite and whose solve returns numbers. Refused with the node indices."""
    nodes, _ = panel.naca4("0012", 100)
    doubled = np.insert(nodes, 30, nodes[30], axis=0)
    with pytest.raises(panel.Unusable, match="no length"):
        panel.panelise(doubled)

    with pytest.raises(panel.Unusable, match="four digits"):
        panel.naca4("241")
    with pytest.raises(panel.Unusable, match="too few"):
        panel.naca4("2412", 8)


def test_the_winding_of_the_contour_does_not_change_the_answer(panel):
    """Selig order is anticlockwise and the method wants clockwise, so most files arrive
    reversed. A file that happens to be wound the other way has to give the same lift --
    if the reversal were missed, every normal would point inward and the sign of the lift
    would follow."""
    nodes, _ = panel.naca4("2412", 200)
    forward = panel.panelise(nodes)
    backward = panel.panelise(nodes[::-1])
    assert forward["reversed"] != backward["reversed"]
    assert panel.solve(forward, 5.0)["cl"] == pytest.approx(
        panel.solve(backward, 5.0)["cl"], abs=1e-9)
    assert panel.solve(forward, 5.0)["cl"] > 0


# -- the wing ------------------------------------------------------------------


def test_the_lifting_line_reduces_to_the_two_d_answer_as_aspect_ratio_grows(panel):
    """An infinite span has no tip, so no downwash, so the section's own lift. The
    approach has to be from below and has to be monotone, because downwash can only
    reduce lift.

    Source: Prandtl, NACA Report No. 116 (1921) -- the finite-wing lift is the section
    lift divided by (1 + a0/(pi*AR)) for elliptic loading, which tends to the section
    value as AR grows.

    Measured, a rectangular wing at 5 degrees with a 2*pi section: Cl = 0.3515 at AR 4,
    0.4222 at AR 8, 0.4719 at AR 16, 0.5172 at AR 50, 0.5459 at AR 1000, against the 2D
    0.5483 -- within 0.5% by AR 1000.
    """
    slope = 2.0 * np.pi
    two_d = slope * np.radians(5.0)

    lifts = []
    for aspect_ratio in (4, 8, 16, 50, 200, 1000):
        wing = panel.lifting_line(5.0, slope, 0.0, aspect_ratio, taper=1.0, modes=64)
        assert wing["cl"] < two_d, "downwash can only take lift away"
        assert wing["cl_2d"] == pytest.approx(two_d, rel=1e-12)
        lifts.append(wing["cl"])
    assert lifts == sorted(lifts), f"lift should rise with aspect ratio: {lifts}"
    assert lifts[-1] == pytest.approx(two_d, rel=5e-3), (
        f"at AR 1000 the wing gave {lifts[-1]:.5f} against the 2D {two_d:.5f}"
    )

    # And the induced drag goes the other way, to zero.
    drags = [panel.lifting_line(5.0, slope, 0.0, ar, modes=64)["cd_induced"]
             for ar in (4, 8, 16, 50, 200, 1000)]
    assert drags == sorted(drags, reverse=True), drags
    assert drags[-1] < 1e-3


def test_an_elliptic_planform_matches_prandtls_closed_form(panel):
    """The one wing with a closed-form answer, and the only thing worth calling a
    validation of the lifting line rather than a property of it.

    Source: Prandtl, "Applications of modern hydrodynamics to aeronautics", NACA Report
    No. 116 (1921) -- elliptic loading gives CL = a0*alpha/(1 + a0/(pi*AR)) and
    CDi = CL^2/(pi*AR), so the span efficiency is exactly 1.

    An elliptic planform is not one of the shapes the command line offers -- taper is
    linear -- so the chord law is substituted here, which is the test doing its own
    independent calculation rather than asking the script to check itself. One Fourier
    mode is the exact solution for this planform, so the test also checks that the higher
    modes come back at nothing.
    """
    slope = 2.0 * np.pi
    for aspect_ratio in (4.0, 6.0, 10.0):
        for alpha in (2.0, 5.0, 8.0):
            modes = 32
            station = np.arange(1, modes + 1) * np.pi / (modes + 1)
            span = 1.0
            # An elliptic chord c = c0*sin(theta), with c0 set so that b^2/S = AR.
            root = 4.0 * span / (np.pi * aspect_ratio)
            chord = root * np.sin(station)
            mu = slope * chord / (4.0 * span)
            orders = np.arange(1, modes + 1)
            sines = np.sin(np.outer(station, orders))
            matrix = sines * (1.0 + mu[:, None] * orders[None, :] / np.sin(station)[:, None])
            amplitudes = np.linalg.solve(matrix, mu * np.radians(alpha))

            cl = np.pi * aspect_ratio * amplitudes[0]
            cdi = np.pi * aspect_ratio * float(np.sum(orders * amplitudes**2))
            closed_form = elliptic_wing_cl(slope, alpha, aspect_ratio)

            assert cl == pytest.approx(closed_form, rel=1e-6), (
                f"AR {aspect_ratio} at {alpha} deg: {cl:.6f} against Prandtl's "
                f"{closed_form:.6f}"
            )
            assert cdi == pytest.approx(cl**2 / (np.pi * aspect_ratio), rel=1e-6)
            assert abs(amplitudes[1:]).max() < 1e-12, "only the first mode is elliptic"


def test_the_lifting_line_reports_the_shape_of_the_loading(panel):
    """The spanwise numbers have to be the right way round, which is where an index error
    hides: the station coordinate runs 0 at the root to 1 at the tip, the chord falls with
    taper, and washout unloads the tip rather than the root.

    The rectangular wing's span efficiency is the comparison with a book here: classical
    lifting-line charts give e of about 0.95 for a rectangular wing of aspect ratio 6 and
    a little less at 8, against a tapered wing which comes nearer the elliptic 1.0.
    Measured: 0.954 at AR 6 rectangular, 0.937 at AR 8 rectangular, 0.983 at AR 8 with a
    0.5 taper.
    """
    slope = 2.0 * np.pi
    plain = panel.lifting_line(5.0, slope, 0.0, 8.0, taper=1.0, modes=48)
    assert plain["station_eta"].min() >= 0.0 and plain["station_eta"].max() <= 1.0
    assert np.allclose(plain["station_chord"], plain["station_chord"][0]), (
        "an untapered wing has one chord"
    )

    tapered = panel.lifting_line(5.0, slope, 0.0, 8.0, taper=0.5, modes=48)
    order = np.argsort(tapered["station_eta"])
    chords = tapered["station_chord"][order]
    assert np.all(np.diff(chords) <= 1e-12), "the chord must fall from root to tip"
    assert chords[-1] == pytest.approx(0.5 * chords[0], rel=0.02)

    # Loading is highest inboard and falls to the tip on both planforms.
    loading = tapered["section_cl"][order] * chords
    assert loading[0] > loading[-1]

    # Washout unloads the tip.
    washed = panel.lifting_line(5.0, slope, 0.0, 8.0, taper=0.5, twist_deg=-4.0, modes=48)
    tip = int(np.argmax(washed["station_eta"]))
    assert washed["section_cl"][tip] < tapered["section_cl"][tip]
    assert washed["cl"] < tapered["cl"], "washout costs total lift"

    # Span efficiency against the classical charts.
    assert panel.lifting_line(5.0, slope, 0.0, 6.0, taper=1.0,
                              modes=48)["span_efficiency"] == pytest.approx(0.954, abs=0.01)
    assert plain["span_efficiency"] == pytest.approx(0.937, abs=0.01)
    assert tapered["span_efficiency"] > plain["span_efficiency"]
    assert tapered["span_efficiency"] < 1.0, "nothing beats elliptic loading"

    for bad in ({"aspect_ratio": 0.0}, {"aspect_ratio": 8.0, "taper": 0.0},
                {"aspect_ratio": 8.0, "taper": 1.4}, {"aspect_ratio": 8.0, "modes": 2}):
        with pytest.raises(panel.Unusable):
            panel.lifting_line(5.0, slope, 0.0, **bad)


def test_the_wing_rides_on_this_sections_own_two_d_solution(panel):
    """The lifting line takes the section slope and zero-lift angle from the panel
    solution, so a cambered section's wing lifts at zero geometric incidence and a
    symmetric one's does not. If it were hardcoded to 2*pi that would not be true."""
    cambered, _ = section(panel, "4412", 200)
    symmetric, _ = section(panel, "0012", 200)

    for geom, expect_lift in ((cambered, True), (symmetric, False)):
        slope = panel.lift_slope(geom)
        wing = panel.lifting_line(0.0, slope["per_radian"],
                                  slope["zero_lift_alpha_deg"], 8.0, modes=32)
        assert (wing["cl"] > 0.2) is expect_lift, (
            f"zero-incidence wing Cl {wing['cl']:.4f}, zero-lift angle "
            f"{slope['zero_lift_alpha_deg']:+.3f} deg"
        )


# -- what it says about itself -------------------------------------------------


def test_the_flat_plate_friction_is_labelled_as_the_flat_plates(panel):
    """Blasius 1.328/sqrt(Re) and the 0.074/Re^0.2 turbulent line, both flat-plate
    results (Schlichting, "Boundary-Layer Theory", the flat-plate chapters). At Re = 6e6
    they give Cf = 5.42e-4 laminar and 3.26e-3 turbulent, so 0.0011 and 0.0066 on
    2.04 x wetted area -- which happens to bracket the order of a real section's profile
    drag at that Reynolds number and says nothing whatever about this section's. That is
    exactly what the label has to carry, and this checks that it does."""
    friction = panel.flat_plate_friction(6.0e6)
    assert friction["cf_laminar"] == pytest.approx(1.328 / np.sqrt(6.0e6), rel=1e-12)
    assert friction["cf_turbulent"] == pytest.approx(0.074 / 6.0e6**0.2, rel=1e-12)
    assert friction["cf_turbulent"] > friction["cf_laminar"]
    assert friction["cd_turbulent"] == pytest.approx(0.0066, abs=0.0004)
    label = friction["label"].lower()
    assert "flat-plate" in label and "not this section" in label
    with pytest.raises(panel.Unusable):
        panel.flat_plate_friction(0.0)


def test_the_output_says_what_it_cannot_do(panel, capsys):
    """Not only the docstring. A Cl on stdout with no caveat beside it is a Cl that gets
    quoted, so the limits travel with every answer -- including a refusal, where the
    temptation to drop them is strongest, and including the json sidecar, which is the
    copy a script reads."""
    assert panel.main(["--naca", "2412", "--alpha", "5"]) == 0
    printed = capsys.readouterr().out.lower()
    for phrase in ("what this cannot tell you", "no viscosity", "no stall",
                   "no separation", "no reynolds number", "linear and attached"):
        assert phrase in printed, f"the output never says {phrase!r}"

    # A refusal keeps them too.
    assert panel.main(["--naca", "24120"]) == 0
    refused = capsys.readouterr().out.lower()
    assert "not estimated" in refused and "what this cannot tell you" in refused


def test_the_json_sidecar_carries_the_numbers_and_the_caveats(panel, tmp_path, capsys):
    out = tmp_path / "panel.json"
    assert panel.main(["--naca", "2412", "--sweep", "-2", "6", "2", "--json", str(out),
                       "--wing", "--aspect-ratio", "7", "--taper", "0.6",
                       "--reynolds", "2e6"]) == 0
    capsys.readouterr()
    data = json.loads(out.read_text())

    assert len(data["incidence"]) == 5
    assert [row["alpha_deg"] for row in data["incidence"]] == [-2.0, 0.0, 2.0, 4.0, 6.0]
    assert len(data["incidence"][0]["cp"]) == data["panels"]
    assert data["section"]["zero_lift_alpha_deg"] < 0
    # One wing per swept incidence, each carrying the incidence it is for (#147): the
    # sidecar used to hold a single wing solved at `--alpha` while the sweep covered
    # -2..6, and nothing in it said which angle that one number was.
    assert [row["alpha_deg"] for row in data["wing"]] == [-2.0, 0.0, 2.0, 4.0, 6.0]
    for row in data["wing"]:
        assert row["cl"] < row["cl_2d"]
    assert data["flat_plate_friction"]["reynolds"] == 2.0e6
    assert data["limits"], "the sidecar must carry the limits, not only stdout"
    assert any("no viscosity" in line for line in data["limits"])
    assert "Hess-Smith" in data["method"]


def test_the_wing_is_run_at_every_swept_incidence_and_each_row_names_its_own(
    panel, tmp_path, capsys
):
    """#147. `--sweep A B STEP --wing` ran the lifting line once, at `--alpha`, which
    defaults to 0, and printed it beside a lift curve covering A..B with nothing on
    the page saying which angle the wing number was for. The number was right for an
    incidence the output never named: `--naca 2412 --sweep 0 8 2 --wing` printed
    `Cl +0.1927` against a last sweep row of `+1.2153`, eight degrees apart.

    The arithmetic was never in question and must not move: 0.1927 at alpha 0 and
    0.6478 at alpha 5 are both checked against the elliptic closed form elsewhere in
    this file. What changes is that there is now one list of incidences and the wing
    is on it, so no row can be read against an angle nobody stated.
    """
    assert panel.main(["--naca", "2412", "--sweep", "0", "8", "2", "--wing",
                       "--aspect-ratio", "8"]) == 0
    printed = capsys.readouterr().out
    block = printed.split("# wing (lifting line)")[1].split("# what this cannot")[0]
    for alpha in ("+0.00", "+2.00", "+4.00", "+6.00", "+8.00"):
        assert alpha in block, f"the wing block names no row for alpha {alpha}"
    assert "+0.1927" in block, "the alpha 0 wing Cl is unchanged arithmetic"

    # The section's own sweep and the wing's now run over the same incidences, which is
    # the whole of the fix: a reader can put the two tables side by side row for row.
    section_rows = [line.split()[0] for line in
                    printed.split("# incidence")[1].splitlines()[2:7]]
    wing_rows = [line.split()[0] for line in block.splitlines()[3:8]]
    assert section_rows == wing_rows, (section_rows, wing_rows)


def test_the_figures_get_written(panel, tmp_path, capsys):
    """Cp against x for each incidence and the lift curve over a sweep, through Agg."""
    out = tmp_path / "figures"
    assert panel.main(["--naca", "0012", "--sweep", "0", "4", "2", "--png", str(out)]) == 0
    capsys.readouterr()
    written = sorted(path.name for path in out.glob("*.png"))
    assert "lift_curve.png" in written
    assert sum(name.startswith("cp_alpha") for name in written) == 3
    for path in out.glob("*.png"):
        assert path.stat().st_size > 5000, f"{path.name} is suspiciously small"


def test_it_runs_standalone_and_exits_zero_whatever_it_finds(tmp_path):
    """`python3 panel.py --help` and a plain run, as a subprocess, because being runnable
    on its own is the contract every script here keeps."""
    script = TOOLBOX / "panel.py"
    helped = subprocess.run([sys.executable, str(script), "--help"],
                            capture_output=True, text=True, timeout=180)
    assert helped.returncode == 0
    assert "--naca" in helped.stdout and "--sweep" in helped.stdout

    done = subprocess.run([sys.executable, str(script), "--naca", "0012", "--alpha", "3"],
                          capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, done.stderr
    assert "Cl" in done.stdout or "cl" in done.stdout

    missing = tmp_path / "nope.dat"
    missing.write_text("1.0 0.0\n")
    refused = subprocess.run([sys.executable, str(script), "--dat", str(missing)],
                             capture_output=True, text=True, timeout=180)
    assert refused.returncode == 0, "exit 0 whatever it finds"
    assert "not estimated" in refused.stdout
