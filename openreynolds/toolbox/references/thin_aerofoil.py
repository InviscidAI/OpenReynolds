"""Closed-form aerofoil and wing results, for grading a panel method against.

Why this file exists. A study on 2026-11-22 (20261122-113750-c5a7) was asked to check a
panel method against four exact statements and reported, of two of them, that it had to
write the reference itself: "the toolbox's copy is only in a test file that isn't on this
machine, so I wrote my own version", and that `--wing` has no elliptic planform so it
built that setup too. Its answers were sound. But a check is only worth the reference
behind it, and "graded against the thin-aerofoil integral" and "graded against the
integral the run wrote ten minutes ago" are different claims that look identical in a
result.

What this file is and is not. It is a reference to compare a solver against, written out
from the published definitions below. It is **not** the repository's check on
`panel.py`: that lives in `tests/test_toolbox_panel.py`, which deliberately keeps its own
independent copy of these same formulas, on the principle that a test importing the thing
it is checking is checking nothing. That principle is untouched. This copy exists so that
a study running in a workspace has something to read instead of something to reinvent,
and a test holds the two numerically equal so they cannot drift apart.

Everything here is arithmetic on published definitions. There is no solver, no mesh and
no fitted constant.
"""
from __future__ import annotations

import numpy as np

__all__ = ["naca4_camber_slope", "thin_aerofoil_zero_lift_deg", "elliptic_wing_cl",
           "elliptic_span_efficiency"]


def naca4_camber_slope(designation: str, x: np.ndarray) -> np.ndarray:
    """dyc/dx of a NACA 4-digit mean line, from the published family definition.

    Source: Jacobs, Ward and Pinkerton, "The characteristics of 78 related airfoil
    sections from tests in the variable-density wind tunnel", NACA Report No. 460
    (1933), which defines the 4-digit mean line as two parabolic arcs; the same
    definition is in Abbott and von Doenhoff, "Theory of Wing Sections" (Dover, 1959),
    section 6.4.

    `designation` is the four-digit string, e.g. "2412": the first digit is maximum
    camber in percent of chord, the second is its position in tenths. A symmetric
    section (either digit zero) has zero camber slope everywhere, which is why its
    zero-lift angle is exactly zero and why that check grades sign conventions rather
    than accuracy.
    """
    m = int(designation[0]) / 100.0
    p = int(designation[1]) / 10.0
    if m == 0.0 or p == 0.0:
        return np.zeros_like(x)
    return np.where(x <= p,
                    2.0 * m / p**2 * (p - x),
                    2.0 * m / (1.0 - p) ** 2 * (p - x))


def _trapezium(y: np.ndarray, x: np.ndarray) -> float:
    """The trapezium rule, written out.

    `numpy.trapezoid` only exists from numpy 2.0 and `numpy.trapz` is removed in it, so
    naming either would work on some installs and not others. One line avoids the
    version check entirely.
    """
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def thin_aerofoil_zero_lift_deg(designation: str, points: int = 200001) -> float:
    """The thin-aerofoil zero-lift angle of a NACA 4-digit camber line, in degrees.

        alpha_L0 = -(1/pi) * integral_0^pi (dyc/dx) (cos(theta) - 1) dtheta,
        with x = (1 - cos(theta)) / 2

    Source: Anderson, "Fundamentals of Aerodynamics", thin airfoil theory for the
    cambered aerofoil -- the alpha_L0 result that follows from the Fourier solution of
    the camber-line integral equation. The same expression is Glauert, "The Elements of
    Aerofoil and Airscrew Theory" (1926), chapter VII.

    Evaluated by the trapezium rule on a fine grid, with the endpoints nudged off the
    singular limits of theta. It is a property of the camber line alone: thickness does
    not enter, which is exactly why a panel method on a real section with thickness is
    expected to sit near this rather than on it.
    """
    theta = np.linspace(1e-9, np.pi - 1e-9, points)
    x = 0.5 * (1.0 - np.cos(theta))
    slope = naca4_camber_slope(designation, x)
    return float(np.degrees(-_trapezium(slope * (np.cos(theta) - 1.0), theta) / np.pi))


def elliptic_wing_cl(slope_per_rad: float, alpha_abs_deg: float,
                     aspect_ratio: float) -> float:
    """The closed-form lift coefficient of an untwisted elliptic wing.

        CL = a0 alpha / (1 + a0 / (pi AR))

    Source: Prandtl, "Applications of modern hydrodynamics to aeronautics", NACA Report
    No. 116 (1921); the standard textbook form is the elliptic-lift-distribution result
    of finite-wing theory in Anderson, "Fundamentals of Aerodynamics".

    Read the caveat before grading anything with it. For an elliptic planform the
    lifting-line series carries only its first mode, so a lifting-line solver reproduces
    this identically, by construction, whatever it does elsewhere. Agreement therefore
    says the planform was set up as elliptic. It does not say induced drag is computed
    correctly, and a study that reports "exactly 1" here has learned less than the number
    suggests.
    """
    return slope_per_rad * np.radians(alpha_abs_deg) / (1.0 + slope_per_rad
                                                        / (np.pi * aspect_ratio))


def elliptic_span_efficiency() -> float:
    """The span efficiency of an elliptic planform: exactly 1, by definition.

    Included so that a comparison has something to name. Elliptic loading minimises
    induced drag for a given lift and span, and every other planform is quoted as a
    fraction of it, so this is the definition of the scale rather than a result on it.
    """
    return 1.0
