"""Drag on a smooth sphere: the standard correlations, with the range each one holds over.

Why this file exists. Study 20261122-143008-18af solved a sphere at Reynolds 100 and
compared its drag against published correlations. It got a good answer and said, three
separate times, that it was working from memory: the Clift-Gauvin accuracy was "quoted
from memory and did not check", a wake length was "as I recall it", and a separation
angle "I recall about 128 degrees". That is the fourth study to reach for a published
number this toolbox did not carry.

What is here and what is not. The correlations are closed-form expressions that can be
written down exactly, so they ship. The experimental wake geometry that study wanted --
recirculation length and separation angle against Johnson and Patel -- does NOT ship,
because those are measurements read off a paper rather than formulas, and a number
recalled into this file would be the same failure one level deeper. If you need them,
read the paper.

Nothing here is a solver. It is arithmetic, and the only thing it can tell you is what
these published fits say.
"""
from __future__ import annotations

__all__ = ["stokes_cd", "schiller_naumann_cd", "clift_gauvin_cd", "newton_regime_cd"]

STOKES_LIMIT_RE = 0.1
"""Above about this, Stokes drag is no longer the whole story.

Stokes' law is exact in the limit of vanishing Reynolds number, not at small-but-finite
values, and the difference bites sooner than people expect: at Re 1 the Schiller-Naumann
correction is about 15% and at Re 10 it is about 70%. A brief written for this project
on 2026-11-22 called Stokes "nearly exact" at Re 1, which is wrong by that 15%, so the
number is recorded here rather than left to be re-derived under pressure.
"""


def stokes_cd(reynolds: float) -> float:
    """Cd = 24/Re, exact as Re approaches zero.

    Source: Stokes (1851). This is a solution of the creeping-flow equations rather than
    a fit, so in its own limit it is exact and is the strongest check available on a
    sphere. Its limit is the point: see `STOKES_LIMIT_RE`. It carries no separation,
    because creeping flow has none.
    """
    if reynolds <= 0.0:
        raise ValueError("Reynolds number must be positive")
    return 24.0 / reynolds


def schiller_naumann_cd(reynolds: float) -> float:
    """Cd = (24/Re)(1 + 0.15 Re^0.687), the standard fit below Re about 1000.

    Source: Schiller and Naumann (1933). A fit to measurements, not a derivation, so it
    carries the scatter of the data behind it; quote its accuracy from a source rather
    than from here. It reduces to Stokes as Re approaches zero, which is the arithmetic
    check this file's tests use.
    """
    if reynolds <= 0.0:
        raise ValueError("Reynolds number must be positive")
    return 24.0 / reynolds * (1.0 + 0.15 * reynolds ** 0.687)


def clift_gauvin_cd(reynolds: float) -> float:
    """Schiller-Naumann plus the term that carries it through the Newton plateau.

        Cd = (24/Re)(1 + 0.15 Re^0.687) + 0.42 / (1 + 42500 Re^-1.16)

    Source: Clift, Grace and Weber, "Bubbles, Drops and Particles" (1978), the
    Clift-Gauvin form. The added term vanishes at low Reynolds number and approaches
    0.42 at high, which is the flat part of the drag curve. Also a fit.
    """
    if reynolds <= 0.0:
        raise ValueError("Reynolds number must be positive")
    return (schiller_naumann_cd(reynolds)
            + 0.42 / (1.0 + 42500.0 * reynolds ** -1.16))


def newton_regime_cd() -> float:
    """About 0.44: the roughly constant drag coefficient of a sphere between Reynolds
    about 1e+03 and the drag crisis near 3e+05.

    It is a plateau on a log plot rather than a constant of nature, and it ends abruptly
    when the boundary layer turns turbulent and the drag falls by a factor of about
    four. A calculation that uses it either side of that transition is wrong by more
    than any mesh error it is likely to be worrying about.
    """
    return 0.44
