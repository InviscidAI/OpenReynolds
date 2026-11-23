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

import math

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


CGW_LOW_RE_RANGE = (0.01, 20.0)
"""The Reynolds range of the Clift, Grace & Weber branch below.

The paper is piecewise and this is only its `0.01 < Re < 20` branch. Outside it the
formula is not CGW's answer, which is why `clift_grace_weber_cd` refuses rather than
extrapolating: an out-of-range value from a piecewise fit looks exactly like an
in-range one."""


def clift_grace_weber_cd(reynolds: float) -> float:
    """Sphere drag from Clift, Grace & Weber (1978), the `0.01 < Re < 20` branch.

        Cd = (24/Re) (1 + 0.1315 Re^(0.82 - 0.05 log10 Re))

    **This is the bar the live sphere case is graded against** (DECISIONS_LIVE.md D16),
    chosen because it is the best-validated of the correlations available: Feng
    (arXiv:2610.00334v3) reports average relative errors against 480 experimental points
    of 3.019% for CGW, 3.236% for Brown & Lawler and 2.469% for Cheng (2009).

    **It is independent of `schiller_naumann_cd`, which `clift_gauvin_cd` is not.** That
    matters and is the reason this function exists: `clift_gauvin_cd` *calls*
    `schiller_naumann_cd` and adds a high-Reynolds plateau worth 0.0034% at Re 10, so the
    two agree there because they are the same formula, not because two independent fits
    concur. A validation argument was built on that agreement and had to be withdrawn.

    **Provenance, stated because it is not clean.** The primary text -- *Bubbles, Drops,
    and Particles* (1978) -- could not be reached. The branch above was read from two
    independent code transcriptions that both cite it (`fluids` and `pychemqt`). Two
    internal checks were run before adopting it: the exponent `0.82 - 0.05 log10 Re`
    gives exactly `0.82` at Re 1 and the expression reduces to Stokes as `Re -> 0`; and
    the correction term reaches unity near Re 20, the top of the stated branch, which is
    where a low-Re branch should hand over. Neither check substitutes for the book, so
    any case grading against this must say so in its target string.

    At Re 10 this gives **4.258391**, against Schiller-Naumann's 4.151066.
    """
    if not CGW_LOW_RE_RANGE[0] <= reynolds <= CGW_LOW_RE_RANGE[1]:
        raise ValueError(
            f"Clift, Grace & Weber is piecewise and this is its "
            f"{CGW_LOW_RE_RANGE[0]} < Re < {CGW_LOW_RE_RANGE[1]} branch; Re "
            f"{reynolds:g} is outside it. Extrapolating a piecewise fit returns a "
            f"number that looks like an answer"
        )
    exponent = 0.82 - 0.05 * math.log10(reynolds)
    return 24.0 / reynolds * (1.0 + 0.1315 * reynolds ** exponent)


def newton_regime_cd() -> float:
    """About 0.44: the roughly constant drag coefficient of a sphere between Reynolds
    about 1e+03 and the drag crisis near 3e+05.

    It is a plateau on a log plot rather than a constant of nature, and it ends abruptly
    when the boundary layer turns turbulent and the drag falls by a factor of about
    four. A calculation that uses it either side of that transition is wrong by more
    than any mesh error it is likely to be worrying about.
    """
    return 0.44
