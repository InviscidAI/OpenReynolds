"""The SPH backend's validation cases: what each measures, against what, and how close.

Written against `docs/sph-contract.md` **before the solver exists**, which is the point of
that document: the interface is fixed up front so the solver and its grading can be built
at the same time without either guessing at the other. Everything here calls only what
the contract promises. Where a case needs something the contract does not promise, it says
so in its own docstring and in `NEEDS_FROM_THE_SOLVER` at the bottom of this module rather
than inventing an attribute and hoping.

This module is deliberately a sibling of `cases.py` and not an extension of it. The
grading machinery is shared -- `Case`, `Check`, `_result` and `report` are imported, not
re-declared -- but the *physics* bookkeeping is not: a lattice case reports MLUPS, a
Smagorinsky constant and a `Scaling`, and an SPH case reports a particle radius, a
particle count and `density_error()`. Folding one into the other would give every case a
report with half its lines empty.

## Where the references came from, and which of them could not be read

Same rule as `cases.py`, which is `benchmarks/README.md`'s rule: *a case with a reference
and no measured agreement beside it teaches a reader to trust an unvalidated number*. So
every entry in `REFERENCES` carries its source, what was actually read, its caveat, and
what its tolerance is on -- and an entry that was not read is marked `graded: False` and
is **reported in the result and graded in nothing**.

Two references here could not be read, and the cases are posed around that:

* **Martin and Moyce (1952).** `benchmarks/README.md`'s SPH dam-break row and
  `DECISIONS_LIVE.md` D14 both record that the publisher refuses the request, and that the
  two non-dimensionalisations in circulation -- `t sqrt(g/a)` and `t sqrt(2g/a)` -- differ
  by a factor of 1.414 at every point. So `run_sph_dam_break` is graded against
  **Ritter's (1892)** closed form, which is three lines of characteristics and needs no
  paper, and the Martin and Moyce comparison is **printed in both conventions and graded
  in neither**. The front position the addendum asks for is reported at three depth
  thresholds and graded nowhere, for D14's reason.
* **Bogner and Rude (2013).** The published version returned HTTP 403 to every attempt;
  the authors' arXiv preprint (1201.0351v1) is reachable but its heeling section did not
  extract as text, so nothing in it can be quoted. What a heeling case can be graded
  against instead is **derivable**: the metacentric height of a wall-sided prism, and the
  heel angle at which its righting moment vanishes, are elementary hydrostatics and are
  re-derived in `metacentric_height` and `heel_equilibrium`. `run_sph_heel_stability`
  grades those and reports the paper as unread.

## What is graded, and what is only reported

A number is graded only when its reference is exact or derivable and the measurement has
no free parameter in it. Three things in this file are therefore reported and not graded,
each for a reason written into its own `Check`:

* the **dam-break front position**, because Ritter's front thins to zero depth, so where
  it is depends entirely on how little water still counts as water (`DECISIONS_LIVE.md`
  D14 has the measured size of that: 0.168 m against a true 0.220 m);
* the **Martin and Moyce non-dimensional time**, in both conventions;
* the **heave added-mass coefficient**, because `T = 2 pi sqrt(d/g)` neglects added mass
  and no analytic added mass for a free-surface-piercing prism has been read for this
  repository. What *is* graded there is the sign: nothing shortens a heave period, so a
  measured period below the estimate is a defect and not a modelling gap.

## The three decisions this file inherits rather than re-deriving

`cases.py` paid for these and they transfer unchanged to a particle method:

1. **The dam break is graded on Ritter's profile, never on the front position**
   (`DECISIONS_LIVE.md` D14), and its reservoir is four depths wide so that the
   shallow-water window `T` in [2, 8] exists at all.
2. **A period is the mean of two estimators with their disagreement graded beside it.**
   `cases.py::_period_of` and `_damping_of` are imported, not copied: the measurements
   that forced both -- the crossings winning on one run and the FFT on the next, and a
   damped period being longer by `1/sqrt(1 - zeta^2)` -- are in their docstrings.
3. **A floating body is settled on the fluid's own velocity, never on a step count**, its
   waterline is the median over columns free of the hull rather than the mean over the
   outer ring, and its draft is averaged after the release transient has decayed over
   several heave periods. All three are fixes to `run_archimedes`, whose first version got
   each of them wrong; `PROGRESS_LIVE.md`'s M6 section is the catalogue.

## Why the floating cases are heavy and the free-surface ones are not

The contract's budget -- "CPU runs a few thousand particles in seconds, for the suite" --
covers M5. It does not cover M6. A tank wide enough that a hull does not feel its walls,
deep enough to float in, with enough particles through the *draft* for the displaced
volume to mean anything, is tens of thousands of particles in three dimensions: particle
count goes as the cube of the refinement, and the draft is the smallest length in the
problem. `run_sph_static_float` defaults to a tank of about 35,000 particles with **four
particle spacings through the draft**, and reports that count as the first thing to raise
if it misses. The convergence case refines it twice more and is firmly a GPU case.

A lattice case escapes this by going quasi-two-dimensional: three planes, periodic in the
thin direction (`DECISIONS_LIVE.md` D13). The contract promises no periodicity for SPH, so
that escape is not available here -- which is the largest single thing this module wants
from the solver that the contract does not promise.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import figures
from .cases import Case, Check, _result, report
from .units import UnphysicalScaling

__all__ = [
    "SPH_CASES", "REFERENCES", "NEEDS_FROM_THE_SOLVER",
    "describe_sph_cases", "run_sph_case", "report",
    # the analytic references, which are the testable part of this module
    "ritter_depth", "ritter_front", "linear_sloshing", "heave_period",
    "archimedes_draft", "metacentric_height", "heel_equilibrium", "richardson",
    "noise_fraction",
]

GRAVITY = 9.81
"""One gravity for every case here, so that a number lifted from one case's report into
another's arithmetic is the same gravity. `cases.py` writes `gravity = 9.81` into each
free-surface runner separately and that is one copy per runner."""


# -- the references ---------------------------------------------------------------------
#
# Four of these are the same references the lattice free-surface cases use, and they are
# built from `cases.REFERENCES` rather than retyped. That is not tidiness: a reference
# duplicated into a second dictionary is a tolerance that can be widened in one copy, and
# `tests/test_live_sph_cases.py` asserts the shared fields still agree with `cases.py`'s.
# Only `tolerance_is` is overridden, because what a tolerance is *on* genuinely differs
# between a lattice and a particle method.

def _shared(name: str, tolerance_is: str, **extra: Any) -> dict[str, Any]:
    """One of `cases.REFERENCES`' entries, with this backend's `tolerance_is`.

    `source`, `read`, `caveat` and `tolerance` come through untouched. The provenance of a
    published value does not depend on which solver is being graded against it, and a
    second hand-typed copy of a caveat is a caveat that will drift out of agreement with
    the first one.
    """
    from .cases import REFERENCES as LATTICE

    try:
        entry = LATTICE[name]
    except KeyError:  # pragma: no cover - a rename in cases.py, caught by the tests
        raise UnphysicalScaling(
            f"cases.REFERENCES has no {name!r} entry to share. These references are "
            f"deliberately one copy, not two: if the lattice entry was renamed, point "
            f"this at the new name rather than retyping the caveat"
        ) from None
    return {**entry, "tolerance_is": tolerance_is, "graded": True, **extra}


REFERENCES: dict[str, dict[str, Any]] = {
    "hydrostatic": _shared(
        "hydrostatic",
        # **The one shared entry whose tolerance is NOT the lattice's, and the difference
        # is the quantity and not the bound.** The lattice case grades `p = rho g h` by
        # proxy -- the surface height and the flatness, at 2% -- because its measurable is
        # `d(rho)/dz`. Here the contract promises `pressures()`, so the pressure itself is
        # graded, and the number this repository has already measured for that quantity is
        # 2.50% RMS (`benchmarks/README.md`'s `sph/` row). A 2% bound would fail a solver
        # that matched the best this repository has achieved, which is not a tolerance, it
        # is a trap. `tests/test_live_sph_cases.py` pins both numbers so they cannot drift
        # apart from this comment.
        tolerance=0.05,
        tolerance_is=(
            "5% RMS on the particle pressures against `rho g (h - z)`, as a fraction of "
            "`rho g h`. The bound is set from this repository's own measurement of the "
            "same quantity rather than from nothing: `benchmarks/README.md`'s "
            "`sph/` hydrostatic row records **2.50% RMS** with a worst particle at 8.61%, "
            "achieved once a density-diffusion term was added -- it read 67% before. So "
            "5% is twice the number already achieved and should be **tightened to the "
            "measurement** once this backend has one. The worst particle is reported with "
            "its depth below the surface and is not graded: the kernel support is "
            "truncated at a free surface and there is no fluid above it to sum, so the "
            "worst particle is a statement about where the surface is, which the grader "
            "measures rather than asserts"
        ),
    ),
    "ritter": _shared(
        "ritter",
        tolerance_is=(
            "15% on Ritter's depth at the dam, `4 h0 / 9`, and on the RMS of the depth "
            "profile where Ritter is thicker than two particle spacings -- the same bound "
            "and the same two quantities the lattice case grades, so that the two "
            "backends are graded identically on the one case they share. The front "
            "position the addendum asks for is reported at three depth thresholds and "
            "graded in none of them (`DECISIONS_LIVE.md` D14)"
        ),
    ),
    "linear_sloshing": _shared(
        "linear_sloshing",
        tolerance_is=(
            "5% on the first-mode period, taken as the mean of an FFT and the zero "
            "crossings with the damping removed, and with the gap between the two "
            "estimators graded at the same 5% beside it. A measurement whose two "
            "estimators disagree by more than the tolerance cannot settle anything at "
            "that tolerance, however close their mean lands"
        ),
    ),
    "archimedes": _shared(
        "archimedes",
        tolerance_is=(
            "2% of the draft, which the addendum asks for by name. The quantisation "
            "caveat carries over from the lattice with a different cause and the same "
            "arithmetic: an SPH hull is not voxelised, but its Akinci boundary particles "
            "occupy a layer about one spacing thick, so the displaced volume is uncertain "
            "by roughly `dp` over the wetted surface. At four particles through the draft "
            "that is percents, and the case reports the count"
        ),
    ),
    "martin_moyce": {
        "source": (
            "Martin, J.C. and Moyce, W.J. (1952), 'An experimental study of the collapse "
            "of liquid columns on a rigid horizontal plane', Philosophical Transactions "
            "of the Royal Society A 244:312-324"
        ),
        "read": (
            "**not read.** The publisher refuses the request; `benchmarks/README.md`'s "
            "`sph/` dam-break row records the same refusal, and `DECISIONS_LIVE.md` D14 "
            "is the decision taken on it. No number from it is quoted here, from it or "
            "from any source that cites it"
        ),
        "caveat": (
            "Which non-dimensionalisation the paper defines is **unverified**, and "
            "`t sqrt(g/a)` and `t sqrt(2g/a)` are both in circulation and differ by a "
            "factor of 1.414 at every point. A front position quoted in the wrong one "
            "agrees with a solver that is 41% wrong, or disagrees with one that is right. "
            "Quoting it through a secondary source is the error this repository argues "
            "hardest against, so the comparison is computed in both conventions and "
            "printed, and the grading goes to Ritter instead"
        ),
        "tolerance": None,
        "graded": False,
        "tolerance_is": (
            "nothing: **reported, not graded**. There is no tolerance because there is no "
            "reference value -- only an unverified choice between two that differ by 1.414"
        ),
    },
    "bogner_rude": {
        "source": (
            "Bogner, S. and Rude, U. (2013), 'Simulation of floating bodies with the "
            "lattice Boltzmann method', Computers and Mathematics with Applications "
            "65(6):901-913, doi 10.1016/j.camwa.2012.09.012; the authors' preprint is "
            "arXiv:1201.0351, 'Liquid-gas-solid flows with lattice Boltzmann: Simulation "
            "of floating bodies'"
        ),
        "read": (
            "**not read.** Every attempt at the published version returned HTTP 403, and "
            "the article page could not be opened to confirm whether it is free to read. "
            "The authors' arXiv preprint is reachable and its abstract was read -- it "
            "states that the evaluation concentrates on 'the rotational stability of "
            "non-spherical rigid bodies floating on a plane water surface' and reports "
            "'convergence towards the ideal solution for the measured heeling stability "
            "of a floating box' -- but **the preprint's own heeling section did not "
            "extract as text**, so no geometry, no resolution and no measured value from "
            "it is quoted here. What is quoted above is the title, the authors, the "
            "journal and the DOI, which came from Crossref"
        ),
        "caveat": (
            "A reference whose numbers have not been seen cannot grade anything, and this "
            "repository's rule is that it is reported rather than reconstructed. Two "
            "further things would stand between it and this case even if it had been "
            "read: it is a **lattice Boltzmann** free-surface method with a different "
            "boundary treatment, and its heeling test is reported in lattice units, so a "
            "comparison would need a non-dimensionalisation that the paper defines and "
            "that has not been read either -- which is exactly the hole Martin and Moyce "
            "fell into, one entry above"
        ),
        "tolerance": None,
        "graded": False,
        "tolerance_is": (
            "nothing: **reported, not graded**. What `run_sph_heel_stability` grades "
            "instead is derivable without it -- the metacentric criterion and the "
            "wall-sided heel equilibrium, both re-derived in this module and hand-checked "
            "in `tests/test_live_sph_cases.py`"
        ),
    },
    "metacentric": {
        "source": (
            "classical hydrostatics, re-derived rather than quoted: for a prism of beam "
            "`b` and height `H` floating at draft `d = r H`, the centre of buoyancy is at "
            "`d/2` above the keel, the centre of gravity at `H/2`, and the metacentric "
            "radius is the second moment of the waterplane over the displaced volume, "
            "`BM = b^2 / (12 d)`. Upright flotation is stable when `GM = d/2 + b^2/(12 d) "
            "- H/2` is positive, which rearranges to `(b/H)^2 > 6 r (1 - r)`. For a "
            "wall-sided prism the righting arm vanishes at `tan^2(a) = 12 r (1-r) (H/b)^2 "
            "- 2`, which is the heeled equilibrium when the upright one is unstable"
        ),
        "read": (
            "n/a -- derived in `metacentric_height` and `heel_equilibrium`, and "
            "hand-checked at values that can be computed on paper: a square-section prism "
            "at half water's density heels to exactly 45 degrees, and at a quarter to "
            "exactly `atan(1/2)` = 26.565 degrees"
        ),
        "formula": (
            "GM = d/2 + b^2/(12 d) - H/2 with d = r H;  stable upright iff (b/H)^2 > "
            "6 r (1-r);  tan^2(a_eq) = 12 r (1-r) (H/b)^2 - 2"
        ),
        "caveat": (
            "**Exact as hydrostatics and three things stand between it and a solver.** It "
            "is wall-sided: the waterline must stay on the two vertical sides, so the "
            "deck must not immerse and the keel must not emerge, `tan(a) <= 2 min(d, H-d) "
            "/ b`, which the case computes and reports as a margin rather than assuming. "
            "It is quasi-static, so a box released at a small heel reaches the equilibrium "
            "through a roll oscillation that the free surface damps, and the angle has to "
            "be averaged after that decays -- the same mistake `run_archimedes` made with "
            "a draft. And it is uniform-density: the prism's centre of gravity is at "
            "`H/2` by construction, which is true of the body a case builds and not of a "
            "real hull"
        ),
        "tolerance": 0.10,
        "graded": True,
        "tolerance_is": (
            "**two things, graded separately and deliberately.** The *sign* is graded "
            "exactly: `GM < 0` predicts that a box released near upright leaves it and "
            "`GM > 0` that it returns, and a solver that gets that backwards is wrong "
            "whatever angle it reports. The *angle* is graded at 10%, which is a grading "
            "choice and is stated as one: the closed form is quasi-static and inviscid "
            "while the measured equilibrium is reached through a damped roll in a tank of "
            "finite width, and no read reference bounds the gap"
        ),
    },
    "heave": {
        "source": (
            "derived: the heave stiffness of a prism of waterplane area `A` is `rho g A` "
            "and its mass is `rho_body A H = rho A d`, so the undamped natural period is "
            "`T = 2 pi sqrt(d / g)` and depends on the draft alone"
        ),
        "read": "n/a -- three lines, and `cases.py::run_archimedes` already sizes its run with it",
        "formula": "T = 2 pi sqrt(d / g),  d the draft",
        "caveat": (
            "**It neglects added mass, and that is not a small neglect.** A prism heaving "
            "at a free surface accelerates a comparable mass of water with it, so the "
            "real period is longer -- for a shallow-draft box, substantially. There is "
            "also a radiation damping, so what is measured is a damped period, longer "
            "again by `1/sqrt(1 - zeta^2)`. No analytic added mass for a "
            "surface-piercing rectangular prism has been read for this repository, so the "
            "magnitude of the correction is **measured and reported, not predicted**: the "
            "case prints `(T_measured/T_estimate)^2 - 1`, which is the added mass as a "
            "fraction of the body's own, for whoever wants to compare it with a strip "
            "theory later"
        ),
        "tolerance": 0.25,
        "graded": True,
        "tolerance_is": (
            "**the sign is the graded part and the magnitude is a bound.** `T_measured >= "
            "T_estimate` is graded with no tolerance at all, because nothing in the "
            "physics shortens a heave period: added mass lengthens it, damping lengthens "
            "it, and a measured period *below* the estimate means the restoring force is "
            "too stiff -- the signature of a hull being held up by something other than "
            "the pressure field. 25% is then a loose upper bound on the added-mass "
            "correction, stated as a grading choice, and the measured coefficient is "
            "reported beside it so a reader sees how much of the 25% was used"
        ),
    },
    "force_noise": {
        "source": (
            "no reference: this is a property of the solver, not of the fluid, and the "
            "requirement it is graded against is a control one. The mean force it is "
            "normalised by has an exact reference -- a fully submerged body feels "
            "`+rho V g`, which `docs/sph-contract.md` makes the standing check on "
            "`wrench()`'s sign convention"
        ),
        "read": "n/a",
        "formula": "noise = RMS(F - mean F) / |mean F|,  over a window after the fluid has settled",
        "caveat": (
            "**Pressure noise is what SPH is worst at, and a force is its surface "
            "integral, which is the one thing that helps.** The `sph/` hydrostatic row in "
            "`benchmarks/README.md` has the mechanism: `dp/drho` is about `c0^2`, so a "
            "density error of `e` is a pressure error of `e c0^2`, which was 400 times "
            "`e` as a fraction of the `rho g h` being measured. Summing that over a few "
            "hundred boundary particles averages much of it away, so a force is far "
            "quieter than a point pressure -- how much quieter is the measurement, and it "
            "is **not** predictable from the density error alone because neighbouring "
            "particles' errors are correlated. The number also depends on the window: "
            "noise measured per step and noise measured over the 10 steps a 100 Hz outer "
            "loop can average are different numbers, and both are reported"
        ),
        "tolerance": 0.10,
        "graded": True,
        "tolerance_is": (
            "**10% of the mean force, per step, and the bound is a control requirement "
            "rather than a physical one.** A controller cannot use a force whose "
            "fluctuation exceeds its signal: at 1 kHz with no filtering, 10% RMS is a "
            "signal-to-noise ratio of 10, and a controller that can average 10 samples "
            "(10 ms, still inside a 100 Hz outer loop) sees 10%/sqrt(10) = 3.2% if the "
            "noise is white. Both numbers are reported and the averaged one is reported "
            "**with its measured reduction factor**, not with sqrt(10) assumed: if the "
            "noise is correlated step to step, averaging buys less than that, and the "
            "ratio of the two measurements is the only honest way to find out. The mean "
            "force is graded separately against `rho V g` at 5%"
        ),
    },
    "richardson": {
        "source": (
            "the three-point Richardson estimate, derived: with three solutions at a "
            "constant refinement ratio `q`, `f = f_exact + C h^p` gives "
            "`p = ln((f3 - f2)/(f2 - f1)) / ln q` and "
            "`f_exact = f1 + (f1 - f2)/(q^p - 1)`, solving for the order and the limit at "
            "once"
        ),
        "read": (
            "n/a -- derived, and hand-checked in `tests/test_live_sph_cases.py` against "
            "both a synthetic second-order triple and this repository's own recorded "
            "worked estimate: `PROGRESS_LIVE.md`'s lattice dam break at 325, 650 and 1300 "
            "cells reads 1.2450, 1.1837 and 1.1576, from which the document records an "
            "observed order of 1.232 and a limit of 1.1382, and this module's "
            "`richardson` reproduces both"
        ),
        "formula": "p = ln((f3-f2)/(f2-f1)) / ln q,  f_exact = f1 + (f1-f2)/(q^p - 1)",
        "caveat": (
            "**The two-grid form is circular and this repository has already published a "
            "circular version of this analysis.** `PROGRESS_LIVE.md` records it: a "
            "two-grid Richardson estimate needs the exact answer supplied, and feeding it "
            "the reference and reading back 'the limit is the reference' proves nothing -- "
            "it returned 1.0000 for three different cases, which is what gave it away. "
            "The three-point form needs no reference at all and that is why it is the one "
            "here; `richardson` takes three values and refuses two. Two further "
            "conditions have to hold and are checked rather than assumed: the refinement "
            "ratio must be **constant**, and the three values must be **monotone**, "
            "because a non-monotone triple has no `p` that fits it and the formula will "
            "return a number anyway"
        ),
        "tolerance": 0.5,
        "graded": True,
        "tolerance_is": (
            "**the observed order, graded at >= 0.5, and the extrapolated limit, graded "
            "against each quantity's own reference tolerance.** The order bound is a "
            "grading choice: SPH is formally second order in the kernel and first order "
            "at best with boundary particles and a free surface, and the number this "
            "repository has measured for the same quantity on the lattice is 1.232, so "
            "0.5 brackets 'converging slowly' and excludes 'not converging'. The limit is "
            "the number that matters and it is **not** graded with a widened tolerance: it "
            "is graded at the 15% the dam break states and the 2% the draft states, which "
            "is the honest reading `PROGRESS_LIVE.md` already gives for the lattice -- "
            "the grid-converged answer is inside the tolerance and the affordable grids "
            "are not"
        ),
    },
}


# -- the analytic references, which are the testable part of this module ----------------
#
# Every graded number in this file comes from one of these, and all of them are
# computable on paper. They are module-level functions rather than closures inside the
# runners for exactly one reason: `tests/test_live_sph_cases.py` can then check them at
# values that can be worked out by hand, with no solver, which is the whole of what is
# testable before the backend lands.


def ritter_depth(x: Any, elapsed: float, h0: float, gravity: float = GRAVITY) -> Any:
    """Ritter's dam-break depth profile at `elapsed`, on positions `x` measured from the dam.

    `h = (2 sqrt(g h0) - x/t)^2 / (9 g)` between the upstream edge of the depression wave
    at `x = -c0 t` and the front at `x = 2 c0 t`, undisturbed upstream of that and dry
    beyond it. Re-derived rather than quoted: it is the characteristics solution of the
    shallow-water equations on a dry horizontal bed, which is why it is safe to use
    without the 1892 paper (`REFERENCES["ritter"]["read"]` says so).

    At `x = 0` it is `4 h0 / 9` for every `t > 0`, independent of time, which is the one
    number in a dam break that needs neither a threshold nor a fit -- and is therefore
    what the case grades.
    """
    x = np.asarray(x, dtype=float)
    celerity = math.sqrt(gravity * h0)
    if elapsed <= 0:
        return np.where(x <= 0.0, h0, 0.0)
    inside = (x >= -celerity * elapsed) & (x <= 2.0 * celerity * elapsed)
    wave = (2.0 * celerity - x / elapsed) ** 2 / (9.0 * gravity)
    return np.where(inside, wave, np.where(x < -celerity * elapsed, h0, 0.0))


def ritter_front(elapsed: float, h0: float, gravity: float = GRAVITY) -> float:
    """Where Ritter's front is, in metres past the dam: `2 t sqrt(g h0)`.

    Computed so that the case can report a front position against it, and **not** so that
    it can grade one. `DECISIONS_LIVE.md` D14 is the decision and has the measured size of
    the problem: Ritter's own profile reaches half a lattice cell of depth at 0.168 m when
    this front is at 0.220 m, a 24% difference that belongs to the reference rather than
    to the solver, because the exact solution has no leading edge at all.
    """
    return 2.0 * elapsed * math.sqrt(gravity * h0)


def linear_sloshing(length: float, depth: float,
                    gravity: float = GRAVITY) -> dict[str, float]:
    """The first-mode standing wave in a rectangular tank: `omega^2 = g k tanh(k h)`.

    `k = pi / L` is the first mode of a tank with walls at both ends -- a half wavelength
    across the tank, which is the mode whose surface slope vanishes at both walls, the way
    a wall boundary condition wants. Linear water-wave theory; `benchmarks/sph/` uses the
    same formula and attributes it to Lamb's *Hydrodynamics* 6th edition article 228.

    Returns the wavenumber, `kh`, omega, the period, and the surface velocity amplitude
    per unit surface elevation -- `omega / tanh(k h)` -- which is what
    `run_sph_sloshing` needs, because the contract gives it no way to displace a free
    surface and it releases the mode through its velocity field instead.

    The two limits are worth having in a test: `kh` large gives the deep-water
    `T = 2 pi / sqrt(g k)`, and `kh` small gives the shallow-water `T = 2 L / sqrt(g h)`.
    """
    wavenumber = math.pi / length
    kh = wavenumber * depth
    omega = math.sqrt(gravity * wavenumber * math.tanh(kh))
    return {
        "wavenumber": wavenumber,
        "kh": kh,
        "omega": omega,
        "period": 2.0 * math.pi / omega,
        # u_surface = a g k / omega = a omega / tanh(kh), at the instant the surface is
        # flat and all the mode's energy is kinetic.
        "velocity_per_elevation": omega / math.tanh(kh),
        # The depth average of the mode's cosh(k(z+h))/cosh(kh) profile, which is what a
        # stack of uniform-velocity fill boxes can actually release.
        "depth_average": math.tanh(kh) / kh,
    }


def heave_period(draft: float, gravity: float = GRAVITY) -> float:
    """`T = 2 pi sqrt(d / g)`: a floating prism's undamped heave period, from the draft alone.

    The restoring stiffness of a prism is `rho g A` with `A` its waterplane area, and its
    mass is `rho_body A H`, which for a body floating at draft `d` is `rho A d`. The
    waterplane area cancels, so the period depends on nothing but the draft -- not on the
    beam, not on the density ratio except through `d`.

    `REFERENCES["heave"]` has what it neglects, which is added mass, and why the case
    grades the sign of the discrepancy exactly and bounds its size loosely.
    """
    return 2.0 * math.pi * math.sqrt(max(float(draft), 1e-12) / gravity)


def archimedes_draft(density_ratio: float, height: float) -> float:
    """`d = (rho_body / rho_water) H`. Archimedes, for a prism of uniform density."""
    return float(density_ratio) * float(height)


def metacentric_height(beam: float, height: float, density_ratio: float) -> dict[str, float]:
    """A wall-sided prism's metacentric height, and whether it floats upright at all.

    Three lengths above the keel: the centre of buoyancy `KB = d/2`, the centre of gravity
    `KG = H/2` (uniform density), and the metacentric radius `BM = I / V` with `I` the
    second moment of the waterplane about the centreline, `L b^3 / 12`, and `V = L b d`
    the displaced volume -- so `BM = b^2 / (12 d)`.

    `GM = KB + BM - KG` is the small-angle righting lever per radian of heel over the
    displacement. Positive and the box returns to upright; negative and it leaves. With
    `d = r H` that rearranges to a criterion in two dimensionless numbers and no lengths
    at all:

        stable upright  <=>  (b/H)^2 > 6 r (1 - r)

    which is worth writing down because it says a square-section prism at half water's
    density **capsizes**: `1 < 1.5`. The lattice case `run_archimedes` floats a 40 mm cube
    at exactly that density ratio and holds it square to the water, and this is the
    arithmetic for why it has to.

    Returns the three lengths, `GM`, the criterion's two sides, and `stable`.
    """
    beam, height, ratio = float(beam), float(height), float(density_ratio)
    draft = archimedes_draft(ratio, height)
    kb = draft / 2.0
    bm = beam * beam / (12.0 * draft) if draft > 0 else float("inf")
    kg = height / 2.0
    gm = kb + bm - kg
    return {
        "draft": draft, "KB": kb, "BM": bm, "KG": kg, "GM": gm,
        "slenderness": (beam / height) ** 2,
        "criterion": 6.0 * ratio * (1.0 - ratio),
        "stable": gm > 0.0,
    }


def heel_equilibrium(beam: float, height: float, density_ratio: float) -> dict[str, float]:
    """The heel angle a prism settles at when it cannot float upright, and whether it holds.

    For a **wall-sided** body -- one whose waterline stays on two vertical sides as it
    heels, so the deck does not immerse and the keel does not emerge -- the shift of the
    centre of buoyancy at heel `a` adds a `tan^2` term to the metacentric radius, and the
    righting lever is `GZ = sin(a) (GM + BM tan^2(a) / 2)`. Setting it to zero gives the
    heeled equilibrium

        tan^2(a_eq) = -2 GM / BM = 12 r (1 - r) (H/b)^2 - 2

    which exists exactly when the upright equilibrium does not: the right-hand side is
    positive precisely when `(b/H)^2 < 6 r (1-r)`, which is `metacentric_height`'s
    criterion with the inequality reversed. `tests/test_live_sph_cases.py` asserts that
    identity rather than trusting the algebra, and checks the two angles that come out in
    closed form: a square-section prism at half water's density heels to **exactly 45
    degrees**, and at a quarter (or three quarters) to **exactly `atan(1/2)`**, 26.565
    degrees.

    Wall-sidedness is a condition on the answer and not on the input, so it is returned
    rather than assumed: `tan(a) <= 2 min(d, H - d) / b`, and `margin` is how much room
    the equilibrium has inside it. An equilibrium outside that window is **not** this
    formula's and the case reports it as such instead of grading against it.

    Returns `angle` in radians (`nan` when the box floats upright), the limit, and the
    margin.
    """
    hydro = metacentric_height(beam, height, density_ratio)
    beam, height = float(beam), float(height)
    draft = hydro["draft"]
    tangent_squared = 12.0 * density_ratio * (1.0 - density_ratio) \
        * (height / beam) ** 2 - 2.0
    limit = 2.0 * min(draft, height - draft) / beam
    if tangent_squared <= 0.0:
        return {"angle": float("nan"), "tan": float("nan"), "wall_sided_limit": limit,
                "margin": float("nan"), "upright_stable": True, **hydro}
    tangent = math.sqrt(tangent_squared)
    return {
        "angle": math.atan(tangent), "tan": tangent, "wall_sided_limit": limit,
        "margin": limit - tangent, "upright_stable": False, **hydro,
    }


def richardson(values: Sequence[float], ratio: float) -> dict[str, float]:
    """The observed order and the extrapolated limit, from three solutions at a constant ratio.

    `values` is finest-first: `(f1, f2, f3)` at spacings `h, q h, q^2 h`. Then

        p = ln((f3 - f2) / (f2 - f1)) / ln q,    f_exact = f1 + (f1 - f2) / (q^p - 1)

    **Three points and not two, because the two-grid form is circular.**
    `PROGRESS_LIVE.md` records this repository publishing a circular version of exactly
    this analysis: a two-grid estimate needs the exact answer supplied, so feeding it the
    reference and reading back "the limit is the reference" proves nothing -- and it
    returned 1.0000 for three different cases, which is what gave it away. The three-point
    form solves for the order and the limit together and needs no reference at all, which
    is what makes the limit evidence.

    Two conditions are checked rather than assumed, because the formula returns a number
    either way. The ratio must be constant, which is the caller's business and is why this
    takes one `ratio` and not three spacings. And the three values must be **monotone**:
    a triple that turns around has no `p` that fits it, and `(f3 - f2)/(f2 - f1)` is then
    negative and the logarithm is a complex number that numpy will hand back as `nan`
    without comment. `monotone` is returned so the case can grade it.

    Returns `order`, `limit`, `monotone`, and the two differences the estimate rests on.
    """
    if len(values) != 3:
        raise UnphysicalScaling(
            f"a three-point Richardson estimate needs exactly three values at a constant "
            f"refinement ratio; got {len(values)}. The two-point form needs the exact "
            f"answer supplied, and PROGRESS_LIVE.md records what happens when the "
            f"reference is used for that: the limit comes back as the reference and the "
            f"estimate proves nothing"
        )
    if ratio <= 1.0:
        raise UnphysicalScaling(
            f"the refinement ratio is coarse/fine and must exceed 1; got {ratio}"
        )
    fine, middle, coarse = (float(v) for v in values)
    first, second = middle - fine, coarse - middle
    monotone = bool(first != 0.0 and second != 0.0
                    and (first > 0.0) == (second > 0.0))
    order = float("nan")
    limit = float("nan")
    if monotone:
        order = math.log(second / first) / math.log(ratio)
        if abs(ratio ** order - 1.0) > 1e-12:
            limit = fine + (fine - middle) / (ratio ** order - 1.0)
    return {"order": order, "limit": limit, "monotone": monotone,
            "first_difference": first, "second_difference": second}


def noise_fraction(signal: Sequence[float], window: int = 1) -> dict[str, float]:
    """The RMS fluctuation of a signal as a fraction of its mean, optionally after averaging.

    `window` is how many consecutive samples a controller would average before using the
    number: 1 is the raw per-step force and 10 is what a 100 Hz outer loop sees from a
    1 kHz solver. Reported at both, and the **measured** reduction between them is what
    the case prints rather than `sqrt(window)`: white noise falls as the square root of
    the window and correlated noise does not, and the ratio of the two measurements is the
    only honest way to find out which this is.

    The mean is the mean of the raw signal in both cases, so the two fractions are
    comparable; averaging does not change a mean.
    """
    series = np.asarray(signal, dtype=float)
    series = series[np.isfinite(series)]
    if series.size < 2:
        return {"mean": float("nan"), "rms": float("nan"), "fraction": float("nan"),
                "samples": int(series.size)}
    mean = float(series.mean())
    if window > 1 and series.size >= window:
        usable = (series.size // window) * window
        averaged = series[:usable].reshape(-1, window).mean(axis=1)
    else:
        averaged = series
    rms = float(np.sqrt(np.mean((averaged - mean) ** 2)))
    return {"mean": mean, "rms": rms,
            "fraction": rms / abs(mean) if mean else float("inf"),
            "samples": int(averaged.size)}


# -- the solver, and what this module needs from it that the contract does not promise --


NEEDS_FROM_THE_SOLVER: list[str] = [
    "**A simulated time or a timestep, readable.** `step()` returns `{'step', "
    "'substeps', 'seconds', 'particles', 'mlups'}` and `seconds` is wall clock. Every "
    "case here puts a measurement on a time axis -- a period, a front against `T = t "
    "sqrt(g/h0)`, a heave decay -- and none of them can without `dt`. `report()` is "
    "promised but it is lines of text. This module reads `world.dt` or `world.time` if "
    "either exists and **refuses with a named reason** if neither does, rather than "
    "guessing; it is the one gap that stops every case rather than one.",
    "**The column coordinates that go with `surface_height()`.** The contract promises "
    "'depth per column in metres, nan where dry' and not the array's shape, its axis "
    "order, or where the columns are. The dam break needs to know which column is at the "
    "dam, the sloshing tank needs the one at the -x wall, and the floating cases need "
    "the ones the hull is not standing in. `_columns` here assumes uniform bins spanning "
    "the domain with the two non-`axis` axes in order, and says so in the report; a "
    "promised `surface_columns(axis=2, bins=None)` returning the coordinates, or a "
    "documented shape, would remove the assumption.",
    "**Periodicity, or a documented substitute.** Every quasi-two-dimensional lattice "
    "case is three planes periodic in the thin direction (`DECISIONS_LIVE.md` D13), "
    "which is what makes the dam break and the sloshing tank affordable. The contract's "
    "`boundary_from_domain()` puts boundary particles 'on the domain's wall faces' and "
    "says nothing about a periodic one. Without it, a thin direction is a walled slot "
    "with friction on both sides of every particle -- the exact failure D13 was written "
    "for -- so these cases run a slot wide enough to be ignorable instead, and pay for "
    "it in particles. This is the difference between an M5 suite of a few thousand "
    "particles and one of tens of thousands.",
    "**A free-surface initialiser with an amplitude.** `freesurface.py` has "
    "`initialise_free_surface(level, amplitude, mode, along_axis)` and the contract has "
    "`fill_below(level)`. `run_sph_sloshing` therefore cannot displace the surface and "
    "releases the first mode through its **velocity** field instead, as a stack of "
    "`fill_box` slabs -- which is the same mode at a quarter period's phase shift, and is "
    "correct, but it projects imperfectly onto the mode and so excites a little of the "
    "higher ones. A `fill_below(level, amplitude=, mode=)` would be cleaner and the "
    "measurement would be quieter.",
    "**Whether abutting `fill_box` calls register on one lattice -- promised, not just "
    "true today.** The sloshing release above lays slabs of width exactly one particle "
    "spacing side by side, and if each box lattices from somewhere other than a common "
    "origin the seams leave gaps and the initial condition carries a sawtooth the FFT "
    "will find. **Measured on the current implementation and it is seamless**: 40 slabs "
    "of 2.5 mm across a 100 mm tank give 7,680 particles on 40 distinct x planes, "
    "identical to one `fill_box` over the whole tank, with the same minimum spacing. So "
    "this costs nothing now and the sloshing case silently depends on it; one sentence in "
    "the contract is the difference between a dependency and an assumption.",
    "**The speed of sound, or the stiffness the pressure comes from.** "
    "`benchmarks/README.md`'s `sph/` hydrostatic row turns on it: `dp/drho` is about "
    "`c0^2`, so a density error is `c0^2/(g h)` times itself as a fraction of the "
    "pressure being measured, which was a factor of 400 there. `density_error()` is "
    "promised and `c0` is not, so `run_sph_hydrostatic` can report both numbers and "
    "cannot connect them. A `report()` line is not enough; a reader needs the number.",
    "**A body's displaced volume as the solver sees it, promised rather than merely "
    "present.** The lattice cases grade the sampled volume against the analytic one "
    "(`benchmarks/README.md`'s falling-sphere row is explicit that a terminal velocity "
    "is a difference of two similar numbers and both must act on the same body). An SPH "
    "hull's Akinci boundary particles occupy a layer about one spacing thick, so its "
    "effective displacement is uncertain by roughly `dp` over the wetted surface. "
    "`run_sph_force_noise` reads a `volume(body)` through `getattr` and grades against "
    "it when it is there, falling back to the analytic volume with a one-layer bound -- "
    "one line in the contract turns that fallback into a grade.",
    "**`NewtonCoupling` accepting an `SphWorld` unchanged, which the contract promises "
    "and the coupling does not yet do. This is what blocks all four M6 cases, and it is "
    "measured rather than predicted**: `NewtonCoupling(model, engine, sph_world, "
    "buoyancy=False)` raises `AttributeError: SphWorld object has no attribute dx` from "
    "`couple.py::_static_shape_arrays`, which divides every shape scale by the lattice "
    "spacing. Everything on this module's side of that line works -- the Newton model "
    "builds, `world.add_body` and `add_shape(body=...)` sample the hull into boundary "
    "particles, and the two body indices line up -- so the four floating cases are "
    "written, importable and one coupling fix from running. Two smaller notes from "
    "getting that far: a shape's `position` is **body-local** and composes with the "
    "body's pose, so passing a world position puts the hull at twice it; and a shape can "
    "only be bound to a body the fluid world already owns, so `world.add_body` has to "
    "come first even when Newton owns the dynamics. If the coupling grows its own shape "
    "hand-off, `_float_rig`'s `add_shape` call becomes a duplicate hull and must come "
    "out: it is the one place in these cases written to be deleted.",
]
"""What the cases want that `docs/sph-contract.md` does not promise, in the order it costs.

Kept as data rather than prose so that the agent tool and the report can print it, and so
that a line can be deleted when the contract grows one. The first entry stops every case
in this file; the rest cost accuracy or particles.
"""


def _sph():
    """The SPH backend module, or a refusal that says it is not written yet.

    These cases were written against `docs/sph-contract.md` **before** the solver, which
    is what that document is for. So the import failing is an expected state and not a
    broken install, and it says which of the two it is.
    """
    try:
        from . import sph
    except ImportError as exc:
        raise UnphysicalScaling(
            f"the SPH backend is not importable: openreynolds/live/sph.py is written to "
            f"docs/sph-contract.md and these cases were written against that contract "
            f"before it landed. If the file is there, this is an install problem and the "
            f"import said: {exc}"
        ) from exc
    return sph


def _timestep(world: Any) -> float:
    """The simulated timestep, in seconds, or a refusal naming the contract gap.

    `docs/sph-contract.md` promises `step()` returns `seconds`, which is wall clock, and
    promises no simulated time at all. Every measurement in this file lives on a time
    axis, so this is the one gap that stops every case rather than one of them, and it is
    `NEEDS_FROM_THE_SOLVER`'s first entry.

    Refuses rather than guessing. A guessed timestep makes every period, every front speed
    and every decay rate in this module wrong by the same unknown factor, and all of them
    would still look plausible.
    """
    for name in ("dt", "timestep", "time_step"):
        value = getattr(world, name, None)
        if value:
            return float(value)
    raise UnphysicalScaling(
        "this world reports no simulated timestep. `docs/sph-contract.md` promises "
        "`step()` -> {'step', 'substeps', 'seconds', 'particles', 'mlups'} where "
        "`seconds` is wall clock, and promises no `dt` and no simulated time -- so a "
        "period, a front position against `T = t sqrt(g/h0)` and a heave decay cannot be "
        "put on a time axis at all. Expose `world.dt` (or a `time` in `step()`'s "
        "dictionary) and every case in openreynolds/live/sph_cases.py runs. This refuses "
        "rather than assuming a value, because a guessed timestep would make every "
        "measurement here wrong by one unknown factor and leave all of them looking "
        "plausible"
    )


def _elapsed(world: Any, dt: float, steps_done: int) -> float:
    """Simulated time: the solver's own if it keeps one, otherwise `steps * dt`."""
    clock = getattr(world, "time", None)
    if clock:
        return float(clock)
    return steps_done * dt


def _advance(world: Any, steps: int, *, sample_every: int = 0,
             sample: Callable[[int, float], None] | None = None,
             settle: float = 0.0, progress: Callable[[str], None] | None = None,
             every: int = 0) -> dict[str, Any]:
    """Step the world `steps` times, sampling and reporting on the way.

    `cases.py::_advance` is not reused and the reason is in the report it prints: it reads
    `world.lattice_steps` and `world.nodes` to turn a loop index into a cell-update rate,
    and an SPH world has neither. The structure is the same one for the same reason --
    "how a case is run" should be one thing rather than eight -- and `settle` means the
    same thing: the fraction of the run discarded before sampling, because a case that
    samples its own transient and then averages over it reports the average of a flow
    that was still arriving.

    `sample` is called with `(index, elapsed)` rather than the index alone, so that no
    runner has to keep its own clock. The rate printed is particles a second from the
    contract's `mlups`, which it names for symmetry with the lattice backend and which
    `step()`'s own report says is particle updates and not cell updates.
    """
    dt = _timestep(world)
    start_sampling = int(steps * settle)
    announce = every or max(1, steps // 10)
    began = time.perf_counter()
    last: dict[str, Any] = {}
    for index in range(steps):
        last = world.step() or {}
        if sample is not None and sample_every and index >= start_sampling \
                and (index - start_sampling) % sample_every == 0:
            sample(index, _elapsed(world, dt, index + 1))
        if progress is not None and (index + 1) % announce == 0:
            spent = time.perf_counter() - began
            left = spent / (index + 1) * (steps - index - 1)
            progress(f"    {index + 1:>9,}/{steps:,} steps  "
                     f"t = {_elapsed(world, dt, index + 1) * 1e3:8.1f} ms  "
                     f"{last.get('mlups', float('nan')):6.2f} M particle-updates/s  "
                     f"{figures.clock(left)} left")
    return {"steps": steps, "dt": dt, "elapsed": _elapsed(world, dt, steps),
            "last": last}


def _columns(world: Any, surface: Any, axis: int = 2) -> dict[str, Any]:
    """The coordinates of `surface_height`'s columns, and the assumption that takes.

    **This is an assumption and not a promise, and it is reported as one.**
    `docs/sph-contract.md` says `surface_height(axis=2, bins=None)` returns "depth per
    column in metres, `nan` where dry" and says nothing about the array's shape, its axis
    order, or where in the domain the columns sit. So this assumes the bins span the
    domain uniformly, with the two axes other than `axis` in increasing order, and derives
    the centres from the domain and the array's own shape.

    It handles a one-dimensional return as well, because a backend that collapses the
    second horizontal axis would be a reasonable reading of the same sentence. Either way
    the case prints what it assumed; `NEEDS_FROM_THE_SOLVER` asks for the coordinates to
    be promised instead.
    """
    grid = np.asarray(surface, dtype=float)
    others = [index for index in range(3) if index != axis]
    origin, size = world.domain.origin, world.domain.size
    axes = []
    for position, count in zip(others, grid.shape):
        step = size[position] / count
        axes.append(origin[position] + (np.arange(count) + 0.5) * step)
    return {"grid": grid, "axes": axes, "first_axis": others[0],
            "assumed": (f"surface_height returned {grid.shape}; the columns are assumed "
                        f"uniform over the domain with axes "
                        f"{'/'.join('xyz'[i] for i in others[:grid.ndim])} in order")}


def _profile(world: Any, axis: int = 2) -> tuple[Any, Any]:
    """Mean depth per column along the first horizontal axis, and that axis' coordinates.

    What a dam break and a sloshing tank both want: a one-dimensional depth profile.
    Averaged over the second horizontal axis rather than sliced at its middle, because a
    slot a few particles wide has only a few columns across it and one of them is noise.

    **The average is over the wet bins only, and that is not cosmetic.** A dry column
    comes back as `nan`, and the lattice twin of this function replaces `nan` with zero
    before averaging because every lattice column exists. A particle backend's bins do
    not: `surface_height` bins the two horizontal axes at the same count, so a long thin
    slot gets as many bins across its 30 mm as along its 650 mm, most of which hold no
    particle at all. Averaging zeros over those divides the depth by the ratio of the two
    extents -- a factor of twenty on the dam break -- and the profile comes out uniformly,
    believably wrong.
    """
    columns = _columns(world, world.surface_height(axis=axis), axis=axis)
    grid = columns["grid"]
    if grid.ndim == 1:
        return np.nan_to_num(grid), columns["axes"][0]
    wet = np.isfinite(grid)
    counts = wet.sum(axis=1)
    totals = np.where(wet, grid, 0.0).sum(axis=1)
    depth = np.divide(totals, counts, out=np.zeros_like(totals, dtype=float),
                      where=counts > 0)
    return depth, columns["axes"][0]


def _front(world: Any, spacing: float, threshold: float = 0.5,
           axis: int = 2) -> float:
    """The furthest the water has reached along x, by a depth threshold in particle spacings.

    **The threshold is this measurement's weak point and is reported rather than hidden**,
    which is `DECISIONS_LIVE.md` D14 and `cases.py::_front_position` saying the same thing
    about the same quantity: Ritter's front thins to zero depth, so "where the front is"
    is entirely a question of how little water still counts as water. The case reports a
    quarter, a half and one full spacing so the sensitivity is on the page, and grades
    none of them.
    """
    depth, coordinate = _profile(world, axis=axis)
    wet = depth > threshold * spacing
    if not wet.any():
        return float("nan")
    last = int(np.flatnonzero(wet)[-1])
    step = coordinate[1] - coordinate[0] if len(coordinate) > 1 else spacing
    return float(coordinate[last] + step / 2.0)


def _settle(world: Any, *, reference_speed: float, tolerance: float = 0.02,
            max_steps: int = 40_000, check_every: int = 100,
            step: Callable[[], Any] | None = None,
            hold: Callable[[], None] | None = None,
            progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Step until the fluid stops moving, and report what it stopped at.

    **Decided on the fluid's own velocity, never on a step count**, which is the first of
    `run_archimedes`' three fixes and is the one that cost the most: `PROGRESS_LIVE.md`
    records that case taking 3,000 steps on trust and releasing a hull into a tank still
    moving at 50 to 90 mm/s, with the vertical force on a *pinned* hull swinging between
    0.2 and 1.7 times its weight. The draft it then reported was an average of that
    transient and did not reproduce between identical runs. No fixed step count can be
    known in advance to be enough.

    `reference_speed` is the gravity-wave speed `sqrt(g h)`, which is the only speed a
    still tank has; `tolerance` of it is the bound.

    **`step` is a parameter and that is the second of `run_archimedes`' lessons.** A case
    with a body in it must settle through `link.step`, not `world.step`: the world's own
    step does not sync poses from Newton, and settling that way put the lattice case's
    hull at the identity transform, voxelised into the corner of the domain, for long
    enough to produce four retracted measurements. `hold` then pins the pose and velocity
    after each step, so the hull stays where it was put while the water quietens.

    Returns whether it settled and what the velocity reached, both of which the cases
    **grade**, because a case that did not settle has not measured an equilibrium.
    """
    dt = _timestep(world)
    advance = step if step is not None else world.step
    reached = float("inf")
    done = 0
    bound = tolerance * abs(reference_speed)
    while done < max_steps:
        for _ in range(check_every):
            advance()
            if hold is not None:
                hold()
        done += check_every
        reached = float(np.nanmax(np.abs(world.velocities())))
        if progress is not None:
            progress(f"    settling {done:,}/{max_steps:,}  |u|max "
                     f"{reached * 1e3:7.3f} mm/s against a bound of {bound * 1e3:.3f}")
        if reached <= bound:
            return {"settled": True, "reached": reached, "steps": done, "bound": bound,
                    "seconds": done * dt}
    return {"settled": False, "reached": reached, "steps": done, "bound": bound,
            "seconds": done * dt}


def _health(world: Any, advanced: dict[str, Any],
            started_with: int) -> tuple[list[Check], list[str]]:
    """The checks every SPH case carries, and the lines every one of them prints.

    `docs/sph-contract.md` nominates `density_error()` as "the health check every case
    reports", and a particle method has one more invariant that is free to check: the
    particle count. A particle that leaked through a wall is mass that left the problem,
    and unlike the lattice's mass drift it is exactly countable -- there is no partially
    filled particle.

    Returned rather than appended in place so that each case decides where in its own
    check list they go, which is how `report` orders what a reader sees first.
    """
    error = world.density_error()
    particles = int(advanced.get("last", {}).get("particles", started_with))
    checks = [
        Check("density error, max", error["max"], "<= 1% of the rest density",
              error["max"] <= 0.01,
              "the health check docs/sph-contract.md nominates. It matters more than it "
              "looks: `dp/drho` is about `c0^2`, so a density error is `c0^2/(g h)` times "
              "itself as a fraction of the hydrostatic pressure being measured -- a "
              "factor of 400 in benchmarks/README.md's `sph/` hydrostatic row, which read "
              "67% pressure RMS before a density-diffusion term was added"),
        Check("density error, mean", error["mean"], "<= 0.1% of the rest density",
              error["mean"] <= 1.0e-3,
              "graded an order of magnitude tighter than the max, because the max is one "
              "particle and the mean is the field. A free surface truncates the kernel "
              "support, so the max lives at the surface by construction and the mean is "
              "what says the interior is right"),
        Check("particles at the end", particles, f"exactly {started_with:,}",
              particles == started_with,
              "a particle method's mass conservation is a count, with no partially filled "
              "particle to argue about: a particle fewer is mass that left through a wall"),
    ]
    lines = [
        f"  density:    max {error['max']:.3%}, mean {error['mean']:.3%} of rest density",
        f"  particles:  {particles:,} ({started_with:,} at the start)",
        f"  steps:      {advanced['steps']:,} at dt {advanced['dt'] * 1e6:.2f} us "
        f"= {advanced['elapsed'] * 1e3:.2f} ms simulated",
    ]
    return checks, lines


def _sph_lines(*, radius: float, particles: int, length: float,
               depth: float, gravity: float = GRAVITY) -> list[str]:
    """The numbers every SPH case report must carry, every time.

    The sizing knob and what it bought: the particle radius, the spacing it implies, the
    count, and the two dimensionless numbers a free-surface case lives or dies by -- the
    Froude number on the case's own length and how many particles are across its depth.

    `cases.py::_free_surface_lines` is the lattice twin of this and exists for the stated
    reason that a case must print the Froude number it matched and the Reynolds number it
    did not, rather than leaving a reader to assume. The same applies here with one
    substitution: the lattice's compromise is viscosity, and SPH's is resolution.
    """
    spacing = 2.0 * radius
    celerity = math.sqrt(gravity * max(depth, 1e-12))
    return [
        f"  particles:  radius {radius * 1e3:.3f} mm, spacing {spacing * 1e3:.3f} mm, "
        f"{particles:,} fluid particles",
        f"  resolution: {depth / spacing:.1f} particle spacings across {depth * 1e3:g} mm "
        f"of depth; the smoothing length is the implementer's choice and is in "
        f"world.report()",
        f"  froude:     gravity-wave speed sqrt(g h) = {celerity:.4f} m/s on "
        f"{depth * 1e3:g} mm, {celerity * math.sqrt(depth / gravity) / max(length, 1e-12):.4f} "
        f"on {length * 1e3:g} mm of length",
        f"  surface:    surface_tension = 0, deliberately: at these scales a few mN/m "
        f"changes the dispersion relation the sloshing case grades against, and the "
        f"contract defaults it off",
    ]


def _estimate(volume: float, radius: float) -> int:
    """Roughly how many particles a volume of water holds at this radius, for a report line."""
    spacing = 2.0 * radius
    return int(volume / (spacing ** 3))


# ======================================================================================
# M5: the free surface
# ======================================================================================


def run_sph_hydrostatic(*, particle_radius: float = 0.0025, depth: float = 0.1,
                        plan: float = 0.05, freeboard: float = 0.02,
                        steps: int = 0, device: str | None = None,
                        solver: str = "dfsph", out_dir: str | Path = ".",
                        progress: Callable[[str], None] | None = None,
                        **_: Any) -> dict[str, Any]:
    """A still column of water: pressure against depth, and the density error over time.

    The first thing to ask of a particle free surface and the one a floating body depends
    on. `p = rho g (h - z)` is exact, and Archimedes is nothing but that pressure
    integrated over a hull -- so a solver that gets a draft right while getting this wrong
    gets it right by accident. `cases.py::run_hydrostatic` says the same sentence about
    the same quantity on the lattice, where the measurable is `d(rho)/dz` against
    `-rho g / c_s^2`; here the contract promises `pressures()` directly, so the pressure
    is compared with the pressure.

    **Two graded quantities and one of them is in the contract.**

    * The **pressure profile**: the RMS of `p - rho g (h - z)` over every fluid particle,
      as a fraction of `rho g h`, at 5%. The worst particle is reported with its depth
      below the surface and is **not** graded, which is a decision taken from a
      measurement rather than a preference: `benchmarks/README.md`'s `sph/` hydrostatic
      row records the worst particle sitting at the free surface once settled, 17.4 mm
      below it on a 100 mm tank, where the kernel support is truncated and there is no
      fluid above to sum. The grader measures where the worst particle is instead of
      asserting that it is allowed to be there.
    * The **density error over time**, max and mean, which the contract nominates as the
      health check and which this case is the one place to watch over a whole run rather
      than at the end. It is reported at every sample and graded on its worst sample, not
      its last: a solver whose density error grows and then relaxes has a transient the
      floating cases will release a hull into.

    The column is also graded on **staying still** -- `|u|max` under 2% of the
    gravity-wave speed -- for the reason `_settle` has at length: still water that is not
    still is not hydrostatic, and every later case in this file releases something into a
    tank this case says is quiet.
    """
    sph = _sph()
    from .units import FLUIDS
    from .world import Domain

    water = FLUIDS["water"]
    domain = Domain(origin=(0.0, 0.0, 0.0), size=(plan, plan, depth + freeboard),
                    faces="wall")
    world = sph.SphWorld(domain=domain, particle_radius=particle_radius, fluid="water",
                         gravity=(0.0, 0.0, -GRAVITY), surface_tension=0.0,
                         device=device, solver=solver)
    world.boundary_from_domain()
    world.fill_below(depth, axis=2)
    started_with = len(world.positions())
    celerity = math.sqrt(GRAVITY * depth)

    total = int(steps if steps else 4000)
    history: list[tuple[float, float, float, float]] = []

    def take(index: int, elapsed: float) -> None:
        error = world.density_error()
        speed = float(np.nanmax(np.abs(world.velocities())))
        history.append((elapsed, float(error["max"]), float(error["mean"]), speed))

    advanced = _advance(world, total, sample_every=max(1, total // 120), sample=take,
                        progress=progress)

    positions = np.asarray(world.positions(), dtype=float)
    pressure = np.asarray(world.pressures(), dtype=float)
    # The surface the pressure is measured from is the measured one, not the fill level:
    # an SPH column settles a little under its own weight exactly as the weakly
    # compressible lattice does, and `cases.py::run_hydrostatic` predicts and checks that
    # deficit rather than absorbing it. Here the top of the particle distribution is
    # directly readable, which is simpler and needs no sound speed -- and the sound speed
    # is the one number this case would like and the contract does not promise.
    # **Plus one particle radius, which is not a fudge.** A particle's centre sits half a
    # spacing below the surface it represents, so a surface read off the particle heights
    # is systematically `r` low -- and `rho g r` is 6.7% of `rho g h` at four spacings
    # through a 60 mm column, which would be read as a solver error. `sph.surface_height`
    # makes the same correction for the same reason and says so.
    surface = float(np.quantile(positions[:, 2], 0.999)) + particle_radius
    exact = water.rho * GRAVITY * np.clip(surface - positions[:, 2], 0.0, None)
    scale = water.rho * GRAVITY * depth
    residual = (pressure - exact) / scale
    rms = float(np.sqrt(np.mean(residual ** 2)))
    worst = int(np.argmax(np.abs(residual)))
    worst_depth = surface - positions[worst, 2]
    speed = float(np.nanmax(np.abs(world.velocities())))
    peak_max = max((row[1] for row in history), default=float("nan"))
    peak_mean = max((row[2] for row in history), default=float("nan"))

    limit = REFERENCES["hydrostatic"]["tolerance"]
    health, health_lines = _health(world, advanced, started_with)
    checks = [
        Check("pressure against rho g h, RMS", rms,
              f"<= {limit:.0%} of rho g h ({scale:.2f} Pa)", rms <= limit,
              f"over all {started_with:,} fluid particles, with the surface taken at the "
              f"0.999 quantile of the particle heights rather than at the fill level: a "
              f"column settles slightly under its own weight. benchmarks/README.md's "
              f"`sph/` row measures 2.50% for this repository's toolbox SPH, so this "
              f"tolerance should be tightened to the measurement once this backend has "
              f"one"),
        Check("worst particle", abs(float(residual[worst])),
              "reported, not graded", None,
              f"{abs(float(residual[worst])):.2%} of rho g h, "
              f"{worst_depth * 1e3:.1f} mm below the surface on a {depth * 1e3:g} mm "
              f"column. **Reported with its position rather than graded**, because the "
              f"kernel support is truncated at a free surface and there is no fluid above "
              f"it to sum: the worst particle is a statement about where the surface is. "
              f"benchmarks/README.md's `sph/` row measures 8.61% at 17.4 mm on the same "
              f"geometry and the same reasoning"),
        Check("density error over the whole run, worst sample",
              f"max {peak_max:.3%}, mean {peak_mean:.3%}",
              "<= 1% / 0.1% at every sample, not just the last",
              bool(peak_max <= 0.01 and peak_mean <= 1.0e-3),
              "graded on the worst sample of the run and not on its end state. A solver "
              "whose density error grows and then relaxes has a transient, and every "
              "floating case in this file releases a hull into a tank this case called "
              "quiet"),
        Check("the column stayed still", speed / celerity,
              "<= 2% of the gravity-wave speed", speed / celerity <= 0.02,
              f"|u|max {speed * 1e3:.3f} mm/s against sqrt(g h) = {celerity:.3f} m/s. "
              f"Still water that is not still is not hydrostatic, and this is the bound "
              f"every later case settles to before it releases anything"),
        *health,
        Check("the pressure-to-density amplification", "c0 not promised",
              "reported, not graded", None,
              "the number this case would most like and cannot compute. A density error "
              "`e` is a pressure error `e c0^2`, which is `e c0^2/(g h)` as a fraction of "
              "what is being measured here -- 400 in benchmarks/README.md's `sph/` row. "
              "`density_error()` is promised and `c0` is not, so the two graded numbers "
              "above cannot be connected to each other"),
    ]

    out = Path(out_dir)
    order = np.argsort(positions[:, 2])
    drawn = [
        figures.profile_figure(
            out / "live-sph-hydrostatic-pressure.png",
            [{"x": (pressure[order] * 1e-3).tolist(),
              "y": (positions[order, 2] * 1e3).tolist(),
              "label": "particle pressure", "style": "o"},
             {"x": (exact[order] * 1e-3).tolist(),
              "y": (positions[order, 2] * 1e3).tolist(),
              "label": "rho g (h - z)"}],
            title=f"hydrostatic: RMS {rms:.2%} of rho g h, worst {abs(float(residual[worst])):.2%} "
                  f"at {worst_depth * 1e3:.1f} mm depth",
            xlabel="pressure (kPa)", ylabel="height (mm)"),
        figures.profile_figure(
            out / "live-sph-hydrostatic-density.png",
            [{"x": [row[0] * 1e3 for row in history],
              "y": [row[1] * 100.0 for row in history], "label": "max density error (%)"},
             {"x": [row[0] * 1e3 for row in history],
              "y": [row[2] * 100.0 for row in history], "label": "mean density error (%)"},
             {"x": [row[0] * 1e3 for row in history],
              "y": [row[3] * 1e3 for row in history], "label": "|u|max (mm/s)",
              "panel": 1}],
            title=f"density error over {advanced['elapsed'] * 1e3:.0f} ms",
            xlabel="t (ms)", ylabel="per cent / mm per s", panels=2),
    ]
    for figure in drawn:
        figures.ensure_rgb(figure)

    lines = [f"# sph hydrostatic: {depth * 1e3:g} mm of water in a {plan * 1e3:g} mm "
             f"square tank",
             *_sph_lines(radius=particle_radius, particles=started_with,
                         length=plan, depth=depth),
             f"  pressure:   RMS {rms:.3%} of rho g h = {scale:.2f} Pa; worst particle "
             f"{abs(float(residual[worst])):.3%} at {worst_depth * 1e3:.2f} mm depth",
             f"  surface:    measured at {surface * 1e3:.3f} mm against a fill level of "
             f"{depth * 1e3:g} mm",
             *health_lines,
             f"  reference:  {REFERENCES['hydrostatic']['source']}",
             *world.report()]
    return _result("sph_hydrostatic", checks, lines, drawn, pressure_rms=rms,
                   worst=float(residual[worst]), worst_depth=worst_depth,
                   surface=surface, history=[list(row) for row in history])


def run_sph_dam_break(*, particle_radius: float = 0.0025, height: float = 0.05,
                      reservoir: float = 4.0, slot: float = 0.03, tank: float = 0.0,
                      graded_from: float = 2.0, graded_to: float = 4.0,
                      steps: int = 0, device: str | None = None, solver: str = "dfsph",
                      out_dir: str | Path = ".",
                      progress: Callable[[str], None] | None = None,
                      **_: Any) -> dict[str, Any]:
    """A column of water collapses: the front over time, graded against Ritter (1892).

    **Why this is not graded against Martin and Moyce, which is what the addendum asks
    for.** `benchmarks/README.md`'s `sph/` dam-break row and `DECISIONS_LIVE.md` D14 both
    record that this repository **could not read** Martin and Moyce (1952) -- the
    publisher refuses the request -- and that the two non-dimensionalisations in
    circulation, `t sqrt(g/a)` and `t sqrt(2g/a)`, **differ by a factor of 1.414 at every
    point**. A front position quoted in the wrong one agrees with a solver that is 41%
    wrong. Quoting a number through a secondary source is the error this repository argues
    hardest against, so the comparison is **computed and printed in both conventions and
    graded in neither**, and what is graded is Ritter's closed form, which is three lines
    of characteristics on a dry bed and needs no paper. The existing lattice case made
    exactly this choice and this one inherits it rather than re-deciding it.

    **The front position is reported and graded nowhere, which is also D14.** The addendum
    asks for "front position over time" and it is here, at three depth thresholds, with
    its fitted `dX/dT` against Ritter's 2.0 -- but Ritter's front is a tongue that thins
    to zero depth, so where it is depends entirely on how little water still counts as
    water. D14 has the measured size of that: the place where Ritter's own profile reaches
    half a lattice cell sits at 0.168 m when the true front is at 0.220 m, a 24%
    difference belonging to the reference and not to the solver. What is graded is the
    depth at the dam, which Ritter fixes at exactly `4 h0 / 9` independent of time and
    which needs neither a threshold nor a fit, and the RMS of the profile where Ritter is
    thicker than two particle spacings.

    **The reservoir is four depths wide, inherited as a measurement.** D14: a square
    column cannot be graded against Ritter at all, because shallow-water theory needs the
    hydrostatic assumption that a vertically accelerating collapse violates (2.07 times
    Ritter's depth at the dam at `T = 0.5`), and its depression wave returns to the dam at
    `T = 2 a/h0` -- so for a square column Ritter expires exactly when the agreement
    arrives. Four depths buys a window of `T` in [2, 8] and the case grades inside it.

    **The thin direction is a walled slot and that costs particles.** Every
    quasi-two-dimensional lattice case is three planes periodic in the thin direction
    (`DECISIONS_LIVE.md` D13), and the contract promises SPH no periodicity -- so a thin
    slot here would be a slot with wall friction on both sides of every particle, which is
    the exact failure D13 was written for (a sloshing tank that did not slosh). The slot
    is therefore `slot` metres wide, several smoothing lengths, and the profile is averaged
    across it. `NEEDS_FROM_THE_SOLVER` asks for periodicity and says what it is worth.
    """
    sph = _sph()
    from .world import Domain

    spacing = 2.0 * particle_radius
    dam = reservoir * height
    # Long enough that the front has not reached the far wall inside the graded window:
    # Ritter's front is at `2 T h0` past the dam, so `T = 4` needs eight depths.
    length = tank if tank > 0 else dam + 9.0 * height
    domain = Domain(origin=(0.0, 0.0, 0.0), size=(length, slot, 2.0 * height),
                    faces="wall")
    world = sph.SphWorld(domain=domain, particle_radius=particle_radius, fluid="water",
                         gravity=(0.0, 0.0, -GRAVITY), surface_tension=0.0,
                         device=device, solver=solver)
    world.boundary_from_domain()
    world.fill_box((0.0, 0.0, 0.0), (dam, slot, height))
    started_with = len(world.positions())

    celerity = math.sqrt(GRAVITY * height)
    time_scale = math.sqrt(height / GRAVITY)
    exact_at_dam = 4.0 * height / 9.0
    total = int(steps if steps else math.ceil(graded_to * time_scale
                                              / _timestep(world)))

    track: list[tuple[float, float, float, float]] = []
    samples: list[tuple[float, float, float]] = []
    last_profile: dict[str, Any] = {}

    def take(index: int, elapsed: float) -> None:
        track.append((elapsed,
                      _front(world, spacing, 0.25), _front(world, spacing, 0.5),
                      _front(world, spacing, 1.0)))
        depth, coordinate = _profile(world)
        from_dam = coordinate - dam
        exact = ritter_depth(from_dam, elapsed, height)
        usable = exact > 2.0 * spacing
        last_profile.update({"depth": depth, "x": from_dam, "exact": exact,
                             "elapsed": elapsed})
        if usable.sum() > 4 and elapsed >= graded_from * time_scale:
            samples.append((
                elapsed / time_scale,
                float(np.interp(0.0, from_dam, depth)),
                float(np.sqrt(((depth[usable] - exact[usable]) ** 2).mean()))))

    advanced = _advance(world, total, sample_every=max(1, total // 80), sample=take,
                        progress=progress)

    times = np.array([row[0] for row in track]) / time_scale
    fronts = (np.array([row[2] for row in track]) - dam) / height
    window = (times >= graded_from) & (times <= graded_to) & np.isfinite(fronts)
    slope = (float(np.polyfit(times[window], fronts[window], 1)[0])
             if window.sum() >= 3 else float("nan"))
    at_dam = float(np.mean([row[1] for row in samples])) if samples else float("nan")
    profile_rms = float(np.mean([row[2] for row in samples])) if samples else float("nan")
    limit = REFERENCES["ritter"]["tolerance"]

    # Both conventions, computed and graded in neither. REFERENCES["martin_moyce"] is why.
    conventions = {
        "T = t sqrt(g/a)": float(times[-1] * math.sqrt(height / dam)) if len(times)
        else float("nan"),
        "T = t sqrt(2g/a)": float(times[-1] * math.sqrt(2.0 * height / dam))
        if len(times) else float("nan"),
        "ratio": math.sqrt(2.0),
    }

    health, health_lines = _health(world, advanced, started_with)
    checks = [
        Check("depth at the dam against Ritter's 4 h0 / 9", at_dam * 1e3,
              f"{exact_at_dam * 1e3:.4f} mm +/- {limit:.0%}",
              abs(at_dam / exact_at_dam - 1.0) <= limit if np.isfinite(at_dam) else False,
              f"Ritter fixes the depth at the dam at exactly `4 h0 / 9` for as long as "
              f"the solution holds, independent of time -- which makes it the one number "
              f"in a dam break needing neither a threshold nor a fit. Averaged over T in "
              f"[{graded_from:g}, {graded_to:g}], the window where the shallow-water "
              f"assumption holds and the back wall has not answered yet (the reservoir is "
              f"{reservoir:g} depths, so the reflection reaches the dam at "
              f"T = {2 * reservoir:g})"),
        Check("depth profile against Ritter, RMS", profile_rms / height,
              f"<= {limit:.0%} of h0",
              profile_rms / height <= limit if np.isfinite(profile_rms) else False,
              "over the columns where Ritter's own depth exceeds two particle spacings, "
              "so the comparison is not dominated by the vanishing tongue near the front "
              "where Ritter tends to zero and no discretisation can follow it"),
        Check("front speed, dX/dT", slope, "2.0 (Ritter), reported, not graded", None,
              f"`x_front = 2 t sqrt(g h0)`, fitted over T in [{graded_from:g}, "
              f"{graded_to:g}]. **Not graded, and DECISIONS_LIVE.md D14 is the decision**: "
              f"Ritter's front has zero depth, so the measured front is wherever the "
              f"threshold is put. D14 has the size of it -- Ritter's own profile reaches "
              f"half a cell at 0.168 m when the front is at 0.220"),
        Check("front position at a quarter, a half and one spacing of depth",
              f"{track[-1][1] * 1e3:.2f} / {track[-1][2] * 1e3:.2f} / "
              f"{track[-1][3] * 1e3:.2f} mm" if track else "n/a",
              "reported, not graded", None,
              f"the answer the addendum asks for, at the three thresholds that could "
              f"define it, at T = {times[-1]:.2f} with Ritter's front at "
              f"{(dam + ritter_front(times[-1] * time_scale, height)) * 1e3:.2f} mm. The "
              f"spread between them is the honest width of this measurement"),
        Check("non-dimensional time, both Martin and Moyce conventions", conventions,
              "reported, not graded", None,
              "the paper could not be read -- the publisher refuses the request -- and "
              "`t sqrt(g/a)` and `t sqrt(2g/a)` differ by 1.414 at every point, so there "
              "is no reference value to grade against, only an unverified choice between "
              "two. Nor is it known whether the paper's `a` is the column's width or its "
              "half-width, which is another factor of sqrt(2) on top: `a` is taken as the "
              "full reservoir width here and that choice is reported rather than claimed. "
              "benchmarks/README.md records the same refusal for this repository's "
              "other SPH dam break and DECISIONS_LIVE.md D14 is the decision"),
        *health,
    ]

    out = Path(out_dir)
    drawn = [
        figures.profile_figure(
            out / "live-sph-dam-break-front.png",
            [{"x": times.tolist(), "y": fronts.tolist(),
              "label": "measured front (half a spacing of depth)", "style": "o"},
             {"x": [0.0, float(times.max()) if len(times) else 2.0],
              "y": [0.0, 2.0 * (float(times.max()) if len(times) else 2.0)],
              "label": "Ritter, X = 2T", "style": "--"}],
            title=f"dam break: dX/dT {slope:.3f} against Ritter's 2.0 (reported, not graded)",
            xlabel="T = t sqrt(g/h0)", ylabel="X = (x_front - a) / h0"),
        figures.profile_figure(
            out / "live-sph-dam-break-profile.png",
            [{"x": (last_profile.get("x", np.zeros(1)) / height).tolist(),
              "y": (last_profile.get("depth", np.zeros(1)) / height).tolist(),
              "label": "measured depth", "style": "o"},
             {"x": (last_profile.get("x", np.zeros(1)) / height).tolist(),
              "y": (last_profile.get("exact", np.zeros(1)) / height).tolist(),
              "label": "Ritter"}],
            title=f"depth profile at T = {last_profile.get('elapsed', 0.0) / time_scale:.2f}: "
                  f"h(dam) {at_dam / exact_at_dam:.4f}x, RMS {profile_rms / height:.2%} of h0",
            xlabel="(x - a) / h0", ylabel="h / h0"),
    ]
    for figure in drawn:
        figures.ensure_rgb(figure)

    lines = [f"# sph dam break: {height * 1e3:g} mm deep, {reservoir:g} depths wide, in a "
             f"{length * 1e3:g} mm tank",
             *_sph_lines(radius=particle_radius, particles=started_with,
                         length=length, depth=height),
             f"  celerity:   sqrt(g h0) = {celerity:.4f} m/s, so Ritter's front runs at "
             f"{2 * celerity:.4f} m/s",
             f"  measured:   h(dam) {at_dam * 1e3:.4f} mm against {exact_at_dam * 1e3:.4f} "
             f"({at_dam / exact_at_dam:.4f}x), profile RMS {profile_rms / height:.3%} of "
             f"h0, dX/dT {slope:.4f} against 2.0 (not graded)",
             f"  window:     graded over T in [{graded_from:g}, {graded_to:g}]",
             f"  slot:       {slot * 1e3:g} mm walled, not periodic: the contract promises "
             f"SPH no periodicity, so D13's thin direction is unavailable here",
             *health_lines,
             f"  reference:  {REFERENCES['ritter']['source']}",
             f"  caveat:     {REFERENCES['ritter']['caveat']}",
             f"  not graded: {REFERENCES['martin_moyce']['read']}",
             *world.report()]
    return _result("sph_dam_break", checks, lines, drawn, at_dam=at_dam,
                   exact_at_dam=exact_at_dam, profile_rms=profile_rms, slope=slope,
                   celerity=celerity, conventions=conventions,
                   track=[list(row) for row in track])


# ======================================================================================
# M7: containers
# ======================================================================================
#
# `PROGRESS_LIVE.md`'s M7 is the carried cup -- "a cup of water is a floating-body problem
# turned inside out" -- and the sloshing tank is its static half: a container whose
# contents have a free surface with a period of their own. The two cases above are the
# free-surface physics (M5) and the three below are floating bodies (M6); this is the one
# container, and it is the case whose period a carried cup will beat against.


def run_sph_sloshing(*, particle_radius: float = 0.00125, length: float = 0.1,
                     depth: float = 0.03, slot: float = 0.02,
                     amplitude: float = 0.002, periods: float = 4.0,
                     steps: int = 0, device: str | None = None, solver: str = "dfsph",
                     out_dir: str | Path = ".",
                     progress: Callable[[str], None] | None = None,
                     **_: Any) -> dict[str, Any]:
    """A tank's first mode is released: its period against `omega^2 = g k tanh(k h)`.

    Linear water-wave theory with `k = pi / L`, exact for small amplitudes, which
    `benchmarks/sph/` already uses for the same measurement and attributes to Lamb's
    *Hydrodynamics* 6th edition article 228.

    **Released through the velocity field, because the contract cannot displace a
    surface.** `freesurface.py` has `initialise_free_surface(level, amplitude, mode)` and
    `docs/sph-contract.md` has `fill_below(level)`, so the cosine surface the theory is
    about cannot be laid down. What can be is the *same mode at a quarter period's phase
    shift*: a flat surface moving with the mode's horizontal velocity,
    `u(x) = a omega / tanh(kh) sin(kx)`, which is where a standing wave is when all its
    energy is kinetic. It is released as a stack of `fill_box` slabs one particle spacing
    wide, each with its own velocity. Two honest costs, both in
    `NEEDS_FROM_THE_SOLVER`: the velocity is uniform over the depth where the mode's is
    `cosh(k(z+h))/cosh(kh)`, so a few per cent of the energy goes into other modes (the
    depth-average factor is reported); and whether abutting `fill_box` calls land on one
    lattice is not promised, so a seam would show up as a sawtooth the FFT would find.

    **A linear tilt is the wrong thing to release and that is a measurement, not a
    preference.** `cases.py::run_sloshing` tried it first, because tilting a tank and
    letting go is how the experiment is done: a ramp has a constant slope that fights the
    wall condition and beats the odd modes together, and the trace is their beat rather
    than a period. The first mode itself has zero surface slope at both walls, which is
    what the walls want.

    **The period is the mean of two estimators and their disagreement is graded beside
    it.** `cases.py::_period_of` and `_damping_of` are imported rather than reimplemented,
    and their docstrings carry what forced each: the crossings won on one lattice run
    (412.5 ms against the FFT's 453.5, theory 417.09) and the FFT won on the next (418.6
    against 392.9), so naming either as *the* estimator is naming whichever was lucky on
    the run that was looked at. And linear theory is inviscid while this is not, so the
    measured period is the damped one and is longer by `1/sqrt(1 - zeta^2)` by
    construction; the graded number has that removed.
    """
    sph = _sph()
    from .cases import _damping_of, _period_of
    from .world import Domain

    spacing = 2.0 * particle_radius
    theory = linear_sloshing(length, depth)
    domain = Domain(origin=(0.0, 0.0, 0.0), size=(length, slot, 3.0 * depth),
                    faces="wall")
    world = sph.SphWorld(domain=domain, particle_radius=particle_radius, fluid="water",
                         gravity=(0.0, 0.0, -GRAVITY), surface_tension=0.0,
                         device=device, solver=solver)
    world.boundary_from_domain()

    # The mode's horizontal velocity at the instant the surface is flat, depth-averaged
    # because a fill box carries one velocity: `a omega / tanh(kh)` at the surface, times
    # `tanh(kh)/(kh)` for the cosh profile's mean over the depth.
    peak = amplitude * theory["velocity_per_elevation"] * theory["depth_average"]
    slabs = max(1, int(round(length / spacing)))
    for index in range(slabs):
        low = index * length / slabs
        high = (index + 1) * length / slabs
        centre = 0.5 * (low + high)
        world.fill_box((low, 0.0, 0.0), (high, slot, depth),
                       velocity=(peak * math.sin(theory["wavenumber"] * centre),
                                 0.0, 0.0))
    started_with = len(world.positions())

    total = int(steps if steps else math.ceil(periods * theory["period"]
                                              / _timestep(world)))
    trace: list[tuple[float, float, float]] = []

    def take(index: int, elapsed: float) -> None:
        """The modal coefficient, and the elevation at the wall beside it.

        **The graded signal is the projection onto the mode, not one column's height, and
        that is a measurement.** The lattice case reads the elevation at the -x wall, and
        it can: its `surface_height` is a fill fraction and moves continuously. A particle
        backend's is the height of the topmost particle in a bin, which jumps by a whole
        particle spacing -- 3.5 mm at the default radius, against a 2 mm wave -- so a
        single column is mostly quantisation. Measured on a two-cycle run: the two
        estimators on the wall column came back 352 ms and 48 ms, 73% apart, against a
        theoretical 417.

        `eta_1 = (2/L) integral eta(x) cos(k x) dx` is the first mode's own amplitude, it
        is exact for `eta = a cos(kx)`, and it averages the quantisation over every column
        in the tank while rejecting the higher modes the velocity release excites. It is
        also the quantity linear theory is a statement about. The wall column is kept and
        reported, so the two can be compared.
        """
        profile, coordinate = _profile(world)
        elevation = profile - depth
        wet = profile > 0.0
        if wet.sum() < 4:
            trace.append((elapsed, float("nan"), float("nan")))
            return
        width = float(coordinate[1] - coordinate[0]) if len(coordinate) > 1 else length
        modal = (2.0 / length) * float(np.sum(
            elevation[wet] * np.cos(theory["wavenumber"] * coordinate[wet]) * width))
        trace.append((elapsed, modal, float(elevation[0])))

    # At least 60 samples a cycle: an FFT needs far more than Nyquist to put a peak in
    # the right bin, and the parabolic interpolation `_period_of` does needs the shoulders.
    every = max(1, int(theory["period"] / _timestep(world) / 60))
    advanced = _advance(world, total, sample_every=every, sample=take, progress=progress)

    times = np.array([row[0] for row in trace])
    elevation = np.array([row[1] for row in trace])
    at_wall = np.array([row[2] for row in trace])
    fft_period, fft_note = _period_of(times, elevation)
    crossings_period, crossings_note = _period_of(times, elevation, method="crossings")
    both = [v for v in (fft_period, crossings_period) if np.isfinite(v)]
    pair = float(np.mean(both)) if both else float("nan")
    spread = abs(fft_period - crossings_period) if len(both) == 2 else float("inf")
    zeta, decay_note = _damping_of(times, elevation)
    undamped = pair * math.sqrt(max(0.0, 1.0 - zeta * zeta))
    realised = float(np.nanmax(np.abs(elevation))) if len(elevation) else float("nan")
    limit = REFERENCES["linear_sloshing"]["tolerance"]

    health, health_lines = _health(world, advanced, started_with)
    checks = [
        Check("first-mode period, with the damping taken out", undamped,
              f"{theory['period']:.5f} s +/- {limit:.0%}",
              abs(undamped / theory['period'] - 1.0) <= limit
              if np.isfinite(undamped) else False,
              f"`omega^2 = g k tanh(k h)`, k = pi/L = {theory['wavenumber']:.4f} 1/m, "
              f"kh = {theory['kh']:.4f}. The graded number is the **mean of two "
              f"estimators** -- an FFT at {fft_period * 1e3:.2f} ms and the zero crossings "
              f"at {crossings_period * 1e3:.2f} ms -- with the damping removed: "
              f"`T sqrt(1 - zeta^2)` with zeta {zeta:.4f} from the decay envelope. The "
              f"two estimators are graded against each other below. cases.py::run_sloshing "
              f"has the runs that forced both choices"),
        Check("the two estimators agree", spread / theory["period"],
              f"<= {limit:.0%} of the period",
              spread / theory["period"] <= limit if np.isfinite(spread) else False,
              f"the gap between {fft_note} and {crossings_note}, as a fraction of theory. "
              f"**Graded, because a measurement whose two estimators disagree by more "
              f"than the tolerance cannot settle anything at that tolerance**, however "
              f"close their mean happens to land"),
        Check("damping ratio from the decay envelope", zeta,
              "reported, not graded", None,
              f"{decay_note}. A property of the run rather than an error: it enters the "
              f"graded period above and nothing else. On the lattice it measured 0.0150 "
              f"at 120 cells and 0.0035 at 240, which is four decimal places of "
              f"correction -- so if it comes out large here that is itself the finding"),
        Check("amplitude at the -x wall, for comparison with the modal coefficient",
              float(np.nanmax(np.abs(at_wall))) * 1e3 if len(at_wall) else float("nan"),
              "reported, not graded", None,
              "the quantity the lattice case grades and this one does not. A particle "
              "backend's surface height is the topmost particle in a bin, so one column "
              "jumps by a whole particle spacing -- 3.5 mm at the default radius against "
              "a 2 mm wave -- and the graded signal is the projection onto the mode "
              "instead. Measured on a two-cycle run, the estimator pair on this column "
              "came back 352 and 48 ms against a theoretical 417"),
        Check("realised modal amplitude", realised * 1e3,
              f"{amplitude * 1e3:.2f} mm asked for, reported, not graded", None,
              f"the mode was released through its velocity field -- "
              f"{peak * 1e3:.3f} mm/s peak, depth-averaged by a factor of "
              f"{theory['depth_average']:.4f} -- because the contract has no way to "
              f"displace a surface. How much of the asked-for elevation came back is how "
              f"well that projected onto the mode; the rest went into the higher ones and "
              f"will show up in the estimator spread above"),
        Check("wave steepness", realised / depth, "reported, not graded", None,
              f"linear theory is the first term of an expansion in this. A larger "
              f"amplitude **lengthens** the measured period -- a real tank's first mode "
              f"softens -- so a period slightly above theory at this steepness is physics "
              f"rather than error, and cases.py puts the second-order correction at 0.4% "
              f"for 6.7% of the depth"),
        *health,
    ]

    out = Path(out_dir)
    drawn = [
        figures.profile_figure(
            out / "live-sph-sloshing.png",
            [{"x": times.tolist(), "y": (elevation * 1e3).tolist(),
              "label": "first-mode amplitude (mm)"},
             {"x": times.tolist(), "y": (at_wall * 1e3).tolist(),
              "label": "elevation at the -x wall (mm)", "style": ":"},
             {"x": times.tolist(),
              "y": (realised * 1e3 * np.sin(2 * math.pi * times
                                            / theory["period"])).tolist(),
              "label": f"linear theory, T = {theory['period'] * 1e3:.1f} ms",
              "style": "--"}],
            title=f"sloshing: {undamped * 1e3:.2f} ms undamped against "
                  f"{theory['period'] * 1e3:.2f} ms ({undamped / theory['period']:.4f}x)",
            xlabel="t (s)", ylabel="mm"),
    ]
    for figure in drawn:
        figures.ensure_rgb(figure)

    lines = [f"# sph sloshing: {depth * 1e3:g} mm of water in a {length * 1e3:g} mm tank, "
             f"first mode at {amplitude * 1e3:g} mm",
             *_sph_lines(radius=particle_radius, particles=started_with,
                         length=length, depth=depth),
             f"  theory:     k = pi/L = {theory['wavenumber']:.4f} 1/m, kh = "
             f"{theory['kh']:.4f}, omega = {theory['omega']:.4f} rad/s, T = "
             f"{theory['period'] * 1e3:.3f} ms",
             f"  release:    velocity, not displacement: {peak * 1e3:.3f} mm/s peak over "
             f"{slabs} slabs of {spacing * 1e3:.3f} mm (the contract has no amplitude on "
             f"fill_below)",
             f"  measured:   {pair * 1e3:.3f} ms from {fft_period * 1e3:.3f} by FFT and "
             f"{crossings_period * 1e3:.3f} by crossings ({spread / theory['period']:.2%} "
             f"apart), {undamped * 1e3:.3f} ms undamped "
             f"({undamped / theory['period']:.4f}x theory)",
             *health_lines,
             f"  reference:  {REFERENCES['linear_sloshing']['source']}",
             *world.report()]
    return _result("sph_sloshing", checks, lines, drawn, period=pair, undamped=undamped,
                   theory=theory["period"], fft=fft_period, crossings=crossings_period,
                   spread=spread, zeta=zeta, amplitude=realised,
                   trace=[list(row) for row in trace])


# ======================================================================================
# M6: floating bodies
# ======================================================================================
#
# **Buoyancy is not supplied, and that is the whole point of these cases.**
# `docs/sph-contract.md`: `buoyancy=False` is the default for SPH and should stay that way
# -- an SPH free surface has a real waterline and real hydrostatic pressure, so buoyancy
# comes out of the pressure field and adding `rho V g` on top would count it twice. A
# draft produced by `rho V g` from a body's own volume would come out at
# `(rho_body/rho_water) H` almost by construction and would test only the volume count.
#
# Three of `run_archimedes`' mistakes are inherited as fixes rather than repeated, and
# `PROGRESS_LIVE.md`'s M6 section is the catalogue of them:
#
#   1. the settle is decided on `_settle`'s velocity bound, never on a step count;
#   2. the waterline is the **median over columns free of the hull**, not the mean over
#      the outer ring -- which is exactly where water thrown up by the hull lands, so the
#      probe read the splash and reported a 49.9 mm draft on a body placed at 20;
#   3. the draft is averaged over a window after the release transient has decayed over
#      several heave periods, and the **wobble over that window is graded**, because a
#      mean draft that is right while still oscillating is the average of a transient.
#      That is what made the lattice case's +1.9% irreproducible: the same code at the
#      same commit gave +4.8% on the next run.


@dataclass
class _Box:
    """A rectangular prism, its hydrostatics, and the Newton body that carries it."""

    beam: float
    length: float
    height: float
    density_ratio: float

    @property
    def volume(self) -> float:
        return self.beam * self.length * self.height

    @property
    def draft(self) -> float:
        return archimedes_draft(self.density_ratio, self.height)

    def hydrostatics(self) -> dict[str, float]:
        """Heel about the long axis, so the beam is the transverse dimension."""
        return heel_equilibrium(self.beam, self.height, self.density_ratio)


def _float_rig(box: _Box, world: Any, *, start_z: float, heel: float = 0.0,
               centre: tuple[float, float] | None = None, solver: str = "mujoco"):
    """A Newton model holding one box, coupled to an SPH world with no analytic buoyancy.

    One helper for the four floating cases, so that "how a hull is built and coupled" is
    one thing rather than four. Two things in it are decisions and not boilerplate.

    **`NewtonCoupling(buoyancy=False)`**, which `docs/sph-contract.md` requires and
    explains: an SPH free surface produces its own buoyancy out of the pressure field, so
    the analytic term would count it twice.

    **No joints.** `run_archimedes` says it gives its box "only the heave degree of
    freedom this case is about" and in fact adds a free body and relies on the geometry
    being square to the water. That is kept here, for two reasons: the heel case *needs* a
    free body, since rolling is the measurement; and a heave case whose box starts to roll
    has found something -- an asymmetric pressure field on a symmetric hull -- that a
    joint would hide. The cases report the off-axis rotation so that it is visible rather
    than assumed away.

    **The hull is also added to the fluid world, bound to the same body index.** The
    contract is explicit that `add_shape(..., body=)` "becomes Akinci boundary particles"
    and that `NewtonCoupling` drives those from `body_q` on the device, so a hull that
    exists only in Newton is a hull the fluid cannot see -- which is the SPH form of the
    failure `PROGRESS_LIVE.md` records on the lattice, where a body that existed only in
    Newton sat at the identity transform and voxelised into the corner of the domain for
    four retracted measurements. The index is Newton's own, because the coupling resizes
    the world's body table to Newton's body count and writes it from there on.
    """
    import warp as wp

    try:
        import newton
    except Exception as exc:
        raise UnphysicalScaling(
            "this case needs the Newton physics engine, which the [live] extra "
            f"declares:\n    pip install 'openreynolds[live]'\n(the import said: {exc})"
        ) from exc

    from .couple import NewtonCoupling
    from .units import FLUIDS

    plan = centre if centre is not None else (world.domain.size[0] / 2.0,
                                              world.domain.size[1] / 2.0)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    rotation = (wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(heel))
                if heel else wp.quat_identity())
    body = builder.add_body(xform=wp.transform(wp.vec3(plan[0], plan[1], start_z),
                                               rotation),
                            label="hull")
    shape = builder.default_shape_cfg.copy()
    shape.density = box.density_ratio * FLUIDS["water"].rho
    shape.has_shape_collision = False
    builder.add_shape_box(body, hx=box.beam / 2.0, hy=box.length / 2.0,
                          hz=box.height / 2.0, cfg=shape)
    # The fluid world needs its own body entry before a shape can be bound to one, and
    # the two indices have to line up: the coupling resizes the world's body table to
    # Newton's count and writes it from then on, so a hull bound to index 1 in a
    # one-body model would be driven by nothing. Checked rather than assumed, because
    # `PROGRESS_LIVE.md` records what a hull at the wrong pose costs -- four retracted
    # measurements, found only because three offsets returned identical forces.
    owned = world.add_body(position=(plan[0], plan[1], start_z),
                           orientation=(rotation[0], rotation[1], rotation[2],
                                        rotation[3]),
                           label="hull")
    if owned != body:
        raise UnphysicalScaling(
            f"the fluid world put this hull at body {owned} and Newton put it at {body}. "
            f"NewtonCoupling drives the world's body table by Newton's index, so the two "
            f"must agree: build the fluid world with no bodies of its own before the hull"
        )
    # The shape's pose is **body-local**, not world: `_shape_sdf` composes it with the
    # body's own, so passing the world position here puts the hull at twice it. Measured
    # the obvious way -- the sampler refused with "no boundary particles ... or it is
    # outside the domain", which is what a box at (192, 192, 120) mm in a 192 mm tank is.
    world.add_shape("box", body=owned, position=(0.0, 0.0, 0.0),
                    orientation=(0.0, 0.0, 0.0, 1.0),
                    scale=(box.beam / 2.0, box.length / 2.0, box.height / 2.0))
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, -GRAVITY))
    if solver == "mujoco":
        engine = newton.solvers.SolverMuJoCo(model, disable_contacts=True,
                                             solver="newton",
                                             integrator="implicitfast", iterations=10)
    else:
        engine = newton.solvers.SolverFeatherstone(model)
    link = NewtonCoupling(model, engine, world, dt=_timestep(world), buoyancy=False)
    return link, body, plan


def _roll_of(quaternion: Any) -> tuple[float, float]:
    """Heel angle about y from a quaternion, and how much of the rotation is off that axis.

    `2 atan2(qy, qw)` is exact for a rotation about y alone, and these cases release a box
    about y alone -- so the off-axis magnitude `hypot(qx, qz)` is returned beside it rather
    than discarded. A box that has started to yaw or pitch is a finding, and a single angle
    that silently folds it in is the kind of reporting that made `run_archimedes`' draft
    irreproducible for two days.
    """
    q = np.asarray(quaternion, dtype=float).ravel()
    x, y, z, w = (float(q[-4]), float(q[-3]), float(q[-2]), float(q[-1]))
    return 2.0 * math.atan2(y, w), float(math.hypot(x, z))


def _waterline(world: Any, box: _Box, plan: tuple[float, float],
               spacing: float) -> float:
    """The undisturbed water level: the **median** over columns the hull is not standing in.

    `PROGRESS_LIVE.md`'s first `run_archimedes` fix, and the reason is a measurement. That
    case read the mean over the outer ring of columns -- which is exactly where water
    thrown up by the hull lands, so the probe read the splash and reported a waterline
    that put the draft at 49.9 mm for a body placed at 20. A median over every column
    outside the hull's own footprint does not care how high a minority of columns go, and
    it uses the whole tank rather than its noisiest edge.

    The hull's footprint is excluded with a margin of one particle spacing, because a
    column that the hull's boundary particles reach into is a column whose depth is the
    hull's and not the water's. `spacing` is passed in rather than read off the world:
    `particle_radius` is a constructor argument in `docs/sph-contract.md` and not a
    promised attribute, and a case that reads an unpromised attribute is a case that
    breaks on the first implementation that names it something else.
    """
    columns = _columns(world, world.surface_height(axis=2))
    grid, axes = columns["grid"], columns["axes"]
    if grid.ndim == 1:
        x = axes[0]
        outside = np.abs(x - plan[0]) > box.beam / 2.0 + spacing
        depths = grid[outside]
    else:
        x = axes[0][:, None]
        y = axes[1][None, :]
        outside = ((np.abs(x - plan[0]) > box.beam / 2.0 + spacing)
                   | (np.abs(y - plan[1]) > box.length / 2.0 + spacing))
        depths = grid[outside]
    usable = depths[np.isfinite(depths) & (depths > 0.0)]
    return float(np.median(usable)) if usable.size else float("nan")


def _floating_tank(box: _Box, *, particle_radius: float, depth: float, span: float,
                   freeboard: float, device: str | None, solver: str):
    """A tank sized around a hull, filled and walled, with the particle budget reported.

    `span` is the tank width in beams. The lattice case uses five and this uses three by
    default, and that is a cost rather than a preference: particle count goes as the cube
    of the refinement and as the square of the span, and a tank of five beams at four
    particles through the draft is over sixty thousand particles. Three beams leaves the
    hull about one beam of water on each side.
    """
    sph = _sph()
    from .world import Domain

    width = span * box.beam
    long = span * box.length
    domain = Domain(origin=(0.0, 0.0, 0.0), size=(width, long, depth + freeboard),
                    faces="wall")
    world = sph.SphWorld(domain=domain, particle_radius=particle_radius, fluid="water",
                         gravity=(0.0, 0.0, -GRAVITY), surface_tension=0.0,
                         device=device, solver=solver)
    world.boundary_from_domain()
    world.fill_below(depth, axis=2)
    return world


def run_sph_static_float(*, particle_radius: float = 0.002, beam: float = 0.064,
                         height: float = 0.032, density_ratio: float = 0.5,
                         depth: float = 0.06, span: float = 3.0, freeboard: float = 0.04,
                         periods: float = 6.0, settle_steps: int = 40_000,
                         steps: int = 0, device: str | None = None,
                         solver: str = "dfsph", engine: str = "mujoco",
                         out_dir: str | Path = ".",
                         progress: Callable[[str], None] | None = None,
                         **_: Any) -> dict[str, Any]:
    """A box of half water's density is released at the surface: where does it float?

    Archimedes says a prism of uniform density floats at `d = (rho_body/rho_water) H` and
    nothing in the solver knows that. The draft is whatever the pressure integrated over
    the hull's boundary particles turns out to produce, so agreeing with it is evidence
    that the free surface, the gravity, the Akinci coupling and the Newton link are all
    right at once and that none of them is being rescued by an analytic term.
    **`NewtonCoupling(buoyancy=False)`**, which the contract requires.

    **The geometry is a raft and not a cube, and that is arithmetic rather than taste.**
    `metacentric_height` says a prism floats upright only when `(b/H)^2 > 6 r (1-r)`, so a
    square-section box at half water's density -- which is what the lattice case floats --
    **capsizes**: `1 < 1.5`. That case holds its cube square to the water; this one uses a
    beam twice its height, `(b/H)^2 = 4` against a criterion of 1.5, so the hull is
    genuinely stable upright and the draft is being measured on a body that wants to stay
    there. `run_sph_heel_stability` takes the unstable case deliberately.

    **Three things `run_archimedes` got wrong, inherited as fixes.**
    `PROGRESS_LIVE.md`'s M6 section is a catalogue of how to pose this case wrong and all
    three are here: the settle is decided on the fluid's own velocity by `_settle` and is
    **graded**, never taken on a step count; the waterline is `_waterline`'s median over
    columns free of the hull, not a mean over the outer ring where the splash lands; and
    the draft is averaged over the last 40% of a run several heave periods long, with the
    **wobble over that window graded** so that a mean draft which is right while still
    oscillating cannot pass. The number that case first reported, +1.9%, moved to +4.8% on
    a re-run of the same commit; a case that does not reproduce is not measuring anything.
    """
    box = _Box(beam=beam, length=beam, height=height, density_ratio=density_ratio)
    hydro = box.hydrostatics()
    if not hydro["stable"]:
        raise UnphysicalScaling(
            f"this geometry cannot float upright, so its draft is not a static float's: "
            f"(b/H)^2 = {hydro['slenderness']:.4f} against the stability criterion "
            f"6 r (1-r) = {hydro['criterion']:.4f}, so GM = {hydro['GM'] * 1e3:.3f} mm is "
            f"negative and the box heels to {math.degrees(hydro['angle']):.2f} degrees. "
            f"Widen the beam, or run run_sph_heel_stability, which is the case for an "
            f"unstable hull"
        )
    world = _floating_tank(box, particle_radius=particle_radius, depth=depth, span=span,
                           freeboard=freeboard, device=device, solver=solver)
    started_with = len(world.positions())
    celerity = math.sqrt(GRAVITY * depth)

    # Released **at** the draft Archimedes predicts, not dropped from above: a box that
    # arrives with momentum spends several heave periods giving it up and this case is
    # about the equilibrium. Starting at the answer is not assuming it -- the body is free
    # to leave, and a wrong buoyant force moves it straight away, which the trace shows.
    link, body, plan = _float_rig(box, world,
                                  start_z=depth - box.draft + box.height / 2.0,
                                  solver=engine)

    # Fill, hold the hull, let the water settle, then let go. A hull released into water
    # that has not settled feels a force swinging between 0.2 and 1.7 times its weight
    # (`PROGRESS_LIVE.md`), and the hull is held by pinning its pose through `link.step`
    # rather than by stepping the world -- which does not sync poses from Newton, and in
    # the lattice case put the hull in the corner of the domain for four retracted runs.
    held = link.state.body_q.numpy().copy()

    def hold() -> None:
        link.state.body_qd.zero_()
        link.state.body_q.assign(held)

    settled = _settle(world, reference_speed=celerity, max_steps=int(settle_steps),
                      step=link.step, hold=hold, progress=progress)

    period = heave_period(box.draft)
    dt = _timestep(world)
    total = int(steps if steps else math.ceil(periods * period / dt))
    track: list[tuple[float, float, float, float]] = []
    every = max(1, total // 200)
    began = time.perf_counter()
    for index in range(total):
        link.step()
        if index % every == 0 or index == total - 1:
            pose = link.state.body_q.numpy()[body]
            heel, off_axis = _roll_of(pose[3:7])
            level = _waterline(world, box, plan, 2.0 * particle_radius)
            track.append((_elapsed(world, dt, index + 1), level,
                          level - (float(pose[2]) - box.height / 2.0), off_axis))
        if progress is not None and index % max(1, total // 20) == 0 and track:
            progress(f"    {index:,}/{total:,}  draft {track[-1][2] * 1e3:.4f} mm  "
                     f"{figures.clock((time.perf_counter() - began) / max(index, 1) * (total - index))} left")

    late = [row for row in track if row[0] >= 0.6 * total * dt]
    draft = float(np.mean([row[2] for row in late])) if late else float("nan")
    wobble = float(np.std([row[2] for row in late])) if late else float("nan")
    off_axis = max((row[3] for row in track), default=float("nan"))
    through_draft = box.draft / (2.0 * particle_radius)
    limit = REFERENCES["archimedes"]["tolerance"]

    advanced = {"steps": total, "dt": dt, "elapsed": total * dt,
                "last": {"particles": len(world.positions())}}
    health, health_lines = _health(world, advanced, started_with)
    checks = [
        Check("draft against Archimedes", draft * 1e3,
              f"{box.draft * 1e3:.4f} mm +/- {limit:.0%}",
              abs(draft / box.draft - 1.0) <= limit if np.isfinite(draft) else False,
              f"`d = (rho_body/rho_water) H` = {density_ratio:g} x {height * 1e3:g} mm, "
              f"averaged over the last 40% of {periods:g} heave periods. The solver is "
              f"told nothing about buoyancy -- `NewtonCoupling(buoyancy=False)`, which "
              f"docs/sph-contract.md requires -- so this force is the pressure field "
              f"integrated over the hull's boundary particles and nothing else"),
        Check("the draft stopped moving", wobble / box.draft,
              f"<= {limit:.0%} of the draft",
              wobble / box.draft <= limit if np.isfinite(wobble) else False,
              "the standard deviation of the draft over the same window. **Graded, "
              "because a mean draft that is right while still oscillating is the average "
              "of a transient**: the lattice case reported +1.9% and then +4.8% from the "
              "same code at the same commit, and this is the check that would have caught "
              "it"),
        Check("the tank was at rest before the hull was released", settled["settled"],
              "True (|u|max <= 2% of the gravity-wave speed)", bool(settled["settled"]),
              f"reached {settled['reached'] * 1e3:.3f} mm/s after {settled['steps']:,} "
              f"steps against a bound of {settled['bound'] * 1e3:.3f}. **Graded, because "
              f"the first version of the lattice case was not**: it took 3,000 steps on "
              f"trust and released the hull into a tank still moving at 50 to 90 mm/s, "
              f"with the vertical force on a pinned hull swinging between 0.2 and 1.7 "
              f"times its weight"),
        Check("the hull stayed upright", math.degrees(off_axis),
              "reported, not graded", None,
              f"the largest off-axis rotation over the run. No joint holds this box "
              f"square to the water -- it is upright because `(b/H)^2 = "
              f"{hydro['slenderness']:.3f}` exceeds the stability criterion "
              f"{hydro['criterion']:.3f}, GM = {hydro['GM'] * 1e3:+.3f} mm -- so a box "
              f"that rolls here is a finding about the pressure field and not a missing "
              f"constraint"),
        Check("particle spacings through the draft", f"{through_draft:.1f}",
              "reported, not graded", None,
              "**the first thing to raise if this case misses.** An SPH hull is not "
              "voxelised, but its boundary particles occupy a layer about one spacing "
              "thick, so the displaced volume is uncertain by roughly `dp` over the "
              "wetted surface -- which at this count is percents. A tank big enough for "
              "a hull at a useful count is tens of thousands of particles, which is why "
              "this case is heavy and the M5 ones are not"),
        *health,
    ]

    out = Path(out_dir)
    drawn = [
        figures.profile_figure(
            out / "live-sph-static-float.png",
            [{"x": [row[0] * 1e3 for row in track],
              "y": [row[2] * 1e3 for row in track], "label": "draft (mm)"},
             {"x": [track[0][0] * 1e3, track[-1][0] * 1e3] if track else [0.0, 1.0],
              "y": [box.draft * 1e3] * 2, "label": "Archimedes", "style": "--"},
             {"x": [row[0] * 1e3 for row in track],
              "y": [math.degrees(row[3]) for row in track],
              "label": "off-axis rotation (deg)", "panel": 1}],
            title=f"static float: {draft * 1e3:.3f} mm against {box.draft * 1e3:.3f} "
                  f"({draft / box.draft:.4f}x), wobble {wobble * 1e3:.4f} mm",
            xlabel="t (ms)", ylabel="mm / degrees", panels=2),
    ]
    for figure in drawn:
        figures.ensure_rgb(figure)

    lines = [f"# sph static float: a {beam * 1e3:g} x {beam * 1e3:g} x {height * 1e3:g} mm "
             f"box at {density_ratio:g} of water's density",
             *_sph_lines(radius=particle_radius, particles=started_with,
                         length=beam, depth=depth),
             f"  hydrostatics: (b/H)^2 {hydro['slenderness']:.4f} against the criterion "
             f"6r(1-r) = {hydro['criterion']:.4f}; GM {hydro['GM'] * 1e3:+.4f} mm, so "
             f"upright is stable",
             f"  expected:   draft {box.draft * 1e3:.4f} mm "
             f"({through_draft:.1f} spacings), heave period {period * 1e3:.2f} ms",
             f"  measured:   draft {draft * 1e3:.4f} mm ({draft / box.draft:.4f}x), "
             f"wobble {wobble * 1e3:.4f} mm",
             f"  settled:    {settled['reached'] * 1e3:.3f} mm/s after "
             f"{settled['steps']:,} steps (bound {settled['bound'] * 1e3:.3f})",
             f"  buoyancy:   NOT supplied; the pressure over the hull produces it and "
             f"this case is the check that it is the right size",
             *health_lines,
             *world.report()]
    return _result("sph_static_float", checks, lines, drawn, draft=draft,
                   exact_draft=box.draft, wobble=wobble, settled=settled,
                   hydrostatics=hydro, through_draft=through_draft,
                   track=[list(row) for row in track])


def run_sph_heave_decay(*, particle_radius: float = 0.002, beam: float = 0.064,
                        height: float = 0.032, density_ratio: float = 0.5,
                        depth: float = 0.06, span: float = 3.0, freeboard: float = 0.04,
                        displacement: float = 0.25, periods: float = 8.0,
                        settle_steps: int = 40_000, steps: int = 0,
                        device: str | None = None, solver: str = "dfsph",
                        engine: str = "mujoco", out_dir: str | Path = ".",
                        progress: Callable[[str], None] | None = None,
                        **_: Any) -> dict[str, Any]:
    """A floating box is pushed down and released: its heave period against `2 pi sqrt(d/g)`.

    The restoring stiffness of a prism is `rho g A` and its mass is `rho A d`, so the
    waterplane area cancels and the undamped natural period depends on the draft alone:
    `T = 2 pi sqrt(d / g)`. `cases.py::run_archimedes` already sizes its run with this
    estimate; this case measures it.

    **What is graded is the sign, and the magnitude is only bounded.**
    `T = 2 pi sqrt(d/g)` neglects added mass, which for a surface-piercing prism is a
    comparable fraction of the body's own mass, and it neglects radiation damping, which
    lengthens the measured period again by `1/sqrt(1 - zeta^2)`. Both corrections have the
    **same sign**: nothing in the physics shortens a heave period. So `T_measured >=
    T_estimate` is graded with no tolerance at all -- a measured period below the estimate
    means the restoring force is too stiff, which is the signature of a hull being held up
    by something other than the pressure field -- and 25% is a loose upper bound on the
    correction, stated in `REFERENCES["heave"]` as the grading choice it is. The measured
    added-mass coefficient `(T_measured/T_estimate)^2 - 1` is reported, for whoever wants
    to compare it with a strip theory later.

    The equilibrium the decay oscillates about is **graded against the static float's
    draft**, which is the cheap cross-check that makes this case more than a period: two
    cases that disagree about where this box floats cannot both be measuring it.

    `_damping_of` and `_period_of` are imported from `cases.py`, so the estimator pair and
    the damping correction arrive with the runs that forced them attached.
    """
    box = _Box(beam=beam, length=beam, height=height, density_ratio=density_ratio)
    hydro = box.hydrostatics()
    from .cases import _damping_of, _period_of

    world = _floating_tank(box, particle_radius=particle_radius, depth=depth, span=span,
                           freeboard=freeboard, device=device, solver=solver)
    started_with = len(world.positions())
    celerity = math.sqrt(GRAVITY * depth)
    estimate = heave_period(box.draft)

    # Held at the equilibrium while the water settles, then released from `displacement`
    # of a draft **below** it: pushing a box down and letting go is the decay test, and
    # starting it above the surface would drop it from air into water, which is an impact
    # and not a decay.
    link, body, plan = _float_rig(box, world,
                                  start_z=depth - box.draft + box.height / 2.0,
                                  solver=engine)
    held = link.state.body_q.numpy().copy()

    def hold() -> None:
        link.state.body_qd.zero_()
        link.state.body_q.assign(held)

    settled = _settle(world, reference_speed=celerity, max_steps=int(settle_steps),
                      step=link.step, hold=hold, progress=progress)

    pushed = held.copy()
    pushed[body][2] -= displacement * box.draft
    link.state.body_q.assign(pushed)
    link.state.body_qd.zero_()

    dt = _timestep(world)
    total = int(steps if steps else math.ceil(periods * estimate / dt))
    track: list[tuple[float, float, float]] = []
    every = max(1, int(estimate / dt / 60))
    for index in range(total):
        link.step()
        if index % every == 0:
            pose = link.state.body_q.numpy()[body]
            level = _waterline(world, box, plan, 2.0 * particle_radius)
            _, off_axis = _roll_of(pose[3:7])
            track.append((_elapsed(world, dt, index + 1),
                          level - (float(pose[2]) - box.height / 2.0), off_axis))
        if progress is not None and index % max(1, total // 20) == 0 and track:
            progress(f"    {index:,}/{total:,}  draft {track[-1][1] * 1e3:.4f} mm")

    times = np.array([row[0] for row in track])
    drafts = np.array([row[1] for row in track])
    fft_period, fft_note = _period_of(times, drafts)
    crossings_period, crossings_note = _period_of(times, drafts, method="crossings")
    both = [v for v in (fft_period, crossings_period) if np.isfinite(v)]
    measured = float(np.mean(both)) if both else float("nan")
    spread = abs(fft_period - crossings_period) if len(both) == 2 else float("inf")
    zeta, decay_note = _damping_of(times, drafts)
    equilibrium = float(np.mean(drafts[len(drafts) // 2:])) if len(drafts) > 3 \
        else float("nan")
    added_mass = (measured / estimate) ** 2 - 1.0 if np.isfinite(measured) else float("nan")
    limit = REFERENCES["heave"]["tolerance"]
    draft_limit = REFERENCES["archimedes"]["tolerance"]

    advanced = {"steps": total, "dt": dt, "elapsed": total * dt,
                "last": {"particles": len(world.positions())}}
    health, health_lines = _health(world, advanced, started_with)
    checks = [
        Check("the heave period is not shorter than the estimate",
              measured / estimate if np.isfinite(measured) else float("nan"),
              ">= 1.0, exactly",
              bool(np.isfinite(measured) and measured >= estimate),
              f"`T = 2 pi sqrt(d/g)` = {estimate * 1e3:.2f} ms. **Graded with no "
              f"tolerance, because both neglected effects have the same sign**: added "
              f"mass lengthens a heave period and radiation damping lengthens it again, "
              f"and nothing shortens it. A measured period below the estimate means the "
              f"restoring force is too stiff, which is a hull being held up by something "
              f"other than the pressure field"),
        Check("the heave period against the estimate", measured,
              f"{estimate:.5f} s, within +{limit:.0%}",
              abs(measured / estimate - 1.0) <= limit
              if np.isfinite(measured) else False,
              f"a loose **upper** bound on the added-mass correction and stated as the "
              f"grading choice it is -- no analytic added mass for a surface-piercing "
              f"rectangular prism has been read for this repository. The mean of "
              f"{fft_note} and {crossings_note}; they disagree by "
              f"{spread / estimate:.2%} of the estimate"),
        Check("added-mass coefficient", added_mass, "reported, not graded", None,
              f"`(T_measured/T_estimate)^2 - 1`: the mass of water this hull accelerates "
              f"with it, as a fraction of its own. The number this case exists to "
              f"produce, and the reason the check above is a bound rather than an "
              f"agreement"),
        Check("the equilibrium agrees with the static float", equilibrium * 1e3,
              f"{box.draft * 1e3:.4f} mm +/- {draft_limit:.0%}",
              abs(equilibrium / box.draft - 1.0) <= draft_limit
              if np.isfinite(equilibrium) else False,
              "the mean draft over the second half of the decay, against Archimedes. The "
              "cheap cross-check that makes this a measurement rather than a period: two "
              "cases that disagree about where this box floats cannot both be right, and "
              "a decay about the wrong equilibrium has the right period for the wrong "
              "reason"),
        Check("damping ratio from the decay envelope", zeta, "reported, not graded", None,
              f"{decay_note}. It is radiation damping and the fluid's viscosity together, "
              f"and it enters nothing that is graded here -- the period check above is "
              f"one-sided precisely because this correction has the same sign as the "
              f"added-mass one"),
        Check("the tank was at rest before the hull was pushed", settled["settled"],
              "True (|u|max <= 2% of the gravity-wave speed)", bool(settled["settled"]),
              f"reached {settled['reached'] * 1e3:.3f} mm/s after {settled['steps']:,} "
              f"steps. Same bound and same reason as the static float: a decay released "
              f"into a moving tank measures the tank"),
        *health,
    ]

    out = Path(out_dir)
    drawn = [
        figures.profile_figure(
            out / "live-sph-heave-decay.png",
            [{"x": (times * 1e3).tolist(), "y": (drafts * 1e3).tolist(),
              "label": "draft (mm)"},
             {"x": [times[0] * 1e3, times[-1] * 1e3] if len(times) else [0.0, 1.0],
              "y": [box.draft * 1e3] * 2, "label": "Archimedes", "style": "--"}],
            title=f"heave decay: {measured * 1e3:.2f} ms measured against an estimated "
                  f"{estimate * 1e3:.2f} ms, added mass {added_mass:+.2f}",
            xlabel="t (ms)", ylabel="mm"),
    ]
    for figure in drawn:
        figures.ensure_rgb(figure)

    lines = [f"# sph heave decay: a {beam * 1e3:g} mm box pushed "
             f"{displacement * box.draft * 1e3:.2f} mm below its draft and released",
             *_sph_lines(radius=particle_radius, particles=started_with,
                         length=beam, depth=depth),
             f"  estimate:   T = 2 pi sqrt(d/g) with d = {box.draft * 1e3:g} mm is "
             f"{estimate * 1e3:.3f} ms (added mass NOT in it)",
             f"  measured:   {measured * 1e3:.3f} ms ({measured / estimate:.4f}x), from "
             f"{fft_period * 1e3:.3f} by FFT and {crossings_period * 1e3:.3f} by "
             f"crossings; zeta {zeta:.4f}",
             f"  added mass: {added_mass:+.4f} of the body's own mass, measured",
             f"  equilibrium:{equilibrium * 1e3:.4f} mm against Archimedes' "
             f"{box.draft * 1e3:.4f}",
             *health_lines,
             f"  reference:  {REFERENCES['heave']['source']}",
             f"  caveat:     {REFERENCES['heave']['caveat']}",
             *world.report()]
    return _result("sph_heave_decay", checks, lines, drawn, period=measured,
                   estimate=estimate, added_mass=added_mass, zeta=zeta,
                   equilibrium=equilibrium, settled=settled,
                   track=[list(row) for row in track])


def run_sph_heel_stability(*, particle_radius: float = 0.002, beam: float = 0.045,
                           slenderness: float = 0.9, density_ratio: float = 0.5,
                           depth: float = 0.06, span: float = 3.0,
                           freeboard: float = 0.05, release: float = 3.0,
                           periods: float = 10.0, settle_steps: int = 40_000,
                           stable_beam: float = 0.072, steps: int = 0,
                           device: str | None = None, solver: str = "dfsph",
                           engine: str = "mujoco", out_dir: str | Path = ".",
                           progress: Callable[[str], None] | None = None,
                           **_: Any) -> dict[str, Any]:
    """The boat benchmark: a floating box's heeling stability, graded on what is derivable.

    **Bogner and Rude (2013) could not be read, and this case says so rather than quoting
    it.** The published version returned HTTP 403 to every attempt; the authors' arXiv
    preprint (1201.0351) is reachable and its abstract confirms that the paper's evaluation
    is "the rotational stability of non-spherical rigid bodies floating on a plane water
    surface" with "convergence towards the ideal solution for the measured heeling
    stability of a floating box" -- but the preprint's own heeling section did not extract
    as text, so **no geometry, resolution or measured value from it is quoted here**.
    `REFERENCES["bogner_rude"]` carries that, marked `graded: False`. This repository's
    rule is that an unread reference is reported and not graded, and
    `REFERENCES["martin_moyce"]` one entry above is the same rule applied to the dam break.

    **What is graded instead is derivable in half a page and is hand-checked in the
    tests.** A prism of uniform density floating at draft `d = r H` has its centre of
    buoyancy at `d/2`, its centre of gravity at `H/2`, and a metacentric radius
    `BM = b^2/(12 d)`, so

        stable upright  <=>  (b/H)^2 > 6 r (1 - r)

    and when it is not, the wall-sided righting lever `GZ = sin(a)(GM + BM tan^2(a)/2)`
    vanishes at `tan^2(a_eq) = 12 r (1-r) (H/b)^2 - 2`. Both are in
    `metacentric_height` and `heel_equilibrium`, and the tests check them at values that
    come out exactly: a square-section prism at half water's density heels to **45
    degrees** and at a quarter to **`atan(1/2)` = 26.565 degrees**.

    **Two hulls, because the criterion has two sides and a solver that capsizes everything
    would pass one of them.** The default geometry is unstable upright -- beam 45 mm,
    height `slenderness` x beam = 40.5 mm, so `(b/H)^2 = 1.235` against a criterion of 1.5
    -- and its equilibrium heel is 33.26 degrees with the waterline still on the two
    vertical sides (the wall-sided margin is reported). The control hull is `stable_beam`
    wide at the same height, comfortably stable, and must **return** to upright. Both
    signs are graded; only the unstable hull's angle is.

    The heel is averaged after the roll oscillation has decayed, over the last 40% of the
    run, with its wobble graded -- the same mistake with a draft is what made
    `run_archimedes` irreproducible.
    """
    height = slenderness * beam
    unstable = _Box(beam=beam, length=2.0 * beam, height=height,
                    density_ratio=density_ratio)
    stable = _Box(beam=stable_beam, length=2.0 * beam, height=height,
                  density_ratio=density_ratio)
    predicted = unstable.hydrostatics()
    control = stable.hydrostatics()
    if predicted["upright_stable"] or not control["stable"]:
        raise UnphysicalScaling(
            f"this case needs one hull that cannot float upright and one that can, and "
            f"got {'two stable' if predicted['upright_stable'] else 'two unstable'} ones: "
            f"(b/H)^2 is {predicted['slenderness']:.4f} and {control['slenderness']:.4f} "
            f"against a criterion of {predicted['criterion']:.4f}. The criterion has two "
            f"sides and a solver that capsizes everything would pass only one of them"
        )

    out = Path(out_dir)
    runs: dict[str, dict[str, Any]] = {}
    for name, box in (("unstable", unstable), ("stable", stable)):
        if progress is not None:
            progress(f"  the {name} hull: beam {box.beam * 1e3:g} mm, "
                     f"(b/H)^2 {box.hydrostatics()['slenderness']:.3f}")
        runs[name] = _heel_once(box, particle_radius=particle_radius, depth=depth,
                                span=span, freeboard=freeboard,
                                release=math.radians(release), periods=periods,
                                settle_steps=settle_steps, steps=steps, device=device,
                                solver=solver, engine=engine, progress=progress)

    measured = runs["unstable"]["heel"]
    wobble = runs["unstable"]["wobble"]
    expected = predicted["angle"]
    limit = REFERENCES["metacentric"]["tolerance"]
    checks = [
        Check("the unstable hull left upright", math.degrees(abs(measured)),
              f"well past the {release:g} degrees it was released at, exactly",
              bool(np.isfinite(measured) and abs(measured) > 3.0 * math.radians(release)),
              f"**the sign, graded exactly.** GM = {predicted['GM'] * 1e3:+.4f} mm is "
              f"negative -- `(b/H)^2 = {predicted['slenderness']:.4f}` against the "
              f"criterion `6 r (1-r) = {predicted['criterion']:.4f}` -- so this hull "
              f"cannot float upright and a solver that keeps it there is wrong whatever "
              f"angle it reports"),
        Check("the stable hull returned to upright",
              math.degrees(abs(runs["stable"]["heel"])),
              f"below the {release:g} degrees it was released at, exactly",
              bool(np.isfinite(runs["stable"]["heel"])
                   and abs(runs["stable"]["heel"]) < math.radians(release)),
              f"**the other side of the same criterion.** The control hull has GM = "
              f"{control['GM'] * 1e3:+.4f} mm, so it must come back; a solver that "
              f"capsizes everything passes the check above and fails this one, which is "
              f"why both are here"),
        Check("the equilibrium heel angle", math.degrees(measured),
              f"{math.degrees(expected):.4f} degrees +/- {limit:.0%}",
              abs(measured / expected - 1.0) <= limit
              if np.isfinite(measured) and expected else False,
              f"`tan^2(a) = 12 r (1-r) (H/b)^2 - 2` = "
              f"{math.tan(expected) ** 2:.4f}, derived in `heel_equilibrium` and "
              f"hand-checked in the tests. Averaged over the last 40% of the run, after "
              f"the roll oscillation has decayed. 10% is a grading choice and is stated "
              f"as one: the closed form is quasi-static and inviscid and the measured "
              f"equilibrium is reached through a damped roll in a tank three beams wide"),
        Check("the heel stopped moving", wobble / abs(expected) if expected else
              float("nan"), f"<= {limit:.0%} of the angle",
              wobble / abs(expected) <= limit
              if np.isfinite(wobble) and expected else False,
              "the standard deviation of the heel over the averaging window. The same "
              "check as the static float's wobble, for the same reason: a mean angle "
              "that is right while still swinging is the average of a transient"),
        Check("wall-sided margin", f"tan a {math.tan(expected):.4f} against a limit of "
              f"{predicted['wall_sided_limit']:.4f}", "reported, not graded", None,
              f"the closed form holds only while the waterline stays on the two vertical "
              f"sides -- the deck must not immerse and the keel must not emerge, "
              f"`tan a <= 2 min(d, H-d)/b`. The margin here is "
              f"{predicted['margin']:+.4f}, and the geometry was chosen to have one: at "
              f"beam = height the equilibrium sits exactly **on** the limit, where the "
              f"formula is not a statement about anything"),
        Check("Bogner and Rude (2013)", "not read", "reported, not graded", None,
              REFERENCES["bogner_rude"]["read"]),
        Check("off-axis rotation on the unstable hull",
              math.degrees(runs["unstable"]["off_axis"]), "reported, not graded", None,
              "how much of the rotation was not about the axis this box was released "
              "about. No joint restrains it, so a yawing or pitching hull is visible "
              "here rather than folded into the heel angle"),
        *runs["unstable"]["health"],
    ]

    drawn = [
        figures.profile_figure(
            out / "live-sph-heel-stability.png",
            [{"x": [row[0] * 1e3 for row in runs["unstable"]["track"]],
              "y": [math.degrees(row[1]) for row in runs["unstable"]["track"]],
              "label": f"unstable hull, (b/H)^2 = {predicted['slenderness']:.2f}"},
             {"x": [row[0] * 1e3 for row in runs["stable"]["track"]],
              "y": [math.degrees(row[1]) for row in runs["stable"]["track"]],
              "label": f"stable hull, (b/H)^2 = {control['slenderness']:.2f}"},
             {"x": [runs["unstable"]["track"][0][0] * 1e3,
                    runs["unstable"]["track"][-1][0] * 1e3]
              if runs["unstable"]["track"] else [0.0, 1.0],
              "y": [math.degrees(expected)] * 2,
              "label": "closed-form equilibrium", "style": "--"}],
            title=f"heeling stability: {math.degrees(measured):.2f} degrees against a "
                  f"derived {math.degrees(expected):.2f}",
            xlabel="t (ms)", ylabel="heel (degrees)"),
    ]
    for figure in drawn:
        figures.ensure_rgb(figure)

    lines = [f"# sph heel stability: a {beam * 1e3:g} mm beam x {height * 1e3:g} mm box "
             f"at {density_ratio:g} of water's density, and a stable control",
             *runs["unstable"]["lines"],
             f"  criterion:  stable upright iff (b/H)^2 > 6r(1-r): "
             f"{predicted['slenderness']:.4f} vs {predicted['criterion']:.4f} "
             f"(unstable, GM {predicted['GM'] * 1e3:+.4f} mm) and "
             f"{control['slenderness']:.4f} vs {control['criterion']:.4f} "
             f"(stable, GM {control['GM'] * 1e3:+.4f} mm)",
             f"  derived:    equilibrium heel {math.degrees(expected):.4f} degrees, "
             f"tan a {math.tan(expected):.4f} against a wall-sided limit of "
             f"{predicted['wall_sided_limit']:.4f}",
             f"  measured:   {math.degrees(measured):.4f} degrees "
             f"({measured / expected if expected else float('nan'):.4f}x), wobble "
             f"{math.degrees(wobble):.4f}; control hull returned to "
             f"{math.degrees(runs['stable']['heel']):.4f}",
             f"  not graded: {REFERENCES['bogner_rude']['read']}",
             f"  reference:  {REFERENCES['metacentric']['source']}"]
    return _result("sph_heel_stability", checks, lines, drawn,
                   heel=measured, expected=expected, wobble=wobble,
                   control=runs["stable"]["heel"], hydrostatics=predicted,
                   control_hydrostatics=control,
                   track=[list(row) for row in runs["unstable"]["track"]])


def _heel_once(box: _Box, *, particle_radius: float, depth: float, span: float,
               freeboard: float, release: float, periods: float, settle_steps: int,
               steps: int, device: str | None, solver: str, engine: str,
               progress: Callable[[str], None] | None) -> dict[str, Any]:
    """One hull, released at a small heel, followed until its roll settles.

    Split out because `run_sph_heel_stability` runs two hulls and the two halves of the
    stability criterion are the measurement. The roll is timed in **heave** periods, which
    is not the roll period -- a prism's roll period depends on its radius of gyration and
    this case does not claim one -- but is the right order and is a number this module
    already has; the wobble check is what decides whether the run was long enough.
    """
    world = _floating_tank(box, particle_radius=particle_radius, depth=depth, span=span,
                           freeboard=freeboard, device=device, solver=solver)
    started_with = len(world.positions())
    celerity = math.sqrt(GRAVITY * depth)
    link, body, plan = _float_rig(box, world,
                                  start_z=depth - box.draft + box.height / 2.0,
                                  heel=release, solver=engine)
    held = link.state.body_q.numpy().copy()

    def hold() -> None:
        link.state.body_qd.zero_()
        link.state.body_q.assign(held)

    settled = _settle(world, reference_speed=celerity, max_steps=int(settle_steps),
                      step=link.step, hold=hold, progress=progress)

    dt = _timestep(world)
    total = int(steps if steps else math.ceil(periods * heave_period(box.draft) / dt))
    track: list[tuple[float, float, float]] = []
    every = max(1, total // 240)
    for index in range(total):
        link.step()
        if index % every == 0 or index == total - 1:
            heel, off_axis = _roll_of(link.state.body_q.numpy()[body][3:7])
            track.append((_elapsed(world, dt, index + 1), heel, off_axis))
        if progress is not None and index % max(1, total // 20) == 0 and track:
            progress(f"    {index:,}/{total:,}  heel {math.degrees(track[-1][1]):+.3f} deg")

    late = [row for row in track if row[0] >= 0.6 * total * dt]
    heel = float(np.mean([row[1] for row in late])) if late else float("nan")
    wobble = float(np.std([row[1] for row in late])) if late else float("nan")
    advanced = {"steps": total, "dt": dt, "elapsed": total * dt,
                "last": {"particles": len(world.positions())}}
    health, health_lines = _health(world, advanced, started_with)
    hydro = box.hydrostatics()
    lines = [*_sph_lines(radius=particle_radius, particles=started_with,
                         length=box.beam, depth=depth),
             f"  settled:    {settled['reached'] * 1e3:.3f} mm/s after "
             f"{settled['steps']:,} steps (bound {settled['bound'] * 1e3:.3f})",
             f"  draft:      {box.draft * 1e3:g} mm, "
             f"{box.draft / (2 * particle_radius):.1f} spacings",
             *health_lines]
    return {"heel": heel, "wobble": wobble, "settled": settled, "track": track,
            "health": health, "lines": lines, "hydrostatics": hydro,
            "off_axis": max((row[2] for row in track), default=float("nan"))}


def run_sph_force_noise(*, particle_radius: float = 0.002, side: float = 0.032,
                        depth: float = 0.08, span: float = 3.0, freeboard: float = 0.02,
                        window: int = 10, samples: int = 4000,
                        settle_steps: int = 40_000, device: str | None = None,
                        solver: str = "dfsph", engine: str = "mujoco",
                        out_dir: str | Path = ".",
                        progress: Callable[[str], None] | None = None,
                        **_: Any) -> dict[str, Any]:
    """The force on a static submerged body: is its noise low enough to control on?

    A controller cannot use a force whose fluctuation exceeds its signal. A 1 kHz loop
    reading a force with 100% RMS noise is reading nothing; at 10% it has a
    signal-to-noise ratio of 10, and a loop that can average 10 samples -- 10 ms, still
    inside a 100 Hz outer loop -- sees 3.2% **if** the noise is white. So this case
    measures three numbers and grades two: the mean force, the per-step noise, and the
    noise after a `window`-sample average, with the **measured** reduction between the
    last two reported rather than `sqrt(window)` assumed. If the noise is correlated step
    to step, averaging buys less than the square root, and the ratio of the two
    measurements is the only honest way to find that out.

    **Why this is the case to measure it on.** A box held **fully submerged** has an exact
    reference for its mean: `+rho V g`, which `docs/sph-contract.md` makes the standing
    check on `wrench()`'s sign convention, and which the lattice backend measures at
    `+1.000 x` (`PROGRESS_LIVE.md`). So the denominator of the noise fraction is a known
    quantity rather than a measured one, the body never moves, and nothing in the signal
    is the body's own dynamics. A floating hull would mix the solver's noise with a real
    heave response and there would be no way to separate them.

    **Why a force is quieter than a pressure, and why that is a measurement and not a
    prediction.** `benchmarks/README.md`'s `sph/` hydrostatic row has the mechanism:
    `dp/drho` is about `c0^2`, so a density error is `c0^2/(g h)` times itself as a
    fraction of a hydrostatic pressure -- a factor of 400 there. A force is that pressure
    summed over a few hundred boundary particles, which averages much of it away. How much
    is not predictable from `density_error()` alone, because neighbouring particles' errors
    are correlated, so it has to be measured -- which is this case.
    """
    box = _Box(beam=side, length=side, height=side, density_ratio=1.0)
    world = _floating_tank(box, particle_radius=particle_radius, depth=depth, span=span,
                           freeboard=freeboard, device=device, solver=solver)
    started_with = len(world.positions())
    from .units import FLUIDS

    water = FLUIDS["water"]
    celerity = math.sqrt(GRAVITY * depth)
    # Centred in the tank at mid-depth, so that the hull is a full side clear of the bed
    # and of the free surface: a body touching either has a different force on it, and a
    # body under a free surface has the surface's own motion in its wrench.
    link, body, plan = _float_rig(box, world, start_z=depth / 2.0, solver=engine)
    held = link.state.body_q.numpy().copy()

    def hold() -> None:
        link.state.body_qd.zero_()
        link.state.body_q.assign(held)

    settled = _settle(world, reference_speed=celerity, max_steps=int(settle_steps),
                      step=link.step, hold=hold, progress=progress)

    dt = _timestep(world)
    series: list[float] = []
    for index in range(int(samples)):
        link.step()
        hold()
        force, _torque = world.wrench(body)
        series.append(float(np.asarray(force, dtype=float).ravel()[2]))
        if progress is not None and index % max(1, int(samples) // 20) == 0:
            progress(f"    sampling {index:,}/{int(samples):,}  "
                     f"Fz {series[-1] * 1e3:+.4f} mN")

    # **Graded against the volume the solver sampled, not the analytic one, when it can
    # be had.** `benchmarks/README.md`'s falling-sphere row is explicit about why: a
    # buoyancy force is the difference between two numbers of similar size, and using the
    # analytic volume for the weight with the sampled surface for the pressure means the
    # two do not act on the same body. The analytic volume is reported beside it. The
    # contract does not promise this reading, so it is asked for and not required.
    sampled = None
    reader = getattr(world, "volume", None)
    if callable(reader):
        try:
            value = float(reader(body))
        except Exception:
            value = 0.0
        if value > 0.0:
            sampled = value
    analytic = water.rho * box.volume * GRAVITY
    exact = water.rho * (sampled if sampled else box.volume) * GRAVITY
    raw = noise_fraction(series, window=1)
    smoothed = noise_fraction(series, window=int(window))
    reduction = (raw["fraction"] / smoothed["fraction"]
                 if smoothed["fraction"] else float("nan"))
    # One layer of boundary particles over the wetted surface, as a bound on the
    # displacement: `(1 + dp/side)^3 - 1` on the volume, which is what a mean force an
    # honest few per cent off should be compared against before it is called an error.
    layer = (1.0 + 2.0 * particle_radius / side) ** 3 - 1.0
    limit = REFERENCES["force_noise"]["tolerance"]

    advanced = {"steps": int(samples), "dt": dt, "elapsed": int(samples) * dt,
                "last": {"particles": len(world.positions())}}
    health, health_lines = _health(world, advanced, started_with)
    checks = [
        Check("force noise per step, RMS / mean", raw["fraction"],
              f"<= {limit:.0%}", raw["fraction"] <= limit,
              f"**the control requirement, and the bound is a control one rather than a "
              f"physical one.** At 1 kHz with no filtering this is the reciprocal of the "
              f"signal-to-noise ratio a controller sees: {limit:.0%} is 10, and a "
              f"controller cannot use a force whose noise exceeds its signal. "
              f"{raw['samples']:,} samples over {int(samples) * dt * 1e3:.1f} ms"),
        Check(f"force noise after averaging {window} samples", smoothed["fraction"],
              f"<= {limit:.0%}", smoothed["fraction"] <= limit,
              f"what a 100 Hz outer loop sees from this solver: "
              f"{window} steps is {window * dt * 1e3:.2f} ms. The reduction from the raw "
              f"number is **measured at {reduction:.2f}x** and not assumed to be "
              f"sqrt({window}) = {math.sqrt(window):.2f}x -- if the noise is correlated "
              f"step to step, averaging buys less than the square root, and the ratio of "
              f"these two measurements is the only honest way to find out which this is"),
        Check("mean force against rho V g", raw["mean"] * 1e3,
              f"{exact * 1e3:.4f} mN +/- 5%",
              abs(raw["mean"] / exact - 1.0) <= 0.05 if exact else False,
              f"the exact reference this case needs for its denominator: a fully "
              f"submerged body feels `+rho V g`, which docs/sph-contract.md makes the "
              f"standing check on `wrench()`'s sign convention and which the lattice "
              f"backend measures at +1.000x. The sign is part of the check -- the lattice "
              f"had a sampler that negated the force, and only the sign gave it away. "
              + (f"V is the volume the **solver sampled**, "
                 f"{sampled * 1e9:.1f} mm3 against an analytic {box.volume * 1e9:.1f} "
                 f"({sampled / box.volume:.4f}x, so rho V g would be "
                 f"{analytic * 1e3:.4f} mN analytic): a buoyancy force is the difference "
                 f"between two numbers of similar size, and grading the sampled surface "
                 f"against the analytic volume grades the sampler"
                 if sampled else
                 "V is the **analytic** volume, because this world exposes no sampled "
                 "one -- so a few per cent here is the sampler and not the solver, and "
                 "the one-layer bound below is how much")),
        Check("one boundary layer as a bound on the displacement", layer,
              "reported, not graded", None,
              f"an SPH hull is not voxelised, but its Akinci boundary particles occupy a "
              f"layer about one spacing thick, so the effective displaced volume is "
              f"uncertain by about `(1 + dp/L)^3 - 1` = {layer:.2%} here. A mean force "
              f"inside that of `rho V g` is as close as this geometry can be asked to "
              f"get, and the contract does not expose the volume the solver thinks it "
              f"displaced"),
        Check("the tank was at rest before sampling", settled["settled"],
              "True (|u|max <= 2% of the gravity-wave speed)", bool(settled["settled"]),
              f"reached {settled['reached'] * 1e3:.3f} mm/s after {settled['steps']:,} "
              f"steps. Noise measured on a tank that is still settling is the settling"),
        *health,
    ]

    out = Path(out_dir)
    times = np.arange(len(series)) * dt * 1e3
    drawn = [
        figures.profile_figure(
            out / "live-sph-force-noise.png",
            [{"x": times.tolist(), "y": (np.asarray(series) * 1e3).tolist(),
              "label": "Fz per step (mN)"},
             {"x": [times[0], times[-1]] if len(times) else [0.0, 1.0],
              "y": [exact * 1e3] * 2, "label": "rho V g", "style": "--"},
             {"x": (np.arange(smoothed["samples"]) * int(window) * dt * 1e3).tolist(),
              "y": (np.asarray(series[:smoothed["samples"] * int(window)])
                    .reshape(-1, int(window)).mean(axis=1) * 1e3).tolist()
              if smoothed["samples"] else [],
              "label": f"averaged over {window} steps (mN)", "panel": 1}],
            title=f"force noise: {raw['fraction']:.2%} per step, "
                  f"{smoothed['fraction']:.2%} over {window} steps "
                  f"({reduction:.2f}x, sqrt = {math.sqrt(window):.2f}x)",
            xlabel="t (ms)", ylabel="mN", panels=2),
    ]
    for figure in drawn:
        figures.ensure_rgb(figure)

    verdict = ("low enough for a 1 kHz controller to use directly"
               if raw["fraction"] <= limit else
               f"NOT low enough for a 1 kHz controller: {raw['fraction']:.1%} RMS against "
               f"a {limit:.0%} bound, so the force carries less signal than noise. "
               f"Averaged over {window} steps it is {smoothed['fraction']:.1%}, which "
               + ("is usable by a 100 Hz outer loop"
                  if smoothed["fraction"] <= limit else "is still not usable"))
    lines = [f"# sph force noise: a {side * 1e3:g} mm box held fully submerged in "
             f"{depth * 1e3:g} mm of water",
             *_sph_lines(radius=particle_radius, particles=started_with,
                         length=side, depth=depth),
             f"  mean force: {raw['mean'] * 1e3:+.4f} mN against rho V g = "
             f"{exact * 1e3:+.4f} ({raw['mean'] / exact:.4f}x)",
             f"  noise:      {raw['fraction']:.3%} per step, {smoothed['fraction']:.3%} "
             f"over {window} steps ({reduction:.2f}x measured, sqrt({window}) = "
             f"{math.sqrt(window):.2f}x if white)",
             f"  verdict:    {verdict}",
             *health_lines,
             f"  caveat:     {REFERENCES['force_noise']['caveat']}",
             *world.report()]
    return _result("sph_force_noise", checks, lines, drawn, mean=raw["mean"],
                   exact=exact, noise=raw["fraction"], smoothed=smoothed["fraction"],
                   reduction=reduction, layer_bound=layer, verdict=verdict,
                   series=series)


# ======================================================================================
# Convergence, which is what distinguishes a right answer from a lucky one
# ======================================================================================


def run_sph_convergence(*, particle_radius: float = 0.005, ratio: float = 1.5,
                        height: float = 0.05, side: float = 0.064,
                        device: str | None = None, solver: str = "dfsph",
                        engine: str = "mujoco", out_dir: str | Path = ".",
                        progress: Callable[[str], None] | None = None,
                        **_: Any) -> dict[str, Any]:
    """Three particle counts at a constant ratio, on the dam break and the static float.

    One coarse-grid match is luck. `cases.py`'s module docstring makes the point about the
    lattice -- three resolutions is the cheapest way to distinguish "the scheme is
    convergent and this resolution is fine enough" from "this resolution happens to agree"
    -- and a particle method needs it more, not less: its convergence order is lowered by
    the boundary particles and by the free surface, and neither is visible in a single run.

    **The estimate is three-point, and the two-grid form is circular.**
    `PROGRESS_LIVE.md` records this repository publishing a circular version of exactly
    this analysis: a two-grid Richardson estimate needs the exact answer supplied, so
    feeding it the reference and reading back "the limit is the reference" proves nothing
    -- and it returned 1.0000 for three different cases, which is what gave it away. The
    three-point form in `richardson` solves for the order and the limit together and needs
    **no reference at all**, which is what makes the limit evidence rather than an
    assumption. `richardson` refuses two values, and the tests check that it reproduces
    `PROGRESS_LIVE.md`'s own worked estimate -- order 1.232 and limit 1.1382 from the
    lattice dam break's 1.2450, 1.1837 and 1.1576.

    **Three things are graded per quantity.** That the three values are **monotone**,
    because a triple that turns around has no order that fits it and the formula will
    return a number anyway. That the observed **order** is at least 0.5, which is a
    grading choice bracketing "converging slowly" against "not converging" -- the number
    this repository has measured for the same quantity on the lattice is 1.232. And that
    the extrapolated **limit** agrees with the reference at that quantity's own tolerance,
    15% for the dam break and 2% for the draft, with **no widening**: the honest reading
    `PROGRESS_LIVE.md` already gives for the lattice is that the grid-converged answer is
    inside the tolerance and the affordable grids are not, and that is a result rather
    than a failure.

    Particle count goes as the cube of the refinement, so at a ratio of 1.5 the finest run
    is 3.4 times the middle one and 11.4 times the coarsest. This case is a GPU case.
    """
    radii = [particle_radius / ratio ** 2, particle_radius / ratio, particle_radius]
    out = Path(out_dir)
    dam: list[dict[str, Any]] = []
    floats: list[dict[str, Any]] = []
    for index, radius in enumerate(radii):
        if progress is not None:
            progress(f"  dam break at radius {radius * 1e3:.3f} mm "
                     f"({index + 1} of {len(radii)})")
        dam.append(run_sph_dam_break(particle_radius=radius, height=height,
                                     device=device, solver=solver,
                                     out_dir=out / f"r{radius * 1e6:.0f}",
                                     progress=progress))
        if progress is not None:
            progress(f"  static float at radius {radius * 1e3:.3f} mm "
                     f"({index + 1} of {len(radii)})")
        floats.append(run_sph_static_float(particle_radius=radius, beam=side,
                                           height=side / 2.0, device=device,
                                           solver=solver, engine=engine,
                                           out_dir=out / f"r{radius * 1e6:.0f}",
                                           progress=progress))

    # Finest first, which is `richardson`'s order. The quantity is the ratio to the
    # reference rather than the raw value, so that an order extracted from it is an order
    # on the error and the limit is directly comparable with 1.0.
    dam_values = [run["at_dam"] / run["exact_at_dam"] for run in reversed(dam)]
    draft_values = [run["draft"] / run["exact_draft"] for run in reversed(floats)]
    dam_fit = richardson(dam_values, ratio)
    draft_fit = richardson(draft_values, ratio)
    dam_limit = REFERENCES["ritter"]["tolerance"]
    draft_limit = REFERENCES["archimedes"]["tolerance"]
    order_bound = REFERENCES["richardson"]["tolerance"]

    checks = []
    for name, values, fit, bound, reference in (
            ("dam break, h(dam) / Ritter's 4 h0/9", dam_values, dam_fit, dam_limit,
             "Ritter"),
            ("static float, draft / Archimedes", draft_values, draft_fit, draft_limit,
             "Archimedes")):
        spread = ", ".join(f"{value:.4f}" for value in reversed(values))
        checks += [
            Check(f"{name}: the three values are monotone", fit["monotone"], "True",
                  bool(fit["monotone"]),
                  f"coarse to fine: {spread}. **Graded, because a triple that turns "
                  f"around has no order that fits it** and `(f3-f2)/(f2-f1)` is then "
                  f"negative -- the logarithm comes back as nan without comment, and a "
                  f"reader sees a missing number rather than a non-convergent run"),
            Check(f"{name}: observed order", fit["order"],
                  f">= {order_bound:g}",
                  bool(np.isfinite(fit["order"]) and fit["order"] >= order_bound),
                  f"from three radii at a constant ratio of {ratio:g}: "
                  f"{', '.join(f'{r * 1e3:.3f} mm' for r in reversed(radii))}. The bound "
                  f"is a grading choice bracketing 'converging slowly' against 'not "
                  f"converging': SPH is formally second order in the kernel and first "
                  f"order at best with boundary particles and a free surface, and the "
                  f"number measured for this same quantity on the lattice is 1.232"),
            Check(f"{name}: Richardson limit against {reference}", fit["limit"],
                  f"1.0 +/- {bound:.0%}",
                  bool(np.isfinite(fit["limit"]) and abs(fit["limit"] - 1.0) <= bound),
                  f"**the number that matters, and it is graded at the tolerance the "
                  f"case states with no widening.** Solved from the three values with no "
                  f"reference supplied, which is why it is evidence: the two-grid form "
                  f"needs the exact answer and returns it, and PROGRESS_LIVE.md records "
                  f"this repository doing that and getting 1.0000 for three different "
                  f"cases. If the limit passes while the finest grid fails, that is the "
                  f"honest reading of the case -- the grid-converged answer is acceptable "
                  f"and the affordable grids are not"),
        ]

    drawn = [
        figures.profile_figure(
            out / "live-sph-convergence.png",
            [{"x": [r * 1e3 for r in radii],
              "y": [abs(run["at_dam"] / run["exact_at_dam"] - 1.0) * 100.0
                    for run in dam],
              "label": f"dam break h(dam) error, order {dam_fit['order']:.2f}",
              "style": "o-"},
             {"x": [r * 1e3 for r in radii],
              "y": [abs(run["draft"] / run["exact_draft"] - 1.0) * 100.0
                    for run in floats],
              "label": f"draft error, order {draft_fit['order']:.2f}", "style": "s-",
              "panel": 1}],
            title=f"three radii at a ratio of {ratio:g}: limits "
                  f"{dam_fit['limit']:.4f} and {draft_fit['limit']:.4f} of their "
                  f"references",
            xlabel="particle radius (mm)", ylabel="error (%)", panels=2),
    ]
    for figure in drawn:
        figures.ensure_rgb(figure)

    lines = [f"# sph convergence: three radii at a constant ratio of {ratio:g}, on the "
             f"dam break and the static float",
             f"  radii:      {', '.join(f'{r * 1e3:.3f} mm' for r in reversed(radii))} "
             f"(particle count goes as the cube: the finest is {ratio ** 6:.1f}x the "
             f"coarsest)",
             f"  dam break:  "
             + ", ".join(f"{value:.4f}" for value in reversed(dam_values))
             + f" of Ritter's 4h0/9; order {dam_fit['order']:.3f}, limit "
               f"{dam_fit['limit']:.4f}",
             f"  float:      "
             + ", ".join(f"{value:.4f}" for value in reversed(draft_values))
             + f" of Archimedes; order {draft_fit['order']:.3f}, limit "
               f"{draft_fit['limit']:.4f}",
             f"  method:     {REFERENCES['richardson']['source']}",
             f"  caveat:     {REFERENCES['richardson']['caveat']}"]
    return _result("sph_convergence", checks, lines, drawn, radii=radii, ratio=ratio,
                   dam=dam_fit, draft=draft_fit, dam_values=dam_values,
                   draft_values=draft_values)


# -- the catalogue ----------------------------------------------------------------------


SPH_CASES: dict[str, Case] = {
    "sph_hydrostatic": Case(
        "sph_hydrostatic",
        "a still column of water: the particle pressures against `rho g h`, and the "
        "density error over the whole run, max and mean",
        run_sph_hydrostatic, {"particle_radius": 0.0025}, milestone="M5"),
    "sph_dam_break": Case(
        "sph_dam_break",
        "a column collapses on a dry bed: the front position over time, graded against "
        "Ritter (1892) and with Martin and Moyce reported in both conventions and graded "
        "in neither",
        run_sph_dam_break, {"particle_radius": 0.0025}, milestone="M5"),
    "sph_sloshing": Case(
        "sph_sloshing",
        "a tank's first mode released through its velocity field: the period against "
        "`omega^2 = g k tanh(k h)`, from two estimators with their disagreement graded",
        run_sph_sloshing, {"particle_radius": 0.00125}, milestone="M7"),
    "sph_static_float": Case(
        "sph_static_float",
        "a box of half water's density released at the surface: the draft against "
        "Archimedes within 2%, with no analytic buoyancy term anywhere",
        run_sph_static_float, {"particle_radius": 0.002}, milestone="M6",
        needs_newton=True, heavy=True),
    "sph_heave_decay": Case(
        "sph_heave_decay",
        "the same box pushed below its draft and released: the natural period against "
        "`2 pi sqrt(d/g)`, graded one-sided because added mass can only lengthen it",
        run_sph_heave_decay, {"particle_radius": 0.002}, milestone="M6",
        needs_newton=True, heavy=True),
    "sph_heel_stability": Case(
        "sph_heel_stability",
        "the boat benchmark: one hull that cannot float upright and one that can, against "
        "the derived metacentric criterion; Bogner and Rude (2013) could not be read and "
        "is reported, not graded",
        run_sph_heel_stability, {"particle_radius": 0.002}, milestone="M6",
        needs_newton=True, heavy=True),
    "sph_force_noise": Case(
        "sph_force_noise",
        "the RMS fluctuation of the force on a static submerged body as a fraction of its "
        "mean, per step and averaged, against what a 1 kHz controller can use",
        run_sph_force_noise, {"particle_radius": 0.002}, milestone="M6",
        needs_newton=True, heavy=True),
    "sph_convergence": Case(
        "sph_convergence",
        "three particle counts at a constant refinement ratio on the dam break and the "
        "static float, with the observed order and a three-point Richardson limit that "
        "needs no reference",
        run_sph_convergence, {"particle_radius": 0.005}, milestone="M6",
        needs_newton=True, heavy=True),
}
"""The SPH cases, in `cases.py::CASES`' shape so that wiring them in is one import.

    from .sph_cases import SPH_CASES
    CASES.update(SPH_CASES)

and `run_case`, `describe_cases`, the CLI and the agent tool all work unchanged. They are
a separate catalogue until the solver lands, so that `openreynolds live cases` does not
offer a list of cases that cannot run.
"""


def describe_sph_cases() -> list[str]:
    """One line a case, the same layout `cases.describe_cases` prints."""
    width = max(len(name) for name in SPH_CASES)
    return [
        f"{case.name.ljust(width)}  {case.milestone or '--':>3}  {case.about}"
        f"{'  [needs a GPU to be quick]' if case.heavy else ''}"
        for case in SPH_CASES.values()
    ]


def run_sph_case(name: str, *, out_dir: str | Path = ".", device: str | None = None,
                 progress: Callable[[str], None] | None = None,
                 **overrides: Any) -> dict[str, Any]:
    """Run one SPH case by name. Unknown keyword arguments are ignored, as in `cases.py`.

    Ignored rather than refused for `cases.run_case`'s stated reason: the CLI and the
    agent tool pass a fixed set of options and the cases take different subsets of them,
    so a case that does not take `fluid` should run rather than raise at a caller who had
    no way to know. Every runner ends its signature with `**_` for that.

    This exists so that every graded number in this module is reproducible from one call
    before the catalogues are merged, which is the house rule. Once they are --
    `CASES.update(SPH_CASES)` -- `cases.run_case` does the same thing and this can go.
    """
    try:
        case = SPH_CASES[name]
    except KeyError:
        raise UnphysicalScaling(
            f"no SPH case called {name!r}. The cases are: {', '.join(sorted(SPH_CASES))}"
        ) from None
    settings = dict(case.defaults)
    resolution = overrides.pop("resolution", None)
    if resolution:
        # One knob for the discretisation, as the lattice cases have. A particle radius is
        # not a cell count, so `resolution` is read as "particle spacings across the
        # case's own depth" and converted, which is the only reading that means the same
        # thing on both backends. An explicit `particle_radius` still wins.
        settings["particle_radius"] = _radius_for(case, int(resolution))
    settings.update({k: v for k, v in overrides.items() if v is not None})
    settings["out_dir"] = out_dir
    settings["device"] = device
    settings["progress"] = progress
    began = time.perf_counter()
    result = case.runner(**settings)
    result["seconds"] = time.perf_counter() - began
    result["milestone"] = case.milestone
    result["about"] = case.about
    result["needs"] = list(NEEDS_FROM_THE_SOLVER)
    return result


def _radius_for(case: Case, spacings: int) -> float:
    """A particle radius from "this many spacings across the case's own depth".

    The lattice's `resolution` is a cell count and a particle radius is a length, so the
    two cannot be the same number. What *is* the same is the resolution across the one
    length each case is about -- the depth, the column height, the draft -- and the
    runners' own defaults name it. Read from the signature so that a case which changes
    its default depth does not need this function changed too.
    """
    import inspect

    defaults = inspect.signature(case.runner).parameters
    for key in ("depth", "height", "side"):
        parameter = defaults.get(key)
        if parameter is not None and parameter.default is not inspect.Parameter.empty:
            return float(parameter.default) / (2.0 * max(int(spacings), 1))
    raise UnphysicalScaling(
        f"{case.name} has no depth, height or side in its signature to size a particle "
        f"radius against; pass particle_radius directly"
    )
