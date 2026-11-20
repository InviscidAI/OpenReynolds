#!/usr/bin/env python3
"""The same quantity on two or more meshes, turned into a number with an error bar.

Everything else in here reports what one mesh did. A quantity measured on one mesh
carries no statement about how much of it is the mesh, and the honest form of a CFD
answer is a value and a band -- Cd = 0.412 +/- 0.004 from discretisation -- not a
value alone. Today the only error bar the product produces anywhere is
`reattach.py`'s reattachment position plus or minus half the local face spacing.
This is the other one: the band that comes from the grid.

The arithmetic is Richardson extrapolation and Roache's Grid Convergence Index. Three
meshes of the same case give the order of convergence the solver actually achieved,
the value the sequence is heading for, and a percentage band on the finest answer. Two
meshes give a band on an assumed order, with the larger safety factor that admits how
much less that is.

what a naive implementation gets wrong

Those formulas are four lines each and they mislead in four ways that happen
constantly on real triplets, which is most of what this file is:

- **Oscillatory convergence.** Richardson presumes the three values approach their
  limit from one side. When the middle value is not between the other two, the
  extrapolation is arithmetic on a sequence that does not have the shape it assumes,
  and it will still print a confident number. Here that is reported as oscillatory,
  the extrapolation is refused, and what comes out instead is the spread across the
  ladder as an interval -- which is a weaker claim and is the true one.
- **Non-integer refinement ratios.** `p = ln(eps32/eps21) / ln(r)` needs one r. Real
  ladders are 2.1 and 1.8, because a mesher is asked for a cell size and delivers a
  cell count. The two ratios are kept separate and p is solved for out of Celik's
  relation, which reduces to the textbook formula exactly when the ratios are equal.
- **A near-zero denominator.** Two meshes that agree to the printed digit make
  `eps21` zero, p infinite, and the GCI either zero or a division error depending on
  the rounding. Mesh independence reached is the good outcome, and it is reported as
  that rather than as an order of convergence.
- **An absurd observed p.** A triplet can yield p = -3 or p = 11. Those are not
  orders of convergence, they are the sequence telling you something other than
  discretisation is moving -- an unconverged solve, a changed boundary treatment, a
  quantity that is not grid-sensitive at all. The number is printed, said to be
  unusable, and the GCI beside it is computed on the scheme's formal order with the
  clamp named in the output.

Each of those four is a test in `tests/test_toolbox_grid.py`, alongside the worked
example from NASA Glenn's spatial-convergence tutorial, which is the published case
the arithmetic in here is checked against rather than against itself.

why there are guards on the inputs and not only on the arithmetic

Those four are all read off the values. Three more are read off where the values came
from, and they are here because the script met a real study and printed
`Cd = 0.27556258 +/- 3.240` with a Richardson limit of -0.80441086 and a GCI of
1175.75% as the answer, on a pair of cases that happened to be in one directory
(issue #142):

- **A refinement ratio below 1.3.** Celik's relation and Roache's factors are
  calibrated on r >= 1.3. At the measured r = 1.1302, `r**p - 1` is 0.277, and a
  difference of 0.3 in Cd divided by that is the 1175%. See `MIN_RATIO`.
- **A value that is not a converged value.** The quantity there was the last row of a
  four-row history -- a run that aborted during its startup transient. See
  `MIN_HISTORY_ROWS`.
- **An extrapolation whose sign no input shares.** A negative drag coefficient is not a
  limit. A value outside the range of every number it was formed from disqualifies
  itself.

Each is reported as the measurement it is -- the ratio, the row count, the sign change
-- with the ladder and the values still printed, and the band withheld. A band of
+/- 3.240 on 0.2756 is worse than no band, because it looks like an answer. The
arithmetic those guards sit in front of is untouched: a well-posed ladder gets exactly
the numbers it got before.

why the safety factor is in the output

A GCI is a factor times a relative error. The conventional factors are 1.25 where
three grids gave an observed p and 3.0 where two grids assumed one, and a band quoted
without saying which is a band whose size is a matter of opinion nobody stated. Both
the value and the fact that it is the default appear in the report.

why the dimensionality is a flag

Cell counts become a representative spacing h through h ~ N**(-1/3) in 3D and
N**(-1/2) in 2D, and getting that wrong moves p by a factor of 1.5. A plane case in
OpenFOAM is a one-cell-thick 3D mesh, so nothing about the cell count itself says
which it is. `--dim` says it; `--dim auto` reads it off the mesh, where an `empty`
patch in `constant/polyMesh/boundary` means a plane case, the same reading
`case_gen.py` takes as a constraint rather than a guess. The report names which route
was used, because a detected 2D and a declared 2D are not equally strong.

where the numbers come from

Cell counts are `checkMesh` output through `mesh_digest.parse`. Quantities are
`postProcessing/*.dat` through `results.read_history`, which is the sibling that
stitches a restarted series back into one history and drops the rows a later run
superseded, so neither half a run's coefficients nor an abandoned run's tail reaches
the ladder. Doing that grouping here instead was issue #146 and 1.5% on a real case.
Every number in the report is printed beside the case, the series, the file and the
column it came from. A grid ladder is the one place where reading the wrong column on
one of three cases produces a plausible answer -- and where one column name sits in
two series, `--quantity series:column` says which.

This script runs no solver, writes no case and applies no threshold: whether a 3% band
is acceptable depends on what the number is for. The exit code is 0 whatever it finds,
as for everything else here. It does refuse at the command line -- exit 2, argparse's
own -- when fewer than two grids are given or when a dimensionality is needed and
neither given nor readable, because there is no reporting to do on an input that
cannot form an h at all, and a script that invented one would be inventing the input
the rest of the answer rests on.

    python3 grid.py --grid 40000:0.431 --grid 160000:0.418 --grid 640000:0.412 --dim 2
    python3 grid.py --spacing 1:0.97050 --spacing 2:0.96854 --spacing 4:0.96178
    python3 grid.py coarse medium fine --quantity Cd --dim auto
    python3 grid.py coarse fine --quantity Cd --dim 3 --safety-factor 3.0
    python3 grid.py L1 L2 L3 --quantity torqueOuter/moment.dat:viscous_z --dim auto
    python3 grid.py c1 c2 c3 --quantity Cd --dim auto --tail 200 --json gci.json --png gci.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mesh_digest
import study_state

FORMAL_ORDER = 2.0
"""The formal order of the schemes this is used with unless told otherwise. OpenFOAM's
usual steady setup -- `Gauss linear` on the divergence terms, `linear` interpolation,
`corrected` on the Laplacian -- is second order, and so is every scheme a validation
case is run with on purpose."""

SAFETY_THREE_GRIDS = 1.25
SAFETY_TWO_GRIDS = 3.0
"""Roache's factors. 1.25 is for a GCI over three grids whose p was observed; 3.0 is
for two grids, where p was assumed and the band has to cover the assumption too."""

EQUAL_RELATIVE = 1e-12
"""Two values this close, relative to the size of the quantity, are one value. The
difference between them is then arithmetic noise, and an order of convergence computed
from noise is noise raised to a logarithm. Deliberately tight: a generous tolerance
here would refuse real triplets whose convergence is merely good."""

ABSURD_ORDER_FACTOR = 2.0
"""An observed p above this multiple of the formal order is reported and not used. A
second-order scheme can show p = 2.4 on a coarse ladder and still be telling the truth;
p = 11 is a sequence being moved by something that is not the grid."""

ORDER_LIMIT = 50.0
"""The bracket the solve for p searches, in both directions. Nothing outside it is a
real order and the formulas overflow somewhere past it. The negative half is searched
on purpose: a p of -2.3 is one of the outcomes this file exists to report, and a solver
that only looked at positive p would come back with "no order found" and hide the most
diagnostic thing the triplet had to say."""

MIN_RATIO = 1.3
"""The smallest refinement ratio these formulas mean anything on.

Celik et al. 2008 -- the paper `_order_residual` cites for the observed-order relation
-- asks for r >= 1.3, and Roache's safety factors are calibrated on the same
assumption. Below it the order solve is ill-conditioned and `r**p - 1` is small enough
that the GCI reports the vanishing denominator rather than an uncertainty.

This is here because of issue #142: on a real pair at r = 1.1302, `r**p - 1 = 0.277`
turned a difference of 0.2996 in Cd into a band of +/- 3.240 on a value of 0.2756, and
a GCI of 1175.75% was printed as the answer. That number is arithmetic, not doubt."""

MIN_HISTORY_ROWS = 10
"""Fewer rows than this in a quantity's history and it is a run that was aborted, not
one that settled.

Also issue #142. The `case2` of that ladder had a four-row history, all of it startup
transient, and its last row was quoted as a converged Cd. The two converged runs in the
same corpus have 3,006 and 8,586 rows, so ten separates an aborted run from a real one
by two orders of magnitude on either side. That is all it does: it detects a run that
never ran. It is not a convergence test, and a long history is not evidence of
convergence -- `--tail` and the solver log are where that question is asked."""


# -- the arithmetic ------------------------------------------------------------
#
# Pure functions on numbers, with no case, no file and no OpenFOAM anywhere near them,
# which is what lets the published example be checked in a unit test.


def representative_spacing(cells: float, dimension: int) -> float:
    """A cell count as a representative grid spacing h.

    h ~ N**(-1/d). It is a proportionality and the constant does not matter: every
    formula below uses only ratios of h, so an h of 0.0215 and an h of 21.5 give the
    same p, the same extrapolation and the same GCI. That is also why a ladder given
    in normalised spacings (1, 2, 4) and the same ladder given in cell counts come out
    identical, which is how the published example below is run against cell counts it
    never published.
    """
    if cells <= 0:
        raise ValueError(f"a cell count of {cells} is not a mesh")
    return float(cells) ** (-1.0 / dimension)


def _order_residual(p: float, r21: float, r32: float, ratio: float) -> float:
    """Celik's relation for the observed order, as a function whose root is p.

        p = ln|eps32/eps21| / ln(r21)  +  q(p) / ln(r21)
        q(p) = ln( (r21**p - 1) / (r32**p - 1) )

    The q term is what admits two different refinement ratios. When r21 == r32 it is
    ln(1) = 0 exactly -- not approximately, since the numerator and denominator are
    then the same float -- and what is left is the textbook three-grid formula. So the
    general path and the equal-ratio path are one path, and the equal-ratio case is not
    a special case anybody has to remember to take.

    Reference: Celik et al., "Procedure for Estimation and Reporting of Uncertainty Due
    to Discretization in CFD Applications", J. Fluids Eng. 130(7), 2008, eq. (3).
    """
    q = math.log((r21**p - 1.0) / (r32**p - 1.0))
    return math.log(r21) * p - math.log(ratio) - q


def solve_order(r21: float, r32: float, ratio: float) -> float | None:
    """The observed order of convergence, by bracketing and bisection.

    scipy is not on the image, so this is written out. It is also the better tool for
    the job than a root finder would be: a scan for a sign change over the whole
    plausible range cannot be walked off a cliff by a bad initial guess the way
    Newton's method on a logarithmic residual can, and a triplet with no root in the
    range is a real outcome that gets reported as such rather than as a number from a
    solve that quietly failed to converge.

    `ratio` is |eps32 / eps21| and must be positive: a negative one is oscillatory
    convergence, which has no order and is caught before this is called.
    """
    if ratio <= 0:
        return None

    def f(p: float) -> float | None:
        try:
            return _order_residual(p, r21, r32, ratio)
        except (ValueError, ZeroDivisionError, OverflowError):
            return None

    # 4000 steps across [-50, 50]: fine enough that no real triplet hides a sign change
    # between two samples, and 4000 evaluations of three logarithms costs nothing. At
    # p = 0 both r**p - 1 terms are zero and `f` returns None, which steps over it; the
    # residual is finite either side.
    steps = 4000
    span = 2.0 * ORDER_LIMIT / steps
    lo_p = -ORDER_LIMIT
    lo = f(lo_p)
    for i in range(1, steps + 1):
        hi_p = -ORDER_LIMIT + span * i
        hi = f(hi_p)
        if lo is not None and lo == 0.0:
            return lo_p
        if lo is not None and hi is not None and (lo < 0.0) != (hi < 0.0):
            for _ in range(200):
                mid_p = 0.5 * (lo_p + hi_p)
                mid = f(mid_p)
                if mid is None:
                    return None
                if (mid < 0.0) == (lo < 0.0):
                    lo_p, lo = mid_p, mid
                else:
                    hi_p, hi = mid_p, mid
            return 0.5 * (lo_p + hi_p)
        lo_p, lo = hi_p, hi
    return None


def gci_percent(fine: float, coarse: float, ratio: float, order: float,
                safety: float) -> float | None:
    """Roache's Grid Convergence Index on the fine grid of a pair, as a percentage.

        GCI = Fs * |(coarse - fine) / fine| / (r**p - 1) * 100

    None rather than a number where the fine value is zero: the GCI is a relative
    error and there is nothing to be relative to. The caller reports the absolute
    difference instead, which is the honest remainder. `reattach.py` returns None for
    its error bar in the same situation rather than a generous default, and says in the
    report that the bar is missing.
    """
    denominator = ratio**order - 1.0
    if fine == 0.0 or denominator <= 0.0:
        return None
    return safety * abs((coarse - fine) / fine) / denominator * 100.0


def richardson(fine: float, coarse: float, ratio: float, order: float) -> float | None:
    """The value the sequence is heading for: f1 + (f1 - f2) / (r**p - 1).

    Written with the fine value first because that is the one being corrected. The
    correction is small when r**p is large, which is the whole statement: a fine mesh
    and a high order leave little to extrapolate.
    """
    denominator = ratio**order - 1.0
    if denominator <= 0.0:
        return None
    return fine + (fine - coarse) / denominator


def analyse(grids: list[dict[str, Any]], formal_order: float = FORMAL_ORDER,
            safety: float | None = None) -> dict[str, Any]:
    """A ladder of grids, finest first, turned into the whole report in one dict.

    Each grid is `{"h": float, "value": float, "label": str}`. Two give a two-grid
    band on the assumed order. Three or more give the observed order; the finest three
    are the answer and every other consecutive triplet is reported beside it, because
    a p that moves from 1.9 on the fine triplet to 0.4 on the coarse one says the
    coarse end of the ladder is not in the asymptotic range, and that is worth seeing
    rather than inferring.
    """
    if len(grids) < 2:
        raise ValueError("a grid-convergence estimate needs at least two grids")
    ordered = sorted(grids, key=lambda g: g["h"])
    pairs = [_pair(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1)]

    if len(ordered) == 2:
        factor = SAFETY_TWO_GRIDS if safety is None else safety
        primary = _two_grid(ordered[0], ordered[1], formal_order, factor)
        triplets: list[dict[str, Any]] = []
    else:
        factor = SAFETY_THREE_GRIDS if safety is None else safety
        triplets = [
            _triplet(ordered[i], ordered[i + 1], ordered[i + 2], formal_order, factor)
            for i in range(len(ordered) - 2)
        ]
        primary = triplets[0]

    return {
        "grids": ordered,
        "pairs": pairs,
        "triplets": triplets,
        "primary": primary,
        "formal_order": formal_order,
        "safety_factor": factor,
        "safety_factor_is_default": safety is None,
        "spread": max(g["value"] for g in ordered) - min(g["value"] for g in ordered),
    }


def _pair(fine: dict[str, Any], coarse: dict[str, Any]) -> dict[str, Any]:
    return {
        "fine": fine["label"],
        "coarse": coarse["label"],
        "ratio": coarse["h"] / fine["h"],
        "difference": coarse["value"] - fine["value"],
    }


def _two_grid(fine: dict[str, Any], coarse: dict[str, Any], formal_order: float,
              safety: float) -> dict[str, Any]:
    """Two grids: no order can be observed, so the formal order is assumed and said to
    have been assumed. The band is real but it is a band on an assumption, which is
    what the 3.0 safety factor is for."""
    ratio = coarse["h"] / fine["h"]
    result = {
        "mode": "two-grid",
        "grids": [fine["label"], coarse["label"]],
        "ratio_21": ratio,
        "eps_21": coarse["value"] - fine["value"],
        "fine_value": fine["value"],
        "order_observed": None,
        "order_used": formal_order,
        "order_source": "assumed: the scheme's formal order, since two grids observe none",
        "order_usable": True,
        "convergence": "unknown: two grids cannot show a trend",
        "safety_factor": safety,
    }
    equal = _equal_values(result["eps_21"], [fine["value"], coarse["value"]])
    if equal:
        result.update(_no_change(result["eps_21"]))
        return result
    if ratio <= 1.0:
        result["note"] = (
            f"the two grids have the same h to within the ratio {ratio:.6g}, so there "
            "is no refinement between them to measure anything over"
        )
        return result
    result["extrapolated"] = richardson(fine["value"], coarse["value"], ratio, formal_order)
    result["gci_fine_percent"] = gci_percent(
        fine["value"], coarse["value"], ratio, formal_order, safety)
    result.update(_band(result))
    result.update(_disqualify(result, [fine, coarse]))
    return result


def _triplet(fine: dict[str, Any], medium: dict[str, Any], coarse: dict[str, Any],
             formal_order: float, safety: float) -> dict[str, Any]:
    """Three grids: the whole analysis, including every way it can decline to answer."""
    r21 = medium["h"] / fine["h"]
    r32 = coarse["h"] / medium["h"]
    eps21 = medium["value"] - fine["value"]
    eps32 = coarse["value"] - medium["value"]
    values = [fine["value"], medium["value"], coarse["value"]]

    result: dict[str, Any] = {
        "mode": "three-grid",
        "grids": [fine["label"], medium["label"], coarse["label"]],
        "ratio_21": r21,
        "ratio_32": r32,
        "eps_21": eps21,
        "eps_32": eps32,
        "fine_value": fine["value"],
        "value_range": [min(values), max(values)],
        "safety_factor": safety,
        "order_observed": None,
        "order_used": None,
        "order_usable": False,
        "order_source": None,
        "extrapolated": None,
        "gci_fine_percent": None,
        "gci_coarse_percent": None,
    }

    if r21 <= 1.0 or r32 <= 1.0:
        result["convergence"] = "no refinement"
        result["note"] = (
            f"the refinement ratios are {r21:.6g} and {r32:.6g}; a ladder needs each "
            "grid finer than the last, and these are not"
        )
        return result

    if _equal_values(eps21, values) or _equal_values(eps32, values):
        result["convergence"] = "no change"
        result.update(_no_change(eps21 if _equal_values(eps21, values) else eps32))
        return result

    if (eps21 > 0.0) != (eps32 > 0.0):
        # The middle value is not between the other two. Richardson's derivation --
        # a single leading error term of fixed sign shrinking as h**p -- does not
        # describe this sequence, so the extrapolation would be arithmetic performed
        # on an assumption the data has already contradicted.
        result["convergence"] = "oscillatory"
        result["note"] = (
            f"the change reverses sign across the triplet ({eps21:+.6g} then "
            f"{eps32:+.6g}), so the middle value is not between the other two. "
            "Richardson extrapolation presumes a single error term approached from "
            "one side and there is no such approach here, so no extrapolated value "
            "and no GCI are reported. What the ladder supports instead is the "
            f"interval: the three values span {min(values):.6g} to {max(values):.6g}, "
            f"a spread of {max(values) - min(values):.6g}. Another rung on the fine "
            "end is what would settle it."
        )
        return result

    result["convergence"] = "monotone"
    order = solve_order(r21, r32, abs(eps32 / eps21))
    result["order_observed"] = order

    if order is None:
        result["order_used"] = formal_order
        result["order_source"] = (
            f"clamped to the formal order {formal_order:g}: no order solved out of "
            "these ratios and differences"
        )
        result["note"] = (
            "the relation for the observed order has no root in "
            f"(0, {ORDER_LIMIT:g}] for these ratios, so the GCI below is on the "
            "formal order and is an estimate rather than a measurement"
        )
    elif order <= 0.0 or order > ABSURD_ORDER_FACTOR * formal_order:
        result["order_used"] = formal_order
        result["order_source"] = f"clamped to the formal order {formal_order:g}"
        result["note"] = (
            f"the observed order is {order:.6g}, which is not a usable order of "
            f"convergence for a scheme whose formal order is {formal_order:g} -- "
            + ("a negative order means the sequence is diverging as the mesh refines"
               if order <= 0.0 else
               f"anything above {ABSURD_ORDER_FACTOR * formal_order:g} is steeper than "
               "the scheme can be")
            + ". Something other than discretisation is moving this quantity: a solve "
            "that has not converged, a boundary treatment that changed with the mesh, "
            "or a quantity barely sensitive to the grid at all. The number is reported "
            "because it is the evidence; the GCI below is computed on the formal order "
            "instead, with that clamp declared, and is an estimate."
        )
    else:
        result["order_used"] = order
        result["order_usable"] = True
        result["order_source"] = "observed from the three grids"

    used = result["order_used"]
    result["extrapolated"] = richardson(fine["value"], medium["value"], r21, used)
    result["gci_fine_percent"] = gci_percent(
        fine["value"], medium["value"], r21, used, safety)
    result["gci_coarse_percent"] = gci_percent(
        medium["value"], coarse["value"], r32, used, safety)
    result.update(_band(result))

    # Roache's check that the ladder is in the asymptotic range: the coarse GCI should
    # be r**p times the fine one, so this ratio should be near 1. Reported, never
    # enforced: how near is near depends on how much of the ladder is being trusted.
    fine_gci, coarse_gci = result["gci_fine_percent"], result["gci_coarse_percent"]
    if fine_gci and coarse_gci:
        result["asymptotic_ratio"] = coarse_gci / (r21**used * fine_gci)

    result.update(_disqualify(result, [fine, medium, coarse]))
    return result


def _equal_values(difference: float, values: list[float]) -> bool:
    scale = max(abs(v) for v in values) or 1.0
    return abs(difference) <= EQUAL_RELATIVE * scale


def _no_change(difference: float) -> dict[str, Any]:
    """Two of the values are the same value. This is the good outcome and it is not an
    order of convergence: the denominator `eps21` is zero, so nothing downstream of it
    is defined. Said as what it is rather than divided by."""
    return {
        "note": (
            f"two of these grids give the same value to within {difference:+.3g}, "
            f"which is below {EQUAL_RELATIVE:g} of the size of the quantity. No order "
            "of convergence can be observed from a difference that is not there, and "
            "nothing is divided by it. Read it as mesh independence reached over that "
            "step: the discretisation band on the finer of the two is smaller than the "
            "difference between them, which is to say smaller than can be shown here."
        ),
        "order_usable": False,
    }


def _band(result: dict[str, Any]) -> dict[str, Any]:
    """The GCI percentage as an absolute band on the fine value, which is the form the
    number gets quoted in. Where the GCI could not be formed -- a fine value of zero --
    the absolute difference between the two finest grids stands in, labelled as the
    weaker thing it is."""
    gci = result.get("gci_fine_percent")
    if gci is not None:
        return {"band": abs(result["fine_value"]) * gci / 100.0, "band_source": "GCI"}
    return {
        "band": abs(result["eps_21"]),
        "band_source": (
            "the difference between the two finest grids: the fine value is zero, so a "
            "relative GCI has nothing to be relative to"
        ),
    }


def _disqualify(result: dict[str, Any], grids: list[dict[str, Any]]) -> dict[str, Any]:
    """Facts about the inputs that make the band meaningless, each with its number.

    The formulas above are domain-limited and nothing was checking the domain. Issue
    #142: two real cases of one study were handed over as a ladder, and out came
    `Cd = 0.27556258 +/- 3.240` with an extrapolated -0.80441086 and a GCI of 1175.75%,
    printed in the report's "the answer" block, which is the part that gets quoted.
    Three disqualifying facts were in the script's own hands and all three were
    ignored: a refinement ratio of 1.1302, a value taken from a four-row history, and
    an extrapolated drag coefficient whose sign no input shares.

    So the inputs, the refinement and the reason are still printed -- a reader has to
    be able to check the refusal -- and the extrapolation, the GCI and the band are
    not. A band of +/- 3.240 on 0.2756 is worse than no band, because it looks like an
    answer; the refusal is the answer. Each reason states the measurement it is read
    off rather than only that something was wrong, because "r is too low" is an opinion
    and "r = 1.1302, below 1.3" is a fact the reader can act on.

    Nothing here touches the arithmetic. A ladder that passes these checks gets exactly
    the numbers it got before, which is why the published worked example is unchanged.
    """
    reasons: list[str] = []
    order = result.get("order_used") or FORMAL_ORDER

    # A ratio of exactly 1.3 arrives as 1.2999999999999998, because it is a quotient of
    # two h values that are themselves quotients. The slack is there so the threshold
    # admits the ratio it is calibrated on rather than the next float below it.
    floor = MIN_RATIO * (1.0 - 1e-9)
    low = [result[key] for key in ("ratio_21", "ratio_32")
           if result.get(key) is not None and result[key] < floor]
    if low:
        worst = min(low)
        reasons.append(
            f"the refinement ratio is {', '.join(f'{r:.4f}' for r in low)}, below the "
            f"{MIN_RATIO:g} that Roache's factors and Celik's order relation are "
            f"calibrated on. At r = {worst:.4f} and p = {order:g} the denominator "
            f"r**p - 1 is {worst**order - 1.0:.4g}, so what a GCI would report here is "
            "that denominator vanishing and not an uncertainty."
        )

    for item in grids:
        rows = item.get("rows")
        if rows is None or rows >= MIN_HISTORY_ROWS:
            continue
        reasons.append(
            f"{item['label']}'s value is the mean of the last "
            f"{item.get('rows_averaged', 1)} of {rows} rows of its history. A run with "
            f"{rows} rows written has not settled -- it is startup transient, or a run "
            f"that stopped -- and anything under {MIN_HISTORY_ROWS} rows is not a "
            "converged value and cannot carry a grid-convergence claim."
        )

    extrapolated = result.get("extrapolated")
    values = [item["value"] for item in grids]
    if extrapolated is not None and (
            (all(v > 0.0 for v in values) and extrapolated < 0.0)
            or (all(v < 0.0 for v in values) and extrapolated > 0.0)):
        reasons.append(
            f"the extrapolation is {extrapolated:.8g}, and its sign differs from every "
            f"value it was formed from ({', '.join(f'{v:.6g}' for v in values)}). A "
            "limit on the far side of zero from the whole ladder -- a negative drag "
            "coefficient out of a ladder of positive ones -- is a result that "
            "disqualifies itself rather than one to quote."
        )

    if not reasons:
        return {}
    return {
        "disqualified": reasons,
        "extrapolated": None,
        "gci_fine_percent": None,
        "gci_coarse_percent": None,
        "asymptotic_ratio": None,
        "band": None,
        "band_source": None,
        "order_usable": False,
    }


# -- reading it off the cases --------------------------------------------------

CHECKMESH_GLOBS = ("log.checkMesh*", "checkMesh.log", "log.checkmesh*", "*checkMesh*.log")


def cells_from_case(case: Path) -> tuple[int | None, str]:
    """A case's cell count and where it was read: a `checkMesh` log, the mesh's own
    header, or a `checkMesh` run, in that order.

    A log already in the case directory is preferred over running anything: the mesh
    this ladder is about may be on a machine with no OpenFOAM, and a log written when
    the mesh was built is the same number. `checkMesh` is run only where no log is
    there and the binary is, and the report says which happened, because a cell count
    is the x axis of everything below it.

    The mesh header sits between those two, and it is here because of #171. Only logs
    were looked at, so a real three-grid ladder whose finest rung had no checkMesh log
    became a two-grid estimate on the wrong grid with a band 2.1x too wide, and a
    second real ladder with no log on any of its three rungs was refused outright --
    while all six cell counts were on disk one file away, in the
    `constant/polyMesh/owner` note OpenFOAM writes (`nPoints:94080 nCells:46080 ...`),
    in the same directory as the `boundary` that `--dim auto` read successfully in
    those very cases. `study_state.owner_ncells` is that read, a regex over the first
    2 kB, and it is the same number: it is what the mesh says about itself.
    """
    for pattern in CHECKMESH_GLOBS:
        for path in sorted(case.glob(pattern)) + sorted((case / "logs").glob(pattern)):
            counts = mesh_digest.parse(path.read_text(errors="replace"))["counts"]
            if "cells" in counts:
                return counts["cells"], str(path)

    count = study_state.owner_ncells(case)
    if count is not None:
        return count, f"the nCells note in {case / 'constant' / 'polyMesh' / 'owner'}"

    try:
        done = subprocess.run(["checkMesh", "-case", str(case)], capture_output=True,
                              text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, (f"no checkMesh log in {case}, no nCells note in its "
                      f"constant/polyMesh/owner, and checkMesh did not run ({exc})")
    counts = mesh_digest.parse(done.stdout + done.stderr)["counts"]
    if "cells" in counts:
        return counts["cells"], f"checkMesh run here on {case}"
    return None, (f"no cell count in any checkMesh log, in constant/polyMesh/owner's "
                  f"note, or in a checkMesh run, for {case}")


def dimension_from_case(case: Path) -> tuple[int | None, str]:
    """2 or 3, read off `constant/polyMesh/boundary`.

    An `empty` patch is OpenFOAM's statement that a direction is not solved in, which
    is what a plane case is -- one cell thick, with the two z faces typed `empty`.
    `case_gen.py` reads the same type as a constraint rather than a guess, for the same
    reason: getting it wrong stops the solver rather than misleading it. Here it is
    offered as a convenience and the route is named in the report, since a mesh with no
    `empty` patch is 3D and a mesh whose boundary file cannot be read is unknown, and
    those two are not the same answer.
    """
    path = case / "constant" / "polyMesh" / "boundary"
    if not path.exists():
        return None, f"no {path} to read a dimensionality from"
    text = path.read_text(errors="replace")
    body = text[text.find("("): text.rfind(")") + 1] if "(" in text else ""
    names = [match.group(1) for match in re.finditer(r"(\w+)\s*\{([^}]*)\}", body)
             if re.search(r"\btype\s+empty\s*;", match.group(2))]
    if names:
        return 2, f"2D: {path} types {', '.join(names)} as empty"
    return 3, f"3D: no patch in {path} is typed empty"


def quantity_from_case(case: Path, quantity: str, tail: int = 1) -> dict[str, Any]:
    """One quantity out of a case's `postProcessing`, with its provenance.

    `results.read_history` does the reading and the stitching, which is the whole point
    of calling it rather than concatenating the files here. A run restarted at 0.2
    writes `forceCoeffs_0.2.dat` beside `forceCoeffs.dat`, and reading only one of them
    is how half a ladder's rungs end up quoting different parts of their runs -- but
    merging all of them is worse, because a run restarted from an earlier write leaves
    rows in the old file from the run that was abandoned. `read_history` drops those and
    names each dropped block in `superseded`; it is the reader written after a tail mean
    was quoted off an abandoned run, and `compare.py` and `report.py` both go through it.

    This file used to do its own grouping, which is issue #146: on a real case it merged
    783 superseded rows and reported 9,369 rows where `read_history` gives 8,586, giving
    1.21093 at `--tail 1000` where the right answer is 1.20803, and 1.5% out on another.
    A 1.5% error in the value is an order larger than the band this script exists to
    produce, and the row count it printed as provenance was wrong too.

    `quantity` may name the series as well as the column, as `SERIES:COLUMN` --
    `torqueOuter/moment.dat:viscous_z`. It has to be sayable: every column of a `forces`
    function object appears in both `force.dat` and `moment.dat`, so a bare column name
    is ambiguous on every case that logs forces, and before #146 the refusal advised a
    "more specific --quantity" in a syntax that did not exist.

    `tail` averages the last n rows. The default is 1, the last row, which is the
    answer for a converged steady case. For a shedding case the last row is whichever
    phase of the cycle the final write landed on, and comparing three meshes' arbitrary
    phases as though they were three answers is a grid study measuring the clock --
    `results.py` prints the tail mean beside the last value for the same reason.
    """
    import results  # numpy and matplotlib come with it, so not at import time

    base = case / "postProcessing"
    if not base.is_dir():
        return {"value": None, "note": f"no {base} in this case"}

    wanted, column = split_quantity(quantity)

    series: dict[str, list[Path]] = {}
    for dat in sorted(base.rglob("*.dat")):
        if not dat.is_file():
            continue
        parts = dat.relative_to(base).parts
        name, _ = results.series_name(dat)
        function = "/".join(parts[:-2] if len(parts) >= 3 else parts[:-1])
        series.setdefault(f"{function}/{name}" if function else name, []).append(dat)

    found: list[dict[str, Any]] = []
    columns_seen: set[str] = set()
    matched_series: list[str] = []
    for key in sorted(series):
        history = results.read_history(sorted(series[key], key=results.written_order))
        columns, rows = history["columns"], history["rows"]
        columns_seen.update(columns)
        if wanted is not None and not _series_matches(key, wanted):
            continue
        matched_series.append(key)
        index = _column_index(columns, column)
        if index is None or not rows:
            continue
        taken = rows[-max(1, tail):]
        found.append({
            "value": sum(row[index] for row in taken) / len(taken),
            "series": key,
            "column": columns[index],
            "column_index": index,
            "files": history["sources"],
            "rows": len(rows),
            "rows_averaged": len(taken),
            "superseded": history["superseded"],
        })

    if not found:
        if wanted is not None and not matched_series:
            return {"value": None, "note": (
                f"no series under {base} matches {wanted!r}; --quantity takes "
                f"SERIES:COLUMN and the series here are "
                + (", ".join(sorted(series)) if series else "none"))}
        return {"value": None, "note": (
            f"no column called {column!r} under {base}; the columns there are "
            + (", ".join(sorted(columns_seen)) if columns_seen else "none that parsed"))}
    if len(found) > 1:
        candidates = [f"{item['series']}:{item['column']}" for item in found]
        return {"value": None, "note": (
            f"{column!r} is a column in {len(found)} series here; name the one you mean "
            f"as SERIES:COLUMN -- --quantity {candidates[0]} -- since picking for you is "
            "picking which number the whole study is about. The candidates are "
            + ", ".join(candidates))}
    return found[0]


def split_quantity(quantity: str) -> tuple[str | None, str]:
    """`torqueOuter/moment.dat:viscous_z` as its series and its column.

    Issue #146: `--quantity` could only name a column, every column of a `forces`
    function object is in both `force.dat` and `moment.dat`, and so the case-reading
    path of this script was unreachable on any case that logged forces -- it refused,
    correctly, and then asked for a syntax that did not exist. This is that syntax. No
    colon means the whole string is the column, which is what every existing call is.
    """
    series, separator, column = quantity.rpartition(":")
    if not separator:
        return None, quantity
    return series, column


def _series_matches(key: str, wanted: str) -> bool:
    """Whether a series key is the one asked for: a case-insensitive substring, so
    `forces`, `forces/moment.dat` and `moment.dat` all name the same series and a
    partial name that still fits two series is answered by listing both rather than by
    picking. Backslashes are normalised because a reader copying a series out of the
    report on Windows copies them."""
    return wanted.lower().replace("\\", "/") in key.lower()


def _column_index(columns: list[str], quantity: str) -> int | None:
    """Exact name, then case-insensitive, and nothing looser. A substring match would
    take `Cd` out of `Cd(f)` or `Cdf`, which are different numbers."""
    if quantity in columns:
        return columns.index(quantity)
    lowered = [c.lower() for c in columns]
    if quantity.lower() in lowered:
        return lowered.index(quantity.lower())
    return None


# -- the report ----------------------------------------------------------------


def report(analysis: dict[str, Any], provenance: list[str]) -> str:
    lines = ["# grid convergence", ""]

    lines.append("## where the numbers came from")
    for line in provenance:
        lines.append(f"  {line}")

    if analysis.get("dropped"):
        lines.append("")
        lines.append(f"## {len(analysis['dropped'])} grid(s) given on the command line "
                     f"are not in this estimate")
        for line in analysis["dropped"]:
            lines.append(f"  {line}")
        lines.append("  This estimate is DEGRADED: it is a ladder with a rung missing, so "
                     "the grid it names, the value it gives and the width of its band are "
                     "not the ones the ladder as given would have produced.")

    lines.append("")
    lines.append("## the ladder")
    lines.append(f"  {'grid':<28}{'h (rel.)':>14}{'value':>18}")
    for grid in analysis["grids"]:
        lines.append(f"  {grid['label'][:28]:<28}{grid['h']:>14.6g}{grid['value']:>18.8g}")

    lines.append("")
    lines.append("## refinement")
    for pair in analysis["pairs"]:
        lines.append(f"  {pair['coarse']} -> {pair['fine']}: r = {pair['ratio']:.4f}, "
                     f"the value changes by {-pair['difference']:+.6g}")

    primary = analysis["primary"]
    lines.append("")
    lines.append(f"## {primary['mode']} estimate on {', '.join(primary['grids'])}")
    lines.append(f"  convergence           {primary['convergence']}")
    observed = primary.get("order_observed")
    lines.append("  observed order p      "
                 + (f"{observed:.6f}" if observed is not None else "not observed"))
    if primary.get("order_used") is not None:
        lines.append(f"  order used            {primary['order_used']:.6f} "
                     f"({primary['order_source']})")
    lines.append(f"  formal order          {analysis['formal_order']:g}")
    if primary.get("extrapolated") is not None:
        lines.append(f"  extrapolated value    {primary['extrapolated']:.8g} "
                     "(Richardson, h -> 0)")
    for key, label in (("gci_fine_percent", "GCI on the fine grid"),
                       ("gci_coarse_percent", "GCI on the coarse pair")):
        if primary.get(key) is not None:
            lines.append(f"  {label:<22} {primary[key]:.6g}%")
    if primary.get("asymptotic_ratio") is not None:
        lines.append(f"  asymptotic ratio      {primary['asymptotic_ratio']:.4f} "
                     "(Roache's check; 1 is in the asymptotic range)")

    default = " (the default for this many grids)" if analysis["safety_factor_is_default"] else ""
    lines.append(f"  safety factor Fs      {analysis['safety_factor']:g}{default}")
    lines.append(f"  spread across ladder  {analysis['spread']:+.6g}")

    lines.append("")
    lines.append("## the answer")
    if primary.get("band") is not None:
        label = analysis.get("quantity") or "value"
        lines.append(f"  {label} = {primary['fine_value']:.8g} +/- {primary['band']:#.4g} "
                     f"on {primary['grids'][0]}")
        if primary["band_source"] != "GCI":
            lines.append(f"  the band is {primary['band_source']}")
        elif not primary.get("order_usable"):
            lines.append("  the band is an estimate: see the note below, the order it "
                         "rests on was not measured")
    elif primary.get("disqualified"):
        label = analysis.get("quantity") or "value"
        lines.append(f"  no band on {label}: the inputs above disqualify this estimate, "
                     "and the reasons are measurements you can check")
        for reason in primary["disqualified"]:
            for position, chunk in enumerate(_wrap(reason, 82)):
                lines.append(f"    {'- ' if position == 0 else '  '}{chunk}")
        lines.append(f"  the values themselves stand: "
                     + ", ".join(f"{g['label']} = {g['value']:.8g}"
                                 for g in analysis["grids"]))
    else:
        lines.append("  no value with an uncertainty: see the note below")

    if primary.get("note"):
        lines.append("")
        lines.append("## note")
        for chunk in _wrap(primary["note"]):
            lines.append(f"  {chunk}")

    others = analysis["triplets"][1:]
    if others:
        lines.append("")
        lines.append("## the coarser triplets, for whether the ladder is asymptotic")
        for triplet in others:
            observed = triplet.get("order_observed")
            lines.append(f"  {', '.join(triplet['grids'])}: {triplet['convergence']}, p = "
                         + (f"{observed:.4f}" if observed is not None else "not observed"))

    return "\n".join(lines) + "\n"


def _wrap(text: str, width: int = 86) -> list[str]:
    out, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def plot(analysis: dict[str, Any], path: Path, quantity: str) -> None:
    """Value against h, with the band drawn as an error bar and the extrapolation as
    the asymptote it is. The point of the picture is that the asymptote is a horizontal
    line and the points should be approaching it: a ladder whose points walk away from
    the line is visible here in a way it is not in a table of three numbers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = analysis["primary"]
    hs = [g["h"] for g in analysis["grids"]]
    values = [g["value"] for g in analysis["grids"]]
    errors = [primary.get("band") if g["label"] == primary["grids"][0] else 0.0
              for g in analysis["grids"]]
    errors = [e if e is not None else 0.0 for e in errors]

    figure, axes = plt.subplots(figsize=(7.0, 4.6))
    axes.errorbar(hs, values, yerr=errors, fmt="o-", capsize=5, color="#1f4e79",
                  ecolor="#c0392b", elinewidth=2, label="computed")
    if primary.get("extrapolated") is not None:
        axes.axhline(primary["extrapolated"], color="#2e7d32", linestyle="--",
                     label=f"Richardson h -> 0: {primary['extrapolated']:.6g}")
    for grid in analysis["grids"]:
        axes.annotate(grid["label"][:18], (grid["h"], grid["value"]),
                      textcoords="offset points", xytext=(6, 6), fontsize=8)
    axes.set_xlabel("representative grid spacing h (relative)")
    axes.set_ylabel(quantity)
    order = primary.get("order_observed")
    title = f"grid convergence: {quantity}"
    if order is not None:
        title += f"  (p = {order:.3f}, Fs = {analysis['safety_factor']:g})"
    if primary.get("band") is not None:
        title += f"\n{primary['fine_value']:.6g} +/- {primary['band']:.3g}"
    axes.set_title(title, fontsize=10)
    axes.set_xlim(left=0.0)
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


# -- the command line ----------------------------------------------------------


def _explicit(entries: list[str], flag: str) -> list[tuple[float, float]]:
    out = []
    for entry in entries:
        key, sep, value = entry.partition(":")
        if not sep:
            raise SystemExit(f"{flag} wants SIZE:VALUE, not {entry!r}")
        try:
            out.append((float(key), float(value)))
        except ValueError:
            raise SystemExit(f"{flag} wants two numbers, not {entry!r}") from None
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[-7:]),
    )
    parser.add_argument("cases", nargs="*", type=Path,
                        help="two or more case directories, in any order")
    parser.add_argument("--quantity", metavar="[SERIES:]COLUMN",
                        help="the column in postProcessing/*.dat, optionally qualified "
                             "by its series: Cd, or torqueOuter/moment.dat:viscous_z "
                             "where one column name appears in more than one series")
    parser.add_argument("--tail", type=int, default=1,
                        help="average the last n rows of the series (default 1, the last row)")
    parser.add_argument("--grid", action="append", default=[], metavar="CELLS:VALUE",
                        help="an explicit (cell count, value) pair; repeat it")
    parser.add_argument("--spacing", action="append", default=[], metavar="H:VALUE",
                        help="an explicit (representative spacing, value) pair; repeat it")
    parser.add_argument("--dim", choices=["2", "3", "auto"],
                        help="2 or 3 for h ~ N**(-1/2) or N**(-1/3); auto reads it "
                             "off each case's empty patches")
    parser.add_argument("--formal-order", type=float, default=FORMAL_ORDER,
                        help=f"the schemes' formal order (default {FORMAL_ORDER:g})")
    parser.add_argument("--safety-factor", type=float,
                        help=f"Roache's Fs (default {SAFETY_THREE_GRIDS:g} for three "
                             f"grids, {SAFETY_TWO_GRIDS:g} for two)")
    parser.add_argument("--json", type=Path, help="write the same facts as JSON")
    parser.add_argument("--png", type=Path, help="value against h with the error bar")
    opts = parser.parse_args(argv)

    grids: list[dict[str, Any]] = []
    provenance: list[str] = []
    dropped: list[str] = []
    quantity = split_quantity(opts.quantity)[1] if opts.quantity else "value"

    for h, value in _explicit(opts.spacing, "--spacing"):
        grids.append({"h": h, "value": value, "label": f"h={h:g}"})
        provenance.append(f"h={h:g}: value {value:g}, given on the command line")

    cells = _explicit(opts.grid, "--grid")
    if cells and opts.dim in (None, "auto"):
        parser.error("--grid gives cell counts, which need --dim 2 or --dim 3 to become "
                     "an h; a guess here moves the observed order by a factor of 1.5")
    for count, value in cells:
        dimension = int(opts.dim)
        grids.append({"h": representative_spacing(count, dimension), "value": value,
                      "label": f"{int(count)} cells"})
        provenance.append(f"{int(count)} cells: value {value:g}, given on the command "
                          f"line; h = N**(-1/{dimension}) from --dim {dimension}")

    for case in opts.cases:
        if not opts.quantity:
            parser.error("a case needs --quantity: which column of which "
                         "postProcessing file is the number this study is about")
        count, where = cells_from_case(case)
        if count is None:
            # Not only stderr. The report is the artifact that gets saved, and a
            # three-grid ladder that quietly became a two-grid estimate named the
            # wrong grid, gave the wrong value and widened the band 2.1x with no
            # notice in it anywhere (#171).
            dropped.append(f"{case.name}: no cell count -- {where}")
            print(f"{case}: {where}", file=sys.stderr)
            continue
        if opts.dim in (None, "auto"):
            dimension, route = dimension_from_case(case)
            if dimension is None:
                parser.error(f"{case}: {route}; give --dim 2 or --dim 3")
        else:
            dimension, route = int(opts.dim), f"{opts.dim}D, declared with --dim"
        measured = quantity_from_case(case, opts.quantity, opts.tail)
        if measured["value"] is None:
            dropped.append(f"{case.name}: no value -- {measured['note']}")
            print(f"{case}: {measured['note']}", file=sys.stderr)
            continue
        grids.append({"h": representative_spacing(count, dimension),
                      "value": measured["value"], "label": case.name,
                      # Carried so `_disqualify` can see a four-row history. The band is
                      # a claim about the quantity, and how many rows the quantity is a
                      # mean of is part of whether there is a quantity at all (#142).
                      "rows": measured["rows"],
                      "rows_averaged": measured["rows_averaged"]})
        provenance.append(
            f"{case.name}: {count:,} cells from {where}; {opts.quantity} = "
            f"{measured['value']:.8g} from column {measured['column']!r} "
            f"(index {measured['column_index']}) of series {measured['series']}, "
            f"{measured['files'][0]}"
            + (f" and {len(measured['files']) - 1} more file(s) of the same series"
               if len(measured["files"]) > 1 else "")
            + f", mean of the last {measured['rows_averaged']} of {measured['rows']} rows"
            + (f" ({len(measured['superseded'])} block(s) of rows dropped as superseded "
               "by a later restart)" if measured["superseded"] else "")
            + f"; h = N**(-1/{dimension}), {route}")

    if len(grids) < 2:
        # The reasons, not only the count. On a real ladder whose three rungs all had
        # their cell count in `constant/polyMesh/owner` and none had a checkMesh log,
        # this said "0 was readable" and the three reasons went to stderr only (#171).
        parser.error(f"a grid-convergence estimate needs two or more grids and {len(grids)} "
                     "was readable; there is nothing to report on one grid"
                     + (f" -- {len(dropped)} dropped: " + "; ".join(dropped)
                        if dropped else ""))

    analysis = analyse(grids, opts.formal_order, opts.safety_factor)
    analysis["quantity"] = quantity
    analysis["dropped"] = dropped
    text = report(analysis, provenance)
    print(text, end="")

    if opts.json:
        opts.json.write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
        print(f"json: {opts.json.resolve()}")
    if opts.png:
        plot(analysis, opts.png, quantity)
        print(f"picture: {opts.png.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
