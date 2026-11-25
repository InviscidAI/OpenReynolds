"""Turbulent pipe friction and the law of the wall: the references, and what they are not.

    python3 pipe_friction.py              # the friction factors and the layer edges
    python3 pipe_friction.py --check      # grade them, exit non-zero on a disagreement
    python3 pipe_friction.py --re 1e6 --roughness 1e-4

Why this exists. The tutorial index names "a turbulent case with a reference and a
measured agreement" as **the largest single gap in this index**, and the derived gap
register agrees. This is the reference half of it, written the way `blasius.py` and
`nozzle.py` were: the closed forms first, the solve second. **No turbulent OpenFOAM run
is graded here**, and the gap stays open on the solve.

What is exact and what is not, because here they are different
--------------------------------------------------------------
This is the first case in this repository whose references are **not exact**, and saying
so is most of the point.

* `f = 64/Re` for laminar pipe flow is **exact** -- it falls out of the Hagen-Poiseuille
  profile, the same algebra `duct2d.py` is graded on.
* `f = 0.316 Re^-0.25` (Blasius) and the **Colebrook-White** equation are **fits to
  Nikuradse's 1930s pipe measurements**: experimental data with a curve through them.
  `ladder.py` says exactly that and sets a tolerance of "within 5-10% of Blasius or
  Colebrook", which is "the scatter in the underlying measurements plus the wall
  function". A solver agreeing with Colebrook to three figures has not been validated to
  three figures; it has matched a correlation inside its own scatter.

So what is graded here is the **arithmetic and the agreement between the correlations**,
which is checkable to machine precision, and never the correlations against nature:

* Colebrook is **implicit** in `f` and is solved by iteration; Haaland's explicit form
  claims to sit within a couple of per cent of it, and that claim is measured here;
* the Blasius fit's **range** is measured rather than repeated: `ladder.py` says it holds
  "to about Re 1e5", and the Re where it leaves a 5% band of Colebrook is printed;
* the laminar and turbulent branches **cross**, and that crossing is printed because a
  case near it can be matched by the wrong branch.

The layer edges are conventions, and one of them is derivable
-------------------------------------------------------------
`wall.py` bands `y+` at 5 and 30 -- viscous sublayer, buffer layer, log layer -- and
chooses wall functions on those edges. The sublayer `u+ = y+` and the log law
`u+ = ln(y+)/kappa + B` **cross** at one `y+`, which is where neither is the obvious
answer, and it is computed here: with `kappa = 0.41` and `B = 5.2` it lands near 11, in
the middle of the buffer band, which is what makes that band the thing it is. The 5 and
30 themselves are conventions and are checked only for bracketing the crossing.

`kappa` and `B` are **measured constants with published spread** -- kappa is quoted from
0.38 to 0.41 and B from 5.0 to 5.5 -- so the crossing moves by about a y+ unit across
that range, and the script prints it at both ends rather than quoting one.

Standard library only.
"""
from __future__ import annotations

import argparse
import math

KAPPA = 0.41
B_CONSTANT = 5.2
"""Von Karman's constant and the log-law intercept, as `wall.py`'s own wall functions
assume them. Measured, not derived: kappa is in print from 0.38 to 0.41 and B from 5.0 to
5.5, which `spread()` below walks rather than hiding."""

KAPPA_SPREAD = (0.38, 0.41)
B_SPREAD = (5.0, 5.5)

SUBLAYER_EDGE, LOG_EDGE = 5.0, 30.0
"""`wall.py`'s band edges. Conventions, checked here only for bracketing the crossing."""

BLASIUS_RANGE_RE = 1.0e5
"""Where `ladder.py` says the Blasius fit stops holding: "to about Re 1e5"."""

LADDER_BAND = 0.05
"""The tight end of `ladder.py`'s "within 5-10%" tolerance."""


def laminar(reynolds: float) -> float:
    """`f = 64/Re`, **exact** for Hagen-Poiseuille flow in a round pipe."""
    if reynolds <= 0.0:
        raise ValueError(f"Re must be positive, got {reynolds}")
    return 64.0 / reynolds


def blasius_fit(reynolds: float) -> float:
    """`f = 0.316 Re^-0.25`: a fit to Nikuradse, not a derivation."""
    if reynolds <= 0.0:
        raise ValueError(f"Re must be positive, got {reynolds}")
    return 0.316 * reynolds ** -0.25


def colebrook(reynolds: float, roughness: float = 0.0) -> float:
    """Colebrook-White, solved by iteration because it is implicit in `f`.

    `1/sqrt(f) = -2 log10(eps/(3.7 D) + 2.51/(Re sqrt(f)))`. Fixed-point on `1/sqrt(f)`
    converges in a handful of steps for every Re this is used at; the loop asserts it
    rather than trusting it, because a correlation that quietly returned its starting
    guess would look like a result.
    """
    if reynolds <= 0.0:
        raise ValueError(f"Re must be positive, got {reynolds}")
    if roughness < 0.0:
        raise ValueError(f"relative roughness cannot be negative, got {roughness}")
    inverse = 1.0 / math.sqrt(blasius_fit(max(reynolds, 4000.0)))
    for _ in range(200):
        nxt = -2.0 * math.log10(roughness / 3.7 + 2.51 * inverse / reynolds)
        if abs(nxt - inverse) < 1e-14:
            inverse = nxt
            break
        inverse = nxt
    else:
        raise ValueError(f"Colebrook did not converge at Re {reynolds}")
    return 1.0 / (inverse * inverse)


def haaland(reynolds: float, roughness: float = 0.0) -> float:
    """Haaland's explicit stand-in for Colebrook, which is the claim being measured."""
    if reynolds <= 0.0:
        raise ValueError(f"Re must be positive, got {reynolds}")
    inner = (roughness / 3.7) ** 1.11 + 6.9 / reynolds
    return (-1.8 * math.log10(inner)) ** -2.0


def crossing(kappa: float = KAPPA, intercept: float = B_CONSTANT) -> float:
    """The `y+` where `u+ = y+` meets `u+ = ln(y+)/kappa + B`, by bisection.

    Not a convention: given the two constants this is a number, and it is why the buffer
    layer is where it is.
    """
    def gap(y: float) -> float:
        return y - (math.log(y) / kappa + intercept)

    low, high = 1.0, 100.0
    assert gap(low) < 0.0 < gap(high), "the crossing is not bracketed"
    for _ in range(200):
        middle = 0.5 * (low + high)
        if gap(middle) < 0.0:
            low = middle
        else:
            high = middle
        if high - low < 1e-13:
            break
    return 0.5 * (low + high)


def crossing_range() -> tuple[float, float]:
    """The lowest and highest crossing over the constants' published spread.

    Over the four corners, not over two of them. The first version of this printed
    `(kappa_low, B_low)` and `(kappa_high, B_high)` and called the result a range, which
    it is not: a smaller `kappa` RAISES the crossing and a smaller `B` lowers it, so the
    extremes are the mixed corners `(kappa_high, B_low)` and `(kappa_low, B_high)`. Both
    of the corners it printed happened to sit above the nominal, which is how a
    "range" that excluded its own central value got printed in the first place.
    """
    values = [crossing(k, b) for k in KAPPA_SPREAD for b in B_SPREAD]
    return min(values), max(values)


def blasius_departs(band: float = LADDER_BAND) -> float:
    """The Re where the Blasius fit first leaves `band` of smooth-pipe Colebrook."""
    low, high = 4000.0, 1.0e9
    for _ in range(300):
        middle = math.sqrt(low * high)
        error = abs(blasius_fit(middle) - colebrook(middle)) / colebrook(middle)
        if error < band:
            low = middle
        else:
            high = middle
        if high / low < 1.0 + 1e-12:
            break
    return math.sqrt(low * high)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Turbulent pipe friction and the law of the wall, as references.")
    parser.add_argument("--re", type=float, default=1.0e5)
    parser.add_argument("--roughness", type=float, default=0.0,
                        help="relative roughness eps/D (default 0, a smooth pipe)")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    print("# pipe friction: one exact branch and two fits")
    print(f"  at Re {args.re:g}, relative roughness {args.roughness:g}:")
    print(f"    laminar 64/Re        {laminar(args.re):.8f}   EXACT, from the "
          "Hagen-Poiseuille profile")
    print(f"    Blasius fit          {blasius_fit(args.re):.8f}   a fit to Nikuradse")
    print(f"    Colebrook-White      {colebrook(args.re, args.roughness):.8f}   "
          "implicit, solved by iteration")
    print(f"    Haaland explicit     {haaland(args.re, args.roughness):.8f}   "
          f"{(haaland(args.re, args.roughness) / colebrook(args.re, args.roughness) - 1):+.3%}"
          " on Colebrook")
    print()
    departs = blasius_departs()
    print(f"  the Blasius fit leaves a {LADDER_BAND:.0%} band of smooth Colebrook at "
          f"Re {departs:.4g}")
    print(f"    -- ladder.py says it holds \"to about Re {BLASIUS_RANGE_RE:g}\"")
    print(f"  the laminar and Blasius branches cross at Re "
          f"{(64.0 / 0.316) ** (1.0 / 0.75):.4g}, where a case can be")
    print("    matched by the wrong branch")
    print()
    print("  the law of the wall, with wall.py's own constants:")
    print(f"    u+ = y+ meets ln(y+)/kappa + B at y+ {crossing():.6f}   "
          f"(kappa {KAPPA:g}, B {B_CONSTANT:g})")
    low, high = crossing_range()
    print(f"    across the published spread kappa {KAPPA_SPREAD} and B {B_SPREAD}: "
          f"y+ {low:.4f} to {high:.4f}   (the four corners, since a smaller kappa "
          "raises it and a smaller B lowers it)")
    print(f"    wall.py bands the buffer layer {SUBLAYER_EDGE:g} < y+ < {LOG_EDGE:g}, "
          "which is a convention that")
    print("    has to contain that crossing to mean anything")

    if not args.check:
        return 0

    checks: list[tuple[str, bool, str]] = []

    # The exact branch, against the profile it comes from.
    checks.append(("f = 64/Re is exact at Re 2000", abs(laminar(2000.0) - 0.032) < 1e-15,
                   "{:.10f}".format(laminar(2000.0))))

    # Colebrook really solved, not returned as its guess.
    f = colebrook(1.0e5)
    residual = abs(1.0 / math.sqrt(f) + 2.0 * math.log10(2.51 / (1.0e5 * math.sqrt(f))))
    checks.append(("Colebrook's own equation is satisfied by what it returns",
                   residual < 1e-10, "residual {:.2e}".format(residual)))

    # Haaland's claim about itself, measured across four decades.
    worst = max(abs(haaland(10.0 ** n) / colebrook(10.0 ** n) - 1.0)
                for n in (4, 5, 6, 7, 8))
    checks.append(("Haaland sits within 2% of Colebrook, which is its claim",
                   worst < 0.02, "worst {:+.3%}".format(worst)))

    # ladder.py's stated range for the Blasius fit.
    checks.append((f"the Blasius fit holds to about Re {BLASIUS_RANGE_RE:g}, as ladder.py says",
                   0.3 * BLASIUS_RANGE_RE < blasius_departs() < 30.0 * BLASIUS_RANGE_RE,
                   "leaves the band at Re {:.4g}".format(blasius_departs())))

    # Monotonicity: friction falls with Re on both turbulent fits.
    rising = [colebrook(10.0 ** n) for n in (4, 5, 6, 7, 8)]
    checks.append(("smooth-pipe friction falls with Re, on every decade",
                   all(a > b for a, b in zip(rising, rising[1:])),
                   "{:.5f} -> {:.5f}".format(rising[0], rising[-1])))

    # Roughness can only add friction, and saturates.
    smooth, rough = colebrook(1.0e7, 0.0), colebrook(1.0e7, 1.0e-3)
    checks.append(("roughness adds friction and never removes it",
                   rough > smooth, "{:.5f} against {:.5f}".format(rough, smooth)))

    # The crossing, and that wall.py's convention contains it.
    where = crossing()
    checks.append(("the sublayer and the log law cross inside wall.py's buffer band",
                   SUBLAYER_EDGE < where < LOG_EDGE, "y+ {:.6f}".format(where)))
    low, high = crossing_range()
    checks.append(("and it stays inside that band across the constants' published spread",
                   SUBLAYER_EDGE < low and high < LOG_EDGE,
                   "y+ {:.4f} to {:.4f}".format(low, high)))
    checks.append(("the spread brackets the nominal crossing, which a two-corner range "
                   "did not", low <= where <= high,
                   "{:.4f} <= {:.4f} <= {:.4f}".format(low, where, high)))

    # Refusals.
    refused = 0
    for bad in (lambda: laminar(0.0), lambda: colebrook(-1.0),
                lambda: colebrook(1.0e5, -0.1), lambda: haaland(0.0)):
        try:
            bad()
        except ValueError:
            refused += 1
    checks.append(("a non-positive Re and a negative roughness are refused",
                   refused == 4, f"{refused} of 4"))

    print()
    bad_count = 0
    for what, ok, measured in checks:
        print(f"  {'ok  ' if ok else 'FAIL'}  {what}  [{measured}]")
        bad_count += 0 if ok else 1
    print("agrees" if not bad_count else f"{bad_count} disagreement(s)")
    return 1 if bad_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
