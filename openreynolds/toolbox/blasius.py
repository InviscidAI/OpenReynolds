"""The laminar flat plate, solved here rather than quoted: Blasius by shooting.

    python3 blasius.py                 # the solution and every constant it implies
    python3 blasius.py --check         # grade it, exit non-zero on a disagreement
    python3 blasius.py --eta-max 12 --steps 4000

Why this exists. `reference_area.py` carries `Cf = 1.328/sqrt(Re)` as a **typed
constant**, used as the friction floor no laminar total drag can sit below, and nothing
in this repository derived it. `panel.py` prints the same 1.328 and labels it "here under
protest". A number nobody computes is a number nobody can check, and the tutorial index
names the absent flat plate as "the most annoying absence here".

This closes the arithmetic half of that gap and **not** the solver half. There is still
no OpenFOAM flat-plate run in this repository to report a measured `cf` from, and the
index still says so. What is here is the reference such a run would be graded against,
computed from the equation instead of copied from a textbook.

What is graded
--------------
`f''(0)`, against the published **0.3320573362151963**. Everything else on the plate
follows from it by algebra that is written out in `IMPLIED` below, so the single
published number is the only thing taken on trust, and the constants this repository
already uses are then checked against it rather than against each other.

Two conventions, and both are a trap this page exists to catch
--------------------------------------------------------------
**The equation.** `f''' + f f''/2 = 0` with `eta = y sqrt(U/(nu x))` gives
`f''(0) = 0.332057`. The same physics written `f''' + f f'' = 0` with
`eta = y sqrt(U/(2 nu x))` gives **0.469600**, and the two differ by exactly `sqrt(2)`.
Both are in print. A reader who takes 0.4696 into the first convention is wrong by 41.4%,
which is the same class of error as reading Gartling's 6.1 in step heights
(`benchmarks/README.md`).

**The thickness.** `delta_99/x = 5.0/sqrt(Re_x)` is in every textbook, and the eta where
`f' = 0.99` is **not** 5.0 -- it is about 4.91. The 5.0 is a rounding, and it is quoted
here as a rounding with the difference measured, because a 1.8% convention gap inside a
"reference value" is exactly what this repository keeps finding.

Standard library only: a scalar ODE and a bisection need no numpy, and this way the case
runs on an interpreter with nothing installed at all.
"""
from __future__ import annotations

import argparse
import math

PUBLISHED_FPP0 = 0.3320573362151963
"""`f''(0)` for `f''' + f f''/2 = 0`, `f(0) = f'(0) = 0`, `f'(inf) = 1`.

Howarth (1938) tabulated 0.332057 and the value is standard to many more figures than
this repository needs. It is the ONE number here that is taken rather than derived, and
everything in `IMPLIED` is computed from the solution instead of being quoted beside it.
"""

OTHER_CONVENTION = 0.4696
"""The same solution under `f''' + f f'' = 0`, `eta = y sqrt(U/(2 nu x))`.

Kept so the factor can be printed rather than warned about: it is `sqrt(2)` times the
value above, so mixing the two is a 41.4% error and not a rounding.
"""

OTHER_KEY = "the other convention's f''(0)"
"""The `IMPLIED` key, as a name: an apostrophe inside an f-string
subscript is legal on 3.12 and a SyntaxError on 3.10, which this
package supports."""

TEXTBOOK_DELTA99 = 5.0
"""`delta_99 sqrt(Re_x)/x` as textbooks give it. The computed value is near 4.91."""

TYPED_IN_REFERENCE_AREA = 1.328
"""`reference_area.blasius`'s whole-plate `Cf sqrt(Re)`, as a literal in that file.

This case's reason for existing: it should be `4 f''(0)`, and now something checks that.
"""


def rhs(state: tuple[float, float, float]) -> tuple[float, float, float]:
    """`f''' = -f f''/2`, as the three first-order equations."""
    f, fp, fpp = state
    return (fp, fpp, -0.5 * f * fpp)


def step(state: tuple[float, float, float], h: float) -> tuple[float, float, float]:
    """One classical RK4 step. Fourth order, which `--check` measures rather than says."""
    k1 = rhs(state)
    s2 = tuple(y + 0.5 * h * k for y, k in zip(state, k1))
    k2 = rhs(s2)
    s3 = tuple(y + 0.5 * h * k for y, k in zip(state, k2))
    k3 = rhs(s3)
    s4 = tuple(y + h * k for y, k in zip(state, k3))
    k4 = rhs(s4)
    return tuple(y + (h / 6.0) * (a + 2 * b + 2 * c + d)
                 for y, a, b, c, d in zip(state, k1, k2, k3, k4))


def integrate(fpp0: float, eta_max: float, steps: int) -> list[tuple[float, tuple[float, float, float]]]:
    """March from the wall to `eta_max`, keeping the whole profile."""
    h = eta_max / steps
    state = (0.0, 0.0, fpp0)
    out = [(0.0, state)]
    for index in range(steps):
        state = step(state, h)
        out.append(((index + 1) * h, state))
    return out


def shoot(eta_max: float = 10.0, steps: int = 2000,
          low: float = 0.1, high: float = 1.0, rounds: int = 200) -> float:
    """`f''(0)` such that `f'(eta_max) = 1`, by bisection.

    Bisection rather than Newton because `f'(eta_max)` is monotone in `f''(0)` and a
    bracket that cannot escape is worth more here than the iteration count: the whole
    solve is milliseconds either way.
    """
    def miss(guess: float) -> float:
        return integrate(guess, eta_max, steps)[-1][1][1] - 1.0

    assert miss(low) < 0.0 < miss(high), "the bracket does not contain the solution"
    for _ in range(rounds):
        middle = 0.5 * (low + high)
        if miss(middle) < 0.0:
            low = middle
        else:
            high = middle
        if high - low < 1e-16:
            break
    return 0.5 * (low + high)


def eta_at(profile, fraction: float) -> float:
    """Where `f'` first reaches `fraction`, linearly interpolated between two steps."""
    previous = profile[0]
    for point in profile[1:]:
        if point[1][1] >= fraction:
            eta0, s0 = previous
            eta1, s1 = point
            span = s1[1] - s0[1]
            if span == 0.0:
                return eta1
            return eta0 + (fraction - s0[1]) * (eta1 - eta0) / span
        previous = point
    raise ValueError(f"f' never reached {fraction}: extend --eta-max")


def displacement(profile) -> float:
    """`delta* sqrt(U/(nu x))`, as `lim (eta - f(eta))`.

    Derived here and **not** compared with a published figure, because this repository
    has not read a source for it. It is printed so it is available, and it carries no
    verdict, the way `A/D` does in `benchmarks/moving_mesh/`.
    """
    eta, state = profile[-1]
    return eta - state[0]


def IMPLIED(fpp0: float) -> dict[str, tuple[float, str]]:
    """Every plate constant that follows from `f''(0)` by algebra, with the algebra.

    One place, so the relationships cannot drift from the numbers: `cf_x` is the wall
    stress over the dynamic pressure and the whole-plate average is its integral, which
    doubles it.
    """
    return {
        "cf_x sqrt(Re_x)": (2.0 * fpp0, "2 f''(0): tau_w = mu U f''(0) sqrt(U/(nu x)), "
                                        "over 0.5 rho U^2"),
        "Cf sqrt(Re_L), whole plate": (4.0 * fpp0, "the integral of cf_x over 0..L, "
                                                   "which is twice the local value at L"),
        OTHER_KEY: (fpp0 * math.sqrt(2.0),
                                          "sqrt(2) f''(0): eta scaled on 2 nu x"),
    }


def report(eta_max: float, steps: int) -> dict[str, object]:
    """Solve, and lay every number beside what it is measured against."""
    fpp0 = shoot(eta_max=eta_max, steps=steps)
    profile = integrate(fpp0, eta_max, steps)
    implied = IMPLIED(fpp0)
    return {
        "fpp0": fpp0,
        "fpp0_error": (fpp0 - PUBLISHED_FPP0) / PUBLISHED_FPP0,
        "implied": implied,
        "eta_99": eta_at(profile, 0.99),
        "delta_star": displacement(profile),
        "eta_max": eta_max,
        "steps": steps,
        "typed_error": (implied["Cf sqrt(Re_L), whole plate"][0] - TYPED_IN_REFERENCE_AREA)
        / TYPED_IN_REFERENCE_AREA,
    }


def order(eta_max: float = 10.0) -> tuple[float, list[tuple[int, float]]]:
    """The observed order of the integrator, against a refined solution of its own.

    **Not against the published value**, which is the mistake the first version of this
    made: by 250 steps the error against `PUBLISHED_FPP0` is already about 3e-09 and is
    set by the far-field condition `f'(eta_max) = 1` at a finite `eta_max`, not by the
    step size. Differencing against the published number therefore measured the
    truncation of the domain and reported an "order" of 0.15, which says nothing about
    RK4 at all.

    So the reference here is this same solve at 3200 steps, and the comparison runs from
    100 to 800, where the step size is what dominates and the rate is asymptotic. `benchmarks/particles/`'s
    tracer orbit grades its integrator the same way and for the same reason: a method
    whose error has a known rate can be checked on the rate, which no single agreement
    can do.
    """
    fine = shoot(eta_max=eta_max, steps=3200)
    out = []
    for steps in (100, 200, 400, 800):
        value = shoot(eta_max=eta_max, steps=steps)
        out.append((steps, abs(value - fine)))
    # Each pair two halvings apart: the error should fall by 4**4 = 2**8, so the
    # exponent is log(ratio)/log(4). The rate APPROACHES 4 from below as h shrinks --
    # measured 3.729, 3.882, 3.944, 3.976 at 25, 50, 100 and 200 steps against a
    # refinement -- so the window is chosen where it is asymptotic rather than where
    # the numbers are largest, and the graded one is the finest.
    rates = [math.log(out[index][1] / out[index + 2][1]) / math.log(4.0)
             for index in range(len(out) - 2)]
    return rates[-1], out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="The Blasius laminar flat plate, solved by shooting.")
    parser.add_argument("--eta-max", type=float, default=10.0,
                        help="how far from the wall to march (default 10)")
    parser.add_argument("--steps", type=int, default=2000,
                        help="RK4 steps across that span (default 2000)")
    parser.add_argument("--check", action="store_true",
                        help="grade against the published f''(0) and exit non-zero on a "
                             "disagreement")
    args = parser.parse_args()

    found = report(args.eta_max, args.steps)
    fpp0 = found["fpp0"]
    print("# Blasius: f''' + f f''/2 = 0, f(0) = f'(0) = 0, f'(inf) = 1")
    print(f"  marched to eta {args.eta_max:g} in {args.steps} RK4 steps")
    print(f"  f''(0)      {fpp0:.10f} against the published {PUBLISHED_FPP0:.10f} "
          f"({found['fpp0_error']:+.3e})")
    for name, (value, how) in found["implied"].items():
        print(f"  {name:34} {value:.10f}   {how}")
    print(f"  eta at f' = 0.99                   {found['eta_99']:.6f}   "
          f"textbook delta_99 is quoted as {TEXTBOOK_DELTA99:g}, which is "
          f"{(TEXTBOOK_DELTA99 - found['eta_99']) / found['eta_99']:+.2%} from it")
    print(f"  delta* sqrt(U/(nu x))              {found['delta_star']:.6f}   "
          "derived here, no published value read, so no verdict")
    print(f"  reference_area.blasius types {TYPED_IN_REFERENCE_AREA:g}: "
          f"4 f''(0) is {4 * fpp0:.7f} ({found['typed_error']:+.4%})")

    if not args.check:
        return 0

    checks: list[tuple[str, bool, str]] = []
    checks.append((f"f''(0) within 1e-7 of {PUBLISHED_FPP0:.10f}",
                   abs(found["fpp0_error"]) < 1e-7,
                   "{:+.3e}".format(found["fpp0_error"])))
    other = found["implied"][OTHER_KEY][0]
    checks.append(("the other convention is exactly sqrt(2) away, not a rounding",
                   abs(other - OTHER_CONVENTION) / OTHER_CONVENTION < 1e-4,
                   "{:.6f}".format(other)))
    observed, errors = order(args.eta_max)
    checks.append(("RK4 shows order 4 against its own refinement, 200 -> 800 steps "
                   "(not against the published value, which at these counts is "
                   "limited by eta_max and not by the step)",
                   observed >= 3.9, "{:.3f}".format(observed)))
    print("  f''(0) against a 3200-step refinement of the same solve:")
    for steps, value in errors:
        print(f"    {steps:4d} steps   h {args.eta_max / steps:.4f}   {value:.3e}")
    checks.append((f"reference_area's typed {TYPED_IN_REFERENCE_AREA:g} is within 0.1% "
                   "of 4 f''(0)", abs(found["typed_error"]) < 1e-3,
                   f"{found['typed_error']:+.4%}"))
    checks.append(("eta at f' = 0.99 is near 4.91 and NOT the textbook 5.0",
                   4.88 < found["eta_99"] < 4.94, f"{found['eta_99']:.6f}"))

    print()
    bad = 0
    for what, ok, measured in checks:
        print(f"  {'ok  ' if ok else 'FAIL'}  {what}  [{measured}]")
        bad += 0 if ok else 1
    print("agrees" if not bad else f"{bad} disagreement(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
