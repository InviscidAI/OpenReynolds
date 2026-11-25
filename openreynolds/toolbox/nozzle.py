"""Quasi-1D compressible relations, solved here: the exact answers a transonic case owes.

    python3 nozzle.py                  # every relation and what it implies
    python3 nozzle.py --check          # grade them, exit non-zero on a disagreement
    python3 nozzle.py --mach 2.0       # the isentropic and shock state at one Mach

Why this exists. The tutorial index says it plainly: "No `compressible.py` case is graded
against anything yet; that is a gap and is named as one." The compressible family is the
largest in `advise.py` -- five solvers -- and the only one with no exact reference
anywhere near it.

It does not need OpenFOAM. The relations a nozzle, a shock tube or a transonic far field
turn on are closed forms for a calorically perfect gas, and three of them are identities
rather than correlations, so they can be checked to machine precision on a laptop:

* **the area-Mach relation** has its minimum at exactly `M = 1`, where `A/A*` is exactly 1;
* **the Prandtl relation** `M1* M2* = 1` holds across a normal shock, and it is derived
  independently of the jump formulas, so it grades them rather than restating them;
* **the second law** makes a shock with `M1 < 1` impossible, so the relations have to
  refuse an expansion shock rather than returning a tidy wrong number.

And two numbers this repository already asserts are checked against the closed form:
`compressible.stagnation` computes total conditions from the gamma the written case file
implies, and `advise.COMPRESSIBLE_SWITCH` claims "M < 0.3 is where density varies by under
5%". The first is cross-checked here against the isentropic relation written out; the
second is derived, and the Mach number where the departure really reaches 5% is printed
beside the 0.3.

**This is not a solved nozzle.** No OpenFOAM run of a converging-diverging nozzle exists
in this repository, and the index's gap paragraph still says so. Quasi-1D is a different
thing from a CFD solve: it has no boundary layer, no shock thickness and no
two-dimensionality. What is here is the reference such a run would be graded against.

Standard library only.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Sibling scripts, not a package: the toolbox runs standalone out of /work/.toolbox.
# This is `advise.py:91-94`'s pattern, and it is needed rather than decorative --
# `run_tier1.py` invokes with `-I`, which removes the script's own directory from
# sys.path, so a bare `import compressible` fails there and nowhere else.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import compressible  # noqa: E402  (after the path insert, deliberately)

GAMMA = 1.4
"""Cp/Cv for a diatomic calorically perfect gas. `compressible.GAMMA_AIR` is the same
number and says air holds it to within 0.3% from 200 K to 1000 K; this file imports that
one for the cross-check rather than trusting the two to stay equal."""

SWITCH_MACH = 0.3
"""The Mach number `advise.COMPRESSIBLE_SWITCH` draws the incompressible line at."""

SWITCH_CLAIM = 0.05
"""The density departure that switch claims is not exceeded below `SWITCH_MACH`."""

CRITICAL_PRESSURE_RATIO = 0.5282817877171742
"""`p*/p0 = (2/(gamma+1))**(gamma/(gamma-1))` at gamma 1.4, to machine precision.

Written out so a reader can see the number the choked condition turns on; `critical()`
computes it rather than returning this."""


def isentropic(mach: float, gamma: float = GAMMA) -> dict[str, float]:
    """Total over static, exactly, for a calorically perfect gas.

    `T0/T = 1 + (gamma-1)/2 M^2`, and the pressure and density ratios are that raised to
    `gamma/(gamma-1)` and `1/(gamma-1)`. No approximation anywhere in it.
    """
    ratio = 1.0 + 0.5 * (gamma - 1.0) * mach * mach
    return {
        "T0/T": ratio,
        "p0/p": ratio ** (gamma / (gamma - 1.0)),
        "rho0/rho": ratio ** (1.0 / (gamma - 1.0)),
    }


def critical(gamma: float = GAMMA) -> dict[str, float]:
    """The sonic (starred) ratios: what a choked throat sits at."""
    return {
        "p*/p0": (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0)),
        "T*/T0": 2.0 / (gamma + 1.0),
        "rho*/rho0": (2.0 / (gamma + 1.0)) ** (1.0 / (gamma - 1.0)),
    }


def area_ratio(mach: float, gamma: float = GAMMA) -> float:
    """`A/A*` from the area-Mach relation. Exactly 1 at `M = 1`, and its only minimum."""
    if mach <= 0.0:
        raise ValueError("the area-Mach relation has no meaning at M <= 0")
    inner = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * mach * mach)
    return (1.0 / mach) * inner ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))


def mach_from_area(ratio: float, supersonic: bool, gamma: float = GAMMA) -> float:
    """Invert the area-Mach relation on the branch asked for, by bisection.

    Two Mach numbers give the same `A/A*` -- one each side of the throat -- and which one
    a nozzle is at is a boundary condition, not arithmetic. Returning either silently
    would be the same class of error as a reference quoted in the wrong convention, so
    the branch is an argument with no default.
    """
    if ratio < 1.0:
        raise ValueError(f"A/A* = {ratio} is below 1, which no section reaches")
    low, high = (1.0, 50.0) if supersonic else (1e-6, 1.0)
    for _ in range(200):
        middle = 0.5 * (low + high)
        here = area_ratio(middle, gamma)
        # A/A* falls toward the throat on the subsonic branch and rises after it.
        if (here > ratio) == (not supersonic):
            low = middle
        else:
            high = middle
        if high - low < 1e-15:
            break
    return 0.5 * (low + high)


def shock(mach1: float, gamma: float = GAMMA) -> dict[str, float]:
    """The normal-shock jump, and it refuses an expansion shock by name.

    A shock with `M1 < 1` would raise the entropy of nothing and lower it in fact: the
    second law forbids it, every jump relation below still returns a finite number for
    it, and a solver that reported one would look like it had found something. So the
    refusal is here rather than in the caller.
    """
    if mach1 < 1.0:
        raise ValueError(
            f"M1 = {mach1} < 1: an expansion shock violates the second law. The jump "
            "relations return numbers for it and they are not physics")
    m1sq = mach1 * mach1
    mach2sq = (1.0 + 0.5 * (gamma - 1.0) * m1sq) / (gamma * m1sq - 0.5 * (gamma - 1.0))
    p_ratio = (2.0 * gamma * m1sq - (gamma - 1.0)) / (gamma + 1.0)
    rho_ratio = ((gamma + 1.0) * m1sq) / (2.0 + (gamma - 1.0) * m1sq)
    t_ratio = p_ratio / rho_ratio
    # Total pressure across the shock: the loss that makes it irreversible.
    p0_ratio = (isentropic(math.sqrt(mach2sq), gamma)["p0/p"] * p_ratio
                / isentropic(mach1, gamma)["p0/p"])
    return {
        "M2": math.sqrt(mach2sq),
        "p2/p1": p_ratio,
        "rho2/rho1": rho_ratio,
        "T2/T1": t_ratio,
        "p02/p01": p0_ratio,
    }


def characteristic_mach(mach: float, gamma: float = GAMMA) -> float:
    """`M* = u/a*`, the speed over the SONIC speed rather than the local one.

    This is the quantity the Prandtl relation is written in, and it is derived from the
    energy equation alone -- not from the jump relations -- which is what makes
    `M1* M2* = 1` an independent check on them.
    """
    m2 = mach * mach
    return math.sqrt(((gamma + 1.0) * m2) / (2.0 + (gamma - 1.0) * m2))


def density_departure(mach: float, gamma: float = GAMMA) -> float:
    """`rho0/rho - 1`: how wrong "incompressible" is at this Mach number."""
    return isentropic(mach, gamma)["rho0/rho"] - 1.0


def switch_mach(target: float = SWITCH_CLAIM, gamma: float = GAMMA) -> float:
    """The Mach number where the density departure actually reaches `target`."""
    low, high = 1e-6, 1.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if density_departure(middle, gamma) < target:
            low = middle
        else:
            high = middle
        if high - low < 1e-15:
            break
    return 0.5 * (low + high)


def cross_check_the_shipped_stagnation(mach: float) -> dict[str, float]:
    """`compressible.stagnation` against the isentropic relation written out here.

    Two implementations of one formula, one of them the shipped one. The point is not
    that the formula is hard; it is that `compressible.py` recovers gamma from the
    FORMATTED Cp it wrote into the case file, so its gamma is the case's gamma to the
    digit and not 1.4 exactly. This measures that difference instead of assuming it is
    zero.
    """
    thermo = {"gamma": compressible.GAMMA_AIR, "R": 287.058,
              "temperature": 300.0, "pressure": 101325.0}
    shipped = compressible.stagnation(thermo, mach)
    here = isentropic(mach, compressible.GAMMA_AIR)
    return {
        "T0 shipped": shipped["T0"],
        "T0 here": thermo["temperature"] * here["T0/T"],
        "p0 shipped": shipped["p0"],
        "p0 here": thermo["pressure"] * here["p0/p"],
        "gamma used": compressible.GAMMA_AIR,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quasi-1D compressible relations, exact, with no OpenFOAM.")
    parser.add_argument("--mach", type=float, default=2.0,
                        help="the Mach number to report a state at (default 2.0)")
    parser.add_argument("--check", action="store_true",
                        help="grade the relations and exit non-zero on a disagreement")
    args = parser.parse_args()

    print(f"# quasi-1D, calorically perfect, gamma = {GAMMA:g}")
    state = isentropic(args.mach, GAMMA)
    print(f"  at M {args.mach:g}:  T0/T {state['T0/T']:.8f}   "
          f"p0/p {state['p0/p']:.8f}   rho0/rho {state['rho0/rho']:.8f}")
    star = critical(GAMMA)
    print(f"  choked throat:  p*/p0 {star['p*/p0']:.10f}   "
          f"T*/T0 {star['T*/T0']:.10f}   rho*/rho0 {star['rho*/rho0']:.10f}")
    print(f"  A/A* at M 1     {area_ratio(1.0, GAMMA):.12f}   (exactly 1, and its minimum)")
    print(f"  A/A* at M {args.mach:g}   {area_ratio(args.mach, GAMMA):.8f}")

    reached = switch_mach(SWITCH_CLAIM, GAMMA)
    print(f"  advise's switch: at M {SWITCH_MACH:g} the density departure is "
          f"{density_departure(SWITCH_MACH, GAMMA):+.4%}, and it reaches "
          f"{SWITCH_CLAIM:.0%} at M {reached:.6f}")

    if args.mach >= 1.0:
        jump = shock(args.mach, GAMMA)
        print(f"  normal shock at M {args.mach:g}:  M2 {jump['M2']:.8f}   "
              f"p2/p1 {jump['p2/p1']:.8f}   T2/T1 {jump['T2/T1']:.8f}   "
              f"p02/p01 {jump['p02/p01']:.8f}")
        product = (characteristic_mach(args.mach, GAMMA)
                   * characteristic_mach(jump["M2"], GAMMA))
        print(f"  Prandtl relation M1* M2*          {product:.14f}   (exactly 1, "
              "derived from the energy equation and not from the jump above)")

    if not args.check:
        return 0

    checks: list[tuple[str, bool, str]] = []

    throat = area_ratio(1.0, GAMMA)
    checks.append(("A/A* is exactly 1 at M = 1", abs(throat - 1.0) < 1e-14,
                   "{:.1e} from 1".format(abs(throat - 1.0))))

    # Its only minimum: a step either side must be above it.
    checks.append(("M = 1 is the minimum of A/A*, from both sides",
                   area_ratio(0.999, GAMMA) > 1.0 and area_ratio(1.001, GAMMA) > 1.0,
                   "{:.9f} / {:.9f}".format(area_ratio(0.999, GAMMA),
                                           area_ratio(1.001, GAMMA))))

    star_now = critical(GAMMA)["p*/p0"]
    checks.append(("p*/p0 is the choked ratio to machine precision",
                   abs(star_now - CRITICAL_PRESSURE_RATIO) < 1e-15,
                   "{:.16f}".format(star_now)))

    # The Prandtl relation, over a range, against the jump relations.
    worst = 0.0
    for tenth in range(10, 60):
        m1 = tenth / 10.0
        product = (characteristic_mach(m1, GAMMA)
                   * characteristic_mach(shock(m1, GAMMA)["M2"], GAMMA))
        worst = max(worst, abs(product - 1.0))
    checks.append(("Prandtl relation M1* M2* = 1 across M1 = 1.0 to 5.9",
                   worst < 1e-12, "worst {:.2e}".format(worst)))

    # Total pressure can only fall across a shock, and not at M1 = 1.
    losses = [shock(m / 10.0, GAMMA)["p02/p01"] for m in range(11, 60)]
    checks.append(("total pressure falls across every shock above M1 = 1",
                   all(value < 1.0 for value in losses) and abs(
                       shock(1.0, GAMMA)["p02/p01"] - 1.0) < 1e-12,
                   "max {:.8f}".format(max(losses))))

    # The area inversion recovers the Mach it was given, on both branches.
    errors = []
    for tenth in list(range(1, 10)) + list(range(11, 50)):
        m = tenth / 10.0
        back = mach_from_area(area_ratio(m, GAMMA), supersonic=m > 1.0, gamma=GAMMA)
        errors.append(abs(back - m) / m)
    checks.append(("the area-Mach inversion returns the Mach it was given, both branches",
                   max(errors) < 1e-9, "worst {:.2e}".format(max(errors))))

    refused = False
    try:
        shock(0.8, GAMMA)
    except ValueError:
        refused = True
    checks.append(("an expansion shock at M1 = 0.8 is refused, not computed",
                   refused, "refused" if refused else "RETURNED A NUMBER"))

    departure = density_departure(SWITCH_MACH, GAMMA)
    checks.append((f"advise's M < {SWITCH_MACH:g} switch really is under "
                   f"{SWITCH_CLAIM:.0%} density departure",
                   departure < SWITCH_CLAIM, "{:+.4%}".format(departure)))

    shipped = cross_check_the_shipped_stagnation(0.8)
    t_gap = abs(shipped["T0 shipped"] - shipped["T0 here"]) / shipped["T0 here"]
    p_gap = abs(shipped["p0 shipped"] - shipped["p0 here"]) / shipped["p0 here"]
    checks.append(("compressible.stagnation agrees with the isentropic relation",
                   max(t_gap, p_gap) < 1e-12,
                   "T0 {:.2e}, p0 {:.2e}".format(t_gap, p_gap)))

    print()
    bad = 0
    for what, ok, measured in checks:
        print(f"  {'ok  ' if ok else 'FAIL'}  {what}  [{measured}]")
        bad += 0 if ok else 1
    print("agrees" if not bad else f"{bad} disagreement(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
