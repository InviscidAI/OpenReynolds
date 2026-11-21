"""Conjugate heat transfer: the exact answer a two-region run owes, and the shell trap.

    python3 conjugate.py                 # the reference for the shipped wall
    python3 conjugate.py --check         # grade it, exit non-zero on a disagreement
    python3 conjugate.py --layers 0.01,16 0.05,0.6   # L[m],k[W/m/K] per layer

Why this exists. `advise.py` offers a `conjugate-heat-transfer` family --
`chtMultiRegionFoam` and its steady twin -- and **this repository has no case for either**:
no template writes one, no tutorial row names one, nothing grades one. It is the only
named family in `advise.FAMILIES` with no coverage at all, and it is multi-physics, which
is the part of the product most often asked for.

This is the reference half, not the case. **No `chtMultiRegionFoam` run exists here**, and
a template that writes one is the next item rather than something this file pretends to.
What is here is the answer such a run has to reproduce before anything else it reports is
worth reading, which is the same order `benchmarks/mhd/` and `benchmarks/thermal_stress/`
were built in: the closed form first, the solve second.

What is exact, and why it is the right check for a conjugate solver
------------------------------------------------------------------
Steady conduction through layers in series is **exact**: one flux through all of them,
`q = dT_total / sum(L_i/k_i)`, and the interface temperatures follow by walking the
resistances. There is no correlation and no fitted constant in it.

That is precisely what a conjugate interface gets wrong when it is wrong. The coupling has
to carry two conditions at once -- **temperature continuous** and **flux continuous** --
and a solver that satisfies one and not the other produces a smooth plausible field with
the wrong interface temperature. So the checks here are the two continuities and the
limits, rather than a single number:

* with equal conductivities the interface sits at exactly the geometric fraction
  `L1/(L1+L2)` of the way down the temperature drop -- geometry alone, no properties;
* one layer's resistance dominating drives the interface temperature to that layer's far
  face, exactly, in the limit;
* the per-layer drops sum to the total drop, which is flux continuity restated and is
  what fails if a resistance is dropped or doubled.

The trap: a shell is not a slab
-------------------------------
For a cylindrical shell the resistance per unit length is `ln(r2/r1)/(2 pi k)`, **not**
`(r2-r1)/(k A)` with some chosen `A`. Using the planar formula on a pipe wall is wrong by
a factor that depends on `r2/r1` and goes to 1 only as the wall gets thin. **Measured
here** at `r2/r1 = 1.5`: the inner area gives **+23.3%**, the outer area **-17.8%**, and
the **log-mean** area is exact to machine precision, which is what the log-mean radius is
for.

That +23.3% was written as "about +3.5%" in the first draft of this docstring, from
memory, and the `--check` below failed it on the measured value. A number asserted rather
than computed is the error this whole directory exists to catch, and it is recorded here
rather than quietly corrected: the trap is six times worse than the guess, which makes it
more worth printing, not less. Both forms are in print and the planar one is usually not
labelled as an approximation, so `shell_resistance` and `planar_resistance_of_a_shell`
are both here and the difference is printed.

Standard library only.
"""
from __future__ import annotations

import argparse
import math

WALL = ((0.01, 16.0), (0.05, 0.6))
"""The shipped two-layer wall: 10 mm of stainless at 16 W/m/K against 50 mm of a
mineral-wool-like 0.6 W/m/K. Chosen so the two resistances differ by about 8x, because a
wall whose layers are comparable hides a dropped resistance and one that is 1000x apart
makes every check pass trivially."""

HOT_K, COLD_K = 400.0, 300.0
"""Outer face temperatures, in kelvin. Absolute rather than a rise: a conjugate case's
`T` is absolute and quoting a rise as a temperature is its own small factor trap."""


def resistances(layers: tuple[tuple[float, float], ...]) -> list[float]:
    """`L/k` per layer, the planar series resistance per unit area."""
    out = []
    for length, conductivity in layers:
        if length <= 0.0 or conductivity <= 0.0:
            raise ValueError(f"a layer needs L > 0 and k > 0, got L={length} k={conductivity}")
        out.append(length / conductivity)
    return out


def flux(layers, hot: float = HOT_K, cold: float = COLD_K) -> float:
    """The one flux through every layer, W/m^2. Exact for steady 1D conduction."""
    return (hot - cold) / sum(resistances(layers))


def interface_temperatures(layers, hot: float = HOT_K, cold: float = COLD_K) -> list[float]:
    """Every face temperature, hot face first, cold face last. Exact."""
    q = flux(layers, hot, cold)
    temperatures = [hot]
    for resistance in resistances(layers):
        temperatures.append(temperatures[-1] - q * resistance)
    return temperatures


def shell_resistance(inner: float, outer: float, conductivity: float) -> float:
    """`ln(r2/r1)/(2 pi k)`, per unit length of pipe. Exact for a cylindrical shell."""
    if not 0.0 < inner < outer:
        raise ValueError(f"need 0 < r1 < r2, got r1={inner} r2={outer}")
    return math.log(outer / inner) / (2.0 * math.pi * conductivity)


def planar_resistance_of_a_shell(inner: float, outer: float, conductivity: float,
                                 use: str = "inner") -> float:
    """The planar formula misapplied to a shell, per unit length, for comparison.

    `(r2-r1)/(k A)` with `A` the inner, outer or log-mean circumference. The log-mean one
    is exact -- that is what the log-mean radius is for -- and the other two are not, which
    is the point of printing all three.
    """
    thickness = outer - inner
    if use == "inner":
        area = 2.0 * math.pi * inner
    elif use == "outer":
        area = 2.0 * math.pi * outer
    elif use == "logmean":
        area = 2.0 * math.pi * (outer - inner) / math.log(outer / inner)
    else:
        raise ValueError("use must be inner, outer or logmean")
    return thickness / (conductivity * area)


def biot(coefficient: float, length: float, conductivity: float) -> float:
    """`Bi = h L / k`: above about 0.1 a lumped region is the wrong model.

    Here because it is the number that says whether a conjugate case is needed at all --
    a body with `Bi << 1` is isothermal and a one-region case with a surface condition
    answers the same question more cheaply, which is the judgement `advise.py` makes for
    every other family.
    """
    return coefficient * length / conductivity


def main() -> int:
    parser = argparse.ArgumentParser(
        description="The exact steady answer a conjugate two-region run owes.")
    parser.add_argument("--layers", nargs="*", default=None,
                        help="L,k per layer in SI, hot side first (default: the shipped wall)")
    parser.add_argument("--hot", type=float, default=HOT_K)
    parser.add_argument("--cold", type=float, default=COLD_K)
    parser.add_argument("--check", action="store_true",
                        help="grade the identities and exit non-zero on a disagreement")
    args = parser.parse_args()

    if args.layers:
        layers = tuple(tuple(float(part) for part in spec.split(",")) for spec in args.layers)
        for layer in layers:
            if len(layer) != 2:
                parser.error("each --layers entry is L,k")
    else:
        layers = WALL

    q = flux(layers, args.hot, args.cold)
    faces = interface_temperatures(layers, args.hot, args.cold)
    print(f"# steady 1D conduction, {len(layers)} layers in series, exact")
    print(f"  hot face {args.hot:g} K, cold face {args.cold:g} K, drop {args.hot - args.cold:g} K")
    print(f"  flux        {q:.8f} W/m^2  (one flux through every layer)")
    for index, (length, conductivity) in enumerate(layers):
        resistance = length / conductivity
        print(f"  layer {index + 1}: L {length:g} m  k {conductivity:g} W/m/K  "
              f"R {resistance:.8f} m^2K/W  drop {q * resistance:.6f} K")
    for index, temperature in enumerate(faces):
        label = "hot face" if index == 0 else (
            "cold face" if index == len(faces) - 1 else f"interface {index}")
        print(f"  {label:12} {temperature:.8f} K")

    print()
    print("  a shell is not a slab: r1 20 mm, r2 30 mm, k 16 W/m/K, per metre")
    exact = shell_resistance(0.020, 0.030, 16.0)
    print(f"    ln(r2/r1)/(2 pi k)          {exact:.10f} K m/W   exact")
    for use in ("inner", "outer", "logmean"):
        wrong = planar_resistance_of_a_shell(0.020, 0.030, 16.0, use)
        print(f"    planar on the {use:8}      {wrong:.10f}   {(wrong - exact) / exact:+.3%}")

    if not args.check:
        return 0

    checks: list[tuple[str, bool, str]] = []

    # Flux continuity, restated: the drops have to sum to the total.
    drops = [q * resistance for resistance in resistances(layers)]
    checks.append(("the per-layer drops sum to the total drop (flux continuity)",
                   abs(sum(drops) - (args.hot - args.cold)) < 1e-9,
                   "{:.3e} K".format(abs(sum(drops) - (args.hot - args.cold)))))

    # Temperature continuity: walking from the cold side gives the same faces.
    backward = interface_temperatures(tuple(reversed(layers)), args.cold, args.hot)
    worst = max(abs(a - b) for a, b in zip(faces, reversed(backward)))
    checks.append(("the faces are the same walked from either side (temperature continuity)",
                   worst < 1e-9, "{:.3e} K".format(worst)))

    # Equal conductivities: the interface sits at the geometric fraction, exactly.
    equal = ((0.01, 5.0), (0.03, 5.0))
    middle = interface_temperatures(equal, 400.0, 300.0)[1]
    expected = 400.0 - 100.0 * (0.01 / 0.04)
    checks.append(("with equal k the interface is at the geometric fraction L1/(L1+L2)",
                   abs(middle - expected) < 1e-9,
                   "{:.8f} vs {:.8f} K".format(middle, expected)))

    # A dominating resistance drives the interface to that layer's far face.
    lopsided = ((0.01, 1.0e6), (0.05, 0.05))
    first = interface_temperatures(lopsided, 400.0, 300.0)[1]
    checks.append(("a negligible first resistance leaves the interface at the hot face",
                   abs(first - 400.0) < 1e-3, "{:.6f} K".format(first)))

    # The shell: the log-mean planar form is exact, the inner-area one is not.
    exact_shell = shell_resistance(0.020, 0.030, 16.0)
    logmean = planar_resistance_of_a_shell(0.020, 0.030, 16.0, "logmean")
    inner = planar_resistance_of_a_shell(0.020, 0.030, 16.0, "inner")
    checks.append(("the log-mean area makes the planar formula exact for a shell",
                   abs(logmean - exact_shell) / exact_shell < 1e-12,
                   "{:.2e}".format(abs(logmean - exact_shell) / exact_shell)))
    # The window brackets the derived value: (r2-r1)/r1 / ln(r2/r1) - 1, which is
    # 0.5/ln(1.5) - 1 = 0.23320 at this radius ratio. Asserted against the algebra
    # rather than against the printed number, so the check cannot drift with it.
    derived = 0.5 / math.log(1.5) - 1.0
    measured = (inner - exact_shell) / exact_shell
    checks.append(("the inner-area planar form is wrong by exactly "
                   "(r2-r1)/r1/ln(r2/r1) - 1, not by noise",
                   abs(measured - derived) < 1e-12,
                   "{:+.4%} against a derived {:+.4%}".format(measured, derived)))
    outer_wrong = (planar_resistance_of_a_shell(0.020, 0.030, 16.0, "outer")
                   - exact_shell) / exact_shell
    checks.append(("and the outer area errs the other way, so neither is a safe default",
                   outer_wrong < -0.1 < 0.0 < measured,
                   "{:+.3%} against {:+.3%}".format(outer_wrong, measured)))

    # Refusals: a zero conductivity and an inverted shell are not answerable.
    refused = 0
    for bad in (lambda: flux(((0.01, 0.0),), 400.0, 300.0),
                lambda: shell_resistance(0.03, 0.02, 16.0)):
        try:
            bad()
        except ValueError:
            refused += 1
    checks.append(("a zero conductivity and an inverted shell are refused",
                   refused == 2, f"{refused} of 2"))

    # Biot: the number that says a conjugate case is needed at all.
    thin = biot(10.0, 0.001, 16.0)
    thick = biot(500.0, 0.05, 0.6)
    checks.append(("Bi separates a lumped region from one that needs solving",
                   thin < 0.1 < thick, "{:.2e} and {:.1f}".format(thin, thick)))

    print()
    bad_count = 0
    for what, ok, measured in checks:
        print(f"  {'ok  ' if ok else 'FAIL'}  {what}  [{measured}]")
        bad_count += 0 if ok else 1
    print("agrees" if not bad_count else f"{bad_count} disagreement(s)")
    return 1 if bad_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
