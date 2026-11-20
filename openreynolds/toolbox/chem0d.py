#!/usr/bin/env python3
"""A `chemFoam` well-stirred reactor: the case, and the temperature it has to reach.

    python3 chem0d.py write                       # a chemFoam case in the cwd
    python3 chem0d.py write --phi 0.8 --t0 1100
    python3 chem0d.py predict --json              # the target, from the case's own data
    python3 chem0d.py grade                       # what the solved case actually did

`chemFoam` is OpenFOAM's single-cell chemistry solver: one hexahedral cell
(`createSingleCellMesh.H` builds it and `createFields.H` refuses a mesh with any other
number), no transport, no mesh to get wrong, and a closed reactor held at either
constant pressure or constant volume while the chemistry integrates. It is the cheapest
honest piece of reacting flow there is -- the whole run below is a fraction of a second
of wall clock -- and unlike a flame it has an answer that can be computed independently.

`write` writes the case. `predict` computes what the final temperature has to be, from
the JANAF coefficients in the case's own thermo file, by conserving enthalpy (or
internal energy, under `constantProperty volume`). `grade` reads what the run produced
and puts the two beside each other.


WHY THE FINAL TEMPERATURE IS KNOWN WITHOUT A PAPER
--------------------------------------------------
`chemFoam` carries sensible enthalpy and adds the chemical release as a source. From
`applications/solvers/combustion/chemFoam/` in OpenFOAM v2512:

    readInitialConditions.H   h0 = sum_i Y0_i * Hs_i(p, T0)
    solveChemistry.H          integratedHeat += (Qdot/rho) * dt
    hEqn.H                    h = h0 + integratedHeat                (constant pressure)
                              h = u0 + p/rho + integratedHeat        (constant volume)

`Qdot` in `chemistryModel` is `-sum_i Hf_i * RR_i`, the chemical enthalpy released by
the rates, and `Hs_i = Ha_i - Hc_i` with `Hc_i = Ha_i(Tstd)` by
`janafThermoI.H:207,212`. Adding those up, the `Hf`/`Hc` terms cancel and what the
solver conserves at constant pressure is the **absolute** enthalpy of the mixture:

    sum_i Y_i(end) * Ha_i(T_end) = sum_i Y_i(0) * Ha_i(T0)                     (1)

which is the constant-pressure adiabatic reactor, and is one scalar equation in one
unknown. Under `constantProperty volume` the same algebra with `u = h - R_mix*T` gives

    sum_i Y_i(end) * (Ha_i(T_end) - R_i*T_end)
        = sum_i Y_i(0) * (Ha_i(T0) - R_i*T0)                                   (2)

and the end pressure follows from the equation of state at the unchanged density,
`p_end = rho0 * R_mix(end) * T_end`, which is a second independent number.

`Ha_i` is exactly what `janafThermoI.H:185-196` evaluates, with the dictionary's
coefficients scaled to a mass basis by `R_i = RR/W_i` (`janafThermo.C:74-75`,
`specieI.H:87-89`):

    Ha_i(T) = (RR/W_i) * ( ((((a4/5*T + a3/4)*T + a2/3)*T + a1/2)*T + a0)*T + a5 )

with `RR = 1e3 * NA * k` (`thermodynamicConstants.C`), `NA = 6.0221417930e+23`
(`fundamentalConstants.C`) and `k = 1.38065e-23` (`etc/controlDict`), so
`RR = 8314.47006650545 J/(kmol.K)`. `perfectGas::H` is zero
(`perfectGasI.H:83-86`), so there is no pressure term in `Ha` to account for.

This is the same move `benchmarks/thermal_stress/` makes: the reference is not a
published number but the solver's own constitutive data, re-evaluated independently, so
there is nothing to misquote. `benchmarks/README.md` is blunt about why that matters --
of its eleven published reference values, re-checking two against their actual sources
changed them, one by a factor of two.


AND WHERE THE REAL-WORLD DATA COMES IN
--------------------------------------
Equation (1) is only as true as the thermo data in it, so the data is checked against a
measurement rather than trusted. `Ha_i(298.15 K)` is, for a NASA/JANAF polynomial fit,
the standard enthalpy of formation, and those are measured quantities with published
uncertainties. `predict` prints each one beside the value in NIST's Chemistry WebBook
(see NIST_FORMATION_KJ_MOL below, read from the WebBook's own gas-phase thermochemistry
pages rather than from a secondary source), and prints the heat of reaction and the
lower heating value that follow from them.

What `predict` does **not** claim is a real flame temperature. A single irreversible
global step has no dissociation in it: it cannot make CO, OH, H, O or NO, and all five
are endothermic sinks that a real stoichiometric hydrocarbon flame spends a substantial
part of its heat release on. So equation (1) is the complete-combustion temperature, the
run reproduces it, and the measured adiabatic flame temperature of the same mixture is
**lower**. The size of that gap is not asserted here, because no primary source for it
has been read in this repository; `benchmarks/sph/` makes the same call about Martin &
Moyce's dam-break non-dimensionalisation, and `benchmarks/README.md` argues at length
that quoting such a number from a secondary source is the single most expensive habit in
validation work. NASA CEA (the Glenn equilibrium code, open access, documented in NASA
RP-1311) and the NIST-JANAF tables are where to get it with dissociation included.


WHAT THE CASE IS, AND WHAT IT DELIBERATELY IS NOT
-------------------------------------------------
Stoichiometric or lean methane in air, one irreversible global step, from a preheat.

* **The mechanism and the thermo data are copied from the distribution, not retyped.**
  `write` copies `constant/reactions` and `constant/thermo.compressibleGas` out of
  `$FOAM_TUTORIALS/combustion/reactingFoam/laminar/counterFlowFlame2D/constant/`. That
  is `CH4 + 2O2 = CO2 + 2H2O` as an `irreversibleArrheniusReaction` with the five
  species CH4, O2, CO2, H2O and N2, and it is the image's own file -- so the data
  `predict` evaluates is byte for byte the data the solver reads, and nothing about this
  mechanism is this repository's assertion. Off the image, `write` says which path it
  wanted rather than inventing coefficients.

* **It starts at 1000 K, not at 300 K**, and that is a property of the mechanism rather
  than a convenience. The single step's rate is `A*exp(-Ta/T)*[CH4]*[O2]^2` with
  `Ta = 14906 K`, so at 300 K the exponential is about 3e-22 and nothing happens in any
  finite run: a one-step mechanism has no ignition chemistry and cannot ignite itself
  from cold. At 1000 K and 1 atm the fuel consumption time is of order 10 ms, which is
  what makes this a case rather than a stalled job. The preheat changes the final
  temperature (it adds its own sensible enthalpy) and changes nothing about equation (1).

* **Rich mixtures are refused.** With no CO and no H2 in the species list, a one-step
  mechanism burns a rich mixture to leftover methane and calls it done, which is not a
  thing that happens. `--phi` above 1 is refused by name rather than run.

* **No ignition delay is graded.** It is the obvious thing to measure in a 0-D reactor
  and it is the wrong thing to measure here: with a one-step global mechanism the delay
  is a property of `A` and `Ta` and not of combustion, and a published delay would be
  for a detailed mechanism this case does not contain. Grading it would be grading an
  Arrhenius fit against somebody's shock tube. The temperature at the end is
  thermodynamics and survives that objection, which is why it is the graded quantity.

The whole of it is the standard library: no numpy, because a single root find by
bisection is four lines and this script may be read on a machine with nothing on it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

# -- the constants, each from the file that defines it --------------------------

AVOGADRO = 6.0221417930e+23      # fundamentalConstants.C, physicoChemical::NA default
BOLTZMANN = 1.38065e-23          # etc/controlDict, DimensionedConstants/physicoChemical
RR = 1.0e3 * AVOGADRO * BOLTZMANN   # thermodynamicConstants.C: J/(kmol.K)
TSTD = 298.15                    # etc/controlDict, standard/Tstd

TUTORIAL_SOURCE = ("combustion/reactingFoam/laminar/counterFlowFlame2D/constant")
"""Where the mechanism and the thermo data are copied from, under `$FOAM_TUTORIALS`.

Copied rather than committed: the data is then the image's own, `predict` is evaluating
the identical bytes the solver reads, and no JANAF coefficient in this repository is
something a person here typed. It is also the only file in the chain carrying a licence
that is not this repository's.
"""

NIST_FORMATION_KJ_MOL = {
    # NIST Chemistry WebBook, gas-phase thermochemistry ("Mask=1") pages, read 2026-11-21
    # at https://webbook.nist.gov/cgi/cbook.cgi?ID=<id>&Units=SI&Mask=1 . Each entry is
    # (value, uncertainty or None, the WebBook's own reference line). Where the WebBook
    # lists several, the review value it ranks first is taken and the other is in the
    # note, because the spread between them is part of the reference's uncertainty and
    # hiding it is the habit `benchmarks/README.md` exists to argue against.
    "CH4": (-74.6, 0.3, "C74828; Manion 2002, adopting Gurvich, Veyts et al. 1991. "
                        "The WebBook also lists -74.87 (Chase 1998, reviewed 1961), "
                        "-74.85 +/- 0.31 (Prosen and Rossini 1945) and "
                        "-74.5 +/- 0.4 (Pittam and Pilcher 1972)"),
    "CO2": (-393.51, 0.13, "C124389; Cox, Wagman et al. 1984, CODATA review value. "
                           "The WebBook also lists -393.52 (Chase 1998)"),
    "H2O": (-241.826, 0.040, "C7732185; Cox, Wagman et al. 1984, CODATA review value. "
                             "The WebBook also lists -241.83 (Chase 1998)"),
    "O2": (0.0, None, "an element in its standard state, by definition"),
    "N2": (0.0, None, "an element in its standard state, by definition"),
}

# -- what `write` writes -------------------------------------------------------

DEFAULT_T0_K = 1000.0
DEFAULT_PRESSURE_PA = 101325.0
DEFAULT_END_TIME_S = 0.5
DEFAULT_MAX_DELTA_T_S = 1.0e-4

FUEL = "CH4"
OXIDISER = "O2"
DILUENT = "N2"
NITROGEN_PER_OXYGEN = 3.76
"""Moles of N2 per mole of O2 in dry air.

79/21 = 3.7619, which is what every combustion text and the `chemFoam` gri tutorial's
own `initialConditions` use. It is a definition of "air" for this purpose and not a
measurement, so it is stated rather than cited.
"""


class Chem0dError(Exception):
    """Something the case does not supply, which is a refusal and not a result."""


# -- reading OpenFOAM dictionaries ---------------------------------------------


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def top_level_blocks(text: str) -> dict[str, str]:
    """`{name: body}` for every `name { ... }` at brace depth zero.

    Written rather than regexed because the species entries in a thermo file are three
    deep -- `CH4 { thermodynamics { highCpCoeffs ( ... ); } }` -- and a regex that
    matched the outer braces would have to count them anyway. Fourteen lines of depth
    counting is the smaller thing to keep right, and it is the only parsing this script
    needs: everything inside a block is a flat `key value;` or a `( ... )` list.
    """
    text = strip_comments(text)
    blocks: dict[str, str] = {}
    name = ""
    index = 0
    while index < len(text):
        char = text[index]
        if char == "{":
            depth, start = 1, index + 1
            index += 1
            while index < len(text) and depth:
                if text[index] == "{":
                    depth += 1
                elif text[index] == "}":
                    depth -= 1
                index += 1
            if name:
                blocks[name] = text[start:index - 1]
            name = ""
            continue
        if char.isspace() or char == ";":
            if char == ";":
                name = ""
            index += 1
            continue
        match = re.match(r"[A-Za-z_][\w.]*", text[index:])
        if match:
            name = match.group(0)
            index += len(name)
            continue
        name = ""
        index += 1
    return blocks


def scalar_entry(body: str, key: str) -> float:
    match = re.search(rf"\b{key}\s+(-?[\d.]+(?:[eE][-+]?\d+)?)\s*;", body)
    if not match:
        raise Chem0dError(f"no `{key}` entry")
    return float(match.group(1))


def list_entry(body: str, key: str) -> list[float]:
    match = re.search(rf"\b{key}\s*\(([^)]*)\)", body)
    if not match:
        raise Chem0dError(f"no `{key}` list")
    return [float(piece) for piece in match.group(1).split()]


def name_list(text: str, key: str) -> list[str]:
    match = re.search(rf"\b{key}\s*\(([^)]*)\)", strip_comments(text))
    if not match:
        raise Chem0dError(f"no `{key}` list")
    return match.group(1).split()


# -- the thermodynamics --------------------------------------------------------


class Specie:
    """One species' JANAF fit, on a mass basis, exactly as the solver scales it."""

    def __init__(self, name: str, body: str) -> None:
        self.name = name
        thermo = top_level_blocks(body).get("thermodynamics")
        specie = top_level_blocks(body).get("specie")
        if thermo is None or specie is None:
            raise Chem0dError(f"{name}: no `specie` or `thermodynamics` sub-dictionary")
        self.molecular_weight = scalar_entry(specie, "molWeight")   # kg/kmol
        self.gas_constant = RR / self.molecular_weight              # J/(kg.K)
        self.t_low = scalar_entry(thermo, "Tlow")
        self.t_high = scalar_entry(thermo, "Thigh")
        self.t_common = scalar_entry(thermo, "Tcommon")
        self.high = list_entry(thermo, "highCpCoeffs")
        self.low = list_entry(thermo, "lowCpCoeffs")
        for which, coeffs in (("high", self.high), ("low", self.low)):
            if len(coeffs) != 7:
                raise Chem0dError(
                    f"{name}: {which}CpCoeffs has {len(coeffs)} entries and a NASA "
                    "polynomial has 7")

    def coefficients(self, temperature: float) -> list[float]:
        """`janafThermoI.H` switches on `T > Tcommon`, with no blending."""
        return self.high if temperature > self.t_common else self.low

    def absolute_enthalpy(self, temperature: float) -> float:
        """Ha, J/kg. `janafThermoI.H:185-196` with `perfectGas::H` zero."""
        a = self.coefficients(temperature)
        t = temperature
        poly = ((((a[4] / 5.0 * t + a[3] / 4.0) * t + a[2] / 3.0) * t
                 + a[1] / 2.0) * t + a[0]) * t + a[5]
        return self.gas_constant * poly

    def formation_enthalpy(self, ) -> float:
        """Hc, J/kg: Ha at Tstd off the **low** coefficients whatever Tcommon is, which
        is what `janafThermoI.H:212-222` does and is not the same as `absolute_enthalpy
        (TSTD)` for a species whose Tcommon is below 298.15."""
        a = self.low
        t = TSTD
        poly = ((((a[4] / 5.0 * t + a[3] / 4.0) * t + a[2] / 3.0) * t
                 + a[1] / 2.0) * t + a[0]) * t + a[5]
        return self.gas_constant * poly


def read_species(case: Path) -> dict[str, Specie]:
    path = case / "constant" / "thermo"
    if not path.is_file():
        raise Chem0dError(
            f"{path} is not there. `chem0d.py write` copies it from\n"
            f"    $FOAM_TUTORIALS/{TUTORIAL_SOURCE}/thermo.compressibleGas")
    blocks = top_level_blocks(path.read_text(encoding="utf-8"))
    blocks.pop("FoamFile", None)
    species = {name: Specie(name, body) for name, body in blocks.items()}
    if not species:
        raise Chem0dError(f"{path} declares no species")
    return species


def read_reaction(case: Path) -> tuple[dict[str, float], dict[str, float]]:
    """`({reactant: nu}, {product: nu})` from the single reaction in `constant/reactions`.

    Exactly one reaction, because the complete-combustion composition `predict` needs is
    only determined by stoichiometry when there is one extent of reaction. A multi-step
    mechanism is therefore refused rather than guessed at, and the refusal says so: the
    thermodynamics in (1) would still hold, but which composition to apply it to would
    be an assumption instead of arithmetic.
    """
    path = case / "constant" / "reactions"
    if not path.is_file():
        raise Chem0dError(
            f"{path} is not there. `chem0d.py write` copies it from\n"
            f"    $FOAM_TUTORIALS/{TUTORIAL_SOURCE}/reactions")
    text = strip_comments(path.read_text(encoding="utf-8"))
    strings = re.findall(r'reaction\s+"([^"]+)"', text)
    if len(strings) != 1:
        raise Chem0dError(
            f"{path} has {len(strings)} reactions. The complete-combustion prediction "
            "needs one extent of reaction, so a multi-step mechanism is refused rather "
            "than guessed at: the enthalpy balance would still hold, but which "
            "composition to apply it to would be an assumption instead of arithmetic")
    left, _, right = strings[0].partition("=")
    if not right:
        raise Chem0dError(f"{strings[0]!r} has no `=` in it")

    def side(part: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for term in part.split("+"):
            term = term.strip()
            if not term:
                continue
            if "^" in term:
                raise Chem0dError(
                    f"{term!r} carries a reaction-order override; the stoichiometry "
                    "this needs is then not what the string says")
            match = re.fullmatch(r"(\d*\.?\d*)\s*([A-Za-z][\w*()-]*)", term)
            if not match:
                raise Chem0dError(f"cannot read the stoichiometry of {term!r}")
            out[match.group(2)] = float(match.group(1) or 1.0)
        return out

    return side(left), side(right)


def mass_fractions(moles: dict[str, float], species: dict[str, Specie]
                   ) -> dict[str, float]:
    """Mole fractions (or relative moles) to mass fractions, normalised.

    The same two lines as `readInitialConditions.H` under `fractionBasis mole`:
    `Y_i = X_i*W_i/sum(X_j*W_j)` after normalising X.
    """
    masses = {}
    for name, amount in moles.items():
        if name not in species:
            raise Chem0dError(f"{name} is not in the thermo file")
        masses[name] = amount * species[name].molecular_weight
    total = sum(masses.values())
    if total <= 0.0:
        raise Chem0dError("the mixture has no mass in it")
    return {name: mass / total for name, mass in masses.items()}


def burn_completely(moles: dict[str, float], reactants: dict[str, float],
                    products: dict[str, float]) -> dict[str, float]:
    """The composition after the single reaction has run as far as it can.

    The extent is set by whichever reactant runs out first, `min(n_i/nu_i)`. Everything
    this reaction does not touch -- the nitrogen -- is carried through unchanged.
    """
    extent = min(moles.get(name, 0.0) / nu for name, nu in reactants.items())
    out = dict(moles)
    for name, nu in reactants.items():
        out[name] = out.get(name, 0.0) - nu * extent
    for name, nu in products.items():
        out[name] = out.get(name, 0.0) + nu * extent
    return {name: amount for name, amount in out.items() if amount > 1e-12}


def mixture_energy(fractions: dict[str, float], species: dict[str, Specie],
                   temperature: float, constant_volume: bool) -> float:
    """The conserved quantity: absolute enthalpy, or absolute internal energy.

    `u = h - R_mix*T` for a perfect gas, and `R_mix = sum Y_i*R_i`, so the two differ by
    one term. Which of the two is conserved is the case's own `constantProperty`.
    """
    total = sum(fraction * species[name].absolute_enthalpy(temperature)
                for name, fraction in fractions.items())
    if constant_volume:
        total -= temperature * sum(fraction * species[name].gas_constant
                                   for name, fraction in fractions.items())
    return total


def solve_temperature(target: float, fractions: dict[str, float],
                      species: dict[str, Specie], constant_volume: bool) -> float:
    """The root of `mixture_energy(T) = target`, by bisection.

    Bisection rather than Newton because the JANAF fit is piecewise at `Tcommon` and its
    derivative jumps there, and because scipy is not on the image and is not wanted
    (`tests/test_toolbox.py`). The bracket is the narrowest interval every species'
    fit is valid over, so a root outside it is reported as such rather than returned
    from an extrapolated polynomial: `janafThermo::limit` would clamp T there and the
    answer would be a boundary value dressed as a solution.
    """
    low = max(specie.t_low for specie in species.values())
    high = min(specie.t_high for specie in species.values())
    energy_low = mixture_energy(fractions, species, low, constant_volume)
    energy_high = mixture_energy(fractions, species, high, constant_volume)
    if not energy_low <= target <= energy_high:
        raise Chem0dError(
            f"the conserved energy {target:.6g} J/kg is outside what this mixture can "
            f"have between {low:g} K ({energy_low:.6g}) and {high:g} K "
            f"({energy_high:.6g}); every species' fit is valid only in that range")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if mixture_energy(fractions, species, middle, constant_volume) < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


# -- the initial conditions ----------------------------------------------------


def read_initial_conditions(case: Path) -> dict[str, Any]:
    path = case / "constant" / "initialConditions"
    if not path.is_file():
        raise Chem0dError(f"{path} is not there; `chem0d.py write` writes it")
    text = strip_comments(path.read_text(encoding="utf-8"))
    basis = re.search(r"\bfractionBasis\s+(\w+)\s*;", text)
    const = re.search(r"\bconstantProperty\s+(\w+)\s*;", text)
    if basis is None or const is None:
        raise Chem0dError(f"{path} has no `fractionBasis` or no `constantProperty`")
    if basis.group(1) != "mole":
        raise Chem0dError(
            f"`fractionBasis {basis.group(1)}` -- this reads mole fractions only, "
            "because that is what it writes and a mass basis would be read as moles "
            "and silently rebalance the mixture")
    blocks = top_level_blocks(text)
    if "fractions" not in blocks:
        raise Chem0dError(f"{path} has no `fractions` sub-dictionary")
    moles = {match.group(1): float(match.group(2)) for match in
             re.finditer(r"([A-Za-z][\w*()-]*)\s+(-?[\d.]+(?:[eE][-+]?\d+)?)\s*;",
                         blocks["fractions"])}
    if not moles:
        raise Chem0dError(f"{path} lists no fractions")
    return {"moles": moles,
            "temperature": scalar_entry(text, "T"),
            "pressure": scalar_entry(text, "p"),
            "constant_volume": const.group(1) == "volume"}


# -- predict -------------------------------------------------------------------


def predict(case: Path) -> dict[str, Any]:
    """The target temperature, and the thermo data's own agreement with NIST."""
    species = read_species(case)
    initial = read_initial_conditions(case)
    reactants, products = read_reaction(case)

    y0 = mass_fractions(initial["moles"], species)
    final_moles = burn_completely(initial["moles"], reactants, products)
    y_final = mass_fractions(final_moles, species)

    constant_volume = initial["constant_volume"]
    t0, p0 = initial["temperature"], initial["pressure"]
    conserved = mixture_energy(y0, species, t0, constant_volume)
    t_end = solve_temperature(conserved, y_final, species, constant_volume)

    r_initial = sum(y * species[name].gas_constant for name, y in y0.items())
    r_final = sum(y * species[name].gas_constant for name, y in y_final.items())
    rho0 = p0 / (r_initial * t0)
    p_end = rho0 * r_final * t_end if constant_volume else p0

    # The thermo data against the measurement behind it.
    formation = {}
    for name, specie in species.items():
        # kJ/mol: Ha is J/kg and W is kg/kmol, so Ha*W is J/kmol = kJ/kmol... which is
        # J/mol. Divided by 1000 for kJ/mol.
        value = specie.formation_enthalpy() * specie.molecular_weight / 1.0e6
        row: dict[str, Any] = {"janaf_kj_mol": value}
        if name in NIST_FORMATION_KJ_MOL:
            nist, uncertainty, reference = NIST_FORMATION_KJ_MOL[name]
            row.update(nist_kj_mol=nist, nist_uncertainty=uncertainty,
                       difference_kj_mol=value - nist, reference=reference)
            if abs(nist) > 1e-9:
                row["difference_percent"] = 100.0 * (value - nist) / abs(nist)
        formation[name] = row

    # The heat of reaction at Tstd, per mole of fuel, and the lower heating value --
    # "lower" because the water in `products` is gaseous, which is what the species
    # list has and what makes this the LHV and not the HHV.
    def formation_sum(side: dict[str, float]) -> float:
        return sum(nu * species[name].formation_enthalpy()
                   * species[name].molecular_weight / 1.0e6
                   for name, nu in side.items())

    heat_of_reaction = formation_sum(products) - formation_sum(reactants)
    fuel_nu = reactants.get(FUEL)
    heating_value = None
    if fuel_nu:
        # kJ/mol divided by g/mol is kJ/g, which is MJ/kg: no conversion factor, and
        # molWeight in a thermo file is kg/kmol, which is the same number as g/mol.
        heating_value = -heat_of_reaction / (
            fuel_nu * species[FUEL].molecular_weight)   # MJ/kg

    return {
        "case": str(case),
        "constant_property": "volume" if constant_volume else "pressure",
        "initial": {"temperature_k": t0, "pressure_pa": p0, "moles": initial["moles"],
                    "mass_fractions": y0, "density_kg_m3": rho0},
        "final": {"moles": final_moles, "mass_fractions": y_final,
                  "temperature_k": t_end, "pressure_pa": p_end},
        "conserved_j_kg": conserved,
        "reaction": {"reactants": reactants, "products": products},
        "formation_enthalpies": formation,
        "heat_of_reaction_kj_mol": heat_of_reaction,
        "lower_heating_value_mj_kg": heating_value,
        "species": sorted(species),
    }


def print_prediction(result: dict[str, Any]) -> None:
    initial, final = result["initial"], result["final"]
    print(f"chem0d: {result['case']}, constantProperty "
          f"{result['constant_property']}")
    print(f"  from      {initial['temperature_k']:.2f} K, "
          f"{initial['pressure_pa']:.6g} Pa, rho {initial['density_kg_m3']:.6f} kg/m3")
    print("  moles in  " + "  ".join(f"{name} {amount:g}"
                                     for name, amount in initial["moles"].items()))
    print("  moles out " + "  ".join(f"{name} {amount:g}"
                                     for name, amount in final["moles"].items()))
    print(f"  conserved {result['conserved_j_kg'] / 1e6:+.6f} MJ/kg "
          f"({'internal energy' if result['constant_property'] == 'volume' else 'enthalpy'}"
          ", absolute)")
    print(f"  TARGET    {final['temperature_k']:.4f} K")
    if result["constant_property"] == "volume":
        print(f"            {final['pressure_pa']:.6g} Pa at the unchanged density")
    print("  the thermo data's own formation enthalpies against NIST's "
          "(kJ/mol, gas phase):")
    for name in sorted(result["formation_enthalpies"]):
        row = result["formation_enthalpies"][name]
        line = f"    {name:5s} {row['janaf_kj_mol']:+10.4f}"
        if "nist_kj_mol" in row:
            uncertainty = row["nist_uncertainty"]
            spread = f" +/- {uncertainty:g}" if uncertainty else ""
            line += f"  NIST {row['nist_kj_mol']:+.4f}{spread}"
            line += f"  diff {row['difference_kj_mol']:+.4f}"
            if "difference_percent" in row:
                line += f" ({row['difference_percent']:+.3g}%)"
        print(line)
    print(f"  heat of reaction at {TSTD} K: "
          f"{result['heat_of_reaction_kj_mol']:+.3f} kJ/mol of the reaction as written")
    if result["lower_heating_value_mj_kg"] is not None:
        print(f"  lower heating value of {FUEL}: "
              f"{result['lower_heating_value_mj_kg']:.4f} MJ/kg")
    print("  NOT a flame temperature: one irreversible step makes no CO, OH, H, O or "
          "NO,")
    print("  so a real flame of this mixture is cooler. See the module docstring.")


# -- grade ---------------------------------------------------------------------


def read_chem_foam_out(case: Path) -> dict[str, Any]:
    """The last row of `chemFoam.out`, which the solver writes itself.

    `createFields.H` opens `args.path()/"chemFoam.out"` and `output.H` appends
    `time<TAB>T<TAB>p` every write, so this is the solver's own record and not an
    analysis of it.
    """
    path = case / "chemFoam.out"
    if not path.is_file():
        raise Chem0dError(
            f"{path} is not there. chemFoam writes it itself, so a case without one "
            "has not been run -- or was run somewhere else, since the path is the "
            "`-case` argument's and not the cwd's")
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if not rows:
        raise Chem0dError(f"{path} has a header and no rows")
    last = rows[-1]
    if len(last) < 3:
        raise Chem0dError(f"{path} last row is {last}, not time/T/p")
    return {"time_s": float(last[0]), "temperature_k": float(last[1]),
            "pressure_pa": float(last[2]), "rows": len(rows),
            "series": [(float(row[0]), float(row[1])) for row in rows]}


def latest_time_dir(case: Path) -> Path | None:
    """The newest numbered time directory, which is where the final Y fields are."""
    times = []
    for entry in case.iterdir():
        if not entry.is_dir():
            continue
        try:
            times.append((float(entry.name), entry))
        except ValueError:
            continue
    if not times:
        return None
    return max(times)[1]


def read_uniform_field(path: Path) -> float | None:
    """The one cell's value out of a `volScalarField`.

    A single-cell field is written `internalField uniform <v>;` because every value in
    it is equal; the `nonuniform` form is read too rather than assumed away, since which
    one OpenFOAM writes is its decision and not this script's.
    """
    if not path.is_file():
        return None
    text = strip_comments(path.read_text(encoding="utf-8"))
    match = re.search(r"internalField\s+uniform\s+(-?[\d.]+(?:[eE][-+]?\d+)?)", text)
    if match:
        return float(match.group(1))
    match = re.search(r"internalField\s+nonuniform[^(]*\(\s*"
                      r"(-?[\d.]+(?:[eE][-+]?\d+)?)", text)
    return float(match.group(1)) if match else None


def grade(case: Path, tolerance: float = 5.0e-3) -> dict[str, Any]:
    """What the run reached, against the prediction, two ways.

    Twice, because the two say different things. Against the **complete-combustion**
    composition the check is the whole case: thermodynamics plus the mechanism having
    actually run to completion. Against the **solver's own final composition**, read out
    of the last time directory, it is thermodynamics alone -- the same trick
    `test_journey_thermal_stress.py` uses when it grades sigma against `-E*alpha*T` at
    the solver's own T, so that an incomplete burn cannot be mistaken for an energy
    error or hide one.
    """
    species = read_species(case)
    initial = read_initial_conditions(case)
    predicted = predict(case)
    run = read_chem_foam_out(case)
    checks: list[dict[str, Any]] = []

    def check(name: str, value: Any, reference: Any, passed: bool | None,
              note: str = "") -> None:
        checks.append({"name": name, "value": value, "reference": reference,
                       "passed": None if passed is None else bool(passed),
                       "note": note})

    target = predicted["final"]["temperature_k"]
    got = run["temperature_k"]
    error = abs(got - target) / target
    check("final_temperature", got, round(target, 4), error < tolerance,
          f"{100.0 * error:+.4g}% against complete combustion of the case's own "
          "mixture by enthalpy conservation")

    # The same number against the composition the solver itself ended on.
    final_dir = latest_time_dir(case)
    measured: dict[str, float] = {}
    if final_dir is not None:
        for name in species:
            value = read_uniform_field(final_dir / name)
            if value is not None:
                measured[name] = max(value, 0.0)
    if measured and abs(sum(measured.values()) - 1.0) < 1e-3:
        normalised = {name: y / sum(measured.values()) for name, y in measured.items()}
        at_own = solve_temperature(predicted["conserved_j_kg"], normalised, species,
                                   initial["constant_volume"])
        own_error = abs(got - at_own) / at_own
        check("final_temperature_at_own_composition", got, round(at_own, 4),
              own_error < tolerance,
              f"{100.0 * own_error:+.4g}%; thermodynamics alone, at the Y fields in "
              f"{final_dir.name}/")
        # The *limiting* reactant, which is the one complete combustion takes to zero.
        # Checking "the fuel" instead would be wrong on a lean mixture, where oxygen
        # legitimately remains, and checking every reactant would fail it.
        reactants, _ = read_reaction(case)
        spent = [name for name in reactants
                 if name not in predicted["final"]["moles"]]
        left = max((normalised.get(name, 0.0) for name in spent), default=0.0)
        check("limiting_reactant_remaining", left, "< 1e-3", left < 1.0e-3,
              f"mass fraction of {', '.join(spent) or 'nothing'}, which complete "
              "combustion consumes entirely. A one-step irreversible reaction "
              "approaches that as a power law rather than exponentially, so the tail "
              "is what decides whether the complete-combustion target applies at all")
    else:
        check("final_temperature_at_own_composition", None, None, None,
              f"reported, not graded: no readable mass-fraction fields in "
              f"{final_dir.name + '/' if final_dir else 'any time directory'}, so the "
              "thermodynamics cannot be separated from the burn-out")

    # The run has to have gone somewhere: a reactor that never ignited reaches its
    # initial temperature to any tolerance you like and agrees with nothing.
    rise = got - initial["temperature"]
    expected_rise = target - initial["temperature"]
    check("temperature_rise", round(rise, 4), round(expected_rise, 4),
          rise > 0.5 * expected_rise,
          "against half the predicted rise; this is the check a reactor that never "
          "ignited fails, and it fails nothing else")

    if initial["constant_volume"]:
        p_target = predicted["final"]["pressure_pa"]
        p_error = abs(run["pressure_pa"] - p_target) / p_target
        check("final_pressure", run["pressure_pa"], round(p_target, 4),
              p_error < tolerance,
              f"{100.0 * p_error:+.4g}%; rho0*R_mix(end)*T_end at the unchanged "
              "density, which is independent of the temperature check only in that it "
              "carries the mixture's molecular weight as well")
    else:
        p_error = abs(run["pressure_pa"] - initial["pressure"]) / initial["pressure"]
        check("pressure_held", run["pressure_pa"], initial["pressure"],
              p_error < 1e-6,
              "constantProperty pressure, so the solver holds it and this says so")

    graded = [item for item in checks if item["passed"] is not None]
    return {
        "case": str(case),
        "prediction": predicted,
        "run": {key: value for key, value in run.items() if key != "series"},
        "measured_mass_fractions": measured,
        "checks": checks,
        "passed": all(item["passed"] for item in graded),
    }


def print_grade(result: dict[str, Any]) -> None:
    run = result["run"]
    print(f"chem0d grade: {result['case']}")
    print(f"  chemFoam.out: {run['rows']} rows, last at t = {run['time_s']:g} s")
    for item in result["checks"]:
        mark = {True: "PASS", False: "FAIL", None: "----"}[item["passed"]]
        value = item["value"]
        reference = item["reference"]
        line = f"  {mark}  {item['name']}: "
        line += "-" if value is None else (
            f"{value:.6g}" if isinstance(value, float) else str(value))
        if reference is not None:
            line += f" (reference {reference:.6g}" if isinstance(reference, float) \
                else f" (reference {reference}"
            line += ")"
        if item["note"]:
            line += f"  -- {item['note']}"
        print(line)
    if result["measured_mass_fractions"]:
        print("  the run's own final mass fractions: " + "  ".join(
            f"{name} {value:.6g}"
            for name, value in sorted(result["measured_mass_fractions"].items())))
    print("PASSED" if result["passed"] else "FAILED")


# -- write ---------------------------------------------------------------------


def foam_header(cls: str, obj: str) -> str:
    return ("FoamFile\n"
            "{\n"
            "    version     2.0;\n"
            "    format      ascii;\n"
            f"    class       {cls};\n"
            f"    object      {obj};\n"
            "}\n")


def tutorial_dir() -> Path:
    root = os.environ.get("FOAM_TUTORIALS")
    if not root:
        raise Chem0dError(
            "$FOAM_TUTORIALS is not set, so the distribution's own mechanism and "
            "thermo data cannot be found. They are deliberately not committed to this "
            "repository: see TUTORIAL_SOURCE in this file. On the workspace image the "
            "variable is set by the OpenFOAM environment.")
    return Path(root) / TUTORIAL_SOURCE


def copy_mechanism(case: Path) -> dict[str, str]:
    """`constant/reactions` and `constant/thermo`, copied out of the distribution."""
    source = tutorial_dir()
    wanted = {"reactions": "reactions", "thermo": "thermo.compressibleGas"}
    copied = {}
    (case / "constant").mkdir(parents=True, exist_ok=True)
    for target, name in wanted.items():
        path = source / name
        if not path.is_file():
            raise Chem0dError(
                f"{path} is not there. This is a stock OpenFOAM tutorial file; if it "
                "has moved, `ls $FOAM_TUTORIALS/combustion/reactingFoam/laminar` will "
                "say where to")
        shutil.copyfile(path, case / "constant" / target)
        copied[target] = str(path)
    return copied


def initial_conditions(moles: dict[str, float], temperature: float, pressure: float,
                       constant_volume: bool, provenance: str) -> str:
    """Mole fractions, not mass: `readInitialConditions.H` normalises either, and a
    mixture stated in moles is the one a stoichiometry is stated in."""
    lines = [foam_header("dictionary", "initialConditions"), "",
             f"// {provenance}", "",
             f"constantProperty {'volume' if constant_volume else 'pressure'};", "",
             "fractionBasis   mole;", "",
             "fractions", "{"]
    for name, amount in moles.items():
        lines.append(f"    {name:<16}{amount:g};")
    lines += ["}", "",
              f"p               {pressure:g};", "",
              f"T               {temperature:g};", ""]
    return "\n".join(lines)


def thermophysical_properties() -> str:
    """`reactingMixture` with the native reader, not the CHEMKIN one.

    Every `chemFoam` tutorial in the distribution runs `chemkinToFoam` first and reads
    the result; this skips that stage by taking a mechanism that is already in
    OpenFOAM's own format, which is one fewer parser between the data and the solver.
    `energy sensibleEnthalpy` is what the derivation in the module docstring assumes --
    `sensibleInternalEnergy` would make `h0` an internal energy and shift the whole
    balance.
    """
    return (foam_header("dictionary", "thermophysicalProperties") + "\n"
            "thermoType\n"
            "{\n"
            "    type            hePsiThermo;\n"
            "    mixture         reactingMixture;\n"
            "    transport       sutherland;\n"
            "    thermo          janaf;\n"
            "    energy          sensibleEnthalpy;\n"
            "    equationOfState perfectGas;\n"
            "    specie          specie;\n"
            "}\n\n"
            "chemistryReader foamChemistryReader;\n\n"
            'foamChemistryFile "<constant>/reactions";\n\n'
            'foamChemistryThermoFile "<constant>/thermo";\n')


def chemistry_properties() -> str:
    """`seulex`, which the distribution's own README says is the best of its ODE solvers.

    Quoting `$FOAM_TUTORIALS/combustion/chemFoam/README`: "Overall the best performing
    ODE solver is seulex followed closely by rodas23." `absTol 1e-12` is tight because
    the quantity being graded is a temperature derived from mass fractions, and a
    species integrated to 1e-6 is a 1e-6 error in the enthalpy balance.
    """
    return (foam_header("dictionary", "chemistryProperties") + "\n"
            "chemistryType\n"
            "{\n"
            "    solver          ode;\n"
            "}\n\n"
            "chemistry       on;\n\n"
            "initialChemicalTimeStep 1e-07;\n\n"
            "odeCoeffs\n"
            "{\n"
            "    solver          seulex;\n"
            "    absTol          1e-12;\n"
            "    relTol          1e-05;\n"
            "}\n")


def control_dict(end_time: float, max_delta_t: float) -> str:
    """`adjustTimeStep on` is not optional here: `setDeltaT.H` sets the step from the
    chemistry's own `deltaTChem`, which spans six orders between the induction period
    and the burn. A fixed step either costs a hundred thousand of them or steps over
    the whole event.

    `writeControl adjustable` with `purgeWrite 0` so the final time directory survives
    -- `grade` reads the mass-fraction fields out of it, and `purgeWrite 1` would leave
    only the last and still work, while 0 leaves the series a plot can use.
    """
    return (foam_header("dictionary", "controlDict") + "\n"
            "application     chemFoam;\n\n"
            "startFrom       startTime;\n\n"
            "startTime       0;\n\n"
            "stopAt          endTime;\n\n"
            f"endTime         {end_time:g};\n\n"
            "deltaT          1e-07;\n\n"
            f"maxDeltaT       {max_delta_t:g};\n\n"
            "adjustTimeStep  on;\n\n"
            "writeControl    adjustable;\n\n"
            f"writeInterval   {end_time / 10.0:g};\n\n"
            "purgeWrite      0;\n\n"
            "writeFormat     ascii;\n\n"
            "writePrecision  12;\n\n"
            "writeCompression off;\n\n"
            "timeFormat      general;\n\n"
            "timePrecision   6;\n\n"
            "runTimeModifiable yes;\n")


def fv_schemes() -> str:
    """Three empty dictionaries and a `ddtSchemes`, which is not laziness: there is one
    cell, so there is no gradient, no divergence and no laplacian anywhere in this
    solver. `chemFoam`'s own tutorials ship exactly this."""
    return (foam_header("dictionary", "fvSchemes") + "\n"
            "ddtSchemes\n"
            "{\n"
            "    default         Euler;\n"
            "}\n\n"
            "gradSchemes\n"
            "{}\n\n"
            "divSchemes\n"
            "{}\n\n"
            "laplacianSchemes\n"
            "{}\n")


def fv_solution() -> str:
    """One entry, `Yi`, which is the name `YEqn.H` solves every species under."""
    return (foam_header("dictionary", "fvSolution") + "\n"
            "solvers\n"
            "{\n"
            "    Yi\n"
            "    {\n"
            "        solver          PBiCGStab;\n"
            "        preconditioner  DILU;\n"
            "        tolerance       1e-12;\n"
            "        relTol          0;\n"
            "    }\n"
            "}\n")


def write_case(case: Path, phi: float, temperature: float, pressure: float,
               constant_volume: bool, end_time: float,
               max_delta_t: float) -> dict[str, Any]:
    """The whole case, with the mechanism copied in rather than written out.

    There is no `0/` directory, and that is right rather than missing: `chemFoam`'s
    `createBaseFields.H` writes `Ydefault`, `p` and `T` into the start time itself
    before the thermo package is constructed, and `readInitialConditions.H` then sets
    every mass fraction from `constant/initialConditions`. Every tutorial in the
    distribution ships the same two directories and no field files.
    """
    if phi > 1.0:
        raise Chem0dError(
            f"--phi {phi:g} is rich. The copied mechanism is one irreversible step with "
            "no CO and no H2 in its species list, so it burns a rich mixture to "
            "leftover methane, which is not a thing that happens. Lean and "
            "stoichiometric are what this case can honestly be")
    if phi <= 0.0:
        raise Chem0dError(f"--phi {phi:g} has no fuel in it")

    copied = copy_mechanism(case)
    moles = {FUEL: phi, OXIDISER: 1.0, DILUENT: NITROGEN_PER_OXYGEN}
    # phi is defined on the stoichiometric fuel/oxygen ratio, which comes out of the
    # copied mechanism rather than being assumed: `CH4 + 2O2 = ...` means 1 mole of fuel
    # per 2 of oxygen, so one mole of O2 carries phi/2 of fuel.
    reactants, _ = read_reaction(case)
    if FUEL not in reactants or OXIDISER not in reactants:
        raise Chem0dError(
            f"the copied mechanism's reactants are {sorted(reactants)}; this writes a "
            f"{FUEL}/{OXIDISER} mixture and cannot state an equivalence ratio for "
            "another one")
    moles[FUEL] = phi * reactants[FUEL] / reactants[OXIDISER]

    provenance = (f"equivalence ratio {phi:g} on {reactants[FUEL]:g} {FUEL} per "
                  f"{reactants[OXIDISER]:g} {OXIDISER}, with "
                  f"{NITROGEN_PER_OXYGEN:g} {DILUENT} per {OXIDISER} for air")
    files = {
        "constant/initialConditions": initial_conditions(
            moles, temperature, pressure, constant_volume, provenance),
        "constant/thermophysicalProperties": thermophysical_properties(),
        "constant/chemistryProperties": chemistry_properties(),
        "system/controlDict": control_dict(end_time, max_delta_t),
        "system/fvSchemes": fv_schemes(),
        "system/fvSolution": fv_solution(),
    }
    for rel, text in files.items():
        path = case / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return {"case": str(case), "copied": copied,
            "written": sorted(files) + sorted(f"constant/{name}" for name in copied),
            "moles": moles}


# -- command line --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)

    write = sub.add_parser("write", help="a chemFoam case in --case")
    write.add_argument("--case", type=Path, default=Path.cwd())
    write.add_argument("--phi", type=float, default=1.0,
                       help="equivalence ratio; 1 is stoichiometric, rich is refused")
    write.add_argument("--t0", type=float, default=DEFAULT_T0_K, dest="t0",
                       help="initial temperature, K (a one-step mechanism cannot "
                            "ignite from cold)")
    write.add_argument("--pressure", type=float, default=DEFAULT_PRESSURE_PA)
    write.add_argument("--constant", choices=("pressure", "volume"),
                       default="pressure")
    write.add_argument("--end-time", type=float, default=DEFAULT_END_TIME_S,
                       dest="end_time")
    write.add_argument("--max-delta-t", type=float, default=DEFAULT_MAX_DELTA_T_S,
                      dest="max_delta_t")
    write.add_argument("--json", action="store_true")

    for name, helptext in (("predict", "the target temperature from the case's data"),
                           ("grade", "the run against the target")):
        action = sub.add_parser(name, help=helptext)
        action.add_argument("--case", type=Path, default=Path.cwd())
        action.add_argument("--json", action="store_true")
        if name == "grade":
            action.add_argument("--tolerance", type=float, default=5.0e-3)

    args = parser.parse_args(argv)
    try:
        if args.action == "write":
            result = write_case(args.case, args.phi, args.t0, args.pressure,
                                args.constant == "volume", args.end_time,
                                args.max_delta_t)
            prediction = predict(args.case)
            if args.json:
                print(json.dumps({"write": result, "predict": prediction}, indent=2))
            else:
                for rel in result["written"]:
                    print(f"  wrote {rel}")
                for target, source in result["copied"].items():
                    print(f"  constant/{target} copied from {source}")
                print()
                print_prediction(prediction)
                print()
                print("the case is complete and meant to be solved. one way to run it,"
                      "\nand the grading that follows:\n"
                      "    chemFoam > log.chemFoam 2>&1\n"
                      f"    python3 {Path(__file__).name} grade --case "
                      f"{args.case}\n"
                      "it is one cell, so it costs seconds rather than a cluster.")
            return 0
        if args.action == "predict":
            result = predict(args.case)
            print(json.dumps(result, indent=2)) if args.json \
                else print_prediction(result)
            return 0
        result = grade(args.case, args.tolerance)
        print(json.dumps(result, indent=2)) if args.json else print_grade(result)
        return 0 if result["passed"] else 1
    except Chem0dError as error:
        print(f"chem0d: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
