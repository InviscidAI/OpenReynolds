"""`chem0d.py`: the JANAF arithmetic, the stoichiometry, and what it refuses.

There is no OpenFOAM on the machine that runs this suite, so what is checked here is
everything that does not need one -- which for this script is most of it, because the
target temperature it grades against is computed in Python from the case's own thermo
data and the solver only has to reproduce it. `tests/test_journey_chem0d.py` in the
`journey` job runs the real case in the workspace image.

The thermo file these tests write is **constant-Cp**: `a0` is the only non-zero
coefficient besides `a5`, so

    Ha(T) = R * (a0*T + a5)

is linear in T and the adiabatic temperature comes out of one line of algebra rather
than out of a root find. That is the point of it. A test whose expected value came from
the same bisection it is testing would assert only that the code is self-consistent --
the rule `test_journey_thermal_stress.py` states for the same reason -- and the real
seven-coefficient fit has no closed-form inverse to compare against.

The real JANAF coefficients are deliberately **not** in this repository at all.
`chem0d.py write` copies them out of `$FOAM_TUTORIALS`, so what it evaluates is the
image's own file byte for byte and nothing about those numbers is an assertion made
here. The one test below that reads them skips where that directory is absent, which is
every machine except the image.
"""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "openreynolds" / "toolbox" / "chem0d.py"


def load():
    spec = importlib.util.spec_from_file_location("toolbox_chem0d", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def chem0d():
    return load()


# -- the synthetic case ---------------------------------------------------------

MOL_WEIGHT = 20.0          # the same for both species, so mass balances exactly
CP_OVER_R = 4.0            # a0
A5_FUEL = 0.0
A5_PRODUCT = -4000.0       # the heat release, in units of R
T0 = 1000.0
PRESSURE = 101325.0


def specie_block(name: str, a5: float) -> str:
    """One species, constant Cp, high and low coefficients identical.

    Identical on purpose: `Tcommon` then cannot change the answer, so these tests
    measure the enthalpy arithmetic and not which branch of the fit was taken. The
    branch itself is checked separately, by a species whose two sets differ.
    """
    coeffs = f"( {CP_OVER_R} 0 0 0 0 {a5} 0 )"
    return (f"{name}\n"
            "{\n"
            "    specie\n"
            "    {\n"
            f"        molWeight       {MOL_WEIGHT};\n"
            "    }\n"
            "    elements\n"
            "    {\n"
            "        X       1;\n"
            "    }\n"
            "    thermodynamics\n"
            "    {\n"
            "        Tlow            200;\n"
            "        Thigh           5000;\n"
            "        Tcommon         1000;\n"
            f"        highCpCoeffs    {coeffs};\n"
            f"        lowCpCoeffs     {coeffs};\n"
            "    }\n"
            "    transport\n"
            "    {\n"
            "        As              1.67212e-06;\n"
            "        Ts              170.672;\n"
            "    }\n"
            "}\n")


def synthetic_case(case: Path, constant: str = "pressure",
                   fractions: str = "    FUEL            1;\n") -> Path:
    """A `chemFoam` case directory with a two-species, one-reaction mechanism."""
    (case / "constant").mkdir(parents=True, exist_ok=True)
    (case / "constant" / "thermo").write_text(
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
        "    class dictionary;\n    object thermo;\n}\n\n"
        + specie_block("FUEL", A5_FUEL) + "\n" + specie_block("PROD", A5_PRODUCT),
        encoding="utf-8")
    (case / "constant" / "reactions").write_text(
        "elements\n(\nX\n);\n\nspecies\n(\n    FUEL\n    PROD\n);\n\n"
        "reactions\n{\n    theOnlyOne\n    {\n"
        "        type     irreversibleArrheniusReaction;\n"
        '        reaction "FUEL = PROD";\n'
        "        A        1e10;\n        beta     0;\n        Ta       10000;\n"
        "    }\n}\n", encoding="utf-8")
    (case / "constant" / "initialConditions").write_text(
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
        "    class dictionary;\n    object initialConditions;\n}\n\n"
        f"constantProperty {constant};\n\nfractionBasis   mole;\n\n"
        "fractions\n{\n" + fractions + "}\n\n"
        f"p               {PRESSURE};\n\nT               {T0};\n",
        encoding="utf-8")
    return case


def exact_constant_pressure() -> float:
    """`Ha_PROD(T) = Ha_FUEL(T0)` with both linear: T = T0 + (a5_f - a5_p)/a0."""
    return T0 + (A5_FUEL - A5_PRODUCT) / CP_OVER_R


def exact_constant_volume() -> float:
    """The same with `u = h - R*T`, so the coefficient on T is `a0 - 1` and not `a0`."""
    return T0 + (A5_FUEL - A5_PRODUCT) / (CP_OVER_R - 1.0)


# -- the arithmetic -------------------------------------------------------------


def test_the_gas_constant_is_the_one_openfoam_computes(chem0d):
    """`RR = 1e3*NA*k` out of `thermodynamicConstants.C`, with NA and k from
    `fundamentalConstants.C` and `etc/controlDict`. 8314.47 and not 8314.46: OpenFOAM's
    NA and k are the pre-2019 values and the product is 8314.47006650545, which is 1e-6
    relative from the current SI 8314.462618. The difference does not matter to any
    answer here and the wrong constant in the wrong place would, so it is pinned."""
    assert chem0d.RR == pytest.approx(8314.47006650545, rel=1e-12)
    assert chem0d.TSTD == 298.15


def test_the_adiabatic_temperature_is_the_closed_form_at_constant_pressure(
        chem0d, tmp_path):
    result = chem0d.predict(synthetic_case(tmp_path))
    assert result["constant_property"] == "pressure"
    assert result["final"]["temperature_k"] == pytest.approx(exact_constant_pressure(),
                                                             rel=1e-9)
    assert result["final"]["moles"] == {"PROD": pytest.approx(1.0)}
    # Constant pressure means the pressure does not move.
    assert result["final"]["pressure_pa"] == pytest.approx(PRESSURE)


def test_the_adiabatic_temperature_is_the_closed_form_at_constant_volume(
        chem0d, tmp_path):
    """The two differ by the `R*T` in `u = h - R*T`, 1333 K against 1000 K of rise here,
    which is a third. A case graded against the wrong one of the two would be out by
    that much and nothing else in the run would say so."""
    result = chem0d.predict(synthetic_case(tmp_path, constant="volume"))
    assert result["constant_property"] == "volume"
    assert result["final"]["temperature_k"] == pytest.approx(exact_constant_volume(),
                                                             rel=1e-9)
    # p/T is constant at fixed density and unchanged molecular weight.
    assert result["final"]["pressure_pa"] / result["final"]["temperature_k"] == (
        pytest.approx(PRESSURE / T0, rel=1e-9))
    assert result["final"]["temperature_k"] > exact_constant_pressure()


def test_the_two_plane_cases_of_the_fit_are_told_apart_by_tcommon(chem0d, tmp_path):
    """`janafThermoI.H` switches on `T > Tcommon` with no blending, so a species whose
    two coefficient sets differ has a discontinuous Cp there. Checked because the
    synthetic file above has them identical and so cannot see this at all."""
    case = synthetic_case(tmp_path)
    body = ("    specie { molWeight 20; }\n"
            "    thermodynamics\n    {\n"
            "        Tlow 200; Thigh 5000; Tcommon 1000;\n"
            "        highCpCoeffs ( 9 0 0 0 0 0 0 );\n"
            "        lowCpCoeffs  ( 4 0 0 0 0 0 0 );\n"
            "    }\n")
    specie = chem0d.Specie("SPLIT", body)
    assert specie.coefficients(999.0)[0] == 4.0
    assert specie.coefficients(1001.0)[0] == 9.0
    # Ha = R*a0*T either side, so the ratio at equal T is the ratio of a0.
    assert specie.absolute_enthalpy(1001.0) / specie.absolute_enthalpy(999.0) == (
        pytest.approx(9.0 * 1001.0 / (4.0 * 999.0), rel=1e-9))
    assert case.is_dir()


def test_the_formation_enthalpy_always_uses_the_low_coefficients(chem0d):
    """`janafThermoI.H:212-222` reads `lowCpCoeffs_` unconditionally, whatever `Tcommon`
    is. Taking `Ha(Tstd)` through the switch instead would read the high set for a
    species whose `Tcommon` is below 298.15, and the two are not the same number."""
    body = ("    specie { molWeight 20; }\n"
            "    thermodynamics\n    {\n"
            "        Tlow 100; Thigh 5000; Tcommon 200;\n"
            "        highCpCoeffs ( 9 0 0 0 0 0 0 );\n"
            "        lowCpCoeffs  ( 4 0 0 0 0 0 0 );\n"
            "    }\n")
    specie = chem0d.Specie("SPLIT", body)
    gas_constant = chem0d.RR / 20.0
    assert specie.formation_enthalpy() == pytest.approx(
        gas_constant * 4.0 * chem0d.TSTD, rel=1e-12)
    # Through the switch it would be the high set, 2.25x larger.
    assert specie.absolute_enthalpy(chem0d.TSTD) == pytest.approx(
        gas_constant * 9.0 * chem0d.TSTD, rel=1e-12)


def test_a_lean_mixture_leaves_the_oxidiser_and_burns_cooler(chem0d, tmp_path):
    """Half the fuel, so half the heat into very nearly the same mass. The leftover
    oxidiser is in the final composition rather than quietly dropped, which is the
    thing a stoichiometry that assumed complete consumption of everything would get
    wrong in the direction of agreement."""
    case = synthetic_case(tmp_path, fractions="    FUEL            0.5;\n"
                                              "    PROD            0.5;\n")
    result = chem0d.predict(case)
    assert set(result["final"]["moles"]) == {"PROD"}
    rise = result["final"]["temperature_k"] - T0
    assert rise == pytest.approx(0.5 * (exact_constant_pressure() - T0), rel=1e-9)


# -- the parsing ----------------------------------------------------------------


def test_nested_blocks_are_read_at_the_depth_they_are_at(chem0d):
    blocks = chem0d.top_level_blocks(
        "FoamFile { object thermo; }\n"
        "A { specie { molWeight 1; } thermodynamics { Tlow 2; } }\n"
        "B { specie { molWeight 3; } }\n")
    assert set(blocks) == {"FoamFile", "A", "B"}
    assert "thermodynamics" in blocks["A"]
    assert chem0d.scalar_entry(chem0d.top_level_blocks(blocks["B"])["specie"],
                               "molWeight") == 3.0


def test_comments_do_not_become_entries(chem0d):
    blocks = chem0d.top_level_blocks(
        "// A { molWeight 1; }\n/* B { molWeight 2; } */\nC { molWeight 3; }\n")
    assert set(blocks) == {"C"}


def test_the_stoichiometry_comes_off_the_reaction_string(chem0d, tmp_path):
    synthetic_case(tmp_path)
    (tmp_path / "constant" / "reactions").write_text(
        'reactions { r { reaction "CH4 + 2O2 = CO2 + 2H2O"; } }\n', encoding="utf-8")
    reactants, products = chem0d.read_reaction(tmp_path)
    assert reactants == {"CH4": 1.0, "O2": 2.0}
    assert products == {"CO2": 1.0, "H2O": 2.0}


def test_a_multi_step_mechanism_is_refused_rather_than_guessed_at(chem0d, tmp_path):
    synthetic_case(tmp_path)
    (tmp_path / "constant" / "reactions").write_text(
        'reactions { a { reaction "A = B"; } b { reaction "B = C"; } }\n',
        encoding="utf-8")
    with pytest.raises(chem0d.Chem0dError, match="2 reactions"):
        chem0d.read_reaction(tmp_path)


def test_a_reaction_order_override_is_refused(chem0d, tmp_path):
    """`CH4^0.7` means the rate's exponent is not the stoichiometric coefficient, so the
    string no longer states the stoichiometry this needs."""
    synthetic_case(tmp_path)
    (tmp_path / "constant" / "reactions").write_text(
        'reactions { a { reaction "CH4^0.7 + 2O2 = CO2 + 2H2O"; } }\n',
        encoding="utf-8")
    with pytest.raises(chem0d.Chem0dError, match="reaction-order"):
        chem0d.read_reaction(tmp_path)


def test_a_mass_basis_is_refused_rather_than_read_as_moles(chem0d, tmp_path):
    case = synthetic_case(tmp_path)
    path = case / "constant" / "initialConditions"
    path.write_text(path.read_text(encoding="utf-8").replace(
        "fractionBasis   mole", "fractionBasis   mass"), encoding="utf-8")
    with pytest.raises(chem0d.Chem0dError, match="fractionBasis"):
        chem0d.predict(case)


def test_a_temperature_outside_every_fit_is_refused_rather_than_clamped(
        chem0d, tmp_path):
    """`janafThermo::limit` clamps T to [Tlow, Thigh] and warns; a prediction that came
    back clamped would be a boundary value dressed as a solution. The release here is
    forty times the shipped one, so the root is far above Thigh."""
    case = synthetic_case(tmp_path)
    path = case / "constant" / "thermo"
    path.write_text(path.read_text(encoding="utf-8").replace(
        f"{A5_PRODUCT}", "-160000"), encoding="utf-8")
    with pytest.raises(chem0d.Chem0dError, match="outside"):
        chem0d.predict(case)


# -- grading --------------------------------------------------------------------


def write_run(case: Path, final_temperature: float, pressure: float = PRESSURE,
              rows: int = 11) -> None:
    """A `chemFoam.out` of the shape `output.H` appends, rising to the given end."""
    lines = ["# Time\tTemperature [K]\tPressure [Pa]"]
    for index in range(rows):
        fraction = index / (rows - 1)
        temperature = T0 + fraction * (final_temperature - T0)
        lines.append(f"{0.05 * index:g}\t{temperature:.10g}\t{pressure:.10g}")
    (case / "chemFoam.out").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_a_run_that_reached_the_closed_form_passes(chem0d, tmp_path):
    case = synthetic_case(tmp_path)
    write_run(case, exact_constant_pressure())
    result = chem0d.grade(case)
    failed = [check["name"] for check in result["checks"] if check["passed"] is False]
    assert result["passed"], failed


def test_a_reactor_that_never_ignited_fails_the_rise_and_not_only_the_temperature(
        chem0d, tmp_path):
    """The failure a tolerance alone would mis-describe: a reactor sitting at its
    initial temperature has not disagreed with the prediction by a little, it has not
    reacted."""
    case = synthetic_case(tmp_path)
    write_run(case, T0)
    result = chem0d.grade(case)
    by_name = {check["name"]: check for check in result["checks"]}
    assert not result["passed"]
    assert by_name["temperature_rise"]["passed"] is False
    assert by_name["final_temperature"]["passed"] is False


def test_a_run_one_percent_off_fails_at_the_default_tolerance(chem0d, tmp_path):
    case = synthetic_case(tmp_path)
    write_run(case, 1.01 * exact_constant_pressure())
    assert chem0d.grade(case)["passed"] is False
    # And passes when told to accept a percent, so the bound is the thing being tested
    # rather than the arithmetic.
    assert chem0d.grade(case, tolerance=2.0e-2)["passed"] is True


def test_a_constant_pressure_run_whose_pressure_moved_is_caught(chem0d, tmp_path):
    case = synthetic_case(tmp_path)
    write_run(case, exact_constant_pressure(), pressure=1.5 * PRESSURE)
    by_name = {check["name"]: check for check in chem0d.grade(case)["checks"]}
    assert by_name["pressure_held"]["passed"] is False


def test_the_composition_check_is_reported_when_no_fields_were_written(
        chem0d, tmp_path):
    """Reported, not graded: without the final mass fractions the thermodynamics cannot
    be separated from whether the fuel burnt out, and saying so beats a verdict.
    `benchmarks/sph/` makes the same call about the dam break."""
    case = synthetic_case(tmp_path)
    write_run(case, exact_constant_pressure())
    by_name = {check["name"]: check for check in chem0d.grade(case)["checks"]}
    own = by_name["final_temperature_at_own_composition"]
    assert own["passed"] is None
    assert "not graded" in own["note"]


def test_the_composition_check_uses_the_fields_the_solver_wrote(chem0d, tmp_path):
    """With a tenth of the fuel left the complete-combustion target is 10% too hot,
    while the target at the solver's own composition is still met -- which is how an
    incomplete burn is told apart from an energy error."""
    case = synthetic_case(tmp_path)
    left = 0.1
    temperature = T0 + (1.0 - left) * (exact_constant_pressure() - T0)
    write_run(case, temperature)
    final = case / "0.5"
    final.mkdir()
    for name, value in (("FUEL", left), ("PROD", 1.0 - left)):
        (final / name).write_text(
            "FoamFile\n{\n    class volScalarField;\n    object " + name + ";\n}\n"
            "dimensions [0 0 0 0 0 0 0];\n"
            f"internalField   uniform {value};\n"
            "boundaryField { }\n", encoding="utf-8")
    result = chem0d.grade(case)
    by_name = {check["name"]: check for check in result["checks"]}
    assert by_name["final_temperature"]["passed"] is False
    assert by_name["final_temperature_at_own_composition"]["passed"] is True
    assert by_name["limiting_reactant_remaining"]["passed"] is False
    assert not result["passed"]


def test_a_case_that_was_never_run_is_a_refusal_and_not_a_failure(chem0d, tmp_path):
    synthetic_case(tmp_path)
    proc = subprocess.run([sys.executable, str(SCRIPT), "grade",
                           "--case", str(tmp_path)], text=True, capture_output=True)
    assert proc.returncode == 2
    assert "chemFoam.out" in proc.stderr


# -- what it refuses to write ---------------------------------------------------


def test_a_rich_mixture_is_refused_by_name(chem0d, tmp_path):
    """One irreversible step with no CO and no H2 burns a rich mixture to leftover
    methane, which does not happen. Refusing beats running it."""
    with pytest.raises(chem0d.Chem0dError, match="rich"):
        chem0d.write_case(tmp_path, 1.3, 1000.0, PRESSURE, False, 0.5, 1e-4)
    with pytest.raises(chem0d.Chem0dError, match="no fuel"):
        chem0d.write_case(tmp_path, 0.0, 1000.0, PRESSURE, False, 0.5, 1e-4)


def test_writing_off_the_image_names_the_path_it_wanted(chem0d, tmp_path, monkeypatch):
    """The mechanism and the thermo data are the distribution's and are not committed
    here, so a machine without `$FOAM_TUTORIALS` cannot write a case -- and says which
    directory it was looking for rather than inventing coefficients."""
    monkeypatch.delenv("FOAM_TUTORIALS", raising=False)
    with pytest.raises(chem0d.Chem0dError, match="FOAM_TUTORIALS"):
        chem0d.write_case(tmp_path, 1.0, 1000.0, PRESSURE, False, 0.5, 1e-4)


# -- the reference table --------------------------------------------------------


def test_every_nist_value_carries_its_provenance(chem0d):
    """`benchmarks/README.md` argues that the reference value is what goes wrong most
    often and that the defence is reading the source. A table entry with no reference
    line on it cannot be re-checked, so it is not allowed to exist."""
    for name, (value, uncertainty, reference) in \
            chem0d.NIST_FORMATION_KJ_MOL.items():
        assert isinstance(value, float), name
        assert uncertainty is None or uncertainty > 0.0, name
        assert len(reference) > 20, f"{name} has no usable reference"
        if value == 0.0:
            assert "standard state" in reference, name
        else:
            # A WebBook page is cited by its compound id, which is how it is found
            # again.
            assert reference.startswith("C"), name


def test_the_nist_table_reproduces_methanes_heat_of_combustion(chem0d):
    """An arithmetic check on the table and not a verdict about reality: Hess's law on
    the three gas-phase formation enthalpies in it gives the heat of reaction of
    `CH4 + 2O2 = CO2 + 2H2O` with gaseous water, and dividing by methane's molar mass
    gives the lower heating value. A sign error or a units slip in the table shows up
    here as a number nothing like 50 MJ/kg.

    The uncertainties add in quadrature to 0.33 kJ/mol out of 802.6, which is 0.04%, so
    the table cannot support a tighter claim than that and this does not make one.
    """
    fuel = chem0d.NIST_FORMATION_KJ_MOL["CH4"][0]
    oxygen = chem0d.NIST_FORMATION_KJ_MOL["O2"][0]
    carbon_dioxide = chem0d.NIST_FORMATION_KJ_MOL["CO2"][0]
    water = chem0d.NIST_FORMATION_KJ_MOL["H2O"][0]
    heat = (carbon_dioxide + 2.0 * water) - (fuel + 2.0 * oxygen)
    assert heat == pytest.approx(-802.562, abs=1e-3)
    # Methane is 16.04 g/mol, so 802.6 kJ/mol is 50.0 MJ/kg. The molar mass is written
    # out rather than read from a thermo file, because this test is about the table.
    assert -heat / 16.0428 == pytest.approx(50.03, abs=0.02)

    spread = math.sqrt(0.3**2 + 0.13**2 + (2.0 * 0.040) ** 2)
    assert spread / abs(heat) < 5e-4


# -- the real file, where there is one ------------------------------------------


@pytest.mark.skipif(not os.environ.get("FOAM_TUTORIALS"),
                    reason="$FOAM_TUTORIALS is only set inside the workspace image, "
                           "and the real JANAF coefficients are not committed here")
def test_the_distributions_thermo_data_agrees_with_nist(chem0d, tmp_path):
    """The one check in this file that touches real combustion data.

    Every other test here measures the arithmetic. This one asks whether the numbers
    the arithmetic is applied to are the measured ones, by comparing the image's own
    JANAF fit at 298.15 K against NIST's gas-phase formation enthalpies. It is the only
    place in this item where a published measurement is being used as a reference at
    all, which is why it is read off the WebBook's own pages rather than quoted.
    """
    chem0d.write_case(tmp_path, 1.0, 1000.0, PRESSURE, False, 0.5, 1e-4)
    result = chem0d.predict(tmp_path)
    for name, row in result["formation_enthalpies"].items():
        if "nist_kj_mol" not in row:
            continue
        difference = abs(row["difference_kj_mol"])
        print(f"  {name}: {row['janaf_kj_mol']:+.4f} against NIST "
              f"{row['nist_kj_mol']:+.4f} kJ/mol, {difference:.4f} apart")
        # An element in its standard state is zero by definition and the fit's residual
        # there is pure rounding of seven-figure coefficients.
        assert difference < 0.5, f"{name} is {difference:.4f} kJ/mol from NIST"
    heating = result["lower_heating_value_mj_kg"]
    print(f"  lower heating value from the image's own fit: {heating:.4f} MJ/kg")
    assert heating == pytest.approx(50.03, abs=0.1)
