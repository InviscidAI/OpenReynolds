"""The combustion journey: chem0d.py -> chemFoam -> graded, in the image.

The fourth test in this repository that actually solves something. It exists because
`tests/test_toolbox_chem0d.py` checks the arithmetic against a constant-Cp mixture whose
answer is one line of algebra, and passes on a laptop with no OpenFOAM and no JANAF data
on it. What it cannot tell you is that `chemFoam` is in the image at all -- nothing in
this repository referenced any combustion solver before this item -- that the
`foamChemistryReader` route works without `chemkinToFoam` first, or that the real
seven-coefficient thermo data in the distribution is the measured data it is taken for.

the case, and the two things it is graded on

`chem0d.py write` as it ships: a single-cell well-stirred reactor, stoichiometric
methane in air, at 1000 K and 1 atm, held at constant pressure, with the mechanism and
the thermo data copied out of
`$FOAM_TUTORIALS/combustion/reactingFoam/laminar/counterFlowFlame2D/constant/` -- one
irreversible step, `CH4 + 2O2 = CO2 + 2H2O`, five species.

1. **The end temperature**, against the enthalpy-conserving value computed from the
   case's own JANAF coefficients. `chemFoam` conserves the mixture's absolute enthalpy
   at constant pressure (the derivation is in `chem0d.py`'s docstring, from the solver's
   own `hEqn.H` and `solveChemistry.H`), so the answer is one scalar equation in one
   unknown and no paper is involved. This is the thermal-stress move: the reference is
   the solver's own constitutive data re-evaluated independently.

2. **The thermo data itself**, against NIST. (1) is only as true as the coefficients in
   it, so `Ha_i(298.15 K)` -- which for a NASA polynomial fit is a standard enthalpy of
   formation -- is compared with NIST's Chemistry WebBook gas-phase values, read off the
   WebBook's own pages and carried with their published uncertainties in
   `chem0d.NIST_FORMATION_KJ_MOL`. The methane heating value that follows from them is
   checked too. That is the real-world measurement in this item, and it is the data
   rather than the temperature, because the temperature a single irreversible step
   produces has no dissociation in it and is **not** a flame temperature. The size of
   that gap is not asserted anywhere here: no primary source for it has been read, and
   `benchmarks/README.md` argues at length against quoting one that has not.

The expected values below are written out in this file rather than imported, with one
exception that is stated where it happens: the target temperature itself comes from
`chem0d.predict`, because reproducing a seven-coefficient JANAF root find here would be
a second implementation of the thing under test rather than an independent check. What
makes that safe is that the two tests above split the claim -- `test_toolbox_chem0d.py`
holds the arithmetic to a closed form, and this file holds the coefficients to NIST --
so neither rests on the other.

the control, which is the point

A second case is written with `chemistry off`, and nothing else changed. It has to come
back at its initial temperature and the grader has to fail it. Without that, "the final
temperature matched the prediction" is a claim about a number, not about combustion:
`benchmarks/lbm/` makes the same argument about its one-relaxation-time contrast.

Skipped where there is no OpenFOAM. The place this runs is the `journey` job in
`.github/workflows/tests.yml`, inside the workspace image.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
SCRIPT = ROOT / "openreynolds" / "toolbox" / "chem0d.py"


def _absent() -> list[str]:
    missing = [name for name in ("chemFoam",) if shutil.which(name) is None]
    if not os.environ.get("FOAM_TUTORIALS"):
        missing.append("$FOAM_TUTORIALS")
    return missing


pytestmark = pytest.mark.skipif(
    bool(_absent()),
    reason="not an OpenFOAM machine, so no case can be solved here: "
           + ", ".join(_absent() or ["-"]) + " missing",
)

T0_K = 1000.0
PRESSURE_PA = 101325.0
END_TIME_S = 0.5

# NIST Chemistry WebBook, gas-phase thermochemistry pages, read 2026-11-21. Written out
# here as well as in `chem0d.py` on purpose: this is the one reference in the item that
# comes from a measurement rather than from algebra, and a test that imported it from
# the code under test could not notice it being edited.
NIST_KJ_MOL = {"CH4": -74.6, "CO2": -393.51, "H2O": -241.826, "O2": 0.0, "N2": 0.0}
# The heat of reaction Hess's law gives from those three, and methane's 16.0428 g/mol.
NIST_HEAT_KJ_MOL = -802.562
NIST_LHV_MJ_KG = 50.026


def chem0d():
    spec = importlib.util.spec_from_file_location("toolbox_chem0d", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(cmd: list[str], cwd: Path, label: str = "") -> tuple[str, float]:
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    elapsed = time.monotonic() - started
    print(f"  {label or cmd[0]}: {elapsed:.1f} s, exit {proc.returncode}")
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:], file=sys.stderr)
        pytest.fail(f"{cmd} failed (exit {proc.returncode})")
    return proc.stdout, elapsed


def write_and_solve(case: Path, chemistry: bool = True) -> dict:
    """`chem0d.py write` as a person runs it, then the solver."""
    case.mkdir(parents=True)
    print(f"\n[{'reacting' if chemistry else 'chemistry off'}] {case}")
    write_log, _ = run([sys.executable, str(SCRIPT), "write", "--case", str(case),
                        "--t0", f"{T0_K:g}", "--pressure", f"{PRESSURE_PA:g}",
                        "--end-time", f"{END_TIME_S:g}"], case, "chem0d write")
    if not chemistry:
        path = case / "constant" / "chemistryProperties"
        text = path.read_text(encoding="utf-8")
        assert "chemistry       on;" in text, "the switch has been renamed"
        path.write_text(text.replace("chemistry       on;", "chemistry       off;"),
                        encoding="utf-8")
    solve_log, seconds = run(["chemFoam"], case, "chemFoam")
    steps = 0
    for line in solve_log.splitlines():
        if line.startswith("Number of steps"):
            steps = int(line.split("=")[1])
    print(f"  cost: 1 cell, {steps:,} steps, {seconds:.1f} s")
    return {"case": case, "write": write_log, "solve": solve_log,
            "seconds": seconds, "steps": steps}


@pytest.fixture(scope="module")
def reacting(tmp_path_factory) -> dict:
    return write_and_solve(tmp_path_factory.mktemp("chem") / "reacting")


@pytest.fixture(scope="module")
def frozen(tmp_path_factory) -> dict:
    return write_and_solve(tmp_path_factory.mktemp("chem") / "frozen", chemistry=False)


# -- the seams ---------------------------------------------------------------------


def test_the_mechanism_came_from_the_distribution_and_not_from_this_repository(
        reacting):
    """The whole provenance argument in one assertion: no JANAF coefficient and no rate
    constant in this item is a number anybody here typed. `constant/thermo` and
    `constant/reactions` are byte-for-byte copies of the distribution's own files, so
    what `chem0d.predict` evaluates is what the solver reads."""
    case = reacting["case"]
    source = Path(os.environ["FOAM_TUTORIALS"]) / chem0d().TUTORIAL_SOURCE
    for target, name in (("thermo", "thermo.compressibleGas"),
                         ("reactions", "reactions")):
        copied = (case / "constant" / target).read_bytes()
        original = (source / name).read_bytes()
        assert copied == original, f"constant/{target} is not the distribution's file"
    text = (case / "constant" / "reactions").read_text(encoding="utf-8")
    assert "CH4 + 2O2 = CO2 + 2H2O" in text
    assert "irreversibleArrheniusReaction" in text


def test_the_distributions_thermo_data_is_the_measured_data(reacting):
    """The real-world check in this item, and the only published reference in it.

    Each species' `Ha(298.15 K)` out of the image's own JANAF fit against NIST's
    gas-phase enthalpy of formation. The agreement bound is 0.5 kJ/mol rather than
    something tighter because NIST's own methane value spans -74.5 to -74.87 across four
    determinations, which is 0.37 of the 0.5: a tighter claim than the reference can
    support is not more accurate, it is less careful.
    """
    result = chem0d().predict(reacting["case"])
    for name, nist in NIST_KJ_MOL.items():
        row = result["formation_enthalpies"][name]
        difference = row["janaf_kj_mol"] - nist
        print(f"  {name:4s} JANAF {row['janaf_kj_mol']:+10.4f}  NIST {nist:+10.4f}  "
              f"diff {difference:+.4f} kJ/mol")
        assert abs(difference) < 0.5, f"{name} is {difference:+.4f} kJ/mol from NIST"
    heat = result["heat_of_reaction_kj_mol"]
    lhv = result["lower_heating_value_mj_kg"]
    print(f"  heat of reaction {heat:+.3f} kJ/mol against NIST's "
          f"{NIST_HEAT_KJ_MOL:+.3f} ({abs(heat - NIST_HEAT_KJ_MOL) / abs(NIST_HEAT_KJ_MOL):+.3%})")
    print(f"  lower heating value {lhv:.4f} MJ/kg against NIST's {NIST_LHV_MJ_KG:.4f}")
    assert heat == pytest.approx(NIST_HEAT_KJ_MOL, rel=2e-3)
    assert lhv == pytest.approx(NIST_LHV_MJ_KG, rel=2e-3)


def test_the_case_has_no_zero_directory_and_still_starts(reacting):
    """Not an omission: `createBaseFields.H` writes `Ydefault`, `p` and `T` into the
    start time itself before the thermo package is constructed, and
    `readInitialConditions.H` then sets every mass fraction from
    `constant/initialConditions`. Every chemFoam tutorial in the distribution ships
    `constant/` and `system/` and no field files, and a `0/` directory written by hand
    here would be overwritten."""
    case = reacting["case"]
    assert sorted(entry.name for entry in case.iterdir() if entry.is_dir()) != []
    assert (case / "constant" / "initialConditions").is_file()
    assert "Reading initial conditions" in reacting["solve"]
    assert "pressure will be held constant" in reacting["solve"]


def test_the_reactor_burnt_and_reached_the_closed_form(reacting):
    """The graded number. The target comes from `chem0d.predict`, which is stated in the
    module docstring: a second seven-coefficient root find here would be another copy of
    the thing under test, and the two halves of the claim are split between this file
    (the coefficients, against NIST, above) and `test_toolbox_chem0d.py` (the arithmetic,
    against a closed form)."""
    module = chem0d()
    predicted = module.predict(reacting["case"])
    run_record = module.read_chem_foam_out(reacting["case"])
    target = predicted["final"]["temperature_k"]
    got = run_record["temperature_k"]
    print(f"  chemFoam reached {got:.4f} K at t = {run_record['time_s']:g} s against "
          f"an enthalpy-conserving {target:.4f} K "
          f"({abs(got - target) / target:+.4%})")
    print(f"  from {T0_K:g} K, so a rise of {got - T0_K:.1f} K against "
          f"{target - T0_K:.1f} K predicted")
    assert got == pytest.approx(target, rel=5e-3)
    # And it is not a flame temperature: a single irreversible step makes no CO, OH, H,
    # O or NO, so this is the complete-combustion value and a real flame is cooler. The
    # size of that gap is deliberately not asserted -- see the module docstring.
    assert got > T0_K + 1000.0, "the reactor barely reacted"


def test_the_fuel_burnt_out_rather_than_stalling(reacting):
    """A one-step irreversible third-order reaction approaches complete consumption as a
    power law rather than exponentially, so whether the complete-combustion target
    applies at all is a question about the tail. The rate constant at the end
    temperature is enormous, which is why it closes -- but that is a measurement and
    not an argument."""
    result = chem0d().grade(reacting["case"])
    by_name = {check["name"]: check for check in result["checks"]}
    left = by_name["limiting_reactant_remaining"]
    print(f"  limiting reactant left: {left['value']:.3e} mass fraction")
    own = by_name["final_temperature_at_own_composition"]
    assert own["passed"] is True, "no final mass fractions were readable"
    print(f"  at the solver's own composition the target is {own['reference']} K")
    assert left["passed"] is True


def test_the_grader_passes_through_its_own_command_line(reacting):
    proc = subprocess.run([sys.executable, str(SCRIPT), "grade",
                           "--case", str(reacting["case"])],
                          text=True, capture_output=True)
    print(proc.stdout)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "FAIL" not in proc.stdout


# -- the control -------------------------------------------------------------------


def test_chemistry_off_does_not_heat_and_is_failed(frozen):
    """The contrast that makes the agreement above evidence. One word changed in
    `constant/chemistryProperties` and the same case, same mesh, same solver, same
    thermo data has to come back at its initial temperature -- and the grader has to say
    so rather than passing a reactor that never reacted."""
    module = chem0d()
    run_record = module.read_chem_foam_out(frozen["case"])
    print(f"  chemistry off reached {run_record['temperature_k']:.4f} K from "
          f"{T0_K:g} K")
    # 1 K, which is not a tolerance chosen to clear today's drift. With chemistry off
    # the solver still integrates the energy equation, and it measured 999.9985 K --
    # a drift of 0.0015 K, 1.5 ppm of 1000 K. The claim this test's name makes is that
    # no heat was RELEASED, and the reacting case next door releases 1877.6 K, so a 1 K
    # bound sits 660x above the drift and 1800x below the thing it has to catch. The
    # first version asserted 1 ppm, which is a statement about the discretisation rather
    # than about combustion and failed on the first real run.
    assert run_record["temperature_k"] == pytest.approx(T0_K, abs=1.0)
    assert abs(run_record["temperature_k"] - T0_K) < 0.01 * (2877.0 - T0_K), (
        "the frozen run moved by a noticeable fraction of the reacting run's rise")

    proc = subprocess.run([sys.executable, str(SCRIPT), "grade",
                           "--case", str(frozen["case"])],
                          text=True, capture_output=True)
    print(proc.stdout)
    assert proc.returncode == 1, "the grader passed a reactor that never reacted"
    assert "FAILED" in proc.stdout
    result = module.grade(frozen["case"])
    by_name = {check["name"]: check for check in result["checks"]}
    assert by_name["temperature_rise"]["passed"] is False
    assert by_name["final_temperature"]["passed"] is False
