"""`compressible.py`: the case files for a transonic or supersonic run on an existing mesh.

What is pinned here is the handful of things that make a compressible case wrong in
ways that look plausible and survive all the way into a report: pressure written as if
it were kinematic, a `forceCoeffs` handed `rhoInf` instead of the density field, an
outlet that reflects an arriving shock, a `transonic no` pressure equation on a
transonic case, and a time step set by the flow speed rather than by the flow speed
plus the speed of sound.

The mesh under test is the same real one `test_toolbox_case_gen.py` uses
(`tests/data/cavity`, one cell thick), copied and its patch names and types rewritten
per test, so pyvista measures the areas and normals rather than them being mocked. No
OpenFOAM is needed and none is used: the assertions are on the text of the files.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"
CAVITY = Path(__file__).resolve().parent / "data" / "cavity"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def compressible():
    return load("compressible")


@pytest.fixture(scope="module")
def preflight():
    return load("preflight")


def mesh_only(root: Path, name: str = "case") -> Path:
    """The cavity's mesh, and nothing else: no 0/, no dictionaries."""
    case = root / name
    (case / "constant").mkdir(parents=True)
    shutil.copytree(CAVITY / "constant" / "polyMesh", case / "constant" / "polyMesh")
    return case


def rename_patches(case: Path, mapping: dict[str, tuple[str, str]]) -> None:
    """Rewrite the boundary file's names and types; the faces stay as they were."""
    path = case / "constant" / "polyMesh" / "boundary"
    text = path.read_text(encoding="utf-8")
    for old, (new, kind) in mapping.items():
        text = re.sub(rf"^(\s*){old}\s*$", rf"\1{new}", text, flags=re.M)
        text = re.sub(rf"({new}\s*\n\s*\{{\s*\n\s*type\s+)\w+;", rf"\g<1>{kind};", text)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def duct(tmp_path):
    """The cavity renamed into a passage: an inlet, an outlet, empty ends."""
    case = mesh_only(tmp_path)
    rename_patches(case, {
        "movingWall": ("inlet", "patch"),
        "fixedWalls": ("outlet", "patch"),
        "frontAndBack": ("frontAndBack", "empty"),
    })
    return case


@pytest.fixture
def body(tmp_path):
    """A patch called `body`, so the force objects are written."""
    case = mesh_only(tmp_path, "body-case")
    rename_patches(case, {
        "movingWall": ("inlet", "patch"),
        "fixedWalls": ("body", "wall"),
        "frontAndBack": ("frontAndBack", "empty"),
    })
    return case


CONDITIONS = ("--t", "255.6", "--p", "80459", "--length", "0.64607")
"""The ONERA M6 AGARD AR-138 case 2308 conditions, which is the run this is aimed at."""


def run(module, case: Path, *args: str) -> int:
    return module.main([str(case), *args])


def text(case: Path, relative: str) -> str:
    return (case / relative).read_text(encoding="utf-8")


def entry_for(body_text: str, patch: str) -> str:
    """One patch's block out of a boundaryField."""
    match = re.search(rf"^    {patch}\n    \{{\n(.*?)^    \}}", body_text, re.S | re.M)
    assert match, f"no {patch} entry in\n{body_text}"
    return match.group(1)


# -- pressure and temperature ------------------------------------------------------


def test_pressure_is_written_in_pascals_and_never_as_kinematic_pressure(
        compressible, duct, capsys):
    """The one error this file exists to stop.

    `case_gen.field_p` writes `[0 2 -2 0 0 0 0]` because every incompressible OpenFOAM
    solver solves p/rho. A compressible solver solves p, and the two cases are one
    dimensions line apart in the text and a factor of rho apart in every force and
    every Cp computed afterwards -- 1.2 in air, which hides inside any useful band.
    """
    run(compressible, duct, "--mach", "0.84", *CONDITIONS, "--solver", "rhoSimpleFoam")
    capsys.readouterr()
    pressure = text(duct, "0/p")
    assert "dimensions      [1 -1 -2 0 0 0 0];" in pressure
    assert "[0 2 -2" not in pressure, "that is kinematic pressure, m2/s2"
    assert "uniform 80459" in pressure
    # and the absolute value reached the internal field, not a gauge zero
    assert "internalField   uniform 0;" not in pressure


def test_temperature_is_written_in_kelvin_as_its_own_field(compressible, duct, capsys):
    """A field the incompressible case does not have at all, with the one dimension
    (the fourth) that says kelvin."""
    run(compressible, duct, "--mach", "0.84", *CONDITIONS, "--solver", "rhoSimpleFoam")
    capsys.readouterr()
    temperature = text(duct, "0/T")
    assert "dimensions      [0 0 0 1 0 0 0];" in temperature
    assert "internalField   uniform 255.6;" in temperature


def test_thermophysical_properties_replaces_transport_properties(
        compressible, duct, capsys):
    """Not both, and not transportProperties alone. A compressible case has no nu in
    any file -- it is mu(T)/rho(p,T) -- so a kinematic viscosity left lying beside a
    thermoType is a second, contradicting answer to the same question."""
    run(compressible, duct, "--mach", "0.84", *CONDITIONS, "--solver", "rhoSimpleFoam")
    capsys.readouterr()
    assert not (duct / "constant" / "transportProperties").exists()
    thermo = text(duct, "constant/thermophysicalProperties")
    for required in ("hePsiThermo", "pureMixture", "sutherland", "hConst", "perfectGas",
                     "molWeight", "Cp", "Hf", "As", "Ts"):
        assert required in thermo, f"thermophysicalProperties has no {required}"
    assert "nu  " not in thermo and "transportModel" not in thermo


def test_the_energy_variable_follows_the_solver_that_has_to_solve_it(
        compressible, tmp_path, capsys):
    """rhoSimpleFoam's energy equation is in enthalpy and the density-based solvers'
    are in internal energy. Asking a solver for the variable it does not solve stops it
    at construction, before any iteration."""
    for solver, energy in (("rhoSimpleFoam", "sensibleEnthalpy"),
                           ("rhoCentralFoam", "sensibleInternalEnergy")):
        case = mesh_only(tmp_path, f"energy-{solver}")
        rename_patches(case, {"movingWall": ("inlet", "patch"),
                              "fixedWalls": ("outlet", "patch"),
                              "frontAndBack": ("frontAndBack", "empty")})
        run(compressible, case, "--mach", "0.84", *CONDITIONS, "--solver", solver)
        capsys.readouterr()
        assert f"energy          {energy};" in text(
            case, "constant/thermophysicalProperties")


# -- the numbers, computed from the file rather than from memory -------------------


def test_the_mach_number_and_sound_speed_come_from_the_cp_and_molar_mass_in_the_file(
        compressible, duct, capsys):
    """Not from a hard-coded gamma and R.

    Cp is formatted into the dictionary and gamma is then recovered from the formatted
    literal, so the reported speed of sound is the one the written file implies. The
    check is end to end: parse Cp and molWeight back out of the dictionary the run
    wrote, redo the arithmetic, and require the printed Mach number to agree.
    """
    run(compressible, duct, "--mach", "0.8395", *CONDITIONS, "--solver", "rhoSimpleFoam")
    printed = capsys.readouterr().out
    thermo = text(duct, "constant/thermophysicalProperties")
    cp = float(re.search(r"Cp\s+([0-9.eE+-]+);", thermo).group(1))
    molar = float(re.search(r"molWeight\s+([0-9.eE+-]+);", thermo).group(1))
    gas_constant = compressible.UNIVERSAL_GAS_CONSTANT / molar
    gamma = cp / (cp - gas_constant)
    sound = (gamma * gas_constant * 255.6) ** 0.5

    reported = float(re.search(r"a = ([0-9.eE+-]+) m/s", printed).group(1))
    assert reported == pytest.approx(sound, rel=1e-5)
    assert "M = 0.8395" in printed
    speed = float(re.search(r"U = ([0-9.eE+-]+) m/s and", printed).group(1))
    assert speed == pytest.approx(0.8395 * sound, rel=1e-5)
    # the M6 conditions this is aimed at: U 269.033 m/s, rho 1.0968 kg/m3
    assert speed == pytest.approx(269.03, rel=2e-3)
    assert "rho = 1.09659" in printed


def test_the_stagnation_conditions_are_reported_not_left_to_be_derived(
        compressible, duct, capsys):
    """A published condition set is usually quoted in total conditions and the model
    sees static. T0 = T(1 + (gamma-1)M^2/2) at M 0.8395 and 255.6 K is 291.6 K."""
    run(compressible, duct, "--mach", "0.8395", *CONDITIONS, "--solver", "rhoSimpleFoam")
    printed = capsys.readouterr().out
    total_t = float(re.search(r"T0 = ([0-9.eE+-]+) K", printed).group(1))
    assert total_t == pytest.approx(291.63, rel=1e-3)
    assert "p0 = " in printed and "q = " in printed


def test_a_stated_viscosity_refits_the_sutherland_constant_rather_than_contradicting_it(
        compressible, duct, capsys):
    """The M6 conditions name mu 1.6266e-5 at 255.6 K; the standard Sutherland pair
    gives 1.6279e-5. A published condition set is allowed to disagree with a
    correlation, and the knob refits As so the law in the file reproduces the number
    asked for instead of a second mu sitting somewhere the solver never reads."""
    run(compressible, duct, "--mach", "0.8395", *CONDITIONS, "--mu", "1.6266e-5",
        "--solver", "rhoSimpleFoam")
    printed = capsys.readouterr().out
    thermo = text(duct, "constant/thermophysicalProperties")
    refitted = float(re.search(r"As\s+([0-9.eE+-]+);", thermo).group(1))
    reproduced = refitted * 255.6 ** 0.5 / (1.0 + compressible.SUTHERLAND_TS / 255.6)
    assert reproduced == pytest.approx(1.6266e-5, rel=1e-5)
    assert "refitted" in printed and "1.6266e-05" in printed


def test_every_constant_is_printed_with_where_it_came_from(compressible, duct, capsys):
    """A constant with a docstring is still a magic constant if nothing prints it. The
    table is printed every run, not on request: a bad thermophysical number is not
    measurable from anything the product reports afterwards, so a study that inherits
    one explains the answer it produces."""
    run(compressible, duct, "--mach", "0.84", *CONDITIONS, "--solver", "rhoSimpleFoam")
    printed = capsys.readouterr().out
    assert "constants  every number this run wrote, and where it came from" in printed
    for name in ("molWeight", "Cp", "Hf", "As", "Ts", "R", "gamma", "a", "rho", "mu", "nu"):
        assert re.search(rf"^  {name}\s", printed, re.M), f"{name} has no sourced row"
    assert "US Standard Atmosphere 1976" in printed
    assert "White, Viscous Fluid Flow" in printed
    assert "N_A*k_B" in printed


# -- the boundary conditions, which are the regime's -------------------------------


def test_a_supersonic_request_gets_different_outlet_types_from_a_subsonic_one(
        compressible, tmp_path, capsys):
    """Below M 1 one characteristic runs back upstream, so a pressure has to be set at
    the outlet; above it every characteristic leaves and nothing may be imposed. A
    `fixedValue` p at a supersonic outlet is information travelling upstream through a
    supersonic flow, and `zeroGradient` at a subsonic one drains the domain."""
    subsonic = mesh_only(tmp_path, "subsonic")
    supersonic = mesh_only(tmp_path, "supersonic")
    for case in (subsonic, supersonic):
        rename_patches(case, {"movingWall": ("inlet", "patch"),
                              "fixedWalls": ("outlet", "patch"),
                              "frontAndBack": ("frontAndBack", "empty")})
    run(compressible, subsonic, "--mach", "0.5", *CONDITIONS, "--solver", "rhoSimpleFoam")
    capsys.readouterr()
    run(compressible, supersonic, "--mach", "2.0", *CONDITIONS, "--solver", "rhoSimpleFoam")
    capsys.readouterr()

    slow = entry_for(text(subsonic, "0/p"), "outlet")
    fast = entry_for(text(supersonic, "0/p"), "outlet")
    assert "waveTransmissive" in slow and "fieldInf" in slow and "lInf" in slow
    assert "zeroGradient" in fast
    assert "waveTransmissive" not in fast
    # and the inlet pressure flips the other way: fixed only once nothing can run back
    assert "zeroGradient" in entry_for(text(subsonic, "0/p"), "inlet")
    assert "fixedValue" in entry_for(text(supersonic, "0/p"), "inlet")
    # the temperature outlet follows the same split
    assert "inletOutlet" in entry_for(text(subsonic, "0/T"), "outlet")
    assert "zeroGradient" in entry_for(text(supersonic, "0/T"), "outlet")


def test_the_boundary_choice_is_stated_with_the_reason_rather_than_only_written(
        compressible, tmp_path, capsys):
    """The choice changes the answer, so it belongs in the output and not only in the
    file -- a reflected shock reads as a convergence problem."""
    case = mesh_only(tmp_path, "stated")
    rename_patches(case, {"movingWall": ("inlet", "patch"),
                          "fixedWalls": ("outlet", "patch"),
                          "frontAndBack": ("frontAndBack", "empty")})
    run(compressible, case, "--mach", "0.84", *CONDITIONS, "--solver", "rhoSimpleFoam")
    printed = capsys.readouterr().out
    assert "outlet     waveTransmissive" in printed
    assert "reflects an arriving shock" in printed
    assert "inlet      fixedValue on T and U, zeroGradient on p" in printed
    assert "regime     transonic: M = 0.8400" in printed


def test_the_transonic_pressure_equation_is_switched_on_by_the_mach_number(
        compressible, tmp_path, capsys):
    """`transonic yes;` is what makes rhoSimpleFoam form the pressure equation that
    carries `div(phid,p)`. Left at `no` the solver runs to convergence on the subsonic
    equation and nothing in the log says so -- the quietest failure in here."""
    for mach, expected in (("0.84", "transonic       yes;"), ("0.3", "transonic       no;")):
        case = mesh_only(tmp_path, f"transonic-{mach}")
        rename_patches(case, {"movingWall": ("inlet", "patch"),
                              "fixedWalls": ("outlet", "patch"),
                              "frontAndBack": ("frontAndBack", "empty")})
        run(compressible, case, "--mach", mach, *CONDITIONS, "--solver", "rhoSimpleFoam")
        capsys.readouterr()
        assert expected in text(case, "system/fvSolution")
    # and the term is named in divSchemes at second order either way: the M6 round-2
    # failure was this exact entry at `Gauss upwind`
    assert "div(phid,p)     bounded Gauss limitedLinear 1;" in text(case, "system/fvSchemes")
    assert "Gauss upwind" not in text(case, "system/fvSchemes")


def test_the_transient_step_is_set_by_the_acoustic_speed_not_the_flow_speed(
        compressible, tmp_path, capsys):
    """Co*h/(|U| + a), not Co*h/|U|. At M 0.84 that is a factor of 2.2 and at M 0.1 a
    factor of 11; an incompressible step handed to a density-based solver diverges in
    the first few steps and reads like a bad mesh."""
    case = mesh_only(tmp_path, "transient")
    rename_patches(case, {"movingWall": ("inlet", "patch"),
                          "fixedWalls": ("outlet", "patch"),
                          "frontAndBack": ("frontAndBack", "empty")})
    run(compressible, case, "--mach", "0.84", *CONDITIONS, "--solver", "rhoCentralFoam",
        "--study", "transient", "--end-time", "0.001", "--courant", "0.5")
    printed = capsys.readouterr().out
    assert "(U + a)" in printed
    control = text(case, "system/controlDict")
    step = float(re.search(r"deltaT          ([0-9.eE+-]+);", control).group(1))
    sound = float(re.search(r"a = ([0-9.eE+-]+) m/s", printed).group(1))
    cell = float(re.search(r"x ([0-9.eE+-]+) m \((?:the \w+ cell)\)", printed).group(1))
    assert step == pytest.approx(0.5 * cell / (0.84 * sound + sound), rel=1e-3)
    # which is strictly smaller than the incompressible rule would have given
    assert step < 0.5 * cell / (0.84 * sound)
    assert "application     rhoCentralFoam;" in control


def test_an_undescribed_patch_is_refused_rather_than_quietly_defaulted(
        compressible, tmp_path, monkeypatch):
    """`case_gen.boundary_field` is driven by the mesh's own patch list, so a patch no
    role describes is a loud error here instead of a missing entry the solver finds on
    the first time step. The walker is reused, so the refusal is the same one."""
    case = mesh_only(tmp_path, "undescribed")
    rename_patches(case, {"movingWall": ("inlet", "patch"),
                          "fixedWalls": ("outlet", "patch"),
                          "frontAndBack": ("frontAndBack", "empty")})
    module = load("compressible")
    original = module.case_gen.roles_for

    def forget_one(mesh, opts):
        roles = original(mesh, opts)
        roles.pop("outlet")
        return roles

    monkeypatch.setattr(module.case_gen, "roles_for", forget_one)
    with pytest.raises(SystemExit) as raised:
        run(module, case, "--mach", "0.84", *CONDITIONS, "--solver", "rhoSimpleFoam")
    assert "outlet" in str(raised.value)
    assert "every patch needs one" in str(raised.value)


def test_a_farfield_patch_is_named_rather_than_guessed_from_its_name(
        compressible, tmp_path, capsys):
    """A patch standing in for unbounded fluid is neither inlet nor outlet -- flow
    crosses it both ways over one run -- and `case_gen.roles_for` reads the name
    `farfield` as saying nothing and calls it a wall, which here would be a no-slip
    surface sixteen chords from the body. So it is a flag, and a flag naming a patch
    the mesh does not have is as loud as a wrong `--inlet`."""
    case = mesh_only(tmp_path, "farfield")
    rename_patches(case, {"movingWall": ("farfield", "patch"),
                          "fixedWalls": ("body", "wall"),
                          "frontAndBack": ("frontAndBack", "empty")})
    run(compressible, case, "--mach", "0.84", *CONDITIONS, "--farfield", "farfield",
        "--solver", "rhoSimpleFoam", "--aref", "0.7532", "--up", "y")
    capsys.readouterr()
    assert "freestreamPressure" in entry_for(text(case, "0/p"), "farfield")
    assert "freestreamVelocity" in entry_for(text(case, "0/U"), "farfield")
    assert "noSlip" in entry_for(text(case, "0/U"), "body")

    with pytest.raises(SystemExit) as raised:
        run(compressible, case, "--mach", "0.84", *CONDITIONS, "--farfield", "sky",
            "--solver", "rhoSimpleFoam", "--up", "y", "--force")
    assert "no patch called sky" in str(raised.value)


# -- the force coefficients, which is the other factor-of-rho trap -----------------


def test_force_coefficients_read_the_density_field_and_not_rho_inf(
        compressible, body, capsys):
    """The compressible branch of the trap `preflight.force_units_diagnosis` names.

    An incompressible case has no density field, so the force object is handed `rho
    rhoInf; rhoInf 1.225;`. A compressible case has one, so `rho rho;` makes the
    object read it and the forces come out in newtons. Carrying the incompressible
    pair over scales every coefficient by 1/rhoInf -- a clean factor of 1.2 in air at
    sea level, and of rho itself anywhere the altitude is not sea level.
    """
    run(compressible, body, "--mach", "0.8395", *CONDITIONS, "--solver", "rhoSimpleFoam",
        "--aref", "0.7532", "--lref", "0.64607", "--up", "x")
    printed = capsys.readouterr().out
    control = text(body, "system/controlDict")
    assert "type            forceCoeffs;" in control
    assert "rho             rho;" in control
    assert "rhoInf" not in control, "rhoInf is the kinematic-pressure form"
    assert "Aref            0.7532;" in control
    assert "lRef            0.64607;" in control
    assert "rho rho" in printed and "would scale every coefficient" in printed


def test_preflight_judges_the_generated_force_object_the_compressible_way(
        compressible, preflight, body, capsys):
    """The two halves have to agree or one of them is wrong about the same case.

    `preflight.py` reads `0/p`'s dimensions and judges the density entry by them: a
    kinematic case without `rho rhoInf` fails, and a static-pressure case *with* it
    warns. This generator writes the other branch, so preflight already knew it -- and
    this is the test that says the two agree rather than both being plausible alone.
    """
    run(compressible, body, "--mach", "0.8395", *CONDITIONS, "--solver", "rhoSimpleFoam",
        "--aref", "0.7532", "--up", "x")
    capsys.readouterr()
    control = text(body, "system/controlDict")
    objects = preflight.find_force_objects(control)
    assert objects and objects[0]["rho"] == "rho"

    dimensions = preflight.parse_dimensions(text(body, "0/p"))
    assert dimensions == preflight.STATIC_PRESSURE
    assert preflight.force_units_diagnosis(dimensions, objects)["status"] == "ok"
    # and the same object read against a kinematic p is the failure preflight exists for
    kinematic = preflight.force_units_diagnosis(preflight.KINEMATIC_PRESSURE, objects)
    assert kinematic["status"] == "fail"


def test_a_three_dimensional_case_without_a_reference_area_writes_forces_not_coefficients(
        compressible, body, capsys, monkeypatch):
    """Aref on a 3D wing is a planform area no script can see, and `case_gen`'s own
    docstring records the easy factor of ten that comes of defaulting it. Newtons need
    no reference area, so that is what gets written, with the reason said out loud."""
    module = load("compressible")
    read = module.case_gen.read_mesh

    def as_three_dimensional(case):
        mesh = read(case)
        mesh.two_d = False
        return mesh

    monkeypatch.setattr(module.case_gen, "read_mesh", as_three_dimensional)
    run(module, body, "--mach", "0.8395", *CONDITIONS, "--solver", "rhoSimpleFoam",
        "--up", "x")
    printed = capsys.readouterr().out
    control = text(body, "system/controlDict")
    assert "type            forces;" in control
    assert "forceCoeffs" not in control
    assert "Aref" not in control
    assert "forceCoeffs is NOT written" in printed and "--aref" in printed


def test_lift_is_measured_towards_a_stated_axis_because_nothing_can_measure_it(
        compressible, tmp_path, capsys):
    """A wing at incidence and a wing with sideslip are the same mesh and the same
    freestream vector. The M6 round 3 ran `liftDir (-0.0533104 0 0.9985782)` on a z-up
    mesh; taking the axis least aligned with the freestream instead picks y there and
    reports sideslip as lift."""
    case = mesh_only(tmp_path, "axes")
    rename_patches(case, {"movingWall": ("inlet", "patch"),
                          "fixedWalls": ("body", "wall"),
                          "frontAndBack": ("frontAndBack", "empty")})
    run(compressible, case, "--mach", "0.84", *CONDITIONS, "--solver", "rhoSimpleFoam",
        "--aref", "0.5", "--up", "x")
    printed = capsys.readouterr().out
    control = text(case, "system/controlDict")
    drag = re.search(r"dragDir         \(([^)]*)\)", control).group(1).split()
    lift = re.search(r"liftDir         \(([^)]*)\)", control).group(1).split()
    assert sum(float(d) * float(l) for d, l in zip(drag, lift)) == pytest.approx(0, abs=1e-6)
    assert sum(float(c) ** 2 for c in lift) == pytest.approx(1.0, rel=1e-6)
    assert "up is --up x" in printed
    # pitchAxis is the third axis taken POSITIVE, which is the sign convention Cm is
    # quoted in. A cross product gets one of the two handednesses backwards -- it
    # leaves |Cm| alone and flips its sign, so every pitching moment reads the wrong
    # way round. OpenFOAM's own airFoil2D has dragDir (1 0 0), liftDir (0 1 0),
    # pitchAxis (0 0 1), and `lift x drag` there is (0 0 -1).
    pitch = re.search(r"pitchAxis       \(([^)]*)\)", control).group(1).split()
    assert [float(c) for c in pitch] == [0.0, 0.0, 1.0]
    assert sum(float(p) * float(d) for p, d in zip(pitch, drag)) == pytest.approx(0, abs=1e-6)
    assert sum(float(p) * float(l) for p, l in zip(pitch, lift)) == pytest.approx(0, abs=1e-6)


# -- the solvers, reported rather than assumed ------------------------------------


def test_a_requested_solver_that_is_not_installed_is_reported_rather_than_assumed(
        compressible, duct, capsys):
    """The dictionaries are the deliverable and the machine that writes them need not
    be the machine that runs them -- writing a HiSA case on a laptop is ordinary. What
    would be wrong is writing it and implying the solver is there, so every run prints
    which of the three is actually on this machine."""
    run(compressible, duct, "--mach", "0.84", *CONDITIONS, "--solver", "rhoSimpleFoam")
    printed = capsys.readouterr().out
    assert re.search(r"^installed  ", printed, re.M)
    for name in ("rhoSimpleFoam", "rhoCentralFoam", "hisa"):
        assert re.search(rf"{name}: (here|NOT here)", printed), f"{name} is not reported"
    # on a machine with no OpenFOAM all three read NOT here, and the case is still written
    assert (duct / "0" / "p").is_file()


def test_hisa_refuses_to_invent_the_schemes_it_cannot_read(compressible, tmp_path, capsys):
    """HiSA's dictionary set is its own -- AUSM+up, wVanLeer, dual-time rPseudoDeltaT,
    GMRES under LU-SGS -- and the only authority for the block nesting is the example
    case in the image, which this cannot read from a laptop. So the 0/, the thermo and
    the controlDict are written and the two scheme files are not, with the `cp`
    commands printed. A refusal that names what is missing beats a guess that parses.
    """
    case = mesh_only(tmp_path, "hisa")
    rename_patches(case, {"movingWall": ("farfield", "patch"),
                          "fixedWalls": ("wing", "wall"),
                          "frontAndBack": ("frontAndBack", "empty")})
    assert run(compressible, case, "--mach", "0.8395", *CONDITIONS, "--solver", "hisa",
               "--farfield", "farfield", "--body", "wing", "--aref", "0.7532",
               "--up", "y") == 0
    printed = capsys.readouterr().out
    assert not (case / "system" / "fvSchemes").exists()
    assert not (case / "system" / "fvSolution").exists()
    assert (case / "system" / "controlDict").is_file()
    assert "are NOT written" in printed
    assert "examples/oneraM6/simulation/system/fvSchemes" in printed
    assert "AUSMPlusUp" in printed and "LU-SGS" in printed
    # the characteristic boundary conditions are HiSA's own and are what the far field
    # takes; the wing takes its corrected wall velocity
    assert "characteristicFarfieldPressure" in entry_for(text(case, "0/p"), "farfield")
    assert "characteristicFarfieldVelocity" in entry_for(text(case, "0/U"), "farfield")
    assert "boundaryCorrectedFixedValue" in entry_for(text(case, "0/U"), "wing")
    assert "application     hisa;" in text(case, "system/controlDict")


def test_rho_central_foam_says_it_is_the_laminar_case_rather_than_pretending_otherwise(
        compressible, duct):
    """Its dictionary set here is the inviscid shock-capturing one. A turbulent
    transonic wing on a layered mesh is the case where its explicit step is set by the
    165 um prism layer rather than the 4.7 mm shock cell, which is a factor of 28 in
    money, so it refuses rather than writing a case that will cost that."""
    with pytest.raises(SystemExit) as raised:
        run(compressible, duct, "--mach", "2.0", *CONDITIONS, "--solver", "rhoCentralFoam",
            "--turbulence", "kOmegaSST")
    assert "rhoCentralFoam" in str(raised.value)
    assert "--solver hisa" in str(raised.value)


def test_a_case_with_neither_mach_nor_speed_is_refused_by_name(compressible, duct):
    """The Mach number picks every boundary condition in the case, so there is no
    default for it that would not be a guess about the regime."""
    with pytest.raises(SystemExit):
        run(compressible, duct, *CONDITIONS)


def test_a_negative_absolute_pressure_is_refused_with_the_reason(compressible, duct):
    """p here is absolute, and the habit an incompressible case teaches is setting the
    outlet pressure to zero. Zero pascals absolute is a vacuum and perfectGas divides
    by it."""
    with pytest.raises(SystemExit) as raised:
        run(compressible, duct, "--mach", "0.84", "--p", "0", "--length", "0.1")
    assert "absolute" in str(raised.value)


def test_a_low_mach_request_says_the_incompressible_case_is_the_cheaper_instrument(
        compressible, duct, capsys):
    """It still writes the case. Below M 0.3 density varies by under 5% and
    `case_gen.py` answers the same question for less, which is worth saying once rather
    than refusing over."""
    assert run(compressible, duct, "--mach", "0.1", *CONDITIONS,
               "--solver", "rhoSimpleFoam") == 0
    printed = capsys.readouterr().out
    assert "low-speed" in printed
    assert "case_gen.py's incompressible case is the cheaper instrument" in printed


def test_a_dry_run_lists_exactly_what_a_real_run_would_write(compressible, duct, capsys):
    """A dry run that lists different files from the ones a real run writes is worse
    than no dry run, so both come off one answer."""
    run(compressible, duct, "--mach", "0.84", *CONDITIONS, "--solver", "rhoSimpleFoam",
        "--dry-run")
    listed = set(re.findall(r"^  (\S+/\S+)   \(", capsys.readouterr().out, re.M))
    assert not (duct / "0").exists()
    run(compressible, duct, "--mach", "0.84", *CONDITIONS, "--solver", "rhoSimpleFoam")
    capsys.readouterr()
    written = {
        path.relative_to(duct).as_posix()
        for path in duct.rglob("*")
        if path.is_file() and "polyMesh" not in path.parts
    }
    assert listed == {name for name in written if "/" in name}


# -- the field note ---------------------------------------------------------------


def test_the_field_notes_carry_what_the_transonic_rounds_cost_to_learn():
    """Rounds 1 to 3 of the ONERA M6 replication lived in `qa-runs/comp/onera/REPORT.md`
    and the agent never reads that. Each of these is a fact one of those rounds paid
    for: p in pascals rather than m2/s2, `rho rho` rather than `rho rhoInf`, the
    first-order pressure flux that diffused the shock away on a mesh that was fine,
    `transonic yes` as the switch nothing reports, the acoustic time step, and nu
    becoming an output rather than a file entry."""
    notes = (TOOLBOX / "notes" / "openfoam-field-notes.md").read_text(encoding="utf-8")
    assert "## Dressing a mesh for a compressible run" in notes
    section = notes.split("## Dressing a mesh for a compressible run", 1)[1].split("\n## ", 1)[0]
    for fact in ("thermophysicalProperties", "transportProperties", "hePsiThermo",
                 "perfectGas", "sensibleInternalEnergy", "sensibleEnthalpy",
                 "waveTransmissive", "transonic", "div(phid,p)", "alphat",
                 "rho rhoInf", "compressible::alphatWallFunction", "0.63",
                 "Sutherland"):
        assert fact in section, f"the compressible note does not mention {fact}"
    assert "compressible.py" in section, "the note does not name the script that does this"


def test_the_compressible_note_states_facts_rather_than_a_procedure():
    """The notes are offered like everything else in the toolbox. A section that told a
    session the order to do things in would be the workflow injection the design exists
    to avoid."""
    notes = (TOOLBOX / "notes" / "openfoam-field-notes.md").read_text(encoding="utf-8")
    section = " ".join(notes.split("## Dressing a mesh for a compressible run", 1)[1]
                       .split("\n## ", 1)[0].lower().split())
    for imperative in ("you must", "always run", "before you", "step 1", "first,",
                       "you should", "make sure"):
        assert imperative not in section
