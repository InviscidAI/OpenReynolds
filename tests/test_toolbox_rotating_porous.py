"""`templates/rotating.py` and `templates/porous.py`: a frozen rotor and a porous slab.

`test_toolbox_templates.py` covers what every template owes -- it parses, it marks its
two numbers, it is in the index, it imports nothing the instance lacks -- by reading the
source with `ast` rather than importing it. This file asks the harder questions that
only apply to these two, which cannot be read off the source, so each module is loaded
by path and its dictionary writers are called against a `tmp_path`. Loading them is safe
for the same reason the templates work at all: `import gmsh` lives inside `build_mesh`,
so nothing at module scope needs the mesher.

Both templates have the same shape of problem: a physical modelling choice whose cost is
invisible in the log, and a unit factor that turns a right answer into a plausible wrong
one.

For the rotor the factor is **9.5493**, rad/s against rpm. `omega` in
`constant/MRFProperties` is radians per second -- OpenFOAM's `MRFZone.H` declares the
member `//- Angular velocity (rad/sec)` and there is no `rpm` key in that class to catch
the mistake -- so a nameplate figure written straight in runs 9.55 times too fast and
nothing says so. The anchor these tests use is the shipped
`$FOAM_TUTORIALS/incompressible/simpleFoam/mixerVessel2D`, which carries `omega 104.72`:
that is 1000.0 rpm in rad/s to five figures, a round nameplate number, and read as rpm it
would be nobody's round number. The modelling choice is the frozen rotor itself, and the
tests below require the generated case to SAY what it cannot compute -- blade passing,
the unsteady wake -- because a steady answer that is quoted as if it were unsteady is the
failure MRF actually produces in practice.

For the slab the factor is **rho**, 1.204 here. `porousSimpleFoam` is incompressible and
solves p/rho in m2/s2, so the drop the case reports is the measured drop divided by the
density: 503.34 Pa measured against 418.06 solved is exact agreement, and calling it a
20% error is the mistake. The modelling content is `d` and `f`, which nobody knows and
which the template therefore derives; the tests here check the derivation against two
independent routes with answers known in closed form, including the factor of 0.5 that
OpenFOAM puts on `f` and Ergun does not.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLBOX = ROOT / "openreynolds" / "toolbox"
TEMPLATES = TOOLBOX / "templates"
ROTATING = TEMPLATES / "rotating.py"
POROUS = TEMPLATES / "porous.py"

HAS_GMSH = importlib.util.find_spec("gmsh") is not None
needs_gmsh = pytest.mark.skipif(not HAS_GMSH, reason="gmsh module not on this machine")

RAD_S_PER_RPM = 2.0 * math.pi / 60.0
"""0.10471975511965977, written out here independently of the module under test so the
conversion is checked against arithmetic rather than against itself."""

MIXER_VESSEL_OMEGA = 104.72
"""`omega` as shipped in `$FOAM_TUTORIALS/incompressible/simpleFoam/mixerVessel2D/
constant/MRFProperties`. The evidence that the key is rad/s: this is 1000 rpm."""


def load(path: Path, tmp_path: Path, name: str, **edits: object):
    """A template as an imported module, optionally with its constants edited.

    "Edit the numbers and re-run" means re-running the file, not reassigning a global:
    nearly everything in these templates is derived from the editable constants at
    module scope, so a test that only rebound one would be testing something no user
    does. The source is rewritten and loaded afresh instead -- the same approach
    `test_toolbox_free_surface.py` takes.
    """
    source = path.read_text(encoding="utf-8")
    for const, value in edits.items():
        # Either a whole bracketed literal or a scalar to the end of its line. The
        # measured curve is a multi-line list, and replacing only its first line would
        # leave the rest of the list behind as an IndentationError.
        source, n = re.subn(rf"^{const} = (?:\[[^\]]*\]|\S+.*)$",
                            f"{const} = {value!r}", source, count=1, flags=re.M)
        assert n == 1, f"{const} is not a single editable module constant"
    out = tmp_path / f"{name}.py"
    out.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, out)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def case(path: Path, tmp_path: Path, name: str, **edits: object) -> tuple[Path, object]:
    """The case written to disk, and the module that wrote it."""
    module = load(path, tmp_path, name, **edits)
    out = tmp_path / name
    out.mkdir()
    module.write_case(out)
    return out, module


def entry(text: str, key: str) -> str:
    """The value of a `key value;` entry, with comments stripped."""
    match = re.search(rf"^\s*{re.escape(key)}\s+([^;/]+?)\s*(?://.*)?;", text, re.M)
    assert match, f"no entry {key!r} in:\n{text}"
    return match.group(1).strip()


def patch_block(text: str, patch: str) -> str:
    """One `boundaryField` entry's braces, which is where a wrong condition hides."""
    match = re.search(rf"^    {patch}\n    \{{\n(.*?)^    \}}", text, re.M | re.S)
    assert match, f"no boundary entry for {patch!r} in:\n{text}"
    return match.group(1)


def assert_convection_is_bounded(schemes: str) -> None:
    """Every `div(phi,...)` term carries the `bounded` prefix, directly or by macro.

    A `steadyState` discretisation drops the `-div(phi)*field` term a transient one
    carries, and without it `k` and `epsilon` can go negative in the first iterations,
    when the flux field is nowhere near divergence-free. `bounded` puts it back.

    Accepting a macro reference is not a loophole: the shipped tutorials define
    `turbulence  bounded Gauss upwind;` once and write `div(phi,k)  $turbulence;`
    beneath it, so a line that expands a bounded macro is bounded. What makes that safe
    to accept is that the macro's own definition is checked here too.
    """
    macros = dict(re.findall(r"^    (\w+)\s+((?:bounded )?Gauss[^;]*);", schemes, re.M))
    convection = [line for line in schemes.splitlines() if "div(phi," in line]
    assert convection, "no convection schemes at all"
    for line in convection:
        if "bounded" in line:
            continue
        used = re.search(r"\$(\w+)", line)
        assert used, f"unbounded steady convection: {line.strip()!r}"
        body = macros.get(used.group(1))
        assert body and "bounded" in body, (
            f"{line.strip()!r} expands ${used.group(1)}, which is not bounded"
        )


# == rotating.py: the frozen rotor =============================================


def test_the_rotor_case_carries_every_file_simplefoam_needs(tmp_path):
    """A missing dictionary is the one failure that IS loud, and it costs a hosted
    launch to find out. The list is the solver's, not a preference: simpleFoam with
    kEpsilon and an MRF zone reads exactly these."""
    out, _ = case(ROTATING, tmp_path, "rot0")
    written = {str(p.relative_to(out)).replace("\\", "/")
               for p in out.rglob("*") if p.is_file()}
    for required in (
        "0/U", "0/p", "0/k", "0/epsilon", "0/nut",
        "constant/MRFProperties", "constant/transportProperties",
        "constant/turbulenceProperties",
        "system/controlDict", "system/fvSchemes", "system/fvSolution",
    ):
        assert required in written, f"{required} is missing from the case"
    # MRF is a source term on a mesh that never moves. A dynamicMeshDict here would be
    # a second, contradictory statement about the motion.
    assert "constant/dynamicMeshDict" not in written
    assert "0/pointDisplacement" not in written


def test_omega_is_radians_per_second_and_the_round_trip_closes(tmp_path):
    """The factor of 9.5493. `omega` is rad/s; ROTOR_RPM is rpm; one function converts
    and the summary converts back. A round trip that does not return where it started
    is the cheapest check on a unit factor there is."""
    out, module = case(ROTATING, tmp_path, "rot_units")

    # the conversion itself, against arithmetic written out independently above
    assert module.RAD_S_PER_RPM == pytest.approx(RAD_S_PER_RPM, rel=1e-12)
    assert module.omega_rad_per_s(1.0) == pytest.approx(RAD_S_PER_RPM, rel=1e-12)
    assert module.rpm_from_rad_per_s(1.0) == pytest.approx(60.0 / (2 * math.pi), rel=1e-12)
    # the classic silent error is this factor, so it is asserted as a number
    assert module.rpm_from_rad_per_s(1.0) == pytest.approx(9.549296586, rel=1e-09)
    # and the round trip closes for an arbitrary value, not only the default
    assert module.rpm_from_rad_per_s(module.omega_rad_per_s(1234.5)) == \
        pytest.approx(1234.5, rel=1e-12)

    # the anchor: the shipped mixerVessel2D tutorial's own omega is 1000 rpm in rad/s,
    # which is what makes "omega is rad/s" evidence rather than recollection
    assert module.omega_rad_per_s(1000.0) == pytest.approx(MIXER_VESSEL_OMEGA, rel=1e-04)

    # what reached the dictionary is the rad/s number, not the rpm one
    mrf = (out / "constant" / "MRFProperties").read_text(encoding="utf-8")
    written = float(entry(mrf, "omega"))
    # the dictionary carries nine significant figures, so the comparison is against
    # that written precision rather than against the full float
    assert written == pytest.approx(module.omega_rad_per_s(), rel=1e-08)
    assert written != pytest.approx(module.ROTOR_RPM, rel=1e-03), (
        "the rpm figure was written into omega, which runs the rotor 9.55x too fast"
    )


def test_the_units_are_unmissable_in_both_the_case_and_the_output(tmp_path):
    """The brief's requirement, and the reason it is a requirement: the person who gets
    this wrong is the one who never opened the dictionary. So both units, the factor and
    the conversion appear in the written file AND in what the script prints."""
    out, module = case(ROTATING, tmp_path, "rot_say")
    mrf = (out / "constant" / "MRFProperties").read_text(encoding="utf-8")
    printed = module.summary()

    for text, where in ((mrf, "constant/MRFProperties"), (printed, "the printed summary")):
        assert "rad" in text.lower(), f"{where} never names radians"
        assert "rpm" in text.lower(), f"{where} never names rpm"
        # the conversion factor itself, either way round, to four figures
        assert "9.549" in text or "0.1047" in text, (
            f"{where} does not carry the rad/s <-> rpm conversion"
        )
    # the printed summary states both numbers and the round trip
    assert f"{module.ROTOR_RPM:.6g}" in printed
    assert "round trip" in printed.lower()
    # and the tip speed, which is the number that says whether a speed is sane at all
    assert "tip speed" in printed.lower()
    assert module.tip_speed_m_s() == pytest.approx(
        module.omega_rad_per_s() * module.BLADE_RADIUS_M, rel=1e-12)


def test_the_mrf_zone_is_declared_the_way_this_openfoam_wants_it(tmp_path):
    """Keys read off the shipped mixerVessel2D: a named block with `cellZone`, `active`,
    `nonRotatingPatches`, `origin`, `axis` and `omega`. A zone naming a cellZone the
    mesh does not have fails at the first solver step."""
    out, _ = case(ROTATING, tmp_path, "rot_zone")
    mrf = (out / "constant" / "MRFProperties").read_text(encoding="utf-8")
    assert "object      MRFProperties" in mrf
    assert re.search(r"^MRF\w*\n\{", mrf, re.M), "no named MRF zone block"
    assert entry(mrf, "cellZone") == "rotorZone"
    assert entry(mrf, "active") in ("yes", "on", "true")
    assert [float(v) for v in entry(mrf, "axis").strip("()").split()] == [0.0, 0.0, 1.0]
    assert [float(v) for v in entry(mrf, "origin").strip("()").split()] == [0.0, 0.0, 0.0]
    # `nonRotatingPatches` stated as an empty list rather than omitted, with the
    # explanation: a stationary patch inside a rotating zone that is left out of it gets
    # spun with the zone, and the run reports nothing.
    assert "nonRotatingPatches" in mrf
    assert re.search(r"nonRotatingPatches\s*\(\s*\)\s*;", mrf), (
        "nonRotatingPatches is not an explicit empty list"
    )
    assert "stationary" in mrf.lower(), (
        "nothing says what nonRotatingPatches is for, which is the defect it prevents"
    )


def test_the_rotor_wall_is_noslip_because_mrf_corrects_it(tmp_path):
    """The point worth reading twice, and the one a careful author gets wrong.

    MRF declares `void correctBoundaryVelocity(volVectorField& U) const;` -- "Correct
    the boundary velocity for the rotation of the MRF region" -- and sets the zone's
    patch velocities from Omega x r itself, on a mesh that never moves. The shipped
    mixerVessel2D puts plain `noSlip` on its `rotor` patch for this reason. A
    `rotatingWallVelocity` here would impose the rotation a second time.
    """
    out, _ = case(ROTATING, tmp_path, "rot_wall")
    u = (out / "0" / "U").read_text(encoding="utf-8")
    rotor = patch_block(u, "rotor")
    assert "noSlip" in rotor
    assert "rotatingWallVelocity" not in u, (
        "rotatingWallVelocity on a patch inside an MRF zone applies the rotation twice"
    )
    assert "movingWallVelocity" not in u, "the mesh does not move in an MRF case"
    # and the reason is written where the entry is, not only in the docstring
    assert "MRF" in rotor, "nothing explains why a spinning blade reads noSlip"
    assert patch_block(u, "stator").count("noSlip") == 1
    # the case stays 2D: the z faces are empty, not two more open sides
    for field in ("U", "p", "k", "epsilon", "nut"):
        block = patch_block((out / "0" / field).read_text(encoding="utf-8"),
                            "frontAndBack")
        assert "empty" in block, f"0/{field} leaves the z faces open"


def test_the_case_says_plainly_what_a_frozen_rotor_cannot_tell_you(tmp_path):
    """The brief's other requirement for MRF, and the failure it addresses is not a
    crashed run: it is a steady answer quoted as though it were unsteady. So the
    generated case itself -- not a note beside it -- names what is absent."""
    out, module = case(ROTATING, tmp_path, "rot_honest")
    mrf = (out / "constant" / "MRFProperties").read_text(encoding="utf-8")
    printed = module.summary()

    for text, where in ((mrf, "constant/MRFProperties"), (printed, "the printed summary")):
        lower = text.lower()
        assert "blade passing" in lower, f"{where} does not say blade passing is absent"
        assert "wake" in lower, f"{where} does not say the unsteady wake is absent"
        # the alternative is named, so the limitation comes with a way out
        assert "cyclicami" in lower or "sliding" in lower, (
            f"{where} names no alternative that would compute it"
        )
    # the blade-passing frequency as a NUMBER, so "can this see my 100 Hz" has an answer
    expected = module.N_BLADES * module.ROTOR_RPM / 60.0
    assert module.blade_passing_hz() == pytest.approx(expected, rel=1e-12)
    assert f"{expected:.6g}" in mrf, "the blade-passing frequency is not in the case"
    assert "Hz" in mrf
    # and the zone-position check, which is how a reader tells a physical torque from a
    # modelling artefact -- ladder.py's frozen-rotor rung
    assert "zone" in mrf.lower() and "per cent" in mrf.lower()


def test_a_closed_vessel_gets_a_pressure_reference(tmp_path):
    """There is no inlet and no outlet here: every patch is a wall or `empty`. An
    incompressible steady solve of a closed domain has a singular pressure equation
    without a reference cell, and the symptom is a pressure that drifts without bound
    while the velocity field stays perfectly healthy."""
    out, _ = case(ROTATING, tmp_path, "rot_pref")
    solution = (out / "system" / "fvSolution").read_text(encoding="utf-8")
    assert "pRefCell" in solution, "a closed domain with no pressure reference"
    assert "pRefValue" in solution
    # the domain really is closed: no patch carries an inlet or outlet condition
    u = (out / "0" / "U").read_text(encoding="utf-8")
    assert "inletOutlet" not in u and "fixedValue" not in u
    # p is kinematic, and the unit is stated where the dimensions are
    p = (out / "0" / "p").read_text(encoding="utf-8")
    assert entry(p, "dimensions") == "[0 2 -2 0 0 0 0]"
    assert "m2/s2" in p and "pascals" in p.lower(), (
        "nothing says this pressure is p/rho, which is a factor of rho in every force"
    )


def test_the_rotor_case_is_steady_and_its_numerics_say_so(tmp_path):
    """`steadyState` ddt with `bounded` convection, and `endTime` counting iterations.

    Both halves matter. A steady discretisation drops the `-div(phi)*field` term a
    transient one carries, and without the `bounded` prefix `k` and `epsilon` can go
    negative in the first iterations. And `deltaT` here is an iteration counter, so a
    Courant number for this run is a number that does not exist.
    """
    out, _ = case(ROTATING, tmp_path, "rot_numerics")
    control = (out / "system" / "controlDict").read_text(encoding="utf-8")
    assert entry(control, "application") == "simpleFoam"
    assert entry(control, "startFrom") == "latestTime", "an interrupted solve restarts"
    assert "ITERATIONS" in control, "nothing says endTime is not seconds"

    schemes = (out / "system" / "fvSchemes").read_text(encoding="utf-8")
    assert entry(schemes, "default") == "steadyState"  # first block is ddtSchemes
    assert_convection_is_bounded(schemes)


def test_the_turbulence_model_and_its_fields_agree(tmp_path):
    """kEpsilon reads `epsilon` and never reads `omega`. The same list has to drive 0/,
    the linear solvers and the divSchemes or one of the three describes a different
    case, and the solver stops before the first iteration saying it cannot find a
    file."""
    out, _ = case(ROTATING, tmp_path, "rot_turb")
    turbulence = (out / "constant" / "turbulenceProperties").read_text(encoding="utf-8")
    assert entry(turbulence, "simulationType") == "RAS"
    assert entry(turbulence, "RASModel") == "kEpsilon"

    written = {p.name for p in (out / "0").iterdir()}
    assert {"k", "epsilon", "nut"} <= written
    assert "omega" not in written, "kEpsilon never reads omega"

    solution = (out / "system" / "fvSolution").read_text(encoding="utf-8")
    assert "epsilon" in solution and "omega" not in solution
    schemes = (out / "system" / "fvSchemes").read_text(encoding="utf-8")
    assert "div(phi,epsilon)" in schemes

    # the wall functions are the names a real solved kEpsilon case in studies/ carries
    assert "kqRWallFunction" in (out / "0" / "k").read_text(encoding="utf-8")
    assert "epsilonWallFunction" in (out / "0" / "epsilon").read_text(encoding="utf-8")
    # Spalding rather than the high-Re form, because a template cannot know its y+;
    # the field notes record that a generated case writes Spalding on every model
    assert "nutUSpaldingWallFunction" in (out / "0" / "nut").read_text(encoding="utf-8")


def test_the_rotor_case_reports_a_torque_with_its_density(tmp_path):
    """A rotating-machinery case with no torque output has delivered nothing. And a
    moment taken off a kinematic pressure field without the density is out by a factor
    of rho -- 998 for water -- and reads as a plausible small number."""
    out, module = case(ROTATING, tmp_path, "rot_torque")
    control = (out / "system" / "controlDict").read_text(encoding="utf-8")
    functions = control.split("functions", 1)
    assert len(functions) == 2, "no function objects at all, so no torque"
    body = functions[1]
    assert "type            forces;" in body
    assert "patches         (rotor);" in body, "the torque is not taken on the impeller"
    assert "CofR" in body, "a moment with no centre of rotation"
    assert float(entry(body, "rhoInf")) == pytest.approx(module.RHO_KG_M3, rel=1e-06)
    assert entry(body, "rho") == "rhoInf"


def test_editing_the_rotor_numbers_moves_the_case(tmp_path):
    """"Copy it in, edit the two numbers, run it" is the contract. A number the case
    does not follow is decoration."""
    base, mod_base = case(ROTATING, tmp_path, "rot_base")
    fast, mod_fast = case(ROTATING, tmp_path, "rot_fast", ROTOR_RPM=2000.0)
    big, mod_big = case(ROTATING, tmp_path, "rot_big", VESSEL_RADIUS_M=0.25)

    # ROTOR_RPM reaches omega, the turbulence seeds and the blade-passing frequency
    assert mod_fast.omega_rad_per_s() == pytest.approx(2 * mod_base.omega_rad_per_s())
    assert mod_fast.blade_passing_hz() == pytest.approx(2 * mod_base.blade_passing_hz())
    assert mod_fast.k_seed() == pytest.approx(4 * mod_base.k_seed(), rel=1e-09)
    assert (float(entry((fast / "constant" / "MRFProperties").read_text(encoding="utf-8"),
                        "omega"))
            == pytest.approx(2 * float(entry(
                (base / "constant" / "MRFProperties").read_text(encoding="utf-8"),
                "omega"))))

    # VESSEL_RADIUS_M reaches the geometry, the cell size and the extrude depth
    assert mod_big.BLADE_RADIUS_M > mod_base.BLADE_RADIUS_M
    assert mod_big.ZONE_RADIUS_M > mod_base.ZONE_RADIUS_M
    assert mod_big.CELL_SIZE_M > mod_base.CELL_SIZE_M
    assert mod_big.THICK_M == mod_big.CELL_SIZE_M, "still one cell thick"
    # and the zone stays between the blade tip and the vessel wall, or the geometry
    # is not a rotating zone around an impeller any more
    for module in (mod_base, mod_big, mod_fast):
        assert module.BLADE_RADIUS_M < module.ZONE_RADIUS_M < module.VESSEL_RADIUS_M


@needs_gmsh
def test_the_rotor_mesh_has_the_two_cell_zones_mrf_names(tmp_path):
    """The only rotor test that needs the mesher, guarded like `test_cad_wiring.py`
    does. Everything above reads dictionaries; this one checks the zone the
    `MRFProperties` names is actually built -- a physical VOLUME in the msh, which is
    what `gmshToFoam` turns into a cellZone -- and that the case is really 2D."""
    import gmsh

    module = load(ROTATING, tmp_path, "rot_meshed", VESSEL_RADIUS_M=0.1)
    msh = tmp_path / "rot_meshed.msh"
    module.build_mesh(msh)
    assert msh.exists() and msh.stat().st_size > 0

    gmsh.initialize()
    try:
        gmsh.open(str(msh))
        groups, cells = {}, {}
        for dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, tag)
            count = 0
            for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
                _types, tags, _nodes = gmsh.model.mesh.getElements(dim, entity)
                count += sum(len(t) for t in tags)
            (cells if dim == 3 else groups)[name] = count

        # the patches the retyping afterwards addresses by name
        assert {"rotor", "stator", "frontAndBack"} == set(groups)
        # the cellZones: the one MRFProperties names, and the stationary remainder
        assert {"rotorZone", "statorZone"} == set(cells)
        assert cells["rotorZone"] > 0 and cells["statorZone"] > 0

        # one cell thick in z, which is what makes the empty z faces legitimate
        nodes = gmsh.model.mesh.getNodes()[1]
        zs = sorted(set(round(z, 9) for z in nodes[2::3]))
        assert zs == [pytest.approx(0.0), pytest.approx(module.THICK_M)]
        # and an exact identity that only holds for a one-cell-thick mesh: every cell
        # contributes exactly one front face and one back face
        assert groups["frontAndBack"] == 2 * (cells["rotorZone"] + cells["statorZone"])

        # the vessel is the radius that was asked for
        xs = nodes[0::3]
        assert max(xs) == pytest.approx(0.1, rel=1e-03)
        assert min(xs) == pytest.approx(-0.1, rel=1e-03)
    finally:
        gmsh.finalize()


# == porous.py: the Darcy-Forchheimer slab =====================================


def test_the_porous_case_carries_every_file_poroussimplefoam_needs(tmp_path):
    """`porousSimpleFoam` is a real solver in this OpenFOAM and not a historical name:
    `$FOAM_TUTORIALS/incompressible/porousSimpleFoam/straightDuctImplicit` ships with
    `application porousSimpleFoam;` and a `constant/porosityProperties` beside it."""
    out, _ = case(POROUS, tmp_path, "por0")
    written = {str(p.relative_to(out)).replace("\\", "/")
               for p in out.rglob("*") if p.is_file()}
    for required in (
        "0/U", "0/p", "0/k", "0/epsilon", "0/nut",
        "constant/porosityProperties", "constant/transportProperties",
        "constant/turbulenceProperties",
        "system/controlDict", "system/fvSchemes", "system/fvSolution",
    ):
        assert required in written, f"{required} is missing from the case"
    control = (out / "system" / "controlDict").read_text(encoding="utf-8")
    assert entry(control, "application") == "porousSimpleFoam"
    # one route, not two: the porosity is declared in porosityProperties, so an
    # fvOptions carrying a second explicitPorositySource would double the resistance
    assert "constant/fvOptions" not in written
    assert "system/fvOptions" not in written


def test_the_momentum_sink_carries_openfoams_factor_of_a_half_on_f(tmp_path):
    """The easiest thing here to get wrong, and it is a factor of two in the term that
    dominates everything but the slowest flows.

    OpenFOAM's `DarcyForchheimer.H` gives `S = -(mu*d + (rho*|U|/2)*f)*U`, and
    `DarcyForchheimer.C` builds the tensor with `forchCoeff.xx() = 0.5*fXYZ_...` while
    the Darcy tensor takes no factor. The prediction is recomputed here from that
    formula independently of the module, so the half is checked rather than copied.
    """
    module = load(POROUS, tmp_path, "por_half")
    mu = module.RHO_KG_M3 * module.NU_M2_S
    assert module.mu_pa_s() == pytest.approx(mu, rel=1e-12)

    for darcy, forchheimer, speed, thickness in ((1e08, 1e04, 1.0, 0.05),
                                                 (0.0, 5e03, 2.5, 0.2),
                                                 (3e07, 0.0, 0.3, 0.01)):
        expected = (mu * darcy * speed
                    + 0.5 * module.RHO_KG_M3 * forchheimer * speed * speed) * thickness
        assert module.predicted_drop_pa(speed, darcy, forchheimer, thickness) == \
            pytest.approx(expected, rel=1e-12)

    # and the half is not silently a one: a pure-Forchheimer slab's drop is exactly
    # half what it would be without the factor
    without_the_half = module.RHO_KG_M3 * 1e04 * 1.0 * 1.0 * 0.05
    assert module.predicted_drop_pa(1.0, 0.0, 1e04, 0.05) == \
        pytest.approx(0.5 * without_the_half, rel=1e-12)


def test_the_curve_fit_recovers_coefficients_it_was_not_told(tmp_path):
    """The sharpest test available: build a curve FROM a chosen d and f, hand only the
    curve to the fit, and require the original pair back.

    This is the identity the derivation rests on. dp/(L*U) = mu*d + 0.5*rho*f*U is
    linear in U, so a straight-line fit through points generated from any (d, f) must
    return that pair exactly, up to floating point. A conserved quantity whose exact
    value is known is the cheapest instrument a case has.
    """
    module = load(POROUS, tmp_path, "por_fit")
    for darcy, forchheimer in ((1.5e08, 2.2e04), (4.0e06, 1.0e02), (9.9e09, 7.5e03)):
        thickness = 0.037
        curve = [(speed, module.predicted_drop_pa(speed, darcy, forchheimer, thickness))
                 for speed in (0.4, 0.9, 1.7, 3.1)]
        got_d, got_f, how = module.coefficients_from_curve(curve, thickness)
        assert got_d == pytest.approx(darcy, rel=1e-06), how
        assert got_f == pytest.approx(forchheimer, rel=1e-06), how
        assert "fit" in how and "intercept" in how and "slope" in how


def test_ergun_and_the_curve_fit_are_two_routes_to_one_answer(tmp_path):
    """The default curve is Ergun's own prediction rounded to six figures, so fitting it
    has to give Ergun's coefficients back. That is what makes the shipped numbers
    traceable arithmetic rather than invention, and it cross-checks both routes at
    once: the factor of 2 on Ergun's 1.75 is the same half OpenFOAM puts on `f`, running
    the other way, and getting it wrong here would show up as a factor of two."""
    module = load(POROUS, tmp_path, "por_ergun")
    ergun_d, ergun_f = module.coefficients_from_ergun()
    fitted_d, fitted_f, _how = module.coefficients_from_curve()

    # six figures in the shipped curve, so agreement to about a part in 10^5
    assert fitted_d == pytest.approx(ergun_d, rel=1e-04)
    assert fitted_f == pytest.approx(ergun_f, rel=1e-04)

    # Ergun's own algebra, recomputed here from the correlation rather than copied
    voids = module.BED_VOID_FRACTION
    diameter = module.BED_PARTICLE_DIAMETER_M
    solid = 1.0 - voids
    assert ergun_d == pytest.approx(
        150.0 * solid ** 2 / (voids ** 3 * diameter ** 2), rel=1e-12)
    assert ergun_f == pytest.approx(
        2.0 * 1.75 * solid / (voids ** 3 * diameter), rel=1e-12)

    # the bed's geometry really drives it: a coarser bed resists less, and d falls as
    # the square of the particle size while f falls only as the first power
    coarse_d, coarse_f = module.coefficients_from_ergun(particle_diameter=2 * diameter)
    assert coarse_d == pytest.approx(ergun_d / 4.0, rel=1e-09)
    assert coarse_f == pytest.approx(ergun_f / 2.0, rel=1e-09)
    # and a refusal rather than a nonsense number for an impossible void fraction
    for bad in (0.0, 1.0, 1.5, -0.2):
        with pytest.raises(SystemExit):
            module.coefficients_from_ergun(void_fraction=bad)


def test_one_measured_point_says_what_it_cannot_determine(tmp_path):
    """Two unknowns and one measurement, which no arithmetic can rescue. The all-Darcy
    limit is returned and `how` says so in words that reach the written dictionary --
    a silent choice between the two limits is the failure."""
    module = load(POROUS, tmp_path, "por_one")
    darcy, forchheimer, how = module.coefficients_from_curve([(1.0, 500.0)], 0.05)
    assert forchheimer == 0.0, "a split was invented from one point"
    assert darcy > 0.0
    # the all-Darcy limit reproduces the one point it was given, exactly
    assert module.predicted_drop_pa(1.0, darcy, forchheimer, 0.05) == \
        pytest.approx(500.0, rel=1e-09)
    lower = how.lower()
    assert "one measured point" in lower
    assert "undetermined" in lower and "darcy" in lower

    # a measurement that is not a measurement is refused rather than reinterpreted: a
    # drop entered negative because it is a loss would give a negative d, which
    # OpenFOAM reads as a cross-flow multiplier rather than as an error
    for bad in ([(1.0, -500.0)], [(-1.0, 500.0)], [(0.0, 500.0)], []):
        with pytest.raises(SystemExit):
            module.coefficients_from_curve(bad, 0.05)


def test_a_curve_that_implies_a_negative_d_is_clamped_and_reported(tmp_path):
    """Real curves have scatter, and `dp/(L*U)` is most sensitive at the low-velocity
    end where the intercept is read. A fitted intercept below zero would mean a negative
    `d`, which OpenFOAM would silently read as a cross-flow multiplier. "This medium's
    loss is entirely inertial over the measured range" is a real answer; a negative
    Darcy coefficient is not."""
    module = load(POROUS, tmp_path, "por_clamp")
    # a purely quadratic curve with the low point nudged up, so the straight line
    # through dp/(L*U) extrapolates below zero at U = 0
    thickness = 0.05
    curve = [(1.0, 250.0), (2.0, 400.0), (3.0, 900.0), (4.0, 1600.0)]
    darcy, forchheimer, how = module.coefficients_from_curve(curve, thickness)
    assert darcy >= 0.0, "a negative d reached the dictionary"
    assert forchheimer > 0.0
    if darcy == 0.0:
        assert "negative" in how.lower() and "clamped" in how.lower()


def test_the_derived_coefficients_reach_the_dictionary_with_their_arithmetic(tmp_path):
    """A template that demands `d` and `f` has handed the user back their own question.
    These have to be in the written file as derived numbers, with the derivation and the
    residual against every measured point beside them -- a residual a reader can see is
    worth more than a derivation they have to trust."""
    out, module = case(POROUS, tmp_path, "por_dict")
    text = (out / "constant" / "porosityProperties").read_text(encoding="utf-8")
    darcy, forchheimer, _how = module.coefficients_from_curve()

    assert "object      porosityProperties" in text
    assert entry(text, "type") == "DarcyForchheimer"
    assert entry(text, "cellZone") == "porousZone"

    written_d = [float(v) for v in entry(text, "d").strip("()").split()]
    written_f = [float(v) for v in entry(text, "f").strip("()").split()]
    assert written_d[0] == pytest.approx(darcy, rel=1e-05)
    assert written_f[0] == pytest.approx(forchheimer, rel=1e-05)

    # the arithmetic, in the file: the sink, the reduction, the fluid and the thickness
    assert "mu*d" in text and "0.5*rho*f" in text
    assert "dp/(L*U)" in text, "the reduction that makes the fit linear is not shown"
    assert f"{module.SLAB_THICKNESS_M:g}" in text, "the thickness d and f are per is absent"
    assert "1/m2" in text and "1/m" in text, "the coefficients' units are not stated"
    # the residual table: every measured point, measured against predicted
    for speed, drop in module.PRESSURE_DROP_CURVE:
        assert f"{drop:.4f}" in text, f"the {speed} m/s measurement is not in the file"
    # and a coordinate system, because e1 is the through-flow direction
    assert "coordinateSystem" in text
    assert "e1" in text and "e2" in text


def test_the_cross_flow_components_are_a_multiplier_and_the_file_says_so(tmp_path):
    """OpenFOAM's convention here is unusual and silently wrong if misread: a NEGATIVE
    component of `d` or `f` is a multiplier of the largest component, applied by
    `porosityModel.C`'s `adjustNegativeResistance`, not a negative resistance. A slab
    that resists across its face only as much as through it lets the flow turn and
    sneak sideways, and the answer is a converged, plausible, wrong pressure drop."""
    out, module = case(POROUS, tmp_path, "por_cross")
    text = (out / "constant" / "porosityProperties").read_text(encoding="utf-8")
    for key in ("d", "f"):
        components = [float(v) for v in entry(text, key).strip("()").split()]
        assert len(components) == 3, f"{key} is not a vector"
        assert components[1] == pytest.approx(module.CROSS_FLOW_MULTIPLIER)
        assert components[2] == pytest.approx(module.CROSS_FLOW_MULTIPLIER)
    assert module.CROSS_FLOW_MULTIPLIER < 0
    lower = text.lower()
    assert "multiplier" in lower, "nothing says a negative component is a multiplier"
    assert "not a negative resistance" in lower or "negative component is not" in lower


def test_the_pressure_is_kinematic_and_both_units_are_given(tmp_path):
    """This template's factor-of-rho, which is the porous equivalent of rad/s against
    rpm. `porousSimpleFoam` solves p/rho in m2/s2, so a 503 Pa measurement compared
    against a field reading 418 is exact agreement, not a 20% error. Both numbers have
    to be in front of the reader, and they have to differ by exactly rho."""
    out, module = case(POROUS, tmp_path, "por_units")
    p = (out / "0" / "p").read_text(encoding="utf-8")
    assert entry(p, "dimensions") == "[0 2 -2 0 0 0 0]"
    assert "m2/s2" in p and "pascals" in p.lower()

    darcy, forchheimer, _how = module.coefficients_from_curve()
    speed = module.design_velocity_m_s()
    in_pa = module.predicted_drop_pa(speed, darcy, forchheimer)
    in_kinematic = in_pa / module.RHO_KG_M3

    for text, where in ((out / "constant" / "porosityProperties").read_text(
            encoding="utf-8"), "porosityProperties"), (module.summary(), "the summary"):
        assert f"{in_pa:.6g}" in text, f"{where} lacks the drop in Pa"
        assert f"{in_kinematic:.6g}" in text, f"{where} lacks the drop in m2/s2"
        assert "m2/s2" in text and "Pa" in text, f"{where} does not name both units"
    # the velocity the case runs at is one the curve was measured at, so the run has
    # something to be wrong against
    assert speed in [s for s, _drop in module.PRESSURE_DROP_CURVE]
    u = (out / "0" / "U").read_text(encoding="utf-8")
    assert f"({speed:g} 0 0)" in u


def test_the_porous_case_has_an_outlet_so_it_needs_no_reference_cell(tmp_path):
    """The opposite of the rotor's problem, and getting it backwards either way is a
    defect: here the outlet's `fixedValue` pressure pins the level, so a `pRefCell`
    would be a second, conflicting statement of it."""
    out, _ = case(POROUS, tmp_path, "por_pref")
    solution = (out / "system" / "fvSolution").read_text(encoding="utf-8")
    assert "pRefCell" not in solution.replace("No pRefCell", "")
    # the implicit porous source is a large diagonal term, so U lags p without this
    assert int(entry(solution, "nUCorrectors")) >= 2

    p = (out / "0" / "p").read_text(encoding="utf-8")
    assert "fixedValue" in patch_block(p, "outlet")
    assert "zeroGradient" in patch_block(p, "inlet"), (
        "a fixed inlet pressure and a fixed outlet pressure over-determine the duct"
    )
    # the slab's wake can recirculate, so the outlet admits inflow rather than
    # feeding the interior value back in
    u = (out / "0" / "U").read_text(encoding="utf-8")
    assert "inletOutlet" in patch_block(u, "outlet")
    assert "fixedValue" in patch_block(u, "inlet")
    assert "noSlip" in patch_block(u, "walls")


def test_the_porous_case_is_steady_and_bounded(tmp_path):
    """Same pair as the rotor's: `steadyState` ddt needs the `bounded` prefix on the
    convection terms or `k` and `epsilon` can go negative early, and `deltaT` counts
    iterations rather than seconds."""
    out, _ = case(POROUS, tmp_path, "por_numerics")
    schemes = (out / "system" / "fvSchemes").read_text(encoding="utf-8")
    assert entry(schemes, "default") == "steadyState"
    assert_convection_is_bounded(schemes)
    control = (out / "system" / "controlDict").read_text(encoding="utf-8")
    assert "ITERATIONS" in control
    assert entry(control, "startFrom") == "latestTime"
    # the two ends' pressure, which is what makes the drop readable off a run
    assert control.count("surfaceFieldValue") == 2
    assert "areaAverage" in control


def test_editing_the_porous_numbers_moves_the_case(tmp_path):
    """The two marked numbers are the measurement, not the geometry, because `d` and
    `f` are what the case is about and the measurement is where they come from."""
    base, mod_base = case(POROUS, tmp_path, "por_base")
    thick, mod_thick = case(POROUS, tmp_path, "por_thick", SLAB_THICKNESS_M=0.1)

    # a slab twice as thick, measured at the same drops, is half as resistive per metre
    base_d, base_f, _ = mod_base.coefficients_from_curve()
    thick_d, thick_f, _ = mod_thick.coefficients_from_curve()
    assert thick_d == pytest.approx(base_d / 2.0, rel=1e-06)
    assert thick_f == pytest.approx(base_f / 2.0, rel=1e-06)
    assert mod_thick.LENGTH_M > mod_base.LENGTH_M, "the duct did not follow the slab"
    assert ((thick / "constant" / "porosityProperties").read_text(encoding="utf-8")
            != (base / "constant" / "porosityProperties").read_text(encoding="utf-8"))

    # and a different curve gives different coefficients, with the dictionary following
    steep, mod_steep = case(POROUS, tmp_path, "por_steep",
                            PRESSURE_DROP_CURVE=[(1.0, 1000.0), (2.0, 4000.0)])
    steep_d, steep_f, _ = mod_steep.coefficients_from_curve()
    assert steep_f > 0 and steep_d >= 0
    assert ((steep / "constant" / "porosityProperties").read_text(encoding="utf-8")
            != (base / "constant" / "porosityProperties").read_text(encoding="utf-8"))
    # the fit reproduces the points it was given, which is the check that matters
    for speed, drop in mod_steep.PRESSURE_DROP_CURVE:
        assert mod_steep.predicted_drop_pa(speed, steep_d, steep_f) == \
            pytest.approx(drop, rel=1e-06)


@needs_gmsh
def test_the_porous_mesh_has_the_cellzone_the_model_names(tmp_path):
    """The zone `porosityProperties` names has to exist in `constant/polyMesh` or the
    case fails at the first solver step. A physical VOLUME in the msh is what
    `gmshToFoam` turns into a cellZone; a physical SURFACE becomes a patch.

    The cell counts here are exact arithmetic rather than approximate: the mesh is
    uniform hexahedra, so the slab holds (thickness/cell) x CELLS_ACROSS cells and
    nothing else is possible.
    """
    import gmsh

    module = load(POROUS, tmp_path, "por_meshed")
    msh = tmp_path / "por_meshed.msh"
    module.build_mesh(msh)
    assert msh.exists() and msh.stat().st_size > 0

    gmsh.initialize()
    try:
        gmsh.open(str(msh))
        groups, cells = {}, {}
        for dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, tag)
            count = 0
            for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
                _types, tags, _nodes = gmsh.model.mesh.getElements(dim, entity)
                count += sum(len(t) for t in tags)
            (cells if dim == 3 else groups)[name] = count

        assert {"inlet", "outlet", "walls", "frontAndBack"} == set(groups)
        assert {"porousZone", "fluid"} == set(cells)

        across = module.CELLS_ACROSS
        along_slab = round(module.SLAB_THICKNESS_M / module.CELL_SIZE_M)
        along_clear = round((module.UPSTREAM_M + module.DOWNSTREAM_M) / module.CELL_SIZE_M)
        assert cells["porousZone"] == along_slab * across
        assert cells["fluid"] == along_clear * across
        # the duct ends are one column of cells tall, and the walls run its whole length
        assert groups["inlet"] == across
        assert groups["outlet"] == across
        assert groups["walls"] == 2 * (along_slab + along_clear)
        # the one-cell-thick identity: one front face and one back face per cell
        assert groups["frontAndBack"] == 2 * (cells["porousZone"] + cells["fluid"])

        nodes = gmsh.model.mesh.getNodes()[1]
        zs = sorted(set(round(z, 9) for z in nodes[2::3]))
        assert zs == [pytest.approx(0.0), pytest.approx(module.THICK_M)]
        xs, ys = nodes[0::3], nodes[1::3]
        assert max(xs) == pytest.approx(module.LENGTH_M, rel=1e-06)
        assert max(ys) == pytest.approx(module.HEIGHT_M, rel=1e-06)
    finally:
        gmsh.finalize()


# == both templates stay inside the image, and inside the contract =============


@pytest.mark.parametrize("template", [ROTATING, POROUS], ids=["rotating", "porous"])
def test_the_template_imports_only_the_standard_library_and_gmsh(template):
    """The same allowlist `test_toolbox_templates.py` applies to every template,
    asserted here as well because these two are tempted by it: a curve fit wants numpy
    and scipy, and numpy is on the instance while scipy is not -- and neither belongs in
    a template, which is copied into a case directory and has to run as a single file.
    `statistics.linear_regression` is the standard library's own answer."""
    allowed = set(sys.stdlib_module_names) | {"gmsh", "__future__"}
    text = template.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [(node.module or "").split(".")[0]]
        else:
            continue
        for root in roots:
            assert root in allowed, f"{template.name} imports {root}"
    # gmsh is imported inside build_mesh, not at module scope, which is what lets these
    # files be loaded and their dictionaries checked on a machine with no mesher
    assert "import gmsh" not in text.split("def build_mesh", 1)[0]


@pytest.mark.parametrize("template", [ROTATING, POROUS], ids=["rotating", "porous"])
def test_the_template_says_its_case_is_meant_to_solve(template):
    """`duct2d.py` and `body_in_box.py` write stubs they expect to be overwritten. These
    do not, and a reader who assumes otherwise will overwrite a working case."""
    doc = ast.get_docstring(ast.parse(template.read_text(encoding="utf-8")))
    assert doc
    assert "stub" in doc, "the docstring does not distinguish itself from the stub ones"
    assert "solve" in doc
    index = (TOOLBOX / "README.md").read_text(encoding="utf-8")
    row = next(line for line in index.splitlines()
               if f"templates/{template.name}" in line)
    assert "stub" in row and "solve" in row, (
        "the index row does not say this case is meant to be solved"
    )


@pytest.mark.parametrize("template", [ROTATING, POROUS], ids=["rotating", "porous"])
def test_the_template_states_where_it_read_the_openfoam_syntax(template):
    """Both of these write dictionary keys whose spelling has moved between OpenFOAM
    versions and between the two forks, and neither was verified by running the solver.
    So each says where its syntax was read and gives the one command that confirms it on
    the image, because a fact with no provenance is a fact nobody can re-check."""
    text = template.read_text(encoding="utf-8")
    assert "FOAM_TUTORIALS" in text, "no shipped case is cited for the syntax"
    assert "foamInfo" in text, "no on-image command to confirm the keys"
    # the version the syntax was read against, said rather than implied
    assert "v2512" in text and "v2606" in text, (
        "the template does not say which version its syntax came from against which "
        "version the image runs"
    )
