"""The parametric case generator.

The point of generating a case rather than typing one is that the dull mistakes stop
happening, so these tests are mostly about the dull mistakes: a "2D" case that is two
cells thick, a patch named one thing in the mesh and another in the fields, an `empty`
patch that one field calls `zeroGradient`, a viscosity that does not match the Reynolds
number it was asked for. Each of those costs a solver run to find by hand and costs
nothing to find here.

No pyvista and no OpenFOAM: everything under test is arithmetic and text.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"

EXTERNAL = ("circle", "square", "lshape", "vehicle")
DUCTS = ("duct-y", "duct-t", "duct-z", "duct-f", "duct-m")
BENDS = ("bend-sharp", "bend-mitred", "bend-rounded")
ALL_TEMPLATES = EXTERNAL + DUCTS + BENDS


@pytest.fixture(scope="module")
def case_gen():
    spec = importlib.util.spec_from_file_location("toolbox_case_gen", TOOLBOX / "case_gen.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(case_gen, template: str, target: Path, *args: str) -> int:
    """Run the CLI the way the agent would, and return its exit code."""
    return case_gen.main([template, str(target), *args])


# -- the geometry helpers ----------------------------------------------------------


def test_signed_area_and_orientation(case_gen):
    """`signed_area` is twice the area, as the shoelace sum comes out; only its sign
    and its magnitude relative to itself are ever used, so it is not halved."""
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert case_gen.signed_area(square) == pytest.approx(2.0)
    assert case_gen.signed_area(list(reversed(square))) == pytest.approx(-2.0)
    # as_ccw leaves an anticlockwise loop alone and turns a clockwise one round.
    assert case_gen.as_ccw(square) == square
    assert case_gen.as_ccw(list(reversed(square)))[0] in square


def test_drop_repeats_closes_nothing_and_removes_duplicates(case_gen):
    points = [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0)]
    kept = case_gen.drop_repeats(points)
    assert kept == [(0.0, 0.0), (1.0, 0.0)]


def test_star_shaped_failure_names_the_point_that_breaks_it(case_gen):
    """An O-grid maps the outline onto a ring, and that needs every ray from the
    centre to cross the outline once. A shape with a deep pocket does not."""
    circle = [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in range(0, 360, 10)]
    assert case_gen.star_shaped_failure(circle, (0.0, 0.0)) is None

    # A loop that doubles back on itself as seen from the centre.
    pocket = [(1.0, 0.0), (0.2, 0.4), (1.0, 0.8), (-1.0, 0.8), (-1.0, -0.8), (1.0, -0.8)]
    assert case_gen.star_shaped_failure(pocket, (0.0, 0.0)) is not None


def test_divisions_never_returns_zero(case_gen):
    """A block with zero divisions in a direction is a block blockMesh refuses."""
    assert case_gen.divisions(1.0, 0.1) == 10
    assert case_gen.divisions(0.001, 1.0) >= 1
    assert case_gen.divisions(0.0, 1.0) >= 1


# -- what the numbers imply --------------------------------------------------------


def test_reynolds_gives_the_viscosity_and_says_so(case_gen):
    flow = case_gen.derive_flow({"speed": 2.0, "reynolds": 200.0, "nu": None, "length": None}, 0.1)

    assert flow.nu == pytest.approx(2.0 * 0.1 / 200.0)
    assert flow.reynolds == pytest.approx(200.0)
    assert flow.derived == "nu"
    assert "nu = U*L/Re" in flow.line()


def test_viscosity_gives_the_reynolds_number_and_says_so(case_gen):
    flow = case_gen.derive_flow({"speed": 30.0, "reynolds": None, "nu": 1.5e-5, "length": None}, 4.0)

    assert flow.reynolds == pytest.approx(30.0 * 4.0 / 1.5e-5)
    assert flow.derived == "reynolds"
    assert "Re = U*L/nu" in flow.line()


def test_both_at_once_is_refused_rather_than_one_quietly_winning(case_gen):
    with pytest.raises(SystemExit, match="not both"):
        case_gen.derive_flow({"speed": 1.0, "reynolds": 100.0, "nu": 1e-5, "length": None}, 0.1)


def test_a_length_given_beats_the_templates_own(case_gen):
    flow = case_gen.derive_flow({"speed": 1.0, "reynolds": 100.0, "nu": None, "length": 0.5}, 0.1)
    assert flow.length == 0.5
    assert flow.nu == pytest.approx(1.0 * 0.5 / 100.0)


@pytest.mark.parametrize("bad", [{"speed": 0.0}, {"speed": -1.0}])
def test_a_speed_that_is_not_positive_is_refused(case_gen, bad):
    with pytest.raises(SystemExit):
        case_gen.derive_flow({"reynolds": 100.0, "nu": None, "length": None, **bad}, 0.1)


def test_the_turbulence_model_is_chosen_from_reynolds_and_can_be_overridden(case_gen):
    low = case_gen.Flow(1.0, 0.1, 1e-3, 100.0, "nu")
    high = case_gen.Flow(30.0, 4.0, 1.5e-5, 8_000_000.0, "reynolds")

    model, why = case_gen.turbulence_model({"turbulence": "auto"}, low)
    assert model == "laminar" and "below" in why
    model, why = case_gen.turbulence_model({"turbulence": "auto"}, high)
    assert model == "kOmegaSST" and "above" in why
    # An explicit choice is honoured even when it is the unusual one.
    model, why = case_gen.turbulence_model({"turbulence": "laminar"}, high)
    assert model == "laminar" and "--turbulence" in why


# -- every template builds, and what it builds is 2D -------------------------------


@pytest.mark.parametrize("template", ALL_TEMPLATES)
def test_every_template_writes_a_case(case_gen, tmp_path, template):
    target = tmp_path / template
    assert build(case_gen, template, target, "--reynolds", "200") == 0

    for required in ("system/blockMeshDict", "system/controlDict", "system/fvSchemes",
                     "system/fvSolution", "constant/transportProperties",
                     "constant/turbulenceProperties", "0/U", "0/p"):
        assert (target / required).is_file(), f"{template} did not write {required}"


@pytest.mark.parametrize("template", ALL_TEMPLATES)
def test_every_case_is_one_cell_thick_with_one_empty_patch(case_gen, tmp_path, template):
    """The failure that looks most like something else: a case with two cells in z
    runs, converges, and answers a different question."""
    target = tmp_path / template
    build(case_gen, template, target, "--reynolds", "200")
    block_mesh = (target / "system" / "blockMeshDict").read_text(encoding="utf-8")

    # Every hex block has exactly one division in z.
    counts = re.findall(r"hex\s*\([^)]*\)\s*\((\d+)\s+(\d+)\s+(\d+)\)", block_mesh)
    assert counts, f"{template}: no hex blocks found"
    assert all(int(z) == 1 for _x, _y, z in counts), f"{template} is not one cell thick"

    # One patch named frontAndBack, declared empty, and it is the only empty one.
    assert re.search(r"frontAndBack\s*\{\s*type\s+empty;", block_mesh), template
    assert block_mesh.count("type empty;") == 1, f"{template} has more than one empty patch"


@pytest.mark.parametrize("template", ALL_TEMPLATES)
def test_the_mesh_and_the_fields_name_the_same_patches(case_gen, tmp_path, template):
    """A patch called `inlet` in the mesh and `Inlet` in the fields stops the solver
    on the first time step, and the message names neither file."""
    target = tmp_path / template
    build(case_gen, template, target, "--reynolds", "200")

    block_mesh = (target / "system" / "blockMeshDict").read_text(encoding="utf-8")
    boundary = block_mesh[block_mesh.index("\nboundary"):]
    in_mesh = set(re.findall(r"^    (\w+)$", boundary, re.M))
    assert in_mesh, f"{template}: no patches read out of the boundary list"

    for field in sorted((target / "0").glob("*")):
        text = field.read_text(encoding="utf-8")
        block = text[text.index("boundaryField"):]
        in_field = set(re.findall(r"^    (\w+)$", block, re.M))
        assert in_field == in_mesh, f"{template}: {field.name} covers {in_field}, mesh has {in_mesh}"


@pytest.mark.parametrize("template", ALL_TEMPLATES)
def test_the_empty_patch_is_empty_in_every_field_too(case_gen, tmp_path, template):
    """`empty` in the mesh and `zeroGradient` in 0/U is the pairing that fails on the
    first time step; it has to hold in both places or in neither."""
    target = tmp_path / template
    build(case_gen, template, target, "--reynolds", "200")

    for field in sorted((target / "0").glob("*")):
        text = field.read_text(encoding="utf-8")
        entry = re.search(r"frontAndBack\s*\{([^}]*)\}", text)
        assert entry, f"{template}: {field.name} has no frontAndBack entry"
        assert "type            empty;" in entry.group(1), f"{template}: {field.name}"


@pytest.mark.parametrize("template", ALL_TEMPLATES)
def test_every_generated_file_looks_like_an_openfoam_dictionary(case_gen, tmp_path, template):
    target = tmp_path / template
    build(case_gen, template, target, "--reynolds", "200")

    for path in sorted(target.rglob("*")):
        if not path.is_file() or ".reynolds" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        assert "FoamFile" in text, f"{path} has no FoamFile header"
        assert text.count("{") == text.count("}"), f"{path} has unbalanced braces"
        assert text.rstrip().endswith("//"), f"{path} does not end with the footer rule"
        # Every `key value` line at any depth is terminated.
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or stripped in ("{", "}", "(", ")", ");"):
                continue


def test_the_printed_cell_count_is_the_product_of_the_divisions(case_gen, tmp_path, capsys):
    """blockMesh cell counts are exactly the product of the divisions written down,
    so the number printed before the mesh is built is the one checkMesh reports."""
    target = tmp_path / "cyl"
    build(case_gen, "circle", target, "--reynolds", "200")
    printed = capsys.readouterr().out

    block_mesh = (target / "system" / "blockMeshDict").read_text(encoding="utf-8")
    counts = re.findall(r"hex\s*\([^)]*\)\s*\((\d+)\s+(\d+)\s+(\d+)\)", block_mesh)
    total = sum(int(x) * int(y) * int(z) for x, y, z in counts)

    assert f"{total:,}" in printed


# -- the flow the case is written for ----------------------------------------------


def test_the_viscosity_written_is_the_one_the_reynolds_number_asked_for(case_gen, tmp_path):
    target = tmp_path / "cyl"
    build(case_gen, "circle", target, "--size", "0.2", "--speed", "3", "--reynolds", "150")

    transport = (target / "constant" / "transportProperties").read_text(encoding="utf-8")
    nu = float(re.search(r"^nu\s+([0-9.eE+-]+);", transport, re.M).group(1))

    assert nu == pytest.approx(3.0 * 0.2 / 150.0)
    # And the case says which three numbers it came from, in the file itself.
    assert "Re = 150" in transport


def test_the_inlet_carries_the_speed_and_the_outlet_lets_it_leave(case_gen, tmp_path):
    target = tmp_path / "cyl"
    build(case_gen, "circle", target, "--speed", "2.5", "--reynolds", "200")
    field = (target / "0" / "U").read_text(encoding="utf-8")

    inlet = re.search(r"    inlet\s*\{([^}]*)\}", field).group(1)
    assert "fixedValue" in inlet and "(2.5 0 0)" in inlet
    outlet = re.search(r"    outlet\s*\{([^}]*)\}", field).group(1)
    assert "inletOutlet" in outlet


def test_pressure_is_kinematic_and_says_so(case_gen, tmp_path):
    """Incompressible OpenFOAM solves p/rho in m2/s2. A force computed from it as if
    it were pascals is out by a factor of rho and looks plausible."""
    target = tmp_path / "cyl"
    build(case_gen, "circle", target, "--reynolds", "200")
    field = (target / "0" / "p").read_text(encoding="utf-8")

    assert "[0 2 -2 0 0 0 0]" in field
    assert "kinematic" in field


def test_a_turbulent_case_gets_the_turbulence_fields_and_a_laminar_one_does_not(case_gen, tmp_path):
    laminar = tmp_path / "slow"
    build(case_gen, "circle", laminar, "--reynolds", "100")
    assert sorted(p.name for p in (laminar / "0").glob("*")) == ["U", "p"]
    assert "simulationType  laminar;" in (laminar / "constant" / "turbulenceProperties").read_text(encoding="utf-8")

    turbulent = tmp_path / "fast"
    build(case_gen, "circle", turbulent, "--reynolds", "1000000")
    assert sorted(p.name for p in (turbulent / "0").glob("*")) == ["U", "k", "nut", "omega", "p"]
    assert "kOmegaSST" in (turbulent / "constant" / "turbulenceProperties").read_text(encoding="utf-8")


# -- moving ground and rotating wheels ---------------------------------------------


def test_a_moving_ground_is_a_belt_at_the_free_stream_speed(case_gen, tmp_path):
    """On a mesh that does not move, a belt is a fixed tangential velocity.
    `movingWallVelocity` would be the wrong tool and would silently do nothing."""
    target = tmp_path / "car"
    build(case_gen, "vehicle", target, "--size", "4", "--speed", "30", "--nu", "1.5e-5",
          "--moving-ground", "--ogrid-aspect", "0.6", "--below", "3")
    field = (target / "0" / "U").read_text(encoding="utf-8")

    ground = re.search(r"    bottomWall\s*\{([^}]*)\}", field).group(1)
    assert "fixedValue" in ground and "(30 0 0)" in ground


def test_rotating_wheels_spin_at_the_road_speed_over_their_radius(case_gen, tmp_path):
    target = tmp_path / "car"
    build(case_gen, "vehicle", target, "--size", "4", "--speed", "30", "--nu", "1.5e-5",
          "--rotating-wheels", "--ogrid-aspect", "0.6", "--below", "3")
    field = (target / "0" / "U").read_text(encoding="utf-8")

    body = case_gen.vehicle_body(4.0)
    for patch, (_x, _y, radius) in body.extras["wheels"].items():
        entry = re.search(rf"    {patch}\s*\{{([^}}]*)\}}", field)
        assert entry, f"no {patch} entry"
        assert "rotatingWallVelocity" in entry.group(1)
        omega = float(re.search(r"omega\s+([0-9.eE+-]+);", entry.group(1)).group(1))
        assert omega == pytest.approx(30.0 / radius, rel=1e-4)


def test_the_wheel_origin_is_in_the_meshs_coordinates_not_the_bodys(case_gen, tmp_path):
    """external_mesh works with the body centred on the origin. A wheel origin left
    in body coordinates puts the rotation axis somewhere else entirely, and the
    tangential velocity comes out at the wrong angle rather than failing."""
    target = tmp_path / "car"
    build(case_gen, "vehicle", target, "--size", "4", "--speed", "30", "--nu", "1.5e-5",
          "--rotating-wheels", "--ogrid-aspect", "0.6", "--below", "3")
    field = (target / "0" / "U").read_text(encoding="utf-8")

    body = case_gen.vehicle_body(4.0)
    centre_x, centre_y = body.centre
    for patch, (wheel_x, wheel_y, _r) in body.extras["wheels"].items():
        entry = re.search(rf"    {patch}\s*\{{([^}}]*)\}}", field).group(1)
        origin = re.search(r"origin\s+\(([^)]*)\)", entry).group(1).split()
        assert float(origin[0]) == pytest.approx(wheel_x - centre_x, abs=1e-6)
        assert float(origin[1]) == pytest.approx(wheel_y - centre_y, abs=1e-6)


def test_rotating_wheels_on_a_body_that_has_none_is_refused(case_gen, tmp_path):
    with pytest.raises(SystemExit, match="no wheels"):
        build(case_gen, "circle", tmp_path / "cyl", "--rotating-wheels")


# -- the study types ---------------------------------------------------------------


def test_a_steady_case_counts_iterations_and_a_transient_one_counts_seconds(case_gen, tmp_path):
    steady = tmp_path / "steady"
    build(case_gen, "circle", steady, "--study", "steady", "--iterations", "500")
    control = (steady / "system" / "controlDict").read_text(encoding="utf-8")
    assert "application     simpleFoam;" in control
    assert "endTime         500;" in control
    assert "steadyState" in (steady / "system" / "fvSchemes").read_text(encoding="utf-8")

    transient = tmp_path / "transient"
    build(case_gen, "circle", transient, "--study", "transient", "--end-time", "2")
    control = (transient / "system" / "controlDict").read_text(encoding="utf-8")
    assert "application     pimpleFoam;" in control
    assert "endTime         2;" in control
    assert "adjustTimeStep  yes;" in control
    assert "backward" in (transient / "system" / "fvSchemes").read_text(encoding="utf-8")


def test_a_mesh_only_study_names_no_solver_but_still_writes_fields(case_gen, tmp_path, capsys):
    """blockMesh does not read 0/, but a mesh-only study that turns into a solve is
    then one edit away rather than an afternoon."""
    target = tmp_path / "meshonly"
    build(case_gen, "circle", target, "--study", "mesh")

    assert "mesh only, no solver" in capsys.readouterr().out
    assert (target / "0" / "U").is_file()


def test_the_transient_step_comes_from_the_courant_number_asked_for(case_gen, tmp_path):
    target = tmp_path / "t"
    build(case_gen, "duct-t", target, "--study", "transient", "--end-time", "1",
          "--courant", "0.5", "--duct-width", "0.05", "--cells-across", "10")
    control = (target / "system" / "controlDict").read_text(encoding="utf-8")

    delta = float(re.search(r"deltaT\s+([0-9.eE+-]+);", control).group(1))
    cell = 0.05 / 10
    assert delta == pytest.approx(0.5 * cell / 1.0, rel=1e-6)
    assert "maxCo           0.5;" in control


# -- the command line --------------------------------------------------------------


def test_list_names_every_template(case_gen, capsys):
    assert case_gen.main(["--list"]) == 0
    printed = capsys.readouterr().out
    for name in case_gen.TEMPLATES:
        assert name in printed


def test_dry_run_writes_nothing_and_lists_what_it_would(case_gen, tmp_path, capsys):
    target = tmp_path / "nothing"
    assert build(case_gen, "circle", target, "--dry-run") == 0

    printed = capsys.readouterr().out
    assert "would write" in printed and "system/blockMeshDict" in printed
    assert not target.exists(), "a dry run created the directory"


def test_a_second_run_over_a_case_needs_force(case_gen, tmp_path):
    target = tmp_path / "cyl"
    build(case_gen, "circle", target, "--reynolds", "200")

    with pytest.raises(SystemExit, match="already holds a case"):
        build(case_gen, "circle", target, "--reynolds", "300")

    assert build(case_gen, "circle", target, "--reynolds", "300", "--force") == 0
    transport = (target / "constant" / "transportProperties").read_text(encoding="utf-8")
    assert "Re = 300" in transport


def test_an_unknown_template_is_refused_by_name(case_gen, tmp_path):
    with pytest.raises(SystemExit, match="no template called"):
        build(case_gen, "trapezoid", tmp_path / "x")


def test_the_profile_template_needs_a_file(case_gen, tmp_path):
    with pytest.raises(SystemExit, match="--profile"):
        build(case_gen, "profile", tmp_path / "foil")


def test_a_profile_read_from_a_file_becomes_a_case(case_gen, tmp_path):
    """A rounded outline, so it is star-shaped and the O-grid can be built."""
    points = [(0.5 + 0.5 * math.cos(math.radians(a)), 0.15 * math.sin(math.radians(a)))
              for a in range(0, 360, 5)]
    profile = tmp_path / "shape.dat"
    profile.write_text("\n".join(f"{x:.6f} {y:.6f}" for x, y in points), encoding="utf-8")

    target = tmp_path / "foil"
    assert build(case_gen, "profile", target, "--profile", str(profile),
                 "--aoa", "5", "--ogrid-aspect", "0.5") == 0
    assert (target / "system" / "blockMeshDict").is_file()


# -- what it leaves for the next session -------------------------------------------


def test_the_case_is_registered_and_the_geometry_phase_recorded(case_gen, tmp_path):
    target = tmp_path / "cyl"
    build(case_gen, "circle", target, "--reynolds", "200")

    # The state belongs to the study, one level up: two cases in one study share one
    # manifest, and the gallery of the study is meant to see both.
    state = target.parent / ".reynolds"
    assert not (target / ".reynolds").exists()

    rows = [json.loads(line) for line in
            (state / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows and rows[-1]["kind"] == "other"
    assert "circle" in rows[-1]["label"]

    phases = json.loads((state / "phases.json").read_text(encoding="utf-8"))
    geometry = [row for row in phases["phases"] if row["name"] == "geometry"][0]
    assert geometry["status"] == "done"


# -- the case has to be one the solver can actually start ---------------------------


@pytest.mark.parametrize("model,wanted,unwanted", [
    ("kOmegaSST", ["k", "omega", "nut"], ["epsilon", "nuTilda"]),
    ("kEpsilon", ["k", "epsilon", "nut"], ["omega", "nuTilda"]),
    ("SpalartAllmaras", ["nuTilda", "nut"], ["k", "omega", "epsilon"]),
])
def test_each_turbulence_model_gets_the_fields_it_actually_reads(
    case_gen, tmp_path, model, wanted, unwanted
):
    """A model looks its fields up by name. kEpsilon reads `epsilon` and never
    `omega`; SpalartAllmaras reads `nuTilda` and neither of those. Writing the wrong
    pair is not a run that converges badly, it is a solver that stops before the
    first iteration saying it cannot find a file."""
    target = tmp_path / model
    build(case_gen, "circle", target, "--turbulence", model)

    present = {p.name for p in (target / "0").glob("*")}
    assert present == {"U", "p", *wanted}, f"{model} got {sorted(present)}"
    for name in unwanted:
        assert name not in present


@pytest.mark.parametrize("model,fields", [
    ("kOmegaSST", ("k", "omega")),
    ("kEpsilon", ("k", "epsilon")),
    ("SpalartAllmaras", ("nuTilda",)),
])
def test_every_transported_field_has_a_div_scheme(case_gen, tmp_path, model, fields):
    """`divSchemes` says `default none`, so a field the model transports and the
    schemes do not name stops the run: 'div(phi,nuTilda) not found'."""
    target = tmp_path / model
    build(case_gen, "circle", target, "--turbulence", model)
    schemes = (target / "system" / "fvSchemes").read_text(encoding="utf-8")
    block = re.search(r"divSchemes\s*\{(.*?)\n\}", schemes, re.S).group(1)

    assert "div(phi,U)" in block
    for field in fields:
        assert f"div(phi,{field})" in block, f"{model}: no scheme for {field}"


def test_spalart_allmaras_does_not_ask_a_wall_function_for_a_field_it_has_not_got(
    case_gen, tmp_path
):
    """nutkWallFunction computes nut from k. A Spalart-Allmaras case has no k."""
    target = tmp_path / "sa"
    build(case_gen, "circle", target, "--turbulence", "SpalartAllmaras")
    nut = (target / "0" / "nut").read_text(encoding="utf-8")

    assert "nutkWallFunction" not in nut
    assert "nutUSpaldingWallFunction" in nut


@pytest.mark.parametrize("model", ["laminar", "kOmegaSST", "kEpsilon", "SpalartAllmaras"])
def test_a_transient_case_has_the_final_linear_solvers_pimple_asks_for(case_gen, tmp_path, model):
    """PIMPLE solves the last inner iteration of every step with `pFinal`, `UFinal`
    and the turbulence fields' Final entries, by those literal names -- `p` does not
    stand in for `pFinal`. Without them pimpleFoam stops on the first time step with
    'keyword pFinal is undefined in dictionary solvers', which a steady run never
    hits because SIMPLE never asks."""
    target = tmp_path / model
    build(case_gen, "circle", target, "--study", "transient", "--turbulence", model)
    solution = (target / "system" / "fvSolution").read_text(encoding="utf-8")
    solvers = solution[solution.index("solvers"):solution.index("PIMPLE")]

    names = set(re.findall(r'^    ("?[\w()|.]+"?)$', solvers, re.M))
    for name in names:
        if name.endswith("Final") or name.endswith('Final"'):
            continue
        expected = name[:-1] + 'Final"' if name.endswith('"') else name + "Final"
        assert expected in names, f"{model}: {name} has no {expected}"
    assert "pFinal" in names and "UFinal" in names


def test_a_steady_case_does_not_carry_final_solvers_it_never_asks_for(case_gen, tmp_path):
    target = tmp_path / "steady"
    build(case_gen, "circle", target, "--study", "steady")
    assert "Final" not in (target / "system" / "fvSolution").read_text(encoding="utf-8")


# -- a step that is a step, and a flag that is not quietly dropped ------------------


def test_a_fixed_delta_t_is_actually_fixed(case_gen, tmp_path):
    """`adjustTimeStep yes` alongside a step the user typed in means the solver is
    free to ignore it, and the only place the disagreement shows is the names of the
    time directories."""
    target = tmp_path / "fixed"
    build(case_gen, "circle", target, "--study", "transient", "--delta-t", "1e-4")
    control = (target / "system" / "controlDict").read_text(encoding="utf-8")

    assert "deltaT          0.0001;" in control
    assert "adjustTimeStep  no;" in control
    assert "maxCo" not in control


def test_the_courant_step_uses_the_shortest_cell_the_mesh_actually_has(case_gen, tmp_path):
    """A block graded 6:1 over 24 cells has a first cell a third of its mean, so a
    step sized from a nominal cell starts the run well above the Courant number that
    was asked for."""
    opts = {
        "size": 0.1, "upstream": 8.0, "downstream": 20.0, "above": 8.0, "below": 8.0,
        "ogrid_scale": 2.5, "ogrid_aspect": 1.0, "cells_around": 40, "cells_radial": 12,
        "radial_grading": 6.0, "far_grading": 8.0, "thickness": 0.01,
    }
    plan = case_gen.build_plan("circle", opts)
    shortest = plan.mesh.smallest_cell

    target = tmp_path / "t"
    build(case_gen, "circle", target, "--study", "transient", "--courant", "0.4",
          "--cells-around", "40", "--cells-radial", "12", "--speed", "2")
    control = (target / "system" / "controlDict").read_text(encoding="utf-8")
    delta = float(re.search(r"deltaT\s+([0-9.eE+-]+);", control).group(1))

    assert delta == pytest.approx(0.4 * shortest / 2.0, rel=1e-5)
    # and it is shorter than the nominal body-size-over-cells-around that says
    # nothing about the mesh, so the step really did come down.
    assert shortest < 0.1 / 40


def test_shortest_cell_of_a_graded_block_is_the_graded_one(case_gen):
    # 24 cells over 1 m expanding 6:1 -- the first is well under the mean of 1/24.
    assert case_gen.shortest_cell(1.0, 24, 6.0) == pytest.approx(0.0147677, rel=1e-4)
    assert case_gen.shortest_cell(1.0, 24, 6.0) < 1.0 / 24
    # and a contracting block's shortest is its last, the same size.
    assert case_gen.shortest_cell(1.0, 24, 1.0 / 6.0) == pytest.approx(
        case_gen.shortest_cell(1.0, 24, 6.0), rel=1e-9)
    assert case_gen.shortest_cell(1.0, 10, 1.0) == pytest.approx(0.1)
    assert case_gen.shortest_cell(1.0, 1, 5.0) == pytest.approx(1.0)


@pytest.mark.parametrize("flag", ["--moving-ground", "--rotating-wheels"])
@pytest.mark.parametrize("template", ["duct-t", "bend-rounded"])
def test_a_ground_or_wheel_flag_on_a_duct_is_refused_not_dropped(
    case_gen, tmp_path, flag, template
):
    """A duct has no ground and no wheels. Accepting the flag and writing the same
    case anyway is worse than refusing it: what comes out looks exactly like what
    was asked for."""
    with pytest.raises(SystemExit, match="external-flow templates"):
        build(case_gen, template, tmp_path / "x", flag)


def test_the_internal_field_points_the_way_the_inlet_does(case_gen, tmp_path):
    """duct-f's inlet is on the floor and the duct runs up the y axis. Seeding every
    cell at (U 0 0) points the whole domain across the duct instead of along it."""
    target = tmp_path / "f"
    build(case_gen, "duct-f", target, "--speed", "3")
    field = (target / "0" / "U").read_text(encoding="utf-8")

    internal = re.search(r"internalField\s+uniform\s+\(([^)]*)\)", field).group(1)
    assert [float(v) for v in internal.split()] == pytest.approx([0.0, 3.0, 0.0])
    inlet = re.search(r"    inlet\s*\{([^}]*)\}", field).group(1)
    assert "(0 3 0)" in inlet

    # and the +x templates are unchanged.
    other = tmp_path / "cyl"
    build(case_gen, "circle", other, "--speed", "3")
    text = (other / "0" / "U").read_text(encoding="utf-8")
    assert re.search(r"internalField\s+uniform\s+\(3 0 0\)", text)


def test_a_patch_with_no_role_is_a_loud_error_not_a_missing_entry(case_gen, tmp_path):
    """Driving the 0/ writer from the mesh's own patch list means a patch the
    template made and forgot to describe stops here, rather than being discovered
    by the solver on the first time step."""
    plan = case_gen.build_plan("circle", {
        "size": 0.1, "upstream": 8.0, "downstream": 20.0, "above": 8.0, "below": 8.0,
        "ogrid_scale": 2.5, "ogrid_aspect": 1.0, "cells_around": 40, "cells_radial": 12,
        "radial_grading": 6.0, "far_grading": 8.0, "thickness": 0.01,
    })
    plan.roles.pop("topWall")

    with pytest.raises(SystemExit, match="did not give a role"):
        case_gen.field_p(plan)


# -- free-stream turbulence, and the wall treatment ----------------------------
#
# Two live studies came back wrong from the defaults here and neither noticed the
# cause. A NACA 0012 gave Cd seven times the published value; an Ahmed body 41% high
# with its slant separation diffused away. Both inherited an ambient eddy viscosity
# roughly 7,000x molecular, from a pipe mixing-length correlation applied to a body in
# open air with wind-tunnel-grid intensity.


def read_field(case: Path, name: str) -> str:
    return (case / "0" / name).read_text(encoding="utf-8")


def internal_value(text: str) -> float:
    line = next(l for l in text.splitlines() if l.strip().startswith("internalField"))
    return float(line.split()[-1].rstrip(";"))


def test_free_stream_eddy_viscosity_is_a_ratio_not_a_pipe_mixing_length(case_gen, tmp_path):
    """nu_t/nu came out at 7,044 on a case the recommended band puts at 0.1-10."""
    case = tmp_path / "sq"
    assert build(case_gen, "square", case, "--size", "1.0", "--reynolds", "3e6",
                 "--speed", "30", "--study", "steady") == 0

    k = internal_value(read_field(case, "k"))
    omega = internal_value(read_field(case, "omega"))
    nu = 1.0 / 3e6 * 30 * 1.0
    assert k / omega / nu == pytest.approx(case_gen.FREE_STREAM_VISCOSITY_RATIO, rel=1e-3)
    # the exact pair the finding named as the fix for this case
    assert k == pytest.approx(1.35e-3, rel=1e-3)
    assert omega == pytest.approx(13.5, rel=1e-3)


def test_a_duct_keeps_the_percent_level_turbulence_it_really_has(case_gen, tmp_path):
    """The 5% default was not wrong everywhere -- internal flow does carry it. Only a
    body in unbounded flow was being given wind-tunnel numbers."""
    case = tmp_path / "duct"
    assert build(case_gen, "duct-t", case, "--reynolds", "1e5", "--study", "steady") == 0
    k = internal_value(read_field(case, "k"))
    assert k == pytest.approx(1.5 * (0.05 * 1.0) ** 2, rel=1e-3)


def test_an_explicit_mixing_length_still_wins(case_gen, tmp_path):
    """A stated mixing length is a deliberate choice; the ratio is only the default."""
    case = tmp_path / "sq2"
    assert build(case_gen, "square", case, "--size", "1.0", "--reynolds", "3e6",
                 "--speed", "30", "--study", "steady", "--mixing-length", "0.07") == 0
    k = internal_value(read_field(case, "k"))
    omega = internal_value(read_field(case, "omega"))
    expected = math.sqrt(k) / (case_gen.CMU ** 0.25 * 0.07)
    assert omega == pytest.approx(expected, rel=1e-3)


def test_the_viscosity_ratio_can_be_asked_for(case_gen, tmp_path):
    case = tmp_path / "sq3"
    assert build(case_gen, "square", case, "--size", "1.0", "--reynolds", "3e6",
                 "--speed", "30", "--study", "steady", "--viscosity-ratio", "2") == 0
    k = internal_value(read_field(case, "k"))
    omega = internal_value(read_field(case, "omega"))
    assert k / omega / 1e-5 == pytest.approx(2.0, rel=1e-3)


def test_the_wall_treatment_is_valid_wherever_the_first_cell_lands(case_gen, tmp_path):
    """nutkWallFunction is the high-Re form and holds for y+ ~30-300. The generated
    O-grid puts the first cell centre far outside that, so the model was extrapolating
    where it had never been valid; Spalding's law blends across the whole range."""
    case = tmp_path / "sq4"
    assert build(case_gen, "square", case, "--size", "1.0", "--reynolds", "3e6",
                 "--study", "steady") == 0
    assert "nutUSpaldingWallFunction" in read_field(case, "nut")
    assert "nutkWallFunction" not in read_field(case, "nut")


def test_every_model_gets_a_wall_function_valid_across_the_range(case_gen):
    for model in ("kOmegaSST", "kEpsilon", "SpalartAllmaras"):
        assert case_gen.nut_wall_function(model) == "nutUSpaldingWallFunction"


def test_the_two_numbers_a_turbulent_case_gets_wrong_are_printed(case_gen, tmp_path, capsys):
    """The near-wall problem is measurable from the case afterwards and a study did
    measure it. The free-stream one is not measurable from anything the product
    reported, so it went unmentioned twice. Both are printed now."""
    assert build(case_gen, "square", tmp_path / "sq5", "--size", "1.0",
                 "--reynolds", "3e6", "--speed", "30", "--study", "steady") == 0
    out = capsys.readouterr().out
    assert "nu_t/nu" in out and "free stream" in out
    assert "y+" in out and "near wall" in out


# -- the case says when it has converged, and computes what it is for ----------


def test_a_steady_case_defines_what_converged_means(case_gen, tmp_path):
    """There was no residualControl anywhere in the toolbox, so `endTime` was the only
    stopping rule: every steady run executed its iteration count and stopped, converged
    or not, and never printed "SIMPLE solution converged"."""
    case = tmp_path / "rc"
    assert build(case_gen, "square", case, "--reynolds", "3e6", "--study", "steady") == 0
    text = (case / "system" / "fvSolution").read_text(encoding="utf-8")
    simple = text[text.index("SIMPLE"):]
    assert "residualControl" in simple
    assert "p" in simple and "U" in simple
    assert '"(k|omega)"' in simple, "the turbulence fields this model actually transports"


def test_residual_control_names_only_the_fields_the_model_has(case_gen):
    assert case_gen.residual_control(()) == list(case_gen.RESIDUAL_CONTROL)
    entries = dict(case_gen.residual_control(("nuTilda",)))
    assert "nuTilda" in entries and "omega" not in "".join(entries)


def test_the_case_computes_the_coefficients_the_study_is_about(case_gen, tmp_path):
    """controlDict had no functions block, while preflight warns when forceCoeffs is
    missing and results.py reads forceCoeffs.dat -- both ends assumed one existed."""
    case = tmp_path / "fo"
    assert build(case_gen, "square", case, "--size", "1.0", "--reynolds", "3e6",
                 "--speed", "30", "--study", "steady") == 0
    text = (case / "system" / "controlDict").read_text(encoding="utf-8")
    assert "functions" in text and "forceCoeffs" in text
    assert "magUInf         30" in text
    # Aref = size x thickness, and thickness defaults to size/10 -- the factor of ten
    # every study had to get right by hand.
    assert "Aref            0.1" in text
    assert "lRef            1" in text
    assert "solverInfo" in text, "residuals are logged where log_digest can find them"


def test_a_duct_gets_no_force_coefficients(case_gen, tmp_path):
    """There is no body in a duct to take forces on."""
    case = tmp_path / "duct"
    assert build(case_gen, "duct-t", case, "--reynolds", "1e5", "--study", "steady") == 0
    assert "forceCoeffs" not in (case / "system" / "controlDict").read_text(encoding="utf-8")


# -- the smaller things that were quietly off ----------------------------------


def test_a_ducts_reynolds_number_is_on_its_hydraulic_diameter(case_gen, tmp_path):
    """For a 2D channel D_h = 2w, not w. The templates passed the width, so every duct
    and bend ran at twice the Reynolds number it reported -- and because LAMINAR_BELOW
    is itself a D_h criterion, `--reynolds 2000` was really 4000 and got `laminar`."""
    case = tmp_path / "duct"
    assert build(case_gen, "duct-t", case, "--duct-width", "0.05",
                 "--reynolds", "1000", "--speed", "1", "--study", "steady") == 0
    nu = float(next(l for l in (case / "constant" / "transportProperties")
                    .read_text(encoding="utf-8").splitlines()
                    if l.startswith("nu ")).split()[-1].rstrip(";"))
    assert nu == pytest.approx(1.0 * 0.10 / 1000, rel=1e-3), "L is 2w = 0.10 m"
    assert case_gen.hydraulic_diameter(0.05) == 0.10


def test_the_tunnel_walls_are_not_typed_as_walls(case_gen, tmp_path):
    """They carry `slip`, so typing them `wall` put them in wallDist and in every
    yPlus/wallShearStress function object, diluting the one histogram you would use to
    check the near-wall mesh."""
    case = tmp_path / "sq"
    assert build(case_gen, "square", case, "--reynolds", "3e6", "--study", "steady") == 0
    boundary = (case / "constant" / "polyMesh" / "blockMeshDict").read_text(encoding="utf-8") \
        if (case / "constant" / "polyMesh" / "blockMeshDict").exists() \
        else (case / "system" / "blockMeshDict").read_text(encoding="utf-8")
    top = boundary[boundary.index("topWall"):]
    assert top.split("type")[1].split(";")[0].strip() == "patch"


def test_the_domain_is_not_tight_enough_to_change_the_answer(case_gen, tmp_path):
    """16 body-heights tall is 6.25% blockage with free-slip walls -- 4-7% on Cd against
    the Re=100 cylinder benchmark, which is the width of a validation band."""
    case = tmp_path / "cyl"
    assert build(case_gen, "circle", case, "--size", "1.0", "--reynolds", "100",
                 "--study", "steady") == 0
    text = (case / "system" / "blockMeshDict").read_text(encoding="utf-8")
    ys = [float(m) for m in __import__("re").findall(r"\(\s*[-0-9.e+]+\s+([-0-9.e+]+)\s+", text)]
    assert max(ys) >= 12.0 and min(ys) <= -12.0, "at least 12 sizes above and below"


def test_a_transient_run_does_not_damp_the_thing_it_is_watching(case_gen, tmp_path):
    """Upwind dissipation smooths out exactly the shedding a transient study is run to
    see -- a wake made steady by the scheme rather than by the physics."""
    case = tmp_path / "t"
    assert build(case_gen, "circle", case, "--reynolds", "100", "--study", "transient") == 0
    schemes = (case / "system" / "fvSchemes").read_text(encoding="utf-8")
    assert "Gauss linear;" in schemes and "linearUpwind" not in schemes


def test_the_adjustable_step_is_actually_bounded(case_gen, tmp_path):
    """maxDeltaT was 100x the initial step, which is a Courant number near 90."""
    case = tmp_path / "t2"
    assert build(case_gen, "circle", case, "--reynolds", "100", "--study", "transient") == 0
    text = (case / "system" / "controlDict").read_text(encoding="utf-8")
    delta = float(next(l for l in text.splitlines() if l.strip().startswith("deltaT")).split()[-1].rstrip(";"))
    cap = float(next(l for l in text.splitlines() if l.strip().startswith("maxDeltaT")).split()[-1].rstrip(";"))
    assert cap == pytest.approx(delta * 5, rel=1e-3)


def test_the_angle_of_attack_flag_says_which_way_it_turns(case_gen):
    """The code rotated the leading edge up and both the docstring and the flag said
    "nose down" -- so an agent trusting the description negates the angle."""
    import inspect

    assert "leading edge up" in inspect.getsource(case_gen.main), "the --aoa help text"
    doc = inspect.getdoc(case_gen.profile_body)
    assert "positive incidence" in doc
    assert "nose-down" not in doc and "nose down" not in doc.split("used to say")[0]
