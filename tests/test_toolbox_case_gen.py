"""`case_gen.py`: the case files for a mesh that is already there.

It used to build the mesh as well, from a template name, and most of this file used
to be about the shapes. They are gone: a mesh arrives from wherever it came from and
this dresses it. So what is pinned here is the reading -- what the script works out
about a mesh it did not make -- and the writing that follows from it.

The mesh under test is a real one (`tests/data/cavity`, a lid-driven cavity: one cell
thick, `movingWall`, `fixedWalls`, an `empty` `frontAndBack`), copied and its patch
names and types rewritten per test. Its geometry is genuine, so pyvista reads it and
the areas, centres and normals are measured rather than mocked.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"
CAVITY = Path(__file__).resolve().parent / "data" / "cavity"


@pytest.fixture(scope="module")
def case_gen():
    spec = importlib.util.spec_from_file_location("case_gen", TOOLBOX / "case_gen.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mesh_only(root: Path, name: str = "case") -> Path:
    """The cavity's mesh, and nothing else: no 0/, no dictionaries."""
    case = root / name
    (case / "constant").mkdir(parents=True)
    shutil.copytree(CAVITY / "constant" / "polyMesh", case / "constant" / "polyMesh")
    return case


def rename_patches(case: Path, mapping: dict[str, tuple[str, str]]) -> None:
    """Rewrite the boundary file's names and types. The faces stay exactly as they
    were, so the mesh is still a mesh and every measurement off it is still real."""
    path = case / "constant" / "polyMesh" / "boundary"
    text = path.read_text(encoding="utf-8")
    for old, (new, kind) in mapping.items():
        text = re.sub(rf"^(\s*){old}\s*$", rf"\1{new}", text, flags=re.M)
        # the type line belonging to that entry: the first one after the name
        text = re.sub(rf"({new}\s*\n\s*\{{\s*\n\s*type\s+)\w+;", rf"\g<1>{kind};", text)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def duct(tmp_path):
    """The cavity renamed into a passage: an inlet, an outlet, walls, empty ends."""
    case = mesh_only(tmp_path)
    rename_patches(case, {
        "movingWall": ("inlet", "patch"),
        "fixedWalls": ("walls", "wall"),
        "frontAndBack": ("frontAndBack", "empty"),
    })
    return case


@pytest.fixture
def duct_with_outlet(tmp_path):
    """Both ends named, so the roles come entirely from the names."""
    case = mesh_only(tmp_path, "duct")
    rename_patches(case, {
        "movingWall": ("inlet", "patch"),
        "fixedWalls": ("outlet", "patch"),
        "frontAndBack": ("frontAndBack", "empty"),
    })
    return case


def run(case_gen, case: Path, *args: str) -> int:
    return case_gen.main([str(case), *args])


def field(case: Path, name: str) -> str:
    return (case / "0" / name).read_text(encoding="utf-8")


def entry_for(text: str, patch: str) -> str:
    """One patch's block out of a boundaryField."""
    match = re.search(rf"^    {patch}\n    \{{\n(.*?)^    \}}", text, re.S | re.M)
    assert match, f"no {patch} entry in\n{text}"
    return match.group(1)


# -- reading the mesh --------------------------------------------------------------


def test_a_directory_with_no_mesh_in_it_is_refused_by_name(case_gen, tmp_path):
    with pytest.raises(SystemExit) as raised:
        run(case_gen, tmp_path / "empty", "--length", "0.1")
    assert "no constant/polyMesh" in str(raised.value)
    assert "mesh it first" in str(raised.value)


def test_the_patches_types_and_counts_come_off_the_mesh(case_gen, duct):
    mesh = case_gen.read_mesh(duct)
    assert set(mesh.patch_faces) == {"inlet", "walls", "frontAndBack"}
    assert mesh.two_d is True
    assert mesh.patch("frontAndBack")["type"] == "empty"


def test_the_bounding_box_is_two_corners_not_vtks_interleaving(case_gen, duct):
    """A span of `0.06 x -0.3 x 0.001 m` is what mixing the two conventions gives,
    and nothing downstream complains about a negative one."""
    mesh = case_gen.read_mesh(duct)
    assert len(mesh.bounds) == 6
    assert all(span > 0 for span in mesh.span)
    assert mesh.bounds[:3] < mesh.bounds[3:]


def test_a_cell_size_is_available_even_where_checkmesh_is_not(case_gen, duct):
    """checkMesh is on the instance and not on a laptop. Without it the average cell
    stands in -- and says it is standing in, because a time step from an average cell
    is optimistic and a time step of zero is a broken dictionary."""
    mesh = case_gen.read_mesh(duct)
    assert mesh.smallest_cell > 0
    if mesh.cell_is_estimate:
        assert mesh.smallest_cell == mesh.average_cell


@pytest.mark.parametrize("bounds,axis,thickness", [
    ([0, 0, 0, 0.3, 0.06, 0.001], 2, 0.001),      # the usual: extruded in z
    ([0, 0, 0, 0.3, 0.001, 0.06], 1, 0.001),      # extruded in y, which nothing forbids
    ([0, 0, 0, 0.001, 0.3, 0.06], 0, 0.001),      # and in x
    ([0, 0, 0, 0.3, 0.2, 0.1], 2, 0.1),           # a real volume: no thin axis, z stands
])
def test_the_thin_direction_is_measured_not_assumed(case_gen, bounds, axis, thickness):
    """`bounds[5] - bounds[2]` reads a metre-scale extent as a cell width on a case
    extruded in y, and that number divides the hydraulic diameter, the cell size, the
    time step and the y+ estimate."""
    mesh = case_gen.MeshFacts(patches=[], bounds=bounds, cells=100, min_volume=0,
                              two_d=True)
    assert mesh.thin_axis == axis
    assert mesh.thickness == pytest.approx(thickness)


def test_a_body_in_the_flow_is_one_that_reaches_at_most_one_of_the_domains_axes(case_gen, duct):
    """Three readings of this were wrong before this one: a cylinder in a plane case
    is on both z faces like everything else, and an L-duct's own wall is not the
    enclosure by the all-points test."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("mesh_look", TOOLBOX / "mesh_look.py")
    mesh_look = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mesh_look)
    box = [0, 0, 0, 0.3, 0.06, 0.001]
    # a ring in mid-channel: reaches nothing
    ring = [[0.1, 0.03, 0.0], [0.11, 0.03, 0.001], [0.1, 0.04, 0.0]]
    assert mesh_look.box_contact(ring, box) == 0
    # a body sitting on the floor: reaches y only
    on_floor = [[0.1, 0.0, 0.0], [0.12, 0.0, 0.001], [0.11, 0.02, 0.0]]
    assert mesh_look.box_contact(on_floor, box) == 1
    # a passage wall: runs into the ends and the sides
    wall = [[0.0, 0.0, 0.0], [0.3, 0.0, 0.001], [0.3, 0.06, 0.0]]
    assert mesh_look.box_contact(wall, box) == 2
    # the thin axis is never counted: in a plane case everything is on both z faces
    assert mesh_look.box_contact([[0.1, 0.03, 0.0], [0.1, 0.03, 0.001]], box) == 0


# -- what each patch is for --------------------------------------------------------


def test_roles_come_from_the_names_when_nothing_is_said(case_gen, duct_with_outlet):
    roles = case_gen.roles_for(case_gen.read_mesh(duct_with_outlet), {})
    assert roles["inlet"]["kind"] == "inlet"
    assert roles["outlet"]["kind"] == "outlet"
    assert roles["frontAndBack"]["kind"] == "empty"


def test_what_is_named_on_the_command_line_wins(case_gen, duct):
    mesh = case_gen.read_mesh(duct)
    roles = case_gen.roles_for(mesh, {"outlet": ["walls"], "wall": ["inlet"]})
    assert roles["walls"]["kind"] == "outlet"
    assert roles["inlet"]["kind"] == "wall"


def test_a_patch_that_is_not_in_the_mesh_is_refused_with_the_list(case_gen, duct):
    mesh = case_gen.read_mesh(duct)
    with pytest.raises(SystemExit) as raised:
        case_gen.roles_for(mesh, {"inlet": ["nozzle"]})
    assert "no patch called nozzle" in str(raised.value)
    assert "walls" in str(raised.value)


def test_an_unnamed_patch_is_a_wall_which_is_the_safe_reading(case_gen, tmp_path):
    """A wall that should have been an outlet shows up in the first residual plot;
    an outlet that should have been a wall quietly drains the domain."""
    case = mesh_only(tmp_path)
    rename_patches(case, {"movingWall": ("lid", "wall"), "fixedWalls": ("sides", "wall")})
    roles = case_gen.roles_for(case_gen.read_mesh(case), {})
    assert roles["lid"]["kind"] == "wall" and roles["sides"]["kind"] == "wall"


def test_the_meshs_own_constraint_types_are_never_overridden_by_a_name(case_gen, tmp_path):
    """`empty` and the symmetry types are constraints: getting them wrong stops the
    solver rather than misleading it, so the mesh's word is final."""
    case = mesh_only(tmp_path)
    rename_patches(case, {"frontAndBack": ("inletSides", "empty")})
    roles = case_gen.roles_for(case_gen.read_mesh(case), {})
    assert roles["inletSides"]["kind"] == "empty"


def test_a_wheel_spins_at_the_road_speed_over_its_radius(case_gen, duct):
    mesh = case_gen.read_mesh(duct)
    roles = case_gen.roles_for(mesh, {"spin": ["walls:0.25:z"]})
    assert roles["walls"]["kind"] == "spinning"
    assert roles["walls"]["radius"] == 0.25
    assert roles["walls"]["axis"] == (0.0, 0.0, 1.0)


def test_a_spin_without_a_radius_is_refused(case_gen, duct):
    with pytest.raises(SystemExit) as raised:
        case_gen.roles_for(case_gen.read_mesh(duct), {"spin": ["walls"]})
    assert "name:radius" in str(raised.value)


# -- which way the flow goes -------------------------------------------------------


def test_the_inlet_direction_is_measured_and_points_into_the_domain(case_gen, duct):
    """The failure this replaces: classifying by bounding box, which on a U-duct with
    both ends on the left gets the direction exactly backwards and says nothing."""
    mesh = case_gen.read_mesh(duct)
    roles = case_gen.roles_for(mesh, {})
    direction = roles["inlet"]["direction"]
    centre = mesh.centre
    inlet = mesh.patch("inlet").get("center")
    if inlet:  # measured only where the reader could open the mesh
        toward = [centre[i] - inlet[i] for i in range(3)]
        assert sum(direction[i] * toward[i] for i in range(3)) > 0


def test_a_stated_direction_wins_over_the_measurement(case_gen, duct):
    roles = case_gen.roles_for(case_gen.read_mesh(duct), {"direction": "-y"})
    assert roles["inlet"]["direction"] == (0.0, -1.0, 0.0)


def test_a_direction_that_is_not_an_axis_is_refused(case_gen, duct):
    with pytest.raises(SystemExit) as raised:
        case_gen.roles_for(case_gen.read_mesh(duct), {"direction": "up"})
    assert "--direction takes x, y, z" in str(raised.value)


# -- the length the Reynolds number is on ------------------------------------------


def test_the_length_is_the_inlets_hydraulic_diameter(case_gen):
    """4A/P for a plane passage is 2w, and the width is the inlet's area over the one
    cell's thickness. Passing the width instead once ran every duct at twice the
    Reynolds number it reported."""
    assert case_gen.hydraulic_diameter(0.0006, 0.001, two_d=True) == pytest.approx(1.2)
    assert case_gen.hydraulic_diameter(0.0, 0.001, two_d=True) == 0.0


def test_the_3d_reading_is_not_called_a_hydraulic_diameter(case_gen, tmp_path):
    """In 3D it is the diameter of a circle of the same area -- for a 100 x 5 mm slot
    that is 2.6x the true 4A/P, and the name is the only place a reader would learn
    the difference."""
    case = mesh_only(tmp_path, "volume")
    rename_patches(case, {"movingWall": ("inlet", "patch"), "frontAndBack": ("side", "wall")})
    mesh = case_gen.read_mesh(case)
    roles = case_gen.roles_for(mesh, {})
    _length, source = case_gen.characteristic_length(mesh, roles, {})
    if mesh.two_d:
        assert "hydraulic diameter" in source
    else:
        assert "equivalent circular diameter" in source


def test_a_stated_length_wins_and_says_where_it_came_from(case_gen, duct):
    mesh = case_gen.read_mesh(duct)
    roles = case_gen.roles_for(mesh, {})
    length, source = case_gen.characteristic_length(mesh, roles, {"length": 0.05})
    assert length == 0.05 and source == "--length"


def test_a_case_with_nothing_to_measure_asks_for_the_length(case_gen, tmp_path):
    case = mesh_only(tmp_path)
    rename_patches(case, {"movingWall": ("lid", "wall"), "fixedWalls": ("sides", "wall")})
    mesh = case_gen.read_mesh(case)
    roles = case_gen.roles_for(mesh, {})
    with pytest.raises(SystemExit) as raised:
        case_gen.characteristic_length(mesh, roles, {})
    assert "give --length" in str(raised.value)


# -- a body in the flow, or a passage ----------------------------------------------


def test_a_passage_carries_percent_level_turbulence_and_a_body_does_not(case_gen, duct):
    plan = case_gen.build_plan(duct, {"length": 0.1})
    assert case_gen.free_stream_intensity(plan, {}) == 0.05
    assert case_gen.viscosity_ratio(plan, {}) is None
    plan.info["external"] = True
    assert case_gen.free_stream_intensity(plan, {}) == case_gen.FREE_STREAM_INTENSITY
    assert case_gen.viscosity_ratio(plan, {}) == case_gen.FREE_STREAM_VISCOSITY_RATIO


def test_what_is_asked_for_beats_both(case_gen, duct):
    plan = case_gen.build_plan(duct, {"length": 0.1})
    assert case_gen.free_stream_intensity(plan, {"turbulent_intensity": 0.02}) == 0.02
    assert case_gen.viscosity_ratio(plan, {"viscosity_ratio": 3}) == 3.0


def test_external_can_be_stated_when_the_mesh_does_not_say_it(case_gen, duct):
    assert case_gen.build_plan(duct, {"length": 0.1, "external": True}).external is True


# -- the files it writes -----------------------------------------------------------


def test_no_blockmeshdict_is_written_because_the_mesh_is_already_there(case_gen, duct, capsys):
    run(case_gen, duct, "--length", "0.1", "--dry-run")
    printed = capsys.readouterr().out
    assert "blockMeshDict" not in printed
    assert "0/U" in printed and "system/controlDict" in printed


def test_every_patch_in_the_mesh_gets_an_entry_in_every_field(case_gen, duct_with_outlet):
    run(case_gen, duct_with_outlet, "--length", "0.1", "--reynolds", "5000")
    names = set(case_gen.read_mesh(duct_with_outlet).patch_faces)
    for name in ("U", "p", "k", "omega", "nut"):
        text = field(duct_with_outlet, name)
        for patch in names:
            assert f"    {patch}\n" in text, f"{patch} missing from 0/{name}"


def test_the_empty_patch_is_empty_in_every_field(case_gen, duct):
    """A `frontAndBack` declared `empty` in the mesh and `zeroGradient` in 0/U stops
    the solver on the first time step -- the failure that looks like something else."""
    run(case_gen, duct, "--length", "0.1", "--reynolds", "5000")
    for name in ("U", "p", "k", "omega", "nut"):
        assert "type            empty;" in entry_for(field(duct, name), "frontAndBack")


def test_the_inlet_carries_the_speed_and_the_outlet_lets_it_leave(case_gen, duct_with_outlet):
    run(case_gen, duct_with_outlet, "--length", "0.1", "--speed", "3")
    text = field(duct_with_outlet, "U")
    assert "fixedValue" in entry_for(text, "inlet")
    assert re.search(r"uniform \(-?3 -?0? ?-?0?\)|uniform \(-?[03] -?[03] -?[03]\)",
                     entry_for(text, "inlet"))
    assert "inletOutlet" in entry_for(text, "outlet")


def test_the_viscosity_written_is_the_one_the_reynolds_number_asked_for(case_gen, duct):
    run(case_gen, duct, "--length", "0.2", "--speed", "2", "--reynolds", "400")
    text = (duct / "constant" / "transportProperties").read_text()
    assert re.search(r"nu\s+(\[0 2 -1 0 0 0 0\]\s+)?0\.001;", text)


def test_a_laminar_case_gets_no_turbulence_fields_and_a_turbulent_one_does(case_gen, tmp_path):
    low = mesh_only(tmp_path, "low")
    rename_patches(low, {"movingWall": ("inlet", "patch"), "fixedWalls": ("walls", "wall")})
    run(case_gen, low, "--length", "0.1", "--reynolds", "100")
    assert not (low / "0" / "k").exists()
    high = mesh_only(tmp_path, "high")
    rename_patches(high, {"movingWall": ("inlet", "patch"), "fixedWalls": ("walls", "wall")})
    run(case_gen, high, "--length", "0.1", "--reynolds", "50000")
    assert (high / "0" / "k").exists() and (high / "0" / "omega").exists()


def test_a_belt_moves_with_the_stream_and_a_wall_does_not(case_gen, duct):
    """With the stream, whichever way the stream goes -- this mesh's inlet is the
    cavity's lid, so it flows down the y axis and the belt goes with it."""
    run(case_gen, duct, "--length", "0.1", "--speed", "5", "--belt", "walls")
    text = field(duct, "U")
    body = entry_for(text, "walls")
    assert "fixedValue" in body
    speed = [abs(float(v)) for v in
             re.search(r"uniform \(([^)]*)\)", body).group(1).split()]
    assert max(speed) == pytest.approx(5.0)
    assert sum(1 for v in speed if v > 1e-9) == 1


def test_a_spinning_wall_turns_at_the_road_speed_over_its_radius(case_gen, duct):
    run(case_gen, duct, "--length", "0.1", "--speed", "10", "--spin", "walls:0.5:z")
    body = entry_for(field(duct, "U"), "walls")
    assert "rotatingWallVelocity" in body
    assert "omega           20;" in body


def test_a_steady_case_counts_iterations_and_a_transient_one_counts_seconds(case_gen, tmp_path):
    steady = mesh_only(tmp_path, "steady")
    rename_patches(steady, {"movingWall": ("inlet", "patch")})
    run(case_gen, steady, "--length", "0.1", "--iterations", "700")
    text = (steady / "system" / "controlDict").read_text()
    assert "application     simpleFoam;" in text and "endTime         700;" in text

    unsteady = mesh_only(tmp_path, "unsteady")
    rename_patches(unsteady, {"movingWall": ("inlet", "patch")})
    run(case_gen, unsteady, "--length", "0.1", "--study", "transient", "--end-time", "3")
    text = (unsteady / "system" / "controlDict").read_text()
    assert "application     pimpleFoam;" in text and "endTime         3;" in text
    assert "adjustTimeStep  yes;" in text


def test_the_transient_step_comes_from_the_courant_number_and_the_cell(case_gen, duct):
    run(case_gen, duct, "--length", "0.1", "--study", "transient",
        "--speed", "2", "--courant", "0.5")
    text = (duct / "system" / "controlDict").read_text()
    cell = case_gen.read_mesh(duct).smallest_cell
    delta = float(re.search(r"deltaT\s+([\d.eE+-]+);", text).group(1))
    assert delta == pytest.approx(0.5 * cell / 2.0, rel=1e-3)


def test_a_mesh_only_study_names_no_solver_but_still_writes_fields(case_gen, duct, capsys):
    run(case_gen, duct, "--length", "0.1", "--study", "mesh")
    assert "mesh only, no solver" in capsys.readouterr().out
    assert (duct / "0" / "U").exists()


def test_a_case_writes_binary_and_collated(case_gen, duct):
    """`/work` is a network filesystem and the file count is what costs; both were
    measured (reconstructPar 133 s -> 6 s) and both belong in the case."""
    text = (duct / "system" / "controlDict").read_text() if (duct / "system").exists() else ""
    run(case_gen, duct, "--length", "0.1")
    text = (duct / "system" / "controlDict").read_text()
    assert "writeFormat     binary;" in text and "fileHandler     collated;" in text


def test_forces_are_computed_only_where_there_is_a_body_to_take_them(case_gen, tmp_path):
    plain = mesh_only(tmp_path, "plain")
    rename_patches(plain, {"movingWall": ("inlet", "patch"), "fixedWalls": ("walls", "wall")})
    run(case_gen, plain, "--length", "0.1")
    assert "forceCoeffs" not in (plain / "system" / "controlDict").read_text()

    withbody = mesh_only(tmp_path, "withbody")
    rename_patches(withbody, {"movingWall": ("inlet", "patch"), "fixedWalls": ("body", "wall")})
    run(case_gen, withbody, "--length", "0.1", "--body", "body")
    text = (withbody / "system" / "controlDict").read_text()
    assert "forceCoeffs" in text and "patches         (body);" in text


def test_writing_over_a_case_needs_force(case_gen, duct):
    run(case_gen, duct, "--length", "0.1")
    with pytest.raises(SystemExit) as raised:
        run(case_gen, duct, "--length", "0.1")
    assert "--force" in str(raised.value)
    assert run(case_gen, duct, "--length", "0.1", "--force") == 0


def test_a_dry_run_writes_nothing(case_gen, duct, capsys):
    run(case_gen, duct, "--length", "0.1", "--dry-run")
    assert not (duct / "0").exists()
    assert "would write" in capsys.readouterr().out


def test_the_summary_says_what_each_patch_became(case_gen, duct, capsys):
    run(case_gen, duct, "--length", "0.1", "--dry-run")
    printed = capsys.readouterr().out
    assert "inlet (" in printed and ", inlet)" in printed
    assert "frontAndBack" in printed and "empty)" in printed
    assert "length 0.1 m from --length" in printed


def test_the_case_is_recorded_against_the_study(case_gen, tmp_path):
    study = tmp_path / "study"
    case = mesh_only(study, "duct")
    rename_patches(case, {"movingWall": ("inlet", "patch")})
    run(case_gen, case, "--length", "0.1")
    rows = [json.loads(line) for line
            in (study / ".reynolds" / "manifest.jsonl").read_text().splitlines() if line.strip()]
    assert any(row.get("case") == "duct" for row in rows)


# -- the flow arithmetic, which did not change -------------------------------------


def test_reynolds_gives_the_viscosity_and_says_so(case_gen):
    flow = case_gen.derive_flow({"speed": 2.0, "reynolds": 200.0, "nu": None, "length": None}, 0.1)
    assert flow.nu == pytest.approx(0.001)
    assert "nu = U*L/Re" in flow.line()


def test_viscosity_gives_the_reynolds_number_and_says_so(case_gen):
    flow = case_gen.derive_flow({"speed": 2.0, "reynolds": None, "nu": 1e-5, "length": None}, 0.1)
    assert flow.reynolds == pytest.approx(20000)
    assert "Re = U*L/nu" in flow.line()


def test_both_at_once_is_refused_rather_than_one_quietly_winning(case_gen):
    with pytest.raises(SystemExit):
        case_gen.derive_flow({"speed": 1.0, "reynolds": 100.0, "nu": 1e-5, "length": None}, 0.1)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_speed_that_is_not_positive_is_refused(case_gen, bad):
    with pytest.raises(SystemExit):
        case_gen.derive_flow({"speed": bad, "reynolds": 100.0, "nu": None, "length": None}, 0.1)


def test_the_turbulence_model_is_chosen_from_reynolds_and_can_be_overridden(case_gen):
    low = case_gen.derive_flow({"speed": 1.0, "reynolds": 100.0, "nu": None, "length": None}, 0.1)
    assert case_gen.turbulence_model({"turbulence": "auto"}, low)[0] == "laminar"
    high = case_gen.derive_flow({"speed": 1.0, "reynolds": 1e6, "nu": None, "length": None}, 0.1)
    assert case_gen.turbulence_model({"turbulence": "auto"}, high)[0] == "kOmegaSST"
    assert case_gen.turbulence_model({"turbulence": "kEpsilon"}, low)[0] == "kEpsilon"
