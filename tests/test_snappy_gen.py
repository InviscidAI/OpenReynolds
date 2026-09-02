"""The surface-meshing generator, checked against shapes whose answers are known.

The three properties worth pinning are the three that were wrong in the field: the
reference area must be measured rather than assumed, a symmetry plane must halve it,
and the first layer must follow from y+ with the factor of two that the centroid
definition implies.
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"
sys.path.insert(0, str(TOOLBOX))

import case_gen  # noqa: E402
import snappy_gen  # noqa: E402


# -- surfaces to measure -----------------------------------------------------------

def write_binary_stl(path: Path, triangles) -> Path:
    with path.open("wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(struct.pack("<I", len(triangles)))
        for tri in triangles:
            fh.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for point in tri:
                fh.write(struct.pack("<3f", *point))
            fh.write(struct.pack("<H", 0))
    return path


def box_triangles(lx: float, ly: float, lz: float, origin=(0.0, 0.0, 0.0)):
    """A closed axis-aligned box, as twelve triangles."""
    ox, oy, oz = origin
    c = [(ox, oy, oz), (ox + lx, oy, oz), (ox + lx, oy + ly, oz), (ox, oy + ly, oz),
         (ox, oy, oz + lz), (ox + lx, oy, oz + lz), (ox + lx, oy + ly, oz + lz),
         (ox, oy + ly, oz + lz)]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (2, 3, 7, 6), (0, 4, 7, 3), (1, 2, 6, 5)]
    tris = []
    for a, b, cc, d in quads:
        tris.append((c[a], c[b], c[cc]))
        tris.append((c[a], c[cc], c[d]))
    return tris


@pytest.fixture()
def unit_box(tmp_path: Path) -> Path:
    """2 x 1 x 0.5 m: surface area 2*(2 + 1 + 0.5) = 7 m2, frontal (x) 0.5 m2."""
    return write_binary_stl(tmp_path / "box.stl", box_triangles(2.0, 1.0, 0.5))


def test_a_binary_stl_whose_header_says_solid_is_still_read_as_binary(tmp_path: Path):
    """The classic misread: a binary file beginning "solid" taken for ASCII, which
    parses to no triangles and reports an empty surface rather than an error."""
    path = tmp_path / "sneaky.stl"
    tris = box_triangles(1.0, 1.0, 1.0)
    write_binary_stl(path, tris)
    raw = bytearray(path.read_bytes())
    raw[0:5] = b"solid"
    path.write_bytes(bytes(raw))

    surface = snappy_gen.read_stl(path)
    assert len(surface.triangles) == len(tris)


def test_a_truncated_upload_is_named_as_one(tmp_path: Path):
    path = write_binary_stl(tmp_path / "cut.stl", box_triangles(1.0, 1.0, 1.0))
    path.write_bytes(path.read_bytes()[:-60])
    with pytest.raises(SystemExit, match="truncated"):
        snappy_gen.read_stl(path)


def test_wetted_area_is_the_triangle_sum(unit_box: Path):
    surface = snappy_gen.read_stl(unit_box)
    assert surface.wetted_area() == pytest.approx(7.0, rel=1e-6)


def test_frontal_area_is_the_silhouette(unit_box: Path):
    surface = snappy_gen.read_stl(unit_box)
    assert surface.frontal_area(axis=0) == pytest.approx(0.5, rel=1e-3)
    assert surface.frontal_area(axis=2) == pytest.approx(2.0, rel=1e-3)


def test_frontal_area_does_not_count_a_shadow_twice(tmp_path: Path):
    """Two boxes in line astern have the frontal area of one.

    This is the case the analytic 0.5*sum(A|n.d|) form gets wrong, and it is not
    exotic: a tube bundle, a rotor behind a hub and a rider behind a bike all have
    the shape of it."""
    tris = (box_triangles(1.0, 1.0, 1.0)
            + box_triangles(1.0, 1.0, 1.0, origin=(4.0, 0.0, 0.0)))
    surface = snappy_gen.read_stl(write_binary_stl(tmp_path / "pair.stl", tris))
    assert surface.wetted_area() == pytest.approx(12.0, rel=1e-6)
    assert surface.frontal_area(axis=0) == pytest.approx(1.0, rel=1e-3)


# -- the reference area, which is the whole of F-43 --------------------------------

def base_opts(**over):
    opts = {"ahead": 2.0, "behind": 5.0, "side": 2.0, "above": 2.0, "below": 2.0,
            "symmetry": "none", "background_cells": 20, "base_cell": None,
            "area": "frontal", "ref_area": None, "ground": False}
    opts.update(over)
    return opts


def test_a_whole_body_uses_the_whole_area(unit_box: Path):
    surface = snappy_gen.read_stl(unit_box)
    opts = base_opts(area="wetted")
    domain, _ = snappy_gen.build_domain(surface, opts)
    area, why = snappy_gen.reference_area(surface, domain, opts)
    assert area == pytest.approx(7.0, rel=1e-6)
    assert "wetted" in why


def test_a_symmetry_plane_halves_the_reference_area(unit_box: Path):
    """The failure this exists to stop: a half model's force over the whole body's
    area, which reads as exactly half the right answer and looks entirely
    plausible."""
    surface = snappy_gen.read_stl(unit_box)
    opts = base_opts(area="wetted", symmetry="y")
    domain, _ = snappy_gen.build_domain(surface, opts)
    area, why = snappy_gen.reference_area(surface, domain, opts)
    assert area == pytest.approx(3.5, rel=1e-6)
    assert "0.5" in why and "symmetry" in why


def test_a_given_reference_area_is_halved_too(unit_box: Path):
    """--ref-area is the whole body's, stated as such, so the same halving applies.
    Taking it as already-halved would put the trap back, one level down."""
    surface = snappy_gen.read_stl(unit_box)
    opts = base_opts(symmetry="y", ref_area=10.0)
    domain, _ = snappy_gen.build_domain(surface, opts)
    area, why = snappy_gen.reference_area(surface, domain, opts)
    assert area == pytest.approx(5.0, rel=1e-9)
    assert "0.5" in why


def test_the_mesh_is_cut_where_the_area_is_halved(unit_box: Path):
    """The two must agree. They are computed in different functions, so this is the
    assertion that keeps them from drifting apart."""
    surface = snappy_gen.read_stl(unit_box)
    opts = base_opts(symmetry="y")
    domain, _ = snappy_gen.build_domain(surface, opts)
    assert domain.symmetry == "y"
    assert "symmetry" in domain.patches
    assert domain.bounds[2] == pytest.approx(surface.centre[1], abs=1e-9)
    # and the far side is untouched
    assert domain.bounds[3] > surface.bounds[3]


def test_a_body_off_the_origin_is_cut_on_its_own_centreline(tmp_path: Path):
    """A hull exported with its origin at the bow is still symmetric about itself.
    Cutting at y = 0 would slice it off-centre and quietly mesh the wrong fraction."""
    tris = box_triangles(2.0, 1.0, 0.5, origin=(0.0, 5.0, 0.0))
    surface = snappy_gen.read_stl(write_binary_stl(tmp_path / "offset.stl", tris))
    opts = base_opts(symmetry="y")
    domain, notes = snappy_gen.build_domain(surface, opts)
    assert domain.bounds[2] == pytest.approx(5.5, abs=1e-9)
    assert any("cuts the body in half" in n for n in notes)


# -- the boundary layer ------------------------------------------------------------

def test_the_first_cell_is_twice_the_centroid_distance():
    """y+ is defined on the wall-to-centroid distance. Returning that as the cell
    thickness is a factor of two, and it is why a mesh 'built for y+ 1' reports 2."""
    flow = case_gen.Flow(10.0, 1.0, 1.5e-5, 10.0 / 1.5e-5, "reynolds")
    thickness, u_tau = snappy_gen.first_layer_thickness(1.0, flow, 1.0)
    centroid = 1.0 * flow.nu / u_tau
    assert thickness == pytest.approx(2.0 * centroid, rel=1e-12)


def test_y_plus_scales_the_first_cell_linearly():
    flow = case_gen.Flow(10.0, 1.0, 1.5e-5, 10.0 / 1.5e-5, "reynolds")
    one, _ = snappy_gen.first_layer_thickness(1.0, flow, 1.0)
    fifty, _ = snappy_gen.first_layer_thickness(50.0, flow, 1.0)
    assert fifty == pytest.approx(50.0 * one, rel=1e-9)


def test_the_layer_stack_is_the_geometric_sum():
    total = snappy_gen.stack_thickness(1e-3, 5, 1.2)
    expected = 1e-3 * (1.2 ** 5 - 1) / 0.2
    assert total == pytest.approx(expected, rel=1e-12)
    assert snappy_gen.stack_thickness(1e-3, 4, 1.0) == pytest.approx(4e-3)


def test_the_ittc_line_is_the_published_one():
    """Cf = 0.075 / (log10 Re - 2)^2. At Re = 1e7 that is 0.003000."""
    assert snappy_gen.ittc57(1e7) == pytest.approx(0.075 / 25.0, rel=1e-9)


# -- the dictionaries ---------------------------------------------------------------

def test_the_generated_case_is_complete_and_runnable(tmp_path: Path, unit_box: Path):
    case = tmp_path / "case"
    rc = snappy_gen.main([
        str(case), "--stl", str(unit_box), "--speed", "10", "--nu", "1.5e-5",
        "--symmetry", "y", "--refine", "2", "--layers", "3",
    ])
    assert rc == 0
    for name in ("system/blockMeshDict", "system/snappyHexMeshDict",
                 "system/surfaceFeatureExtractDict", "system/meshQualityDict",
                 "system/controlDict", "system/fvSchemes", "system/fvSolution",
                 "constant/transportProperties", "constant/turbulenceProperties",
                 "0/U", "0/p", "0/k", "0/omega", "0/nut", "Allmesh"):
        assert (case / name).exists(), f"{name} was not written"
    assert (case / "constant" / "triSurface" / "box.stl").exists()


def test_the_symmetry_patch_reaches_every_field(tmp_path: Path, unit_box: Path):
    """A patch the mesh has and a field does not is a solver failure on step one."""
    case = tmp_path / "case"
    snappy_gen.main([str(case), "--stl", str(unit_box), "--speed", "10",
                     "--nu", "1.5e-5", "--symmetry", "y", "--refine", "1"])
    for field in ("U", "p", "k", "omega", "nut"):
        text = (case / "0" / field).read_text(encoding="utf-8")
        assert "symmetry" in text, f"0/{field} has no symmetry entry"
        assert "type            symmetry;" in text, f"0/{field} mistypes it"


def test_the_body_patch_is_named_in_the_force_object(tmp_path: Path, unit_box: Path):
    case = tmp_path / "case"
    snappy_gen.main([str(case), "--stl", str(unit_box), "--speed", "10",
                     "--nu", "1.5e-5", "--refine", "1", "--area", "wetted"])
    control = (case / "system" / "controlDict").read_text(encoding="utf-8")
    assert "patches         (box)" in control
    assert "forceCoeffs" in control and "Aref            7" in control


def test_a_thermal_case_brings_its_own_solver_and_fields(tmp_path: Path, unit_box: Path):
    case = tmp_path / "case"
    snappy_gen.main([str(case), "--stl", str(unit_box), "--speed", "6",
                     "--nu", "1.5e-5", "--study", "thermal", "--refine", "1",
                     "--wall-temperature", "333.15", "--inlet-temperature", "293.15"])
    control = (case / "system" / "controlDict").read_text(encoding="utf-8")
    assert "application     buoyantSimpleFoam" in control
    assert "wallHeatFlux" in control
    # forceCoeffs normalises by rhoInf even when it reads rho from the field; without
    # it the run aborts on the first write, after the mesh has been paid for.
    assert "rho             rho;" in control and "rhoInf          " in control
    # Every heat-transfer coefficient worth comparing with a correlation is defined
    # on the log-mean temperature difference, which needs the outlet bulk
    # temperature. Without it the obvious thing to use is (T_wall - T_inlet), which
    # on a real bundle is 10% of the answer in the pessimistic direction.
    assert "outletTemperature" in control
    for name in ("0/T", "0/alphat", "0/p_rgh", "constant/thermophysicalProperties",
                 "constant/g"):
        assert (case / name).exists(), f"{name} is missing from a thermal case"
    temperature = (case / "0" / "T").read_text(encoding="utf-8")
    assert "uniform 333.15" in temperature and "uniform 293.15" in temperature
    # p is absolute pascals here, not the kinematic p/rho of the incompressible case
    assert "[1 -1 -2 0 0 0 0]" in (case / "0" / "p").read_text(encoding="utf-8")


def test_the_thermo_viscosity_matches_the_reported_reynolds(tmp_path: Path,
                                                            unit_box: Path):
    """mu = rho * nu, so the Re in the summary is the Re the solver runs at."""
    case = tmp_path / "case"
    snappy_gen.main([str(case), "--stl", str(unit_box), "--speed", "6",
                     "--nu", "1.5e-5", "--study", "thermal", "--refine", "1",
                     "--density", "1.2"])
    thermo = (case / "constant" / "thermophysicalProperties").read_text(encoding="utf-8")
    assert "1.8e-05" in thermo.replace("1.8e-005", "1.8e-05")


def test_an_mrf_case_states_rotation_in_both_conventions(tmp_path: Path, unit_box: Path):
    """omega in rad/s for the solver, rev/s in the comment, because n in
    C_T = T/(rho n^2 D^4) is rev/s and the 2*pi is the classic factor-of-40 error."""
    case = tmp_path / "case"
    snappy_gen.main([str(case), "--stl", str(unit_box), "--speed", "0.1",
                     "--nu", "1.5e-5", "--mrf", "--mrf-rpm", "4014",
                     "--mrf-axis", "x", "--refine", "1"])
    mrf = (case / "constant" / "MRFProperties").read_text(encoding="utf-8")
    assert "cellZone        rotor" in mrf
    assert "420.3" in mrf                       # 4014 rpm in rad/s
    assert "66.9" in mrf                        # and in rev/s
    assert (case / "system" / "topoSetDict").exists()


def test_the_inside_point_is_outside_the_body(unit_box: Path):
    """locationInMesh in the geometry is the commonest way snappy produces an empty
    mesh, and the default 'centre of the domain' lands there for anything hollow."""
    surface = snappy_gen.read_stl(unit_box)
    opts = base_opts()
    domain, _ = snappy_gen.build_domain(surface, opts)
    point = snappy_gen.inside_point(domain, surface)
    x0, x1, y0, y1, z0, z1 = surface.bounds
    assert not (x0 <= point[0] <= x1 and y0 <= point[1] <= y1 and z0 <= point[2] <= z1)
    for i in range(3):
        assert domain.bounds[2 * i] < point[i] < domain.bounds[2 * i + 1]


def test_layers_that_will_not_fit_are_called_out(tmp_path: Path, unit_box: Path,
                                                 capsys):
    """snappy silently drops layers that do not fit in the surface cell, and a mesh
    reported as 'y+ 1 with 10 layers' that has none is the worst kind of wrong."""
    snappy_gen.main([str(tmp_path / "c"), "--stl", str(unit_box), "--speed", "1",
                     "--nu", "1.5e-5", "--refine", "6", "--layers", "20",
                     "--y-plus", "300", "--dry-run"])
    assert "will drop layers" in capsys.readouterr().out


def test_the_allmesh_script_is_one_call_for_the_whole_mesh(tmp_path: Path,
                                                           unit_box: Path):
    """F-37: setup round trips, not solving, are what a study spends its time on."""
    case = tmp_path / "case"
    snappy_gen.main([str(case), "--stl", str(unit_box), "--speed", "10",
                    "--nu", "1.5e-5", "--refine", "1"])
    script = (case / "Allmesh").read_text(encoding="utf-8")
    for stage in ("blockMesh", "surfaceFeatureExtract", "snappyHexMesh -overwrite",
                  "checkMesh"):
        assert stage in script, f"Allmesh does not run {stage}"
    assert "exit 1" in script, "Allmesh does not stop when a stage fails"
    # ...but checkMesh's own findings are not a stage failure: a layered snappy mesh
    # routinely reports concave cells where a layer meets a curved surface, and
    # throwing the mesh away for that is worse than reporting it.
    assert "not errors" in script


def test_a_millimetre_surface_is_questioned_not_silently_rescaled(tmp_path: Path):
    tris = box_triangles(2000.0, 1000.0, 500.0)
    surface = snappy_gen.read_stl(write_binary_stl(tmp_path / "mm.stl", tris))
    notes = snappy_gen.scale_notes(surface, 1.0)
    assert any("millimetres" in n for n in notes)
    assert snappy_gen.scale_notes(surface, 0.001) == []


def test_water_viscosity_with_air_density_is_called_out():
    """A hull run at nu = 1e-6 with rho = 1.205 gives a perfectly correct Cd beside
    a force 830x too small, and the coefficient being right is what stops anyone
    looking at the force."""
    notes = snappy_gen.fluid_notes(1.0e-6, 1.205)
    assert notes and "828" in notes[0] and "998" in notes[0]


def test_a_consistent_fluid_says_nothing():
    assert snappy_gen.fluid_notes(1.0e-6, 998.2) == []
    assert snappy_gen.fluid_notes(1.5e-5, 1.205) == []


# -- planes that bound the flow without halving the body ---------------------------

def test_a_plane_at_the_body_edge_does_not_halve_the_area(unit_box: Path):
    """The waterline case. A hull STL is closed with a flat lid at the waterline; put
    the symmetry plane there and the lid IS the plane, rather than a deck towed
    through the water and charged for its friction. Nothing is bisected, so nothing
    is halved -- and halving anyway would be the same class of error as failing to
    halve for a plane that does bisect."""
    surface = snappy_gen.read_stl(unit_box)
    opts = base_opts(area="wetted", symmetry="z:max")
    domain, notes = snappy_gen.build_domain(surface, opts)
    assert domain.halvings == 0
    area, why = snappy_gen.reference_area(surface, domain, opts)
    assert area == pytest.approx(7.0, rel=1e-6)
    assert "NOT halved" in " ".join(notes)
    # the domain's ceiling is the waterline, and the body is underneath it
    assert domain.bounds[5] == pytest.approx(surface.bounds[5], abs=1e-9)
    assert domain.bounds[4] < surface.bounds[4]


def test_two_planes_quarter_the_area_when_both_bisect(unit_box: Path):
    surface = snappy_gen.read_stl(unit_box)
    opts = base_opts(area="wetted", symmetry="y,z")
    domain, _ = snappy_gen.build_domain(surface, opts)
    assert domain.halvings == 2
    area, _ = snappy_gen.reference_area(surface, domain, opts)
    assert area == pytest.approx(7.0 / 4.0, rel=1e-6)


def test_a_bisecting_and_a_bounding_plane_together_halve_once(unit_box: Path):
    """The Wigley double body: cut on the centreline (halves) and close at the
    waterline (does not). One factor of two, not two."""
    surface = snappy_gen.read_stl(unit_box)
    opts = base_opts(area="wetted", symmetry="y,z:max")
    domain, _ = snappy_gen.build_domain(surface, opts)
    assert domain.halvings == 1
    area, _ = snappy_gen.reference_area(surface, domain, opts)
    assert area == pytest.approx(3.5, rel=1e-6)
    assert sorted(p["name"] for p in domain.planes) == ["y", "z"]
    assert sorted(n for n in domain.patches if n.startswith("symmetry")) == [
        "symmetryY", "symmetryZ"]


def test_an_explicit_coordinate_is_honoured(unit_box: Path):
    surface = snappy_gen.read_stl(unit_box)
    planes = snappy_gen.parse_symmetry("z:0.125", surface)
    assert planes[0]["plane"] == pytest.approx(0.125)
    assert planes[0]["bisects"] is True


def test_a_bad_symmetry_spec_is_refused(unit_box: Path):
    surface = snappy_gen.read_stl(unit_box)
    with pytest.raises(SystemExit, match="not one of"):
        snappy_gen.parse_symmetry("w", surface)
    with pytest.raises(SystemExit, match="expected an axis"):
        snappy_gen.parse_symmetry("z:middle", surface)


def test_every_symmetry_face_becomes_a_symmetry_patch(tmp_path: Path, unit_box: Path):
    case = tmp_path / "case"
    snappy_gen.main([str(case), "--stl", str(unit_box), "--speed", "10",
                     "--nu", "1.5e-5", "--symmetry", "y,z:max", "--refine", "1"])
    block = (case / "system" / "blockMeshDict").read_text(encoding="utf-8")
    # Two planes, two patches. One patch holding both is rejected by blockMesh:
    # symmetryPlane is a constraint whose faces must be coplanar, and a centreline
    # and a waterline are perpendicular.
    assert block.count("type            symmetry;") == 2
    assert "symmetryY" in block and "symmetryZ" in block
    for field in ("U", "p", "k", "omega", "nut"):
        text = (case / "0" / field).read_text(encoding="utf-8")
        assert text.count("type            symmetry;") == 2, \
            f"0/{field} misses a symmetry patch"


def test_the_kept_half_is_the_one_the_body_is_in(unit_box: Path):
    """A hull closed at its waterline lies below that plane. Cutting the domain's
    floor there instead of its ceiling leaves the body outside the mesh, and snappy
    reports that as an empty box rather than as an error."""
    surface = snappy_gen.read_stl(unit_box)
    top = snappy_gen.parse_symmetry("z:max", surface)[0]
    bottom = snappy_gen.parse_symmetry("z:min", surface)[0]
    mid = snappy_gen.parse_symmetry("z", surface)[0]
    assert top["keep"] == "low"
    assert bottom["keep"] == "high"
    assert mid["keep"] == "high"


def test_far_field_symmetry_types_the_mesh_patches_too(tmp_path: Path, unit_box: Path):
    """`--far symmetry` makes the tunnel walls symmetry patches. Typing them `patch`
    in blockMesh while the 0/ files call them symmetry is not caught until
    decomposePar reads the case and aborts with "attempt to cast type patch to type
    symmetryPlane" -- after the mesh has been built and paid for."""
    case = tmp_path / "case"
    snappy_gen.main([str(case), "--stl", str(unit_box), "--speed", "6",
                     "--nu", "1.5e-5", "--far", "symmetry", "--refine", "1"])
    block = (case / "system" / "blockMeshDict").read_text(encoding="utf-8")
    field = (case / "0" / "U").read_text(encoding="utf-8")
    for patch in ("top", "bottom", "side"):
        assert patch in block and patch in field
    # every patch the fields call symmetry is typed symmetry in the mesh
    assert block.count("type            symmetry;") == field.count("type            symmetry;")


def test_a_two_faced_patch_is_never_a_symmetry_plane(tmp_path: Path, unit_box: Path):
    """`side` carries both tunnel walls. They are parallel, not coplanar, so the
    strict symmetryPlane constraint rejects the patch; the general symmetry type
    takes it."""
    case = tmp_path / "case"
    snappy_gen.main([str(case), "--stl", str(unit_box), "--speed", "6",
                     "--nu", "1.5e-5", "--far", "symmetry", "--refine", "1"])
    block = (case / "system" / "blockMeshDict").read_text(encoding="utf-8")
    assert "symmetryPlane" not in block


def test_the_mrf_zone_contains_the_rotor(tmp_path: Path):
    """A cylinder sized off the disc but a quarter of it thick is thinner than the
    blades are long axially, so the tips sit outside the rotating frame, feel no
    rotation, and the thrust is wrong with nothing reporting it."""
    # 40 mm along the axis, 250 mm across: a propeller's proportions.
    tris = box_triangles(0.04, 0.25, 0.25)
    stl = write_binary_stl(tmp_path / "rotor.stl", tris)
    case = tmp_path / "case"
    snappy_gen.main([str(case), "--stl", str(stl), "--speed", "5", "--nu", "1.5e-5",
                     "--mrf", "--mrf-rpm", "4000", "--mrf-axis", "x", "--refine", "1"])
    topo = (case / "system" / "topoSetDict").read_text(encoding="utf-8")
    p1 = [float(v) for v in topo.split("point1      (")[1].split(")")[0].split()]
    p2 = [float(v) for v in topo.split("point2      (")[1].split(")")[0].split()]
    radius = float(topo.split("radius      ")[1].split(";")[0])
    surface = snappy_gen.read_stl(stl)
    assert p1[0] <= surface.bounds[0] and p2[0] >= surface.bounds[1],         "the zone is shorter than the rotor along its own axis"
    assert 2 * radius >= max(surface.extent[1], surface.extent[2])


def test_the_inside_point_never_lands_on_a_cell_face(unit_box: Path):
    """snappy refuses a locationInMesh that sits on a face or an edge, and reports it
    by printing a bounding box that plainly contains the point -- which sends you to
    the domain instead of to the arithmetic. Half of an even cell count is a face."""
    surface = snappy_gen.read_stl(unit_box)
    for cells in range(4, 41):
        opts = base_opts(symmetry="y", background_cells=cells)
        domain, _ = snappy_gen.build_domain(surface, opts)
        point = snappy_gen.inside_point(domain, surface)
        for i in range(3):
            lo, hi = domain.bounds[2 * i], domain.bounds[2 * i + 1]
            step = (hi - lo) / domain.cells[i]
            offset = (point[i] - lo) / step
            assert abs(offset - round(offset)) > 1e-3, (
                f"{cells} background cells put the point on a face on axis {i}")


def test_the_thermal_case_uses_plain_simple(tmp_path: Path, unit_box: Path):
    """SIMPLEC with a 0.3 pressure factor is conservative on both counts and
    converges too slowly for a fixed iteration budget -- which on a heat-transfer
    case means the wall heat flux, the entire answer, has not settled."""
    case = tmp_path / "case"
    snappy_gen.main([str(case), "--stl", str(unit_box), "--speed", "6",
                     "--nu", "1.5e-5", "--study", "thermal", "--refine", "1"])
    text = (case / "system" / "fvSolution").read_text(encoding="utf-8")
    assert "consistent      no;" in text
    assert "p_rgh           0.3;" in text and "U               0.7;" in text

    incompressible = tmp_path / "inc"
    snappy_gen.main([str(incompressible), "--stl", str(unit_box), "--speed", "6",
                     "--nu", "1.5e-5", "--refine", "1"])
    other = (incompressible / "system" / "fvSolution").read_text(encoding="utf-8")
    assert "consistent      yes;" in other
    assert "p           0.7;" in other and "U               0.9;" in other


# -- pointing an uploaded body at the flow -----------------------------------------

def test_rotation_carries_z_onto_x(tmp_path: Path):
    """y:90 is the turn that points a shaft-along-z rotor down the tunnel."""
    tris = box_triangles(0.04, 0.25, 0.01)
    surface = snappy_gen.read_stl(write_binary_stl(tmp_path / "r.stl", tris))
    turned = snappy_gen.rotated(surface, snappy_gen.parse_rotation("y:90"))
    before, after = surface.extent, turned.extent
    assert after[0] == pytest.approx(before[2], abs=1e-6)
    assert after[1] == pytest.approx(before[1], abs=1e-6)
    assert after[2] == pytest.approx(before[0], abs=1e-6)
    # a rotation is rigid: area is unchanged
    assert turned.wetted_area() == pytest.approx(surface.wetted_area(), rel=1e-5)


def test_a_rotor_spun_about_the_wrong_axis_is_called_out(tmp_path: Path):
    """Looking down a propeller's shaft you see its blade planform, much the largest
    of the three projections -- so the axis of maximum projected area IS the shaft.
    Spinning it about a line lying in its own disc gives a plausible thrust beside a
    torque an order of magnitude too big, which reads as a mesh problem and is not."""
    # broad in x-y, thin in z: the "disc" faces down z
    surface = snappy_gen.read_stl(
        write_binary_stl(tmp_path / "disc.stl", box_triangles(0.2, 0.2, 0.01)))
    bad = snappy_gen.axis_notes(surface, "x")
    assert any(n.startswith("!!") and "shaft is z" in n for n in bad)
    assert snappy_gen.axis_notes(surface, "z") == [
        n for n in snappy_gen.axis_notes(surface, "z") if not n.startswith("!!")]


def test_the_projected_areas_are_always_reported(tmp_path: Path, unit_box: Path):
    surface = snappy_gen.read_stl(unit_box)
    notes = snappy_gen.axis_notes(surface, None)
    assert notes and notes[0].startswith("projected area")
    for name in ("down x", "down y", "down z"):
        assert name in notes[0]


def test_a_rotated_case_writes_the_surface_it_meshed(tmp_path: Path):
    """The STL in the case must be the one the domain was built around. Copying the
    original would put the body somewhere the mesh is not, which snappy reports as
    an empty mesh rather than as a mismatch."""
    tris = box_triangles(0.04, 0.25, 0.01)
    stl = write_binary_stl(tmp_path / "r.stl", tris)
    case = tmp_path / "case"
    snappy_gen.main([str(case), "--stl", str(stl), "--speed", "5", "--nu", "1.5e-5",
                     "--rotate", "y:90", "--refine", "1"])
    written = snappy_gen.read_stl(case / "constant" / "triSurface" / "r.stl")
    assert written.extent[0] == pytest.approx(0.01, abs=1e-6)
    assert written.extent[2] == pytest.approx(0.04, abs=1e-6)


def test_a_bad_rotation_spec_is_refused():
    with pytest.raises(SystemExit, match="axis:degrees"):
        snappy_gen.parse_rotation("90")
    with pytest.raises(SystemExit, match="not a number"):
        snappy_gen.parse_rotation("y:sideways")


def test_layer_addition_controls_are_reachable(tmp_path: Path, unit_box: Path):
    """maxThicknessToMedialRatio decides whether a stack gets built at all, and the
    tutorial 0.3 assumes a chunky body. A thin one -- a hull whose bow and stern come
    to a knife edge -- has a medial distance going to zero there, and snappy responds
    by declining to build the layers and saying nothing."""
    case = tmp_path / "case"
    snappy_gen.main([str(case), "--stl", str(unit_box), "--speed", "10",
                     "--nu", "1.5e-5", "--refine", "1", "--layers", "6",
                     "--medial-ratio", "0.9", "--layer-iter", "80"])
    text = (case / "system" / "snappyHexMeshDict").read_text(encoding="utf-8")
    assert "maxThicknessToMedialRatio 0.9;" in text
    assert "nLayerIter      80;" in text
    assert "nRelaxedIter    40;" in text


def test_the_default_medial_ratio_suits_a_thin_body(tmp_path: Path, unit_box: Path):
    case = tmp_path / "case"
    snappy_gen.main([str(case), "--stl", str(unit_box), "--speed", "10",
                     "--nu", "1.5e-5", "--refine", "1"])
    text = (case / "system" / "snappyHexMeshDict").read_text(encoding="utf-8")
    assert "maxThicknessToMedialRatio 0.6;" in text


def test_the_inside_point_survives_a_body_that_overhangs_the_domain(unit_box: Path):
    """A domain cut deliberately INSIDE the geometry -- tubes trimmed by their own
    end planes, so they span the bank the way the correlation assumes -- puts the
    body's bound outside the box. Stepping a fraction of the way towards it walks
    out through the floor, and snappy rejects the point while printing a bounding
    box that does not contain it."""
    surface = snappy_gen.read_stl(unit_box)
    # negative margins: the box ends inside the body on every axis
    opts = base_opts(ahead=-0.02, behind=-0.02, side=-0.02, above=-0.02, below=-0.02)
    domain, _ = snappy_gen.build_domain(surface, opts)
    point = snappy_gen.inside_point(domain, surface)
    for i in range(3):
        assert domain.bounds[2 * i] < point[i] < domain.bounds[2 * i + 1], (
            f"axis {i}: {point[i]} outside "
            f"{domain.bounds[2 * i]}..{domain.bounds[2 * i + 1]}")


def test_a_transient_thermal_case_is_buoyant_pimple(tmp_path: Path, unit_box: Path):
    """Steady RANS suppresses the unsteady wake that carries heat off a bluff body,
    so a bank of cylinders comes out 15-30% low however good the mesh is. Running it
    in time is the only way out, and it needs a different solver."""
    case = tmp_path / "case"
    snappy_gen.main([str(case), "--stl", str(unit_box), "--speed", "6",
                     "--nu", "1.5e-5", "--study", "thermal-transient",
                     "--refine", "1", "--end-time", "0.05", "--courant", "2"])
    control = (case / "system" / "controlDict").read_text(encoding="utf-8")
    assert "application     buoyantPimpleFoam" in control
    assert "adjustTimeStep  yes;" in control and "maxCo           2;" in control
    # the heat-flux probe must be dense enough in time to average over periods
    assert "writeControl    timeStep;" in control
    solution = (case / "system" / "fvSolution").read_text(encoding="utf-8")
    assert "PIMPLE" in solution and "residualControl" not in solution
    assert "p_rghFinal" in solution and "rhoMax" in solution
    # buoyantPimpleFoam carries a density equation the steady solver does not, and
    # without an entry for it the run dies on its first step.
    assert '"rho.*"' in solution and "solver          diagonal;" in solution
    schemes = (case / "system" / "fvSchemes").read_text(encoding="utf-8")
    assert "default         Euler;" in schemes
    # it is still a thermal case: same fields, same thermophysics
    for name in ("0/T", "0/alphat", "0/p_rgh", "constant/thermophysicalProperties"):
        assert (case / name).exists(), f"{name} missing from a transient thermal case"


def test_the_steady_thermal_case_is_unchanged(tmp_path: Path, unit_box: Path):
    case = tmp_path / "case"
    snappy_gen.main([str(case), "--stl", str(unit_box), "--speed", "6",
                     "--nu", "1.5e-5", "--study", "thermal", "--refine", "1"])
    control = (case / "system" / "controlDict").read_text(encoding="utf-8")
    assert "application     buoyantSimpleFoam" in control
    assert "writeControl    writeTime;" in control


def test_a_transition_case_carries_its_two_extra_fields(tmp_path: Path, unit_box: Path):
    """Below Re ~ 5e5 a blade carries a laminar separation bubble over much of its
    chord. A fully turbulent model assumes it turbulent from the leading edge, and
    the lift comes out low however fine the mesh -- which is a different question
    from the one being asked, not a coarser answer to it."""
    case = tmp_path / "case"
    snappy_gen.main([str(case), "--stl", str(unit_box), "--speed", "10",
                     "--nu", "1.5e-5", "--turbulence", "kOmegaSSTLM", "--refine", "1"])
    for name in ("0/k", "0/omega", "0/gammaInt", "0/ReThetat", "0/nut"):
        assert (case / name).exists(), f"{name} missing from a transition case"
    turb = (case / "constant" / "turbulenceProperties").read_text(encoding="utf-8")
    assert "kOmegaSSTLM" in turb
    gamma = (case / "0" / "gammaInt").read_text(encoding="utf-8")
    assert "internalField   uniform 1;" in gamma
    # every transported field needs a scheme for its convection term; a field in 0/
    # with no entry in divSchemes is not caught until the solver stops on it, after
    # the mesh is built and the case decomposed.
    schemes = (case / "system" / "fvSchemes").read_text(encoding="utf-8")
    for field in case_gen.TURBULENCE_FIELDS["kOmegaSSTLM"]:
        assert f"div(phi,{field})" in schemes, f"no divScheme for {field}"


def test_re_theta_follows_langtry_menter(tmp_path: Path):
    """Tu is in PERCENT in this correlation, and the 1/Tu^2 term makes a factor of
    100 spectacular rather than subtle."""
    # Tu = 1% -> 1173.51 - 589.428 + 0.2196 = 584.30
    assert case_gen.free_stream_re_theta(0.01) == pytest.approx(584.30, abs=0.05)
    # high turbulence takes the other branch and must fall, not rise
    assert case_gen.free_stream_re_theta(0.05) < case_gen.free_stream_re_theta(0.01)
    # and it must stay finite as Tu goes to zero
    assert case_gen.free_stream_re_theta(0.0) < 1e5


def test_the_buffer_layer_is_called_out():
    """5 < y+ < 30 is where neither wall treatment holds. Four meshes of one hull at
    y+ 4.3, 9.6, 16.9 and 33.5 gave form factors 0.786, 0.928, 0.977, 1.018 -- the
    first three look like a convergence trend and are three invalid points."""
    assert any("buffer layer" in n for n in snappy_gen.y_plus_notes(16.9, 3))
    assert any("buffer layer" in n for n in snappy_gen.y_plus_notes(9.6, 5))
    assert snappy_gen.y_plus_notes(35.0, 3) == []      # wall-function regime, fine
    assert snappy_gen.y_plus_notes(1.0, 12) == []      # resolved, with enough layers


def test_resolving_with_too_few_layers_is_called_out():
    """Asking for y+ 1 with three layers puts the SECOND cell in the buffer layer,
    which undoes the first."""
    notes = snappy_gen.y_plus_notes(1.0, 3)
    assert notes and "10 or more" in notes[0]


def test_a_confined_flow_does_not_get_a_wind_tunnel_inlet(tmp_path: Path, unit_box: Path):
    """0.1% is right for a body in the open and wrong for anything the geometry
    confines: past the first row a bank makes its own turbulence."""
    open_case, confined = tmp_path / "open", tmp_path / "confined"
    snappy_gen.main([str(open_case), "--stl", str(unit_box), "--speed", "6",
                     "--nu", "1.5e-5", "--refine", "1"])
    snappy_gen.main([str(confined), "--stl", str(unit_box), "--speed", "6",
                     "--nu", "1.5e-5", "--refine", "1", "--far", "symmetry"])
    k_open = float((open_case / "0" / "k").read_text(encoding="utf-8")
                   .split("internalField   uniform ")[1].split(";")[0])
    k_confined = float((confined / "0" / "k").read_text(encoding="utf-8")
                       .split("internalField   uniform ")[1].split(";")[0])
    # k goes as intensity squared: 0.05/0.001 = 50, so 2500x
    assert k_confined / k_open == pytest.approx(2500.0, rel=0.05)


def test_an_explicit_intensity_still_wins(tmp_path: Path, unit_box: Path):
    case = tmp_path / "c"
    snappy_gen.main([str(case), "--stl", str(unit_box), "--speed", "6", "--nu", "1.5e-5",
                     "--refine", "1", "--far", "symmetry",
                     "--turbulent-intensity", "0.001"])
    k = float((case / "0" / "k").read_text(encoding="utf-8")
              .split("internalField   uniform ")[1].split(";")[0])
    assert k == pytest.approx(1.5 * (0.001 * 6) ** 2, rel=1e-6)
