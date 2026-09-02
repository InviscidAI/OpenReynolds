"""preflight.py -- the gatekeeper's checks, each against a case built to fail it.

Every check here is a claim about a real OpenFOAM failure mode, so the tests are
mostly one synthetic case per failure: a boundary file and a 0/ that disagree, an
empty patch one field calls zeroGradient, an STL a thousand times too big. The
solver probe is the only check that shells out, and its runner is injected, so no
test needs OpenFOAM on the machine.

Nothing here imports pyvista: preflight does not need a graphics stack and must not
grow one, since its whole value is being cheap enough to run before anything else.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load(name: str):
    """Import a toolbox script by path; the directory is data, not a package."""
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def preflight():
    return load("preflight")


@pytest.fixture
def study_state():
    return load("study_state")


# -- a case that passes, and the pieces to break in it ------------------------------


CONTROL_DICT = """\
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }

application     pimpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          0.001;
writeControl    runTime;
writeInterval   0.1;
writeFormat     ascii;
purgeWrite      0;

functions
{
    forces1
    {
        type            forceCoeffs;
        patches         (cylinder);
        rho             rhoInf;
        rhoInf          1.225;
        magUInf         1;
        lRef            0.1;
        Aref            0.01;
        writeControl    timeStep;
        writeInterval   5;
    }
}
"""

BLOCK_MESH_DICT = """\
convertToMeters 1;
vertices
(
    (0 0 0) (2 0 0) (2 1 0) (0 1 0)
    (0 0 0.1) (2 0 0.1) (2 1 0.1) (0 1 0.1)
);
blocks
(
    hex (0 1 2 3 4 5 6 7) (100 50 1) simpleGrading (1 1 1)
);
"""

BOUNDARY = """\
FoamFile { version 2.0; format ascii; class polyBoundaryMesh; object boundary; }

4
(
    inlet        { type patch; nFaces 50;    startFace 9000; }
    outlet       { type patch; nFaces 50;    startFace 9050; }
    cylinder     { type wall;  nFaces 200;   startFace 9100; }
    frontAndBack { type empty; nFaces 10000; startFace 9300; }
)
"""

OWNER = 'FoamFile { note "nPoints:10302 nCells:5000 nFaces:20000 nInternalFaces:9000"; }\n'

FIELD_U = """\
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{
    inlet        { type fixedValue; value uniform (1 0 0); }
    outlet       { type zeroGradient; }
    cylinder     { type noSlip; }
    frontAndBack { type empty; }
}
"""

FIELD_P = """\
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    inlet        { type zeroGradient; }
    outlet       { type fixedValue; value uniform 0; }
    cylinder     { type zeroGradient; }
    frontAndBack { type empty; }
}
"""

CHECK_MESH_LOG = """\
Mesh stats
    points:           10302
    cells:             5000
Checking geometry...
    Max aspect ratio = 3.24 OK.
    Min volume = 8e-09. Max volume = 3.4e-08.  Total volume = 0.2.
    Mesh non-orthogonality Max: 40.1 average: 6.2
    Max skewness = 1.92 OK.
Mesh OK.
"""

SOLVER_LOG = """\
Time = 0.001

smoothSolver:  Solving for Ux, Initial residual = 1.0e-01, Final residual = 2.0e-03, No Iterations 3
GAMG:  Solving for p, Initial residual = 9.0e-01, Final residual = 1.0e-02, No Iterations 9
time step continuity errors : sum local = 1.5e-06, global = -2.0e-09, cumulative = -3.0e-09
ExecutionTime = 1.40 s

Time = 0.002

smoothSolver:  Solving for Ux, Initial residual = 5.0e-03, Final residual = 1.0e-04, No Iterations 3
GAMG:  Solving for p, Initial residual = 4.0e-02, Final residual = 5.0e-04, No Iterations 7
time step continuity errors : sum local = 7.5e-07, global = -1.0e-09, cumulative = -4.0e-09
ExecutionTime = 2.80 s
"""


def write_case(root: Path) -> Path:
    """A small 2D cylinder case with nothing wrong with it."""
    root = Path(root)
    (root / "system").mkdir(parents=True, exist_ok=True)
    (root / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
    (root / "0").mkdir(parents=True, exist_ok=True)

    (root / "system" / "controlDict").write_text(CONTROL_DICT)
    (root / "system" / "blockMeshDict").write_text(BLOCK_MESH_DICT)
    (root / "constant" / "transportProperties").write_text(
        "nu              [0 2 -1 0 0 0 0] 1e-03;\n"
    )
    (root / "constant" / "polyMesh" / "boundary").write_text(BOUNDARY)
    (root / "constant" / "polyMesh" / "owner").write_text(OWNER)
    (root / "0" / "U").write_text(FIELD_U)
    (root / "0" / "p").write_text(FIELD_P)
    (root / "log.checkMesh").write_text(CHECK_MESH_LOG)
    (root / "log.pimpleFoam").write_text(SOLVER_LOG)
    return root


@pytest.fixture
def case(tmp_path):
    return write_case(tmp_path / "cylinder")


def finding_for(findings, check):
    matches = [item for item in findings if item.check == check]
    assert matches, f"no finding for {check}"
    return matches[0]


def ascii_stl(path: Path, triangles) -> Path:
    lines = ["solid body"]
    for triangle in triangles:
        lines.append("  facet normal 0 0 1\n    outer loop")
        lines += [f"      vertex {v[0]} {v[1]} {v[2]}" for v in triangle]
        lines.append("    endloop\n  endfacet")
    lines.append("endsolid body")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def binary_stl(path: Path, triangles) -> Path:
    payload = b"\0" * 80 + struct.pack("<I", len(triangles))
    for triangle in triangles:
        payload += struct.pack("<3f", 0.0, 0.0, 1.0)
        for vertex in triangle:
            payload += struct.pack("<3f", *vertex)
        payload += struct.pack("<H", 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


A, B, C, D = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)

CLOSED_TETRAHEDRON = [(A, C, B), (A, B, D), (B, C, D), (C, A, D)]
"""Four faces wound consistently outwards: every edge is walked once in each
direction, which is what a watertight surface looks like to the edge bookkeeping."""


# -- it must stay cheap to import ---------------------------------------------------


def test_pyvista_is_not_imported_anywhere_in_preflight():
    """A preflight that needs a graphics stack is a preflight nobody runs first."""
    tree = ast.parse((TOOLBOX / "preflight.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "pyvista" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "pyvista"


def test_the_docstring_says_the_agent_decides():
    """The free-will contract: a gatekeeper that reads as an order is a workflow
    injection wearing a check's clothes."""
    doc = ast.get_docstring(ast.parse((TOOLBOX / "preflight.py").read_text(encoding="utf-8")))
    normalised = " ".join(doc.lower().split())
    assert "suggestion" in normalised
    assert "yours to decide" in normalised
    assert "blocks nothing" in normalised
    for imperative in ("you must", "always run", "before you run this you", "step 1"):
        assert imperative not in normalised


# -- dictionary reading -------------------------------------------------------------


def test_block_body_is_brace_matched_not_regex_delimited(preflight):
    """boundaryField blocks nest, so stopping at the first '}' stops inside the
    first patch and loses every patch after it."""
    body = preflight.block_body(FIELD_U, "boundaryField")
    assert "inlet" in body and "frontAndBack" in body


def test_block_body_of_a_missing_keyword_is_empty(preflight):
    assert preflight.block_body(FIELD_U, "functions") == ""


def test_block_entries_takes_the_last_token_before_the_brace(preflight):
    """An `#includeEtc` line sitting above a patch entry must not be glued onto it."""
    body = """
    #includeEtc "caseDicts/setConstraintTypes"
    inlet { type fixedValue; }
    """
    assert [name for name, _ in preflight.block_entries(body)] == ["inlet"]


def test_parse_control_ignores_the_functions_block(preflight):
    """A function object's own writeInterval is not the case's write cadence."""
    control = preflight.parse_control(CONTROL_DICT)
    assert control["application"] == "pimpleFoam"
    assert control["writeInterval"] == "0.1"
    assert control["writeControl"] == "runTime"


def test_parse_boundary_reads_names_types_and_face_counts(preflight):
    patches = preflight.parse_boundary(BOUNDARY)
    assert [patch["name"] for patch in patches] == ["inlet", "outlet", "cylinder", "frontAndBack"]
    assert patches[3]["type"] == "empty"
    assert patches[2]["nFaces"] == 200


def test_a_patch_written_on_one_line_still_yields_every_entry(preflight):
    """Regression: reading one `key value;` per line lost `nFaces` on every
    hand-written boundary file, which put the face counts at zero across the board."""
    values = preflight.entry_values(" type wall; nFaces 200;  startFace 9100; ")
    assert values == {"type": "wall", "nFaces": "200", "startFace": "9100"}


def test_outer_text_drops_nested_blocks_without_joining_what_surrounded_them(preflight):
    text = preflight.outer_text("alpha 1; inner { beta 2; } gamma 3;")
    values = preflight.entry_values(text)
    assert values == {"alpha": "1", "gamma": "3"}


def test_parse_boundary_field_separates_literal_names_from_patterns(preflight):
    text = """
    boundaryField
    {
        inlet { type fixedValue; }
        "(top|bottom)" { type zeroGradient; }
        #includeEtc "caseDicts/setConstraintTypes"
    }
    """
    spec = preflight.parse_boundary_field(text)
    assert spec["names"] == ["inlet"]
    assert spec["patterns"] == ["(top|bottom)"]
    assert spec["includes"] is True
    assert spec["types"]["inlet"] == "fixedValue"


def test_parse_dimensions_and_nu_and_uniform_velocity(preflight):
    assert preflight.parse_dimensions(FIELD_P) == (0, 2, -2, 0, 0, 0, 0)
    assert preflight.parse_nu("nu [0 2 -1 0 0 0 0] 1e-05;") == pytest.approx(1e-05)
    assert preflight.parse_uniform_velocity(FIELD_U) == pytest.approx(1.0)


def test_a_zero_viscosity_is_reported_rather_than_treated_as_absent(preflight):
    """`nu 0;` and "no nu entry" are different problems with different repairs."""
    assert preflight.parse_nu("nu  0;") == 0.0


def test_parse_owner_note_is_the_cheap_cell_count(preflight):
    assert preflight.parse_owner_note(OWNER)["nCells"] == 5000


def test_field_components_tells_a_vector_from_a_scalar(preflight):
    assert preflight.field_components(FIELD_U) == 3
    assert preflight.field_components(FIELD_P) == 1


def test_a_gzipped_field_is_still_read(preflight, case):
    import gzip

    (case / "0" / "U").unlink()
    with gzip.open(case / "0" / "U.gz", "wt", encoding="utf-8") as handle:
        handle.write(FIELD_U)
    assert "frontAndBack" in preflight.Case(case).field_texts["U"]


# -- geometry: scale --------------------------------------------------------------


def test_a_metre_sized_surface_in_a_metre_sized_domain_is_fine(preflight):
    assert preflight.scale_diagnosis([0.2, 0.2, 0.1], [2.0, 1.0, 0.1])["status"] == "ok"


def test_a_millimetre_surface_against_a_metre_domain_fails(preflight):
    """The classic: CAD exports mm, OpenFOAM reads m, snappy meshes nothing."""
    result = preflight.scale_diagnosis([200.0, 200.0, 100.0], [2.0, 1.0, 0.1])
    assert result["status"] == "fail"
    assert "0.001" in result["repair"]


def test_a_surface_far_smaller_than_its_domain_also_fails(preflight):
    result = preflight.scale_diagnosis([0.002, 0.002, 0.001], [2.0, 1.0, 0.1])
    assert result["status"] == "fail"
    assert "speck" in result["note"]


def test_a_large_surface_with_no_domain_to_compare_is_only_a_warning(preflight):
    """A 300 m bounding box is a building as often as it is a units error."""
    result = preflight.scale_diagnosis([300.0, 50.0, 40.0])
    assert result["status"] == "warn"
    assert "millimetres" in result["note"]


def test_the_stated_length_scale_is_used_when_there_is_no_domain(preflight):
    result = preflight.scale_diagnosis([100.0, 100.0, 3.0], None, length=0.1)
    assert result["status"] == "warn"
    assert "1000" in result["note"]


def test_a_zero_sized_bounding_box_fails(preflight):
    assert preflight.scale_diagnosis([0.0, 0.0, 0.0])["status"] == "fail"


# -- geometry: topology -----------------------------------------------------------


@pytest.mark.parametrize("writer", [ascii_stl, binary_stl])
def test_read_triangles_handles_either_stl_format(tmp_path, preflight, writer):
    triangles = preflight.read_triangles(writer(tmp_path / "body.stl", CLOSED_TETRAHEDRON))
    assert triangles.shape == (4, 3, 3)


def test_a_closed_consistent_surface_has_no_open_or_flipped_edges(tmp_path, preflight):
    """An STL stores every corner three times over, so the vertices have to be welded
    before any edge count means anything -- without the weld nothing is ever closed."""
    triangles = preflight.read_triangles(ascii_stl(tmp_path / "t.stl", CLOSED_TETRAHEDRON))
    topology = preflight.surface_topology(triangles)
    assert topology["open_edges"] == 0
    assert topology["flipped_edges"] == 0
    assert topology["non_manifold_edges"] == 0


def test_a_missing_face_shows_up_as_open_edges(tmp_path, preflight):
    triangles = preflight.read_triangles(ascii_stl(tmp_path / "t.stl", CLOSED_TETRAHEDRON[:3]))
    assert preflight.surface_topology(triangles)["open_edges"] == 3


def test_one_face_wound_backwards_shows_up_as_flipped_edges(tmp_path, preflight):
    """Geometrically watertight, and snappy still meshes it as though it had a hole."""
    faces = list(CLOSED_TETRAHEDRON)
    faces[0] = tuple(reversed(faces[0]))
    triangles = preflight.read_triangles(ascii_stl(tmp_path / "t.stl", faces))
    topology = preflight.surface_topology(triangles)
    assert topology["open_edges"] == 0
    assert topology["flipped_edges"] == 3


def test_three_triangles_on_one_edge_is_non_manifold(tmp_path, preflight):
    extra = (A, B, (0, -1, 0))
    triangles = preflight.read_triangles(ascii_stl(tmp_path / "t.stl", CLOSED_TETRAHEDRON + [extra]))
    assert preflight.surface_topology(triangles)["non_manifold_edges"] >= 1


def test_a_repeated_corner_is_counted_and_does_not_break_the_edge_count(tmp_path, preflight):
    flat = (A, B, B)
    triangles = preflight.read_triangles(ascii_stl(tmp_path / "t.stl", CLOSED_TETRAHEDRON + [flat]))
    topology = preflight.surface_topology(triangles)
    assert topology["degenerate_triangles"] == 1
    assert topology["computed"] is True


def test_topology_is_not_attempted_on_an_enormous_surface(preflight):
    import numpy as np

    huge = np.zeros((preflight.TOPOLOGY_TRIANGLE_LIMIT + 1, 3, 3))
    topology = preflight.surface_topology(huge)
    assert topology["computed"] is False
    assert "not counted" in topology["note"]


def test_no_triangles_at_all_is_reported_rather_than_raised(preflight):
    assert preflight.surface_topology(None)["computed"] is False


def test_the_geometry_check_reads_a_real_stl_out_of_the_case(preflight, case):
    ascii_stl(case / "constant" / "triSurface" / "body.stl", CLOSED_TETRAHEDRON)
    finding = finding_for(preflight.run_checks(case, ["geometry"]), "geometry")
    assert finding.status == "ok"
    assert "body.stl" in finding.measured
    assert "4 triangles" in finding.measured


def test_the_geometry_check_is_skipped_with_no_surfaces(preflight, case):
    assert finding_for(preflight.run_checks(case, ["geometry"]), "geometry").status == "skipped"


def test_a_millimetre_stl_fails_the_geometry_check_against_the_case_domain(preflight, case):
    """End to end: the blockMeshDict says the domain is 2 m across and the STL says
    the object in it is 1000 m across."""
    millimetres = [
        tuple(tuple(coordinate * 1000 for coordinate in vertex) for vertex in face)
        for face in CLOSED_TETRAHEDRON
    ]
    ascii_stl(case / "constant" / "triSurface" / "body.stl", millimetres)
    finding = finding_for(preflight.run_checks(case, ["geometry"]), "geometry")
    assert finding.status == "fail"
    assert "surfaceTransformPoints" in finding.repair


# -- patch-name consistency ---------------------------------------------------------


def spec_for(preflight, *names, patterns=(), includes=False, types=None):
    return {
        "names": list(names),
        "patterns": list(patterns),
        "types": types or {},
        "includes": includes,
        "present": True,
    }


def test_a_field_covering_every_patch_is_clean(preflight):
    mesh = preflight.parse_boundary(BOUNDARY)
    fields = {"U": preflight.parse_boundary_field(FIELD_U)}
    row = preflight.patch_consistency(mesh, fields)[0]
    assert row["missing"] == [] and row["extra"] == []


def test_a_patch_with_no_entry_is_reported_by_name(preflight):
    mesh = preflight.parse_boundary(BOUNDARY)
    fields = {"p": preflight.parse_boundary_field(FIELD_P.replace(
        "    outlet       { type fixedValue; value uniform 0; }\n", ""
    ))}
    assert preflight.patch_consistency(mesh, fields)[0]["missing"] == ["outlet"]


def test_an_entry_naming_no_patch_is_reported_too(preflight):
    """The quieter half: a spare entry is ignored without a message, so the boundary
    condition somebody set is simply never applied."""
    mesh = preflight.parse_boundary(BOUNDARY)
    fields = {"p": preflight.parse_boundary_field(
        FIELD_P.replace("cylinder ", "cylinderWall ")
    )}
    row = preflight.patch_consistency(mesh, fields)[0]
    assert row["missing"] == ["cylinder"]
    assert row["extra"] == ["cylinderWall"]


def test_a_regex_key_covers_the_patches_it_matches(preflight):
    mesh = preflight.parse_boundary(BOUNDARY)
    fields = {"k": spec_for(preflight, "frontAndBack", patterns=[".*"])}
    assert preflight.patch_consistency(mesh, fields)[0]["missing"] == []


def test_an_includeetc_covers_the_constraint_patches_only(preflight):
    """setConstraintTypes writes the empty/cyclic/wedge entries and nothing else, so
    it excuses frontAndBack and does not excuse inlet."""
    mesh = preflight.parse_boundary(BOUNDARY)
    fields = {"nut": spec_for(preflight, "outlet", "cylinder", includes=True)}
    assert preflight.patch_consistency(mesh, fields)[0]["missing"] == ["inlet"]


def test_a_broken_regex_key_does_not_take_the_check_down(preflight):
    mesh = preflight.parse_boundary(BOUNDARY)
    fields = {"k": spec_for(preflight, patterns=["(unclosed"])}
    row = preflight.patch_consistency(mesh, fields)[0]
    assert len(row["missing"]) == 4


def test_the_patches_check_passes_on_a_consistent_case(preflight, case):
    assert finding_for(preflight.run_checks(case, ["patches"]), "patches").status == "ok"


def test_the_patches_check_names_the_field_and_the_patch(preflight, case):
    (case / "0" / "p").write_text(FIELD_P.replace(
        "    outlet       { type fixedValue; value uniform 0; }\n", ""
    ))
    finding = finding_for(preflight.run_checks(case, ["patches"]), "patches")
    assert finding.status == "fail"
    assert "0/p has no entry for outlet" in finding.measured
    assert "0/p" in finding.repair


def test_the_patches_check_is_skipped_before_the_mesh_exists(preflight, case):
    (case / "constant" / "polyMesh" / "boundary").unlink()
    assert finding_for(preflight.run_checks(case, ["patches"]), "patches").status == "skipped"


# -- 2D and the empty patch ---------------------------------------------------------


def test_a_correct_2d_case_passes_the_empty_check(preflight):
    mesh = preflight.parse_boundary(BOUNDARY)
    fields = {
        "U": preflight.parse_boundary_field(FIELD_U),
        "p": preflight.parse_boundary_field(FIELD_P),
    }
    result = preflight.empty_diagnosis(mesh, fields, (100, 50, 1))
    assert result["two_dimensional"] is True
    assert result["thin_axis"] == 2
    assert result["wrong_types"] == [] and result["missing_entries"] == []


def test_one_field_calling_the_empty_patch_zerogradient_is_caught(preflight):
    """The whole 2D bug in one line: nine fields right and one wrong stops the run."""
    mesh = preflight.parse_boundary(BOUNDARY)
    fields = {
        "U": preflight.parse_boundary_field(FIELD_U),
        "p": preflight.parse_boundary_field(
            FIELD_P.replace("frontAndBack { type empty; }", "frontAndBack { type zeroGradient; }")
        ),
    }
    result = preflight.empty_diagnosis(mesh, fields, (100, 50, 1))
    assert result["wrong_types"] == [("p", "frontAndBack", "zeroGradient")]


def test_a_wildcard_that_swallows_the_empty_patch_is_caught(preflight):
    """`".*" { type zeroGradient; }` covers the patch, and covers it wrongly."""
    mesh = preflight.parse_boundary(BOUNDARY)
    fields = {"k": spec_for(
        preflight, patterns=[".*"], types={".*": "zeroGradient"}
    )}
    result = preflight.empty_diagnosis(mesh, fields, (100, 50, 1))
    assert result["wrong_types"] == [("k", "frontAndBack", "zeroGradient")]


def test_a_field_with_no_entry_for_the_empty_patch_is_caught(preflight):
    mesh = preflight.parse_boundary(BOUNDARY)
    fields = {"nut": spec_for(preflight, "inlet", "outlet", "cylinder")}
    result = preflight.empty_diagnosis(mesh, fields, (100, 50, 1))
    assert result["missing_entries"] == [("nut", "frontAndBack")]


def test_an_includeetc_excuses_the_missing_empty_entry(preflight):
    mesh = preflight.parse_boundary(BOUNDARY)
    fields = {"nut": spec_for(preflight, "inlet", "outlet", "cylinder", includes=True)}
    assert preflight.empty_diagnosis(mesh, fields, (100, 50, 1))["missing_entries"] == []


def test_an_empty_patch_on_a_mesh_two_cells_thick_disagrees_with_itself(preflight):
    mesh = preflight.parse_boundary(BOUNDARY)
    result = preflight.empty_diagnosis(mesh, {}, (100, 50, 2))
    assert result["thin_axis"] is None
    assert result["thickness_disagrees"] is True


def test_a_one_cell_direction_with_no_empty_patch_is_still_2d(preflight):
    """One cell in z and no empty patch is the other half of the same bug: the
    solver discretises across the single cell as though it were a third dimension."""
    mesh = [{"name": "walls", "type": "wall", "nFaces": 10, "inGroups": ""}]
    result = preflight.empty_diagnosis(mesh, {}, (100, 50, 1))
    assert result["two_dimensional"] is True
    assert result["empty_patches"] == []


def test_a_3d_case_skips_the_empty_check_entirely(preflight):
    mesh = [{"name": "walls", "type": "wall", "nFaces": 10, "inGroups": ""}]
    assert preflight.empty_diagnosis(mesh, {}, (100, 50, 40))["two_dimensional"] is False


def test_the_empty_check_reports_skipped_on_a_3d_case(preflight, case):
    (case / "system" / "blockMeshDict").write_text(
        BLOCK_MESH_DICT.replace("(100 50 1)", "(100 50 40)")
    )
    (case / "constant" / "polyMesh" / "boundary").write_text(
        BOUNDARY.replace("type empty;", "type wall;")
    )
    assert finding_for(preflight.run_checks(case, ["empty"]), "empty").status == "skipped"


def test_the_empty_check_fails_the_whole_case_for_one_wrong_field(preflight, case):
    (case / "0" / "p").write_text(
        FIELD_P.replace("frontAndBack { type empty; }", "frontAndBack { type zeroGradient; }")
    )
    finding = finding_for(preflight.run_checks(case, ["empty"]), "empty")
    assert finding.status == "fail"
    assert "0/p gives frontAndBack type zeroGradient" in finding.measured
    assert "type empty;" in finding.repair


# -- Reynolds number ----------------------------------------------------------------


def test_the_reynolds_arithmetic_is_u_l_over_nu(preflight):
    result = preflight.reynolds_diagnosis(velocity=2.0, length=0.5, viscosity=1e-4)
    assert result["implied"] == pytest.approx(10_000.0)
    assert result["status"] == "ok"


def test_a_matching_stated_reynolds_number_passes(preflight):
    result = preflight.reynolds_diagnosis(1.0, 0.1, 1e-5, stated=10_000)
    assert result["status"] == "ok"
    assert result["ratio"] == pytest.approx(1.0)


def test_a_small_disagreement_is_a_warning_and_a_large_one_is_a_failure(preflight):
    """Ten per cent out is somebody's rounded length scale; four times out is a
    different flow."""
    assert preflight.reynolds_diagnosis(1.1, 0.1, 1e-5, stated=10_000)["status"] == "warn"
    assert preflight.reynolds_diagnosis(4.0, 0.1, 1e-5, stated=10_000)["status"] == "fail"


def test_no_length_scale_means_no_reynolds_number_is_invented(preflight):
    """A guessed L is a guessed Re, and a made-up number that disagrees with the case
    is worse than no number."""
    result = preflight.reynolds_diagnosis(1.0, None, 1e-5, stated=3900)
    assert result["implied"] is None
    assert result["status"] == "skipped"


def test_a_zero_viscosity_fails_the_reynolds_check(preflight):
    assert preflight.reynolds_diagnosis(1.0, 0.1, 0.0)["status"] == "fail"


def test_the_repair_names_the_viscosity_that_would_be_right(preflight):
    assert preflight.suggested_viscosity(1.0, 0.1, 3900) == pytest.approx(1.0 / 39000)


def test_the_reynolds_check_reads_u_from_the_case_when_none_is_given(preflight, case):
    finding = finding_for(
        preflight.run_checks(case, ["reynolds"], preflight.Intent(length=0.1)), "reynolds"
    )
    assert finding.status == "ok"
    assert "0/U" in finding.measured
    assert "Re = 100" in finding.measured


def test_the_reynolds_check_fails_against_a_stated_intent(preflight, case):
    intent = preflight.Intent(reynolds=3900, length=0.1)
    finding = finding_for(preflight.run_checks(case, ["reynolds"], intent), "reynolds")
    assert finding.status == "fail"
    assert "3900" in finding.measured
    assert "constant/transportProperties" in finding.repair


def test_the_reynolds_check_is_skipped_without_a_viscosity(preflight, case):
    (case / "constant" / "transportProperties").unlink()
    assert finding_for(preflight.run_checks(case, ["reynolds"]), "reynolds").status == "skipped"


# -- predicted against actual cell count --------------------------------------------


def test_a_mesh_the_size_the_dictionaries_predict_passes(preflight):
    assert preflight.cell_count_diagnosis(1_000_000, 1_500_000)["status"] == "ok"


def test_a_mesh_far_smaller_than_predicted_fails(preflight):
    """What a snappy run that never found the surface leaves behind."""
    result = preflight.cell_count_diagnosis(9_000_000, 2_000)
    assert result["status"] == "fail"
    assert result["ratio"] < 1


def test_a_mesh_moderately_off_the_prediction_only_warns(preflight):
    """The estimate is documented as order-of-magnitude; it is not evidence on its own."""
    assert preflight.cell_count_diagnosis(1_000_000, 5_000_000)["status"] == "warn"


def test_an_unbuilt_mesh_is_judged_on_the_prediction_alone(preflight):
    assert preflight.cell_count_diagnosis(1_000, None)["status"] == "ok"
    assert preflight.cell_count_diagnosis(100_000_000, None)["status"] == "warn"


def test_the_cells_check_compares_the_case_against_its_own_mesh(preflight, case):
    finding = finding_for(preflight.run_checks(case, ["cells"]), "cells")
    assert finding.status == "ok"
    assert "5,000 cells" in finding.measured


# -- checkMesh quality ---------------------------------------------------------------


def test_a_healthy_checkmesh_log_passes(preflight):
    mesh_digest = load("mesh_digest")
    assert preflight.checkmesh_verdict(mesh_digest.parse(CHECK_MESH_LOG))["status"] == "ok"


def test_severe_non_orthogonality_warns_and_names_the_corrector(preflight):
    mesh_digest = load("mesh_digest")
    log = CHECK_MESH_LOG.replace("Max: 40.1", "Max: 74.0")
    verdict = preflight.checkmesh_verdict(mesh_digest.parse(log))
    assert verdict["status"] == "warn"
    assert "nonOrthogonalCorrectors" in " ".join(verdict["repairs"])


def test_extreme_non_orthogonality_fails(preflight):
    mesh_digest = load("mesh_digest")
    log = CHECK_MESH_LOG.replace("Max: 40.1", "Max: 89.0")
    assert preflight.checkmesh_verdict(mesh_digest.parse(log))["status"] == "fail"


def test_a_line_checkmesh_starred_is_a_failure_whatever_the_numbers_say(preflight):
    mesh_digest = load("mesh_digest")
    log = CHECK_MESH_LOG.replace(
        "Max skewness = 1.92 OK.", "Max skewness = 22.4 ***Max skewness too high"
    )
    verdict = preflight.checkmesh_verdict(mesh_digest.parse(log))
    assert verdict["status"] == "fail"


def test_the_checkmesh_check_is_skipped_with_no_log(preflight, case):
    (case / "log.checkMesh").unlink()
    finding = finding_for(preflight.run_checks(case, ["checkmesh"]), "checkmesh")
    assert finding.status == "skipped"
    assert "checkMesh" in finding.repair


# -- the one-iteration solver probe --------------------------------------------------


def test_rewrite_control_dict_replaces_only_top_level_entries(preflight):
    """The functions block has its own writeInterval; rewriting that one would change
    a function object and leave the run bounds alone."""
    rewritten = preflight.rewrite_control_dict(CONTROL_DICT, {"writeInterval": "1"})
    control = preflight.parse_control(rewritten)
    assert control["writeInterval"] == "1"
    assert "writeInterval   5;" in rewritten or "writeInterval    5;" in rewritten


def test_rewrite_control_dict_appends_an_entry_that_was_not_there(preflight):
    rewritten = preflight.rewrite_control_dict(CONTROL_DICT, {"purgeWrite2": "3"})
    assert preflight.parse_control(rewritten)["purgeWrite2"] == "3"


def test_the_probe_control_dict_stops_one_step_past_the_start(preflight):
    patched, bounds = preflight.probe_control_dict(CONTROL_DICT)
    control = preflight.parse_control(patched)
    assert bounds == {"startTime": 0.0, "deltaT": 0.001, "endTime": 0.001}
    assert float(control["endTime"]) == pytest.approx(0.001)
    assert control["stopAt"] == "endTime"
    assert control["startFrom"] == "startTime"
    assert control["adjustTimeStep"] == "no"


def test_staging_never_touches_the_real_control_dict(tmp_path, preflight, case):
    """The whole reason the probe is allowed to exist: it runs against a copy."""
    before = (case / "system" / "controlDict").read_text()
    staged = preflight.stage_probe_case(case, tmp_path / "staged")
    patched, _bounds = preflight.probe_control_dict(
        (staged / "system" / "controlDict").read_text()
    )
    (staged / "system" / "controlDict").write_text(patched)

    assert (case / "system" / "controlDict").read_text() == before
    assert float(preflight.parse_control(
        (staged / "system" / "controlDict").read_text()
    )["endTime"]) == pytest.approx(0.001)


def test_staging_carries_the_mesh_and_the_fields_but_not_the_logs(tmp_path, preflight, case):
    staged = preflight.stage_probe_case(case, tmp_path / "staged")
    assert (staged / "constant" / "polyMesh" / "boundary").exists()
    assert (staged / "0" / "U").exists()
    assert not (staged / "log.pimpleFoam").exists()


def test_a_probe_that_takes_a_step_and_ends_is_a_pass(preflight):
    verdict = preflight.probe_verdict({
        "returncode": 0,
        "output": "Create mesh\nTime = 0.001\n\nExecutionTime = 1 s\nEnd\n",
    })
    assert verdict["status"] == "ok"
    assert verdict["steps"] == 1
    assert verdict["finished"] is True


def test_a_fatal_error_is_extracted_without_the_stack_trace(preflight):
    output = """\
Create mesh for time = 0

--> FOAM FATAL IO ERROR:
Cannot find patchField entry for outlet

file: /work/case/0/p.boundaryField from line 25 to line 40.

    From function void Foam::GeometricField::Boundary::readField(...)
    #0  Foam::error::printStack(Foam::Ostream&) at ??:?
    #1  Foam::IOerror::abort() at ??:?
FOAM exiting
"""
    text = preflight.fatal_error_text(output)
    assert "Cannot find patchField entry for outlet" in text
    assert "#0" not in text


def test_a_known_fatal_error_gets_a_named_repair(preflight):
    verdict = preflight.probe_verdict({
        "returncode": 1,
        "output": "--> FOAM FATAL IO ERROR:\nCannot find patchField entry for outlet\nFOAM exiting",
    })
    assert verdict["status"] == "fail"
    assert "boundaryField entry" in verdict["hint"]


def test_an_unrecognised_fatal_error_is_reported_without_interpretation(preflight):
    """Making something up about an error nobody has seen before is worse than
    handing over what it said."""
    verdict = preflight.probe_verdict({
        "returncode": 1,
        "output": "--> FOAM FATAL ERROR:\nsomething nobody has written a hint for\nFOAM exiting",
    })
    assert verdict["status"] == "fail"
    assert verdict["hint"] == ""
    assert "nobody has written a hint for" in verdict["fatal"]


def test_a_solver_that_exits_cleanly_without_a_time_step_only_warns(preflight):
    verdict = preflight.probe_verdict({"returncode": 0, "output": "Create mesh\n"})
    assert verdict["status"] == "warn"


def test_the_probe_check_runs_the_named_application_in_the_copy(tmp_path, preflight, case):
    seen = {}

    def runner(command, cwd):
        seen["command"] = command
        seen["cwd"] = Path(cwd)
        return {"returncode": 0, "output": "Time = 0.001\nEnd\n"}

    findings = preflight.check_probe(
        preflight.Case(case), preflight.Intent(), runner=runner, workdir=tmp_path / "probe"
    )
    assert seen["command"] == ["pimpleFoam"]
    assert seen["cwd"] == tmp_path / "probe"
    assert findings[0].status == "ok"
    assert "one deltaT" in findings[0].measured


def test_the_probe_check_reports_what_the_solver_said(tmp_path, preflight, case):
    def runner(command, cwd):
        return {
            "returncode": 1,
            "output": "--> FOAM FATAL IO ERROR:\nCannot find patchField entry for outlet\nFOAM exiting",
        }

    finding = preflight.check_probe(
        preflight.Case(case), preflight.Intent(), runner=runner, workdir=tmp_path / "probe"
    )[0]
    assert finding.status == "fail"
    assert "Cannot find patchField entry for outlet" in finding.measured


def test_the_probe_is_skipped_when_no_application_is_named(tmp_path, preflight, case):
    (case / "system" / "controlDict").write_text(
        CONTROL_DICT.replace("application     pimpleFoam;", "")
    )
    finding = preflight.check_probe(
        preflight.Case(case), preflight.Intent(),
        runner=lambda command, cwd: {"returncode": 0, "output": ""},
        workdir=tmp_path / "probe",
    )[0]
    assert finding.status == "skipped"


# -- Courant number ------------------------------------------------------------------


def test_the_courant_arithmetic(preflight):
    assert preflight.courant_estimate(2.0, 0.01, 0.001) == pytest.approx(0.2)
    assert preflight.courant_estimate(None, 0.01, 0.001) is None


def test_a_courant_number_over_one_warns_and_far_over_fails(preflight):
    assert preflight.courant_verdict(0.4, False, None)["status"] == "ok"
    assert preflight.courant_verdict(2.0, False, None)["status"] == "warn"
    assert preflight.courant_verdict(40.0, False, None)["status"] == "fail"


def test_an_adjusting_timestep_is_judged_on_maxco_instead(preflight):
    assert preflight.courant_verdict(99.0, True, 0.9)["status"] == "ok"
    assert preflight.courant_verdict(0.1, True, None)["status"] == "warn"


def test_a_steady_solver_has_no_courant_number_to_check(preflight):
    assert preflight.is_steady("simpleFoam") is True
    assert preflight.is_steady("buoyantBoussinesqSimpleFoam") is True
    assert preflight.is_steady("pimpleFoam") is False


def test_the_courant_check_skips_a_steady_case(preflight, case):
    (case / "system" / "controlDict").write_text(
        CONTROL_DICT.replace("pimpleFoam", "simpleFoam")
    )
    finding = finding_for(preflight.run_checks(case, ["courant"]), "courant")
    assert finding.status == "skipped"
    assert "iterations" in finding.meaning


def test_the_courant_check_prefers_the_mesh_that_exists(preflight, case):
    """The blockMeshDict base size describes the mesh before snappy refined it, and
    using it on a refined case reports a Courant number several factors too low."""
    size, source = preflight.smallest_cell_size(preflight.Case(case))
    assert size == pytest.approx(8e-09 ** (1 / 3))
    assert "checkMesh" in source


def test_the_courant_check_falls_back_to_the_blockmesh_base_cell(preflight, case):
    (case / "log.checkMesh").unlink()
    size, source = preflight.smallest_cell_size(preflight.Case(case))
    assert size == pytest.approx(0.0466667, rel=1e-3)
    assert "blockMeshDict" in source


def test_the_courant_check_fails_a_timestep_that_is_far_too_large(preflight, case):
    (case / "system" / "controlDict").write_text(
        CONTROL_DICT.replace("deltaT          0.001;", "deltaT          0.1;")
    )
    finding = finding_for(preflight.run_checks(case, ["courant"]), "courant")
    assert finding.status == "fail"
    assert "deltaT" in finding.repair


# -- residuals -----------------------------------------------------------------------


def falling(preflight):
    return {
        "residuals": {"Ux": [(0, 1e-1), (1, 1e-2), (2, 1e-3), (3, 1e-4)]},
        "final_residual": {"Ux": 1e-5},
        "continuity": (1e-8, 1e-10, 1e-9),
        "bounding": {},
        "times": [1.0, 2.0, 3.0, 4.0],
    }


def test_a_falling_residual_series_passes(preflight):
    assert preflight.residual_verdict(falling(preflight))["status"] == "ok"


def test_a_blown_up_residual_fails_and_names_relaxation(preflight):
    data = falling(preflight)
    data["residuals"]["p"] = [(0, 1e-1), (1, 1e3), (2, 1e6)]
    verdict = preflight.residual_verdict(data)
    assert verdict["status"] == "fail"
    assert verdict["diverging"][0][0] == "p"
    assert "relaxation" in " ".join(verdict["repairs"])


def test_a_residual_climbing_off_its_own_floor_warns(preflight):
    data = falling(preflight)
    data["residuals"]["Ux"] = [(0, 1e-1), (1, 1e-4), (2, 1e-3), (3, 1e-2)]
    verdict = preflight.residual_verdict(data)
    assert verdict["status"] == "warn"
    assert verdict["rising"]


def test_a_growing_continuity_error_is_its_own_failure(preflight):
    """Different problem, different repair: mass conservation, not stability."""
    data = falling(preflight)
    data["continuity"] = (1e-4, 1e-4, 12.0)
    verdict = preflight.residual_verdict(data)
    assert verdict["status"] == "fail"
    assert "conserved" in " ".join(verdict["repairs"])


def test_bounding_messages_are_reported_as_a_warning(preflight):
    data = falling(preflight)
    data["bounding"] = {"k": 40}
    verdict = preflight.residual_verdict(data)
    assert verdict["status"] == "warn"
    assert "k x40" in " ".join(verdict["problems"])


def test_a_log_with_no_residuals_yet_is_skipped(preflight):
    assert preflight.residual_verdict({"residuals": {}})["status"] == "skipped"


def test_the_residual_check_reads_the_case_log(preflight, case):
    finding = finding_for(preflight.run_checks(case, ["residuals"]), "residuals")
    assert finding.status == "ok"
    assert "log.pimpleFoam" in finding.measured


def test_the_residual_check_ignores_the_meshing_logs(preflight, case):
    """`log.blockMesh` has no residuals in it and picking it would report nothing."""
    (case / "log.pimpleFoam").unlink()
    (case / "log.blockMesh").write_text("Create mesh\nEnd\n")
    assert preflight.Case(case).solver_log() is None


def test_a_named_log_overrides_the_search(preflight, case, tmp_path):
    other = tmp_path / "elsewhere.log"
    other.write_text(SOLVER_LOG)
    assert preflight.Case(case).solver_log(other) == other


# -- force and pressure units ---------------------------------------------------------


def test_force_objects_are_found_inside_the_functions_block(preflight):
    objects = preflight.find_force_objects(CONTROL_DICT)
    assert [entry["name"] for entry in objects] == ["forces1"]
    assert objects[0]["type"] == "forceCoeffs"
    assert objects[0]["rhoInf"] == pytest.approx(1.225)


def test_a_correctly_set_up_force_object_passes(preflight):
    objects = preflight.find_force_objects(CONTROL_DICT)
    result = preflight.force_units_diagnosis(preflight.KINEMATIC_PRESSURE, objects)
    assert result["status"] == "ok"


def test_a_force_object_with_no_density_fails_on_a_kinematic_pressure(preflight):
    """Incompressible p is m2/s2. A force computed from it without rho is a force
    divided by density -- out by 1.2 in air, by a thousand in water, and plausible."""
    objects = preflight.find_force_objects(
        CONTROL_DICT.replace("rho             rhoInf;", "").replace("rhoInf          1.225;", "")
    )
    result = preflight.force_units_diagnosis(preflight.KINEMATIC_PRESSURE, objects)
    assert result["status"] == "fail"
    assert "1.225" in " ".join(result["repairs"])


def test_the_placeholder_density_of_one_is_a_warning(preflight):
    objects = preflight.find_force_objects(CONTROL_DICT.replace("1.225;", "1;"))
    result = preflight.force_units_diagnosis(preflight.KINEMATIC_PRESSURE, objects)
    assert result["status"] == "warn"
    assert "per unit density" in " ".join(result["problems"])


def test_a_forcecoeffs_missing_its_reference_values_fails(preflight):
    objects = preflight.find_force_objects(CONTROL_DICT.replace("Aref            0.01;", ""))
    result = preflight.force_units_diagnosis(preflight.KINEMATIC_PRESSURE, objects)
    assert result["status"] == "fail"
    assert "Aref" in " ".join(result["problems"])


def test_a_compressible_case_is_judged_the_other_way_round(preflight):
    objects = preflight.find_force_objects(CONTROL_DICT)
    result = preflight.force_units_diagnosis(preflight.STATIC_PRESSURE, objects)
    assert result["status"] == "warn"
    assert "rho rho" in " ".join(result["repairs"])


def test_no_force_object_means_nothing_to_check(preflight):
    assert preflight.force_units_diagnosis(preflight.KINEMATIC_PRESSURE, [])["status"] == "skipped"


def test_a_force_object_in_its_own_system_file_is_found_too(preflight, case):
    (case / "system" / "forceCoeffs").write_text("""
    forceCoeffs1
    {
        type            forceCoeffs;
        magUInf         1;
        lRef            0.1;
        Aref            0.01;
    }
    """)
    finding = finding_for(preflight.run_checks(case, ["units"]), "units")
    assert finding.status == "fail"
    assert "forceCoeffs1" in finding.measured


def test_the_units_check_passes_on_the_healthy_case(preflight, case):
    finding = finding_for(preflight.run_checks(case, ["units"]), "units")
    assert finding.status == "ok"
    assert "kinematic" in finding.measured


# -- disk -----------------------------------------------------------------------------


def test_the_write_count_for_a_runtime_cadence(preflight):
    assert preflight.expected_write_count(preflight.parse_control(CONTROL_DICT)) == 10


def test_the_write_count_for_a_timestep_cadence(preflight):
    control = preflight.parse_control(
        CONTROL_DICT.replace("writeControl    runTime;", "writeControl    timeStep;")
        .replace("writeInterval   0.1;", "writeInterval   100;")
    )
    assert preflight.expected_write_count(control) == 10


def test_purgewrite_caps_the_write_count(preflight):
    control = preflight.parse_control(CONTROL_DICT.replace("purgeWrite      0;", "purgeWrite      3;"))
    assert preflight.expected_write_count(control) == 3


def test_an_unknowable_write_count_is_none_rather_than_a_guess(preflight):
    assert preflight.expected_write_count({"endTime": "1"}) is None


def test_the_bytes_per_write_estimate_scales_with_cells_and_components(preflight):
    small = preflight.estimate_write_bytes(1000, 4, "ascii")
    large = preflight.estimate_write_bytes(2000, 4, "ascii")
    binary = preflight.estimate_write_bytes(1000, 4, "binary")
    assert large == pytest.approx(2 * small)
    assert binary < small


def test_an_existing_time_directory_beats_the_estimate(preflight, case):
    written = case / "0.5"
    written.mkdir()
    (written / "U").write_text("x" * 4096)
    size, source = preflight.measured_write_bytes(case)
    assert size == 4096
    assert "0.5" in source


def test_the_zero_directory_is_not_mistaken_for_a_written_time(preflight, case):
    assert preflight.measured_write_bytes(case) == (None, "")


def test_disk_verdict_thresholds(preflight):
    assert preflight.disk_verdict(10, 1000)["status"] == "ok"
    assert preflight.disk_verdict(900, 1000)["status"] == "warn"
    assert preflight.disk_verdict(2000, 1000)["status"] == "fail"


def test_the_disk_check_fails_when_the_run_will_not_fit(preflight, case):
    finding = finding_for(
        preflight.run_checks(case, ["disk"], free_bytes=1024), "disk"
    )
    assert finding.status == "fail"
    assert "purgeWrite" in finding.repair


def test_the_disk_check_passes_with_room_to_spare(preflight, case):
    finding = finding_for(
        preflight.run_checks(case, ["disk"], free_bytes=10 ** 12), "disk"
    )
    assert finding.status == "ok"


def test_the_disk_check_prefers_a_time_directory_that_already_exists(preflight, case):
    written = case / "0.1"
    written.mkdir()
    (written / "U").write_text("x" * 100_000)
    finding = finding_for(preflight.run_checks(case, ["disk"], free_bytes=10 ** 12), "disk")
    assert "existing time directory 0.1" in finding.measured
    assert "10 write time" in finding.measured


# -- findings, selection and the command line ------------------------------------------


def test_escalate_keeps_the_worst_status(preflight):
    assert preflight.escalate("ok", "warn") == "warn"
    assert preflight.escalate("fail", "warn") == "fail"
    assert preflight.escalate("skipped", "ok") == "ok"


def test_count_phrase_does_not_say_one_edges(preflight):
    assert preflight.count_phrase(1, "open edge") == "1 open edge"
    assert preflight.count_phrase(3, "open edge") == "3 open edges"


def test_select_checks_keeps_the_fixed_order(preflight):
    assert preflight.select_checks("units,patches") == ["patches", "units"]


def test_select_checks_refuses_a_name_it_does_not_know(preflight):
    with pytest.raises(ValueError) as error:
        preflight.select_checks("patches,nonsense")
    assert "nonsense" in str(error.value)


def test_no_probe_drops_only_the_probe(preflight):
    names = preflight.select_checks("", skip_probe=True)
    assert "probe" not in names
    assert "patches" in names


def test_a_check_that_raises_costs_only_itself(preflight, case, monkeypatch):
    """Nine useful answers and one crash is a better preflight than no preflight."""
    def explode(_case, _intent):
        raise RuntimeError("boom")

    monkeypatch.setitem(preflight.CHECKS, "cells", explode)
    findings = preflight.run_checks(case, ["cells", "patches"])
    crashed = finding_for(findings, "cells")
    assert crashed.status == "skipped"
    assert "RuntimeError" in crashed.measured
    assert finding_for(findings, "patches").status == "ok"


def test_findings_come_back_worst_first(preflight, case):
    (case / "0" / "p").write_text(FIELD_P.replace(
        "    outlet       { type fixedValue; value uniform 0; }\n", ""
    ))
    findings = preflight.run_checks(case, ["patches", "checkmesh"])
    assert findings[0].status == "fail"


def test_a_healthy_case_exits_zero(preflight, case, capsys):
    code = preflight.main([str(case), "--no-probe"])
    assert code == 0
    assert "fail" not in capsys.readouterr().out.split("\n")[1]


def test_a_broken_case_exits_one_so_a_script_can_gate_on_it(preflight, case, capsys):
    (case / "0" / "p").write_text(FIELD_P.replace(
        "    outlet       { type fixedValue; value uniform 0; }\n", ""
    ))
    assert preflight.main([str(case), "--checks", "patches"]) == 1


def test_the_json_output_carries_every_field_of_every_finding(preflight, case, capsys):
    preflight.main([str(case), "--checks", "patches,units", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["worst"] == "ok"
    assert {row["check"] for row in payload["findings"]} == {"patches", "units"}
    for row in payload["findings"]:
        assert set(row) == {"check", "status", "measured", "means", "repair"}


def test_the_human_report_keeps_measurement_and_meaning_apart(preflight, case, capsys):
    preflight.main([str(case), "--checks", "checkmesh"])
    out = capsys.readouterr().out
    assert "measured" in out and "means" in out


def test_the_report_says_the_repairs_are_suggestions(preflight, case, capsys):
    preflight.main([str(case), "--checks", "patches"])
    assert "suggestions" in capsys.readouterr().out


def test_an_unknown_check_name_exits_two_and_lists_the_known_ones(preflight, case, capsys):
    assert preflight.main([str(case), "--checks", "nonsense"]) == 2
    assert "patches" in capsys.readouterr().err


def test_list_checks_names_every_check(preflight, capsys):
    preflight.main(["--list-checks"])
    printed = capsys.readouterr().out.split()
    assert set(printed) == set(preflight.CHECK_ORDER)


def test_the_written_report_is_registered_in_the_manifest(preflight, study_state, tmp_path, capsys):
    """A preflight nobody can find afterwards is a preflight that gets run twice."""
    study = tmp_path / "study"
    (study / ".reynolds").mkdir(parents=True)
    case = write_case(study / "cylinder")
    out = study / "preflight.txt"

    preflight.main([str(case), "--checks", "patches", "--out", str(out)])

    assert out.exists()
    rows = study_state.artifacts(root=study, kind="report")
    assert [row["label"] for row in rows] == ["preflight"]
    assert rows[0]["path"] == "preflight.txt"
    assert rows[0]["meta"]["worst"] == "ok"


# -- the empty check before the mesh exists ----------------------------------------


BLOCK_MESH_2D = """\
FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
scale   1;
vertices
(
    (0 0 0) (1 0 0) (1 1 0) (0 1 0)
    (0 0 0.1) (1 0 0.1) (1 1 0.1) (0 1 0.1)
);
blocks
(
    hex (0 1 2 3 4 5 6 7) (20 20 1) simpleGrading (1 1 1)
);
edges ();
boundary
(
    inlet
    {
        type patch;
        faces ( (0 4 7 3) );
    }
    frontAndBack
    {
        type empty;
        faces ( (0 3 2 1) (4 5 6 7) );
    }
);
mergePatchPairs ();
"""


def test_the_blockmesh_boundary_list_is_read_for_names_and_types(preflight):
    patches = preflight.parse_block_mesh_boundary(BLOCK_MESH_2D)

    assert [patch["name"] for patch in patches] == ["inlet", "frontAndBack"]
    assert [patch["type"] for patch in patches] == ["patch", "empty"]


def test_a_blockmeshdict_with_no_boundary_list_reads_as_no_patches(preflight):
    assert preflight.parse_block_mesh_boundary("scale 1;\nblocks ();\n") == []


def test_the_empty_check_falls_back_to_the_dictionary_before_the_mesh_is_built(preflight, tmp_path):
    """The moment this check is worth most is before blockMesh has run -- an `empty`
    patch missing from the dictionary costs a mesh build to find. Reading only
    constant/polyMesh/boundary made every unmeshed 2D case report the opposite of
    the truth: "no patch is declared empty" about a dictionary that declares one.
    """
    case = tmp_path / "cyl"
    (case / "system").mkdir(parents=True)
    (case / "0").mkdir()
    (case / "system" / "blockMeshDict").write_text(BLOCK_MESH_2D, encoding="utf-8")
    (case / "0" / "U").write_text(
        "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\n"
        "boundaryField\n{\n    inlet\n    {\n        type fixedValue;\n"
        "        value uniform (1 0 0);\n    }\n    frontAndBack\n    {\n"
        "        type empty;\n    }\n}\n",
        encoding="utf-8",
    )

    findings = preflight.check_empty(preflight.Case(case), preflight.Intent())

    assert len(findings) == 1
    assert findings[0].status == "ok", findings[0].measured
    assert "system/blockMeshDict" in findings[0].measured
    assert "frontAndBack" in findings[0].measured


def test_a_dictionary_that_is_thin_and_names_no_empty_patch_still_fails(preflight, tmp_path):
    """The fallback must not turn the check off: a one-cell direction with no empty
    patch anywhere is the failure it exists to catch."""
    case = tmp_path / "cyl"
    (case / "system").mkdir(parents=True)
    (case / "system" / "blockMeshDict").write_text(
        BLOCK_MESH_2D.replace("type empty;", "type patch;"), encoding="utf-8"
    )

    findings = preflight.check_empty(preflight.Case(case), preflight.Intent())

    assert findings[0].status == "fail"
    assert "no patch is declared empty" in findings[0].measured


# -- what each check does at a stage where its input does not exist yet -------------
#
# The `empty` check reading only the built mesh was not an incident, it was a
# pattern: this tool runs before the expensive steps, so every check meets its input
# missing. A `fail` or a `warn` there trains the agent to ignore the gate, and an
# `ok` about numbers that were never read is worse than either.


def test_a_checkmesh_log_with_no_metrics_is_skipped_not_passed(preflight, case):
    """checkMesh that died on a missing mesh left a log with nothing measured in it,
    and the verdict came back `ok` -- "every quality metric is inside the
    conventional limits" about zero metrics. A gate may not pass a mesh it never
    read."""
    (case / "log.checkMesh").write_text(
        "Create time\n\nCreate polyMesh for time = 0\n\n"
        "--> FOAM FATAL ERROR:\n"
        'cannot find file "constant/polyMesh/points"\n\n'
        "    From function regIOobject::readStream()\n"
        "FOAM exiting\n",
        encoding="utf-8",
    )

    finding = finding_for(preflight.run_checks(case, ["checkmesh"]), "checkmesh")

    assert finding.status == "skipped"
    assert "not a mesh that passed" in finding.meaning
    assert "constant/polyMesh/points" in finding.measured


def test_a_checkmesh_log_that_stops_before_the_metrics_is_skipped(preflight, case):
    (case / "log.checkMesh").write_text(
        "Create time\nCreate polyMesh for time = 0\n", encoding="utf-8"
    )
    finding = finding_for(preflight.run_checks(case, ["checkmesh"]), "checkmesh")
    assert finding.status == "skipped"


def test_a_real_checkmesh_log_still_gets_a_verdict(preflight, case):
    """The skip must not swallow the check it guards."""
    finding = finding_for(preflight.run_checks(case, ["checkmesh"]), "checkmesh")
    assert finding.status == "ok"
    assert "5,000 cells" in finding.measured


def test_force_objects_named_before_the_fields_exist_are_skipped_not_warned(preflight, case):
    """A controlDict names its forceCoeffs long before 0/ is written. Warning that
    p's dimensions are unreadable at that point is a complaint about a units error
    nobody can have made yet."""
    for name in ("U", "p"):
        (case / "0" / name).unlink()
    (case / "0").rmdir()

    finding = finding_for(preflight.run_checks(case, ["units"]), "units")

    assert finding.status == "skipped"
    assert "no field files written yet" in finding.measured


def test_a_zero_directory_whose_p_is_unreadable_still_warns(preflight, case):
    """The skip above is about a stage. A 0/ that exists and still has no readable p
    is a fault, and it keeps its warn."""
    (case / "0" / "p").write_text("internalField uniform 0;\n", encoding="utf-8")

    finding = finding_for(preflight.run_checks(case, ["units"]), "units")

    assert finding.status == "warn"
    assert "no readable dimensions" in finding.measured
    # The reading has to follow from the measurement: with p unreadable there is no
    # factor-of-rho claim to be made either way.
    assert "without p's dimensions" in finding.meaning


def test_the_empty_check_does_not_claim_fields_agree_when_there_are_none(preflight, tmp_path):
    """'Every field agrees' over no fields is a true sentence and a false
    reassurance, in the one check whose whole point is the single field that does
    not agree."""
    case = tmp_path / "cyl"
    (case / "system").mkdir(parents=True)
    (case / "system" / "blockMeshDict").write_text(BLOCK_MESH_2D, encoding="utf-8")

    finding = preflight.check_empty(preflight.Case(case), preflight.Intent())[0]

    assert finding.status == "ok", "a case with no 0/ yet is a stage, not a fault"
    assert "no readable field files" in finding.measured
    assert "has not been done" in finding.meaning
    assert "every field agrees" not in finding.meaning


def test_the_empty_check_names_the_fields_it_checked(preflight, case):
    finding = finding_for(preflight.run_checks(case, ["empty"]), "empty")
    assert finding.status == "ok"
    assert "U, p" in finding.meaning


# -- the probe stages a copy, and the copy has to contain the case ------------------


def test_staging_resolves_the_case_path_before_linking(tmp_path, preflight, monkeypatch):
    """A symlink stores its target verbatim, and a relative one resolves against the
    link's own directory. Staging `cases/cylinder` from the study directory -- the
    ordinary way this is called -- linked `constant` to
    `<tempdir>/cylinder/cases/cylinder/constant`, which is nothing. The solver then
    said it could not find the mesh and the probe reported `fail` on a healthy case:
    a false fail from the one check that is supposed to be authoritative.
    """
    write_case(tmp_path / "cases" / "cylinder")
    recorded = []

    def fake_symlink(source, target, target_is_directory=False):
        recorded.append(Path(source))
        if Path(source).is_dir():
            Path(target).mkdir()
        else:
            Path(target).touch()

    monkeypatch.setattr(preflight.os, "symlink", fake_symlink)
    monkeypatch.chdir(tmp_path)

    preflight.stage_probe_case(Path("cases/cylinder"), tmp_path / "staged")

    assert recorded, "nothing was staged"
    for source in recorded:
        assert source.is_absolute(), f"{source} would resolve against the link's directory"
        assert source.exists(), f"{source} is a dangling link"


def test_a_probe_from_a_relative_case_path_stages_a_readable_mesh(tmp_path, preflight, monkeypatch):
    """The end of the same bug, through the check rather than through the helper."""
    write_case(tmp_path / "cases" / "cylinder")
    monkeypatch.chdir(tmp_path)
    seen = {}

    def runner(command, cwd):
        seen["boundary"] = (Path(cwd) / "constant" / "polyMesh" / "boundary").is_file()
        seen["U"] = (Path(cwd) / "0" / "U").is_file()
        return {"returncode": 0, "output": "Time = 0.001\nEnd\n"}

    findings = preflight.check_probe(
        preflight.Case(Path("cases/cylinder")), preflight.Intent(), runner=runner
    )

    assert seen == {"boundary": True, "U": True}
    assert findings[0].status == "ok"


# -- a converged residual is not a rising one ---------------------------------------


def test_a_residual_wobbling_on_its_floor_is_not_called_rising(preflight):
    """A run that bottoms out at 1e-12 and settles at 3e-11 is thirty times its own
    best and is finished. Warning about it puts a complaint on the healthiest log
    there is."""
    data = falling(preflight)
    data["residuals"]["Ux"] = [(0, 1e-3), (1, 1e-8), (2, 1e-12), (3, 1e-11), (4, 3e-11)]

    verdict = preflight.residual_verdict(data)

    assert verdict["rising"] == []
    assert verdict["status"] == "ok"


def test_a_residual_rising_above_the_floor_is_still_caught(preflight):
    """The floor must not turn the check off for a residual that matters."""
    data = falling(preflight)
    data["residuals"]["Ux"] = [(0, 1e-1), (1, 1e-5), (2, 1e-4), (3, 1e-3)]

    verdict = preflight.residual_verdict(data)

    assert verdict["status"] == "warn"
    assert verdict["rising"][0][0] == "Ux"


# -- a function object that is a whole file -----------------------------------------


FORCE_COEFFS_FILE = """\
FoamFile { version 2.0; format ascii; class dictionary; object forceCoeffs; }

type            forceCoeffs;
libs            ("libforces.so");
patches         (cylinder);
magUInf         1;
lRef            0.1;
Aref            0.01;
"""


def test_a_function_object_written_as_its_own_file_is_found(preflight):
    """`#includeFunc forceCoeffs` reads system/forceCoeffs, whose `type` is a
    top-level entry rather than a key inside a named block. Only named blocks were
    looked at, so this layout reported "no forces or forceCoeffs function object"
    and its missing density was never examined."""
    objects = preflight.find_force_objects(FORCE_COEFFS_FILE, name_if_bare="forceCoeffs")

    assert [entry["type"] for entry in objects] == ["forceCoeffs"]
    assert objects[0]["magUInf"] == 1.0


def test_the_units_check_reads_a_function_object_file_under_system(preflight, case):
    """The error this check exists for, in the layout it used to miss entirely:
    kinematic p and a force object that never sets rhoInf."""
    control = (case / "system" / "controlDict").read_text()
    control = (
        control[:control.index("functions")]
        + "functions\n{\n    #includeFunc forceCoeffs\n}\n"
    )
    (case / "system" / "controlDict").write_text(control, encoding="utf-8")
    (case / "system" / "forceCoeffs").write_text(FORCE_COEFFS_FILE, encoding="utf-8")

    finding = finding_for(preflight.run_checks(case, ["units"]), "units")

    assert finding.status == "fail"
    assert "rho rhoInf" in finding.measured
    assert "rhoInf" in finding.repair


def test_the_system_sweep_does_not_invent_force_objects(preflight, case):
    """Every dictionary under system/ is read looking for one. None of the ordinary
    ones may come back as a forces function object."""
    (case / "system" / "decomposeParDict").write_text(
        "FoamFile { object decomposeParDict; }\nnumberOfSubdomains 4;\nmethod scotch;\n",
        encoding="utf-8",
    )
    (case / "system" / "fvSchemes").write_text(
        "FoamFile { object fvSchemes; }\nddtSchemes { default Euler; }\n"
        "divSchemes { div(phi,U) Gauss linearUpwind grad(U); }\n",
        encoding="utf-8",
    )
    (case / "system" / "topoSetDict").write_text(
        "FoamFile { object topoSetDict; }\nactions\n(\n"
        "    { name c0; type cellSet; action new; source boxToCell; }\n);\n",
        encoding="utf-8",
    )

    finding = finding_for(preflight.run_checks(case, ["units"]), "units")

    assert finding.status == "ok", finding.measured
    # exactly one force object: the name and the type of forces1, nothing invented
    assert finding.measured.endswith("forces1 (forceCoeffs)")


def test_a_relative_out_path_is_registered_where_the_report_actually_is(
    preflight, study_state, tmp_path, monkeypatch
):
    """`--out` is written relative to the working directory and recorded relative to
    the study root. Handed over as written, the relative path was joined to the root
    instead, and the manifest pointed at a file that was not there -- an artifact
    registered and unreachable is worse than one not registered. The existing
    coverage passed an absolute --out and never met this."""
    work = tmp_path / "work"
    study = work / "studies" / "wake"
    (study / ".reynolds").mkdir(parents=True)
    write_case(study / "cylinder")
    monkeypatch.chdir(work)

    assert preflight.main(
        ["studies/wake/cylinder", "--checks", "patches", "--out", "pre.txt"]
    ) == 0

    assert (work / "pre.txt").is_file(), "the report goes where the shell said"
    rows = study_state.artifacts(root=study, kind="report")
    assert len(rows) == 1
    registered = study / rows[0]["path"]
    assert registered.is_file(), (
        "the manifest points at %s, which does not exist" % rows[0]["path"]
    )
    assert registered.read_text(encoding="utf-8") == (work / "pre.txt").read_text(encoding="utf-8")


# -- the cell count on a case snappyHexMesh never touches ----------------------------


BLOCK_MESH_GRADED = """\
FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
scale 1;
vertices ( (0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 0.1) (1 0 0.1) (1 1 0.1) (0 1 0.1) );
blocks
(
    hex (0 1 2 3 4 5 6 7) (20 30 1) simpleGrading (5 1 1)
    hex (0 1 2 3 4 5 6 7) (10 30 1) simpleGrading (1 1 1)
);
edges ();
boundary ( frontAndBack { type empty; faces ( (0 3 2 1) ); } );
"""


def test_the_blockmesh_count_is_arithmetic_not_an_estimate(preflight):
    """blockMesh builds the product of the divisions on each block and nothing else."""
    assert preflight.block_mesh_cells(BLOCK_MESH_GRADED) == 20 * 30 * 1 + 10 * 30 * 1


def test_a_blockmesh_only_case_is_not_judged_by_the_snappy_estimator(preflight, tmp_path):
    """Found on a live instance: a 12-block graded O-grid with 4,238 real cells was
    read as 105 background cells (ratio 40) and reported `fail`, with advice to check
    `refinementRegions` -- in a case that has no snappyHexMeshDict at all. A gate that
    fails a healthy case teaches the agent to stop reading it."""
    case = tmp_path / "cyl"
    (case / "system").mkdir(parents=True)
    (case / "system" / "blockMeshDict").write_text(BLOCK_MESH_GRADED, encoding="utf-8")

    finding = preflight.check_cells(preflight.Case(case), preflight.Intent())[0]

    assert finding.status == "ok", finding.measured
    assert "900" in finding.measured  # 600 + 300, the exact product
    assert "refinementRegions" not in (finding.repair or "")


def test_a_mesh_that_does_not_match_its_dictionary_is_a_warning(preflight, tmp_path, monkeypatch):
    case = tmp_path / "cyl"
    (case / "system").mkdir(parents=True)
    (case / "system" / "blockMeshDict").write_text(BLOCK_MESH_GRADED, encoding="utf-8")

    subject = preflight.Case(case)
    monkeypatch.setattr(type(subject), "cell_count", property(lambda self: 4238))
    finding = preflight.check_cells(subject, preflight.Intent())[0]

    assert finding.status == "warn"
    assert "900" in finding.measured and "4,238" in finding.measured
