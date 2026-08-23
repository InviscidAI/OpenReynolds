"""The toolbox scripts.

They run inside the container, but their parsing is ordinary Python and is exactly the
kind of thing that rots quietly, so it is tested here.
"""

from __future__ import annotations

import ast
import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load(name: str):
    """Import a toolbox script by path; the directory is data, not a package."""
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- log_digest ----------------------------------------------------------------

SOLVER_LOG = """\
Time = 1

smoothSolver:  Solving for Ux, Initial residual = 1.0e-01, Final residual = 2.0e-03, No Iterations 3
smoothSolver:  Solving for Uy, Initial residual = 2.0e-01, Final residual = 4.0e-03, No Iterations 2
GAMG:  Solving for p, Initial residual = 9.0e-01, Final residual = 1.0e-02, No Iterations 9
time step continuity errors : sum local = 1.5e-06, global = -2.0e-09, cumulative = -3.0e-09
Courant Number mean: 0.12 max: 0.87
bounding k, min: -0.01 max: 5 average: 1
ExecutionTime = 1.40 s  ClockTime = 1 s

Time = 2

smoothSolver:  Solving for Ux, Initial residual = 5.0e-03, Final residual = 1.0e-04, No Iterations 3
GAMG:  Solving for p, Initial residual = 4.0e-02, Final residual = 5.0e-04, No Iterations 7
time step continuity errors : sum local = 7.5e-07, global = -1.0e-09, cumulative = -4.0e-09
bounding k, min: -0.02 max: 5 average: 1
ExecutionTime = 2.80 s  ClockTime = 2 s
"""


@pytest.fixture
def log_digest():
    return load("log_digest")


def test_log_digest_reads_the_residual_series(tmp_path, log_digest):
    log = tmp_path / "log.simpleFoam"
    log.write_text(SOLVER_LOG)

    data = log_digest.digest(log)

    assert data["times"] == [1.0, 2.0]
    assert [v for _, v in data["residuals"]["Ux"]] == [0.1, 0.005]
    assert data["final_residual"]["p"] == 5.0e-04
    assert data["iterations"]["p"] == 7


def test_log_digest_keeps_the_most_recent_continuity_and_courant(tmp_path, log_digest):
    log = tmp_path / "log"
    log.write_text(SOLVER_LOG)

    data = log_digest.digest(log)

    assert data["continuity"] == (7.5e-07, -1.0e-09, -4.0e-09)
    assert data["courant"] == (0.12, 0.87)
    assert data["exec_time"] == 2.80


def test_log_digest_counts_bounding_messages(tmp_path, log_digest):
    log = tmp_path / "log"
    log.write_text(SOLVER_LOG)
    assert log_digest.digest(log)["bounding"] == {"k": 2}


def test_log_digest_writes_a_plot(tmp_path, log_digest):
    log = tmp_path / "log"
    log.write_text(SOLVER_LOG)
    out = tmp_path / "residuals.png"

    log_digest.plot(log_digest.digest(log)["residuals"], out)

    assert out.exists() and out.stat().st_size > 1000


def test_log_digest_survives_an_empty_log(tmp_path, log_digest):
    log = tmp_path / "log"
    log.write_text("")
    data = log_digest.digest(log)
    assert data["residuals"] == {}
    assert log_digest.report(data, log, None)


# -- mesh_digest ---------------------------------------------------------------

CHECK_MESH = """\
Mesh stats
    points:           98765
    faces:           287654
    internal faces:  278000
    cells:            94321
    boundary patches:     4
Overall number of cells of each type:
    hexahedra:     94000
    polyhedra:       321
Checking geometry...
    Overall domain bounding box (0 0 0) (0.3 0.1 0.01)
    Max cell openness = 2.1e-16 OK.
    Max aspect ratio = 3.24 OK.
    Minimum face area = 1.2e-08. Maximum face area = 2.4e-06.  Face area magnitudes OK.
    Min volume = 1.1e-12. Max volume = 3.4e-09.  Total volume = 0.0003.
    Mesh non-orthogonality Max: 42.13 average: 8.31
    Max skewness = 1.92 OK.
    Minimum face determinant = 0.41
Checking patch topology for multiply connected surfaces...
    Patch               Faces    Points   Surface topology
    inlet                  40        82   ok (non-closed singly connected)
    walls                8600     17000   ok (non-closed singly connected)

Mesh OK.
"""


@pytest.fixture
def mesh_digest():
    return load("mesh_digest")


def test_mesh_digest_reads_counts_and_patches(mesh_digest):
    data = mesh_digest.parse(CHECK_MESH)

    assert data["counts"]["cells"] == 94321
    assert data["counts"]["hexahedra"] == 94000
    assert [row[0] for row in data["patches"]] == ["inlet", "walls"]
    assert data["patches"][1][1] == "8600"


def test_mesh_digest_does_not_swallow_the_trailing_period(mesh_digest):
    """Regression: the number regex used to capture checkMesh's sentence period."""
    data = mesh_digest.parse(CHECK_MESH)

    assert data["face_area"] == ("1.2e-08", "2.4e-06")
    assert data["cell_volume"] == ("1.1e-12", "3.4e-09")
    for value in (*data["face_area"], *data["cell_volume"]):
        assert not value.endswith(".")
        float(value)


def test_mesh_digest_reads_the_quality_metrics(mesh_digest):
    data = mesh_digest.parse(CHECK_MESH)
    assert data["non_ortho"] == ("42.13", "8.31")
    assert data["skewness"] == "1.92"
    assert data["aspect_ratio"] == "3.24"
    assert data["bounding_box"] == ("0 0 0", "0.3 0.1 0.01")


def test_mesh_digest_offers_no_verdict(mesh_digest):
    """The model judges the mesh; this only reports the numbers."""
    report = mesh_digest.report(mesh_digest.parse(CHECK_MESH)).lower()
    for verdict in ("too high", "acceptable", "bad mesh", "should refine", "unacceptable"):
        assert verdict not in report


def test_mesh_digest_surfaces_failed_checks(mesh_digest):
    failing = CHECK_MESH.replace(
        "Max skewness = 1.92 OK.", "Max skewness = 12.4 ***Max skewness too high"
    )
    data = mesh_digest.parse(failing)
    assert any("skewness" in item for item in data["failures"])
    assert "***" in mesh_digest.report(data) or "flagged" in mesh_digest.report(data)


def test_mesh_digest_handles_junk(mesh_digest):
    data = mesh_digest.parse("not a checkMesh log at all")
    assert data["counts"] == {}
    assert mesh_digest.report(data).startswith("# checkMesh")


# -- cells_estimate ------------------------------------------------------------

BLOCK_MESH = """\
convertToMeters 1;
vertices
(
    (0 0 0)
    (2 0 0)
    (2 1 0)
    (0 1 0)
    (0 0 1)
    (2 0 1)
    (2 1 1)
    (0 1 1)
);
blocks
(
    hex (0 1 2 3 4 5 6 7) (40 20 20) simpleGrading (1 1 1)
);
"""

SNAPPY = """\
castellatedMeshControls
{
    refinementSurfaces
    {
        body
        {
            level (2 3);
        }
        ground
        {
            level (1 1);
        }
    }
}
addLayersControls
{
    nSurfaceLayers 4;
}
"""


@pytest.fixture
def cells_estimate():
    return load("cells_estimate")


def ascii_stl(path: Path, triangles):
    lines = ["solid body"]
    for tri in triangles:
        lines.append("  facet normal 0 0 1\n    outer loop")
        lines += [f"      vertex {v[0]} {v[1]} {v[2]}" for v in tri]
        lines.append("    endloop\n  endfacet")
    lines.append("endsolid body")
    path.write_text("\n".join(lines))
    return path


def binary_stl(path: Path, triangles):
    payload = b"\0" * 80 + struct.pack("<I", len(triangles))
    for tri in triangles:
        payload += struct.pack("<3f", 0.0, 0.0, 1.0)
        for vertex in tri:
            payload += struct.pack("<3f", *vertex)
        payload += struct.pack("<H", 0)
    path.write_bytes(payload)
    return path


UNIT_SQUARE = [
    ((0, 0, 0), (1, 0, 0), (1, 1, 0)),
    ((0, 0, 0), (1, 1, 0), (0, 1, 0)),
]


def test_background_cell_size_comes_from_the_block(cells_estimate):
    delta0, volume, extent = cells_estimate.block_mesh_delta(BLOCK_MESH)
    assert delta0 == pytest.approx(0.05)
    assert volume == pytest.approx(2.0)
    assert extent == (2.0, 1.0, 1.0)


def test_convert_to_metres_is_applied(cells_estimate):
    """A dict written in millimetres is the classic scale trap."""
    delta0, volume, _ = cells_estimate.block_mesh_delta(
        BLOCK_MESH.replace("convertToMeters 1;", "convertToMeters 0.001;")
    )
    assert delta0 == pytest.approx(0.00005)
    assert volume == pytest.approx(2e-9)


def test_a_dict_it_cannot_read_yields_nothing(cells_estimate):
    assert cells_estimate.block_mesh_delta("") == (None, None, ())


def test_refinement_levels_take_the_maximum_of_the_pair(cells_estimate):
    assert cells_estimate.surface_levels(SNAPPY) == {"body": 3, "ground": 1}


def test_no_refinement_section_is_not_an_error(cells_estimate):
    assert cells_estimate.surface_levels("") == {}


@pytest.mark.parametrize("writer", [ascii_stl, binary_stl])
def test_stl_area_from_either_format(tmp_path, cells_estimate, writer):
    path = writer(tmp_path / "body.stl", UNIT_SQUARE)
    area, extent, count = cells_estimate.stl_area(path)

    assert area == pytest.approx(1.0)
    assert count == 2
    assert np.allclose(extent, [1, 1, 0])


def test_an_unreadable_stl_reports_zero_rather_than_raising(tmp_path, cells_estimate):
    path = tmp_path / "empty.stl"
    path.write_bytes(b"")
    assert cells_estimate.stl_area(path) == (0.0, pytest.approx(np.zeros(3)), 0)


def test_the_estimate_is_the_documented_arithmetic(cells_estimate):
    result = cells_estimate.estimate(
        delta0=0.05, volume=2.0, levels={"body": 3}, area=1.0, n_layers=4
    )
    assert result["background"] == pytest.approx(2.0 / 0.05**3)
    assert result["surface"] == pytest.approx(1.0 * 8 / 0.05**2)
    assert result["delta_finest"] == pytest.approx(0.00625)
    assert result["layers"] == pytest.approx(4 * 1.0 / 0.00625**2)
    assert result["total"] == pytest.approx(
        result["background"] + result["surface"] + result["layers"]
    )


def test_the_estimate_degrades_without_geometry(cells_estimate):
    result = cells_estimate.estimate(delta0=0.05, volume=2.0, levels={}, area=0.0, n_layers=0)
    assert result["surface"] == 0.0
    assert result["layers"] == 0.0
    assert result["background"] > 0


# -- render --------------------------------------------------------------------


def test_render_is_valid_python():
    """It needs pyvista, which lives in the container, so this is a parse check."""
    ast.parse((TOOLBOX / "render.py").read_text(encoding="utf-8"))


def test_every_toolbox_script_has_a_usage_docstring():
    for script in sorted(TOOLBOX.glob("*.py")):
        doc = ast.get_docstring(ast.parse(script.read_text(encoding="utf-8")))
        assert doc, f"{script.name} has no docstring"
        assert "python3" in doc, f"{script.name} does not show how to run it"


def test_the_toolbox_sticks_to_what_the_image_provides():
    """numpy, matplotlib, pandas and pyvista are installed; scipy is not."""
    allowed = {
        "argparse", "ast", "collections", "io", "json", "math", "os", "pathlib", "re",
        "struct", "subprocess", "sys", "textwrap", "numpy", "matplotlib", "pandas",
        "pyvista", "__future__", "typing", "dataclasses",
    }
    for script in sorted(TOOLBOX.glob("*.py")):
        tree = ast.parse(script.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                roots = [(node.module or "").split(".")[0]]
            else:
                continue
            for root in roots:
                assert root in allowed, f"{script.name} imports {root}, absent from the image"
