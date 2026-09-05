"""The toolbox scripts.

They run inside the container, but their parsing is ordinary Python and is exactly the
kind of thing that rots quietly, so it is tested here.
"""

from __future__ import annotations

import ast
import importlib.util
import struct
import sys
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
    """numpy, matplotlib, pandas and pyvista are installed; scipy is not.

    The rule being kept is "nothing the image does not have". It used to be spelled
    as a hand-written list of module names, which meant every script that reached
    for another corner of the standard library -- `time`, `shutil`, `itertools` --
    read as a violation and the list grew by one more line that said nothing. The
    standard library is on the image by definition, so it is asked for by name
    rather than enumerated, and what is left to police is the third-party set and
    the scripts' own siblings.
    """
    # imageio (with the ffmpeg plugin) joined the image when gif/mp4 encoding moved onto
    # the instance; `encode.py` is the script that uses it. It is on the image, so it is
    # allowed here like the other four.
    third_party = {"numpy", "matplotlib", "pandas", "pyvista", "imageio"}
    siblings = {script.stem for script in TOOLBOX.glob("*.py")}
    allowed = set(sys.stdlib_module_names) | third_party | siblings | {"__future__"}
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


def test_the_toolbox_index_names_every_script_in_it():
    """The scripts are discoverable by listing the directory, which tells you their
    filenames and nothing else. An index costs one file and says what each is for."""
    index = (TOOLBOX / "README.md").read_text(encoding="utf-8")
    for script in sorted(TOOLBOX.glob("*.py")):
        assert script.name in index, f"{script.name} is in the toolbox and not in its index"


def test_the_environment_manifest_states_the_facts_the_corpus_kept_missing():
    """235 install attempts and a run of `import fitz` failures came from not knowing
    the instance: the network is sealed, PDFs are pdftoppm not fitz, imageio and gmsh
    are already here. The manifest states those so they are read, not rediscovered."""
    env = (TOOLBOX / "ENVIRONMENT.md").read_text(encoding="utf-8").lower()
    for fact in ("sealed", "pdftoppm", "imageio", "gmsh", "foamtovtk"):
        assert fact in env, f"ENVIRONMENT.md does not mention {fact}"
    assert "fitz" in env, "the manifest should name fitz to steer off it"


def test_the_index_offers_rather_than_instructs():
    """The toolbox is offered, not imposed. An index that told the model when to run
    things would be the workflow injection the whole design exists to avoid."""
    # Normalised, because where a sentence happens to wrap is not the point.
    index = " ".join((TOOLBOX / "README.md").read_text(encoding="utf-8").lower().split())
    for imperative in ("you must", "always run", "before you", "step 1", "first,"):
        assert imperative not in index
    assert "ignore them" in index


# -- the camera actually points where the caption says -------------------------
#
# `render.py --scene mesh` is what the standing note tells every session to use, and
# for a long time it produced a picture rotated a quarter turn -- a 28x16 domain drawn
# as a portrait column with the flow running bottom to top. Three studies in one run
# worked around it by hand-writing eight pyvista scripts and reading thirteen images,
# every one a paid turn. `results.py` and `animate.py` had the fix all along.


class FakePlotter:
    """Records what the camera was told, without drawing anything."""

    def __init__(self):
        self.vector = None
        self.viewup = None
        self.parallel = False
        self.zoomed = None
        self.reset_bounds = None
        self.camera = self

    def view_vector(self, vector, viewup=None):
        self.vector, self.viewup = vector, viewup

    def enable_parallel_projection(self):
        self.parallel = True

    def zoom(self, factor):
        self.zoomed = factor

    def reset_camera(self, bounds=None):
        self.reset_bounds = bounds


def test_a_slice_is_aimed_with_an_up_vector_not_just_a_direction():
    """Without viewup VTK picks, and for a z-normal slice it picks something collinear
    with the view direction and falls back to an arbitrary up."""
    render = load("render")
    plotter = FakePlotter()
    render.aim(plotter, "z")

    assert plotter.vector == render.NORMALS["z"]
    assert plotter.viewup == (0.0, 1.0, 0.0), "y is up, so x runs across the page"
    assert plotter.parallel, "a flat cut drawn in perspective reads as a graded mesh"


def test_render_aims_the_same_way_its_siblings_do():
    """The fix existed in two other files in the same directory and was never
    back-ported to this one."""
    render, results = load("render"), load("results")
    assert render.VIEWUP == results.VIEWUP
    for normal in render.NORMALS:
        here, there = FakePlotter(), FakePlotter()
        render.aim(here, normal)
        results.aim(there, normal)
        assert (here.vector, here.viewup) == (there.vector, there.viewup)


def test_a_close_up_can_be_framed_on_a_region():
    """Asked for the mesh around a step, the tool drew the whole domain as a sliver
    with the step invisible, because there was no way to say where to look."""
    render = load("render")
    plotter = FakePlotter()
    render.frame(plotter, None, zoom=None, bounds=(0, 1, 0, 1, 0, 1))
    assert plotter.reset_bounds == [0, 1, 0, 1, 0, 1]

    plotter = FakePlotter()
    render.frame(plotter, None, zoom=4.0, bounds=None)
    assert plotter.zoomed == 4.0


def test_framing_is_not_applied_unless_asked_for():
    render = load("render")
    plotter = FakePlotter()
    render.frame(plotter, None, zoom=None, bounds=None)
    assert plotter.zoomed is None and plotter.reset_bounds is None


# The reader fails on a `0/` field that is a `$variable`, `flowVelocity`/`Tinf`, or an
# `#includeEtc` -- 73 studies met that and hand-wrote pyvista to get around it. The
# fallback reads foamToVTK output instead; these cover the file-picking it does, which
# is the part that has changed shape across OpenFOAM releases.


def test_time_is_read_out_of_a_foamtovtk_output_name():
    render = load("render")
    assert render.time_from_stem("motorBike_300") == 300.0
    assert render.time_from_stem("cavity_12") == 12.0
    assert render.time_from_stem("internal") == 0.0  # unparseable -> 0, not a failure


def test_newest_internal_vtk_takes_the_last_write_across_either_layout(tmp_path):
    render = load("render")
    vtk = tmp_path / "VTK"
    # modern layout: a per-time subdir with internal.vtu
    old = vtk / "case_100"
    old.mkdir(parents=True)
    (old / "internal.vtu").write_text("old")
    # a later write, legacy layout: a bare .vtk
    new = vtk / "case_200.vtk"
    new.write_text("new")
    import os, time as _t
    later = _t.time() + 10
    os.utime(new, (later, later))
    assert render.newest_internal_vtk(vtk) == new


def test_newest_internal_vtk_is_none_when_foamtovtk_wrote_nothing(tmp_path):
    render = load("render")
    assert render.newest_internal_vtk(tmp_path / "VTK") is None


# -- the digests report the number that measures the thing ---------------------


PIMPLE_LOG = """Time = 1

PIMPLE: iteration 1
smoothSolver:  Solving for Ux, Initial residual = 1e-02, Final residual = 1e-06, No Iterations 3
GAMG:  Solving for p, Initial residual = 5e-02, Final residual = 9e-08, No Iterations 5
time step continuity errors : sum local = 1e-09, global = 1e-12, cumulative = 1e-12
PIMPLE: iteration 2
smoothSolver:  Solving for Ux, Initial residual = 1e-05, Final residual = 1e-09, No Iterations 2
GAMG:  Solving for p, Initial residual = 2e-05, Final residual = 1e-09, No Iterations 4
ExecutionTime = 12 s
Time = 2

smoothSolver:  Solving for Ux, Initial residual = 8e-03, Final residual = 1e-06, No Iterations 3
time step continuity errors : sum local = 1e-06, global = 1e-09, cumulative = 5e-09
ExecutionTime = 25 s
End
"""


def test_a_step_reports_its_outer_residual_not_its_last_inner_corrector(tmp_path):
    """`step` advances only on a `Time =` line, so every PIMPLE inner corrector landed
    on the same step and the last one -- one to two orders lower -- was reported as that
    step's residual. A transient table looked immaculate either way."""
    log_digest = load("log_digest")
    log = tmp_path / "log.pimpleFoam"
    log.write_text(PIMPLE_LOG, encoding="utf-8")

    data = log_digest.digest(log)
    assert data["residuals"]["Ux"] == [(1, 1e-2), (2, 8e-3)], "one point per step"
    assert data["outer_residual"]["Ux"] == 8e-3
    assert 1e-5 not in [v for _, v in data["residuals"]["Ux"]], "the inner corrector"


def test_the_continuity_series_is_kept_not_just_its_last_value(tmp_path):
    """A growing cumulative error is a documented failure signature and only the most
    recent value was kept, which is exactly what made it invisible."""
    log_digest = load("log_digest")
    log = tmp_path / "log.pimpleFoam"
    log.write_text(PIMPLE_LOG, encoding="utf-8")
    data = log_digest.digest(log)
    assert [v for _, v in data["continuity_series"]] == [1e-12, 5e-09]


def test_how_a_run_ended_is_the_first_thing_the_digest_says(tmp_path):
    log_digest = load("log_digest")

    fatal = {"fatal": "--> FOAM FATAL ERROR: no such field", "times": [3.0]}
    assert "FOAM FATAL" in log_digest.how_it_ended(fatal)[0]

    converged = {"converged_at": 412, "times": [412.0]}
    assert "convergence at iteration 412" in log_digest.how_it_ended(converged)[0]

    short = {"times": [37.0], "ended": True}
    assert "stopped at 37 of a requested 1000" in log_digest.how_it_ended(short, 1000.0)[0]

    cut = {"times": [10.0]}
    assert "no End line" in log_digest.how_it_ended(cut)[0]


def test_a_diverged_run_no_longer_reads_like_a_finished_one(tmp_path):
    """It printed a normal-looking table headed "time steps parsed: 37"."""
    log_digest = load("log_digest")
    log = tmp_path / "log.simpleFoam"
    log.write_text(
        "Time = 1\n"
        "smoothSolver:  Solving for Ux, Initial residual = 1, Final residual = 1e30, No Iterations 1000\n"
        "--> FOAM FATAL ERROR: Maximum number of iterations exceeded\n",
        encoding="utf-8")
    assert "FOAM FATAL" in log_digest.report(log_digest.digest(log), log, None)


ADVISORY_MESH = """Checking geometry...
    Overall domain bounding box (0 0 0) (1 1 1)
 *Number of severely non-orthogonal (> 70 degrees) faces: 74.
    Mesh non-orthogonality Max: 86.3 average: 12.1
Mesh OK.
"""


def test_a_one_star_advisory_is_reported_beside_mesh_ok(tmp_path):
    """checkMesh's severe-non-orthogonality advisory carries ONE star; the WARNING
    pattern required two and was never called anyway, so a mesh with 74 faces at 86.3
    degrees was summarised as "Mesh OK" -- which a reader takes it to have ruled out."""
    mesh_digest = load("mesh_digest")
    data = mesh_digest.parse(ADVISORY_MESH)
    assert data["warnings"] == ["Number of severely non-orthogonal (> 70 degrees) faces: 74."]

    text = mesh_digest.report(data)
    assert "advisory" in text and "74" in text
    assert "Mesh OK." in text, "checkMesh's own verdict is still reported, not suppressed"


def test_three_star_failures_are_still_failures_not_advisories():
    mesh_digest = load("mesh_digest")
    data = mesh_digest.parse("***Number of unused points: 12\n")
    assert data["failures"] and not data.get("warnings")
