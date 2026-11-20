"""The half of the toolbox that draws, actually drawing.

Every other toolbox test runs on a machine with no pyvista, so it tests the parsing,
the tables and the arithmetic and stubs the renderer out. That leaves the drawing
code -- `results.py`'s producers, `first_look.py`'s panels, `animate.py`'s frames --
never executed by anything: an API misuse there is invisible until a study spends a
core-hour reaching it, and everything upstream is already green.

So this file skips itself where pyvista is absent (a developer laptop, CI) and runs
where it is present (the OpenFOAM image, which is where these scripts actually live).
It is deliberately thin: one tiny case, one call per script, and the assertion is
"a PNG came out with bytes in it". Anything finer would be testing pyvista.

    python3 -m pytest tests/test_toolbox_render_smoke.py -q
"""

from __future__ import annotations

import functools
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

# exc_type=ImportError: a pyvista that is present but cannot load its VTK DLLs
# (seen under a Windows Application Control policy) is the same situation as no
# pyvista -- the drawing half runs on the OpenFOAM image, not here.
pytest.importorskip("pyvista", reason="the drawing half runs on the OpenFOAM image", exc_type=ImportError)

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"

PROBE = """
import pyvista as pv
pv.OFF_SCREEN = True
p = pv.Plotter(off_screen=True, window_size=(64, 64))
p.add_mesh(pv.Sphere())
p.screenshot(None, return_img=True)
"""


@functools.lru_cache(maxsize=1)
def _can_draw() -> bool:
    """Whether VTK can draw off-screen here, asked in a child process.

    Without an OpenGL context -- a CI runner, Linux or Windows -- the draw is a segfault
    or an access violation, which no `except` in this process would see; it takes the
    whole session with it. The workspace image has OSMesa and draws. Same probe as
    `test_felix_results.py`, for the same reason.
    """
    try:
        return subprocess.run([sys.executable, "-c", PROBE],
                              capture_output=True, timeout=120).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def needs_a_renderer():
    if not _can_draw():
        pytest.skip("no off-screen OpenGL here; the workspace image draws")


def not_blank(path: Path) -> bool:
    """A PNG with something in it, measured the way `docker/workspace/check.sh` does.

    A render that failed quietly is a file of one flat colour: the right size, the right
    header, and `the render is blank`. Byte count cannot tell the two apart.

    Read with matplotlib rather than imageio: both are on the workspace image, and only
    matplotlib is in this package's `dev` extra, so this works wherever the suite does."""
    import matplotlib.image

    return float(matplotlib.image.imread(path).std()) > 0


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BLOCK_MESH = """\
FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
scale   1;
vertices
(
    (0 0 0) (1 0 0) (1 1 0) (0 1 0)
    (0 0 0.1) (1 0 0.1) (1 1 0.1) (0 1 0.1)
);
blocks
(
    hex (0 1 2 3 4 5 6 7) (3 3 1) simpleGrading (1 1 1)
);
edges ();
boundary
(
    walls
    {
        type wall;
        faces ( (0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) );
    }
    frontAndBack
    {
        type empty;
        faces ( (0 3 2 1) (4 5 6 7) );
    }
);
mergePatchPairs ();
"""


@pytest.fixture(scope="module")
def meshed_case(tmp_path_factory):
    """A three-by-three box, meshed by blockMesh if it is on this machine.

    Skipped rather than faked when blockMesh is absent: a hand-written polyMesh is a
    test of the hand-writing, and the thing worth knowing here is whether these
    scripts can read what OpenFOAM actually produces.
    """
    import shutil
    import subprocess

    if shutil.which("blockMesh") is None:
        pytest.skip("blockMesh is not on this machine")
    case = tmp_path_factory.mktemp("box")
    (case / "system").mkdir()
    (case / "constant").mkdir()
    (case / "system" / "blockMeshDict").write_text(BLOCK_MESH, encoding="utf-8")
    (case / "system" / "controlDict").write_text(
        "application     simpleFoam;\nstartFrom latestTime;\nstartTime 0;\n"
        "stopAt endTime;\nendTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\n",
        encoding="utf-8",
    )
    result = subprocess.run(["blockMesh", "-case", str(case)], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"blockMesh refused the test case: {result.stderr[-300:]}")
    return case


def pngs_under(directory: Path) -> list[Path]:
    return [path for path in directory.rglob("*.png") if path.stat().st_size > 1000]


def test_first_look_draws_a_contact_sheet(meshed_case, tmp_path):
    first_look = load("first_look")

    first_look.main([str(meshed_case), "--out", str(tmp_path / "look")])

    assert pngs_under(tmp_path / "look"), "no panel or sheet came out with bytes in it"


def test_results_draws_the_mesh_validation_preset(meshed_case, tmp_path):
    results = load("results")

    results.main([str(meshed_case), "--preset", "mesh-validation", "--out", str(tmp_path / "res")])

    assert pngs_under(tmp_path / "res"), "the mesh-validation preset drew nothing"


def test_animate_renders_frames_when_there_are_times_to_render(meshed_case, tmp_path):
    """Needs two written times; a mesh-only case has none, so this asserts the
    honest refusal rather than inventing a solution to animate."""
    animate = load("animate")

    with pytest.raises(SystemExit) as refused:
        animate.main([str(meshed_case), "--field", "p", "--out", str(tmp_path / "frames")])

    assert "write time" in str(refused.value).lower() or "2" in str(refused.value)


# -- the comparing views, actually drawn -----------------------------------------------
#
# Built from pyvista grids rather than from an OpenFOAM case: what is being exercised is
# the drawing of a contour set and of a difference pair, and a solved case is a slower
# way to get a field with a gradient in it. The reading of a real case is covered either
# side of this file.


def two_runs(cells: int = 12, second_cells: int = 12):
    """Two runs of one case as uniform grids, the second a little slower everywhere.

    `p` varies across the domain so a contour has somewhere to go and the difference is
    not uniform; the second run's field is the first scaled, so the difference has a
    sign on both sides of zero -- which is what the diverging map is for.
    """
    import numpy as np
    import pyvista as pv

    def run(count: int, gain: float):
        step = 2.0 / count
        mesh = pv.ImageData(dimensions=(count + 1, count + 1, 3),
                            spacing=(step, step, step))
        x = np.asarray(mesh.points)[:, 0]
        y = np.asarray(mesh.points)[:, 1]
        mesh.point_data["p"] = gain * (np.sin(3.0 * x) + 0.5 * y)
        return mesh

    return run(cells, 1.0), run(second_cells, 1.3)


def test_a_field_slice_draws_its_iso_contours(tmp_path):
    needs_a_renderer()
    render = load("render")
    first, _ = two_runs()

    out = render.render_field(first, "p", "z", tmp_path / "p_contours.png", contours=9)

    assert out and out.exists()
    assert not_blank(out), "the render is blank"


def test_two_runs_are_drawn_on_one_scale_and_their_difference_on_another(tmp_path):
    """The whole point of the view, drawn: three PNGs with content in them and a sidecar
    whose numbers say the two panels were drawn the same way."""
    needs_a_renderer()
    render = load("render")
    first, second = two_runs(12, 17)

    record, paths = render.render_difference(first, second, "p", "z", tmp_path,
                                            ("run_A", "run_B"))

    assert record["correspondence"] == "resampled"
    assert record["max_abs_difference"] > 0
    pngs = [path for path in paths if path.suffix == ".png"]
    assert len(pngs) == 3
    for path in pngs:
        assert not_blank(path), f"the render is blank: {path.name}"
    sidecar = json.loads((tmp_path / "p_z_difference.json").read_text(encoding="utf-8"))
    assert sidecar["clim"] == record["clim"]
    assert sidecar["viewup"] == list(render.VIEWUP["z"])


def test_a_difference_against_a_different_domain_is_refused_before_it_draws(tmp_path):
    """The refusal is the feature. Nothing is written, so there is no plausible picture
    of nothing left in the output directory to be read later as a result."""
    needs_a_renderer()
    render = load("render")
    first, _ = two_runs()
    _, far = two_runs()
    far.origin = (5.0, 0.0, 0.0)

    with pytest.raises(SystemExit) as refused:
        render.render_difference(first, far, "p", "z", tmp_path, ("here", "elsewhere"))

    assert "elsewhere" in str(refused.value)
    assert not list(tmp_path.glob("*.png"))
