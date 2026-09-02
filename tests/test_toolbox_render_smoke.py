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

import importlib.util
from pathlib import Path

import pytest

# exc_type=ImportError: a pyvista that is present but cannot load its VTK DLLs
# (seen under a Windows Application Control policy) is the same situation as no
# pyvista -- the drawing half runs on the OpenFOAM image, not here.
pytest.importorskip("pyvista", reason="the drawing half runs on the OpenFOAM image", exc_type=ImportError)

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


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
