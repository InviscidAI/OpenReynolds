"""The snappy recipe folder, the patch enumerator, and the CHANGE_ME rule.

These are the tests that would fail an implementation that runs without erroring and
is wrong, which is the failure mode the whole meshing path exists to catch:
`snappyHexMesh` exits 0 after a stage that achieved nothing, a surface left out of one
of the three lists is still meshed into a patch nobody named, an inverted normal or a
misplaced `locationInMesh` meshes the outside of a duct, and a layer stage that
inserted prisms and deleted them again prints a number it believes.

So every claim here is measured off `constant/polyMesh` after a real mesher run. The
OpenFOAM tests skip cleanly where OpenFOAM is not installed, which keeps the suite
portable; where it is installed they are the gate.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLBOX = ROOT / "openreynolds" / "toolbox"
SNAPPY = TOOLBOX / "templates" / "snappy"
CFMESH = TOOLBOX / "templates" / "cfmesh"
DUCT = ROOT / "tests" / "data" / "snappy" / "duct"

TOKEN = "CHANGE_ME"
PLACEHOLDER = f"({TOKEN} {TOKEN} {TOKEN})"

# The duct's own numbers, which the fixture case is built around. They are restated
# here rather than imported from the dictionaries: a test that reads its expectation
# out of the artifact under test measures nothing.
DX = 0.005
BOX_MIN = (-0.0225, -0.0225, -0.0225)
BOX_MAX = (0.3225, 0.0725, 0.0725)
INSIDE = (0.15, 0.025, 0.025)

MESH_TIMEOUT_S = 900


def load(name: str):
    """Import a toolbox script by path; the directory is data, not a package."""
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def patch_entries():
    return load("patch_entries")


@pytest.fixture(scope="session")
def preflight():
    return load("preflight")


@pytest.fixture(scope="session")
def layer_report():
    return load("layer_report")


@pytest.fixture(scope="session")
def cfmesh():
    return load("cfmesh")


# -- OpenFOAM, which is installed here and is not on PATH ---------------------------


def foam_bashrc() -> Path | None:
    """The `etc/bashrc` that puts the OpenFOAM utilities on PATH, or None."""
    named = os.environ.get("FOAM_BASHRC")
    if named and Path(named).is_file():
        return Path(named)
    etc = os.environ.get("FOAM_ETC")
    if etc and (Path(etc) / "bashrc").is_file():
        return Path(etc) / "bashrc"
    for root in (Path("/usr/lib/openfoam"), Path("/opt"), Path("/usr/lib")):
        if not root.is_dir():
            continue
        for candidate in sorted(root.glob("openfoam*/etc/bashrc"), reverse=True):
            return candidate
    return None


BASHRC = foam_bashrc()
needs_openfoam = pytest.mark.skipif(
    BASHRC is None,
    reason="OpenFOAM is not installed here; the meshing gate needs it on the machine",
)


def foam(command: str, cwd: Path) -> subprocess.CompletedProcess:
    """One shell, sourcing the OpenFOAM environment and then running `command`."""
    assert BASHRC is not None
    return subprocess.run(
        ["bash", "-lc", f"source {BASHRC} && cd {cwd} && {command}"],
        capture_output=True, text=True, timeout=MESH_TIMEOUT_S,
    )


# -- the case the dictionaries are asked to mesh ------------------------------------


FV_SCHEMES = """\
FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""

FV_SOLUTION = """\
FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers {}
"""

CONTROL_DICT = """\
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""


def vector(values) -> str:
    return "(" + " ".join(f"{value:g}" for value in values) + ")"


def fill_placeholders(path: Path, replacements: list[str]) -> None:
    """Replace `(CHANGE_ME CHANGE_ME CHANGE_ME)` occurrences, in order, and NOTHING
    else -- the property that separates a recipe from a description of one.

    It asserts that too: the only lines that differ afterwards are the lines that
    carried the token.
    """
    before = path.read_text(encoding="utf-8")
    after = before
    for replacement in replacements:
        assert PLACEHOLDER in after, f"{path.name} has no {PLACEHOLDER} left to fill"
        after = after.replace(PLACEHOLDER, replacement, 1)
    changed = [
        (old, new)
        for old, new in zip(before.splitlines(), after.splitlines())
        if old != new
    ]
    assert len(changed) == len(replacements)
    for old, _ in changed:
        assert TOKEN in old, f"{path.name}: a line without the token was edited: {old}"
    path.write_text(after, encoding="utf-8")


def build_case(where: Path, patch_entries, level: str = "0 0",
               layers_on: bool = False) -> Path:
    """The shipped dictionaries, the committed duct, and nothing hand-written except
    the two fvDicts snappyHexMesh insists on and the case's controlDict."""
    case = where
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
    for name in ("inlet.stl", "outlet.stl", "walls.stl", "patches.json"):
        (case / "constant" / "triSurface" / name).write_bytes((DUCT / name).read_bytes())
    for dictionary in sorted(SNAPPY.glob("*Dict")):
        (case / "system" / dictionary.name).write_text(
            dictionary.read_text(encoding="utf-8"), encoding="utf-8"
        )
    (case / "system" / "controlDict").write_text(CONTROL_DICT)
    (case / "system" / "fvSchemes").write_text(FV_SCHEMES)
    (case / "system" / "fvSolution").write_text(FV_SOLUTION)

    assert patch_entries.main([
        str(case),
        "--insert", "system/snappyHexMeshDict",
        "--insert", "system/surfaceFeatureExtractDict",
        "--level", level,
    ]) == 0

    fill_placeholders(case / "system" / "blockMeshDict",
                      [vector(BOX_MIN), vector(BOX_MAX)])
    fill_placeholders(case / "system" / "snappyHexMeshDict", [vector(INSIDE)])

    if layers_on:
        path = case / "system" / "snappyHexMeshDict"
        text = path.read_text(encoding="utf-8")
        text = text.replace("addLayers       false;", "addLayers       true;")
        text = text.replace(
            '        // "walls"\n        // {\n        //     nSurfaceLayers 3;\n        // }',
            '        "walls"\n        {\n            nSurfaceLayers 3;\n        }',
        )
        assert "nSurfaceLayers 3;" in text and "// nSurfaceLayers" not in text
        path.write_text(text, encoding="utf-8")
    return case


def mesh_it(case: Path) -> dict:
    """blockMesh, surfaceFeatureExtract, snappyHexMesh, checkMesh -- as the README
    spells them, in one shell each, with the logs kept."""
    runs = {}
    for step in ("blockMesh", "surfaceFeatureExtract", "snappyHexMesh -overwrite",
                 "checkMesh"):
        name = step.split()[0]
        done = foam(f"{step} > log.{name} 2>&1", case)
        runs[name] = {
            "returncode": done.returncode,
            "log": (case / f"log.{name}").read_text(errors="replace"),
        }
        assert done.returncode == 0, f"{step} exited {done.returncode}:\n{runs[name]['log'][-2000:]}"
    return runs


@pytest.fixture(scope="session")
def baseline(tmp_path_factory, patch_entries):
    """The duct at the background cell size, no layers. Also the reference wall for
    the layer measurement: the same wall meshed with no layer request."""
    if BASHRC is None:
        pytest.skip("OpenFOAM is not installed here")
    case = build_case(tmp_path_factory.mktemp("duct-l0"), patch_entries)
    return case, mesh_it(case)


@pytest.fixture(scope="session")
def refined(tmp_path_factory, patch_entries):
    """The same duct with the surface refinement level doubled, and nothing else
    changed."""
    if BASHRC is None:
        pytest.skip("OpenFOAM is not installed here")
    case = build_case(tmp_path_factory.mktemp("duct-l1"), patch_entries, level="1 1")
    return case, mesh_it(case)


@pytest.fixture(scope="session")
def layered(tmp_path_factory, patch_entries):
    """The same duct with three layers asked for on `walls`."""
    if BASHRC is None:
        pytest.skip("OpenFOAM is not installed here")
    case = build_case(tmp_path_factory.mktemp("duct-layers"), patch_entries,
                      layers_on=True)
    return case, mesh_it(case)


# -- reading the mesh back ---------------------------------------------------------


def boundary_of(case: Path, layer_report) -> dict:
    return layer_report.load(case)["boundary"]


def cell_count(case: Path, preflight) -> int:
    count = preflight.Case(case).cell_count
    assert count, "the mesh reports no cell count"
    return int(count)


def duct_volume(preflight) -> float:
    """The fluid volume, measured off the committed STLs rather than restated: the
    duct is a box, so its bounding box is its volume."""
    triangles = np.concatenate([
        preflight.read_triangles(DUCT / name)
        for name in ("inlet.stl", "outlet.stl", "walls.stl")
    ])
    points = triangles.reshape(-1, 3)
    spans = points.max(axis=0) - points.min(axis=0)
    return float(spans[0] * spans[1] * spans[2])


# -- the patch names survive -------------------------------------------------------


@needs_openfoam
def test_the_patch_names_survive_the_mesher(baseline, layer_report):
    """Exactly the three exported names, each with faces, and nothing snappy invented.

    A surface missing from `refinementSurfaces` is still meshed; its faces land in a
    default patch and take that patch's boundary condition, and the run finishes. The
    name of that patch in the mesh is the whole evidence.
    """
    case, _ = baseline
    boundary = boundary_of(case, layer_report)
    assert set(boundary) == {"inlet", "outlet", "walls"}, sorted(boundary)
    for name, entry in boundary.items():
        assert entry["nFaces"] > 0, f"{name} is in the boundary with no faces on it"
    assert "defaultFaces" not in boundary
    assert "patch0" not in boundary
    assert boundary["walls"]["type"] == "wall"
    assert boundary["inlet"]["type"] == "patch"


@needs_openfoam
def test_check_mesh_is_clean_on_the_shipped_dictionaries(baseline):
    case, runs = baseline
    assert "Mesh OK." in runs["checkMesh"]["log"], runs["checkMesh"]["log"][-2000:]
    assert "***" not in runs["checkMesh"]["log"]


# -- inside, not outside -----------------------------------------------------------


@needs_openfoam
def test_the_mesh_is_the_inside_of_the_duct_not_the_outside(baseline, preflight):
    """An inverted patch normal or a wrong `locationInMesh` meshes the outside and
    exits 0. Arithmetic on the cell count is what tells the two apart: the background
    box is far larger than the duct, so meshing the wrong side is not a near miss.
    """
    case, _ = baseline
    volume = duct_volume(preflight)
    expected = volume / DX ** 3
    measured = cell_count(case, preflight)
    outside = (
        (BOX_MAX[0] - BOX_MIN[0]) * (BOX_MAX[1] - BOX_MIN[1]) * (BOX_MAX[2] - BOX_MIN[2])
        - volume
    ) / DX ** 3
    assert abs(measured - expected) <= 0.10 * expected, (
        f"{measured} cells against {expected:.0f} for the duct interior "
        f"({outside:.0f} would be the outside)"
    )


# -- refinement is honoured in the mesh --------------------------------------------


@needs_openfoam
def test_doubling_the_refinement_level_halves_the_near_wall_cell(
    baseline, refined, layer_report
):
    """Measured off `polyMesh`, not read back out of the dictionary.

    `face_spacing` is the square root of the mean face area on the patch, which is
    the near-wall cell size as the mesh actually carries it.
    """
    coarse, _ = baseline
    fine, _ = refined
    coarse_spacing = layer_report.face_spacing(layer_report.load(coarse), "walls")
    fine_spacing = layer_report.face_spacing(layer_report.load(fine), "walls")
    ratio = coarse_spacing / fine_spacing
    assert 1.8 <= ratio <= 2.2, (
        f"level (0 0) gave {coarse_spacing:.6f} m at the wall and level (1 1) gave "
        f"{fine_spacing:.6f} m: a ratio of {ratio:.3f}, not a halving"
    )


# -- layer coverage is measured, not claimed ---------------------------------------


@needs_openfoam
def test_layer_coverage_comes_from_the_mesh(layered, baseline, layer_report):
    """The number is read off `polyMesh` against the same wall meshed with no layer
    request, which is the only reading that means the same thing twice."""
    case, _ = layered
    reference, _ = baseline
    found = layer_report.measure(case, ["walls"], reference)
    assert found["problems"] == []
    (patch,) = found["patches"]
    coverage = patch["coverage_pct"]
    assert coverage is not None, patch.get("note")
    assert patch["basis"] == "per face against the reference mesh"
    assert patch["layered_faces"] > 0


@needs_openfoam
def test_coverage_under_half_is_a_warn_with_the_number_in_it(
    baseline, layered, layer_report, cfmesh
):
    """Never a silent success. That is the exact failure this path was built around:
    88.6% eroded to 66.7%, printed 66.7%, exited 0.

    The bare mesh measured against its own no-layer wall spacing is a wall with no
    layer on it, which is the under-50% case, and it has to say so with the number.
    """
    bare, _ = baseline
    spacing = layer_report.face_spacing(layer_report.load(bare), "walls")
    findings = cfmesh.check(bare, wall="walls", wall_spacing=spacing)
    coverage = [f for f in findings if f.check == "cfmesh-coverage"]
    assert coverage, [f.check for f in findings]
    (finding,) = coverage
    measured = layer_report.measure(bare, ["walls"], None, spacing)["patches"][0]
    assert measured["coverage_pct"] < 50
    assert finding.status == "warn"
    assert f"{measured['coverage_pct']:.1f}%" in finding.measured

    # and the layered mesh's number is in its finding too, whatever it is
    case, _ = layered
    reported = [
        f for f in cfmesh.check(case, wall="walls", ref=bare)
        if f.check == "cfmesh-coverage"
    ]
    (finding,) = reported
    achieved = layer_report.measure(case, ["walls"], bare)["patches"][0]["coverage_pct"]
    assert f"{achieved:.1f}%" in finding.measured
    assert finding.status in ("ok", "warn")


# -- CHANGE_ME is loud -------------------------------------------------------------


def test_the_shipped_dictionaries_carry_the_token():
    """Asserted, so that no default is ever quietly baked in where nothing can be
    guessed. Two things carry it: the background box and the point that says which
    side of the surface is fluid."""
    box = (SNAPPY / "blockMeshDict").read_text(encoding="utf-8")
    assert box.count(PLACEHOLDER) == 2, "boxMin and boxMax both have to be asked for"
    snappy = (SNAPPY / "snappyHexMeshDict").read_text(encoding="utf-8")
    assert f"locationInMesh {PLACEHOLDER};" in snappy
    assert TOKEN in (CFMESH / "meshDict").read_text(encoding="utf-8")


def test_a_dictionary_still_holding_the_token_is_a_preflight_fail(preflight, tmp_path):
    """Before any mesher runs, and naming the file and the line."""
    case = tmp_path / "case"
    (case / "system").mkdir(parents=True)
    (case / "system" / "blockMeshDict").write_text(
        (SNAPPY / "blockMeshDict").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (finding,) = preflight.check_change_me(preflight.Case(case), preflight.Intent())
    assert finding.status == "fail"
    assert finding.check == "change_me"
    line = next(
        number
        for number, text in enumerate(
            (case / "system" / "blockMeshDict").read_text().splitlines(), start=1
        )
        if TOKEN in text
    )
    assert f"system/blockMeshDict:{line}" in finding.measured
    assert "blockMeshDict" in finding.measured


def test_the_check_is_in_the_register_and_passes_a_finished_case(preflight, tmp_path):
    assert "change_me" in preflight.CHECKS
    assert preflight.CHECK_ORDER[0] == "method", "the order of the others is not mine"
    case = tmp_path / "case"
    (case / "system").mkdir(parents=True)
    text = (SNAPPY / "blockMeshDict").read_text(encoding="utf-8")
    (case / "system" / "blockMeshDict").write_text(
        text.replace(PLACEHOLDER, "(0 0 0)"), encoding="utf-8"
    )
    (finding,) = preflight.check_change_me(preflight.Case(case), preflight.Intent())
    assert finding.status == "ok"


def test_a_mesh_directory_does_not_make_the_check_expensive(preflight, tmp_path):
    """`polyMesh` is megabytes of numbers and no dictionary anybody edits."""
    case = tmp_path / "case"
    (case / "constant" / "polyMesh").mkdir(parents=True)
    (case / "constant" / "polyMesh" / "points").write_text(f"// {TOKEN}\n")
    (case / "system").mkdir()
    (case / "system" / "controlDict").write_text("application simpleFoam;\n")
    (finding,) = preflight.check_change_me(preflight.Case(case), preflight.Intent())
    assert finding.status == "ok"


# -- the enumerator: exhaustive, disjoint, idempotent -------------------------------


def one_triangle(path: Path, name: str, offset: float) -> None:
    path.write_text(
        f"solid {name}\n"
        "  facet normal 0 0 1\n    outer loop\n"
        f"      vertex {offset} 0 0\n      vertex {offset + 1} 0 0\n"
        f"      vertex {offset} 1 0\n"
        "    endloop\n  endfacet\n"
        f"endsolid {name}\n"
    )


def twenty_patch_case(where: Path, count: int = 20) -> Path:
    """Three patches prove nothing. Twenty is where hand-writing the lists fails."""
    surfaces = where / "constant" / "triSurface"
    surfaces.mkdir(parents=True, exist_ok=True)
    (where / "system").mkdir(parents=True, exist_ok=True)
    names = [f"panel{index:02d}" for index in range(count)]
    roles = ["inlet", "outlet"] + ["wall"] * (count - 2)
    for index, (name, role) in enumerate(zip(names, roles)):
        one_triangle(surfaces / f"{name}.stl", name, float(index))
    (surfaces / "patches.json").write_text(json.dumps({
        "unit_metres": 1.0,
        "source": "twenty.step",
        "patches": [
            {"name": name, "file": f"{name}.stl", "triangles": 1,
             "area_m2": 0.5, "role": role}
            for name, role in zip(names, roles)
        ],
        "location_in_mesh": [0.0, 0.0, 0.0],
    }, indent=2))
    for dictionary in sorted(SNAPPY.glob("*Dict")):
        (where / "system" / dictionary.name).write_text(
            dictionary.read_text(encoding="utf-8"), encoding="utf-8"
        )
    return where


ENTRY_PATTERNS = {
    "geometry": re.compile(r"^\s*name (\w+);$", re.M),
    "features": re.compile(r'file "(\w+)\.eMesh"', re.M),
    "refinementSurfaces": re.compile(r"^\s*(\w+)$", re.M),
    "extract": re.compile(r"^\s*(\w+)\.stl$", re.M),
}


def entries(path: Path, section: str) -> list[str]:
    """The patch names one marked section actually names, in the order it names
    them. Counting the string would count `panel00.stl` and `name panel00;` twice."""
    return ENTRY_PATTERNS[section].findall(section_text(path, section))


def section_text(path: Path, section: str) -> str:
    text = path.read_text(encoding="utf-8")
    begin = text.index(f"// PATCH-ENTRIES-BEGIN {section}")
    end = text.index("// PATCH-ENTRIES-END", begin)
    return text[text.index("\n", begin) + 1:end]


def test_every_surface_appears_exactly_once_in_every_section(patch_entries, tmp_path):
    case = twenty_patch_case(tmp_path / "twenty")
    assert patch_entries.main([
        str(case), "--insert", "system/snappyHexMeshDict",
        "--insert", "system/surfaceFeatureExtractDict",
    ]) == 0
    names = [f"panel{index:02d}" for index in range(20)]
    snappy = case / "system" / "snappyHexMeshDict"
    for section in ("geometry", "features", "refinementSurfaces"):
        named = entries(snappy, section)
        assert len(named) == 20, f"{section} names {len(named)} surfaces, not 20"
        assert sorted(named) == names, f"{section} names {sorted(named)}"
        assert len(set(named)) == len(named), f"{section} names one of them twice"
    assert sorted(entries(case / "system" / "surfaceFeatureExtractDict",
                          "extract")) == names


def test_the_insert_is_byte_identical_on_a_second_run(patch_entries, tmp_path):
    case = twenty_patch_case(tmp_path / "twice")
    argv = [str(case), "--insert", "system/snappyHexMeshDict"]
    assert patch_entries.main(argv) == 0
    once = (case / "system" / "snappyHexMeshDict").read_bytes()
    assert patch_entries.main(argv) == 0
    assert (case / "system" / "snappyHexMeshDict").read_bytes() == once


def test_one_more_patch_adds_exactly_one_entry_per_section(patch_entries, tmp_path):
    """Re-export the patches, run it again, the dictionary follows -- and follows
    only where the directory changed."""
    case = twenty_patch_case(tmp_path / "grow")
    argv = [str(case), "--insert", "system/snappyHexMeshDict"]
    assert patch_entries.main(argv) == 0
    snappy = case / "system" / "snappyHexMeshDict"
    before = {
        section: (section_text(snappy, section), entries(snappy, section))
        for section in ("geometry", "features", "refinementSurfaces")
    }
    one_triangle(case / "constant" / "triSurface" / "zzextra.stl", "zzextra", 99.0)
    assert patch_entries.main(argv) == 0
    for section, (old_text, old_names) in before.items():
        new_text = section_text(snappy, section)
        assert new_text.startswith(old_text), (
            f"{section} was rewritten above the new entry"
        )
        assert entries(snappy, section) == old_names + ["zzextra"]
        assert "panel" not in new_text[len(old_text):]


def test_the_enumerator_names_nothing_that_has_no_file(patch_entries, tmp_path):
    """A manifest entry with no STL contributes nothing: there is nothing to hand
    snappy. Whether that disagreement is a problem is cad_audit.py's question."""
    case = twenty_patch_case(tmp_path / "ghost", count=3)
    surfaces = case / "constant" / "triSurface"
    manifest = json.loads((surfaces / "patches.json").read_text())
    manifest["patches"].append(
        {"name": "ghost", "file": "ghost.stl", "triangles": 4, "area_m2": 1.0,
         "role": "wall"}
    )
    (surfaces / "patches.json").write_text(json.dumps(manifest))
    assert patch_entries.main([str(case), "--insert", "system/snappyHexMeshDict"]) == 0
    text = (case / "system" / "snappyHexMeshDict").read_text()
    assert "ghost" not in text


def test_the_patch_type_comes_from_the_manifest_role(patch_entries, tmp_path):
    case = twenty_patch_case(tmp_path / "roles", count=3)
    assert patch_entries.main([str(case), "--insert", "system/snappyHexMeshDict"]) == 0
    body = section_text(case / "system" / "snappyHexMeshDict", "refinementSurfaces")
    assert entries(case / "system" / "snappyHexMeshDict", "refinementSurfaces") == [
        "panel00", "panel01", "panel02"
    ]
    assert body.count("patchInfo { type patch; }") == 2   # the inlet and the outlet
    assert body.count("patchInfo { type wall; }") == 1


def test_a_surface_with_no_manifest_entry_is_reported_not_guessed_about(
    patch_entries, tmp_path, capsys
):
    case = twenty_patch_case(tmp_path / "unnamed", count=3)
    one_triangle(case / "constant" / "triSurface" / "stray.stl", "stray", 50.0)
    assert patch_entries.main([str(case), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    manifest = [f for f in payload["findings"] if f["check"] == "manifest"]
    assert manifest and manifest[0]["status"] == "warn"
    assert "stray" in manifest[0]["measured"]
    assert payload["ok"] is True


def test_the_json_envelope_is_the_frozen_one(patch_entries, tmp_path, capsys):
    case = twenty_patch_case(tmp_path / "envelope", count=3)
    assert patch_entries.main([str(case), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["script"] == "patch_entries"
    assert set(payload) == {"script", "ok", "findings", "measured"}
    for finding in payload["findings"]:
        assert set(finding) == {"check", "status", "measured", "means", "repair"}
    assert payload["measured"]["patches"][0]["triangles"] == 1


def test_the_enumerator_refuses_rather_than_guesses(patch_entries, tmp_path, capsys):
    empty = tmp_path / "empty"
    (empty / "system").mkdir(parents=True)
    assert patch_entries.main([str(empty)]) == 2
    assert capsys.readouterr().err.startswith("refused: ")

    case = twenty_patch_case(tmp_path / "nomarkers", count=3)
    plain = case / "system" / "plainDict"
    plain.write_text("geometry {}\n")
    assert patch_entries.main([str(case), "--insert", "system/plainDict"]) == 2
    assert "refused: " in capsys.readouterr().err
    assert plain.read_text() == "geometry {}\n"

    truncated = case / "system" / "halfDict"
    truncated.write_text("geometry\n{\n// PATCH-ENTRIES-BEGIN geometry\n}\n")
    assert patch_entries.main([str(case), "--insert", "system/halfDict"]) == 2
    assert "refused: " in capsys.readouterr().err
    assert "PATCH-ENTRIES-BEGIN geometry" in truncated.read_text()


def test_print_emits_the_fragments_without_touching_anything(
    patch_entries, tmp_path, capsys
):
    case = twenty_patch_case(tmp_path / "printing", count=3)
    before = (case / "system" / "snappyHexMeshDict").read_bytes()
    assert patch_entries.main([str(case), "--print", "geometry"]) == 0
    printed = capsys.readouterr().out
    assert "type triSurfaceMesh;" in printed
    assert printed.count("name panel") == 3
    assert (case / "system" / "snappyHexMeshDict").read_bytes() == before


# -- the README is sufficient on its own --------------------------------------------


GOTCHAS = {
    "castellatedMesh": "the three stages are named",
    "addLayers": "and the last of them",
    "exits 0": "a late failure still exits 0",
    "88.6%": "the measured erosion",
    "66.7%": "the number it printed instead",
    "layer_report.py": "what reads coverage off the mesh",
    "normal": "orientation is how inside is decided",
    "void": "an inverted patch turns a solid into one",
    "background": "a real chassis outruns a step window",
    "nohup": "and how it is put there",
    "defaultFaces": "the patch unassigned faces land in",
    "locationInMesh": "the point nothing can guess",
    "fvSchemes": "which snappyHexMesh reads although it solves nothing",
}


def test_the_readme_names_every_gotcha():
    """The README carries what the template scripts used to carry in their
    docstrings: known gotchas, baked in rather than left to be rediscovered by
    failing."""
    text = (SNAPPY / "README.md").read_text(encoding="utf-8")
    for token, why in GOTCHAS.items():
        assert token in text, f"the README does not mention {token} -- {why}"


def test_the_cfmesh_readme_names_its_own_gotchas():
    text = (CFMESH / "README.md").read_text(encoding="utf-8")
    for token in ("one surface file", "renameBoundary", "Cannot find any patch names",
                  "88.6%", "66.7%", "100.0%", "layer_report.py", "closed surface",
                  "nohup"):
        assert token in text, f"the cfMesh README does not mention {token}"


@pytest.mark.parametrize("folder", ["snappy", "cfmesh"])
def test_the_readme_offers_rather_than_instructs(folder):
    """The same word list `test_the_index_offers_rather_than_instructs` applies to
    the toolbox index. The free-will contract does not stop at the index."""
    text = " ".join(
        (TOOLBOX / "templates" / folder / "README.md")
        .read_text(encoding="utf-8").lower().split()
    )
    for imperative in ("you must", "always run", "before you", "step 1", "first,"):
        assert imperative not in text, f"{folder}/README.md says '{imperative}'"
    assert "ignore them" in text


def test_the_recipe_folders_hold_no_script_to_copy_and_run():
    """No `.py` to copy, edit and run: the desk reads the README and writes its own
    cells. Dictionaries are data -- snappy reads a file, and there is no version of
    that which is not a file -- so shipping those is not the pattern being removed."""
    for folder in (SNAPPY, CFMESH):
        assert list(folder.glob("*.py")) == []
        assert (folder / "README.md").is_file()
    assert {path.name for path in SNAPPY.glob("*Dict")} == {
        "blockMeshDict", "surfaceFeatureExtractDict", "snappyHexMeshDict"
    }
    assert (CFMESH / "meshDict").is_file()


def test_the_templates_state_the_house_thresholds_rather_than_importing_them():
    """A template that imports the house thresholds is a facade again. The numbers
    are literals in plain sight with a comment beside them."""
    snappy = (SNAPPY / "snappyHexMeshDict").read_text(encoding="utf-8")
    assert "maxNonOrtho         65;" in snappy
    assert "70" in snappy and "85" in snappy, "the comment states what preflight does"
    assert "nCellsBetweenLevels 3;" in snappy
    for path in list(SNAPPY.glob("*Dict")) + [CFMESH / "meshDict"]:
        text = path.read_text(encoding="utf-8")
        assert "#include" not in text, f"{path.name} reaches outside itself"
        for line in text.splitlines():
            if "preflight" not in line:
                continue
            assert "//" in line and line.index("//") < line.index("preflight"), (
                f"{path.name} does more than mention preflight in a comment: {line}"
            )


def test_nothing_is_reimplemented(patch_entries):
    """`patch_entries.py` reads the manifest and the triangles through preflight,
    rather than growing a second, subtly different copy of either."""
    source = (TOOLBOX / "patch_entries.py").read_text(encoding="utf-8")
    assert "import preflight" in source
    assert "preflight.read_triangles" in source
    assert "preflight.SURFACE_SUFFIXES" in source
    assert "preflight.SURFACE_DIRS" in source
    assert patch_entries.Finding.__module__ == "preflight", (
        "the finding record is I1's, imported from preflight, not restated"
    )
    assert patch_entries.Finding._fields == load("preflight").Finding._fields
    assert "struct" not in source, "an STL reader of its own is what this avoids"


# -- the cfMesh recipe --------------------------------------------------------------


@needs_openfoam
def test_the_cfmesh_dictionary_meshes_the_same_duct(tmp_path_factory, layer_report):
    """One surface file, the names listed under `newPatchNames`, and the three
    exported patches survive into the mesh."""
    case = tmp_path_factory.mktemp("duct-cfmesh")
    (case / "system").mkdir(exist_ok=True)
    surfaces = case / "constant" / "triSurface"
    surfaces.mkdir(parents=True)
    merged = "".join(
        (DUCT / f"{name}.stl").read_text() for name in ("inlet", "outlet", "walls")
    )
    (surfaces / "duct.stl").write_text(merged)
    (case / "system" / "controlDict").write_text(CONTROL_DICT)
    (case / "system" / "fvSchemes").write_text(FV_SCHEMES)
    (case / "system" / "fvSolution").write_text(FV_SOLUTION)
    text = (CFMESH / "meshDict").read_text(encoding="utf-8")
    assert f'surfaceFile "{TOKEN}";' in text
    text = text.replace(f'"{TOKEN}"', '"constant/triSurface/duct.stl"')
    text = text.replace(
        '        // "inlet"  { newName inlet;  type patch; }\n'
        '        // "outlet" { newName outlet; type patch; }',
        '        "inlet"  { newName inlet;  type patch; }\n'
        '        "outlet" { newName outlet; type patch; }',
    )
    (case / "system" / "meshDict").write_text(text, encoding="utf-8")

    done = foam("cartesianMesh > log.cartesianMesh 2>&1", case)
    log = (case / "log.cartesianMesh").read_text(errors="replace")
    assert done.returncode == 0, log[-2000:]
    checked = foam("checkMesh > log.checkMesh 2>&1", case)
    assert "Mesh OK." in (case / "log.checkMesh").read_text(errors="replace")
    assert checked.returncode == 0
    boundary = layer_report.load(case)["boundary"]
    assert set(boundary) == {"inlet", "outlet", "walls"}, sorted(boundary)
    assert all(entry["nFaces"] > 0 for entry in boundary.values())


# -- the committed fixture ----------------------------------------------------------


def test_the_committed_duct_is_a_closed_three_patch_set(preflight):
    """One STL per named patch, the union closed, normals consistently outward --
    what the export side is required to guarantee, checked on the fixture the rest of
    this file meshes."""
    manifest = json.loads((DUCT / "patches.json").read_text())
    assert [entry["name"] for entry in manifest["patches"]] == ["inlet", "outlet", "walls"]
    assert manifest["unit_metres"] == 1.0
    triangles = np.concatenate([
        preflight.read_triangles(DUCT / entry["file"]) for entry in manifest["patches"]
    ])
    topology = preflight.surface_topology(triangles)
    assert topology["open_edges"] == 0, "the union has a hole in it"
    assert topology["non_manifold_edges"] == 0
    assert topology["flipped_edges"] == 0, "a patch is wound against its neighbour"
    centre = np.array(manifest["location_in_mesh"])
    points = triangles.reshape(-1, 3)
    assert np.all(points.min(axis=0) < centre) and np.all(centre < points.max(axis=0))
    assert sum((DUCT / entry["file"]).stat().st_size
               for entry in manifest["patches"]) < 20_000
