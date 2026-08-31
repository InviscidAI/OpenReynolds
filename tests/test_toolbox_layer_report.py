"""`layer_report.py`: what the mesh says about its own boundary layer.

The script exists because three meshers' claims about the same hull could not be
compared -- snappyHexMesh prints a layer table, cfMesh prints nothing comparable,
and the hybrid route prints nothing at all -- so the meshes were measured instead.
The measurement is only worth anything if it is the same measurement every time,
which is what these tests are for.

Everything here runs on a hand-written ASCII polyMesh: a structured box whose
wall-normal spacing is chosen so the right answer is known before the script is
asked. The three answers tested hardest are the three that get quoted -- all of a
wall covered, none of it, and a half of it -- because those are the numbers a
study comparing two meshers ends up putting in a table.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


@pytest.fixture(scope="module")
def layer_report():
    spec = importlib.util.spec_from_file_location(
        "layer_report", TOOLBOX / "layer_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# -- a polyMesh small enough to read and check by hand -------------------------


def _foam(cls: str, obj: str, body: str) -> str:
    return (
        "FoamFile\n"
        "{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {cls};\n"
        '    location    "constant/polyMesh";\n'
        f"    object      {obj};\n"
        "}\n\n" + body + "\n"
    )


def _list(entries: list[str]) -> str:
    return f"{len(entries)}\n(\n" + "\n".join(entries) + "\n)\n"


def write_box(case: Path, xs, ys, zs, wall=("zmin",)) -> Path:
    """A structured hex box as an ASCII `constant/polyMesh`.

    `xs`, `ys`, `zs` are node coordinates, so the wall-normal spacing is written
    rather than requested: the first-cell height at the `zmin` wall is exactly
    `zs[1] - zs[0]`, which is the number `layer_report` has to come back with.
    Faces named in `wall` land on the `wall` patch and the rest on `outer`.
    """
    xs, ys, zs = map(np.asarray, (xs, ys, zs))
    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
    nx1, ny1 = nx + 1, ny + 1

    def pt(i, j, k):
        return i + nx1 * (j + ny1 * k)

    def cell(i, j, k):
        return i + nx * (j + ny * k)

    points = [
        f"({xs[i]:.10g} {ys[j]:.10g} {zs[k]:.10g})"
        for k in range(nz + 1)
        for j in range(ny + 1)
        for i in range(nx + 1)
    ]

    internal: list[tuple[int, int, list[int]]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(1, nx):  # faces with a +x normal
                internal.append((cell(i - 1, j, k), cell(i, j, k),
                                 [pt(i, j, k), pt(i, j + 1, k),
                                  pt(i, j + 1, k + 1), pt(i, j, k + 1)]))
    for k in range(nz):
        for j in range(1, ny):  # +y
            for i in range(nx):
                internal.append((cell(i, j - 1, k), cell(i, j, k),
                                 [pt(i, j, k), pt(i, j, k + 1),
                                  pt(i + 1, j, k + 1), pt(i + 1, j, k)]))
    for k in range(1, nz):  # +z
        for j in range(ny):
            for i in range(nx):
                internal.append((cell(i, j, k - 1), cell(i, j, k),
                                 [pt(i, j, k), pt(i + 1, j, k),
                                  pt(i + 1, j + 1, k), pt(i, j + 1, k)]))
    internal.sort(key=lambda f: (f[0], f[1]))  # upper-triangular order

    def zmin_faces():
        return [(cell(i, j, 0), [pt(i, j, 0), pt(i, j + 1, 0),
                                 pt(i + 1, j + 1, 0), pt(i + 1, j, 0)])
                for j in range(ny) for i in range(nx)]

    def zmax_faces():
        return [(cell(i, j, nz - 1), [pt(i, j, nz), pt(i + 1, j, nz),
                                      pt(i + 1, j + 1, nz), pt(i, j + 1, nz)])
                for j in range(ny) for i in range(nx)]

    def xmin_faces():
        return [(cell(0, j, k), [pt(0, j, k), pt(0, j, k + 1),
                                 pt(0, j + 1, k + 1), pt(0, j + 1, k)])
                for k in range(nz) for j in range(ny)]

    def xmax_faces():
        return [(cell(nx - 1, j, k), [pt(nx, j, k), pt(nx, j + 1, k),
                                      pt(nx, j + 1, k + 1), pt(nx, j, k + 1)])
                for k in range(nz) for j in range(ny)]

    def ymin_faces():
        return [(cell(i, 0, k), [pt(i, 0, k), pt(i + 1, 0, k),
                                 pt(i + 1, 0, k + 1), pt(i, 0, k + 1)])
                for k in range(nz) for i in range(nx)]

    def ymax_faces():
        return [(cell(i, ny - 1, k), [pt(i, ny, k), pt(i, ny, k + 1),
                                      pt(i + 1, ny, k + 1), pt(i + 1, ny, k)])
                for k in range(nz) for i in range(nx)]

    sides = {
        "zmin": zmin_faces, "zmax": zmax_faces, "xmin": xmin_faces,
        "xmax": xmax_faces, "ymin": ymin_faces, "ymax": ymax_faces,
    }
    on_wall = [name for name in sides if name in wall]
    on_outer = [name for name in sides if name not in wall]
    wall_faces = [f for name in on_wall for f in sides[name]()]
    outer_faces = [f for name in on_outer for f in sides[name]()]

    faces = [verts for _, _, verts in internal]
    owner = [own for own, _, _ in internal]
    neighbour = [nei for _, nei, _ in internal]
    for own, verts in wall_faces + outer_faces:
        faces.append(verts)
        owner.append(own)

    poly = case / "constant" / "polyMesh"
    poly.mkdir(parents=True, exist_ok=True)
    (poly / "points").write_text(_foam("vectorField", "points", _list(points)))
    (poly / "faces").write_text(
        _foam("faceList", "faces",
              _list([f"4({' '.join(str(v) for v in f)})" for f in faces]))
    )
    (poly / "owner").write_text(
        _foam("labelList", "owner", _list([str(v) for v in owner]))
    )
    (poly / "neighbour").write_text(
        _foam("labelList", "neighbour", _list([str(v) for v in neighbour]))
    )
    start = len(internal)
    (poly / "boundary").write_text(
        _foam("polyBoundaryMesh", "boundary", _list([
            "    wall\n    {\n        type            wall;\n"
            f"        nFaces          {len(wall_faces)};\n"
            f"        startFace       {start};\n    }}",
            "    outer\n    {\n        type            patch;\n"
            f"        nFaces          {len(outer_faces)};\n"
            f"        startFace       {start + len(wall_faces)};\n    }}",
        ]))
    )
    return case


XS = [0.0, 0.25, 0.5, 0.75, 1.0]
YS = [0.0, 0.4, 1.0]
BARE = [0.0, 0.2, 0.5, 1.0]  # no layer: the first cell is a fifth of the domain
LAYERED = [0.0, 0.004, 0.2, 0.5, 1.0]  # a 4 mm first cell against that 200 mm one


def layered(root: Path) -> Path:
    return write_box(root / "layered", XS, YS, LAYERED)


def bare(root: Path) -> Path:
    return write_box(root / "bare", XS, YS, BARE)


# -- the measurement itself ----------------------------------------------------


def test_the_first_cell_height_is_the_spacing_that_was_written(layer_report, tmp_path):
    """Before any coverage number means anything, `2 |C_face - C_owner|` has to come
    back as the wall-normal spacing that went in. The centres are computed here from
    `points`, `faces` and `owner` rather than read from a written field, so this is
    the check that the decomposition is OpenFOAM's and not an approximation."""
    mesh = layer_report.load(layered(tmp_path))
    h, centres = layer_report.first_cell_heights(mesh, "wall")

    assert h.size == (len(XS) - 1) * (len(YS) - 1)
    assert np.allclose(h, LAYERED[1] - LAYERED[0])
    assert np.allclose(centres[:, 2], 0.0), "the wall faces sit on z = 0"


def test_a_wall_layered_everywhere_reads_as_full_coverage(layer_report, tmp_path):
    """The cfMesh answer: 100.0%, on a mesh whose mesher reported nothing at all."""
    found = layer_report.measure(layered(tmp_path), ["wall"], ref=bare(tmp_path))
    assert found["problems"] == []
    (patch,) = found["patches"]

    assert patch["coverage_pct"] == 100.0
    assert patch["layered_faces"] == patch["faces"] == 8
    assert patch["basis"] == "per face against the reference mesh"
    assert patch["first_cell_m"]["max"] == pytest.approx(0.004)
    # 200 mm of bare wall cell against 4 mm of layered one.
    assert patch["thinning"]["median"] == pytest.approx(50.0)
    assert patch["reference_spacing_m"] == pytest.approx(0.2)


def test_a_wall_with_no_layer_on_it_reads_as_zero_coverage(layer_report, tmp_path):
    """The three snappy rounds that put layers on none of the hull. Zero has to be
    reachable and reportable: a metric that cannot say "none" cannot say "some"."""
    case = bare(tmp_path)
    found = layer_report.measure(case, ["wall"], ref=write_box(
        tmp_path / "same", XS, YS, BARE
    ))
    (patch,) = found["patches"]

    assert patch["coverage_pct"] == 0.0
    assert patch["layered_faces"] == 0
    assert patch["thinning"]["median"] == pytest.approx(1.0)
    assert "0.0 %" in layer_report.report(found)


def test_a_half_covered_wall_lands_between_the_two(layer_report, tmp_path):
    """The snappy round-4 mesh was neither: 65.2% of the hull carried a layer and
    the rest did not, and it is the partial number the comparison actually turned
    on. Here the `wall` patch is both ends of the box, one layered and one not."""
    case = write_box(tmp_path / "half", XS, YS, LAYERED, wall=("zmin", "zmax"))
    ref = write_box(tmp_path / "half-ref", XS, YS, BARE, wall=("zmin", "zmax"))
    (patch,) = layer_report.measure(case, ["wall"], ref=ref)["patches"]

    assert patch["faces"] == 16
    assert patch["coverage_pct"] == 50.0
    assert patch["layered_faces"] == 8
    assert patch["first_cell_m"]["min"] == pytest.approx(0.004)
    assert patch["first_cell_m"]["max"] == pytest.approx(0.5)


def test_a_named_wall_spacing_stands_in_for_the_reference_mesh(layer_report, tmp_path):
    """The no-layer mesh is often not around any more, but its wall spacing is
    written down. One number is enough for a threshold and a thinning ratio."""
    (patch,) = layer_report.measure(
        layered(tmp_path), ["wall"], wall_spacing=0.2
    )["patches"]

    assert patch["coverage_pct"] == 100.0
    assert patch["basis"] == "half the reference wall spacing"
    assert patch["thinning"]["median"] == pytest.approx(50.0)


def test_without_a_reference_the_thinning_is_absent_and_the_weakness_is_stated(
    layer_report, tmp_path
):
    """A wall measured against itself cannot be compared to anything, so there is
    no thinning ratio to give -- and the coverage number it can still produce is
    the one that goes wrong at both ends, which the note has to say out loud."""
    (patch,) = layer_report.measure(layered(tmp_path), ["wall"])["patches"]

    assert patch["thinning"] is None
    assert patch["reference_spacing_m"] is None
    assert "no reference given" in patch["basis"]
    assert "wholly layered" in patch["note"]
    assert "not derivable" in layer_report.report({**layer_report.measure(
        layered(tmp_path), ["wall"]
    )})


def test_wall_patches_are_the_default_and_an_inlet_is_not_one(layer_report, tmp_path):
    """A prism layer on an `outer` patch of type `patch` is not a thing anyone
    asked for, so the walls are what gets measured when no patch is named."""
    found = layer_report.measure(layered(tmp_path))
    assert [p["patch"] for p in found["patches"]] == ["wall"]


# -- the contract with whoever runs it -----------------------------------------


def test_json_is_the_whole_report_and_parses(layer_report, tmp_path, capsys):
    case = layered(tmp_path)
    assert layer_report.main([str(case), "--patch", "wall", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert set(payload) >= {"case", "reference", "patches", "problems"}
    assert payload["problems"] == []
    (patch,) = payload["patches"]
    assert set(patch) >= {
        "patch",
        "type",
        "faces",
        "coverage_pct",
        "layered_faces",
        "basis",
        "first_cell_m",
        "reference_spacing_m",
        "thinning",
        "note",
    }
    assert set(patch["first_cell_m"]) == {"min", "mean", "max", "median"}
    assert patch["type"] == "wall"


def test_an_unreadable_mesh_is_reported_rather_than_raised(layer_report, tmp_path, capsys):
    """A binary mesh and a case that is not one are both answers about the run.
    Exiting non-zero on them would make "no layers here" and "I could not look"
    the same event to whatever is reading the exit code."""
    assert layer_report.main([str(tmp_path / "nothing"), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["patches"] == []
    assert payload["problems"] and "polyMesh" in payload["problems"][0]

    case = layered(tmp_path)
    binary = case / "constant" / "polyMesh" / "points"
    binary.write_text(binary.read_text().replace("format      ascii", "format      binary"))
    assert layer_report.main([str(case), "--patch", "wall"]) == 0
    text = capsys.readouterr().out
    assert "binary" in text and "not measured" in text


def test_a_patch_that_is_not_there_is_a_problem_and_not_a_traceback(
    layer_report, tmp_path
):
    found = layer_report.measure(layered(tmp_path), ["hull"])
    assert found["patches"] == []
    assert "no patch 'hull'" in found["problems"][0]
    assert "outer" in found["problems"][0], "the patches it does have are named"


def test_the_docstring_keeps_the_script_advisory(layer_report):
    """The free-will contract reaches the toolbox too: this measures, and stops."""
    text = " ".join((layer_report.__doc__ or "").split())
    assert "writes nothing, changes nothing" in text
    assert "the reading can also be wrong" in text
    assert "python3 layer_report.py" in text
    for imperative in ("you must", "always ", "make sure", "you should"):
        assert imperative not in text.lower()


def test_the_report_offers_the_number_without_a_verdict_on_it(layer_report, tmp_path):
    """Whether 65% is good depends on what the layer is for, which is not something
    a measurement knows."""
    text = layer_report.report(layer_report.measure(layered(tmp_path), ["wall"]))
    assert "nothing here decides whether this mesh is good enough" in text
    for verdict in ("too few", "unacceptable", "you must", "fix "):
        assert verdict not in text.lower()


def test_it_needs_nothing_the_image_does_not_have(layer_report):
    """numpy is on the image; pyvista would be, but a coverage number should not
    need a render stack to be produced on a login node."""
    source = (TOOLBOX / "layer_report.py").read_text(encoding="utf-8")
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= set(sys.stdlib_module_names) | {"__future__", "numpy"}
    source.encode("ascii")  # raises if a non-ASCII character crept in
