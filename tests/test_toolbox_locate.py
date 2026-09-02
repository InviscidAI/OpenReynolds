"""`locate.py`: where a diverging run went wrong, measured rather than argued.

The script exists because of the Wigley free-surface case, where eight hypotheses
about a divergence were proposed and all eight falsified over five rounds and two
days, and what settled it in one step was a localisation: the extremum in the first
cell row at the inlet, p_rgh moving before omega and before U, and the value
mesh-invariant across two domains -- which is the boundary specification's signature,
not the grid's. So what is tested here is the three measurements: the extremum is
found and placed in the mesh, the log yields the first-degrading field with a time on
it, and `--compare` says whether a planted value survives a change of mesh.

The rest is the same contract the ladder tests keep: the exit code is 0 whatever the
report says, the docstring carries preflight's advisory formula, and nothing pulls in
a rendering stack.
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
def locate():
    spec = importlib.util.spec_from_file_location("toolbox_locate", TOOLBOX / "locate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# -- a synthetic mesh whose geometry is known by construction ----------------------
#
# A structured nx x ny x nz box of hexes with spacing (dx, dy, dz), written as an
# ascii polyMesh: cell (i, j, k) has its centre at ((i+0.5)dx, (j+0.5)dy, (k+0.5)dz)
# by arithmetic, so every location the script reports can be checked against a number
# that never came from the script. Patches: `inlet` at x=0, `outlet` at x=max, and
# everything else `walls`.


def _foam(cls: str, obj: str, body: str) -> str:
    return (
        "FoamFile { version 2.0; format ascii; "
        f"class {cls}; object {obj}; }}\n{body}\n"
    )


def write_box_mesh(case: Path, nx: int, ny: int, nz: int, spacing=(1.0, 1.0, 1.0)) -> None:
    dx, dy, dz = spacing
    poly = case / "constant" / "polyMesh"
    poly.mkdir(parents=True, exist_ok=True)

    def pid(i, j, k):
        return i + j * (nx + 1) + k * (nx + 1) * (ny + 1)

    def cid(i, j, k):
        return i + j * nx + k * nx * ny

    points = [
        (i * dx, j * dy, k * dz)
        for k in range(nz + 1) for j in range(ny + 1) for i in range(nx + 1)
    ]

    faces: list[list[int]] = []
    owner: list[int] = []
    neighbour: list[int] = []

    # Internal faces, each wound so its normal points from owner to neighbour.
    for k in range(nz):
        for j in range(ny):
            for i in range(1, nx):  # plane x = i, normal +x
                faces.append([pid(i, j, k), pid(i, j + 1, k),
                              pid(i, j + 1, k + 1), pid(i, j, k + 1)])
                owner.append(cid(i - 1, j, k))
                neighbour.append(cid(i, j, k))
    for k in range(nz):
        for j in range(1, ny):  # plane y = j, normal +y
            for i in range(nx):
                faces.append([pid(i, j, k), pid(i, j, k + 1),
                              pid(i + 1, j, k + 1), pid(i + 1, j, k)])
                owner.append(cid(i, j - 1, k))
                neighbour.append(cid(i, j, k))
    for k in range(1, nz):  # plane z = k, normal +z
        for j in range(ny):
            for i in range(nx):
                faces.append([pid(i, j, k), pid(i + 1, j, k),
                              pid(i + 1, j + 1, k), pid(i, j + 1, k)])
                owner.append(cid(i, j, k - 1))
                neighbour.append(cid(i, j, k))

    patches: list[tuple[str, int, int]] = []

    def patch(name: str, boundary_faces) -> None:
        start = len(faces)
        for verts, own in boundary_faces:
            faces.append(verts)
            owner.append(own)
        patches.append((name, start, len(faces) - start))

    patch("inlet", [  # x = 0, outward normal -x
        ([pid(0, j, k), pid(0, j, k + 1), pid(0, j + 1, k + 1), pid(0, j + 1, k)],
         cid(0, j, k))
        for k in range(nz) for j in range(ny)
    ])
    patch("outlet", [  # x = nx dx, outward normal +x
        ([pid(nx, j, k), pid(nx, j + 1, k), pid(nx, j + 1, k + 1), pid(nx, j, k + 1)],
         cid(nx - 1, j, k))
        for k in range(nz) for j in range(ny)
    ])
    walls = []
    for k in range(nz):
        for i in range(nx):
            walls.append((  # y = 0, outward -y
                [pid(i, 0, k), pid(i + 1, 0, k), pid(i + 1, 0, k + 1), pid(i, 0, k + 1)],
                cid(i, 0, k)))
            walls.append((  # y = ny dy, outward +y
                [pid(i, ny, k), pid(i, ny, k + 1), pid(i + 1, ny, k + 1), pid(i + 1, ny, k)],
                cid(i, ny - 1, k)))
    for j in range(ny):
        for i in range(nx):
            walls.append((  # z = 0, outward -z
                [pid(i, j, 0), pid(i, j + 1, 0), pid(i + 1, j + 1, 0), pid(i + 1, j, 0)],
                cid(i, j, 0)))
            walls.append((  # z = nz dz, outward +z
                [pid(i, j, nz), pid(i + 1, j, nz), pid(i + 1, j + 1, nz), pid(i, j + 1, nz)],
                cid(i, j, nz - 1)))
    patch("walls", walls)

    (poly / "points").write_text(_foam(
        "vectorField", "points",
        f"{len(points)}\n(\n" + "\n".join(f"({x:g} {y:g} {z:g})" for x, y, z in points) + "\n)",
    ))
    (poly / "faces").write_text(_foam(
        "faceList", "faces",
        f"{len(faces)}\n(\n" + "\n".join(
            f"4({' '.join(str(v) for v in verts)})" for verts in faces
        ) + "\n)",
    ))
    (poly / "owner").write_text(_foam(
        "labelList", "owner",
        f"{len(owner)}\n(\n" + "\n".join(str(v) for v in owner) + "\n)",
    ))
    (poly / "neighbour").write_text(_foam(
        "labelList", "neighbour",
        f"{len(neighbour)}\n(\n" + "\n".join(str(v) for v in neighbour) + "\n)",
    ))
    boundary = "".join(
        f"    {name} {{ type patch; nFaces {count}; startFace {start}; }}\n"
        for name, start, count in patches
    )
    (poly / "boundary").write_text(_foam(
        "polyBoundaryMesh", "boundary", f"{len(patches)}\n(\n{boundary})",
    ))


def write_scalar_field(case: Path, time: str, name: str, values) -> None:
    directory = case / time
    directory.mkdir(parents=True, exist_ok=True)
    listed = "\n".join(f"{value:.10g}" for value in values)
    (directory / name).write_text(_foam(
        "volScalarField", name,
        "dimensions [0 2 -2 0 0 0 0];\n"
        f"internalField nonuniform List<scalar> {len(values)}\n(\n{listed}\n);\n"
        "boundaryField { inlet { type fixedValue; value uniform 0; } }\n",
    ))


def write_vector_field(case: Path, time: str, name: str, vectors) -> None:
    directory = case / time
    directory.mkdir(parents=True, exist_ok=True)
    listed = "\n".join(f"({x:g} {y:g} {z:g})" for x, y, z in vectors)
    (directory / name).write_text(_foam(
        "volVectorField", name,
        "dimensions [0 1 -1 0 0 0 0];\n"
        f"internalField nonuniform List<vector> {len(vectors)}\n(\n{listed}\n);\n"
        "boundaryField { inlet { type fixedValue; value uniform (0 0 0); } }\n",
    ))


PLANTED = -4723.0
"""The Wigley number: the inlet p_rgh that was the same on two meshes and thereby
named the boundary specification. Planted at cell (0, 1, 1) -- first cell row at the
inlet, touching no other patch."""


def crashed_case(root: Path, name: str = "tank", nx: int = 4, spacing=(1.0, 1.0, 1.0),
                 planted: float = PLANTED) -> Path:
    """4x3x3 (by default) cells, p_rgh planted at cell (0,1,1), |U| max interior,
    alpha uniform."""
    case = root / name
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "system" / "controlDict").write_text("application     interFoam;\n")
    write_box_mesh(case, nx, 3, 3, spacing)
    cells = nx * 9
    p_rgh = np.zeros(cells)
    p_rgh[0 + 1 * nx + 1 * nx * 3] = planted  # cell (0, 1, 1)
    write_scalar_field(case, "0.5", "p_rgh", p_rgh)
    vectors = [(1.0, 0.0, 0.0)] * cells
    vectors[2 + 1 * nx + 1 * nx * 3] = (9.0, 0.0, 0.0)  # cell (2, 1, 1), interior
    write_vector_field(case, "0.5", "U", vectors)
    (case / "0.5" / "alpha.water").write_text(_foam(
        "volScalarField", "alpha.water",
        "dimensions [0 0 0 0 0 0 0];\ninternalField uniform 1;\n"
        "boundaryField { inlet { type fixedValue; value uniform 1; } }\n",
    ))
    return case


DYING_LOG = """\
Time = 0.1
Courant Number mean: 0.05 max: 0.2
Solving for p_rgh, Initial residual = 1e-06, Final residual = 1e-09, No Iterations 3
Solving for omega, Initial residual = 2e-06, Final residual = 1e-08, No Iterations 2
Time = 0.2
Courant Number mean: 0.05 max: 0.2
Solving for p_rgh, Initial residual = 1e-06, Final residual = 1e-09, No Iterations 3
Solving for omega, Initial residual = 2e-06, Final residual = 1e-08, No Iterations 2
Time = 0.3
Courant Number mean: 0.05 max: 0.3
Solving for p_rgh, Initial residual = 1e-06, Final residual = 1e-09, No Iterations 3
Solving for omega, Initial residual = 2e-06, Final residual = 1e-08, No Iterations 2
Time = 0.4
Courant Number mean: 0.06 max: 0.3
Solving for p_rgh, Initial residual = 1e-06, Final residual = 1e-09, No Iterations 3
Solving for omega, Initial residual = 2e-06, Final residual = 1e-08, No Iterations 2
Time = 0.5
Courant Number mean: 0.06 max: 0.4
Solving for p_rgh, Initial residual = 1e-06, Final residual = 1e-09, No Iterations 3
Solving for omega, Initial residual = 2e-06, Final residual = 1e-08, No Iterations 2
bounding omega, min: -1.2e+02 max: 3.4e+05 average: 4.5
Time = 0.6
Courant Number mean: 0.07 max: 0.6
Solving for p_rgh, Initial residual = 1e-06, Final residual = 1e-09, No Iterations 3
Solving for omega, Initial residual = 3e-06, Final residual = 1e-08, No Iterations 2
bounding omega, min: -5.6e+03 max: 8.1e+06 average: 22
Time = 0.7
Courant Number mean: 0.2 max: 2.1
Solving for p_rgh, Initial residual = 1e-03, Final residual = 1e-06, No Iterations 8
Solving for omega, Initial residual = 4e-06, Final residual = 1e-08, No Iterations 2
bounding omega, min: -9.9e+04 max: 6.3e+08 average: 410
Time = 0.8
Courant Number mean: 1.4 max: 12.6
Solving for p_rgh, Initial residual = 1e-01, Final residual = 1e-03, No Iterations 20
Solving for omega, Initial residual = 5e-06, Final residual = 1e-08, No Iterations 2
Time = 0.9
Courant Number mean: 6.2 max: 44.3
Solving for p_rgh, Initial residual = 1, Final residual = 0.5, No Iterations 50
Solving for omega, Initial residual = 6e-06, Final residual = 1e-08, No Iterations 2
"""


def by_check(findings, check):
    return [finding for finding in findings if finding.check == check]


# -- the last-write scan -----------------------------------------------------------


def test_the_extremum_is_found_and_its_location_is_correct(locate, tmp_path):
    """Cell (0,1,1) of a unit-spaced 4x3x3 box has its centre at (0.5, 1.5, 1.5) by
    arithmetic. The number the script reports has to be that one."""
    case = crashed_case(tmp_path)
    scanned = locate.scan(case)

    assert scanned["time"] == "0.5"
    record = scanned["fields"]["p_rgh"]
    assert record["min"] == pytest.approx(PLANTED)
    assert record["extremum"]["value"] == pytest.approx(PLANTED)
    assert record["extremum"]["cell"] == 16
    assert record["extremum"]["centre"] == pytest.approx([0.5, 1.5, 1.5])
    assert record["extremum"]["normalised"] == pytest.approx([0.125, 0.5, 0.5])
    assert record["extremum"]["distance_to_bbox"] == pytest.approx(0.5)


def test_a_boundary_adjacent_extremum_names_its_patch(locate, tmp_path):
    """Cell (0,1,1) touches the inlet and nothing else; the attribution has to say
    `inlet` and not the walls it never reaches."""
    case = crashed_case(tmp_path)
    scanned = locate.scan(case)
    assert scanned["fields"]["p_rgh"]["extremum"]["patches"] == ["inlet"]

    findings, _ = locate.run(case)
    (finding,) = [f for f in by_check(findings, "last-write") if "p_rgh" in f.measured]
    assert finding.status == "warn"
    assert "inlet" in finding.measured
    assert "inlet" in finding.meaning


def test_an_interior_extremum_is_placed_without_a_patch(locate, tmp_path):
    """|U| peaks at cell (2,1,1), centre (2.5, 1.5, 1.5), which touches no patch."""
    case = crashed_case(tmp_path)
    record = locate.scan(case)["fields"]["U"]
    assert record["max"] == pytest.approx(9.0)
    assert record["extremum"]["cell"] == 18
    assert record["extremum"]["centre"] == pytest.approx([2.5, 1.5, 1.5])
    assert record["extremum"]["patches"] == []


def test_a_uniform_field_is_reported_not_localised(locate, tmp_path):
    case = crashed_case(tmp_path)
    record = locate.scan(case)["fields"]["alpha.water"]
    assert record["uniform"] == pytest.approx(1.0)
    assert "extremum" not in record


def test_a_binary_points_list_reads_the_same_as_ascii(locate, tmp_path):
    """Production cases crash in `writeFormat binary`; the reader has to take both.
    The same three points, written raw, come back identical."""
    coordinates = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-4.4375, 0.5, 0.25]])
    path = tmp_path / "points"
    header = (
        'FoamFile { version 2.0; format binary; arch "LSB;label=32;scalar=64"; '
        "class vectorField; object points; }\n3\n("
    )
    path.write_bytes(header.encode() + coordinates.tobytes() + b")\n")
    fobj = locate.foam_file(path)
    flat, _ = locate.read_list(fobj, fobj.body_at, "scalar", per_item=3)
    assert flat.reshape(-1, 3) == pytest.approx(coordinates)


# -- field ordering from the log ---------------------------------------------------


def test_the_first_degrading_field_is_found_in_a_bounding_cascade(locate, tmp_path):
    """omega starts bounding at t=0.5 while every residual still looks healthy; the
    p_rgh residual does not blow until 0.7. The ordering is omega first, with the
    time attached -- which is the measurement that beats a mechanism."""
    log = tmp_path / "log.interFoam"
    log.write_text(DYING_LOG)
    parsed = locate.parse_log(log)
    first = locate.first_degrading(parsed)

    assert first["field"] == "omega"
    assert first["time"] == pytest.approx(0.5)
    assert "bounding" in first["how"]
    assert first["onsets"]["p_rgh"]["time"] == pytest.approx(0.7)


def test_the_ordering_finding_reports_the_field_the_time_and_the_courant_tail(locate, tmp_path):
    case = crashed_case(tmp_path)
    (case / "log.interFoam").write_text(DYING_LOG)
    findings, data = locate.run(case)
    (finding,) = by_check(findings, "ordering")

    assert finding.status == "warn"
    assert "omega degrades first" in finding.measured
    assert "t=0.5" in finding.measured
    assert "44.3" in finding.measured, "the Courant tail belongs in the same line"
    assert data["log"]["first_degrading"]["field"] == "omega"


# -- mesh-invariance ---------------------------------------------------------------


def test_compare_declares_invariance_for_a_shared_planted_value(locate, tmp_path):
    """The same -4723 in the first cell row at the inlet of two meshes with
    different domains and different cell sizes: the report has to call the value
    invariant and say what that means -- set by the boundary specification."""
    here = crashed_case(tmp_path, "one_l", nx=4, spacing=(1.0, 1.0, 1.0))
    there = crashed_case(tmp_path, "three_l", nx=6, spacing=(0.75, 1.0, 1.0))
    findings, data = locate.run(here, compare=str(there))

    rows = {row["field"]: row for row in data["compare"]["fields"]}
    assert rows["p_rgh"]["value_invariant"] is True
    assert rows["p_rgh"]["location_agrees"] is True

    (finding,) = [f for f in by_check(findings, "invariance") if "p_rgh" in f.measured]
    assert finding.status == "warn"
    assert "boundary specification" in finding.meaning
    assert "not the grid" in finding.meaning


def test_compare_denies_invariance_when_the_value_moves(locate, tmp_path):
    here = crashed_case(tmp_path, "one_l", nx=4)
    there = crashed_case(tmp_path, "other", nx=6, spacing=(0.75, 1.0, 1.0), planted=-9000.0)
    findings, data = locate.run(here, compare=str(there))

    rows = {row["field"]: row for row in data["compare"]["fields"]}
    assert rows["p_rgh"]["value_invariant"] is False
    (finding,) = [f for f in by_check(findings, "invariance") if "p_rgh" in f.measured]
    assert finding.status == "ok"
    assert "moved when the mesh changed" in finding.meaning


# -- the report and the contract ---------------------------------------------------


def test_json_parses_and_has_the_documented_shape(locate, tmp_path, capsys):
    case = crashed_case(tmp_path)
    (case / "log.interFoam").write_text(DYING_LOG)
    assert locate.main([str(case), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["case"] == str(case)
    assert set(payload) == {"case", "worst", "counts", "findings", "data"}
    for finding in payload["findings"]:
        assert set(finding) == {"check", "status", "measured", "means", "repair"}
    assert payload["data"]["scan"]["time"] == "0.5"
    assert payload["data"]["log"]["first_degrading"]["field"] == "omega"


def test_a_missing_case_is_reported_and_the_exit_code_is_still_zero(locate, tmp_path, capsys):
    """There is no verdict here that may stop anyone, so there is no code for one --
    not even for a case that is not there."""
    assert locate.main([str(tmp_path / "does-not-exist")]) == 0
    out = capsys.readouterr().out
    assert "skipped" in out.lower()
    assert locate.main([str(crashed_case(tmp_path))]) == 0
    capsys.readouterr()


def test_the_docstring_keeps_it_advisory_and_prefers_location_to_explanation(locate):
    doc = locate.__doc__
    assert "edits nothing, refuses nothing, and blocks nothing" in doc
    assert "exit code is 0 whatever" in doc
    assert "python3" in doc, "it has to show how to run it"
    assert "location over explanation" in doc, (
        "the point of the tool is in its own docstring: where, which, whether -- not why"
    )


def test_it_is_stdlib_numpy_and_its_own_siblings(locate):
    """No pyvista: a locator that pulled in a rendering stack could not run in the
    seconds after a crash that make it worth running."""
    source = (TOOLBOX / "locate.py").read_text(encoding="utf-8")
    assert "pyvista" not in source
    siblings = {script.stem for script in TOOLBOX.glob("*.py")}
    allowed = set(sys.stdlib_module_names) | siblings | {"__future__", "numpy"}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [(node.module or "").split(".")[0]]
        else:
            continue
        for root in roots:
            assert root in allowed, f"locate.py imports {root}"


def test_the_toolbox_index_and_the_field_note_carry_it(locate):
    index = (TOOLBOX / "README.md").read_text(encoding="utf-8")
    assert "locate.py" in index
    notes = (TOOLBOX / "notes" / "openfoam-field-notes.md").read_text(encoding="utf-8")
    assert "locate.py" in notes
    assert "eight" in notes.lower(), "the Wigley incident is why the note exists"
