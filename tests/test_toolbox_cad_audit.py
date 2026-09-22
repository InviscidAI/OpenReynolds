"""cad_audit.py -- the exported patch set measured, held to the numbers it names.

Almost every way this script can be wrong is a way it can be wrong *quietly*. A
closure check run file by file passes nothing real and looks like a pass. A
self-intersection sweep with a broken spatial index reports a clean surface. An
expensive check that gives up above a size limit and says `ok` is indistinguishable
from one that looked. So the tests here are built around known answers -- a defect
injected at a place where the right number can be counted by hand -- and around
agreement with a brute-force O(n^2) sweep that is slow enough to be obviously right.

Three of them carry the weight:

*Union versus parts.* A correct six-patch cube whose every individual file has open
edges must come back `closure: ok`. An implementation that checks the files one at a
time fails here and nowhere else.

*Scale.* The small self-intersection cases never touch the spatial index; the
two-hundred-thousand-triangle one with a crossing planted at a known place is the only
test a broken index shows up in.

*Degradation.* Above `preflight.TOPOLOGY_TRIANGLE_LIMIT` the expensive findings are
asserted `skipped` and explicitly asserted **not** `ok`, because silence that reads as
a pass is the failure mode the limit creates.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import subprocess
import sys
import time
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


@pytest.fixture(scope="module")
def cad_audit():
    return load("cad_audit")


@pytest.fixture(scope="module")
def preflight():
    return load("preflight")


@pytest.fixture(scope="module")
def surfaces():
    return load("surfaces")


# -- surfaces to measure on --------------------------------------------------------


def _orient_outward(tris: np.ndarray, centre) -> np.ndarray:
    """Wind every triangle so its normal points away from `centre`.

    Written this way rather than by getting six sign conventions right by hand: the
    fixture's whole job is to be a surface whose correct answers are known, and a cube
    with one face wound the wrong way by accident would make `normals` fail on the
    control case.
    """
    tris = np.array(tris, dtype=float)
    normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    outward = tris.mean(axis=1) - np.asarray(centre, dtype=float)
    wrong = np.einsum("ij,ij->i", normal, outward) < 0
    tris[wrong] = tris[wrong][:, [0, 2, 1]]
    return tris


def cube_faces(n: int = 2, size: float = 1.0, origin=(0.0, 0.0, 0.0)) -> dict[str, np.ndarray]:
    """A closed axis-aligned cube as six named faces, each an n x n grid of quads.

    Six *open* surfaces whose union is closed, which is the shape of every correct
    export and the shape this script exists to tell apart from a closed single file.
    """
    origin = np.asarray(origin, dtype=float)
    centre = origin + size / 2.0
    grid = np.linspace(0.0, size, n + 1)
    u, v = np.meshgrid(grid, grid, indexing="ij")
    names = ("xMin", "xMax", "yMin", "yMax", "zMin", "zMax")
    faces: dict[str, np.ndarray] = {}
    for axis in range(3):
        for which, side in enumerate((0.0, size)):
            others = [k for k in range(3) if k != axis]
            points = np.zeros((n + 1, n + 1, 3))
            points[..., others[0]] = u
            points[..., others[1]] = v
            points[..., axis] = side
            points = points + origin
            p00 = points[:-1, :-1]
            p01 = points[:-1, 1:]
            p11 = points[1:, 1:]
            p10 = points[1:, :-1]
            lower = np.stack([p00, p01, p11], axis=-2).reshape(-1, 3, 3)
            upper = np.stack([p00, p11, p10], axis=-2).reshape(-1, 3, 3)
            tris = np.concatenate([lower, upper], axis=0)
            faces[names[axis * 2 + which]] = _orient_outward(tris, centre)
    return faces


def octahedron(radius: float = 1.0, refine: int = 0, centre=(0.0, 0.0, 0.0)) -> np.ndarray:
    """A closed sphere-ish body, 8 * 4**refine triangles, normals outward."""
    poles = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
                     dtype=float)
    tris = []
    for x in (0, 1):
        for y in (2, 3):
            for z in (4, 5):
                tris.append([poles[x], poles[y], poles[z]])
    tris = np.array(tris)
    for _ in range(refine):
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        tris = np.concatenate([
            np.stack([a, ab, ca], axis=1), np.stack([ab, b, bc], axis=1),
            np.stack([ca, bc, c], axis=1), np.stack([ab, bc, ca], axis=1),
        ])
    flat = tris.reshape(-1, 3)
    flat = flat / np.linalg.norm(flat, axis=1)[:, None] * radius
    tris = flat.reshape(-1, 3, 3) + np.asarray(centre, dtype=float)
    return _orient_outward(tris, centre)


def rotation(rng) -> np.ndarray:
    """A uniformly random rotation matrix, via QR of a Gaussian."""
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def rotate(tris: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return (tris.reshape(-1, 3) @ matrix.T).reshape(-1, 3, 3)


# -- a patch set on disk -----------------------------------------------------------


def write_stl(path: Path, tris: np.ndarray) -> None:
    """A binary STL, which is what `preflight.read_triangles` reads fastest."""
    count = len(tris)
    normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    length = np.linalg.norm(normal, axis=1)
    normal = np.where(length[:, None] > 0, normal / np.where(length[:, None] > 0, length[:, None], 1.0), 0.0)
    block = np.zeros((count, 4, 3), dtype="<f4")
    block[:, 0] = normal
    block[:, 1:] = tris
    record = np.zeros((count, 50), dtype=np.uint8)
    record[:, :48] = block.view(np.uint8).reshape(count, 48)
    path.write_bytes(b"\0" * 80 + np.array([count], dtype="<u4").tobytes() + record.tobytes())


def write_patch_set(directory: Path, patches: dict[str, np.ndarray], **manifest) -> Path:
    """An I3 patch set: one STL per patch and the manifest that names them."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, tris in patches.items():
        write_stl(directory / f"{name}.stl", np.asarray(tris, dtype=float))
        entries.append({"name": name, "file": f"{name}.stl",
                        "triangles": int(len(tris)), "role": "wall"})
    body = {"unit_metres": 1.0, "source": "fixture.step", "patches": entries,
            "location_in_mesh": [0.5, 0.5, 0.5]}
    body.update(manifest)
    (directory / "patches.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    return directory


def status_of(report: dict, check: str) -> str:
    return next(row["status"] for row in report["findings"] if row["check"] == check)


def measured_of(report: dict, check: str) -> str:
    return next(row["measured"] for row in report["findings"] if row["check"] == check)


def means_of(report: dict, check: str) -> str:
    return next(row["means"] for row in report["findings"] if row["check"] == check)


# -- a brute-force sweep, vectorised so it can afford to be exhaustive -------------


def all_pairs_crossings(triangles: np.ndarray, corners: np.ndarray) -> set[int]:
    """Every non-neighbouring pair tested, with no index anywhere near it.

    The same twenty-three separating axes `surfaces.tri_tri_intersect` uses, applied to
    every pair at once instead of one pair per Python call -- which is what makes an
    O(n^2) sweep affordable on six hundred triangles twenty times over. That it agrees
    with `surfaces.tri_tri_intersect` pair by pair is asserted below on a sample, so
    this is a second opinion rather than an unchecked one.
    """
    count = len(triangles)
    first, second = np.triu_indices(count, k=1)
    shares = np.zeros(len(first), dtype=bool)
    for one in range(3):
        for other in range(3):
            shares |= corners[first][:, one] == corners[second][:, other]
    first, second = first[~shares], second[~shares]

    hit: list[np.ndarray] = []
    for start in range(0, len(first), 200_000):
        i = first[start:start + 200_000]
        j = second[start:start + 200_000]
        meets = _all_pairs_sat(triangles[i], triangles[j])
        hit.append(np.stack([i[meets], j[meets]], axis=1))
    pairs = np.concatenate(hit) if hit else np.zeros((0, 2), dtype=np.int64)
    return {int(v) for v in pairs.reshape(-1)}


def _all_pairs_sat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a):
        return np.zeros(0, dtype=bool)
    ea = a[:, [1, 2, 0]] - a
    eb = b[:, [1, 2, 0]] - b
    na = np.cross(ea[:, 0], -ea[:, 2])
    nb = np.cross(eb[:, 0], -eb[:, 2])
    axes = [na, nb]
    for p in range(3):
        for q in range(3):
            axes.append(np.cross(ea[:, p], eb[:, q]))
    for edge in [ea[:, k] for k in range(3)] + [eb[:, k] for k in range(3)]:
        axes.append(np.cross(edge, na))
        axes.append(np.cross(edge, nb))
    reach = np.maximum(np.abs(a).reshape(len(a), -1).max(axis=1),
                       np.abs(b).reshape(len(b), -1).max(axis=1))
    floor = (np.maximum(reach, 1e-300) * 1e-12) ** 2
    meets = np.ones(len(a), dtype=bool)
    for axis in axes:
        length2 = np.einsum("ij,ij->i", axis, axis)
        usable = length2 > floor
        pa = np.einsum("ijk,ik->ij", a, axis)
        pb = np.einsum("ijk,ik->ij", b, axis)
        apart = usable & ((pa.max(axis=1) < pb.min(axis=1)) | (pb.max(axis=1) < pa.min(axis=1)))
        meets &= ~apart
    return meets


def test_the_brute_force_sweep_agrees_with_the_shared_primitive(surfaces):
    """The oracle the next test leans on, checked against the one C2a owns.

    `all_pairs_crossings` is vectorised so it can be exhaustive; if it disagreed with
    `surfaces.tri_tri_intersect` then every comparison built on it would be a
    comparison between two wrong answers.
    """
    rng = np.random.default_rng(4)
    tris = rng.uniform(-1, 1, size=(120, 3, 3))
    first, second = np.triu_indices(120, k=1)
    pick = rng.choice(len(first), size=400, replace=False)
    mine = _all_pairs_sat(tris[first[pick]], tris[second[pick]])
    theirs = np.array([
        surfaces.tri_tri_intersect(tris[i], tris[j])
        for i, j in zip(first[pick], second[pick])
    ])
    assert (mine == theirs).all()


# -- self-intersection -------------------------------------------------------------


def interpenetrating(rng) -> np.ndarray:
    """Two closed primitives left overlapping, at a random rotation and offset."""
    kinds = ("octahedron", "cube", "tetra")
    bodies = []
    for _ in range(2):
        kind = kinds[rng.integers(len(kinds))]
        if kind == "octahedron":
            body = octahedron(radius=float(rng.uniform(0.6, 1.2)), refine=2)
        elif kind == "cube":
            body = np.concatenate(list(cube_faces(n=3, size=float(rng.uniform(1.0, 2.0)),
                                                  origin=(-0.5, -0.5, -0.5)).values()))
        else:
            body = octahedron(radius=float(rng.uniform(0.8, 1.4)), refine=1)
        body = rotate(body, rotation(rng))
        bodies.append(body + rng.uniform(-0.7, 0.7, size=3))
    return np.concatenate(bodies)


def test_self_intersection_equals_a_brute_force_sweep_on_randomised_cases(
    cad_audit, surfaces
):
    """Twenty randomised interpenetrations, and the reported set is the true set.

    Not a count and not a superset: the same triangles. A sweep that misses a crossing
    reports a clean surface, and a sweep that invents one sends somebody to repair
    geometry that was already right.
    """
    rng = np.random.default_rng(2026)
    for case in range(20):
        tris = interpenetrating(rng)
        assert len(tris) < 2000
        corners, _vertices = surfaces.weld(tris)
        found = cad_audit.crossing_pairs(tris, corners)
        reported = {int(t) for t in found["triangles"]}
        truth = all_pairs_crossings(tris, corners)
        assert reported == truth, f"case {case}: {len(reported)} reported, {len(truth)} true"
        assert truth, f"case {case} produced no crossing at all, so it tests nothing"


def test_a_crossing_planted_in_a_large_surface_is_found(cad_audit, tmp_path_factory):
    """The one place a broken spatial index shows up.

    The small cases above never leave the leaves of the tree. Here a single pair of
    triangles is driven through the wall of a two-hundred-thousand-triangle cube at a
    place the test knows, and both of them have to come back.
    """
    faces = cube_faces(n=130, size=1.0)
    total = sum(len(tris) for tris in faces.values())
    assert 200_000 <= total <= 210_000, total

    # A fin through the zMin wall at (0.33, 0.44), sharing no vertex with anything.
    fin = np.array([
        [[0.31, 0.42, -0.05], [0.35, 0.42, 0.05], [0.33, 0.46, 0.0]],
        [[0.31, 0.46, -0.05], [0.35, 0.46, 0.05], [0.33, 0.42, 0.0]],
    ])
    patches = dict(faces)
    patches["fin"] = fin
    directory = write_patch_set(tmp_path_factory.mktemp("planted") / "triSurface", patches)

    started = time.time()
    report = cad_audit.audit(directory)
    elapsed = time.time() - started

    assert status_of(report, "self_intersection") == "fail", measured_of(report, "self_intersection")
    ids = set(report["measured"]["self_intersection"]["triangle_ids"])
    planted = {total, total + 1}  # the fin is concatenated last, in manifest order
    assert planted <= ids, f"the planted crossing was not reported: {sorted(ids)[:10]}"
    assert report["measured"]["self_intersection"]["pairs"] >= 2
    assert elapsed < 120, elapsed


def test_only_the_shared_primitive_ever_returns_a_crossing(cad_audit, surfaces, monkeypatch):
    """The cheap all-pairs filter may drop pairs; it may never decide one.

    `cad_audit` narrows half a million box-overlapping pairs with eight of the
    twenty-three separating axes before it asks. That is allowed to be conservative and
    is not allowed to be the answer, so with `surfaces.tri_tri_intersect` stubbed to
    refuse everything the script must report nothing at all -- which it cannot do if it
    is deciding anywhere else.
    """
    tris = interpenetrating(np.random.default_rng(7))
    corners, _vertices = surfaces.weld(tris)
    assert cad_audit.crossing_pairs(tris, corners)["pairs"]
    monkeypatch.setattr(cad_audit.surfaces, "tri_tri_intersect", lambda a, b: False)
    assert cad_audit.crossing_pairs(tris, corners)["pairs"] == []


def test_the_cheap_filter_never_drops_a_pair_that_meets(cad_audit):
    """Superset, the property the narrowing rests on, asserted directly."""
    rng = np.random.default_rng(11)
    a = rng.uniform(-1, 1, size=(4000, 3, 3))
    b = rng.uniform(-1, 1, size=(4000, 3, 3))
    keep = cad_audit._separating_axes(a, b)
    truth = _all_pairs_sat(a, b)
    assert truth.any()
    assert (truth & ~keep).sum() == 0, "a pair that meets was filtered out"


# -- known-answer defect injection -------------------------------------------------


def test_a_deleted_quad_fails_closure_naming_four_open_edges(cad_audit, tmp_path):
    faces = cube_faces(n=2)
    faces["zMax"] = np.delete(faces["zMax"], [0, 4], axis=0)  # one quad, both triangles
    report = cad_audit.audit(write_patch_set(tmp_path / "triSurface", faces))
    assert status_of(report, "closure") == "fail"
    assert "4 open edges" in measured_of(report, "closure")
    assert report["ok"] is False


def test_a_flipped_triangle_fails_normals_naming_three_same_direction_edges(
    cad_audit, tmp_path
):
    faces = cube_faces(n=2)
    flipped = faces["yMin"].copy()
    flipped[3] = flipped[3][[0, 2, 1]]
    faces["yMin"] = flipped
    report = cad_audit.audit(write_patch_set(tmp_path / "triSurface", faces))
    assert status_of(report, "normals") == "fail"
    assert "3 same-direction edges" in measured_of(report, "normals")
    assert status_of(report, "closure") == "ok", "flipping a triangle does not open the surface"


def test_a_face_in_two_patches_fails_coverage_as_double_assigned(cad_audit, tmp_path):
    faces = cube_faces(n=2)
    quad = faces["xMax"][[1, 5]]
    faces["yMax"] = np.concatenate([faces["yMax"], quad])
    report = cad_audit.audit(write_patch_set(tmp_path / "triSurface", faces))
    measured = measured_of(report, "coverage")
    assert status_of(report, "coverage") == "fail"
    assert "double-assigned" in measured
    assert "xMax" in measured and "yMax" in measured
    centre = quad[0].mean(axis=0)
    assert f"{centre[0]:.4g}" in measured


def test_a_face_in_no_patch_fails_coverage_as_unassigned_naming_the_count(
    cad_audit, tmp_path
):
    """One quad dropped from every patch it could have been in.

    Its absence is not an open-edge count -- four open edges bound one missing quad and
    four hundred bound one missing cylinder -- so the finding has to say how many faces
    are gone, which it reads off the Euler characteristic rather than by walking edges.
    """
    faces = cube_faces(n=2)
    faces["xMin"] = np.delete(faces["xMin"], [2, 6], axis=0)
    report = cad_audit.audit(write_patch_set(tmp_path / "triSurface", faces))
    measured = measured_of(report, "coverage")
    assert status_of(report, "coverage") == "fail"
    assert "unassigned" in measured
    assert "1 face of the exported surface is in no patch file" in measured
    assert "4 open edges" in measured


def test_two_missing_quads_are_counted_as_two(cad_audit, tmp_path):
    """The count is a count, not the word `1` hard-wired into a sentence."""
    faces = cube_faces(n=2)
    faces["xMin"] = np.delete(faces["xMin"], [2, 6], axis=0)
    faces["zMax"] = np.delete(faces["zMax"], [0, 4], axis=0)
    report = cad_audit.audit(write_patch_set(tmp_path / "triSurface", faces))
    assert "2 faces of the exported surface are in no patch file" in measured_of(report, "coverage")


# -- the union, versus the parts ---------------------------------------------------


def test_a_correct_cube_is_closed_although_every_file_is_open(cad_audit, tmp_path):
    """The load-bearing case.

    Every one of the six files is an open surface with a rim; their union is a closed
    cube. An implementation that checks files one at a time reports six failures here
    and passes every other test in this module.
    """
    report = cad_audit.audit(write_patch_set(tmp_path / "triSurface", cube_faces(n=2)))
    per_file = report["measured"]["patches"]
    assert len(per_file) == 6
    for row in per_file:
        assert row["open_edges"] > 0, f"{row['name']} was expected to be an open surface"
    assert status_of(report, "closure") == "ok"
    assert status_of(report, "manifold") == "ok"
    assert status_of(report, "normals") == "ok"
    assert status_of(report, "coverage") == "ok"
    assert status_of(report, "self_intersection") == "ok"
    assert status_of(report, "degenerate") == "ok"
    assert report["ok"] is True


# -- invariance --------------------------------------------------------------------


def _statuses(report: dict) -> dict[str, str]:
    return {row["check"]: row["status"] for row in report["findings"]}


def test_every_status_survives_rotation_scaling_and_reordering(cad_audit, tmp_path):
    """The same surface, said four other ways, answers the same.

    A defective cube rather than a clean one, so the invariance being asserted is over
    findings that are actually saying something. The vertex permutation is a cyclic
    roll: a transposition would invert the triangle, and `normals` noticing that is the
    whole reason the check exists.
    """
    faces = cube_faces(n=2)
    faces["xMin"] = np.delete(faces["xMin"], [2, 6], axis=0)
    faces["yMin"] = faces["yMin"].copy()
    faces["yMin"][3] = faces["yMin"][3][[0, 2, 1]]

    base = _statuses(cad_audit.audit(write_patch_set(tmp_path / "base", faces)))
    assert base["closure"] == "fail" and base["normals"] == "fail"

    matrix = rotation(np.random.default_rng(5))
    rng = np.random.default_rng(6)
    variants = {
        "rotated": {name: rotate(tris, matrix) for name, tris in faces.items()},
        "scaled": {name: tris * 3.7 for name, tris in faces.items()},
        "shuffled": {name: tris[rng.permutation(len(tris))] for name, tris in faces.items()},
        "rolled": {name: np.roll(tris, 1, axis=1) for name, tris in faces.items()},
    }
    for label, patches in variants.items():
        report = cad_audit.audit(write_patch_set(tmp_path / label, patches))
        assert _statuses(report) == base, label


# -- degradation, not silence ------------------------------------------------------


def test_above_the_triangle_limit_the_expensive_findings_are_skipped_not_ok(
    cad_audit, preflight, tmp_path_factory
):
    """A surface nobody looked at is not a surface that passed.

    Four hundred thousand triangles is where `preflight` stops counting edges, so every
    finding that rests on the edge bookkeeping has to say so. `skipped` and a reason;
    never `ok`.
    """
    faces = cube_faces(n=186)
    total = sum(len(tris) for tris in faces.values())
    assert total > preflight.TOPOLOGY_TRIANGLE_LIMIT, total
    directory = write_patch_set(tmp_path_factory.mktemp("huge") / "triSurface", faces)

    started = time.time()
    report = cad_audit.audit(directory)
    elapsed = time.time() - started

    for check in ("closure", "manifold", "normals", "degenerate", "coverage",
                  "self_intersection"):
        assert status_of(report, check) == "skipped", check
        assert status_of(report, check) != "ok"
        assert "not computed" in measured_of(report, check)
        assert f"{preflight.TOPOLOGY_TRIANGLE_LIMIT:,}" in measured_of(report, check)
    assert status_of(report, "scale") == "ok", "the cheap findings still answer"
    assert report["ok"] is True, "skipped is not a failure"
    assert elapsed < 60, elapsed


# -- the manifest against the disk -------------------------------------------------


def test_a_patch_named_with_no_file_fails_naming_it(cad_audit, tmp_path):
    directory = write_patch_set(tmp_path / "triSurface", cube_faces(n=1))
    (directory / "zMax.stl").unlink()
    report = cad_audit.audit(directory)
    assert status_of(report, "manifest") == "fail"
    assert "zMax.stl" in measured_of(report, "manifest")
    assert report["ok"] is False


def test_a_file_with_no_manifest_entry_fails_naming_it(cad_audit, tmp_path):
    directory = write_patch_set(tmp_path / "triSurface", cube_faces(n=1))
    write_stl(directory / "stray.stl", octahedron(radius=0.1))
    report = cad_audit.audit(directory)
    assert status_of(report, "manifest") == "fail"
    assert "stray.stl" in measured_of(report, "manifest")


# -- nothing is reimplemented ------------------------------------------------------


def test_the_script_imports_the_machinery_rather_than_restating_it(cad_audit):
    """Two edge counters that are nearly the same is the mess the rule exists for."""
    source = (TOOLBOX / "cad_audit.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "import preflight" in source
    assert "import surfaces" in source

    defined = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for banned in ("weld", "edge", "tri_tri", "intersect_tri", "topology",
                   "read_triangles", "scale_diagnosis"):
        offenders = [name for name in defined if banned in name.lower()]
        assert not offenders, f"{offenders} looks like a second {banned} implementation"

    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            value = node.func.value
            if isinstance(value, ast.Name):
                called.add(f"{value.id}.{node.func.attr}")
    for needed in ("preflight.read_triangles", "preflight.surface_topology",
                   "preflight.scale_diagnosis", "surfaces.weld", "surfaces.Index",
                   "surfaces.tri_tri_intersect"):
        assert needed in called, f"{needed} is not called; something was rewritten"

    # The weld is a rounding rule, and there is exactly one of it.
    assert "np.round(" not in source
    assert "TOPOLOGY_TRIANGLE_LIMIT" in source, "the threshold is imported, not retyped"


def test_its_topology_numbers_are_preflights_numbers(cad_audit, preflight, tmp_path):
    """Not merely the same kind of number: the same number."""
    faces = cube_faces(n=3)
    faces["xMin"] = np.delete(faces["xMin"], [2, 11], axis=0)
    faces["yMin"] = faces["yMin"].copy()
    faces["yMin"][3] = faces["yMin"][3][[0, 2, 1]]
    report = cad_audit.audit(write_patch_set(tmp_path / "triSurface", faces))

    union = np.concatenate([
        preflight.read_triangles(tmp_path / "triSurface" / row["file"])
        for row in report["measured"]["patches"]
    ])
    theirs = preflight.surface_topology(union)
    mine = report["measured"]["union_topology"]
    for key in ("open_edges", "non_manifold_edges", "flipped_edges", "degenerate_triangles"):
        assert mine[key] == theirs[key], key
    assert f"{theirs['open_edges']} open edges" in measured_of(report, "closure")
    assert f"{theirs['flipped_edges']} same-direction edges" in measured_of(report, "normals")


def test_it_runs_on_numpy_alone(cad_audit):
    """The whole script is numpy over triangles, so its gate runs anywhere.

    Only `--surface-check` reaches for OpenFOAM, and that one degrades to `skipped`.
    """
    source = (TOOLBOX / "cad_audit.py").read_text(encoding="utf-8")
    for forbidden in ("import gmsh", "import build123d", "import OCP", "import pyvista"):
        assert forbidden not in source


# -- the envelope, and the refusals ------------------------------------------------


def test_the_json_envelope_is_the_one_every_new_script_emits(cad_audit, preflight, tmp_path):
    directory = write_patch_set(tmp_path / "triSurface", cube_faces(n=1))
    out = subprocess.run(
        [sys.executable, str(TOOLBOX / "cad_audit.py"), str(directory), "--json"],
        capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr
    report = json.loads(out.stdout)
    assert set(report) == {"script", "ok", "findings", "measured"}
    assert report["script"] == "cad_audit"
    findings = cad_audit.findings_of(report)
    assert report["ok"] == (preflight.worst_status(findings) != "fail")
    for row in report["findings"]:
        assert set(row) == {"check", "status", "measured", "means", "repair"}
        assert row["status"] in preflight.STATUSES
        assert row["measured"]
    names = {row["check"] for row in report["findings"]}
    assert {"closure", "manifold", "normals", "coverage", "self_intersection",
            "degenerate", "scale"} <= names


def test_a_failing_surface_still_exits_zero(cad_audit, tmp_path):
    faces = cube_faces(n=2)
    faces["zMax"] = np.delete(faces["zMax"], [0, 4], axis=0)
    directory = write_patch_set(tmp_path / "triSurface", faces)
    out = subprocess.run(
        [sys.executable, str(TOOLBOX / "cad_audit.py"), str(directory)],
        capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0
    assert "FAIL" in out.stdout


@pytest.mark.parametrize("make", ["absent", "no-manifest", "bad-manifest"])
def test_inputs_it_cannot_audit_are_refused_on_stderr_at_exit_two(tmp_path, make):
    """cad_convert.py's discipline: `refused: ...`, naming what to pass instead."""
    directory = tmp_path / "triSurface"
    if make != "absent":
        directory.mkdir()
        write_stl(directory / "wall.stl", octahedron())
    if make == "bad-manifest":
        (directory / "patches.json").write_text("{not json", encoding="utf-8")
    out = subprocess.run(
        [sys.executable, str(TOOLBOX / "cad_audit.py"), str(directory)],
        capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 2
    assert out.stderr.startswith("refused: ")
    assert "patches.json" in out.stderr or "triSurface" in out.stderr


def test_scale_uses_preflights_ladder(cad_audit, preflight, tmp_path):
    """A cube exported in millimetres and read as metres, which is what the ladder is
    there to name."""
    faces = {name: tris * 1000.0 for name, tris in cube_faces(n=1).items()}
    report = cad_audit.audit(write_patch_set(tmp_path / "triSurface", faces))
    assert status_of(report, "scale") == "warn"
    assert "millimetres" in means_of(report, "scale")
    assert preflight.MILLIMETRE_SUSPICION == 100.0


# -- surfaceCheck ------------------------------------------------------------------


def test_surface_check_says_it_did_not_run_rather_than_saying_nothing(
    cad_audit, tmp_path, monkeypatch
):
    monkeypatch.setattr(cad_audit.shutil, "which", lambda name: None)
    report = cad_audit.audit(write_patch_set(tmp_path / "triSurface", cube_faces(n=1)),
                             surface_check=True)
    assert status_of(report, "surface_check") == "skipped"
    assert "PATH" in measured_of(report, "surface_check")


@pytest.mark.skipif(shutil.which("surfaceCheck") is None,
                    reason="OpenFOAM is not on PATH in this shell")
def test_surface_check_runs_when_openfoam_is_here(cad_audit, tmp_path):
    report = cad_audit.audit(write_patch_set(tmp_path / "triSurface", cube_faces(n=2)),
                             surface_check=True)
    assert status_of(report, "surface_check") in ("ok", "warn")
    assert len(report["measured"]["surface_check"]) == 6
