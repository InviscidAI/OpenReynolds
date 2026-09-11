"""surfaces.py -- the shared triangle machinery, held to the one property it sells.

The module is a library three other toolbox scripts import rather than reimplement, and
almost everything it is asked can fail quietly. A candidate query that drops a triangle
does not raise: it returns a self-intersection that was not found, a crossing that was
not counted, a point reported outside a body it is inside. So the tests here are built
around *supersets* -- every candidate query is checked against an exhaustive scan and
must contain it, never equal it -- and around agreement with naive implementations that
are slow enough to be obviously right.

The two that are easiest to fake are given their own treatment. The degenerate ray is
asserted to have actually re-cast, because a ray through a shared edge can return the
right count by luck and then return the wrong one on the next surface. And the index is
asserted to be an order of magnitude faster than the scan it replaces, because an index
that returns everything is correct and worthless.
"""

from __future__ import annotations

import ast
import importlib.util
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
def surfaces():
    return load("surfaces")


@pytest.fixture(scope="module")
def preflight():
    return load("preflight")


# -- surfaces to measure on --------------------------------------------------------


def random_triangles(rng, count, spread=1.0, size=0.4):
    """Triangles scattered in a box, each small enough that plenty of pairs miss and
    plenty meet."""
    centres = rng.uniform(-spread, spread, size=(count, 1, 3))
    return centres + rng.uniform(-size, size, size=(count, 3, 3))


def uv_sphere(rows, cols, radius=1.0, centre=(0.0, 0.0, 0.0)):
    """A closed sphere as triangles, with the seam closed exactly and the two poles'
    zero-area triangles dropped rather than left to be argued about."""
    theta = np.linspace(0.0, np.pi, rows + 1)
    phi = np.linspace(0.0, 2.0 * np.pi, cols + 1)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    points = np.stack(
        [radius * np.sin(t) * np.cos(p),
         radius * np.sin(t) * np.sin(p),
         radius * np.cos(t)],
        axis=-1,
    )
    points[:, -1, :] = points[:, 0, :]          # the seam is one column, not two
    points[0, :, :] = [0.0, 0.0, radius]        # and a pole is one point
    points[-1, :, :] = [0.0, 0.0, -radius]
    points = points + np.asarray(centre, dtype=float)

    a = points[:-1, :-1]
    b = points[:-1, 1:]
    c = points[1:, 1:]
    d = points[1:, :-1]
    lower = np.stack([a, b, c], axis=2).reshape(-1, 3, 3)
    upper = np.stack([a, c, d], axis=2).reshape(-1, 3, 3)
    tris = np.concatenate([lower, upper])
    area = np.linalg.norm(
        np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1
    )
    return np.ascontiguousarray(tris[area > 0])


def unit_cube():
    """A unit cube as twelve triangles, every face split on the same diagonal.

    Splitting every face the same way is deliberate: it puts a shared edge exactly
    where an axis-parallel ray from `x = y` will run into it, which is the degenerate
    case the re-cast exists for.
    """
    corners = np.array([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])

    def quad(i, j, k, m):
        return [[corners[i], corners[j], corners[k]], [corners[i], corners[k], corners[m]]]

    # corner i is (x, y, z) read off the bits of i: 4x + 2y + z.
    faces = []
    faces += quad(0, 2, 6, 4)   # z = 0, split on (0,0,0)-(1,1,0), which is x = y
    faces += quad(1, 3, 7, 5)   # z = 1, split the same way
    faces += quad(0, 1, 5, 4)   # y = 0
    faces += quad(2, 3, 7, 6)   # y = 1
    faces += quad(0, 1, 3, 2)   # x = 0
    faces += quad(4, 5, 7, 6)   # x = 1
    return np.asarray(faces, dtype=float)


# -- naive implementations, slow enough to be obviously right ----------------------


def naive_sat(a, b):
    """A direct separating-axis test: build the axes, project, look for a gap."""
    def edges(t):
        return [t[1] - t[0], t[2] - t[1], t[0] - t[2]]

    normal_a = np.cross(a[1] - a[0], a[2] - a[0])
    normal_b = np.cross(b[1] - b[0], b[2] - b[0])
    axes = [normal_a, normal_b]
    for ea in edges(a):
        for eb in edges(b):
            axes.append(np.cross(ea, eb))
    for edge in edges(a) + edges(b):
        axes.append(np.cross(edge, normal_a))
        axes.append(np.cross(edge, normal_b))

    scale = max(float(np.abs(a).max()), float(np.abs(b).max()), 1e-300)
    floor = (scale * 1e-12) ** 2
    for axis in axes:
        if float(np.dot(axis, axis)) <= floor:
            continue
        pa = [float(np.dot(vertex, axis)) for vertex in a]
        pb = [float(np.dot(vertex, axis)) for vertex in b]
        if max(pa) < min(pb) or max(pb) < min(pa):
            return False
    return True


def naive_ray_hits(tris, origin, direction):
    """Every triangle tested, by solving the 3x3 system rather than by Moller's
    shortcut -- the same question asked a different way."""
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    v0 = tris[:, 0]
    columns = np.stack([np.broadcast_to(-direction, v0.shape),
                        tris[:, 1] - v0, tris[:, 2] - v0], axis=2)
    dets = np.linalg.det(columns)
    scale = np.abs(columns).max(axis=(1, 2)) ** 3
    good = np.abs(dets) > 1e-12 * np.maximum(scale, 1e-300)
    if not good.any():
        return 0, np.zeros(0, dtype=np.int64)
    rhs = (origin - v0)[good][:, :, None]
    solved = np.linalg.solve(columns[good], rhs)[:, :, 0]
    t, u, v = solved[:, 0], solved[:, 1], solved[:, 2]
    hit = (t > 0.0) & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
    return int(hit.sum()), np.flatnonzero(good)[hit]


def naive_point_triangle_distance(point, tri):
    """Ericson's closest-point-on-triangle, by regions, one triangle at a time."""
    p = np.asarray(point, dtype=float)
    a, b, c = tri
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0 and d2 <= 0:
        return float(np.linalg.norm(p - a))
    bp = p - b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0 and d4 <= d3:
        return float(np.linalg.norm(p - b))
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        denominator = d1 - d3
        t = d1 / denominator if denominator != 0 else 0.0
        return float(np.linalg.norm(p - (a + t * ab)))
    cp = p - c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0 and d5 <= d6:
        return float(np.linalg.norm(p - c))
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        denominator = d2 - d6
        t = d2 / denominator if denominator != 0 else 0.0
        return float(np.linalg.norm(p - (a + t * ac)))
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        denominator = (d4 - d3) + (d5 - d6)
        t = (d4 - d3) / denominator if denominator != 0 else 0.0
        return float(np.linalg.norm(p - (b + t * (c - b))))
    total = va + vb + vc
    if total == 0:
        return float(np.linalg.norm(p - a))
    return float(np.linalg.norm(p - (a + ab * (vb / total) + ac * (vc / total))))


def exhaustive_near(tris, point, radius):
    return {
        i for i, tri in enumerate(tris)
        if naive_point_triangle_distance(point, tri) <= radius
    }


def exhaustive_boxes_along(tris, origin, direction):
    """Every triangle whose own box the ray meets, by the slab test, one at a time."""
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    found = set()
    for i, tri in enumerate(tris):
        lo, hi = tri.min(axis=0), tri.max(axis=0)
        tmin, tmax = 0.0, float("inf")
        for axis in range(3):
            if abs(direction[axis]) < 1e-300:
                if origin[axis] < lo[axis] or origin[axis] > hi[axis]:
                    tmin, tmax = 1.0, -1.0
                    break
                continue
            t0 = (lo[axis] - origin[axis]) / direction[axis]
            t1 = (hi[axis] - origin[axis]) / direction[axis]
            tmin = max(tmin, min(t0, t1))
            tmax = min(tmax, max(t0, t1))
        if tmax >= tmin:
            found.add(i)
    return found


def exhaustive_overlapping(tris, tri_id):
    lo, hi = tris[tri_id].min(axis=0), tris[tri_id].max(axis=0)
    los, his = tris.min(axis=1), tris.max(axis=1)
    overlap = np.all((his >= lo) & (los <= hi), axis=1)
    return set(np.flatnonzero(overlap).tolist())


# -- 1. the one property: no false negatives ---------------------------------------


def test_no_candidate_query_ever_under_returns(surfaces):
    """Fifty randomised surfaces, three queries each, against exhaustive scans.

    The assertion is containment, not equality. Over-returning costs an exact test on
    a triangle that turns out not to matter; under-returning is a missed intersection,
    a missed crossing or a wrong sign, and every one of those is silent. This is the
    property the rest of the module is built on, so it is measured across the widest
    set of shapes the test can afford rather than on one favourable case.
    """
    rng = np.random.default_rng(11)
    over_near, over_along, over_overlap = [], [], []
    scanned_near = 0

    for trial in range(50):
        count = int(rng.integers(30, 160))
        tris = random_triangles(rng, count, spread=1.0, size=0.35)
        index = surfaces.Index(tris, leaf=int(rng.integers(1, 9)))

        point = rng.uniform(-1.4, 1.4, size=3)
        radius = float(rng.uniform(0.05, 0.8))
        got = set(index.candidates_near(point, radius).tolist())
        true = exhaustive_near(tris, point, radius)
        assert true <= got, f"trial {trial}: candidates_near dropped {sorted(true - got)}"
        over_near.append(len(got))
        scanned_near += count

        origin = rng.uniform(-2.5, 2.5, size=3)
        direction = rng.normal(size=3)
        got = set(index.candidates_along(origin, direction).tolist())
        boxes = exhaustive_boxes_along(tris, origin, direction)
        _, hits = naive_ray_hits(tris, origin, direction)
        true = boxes | set(hits.tolist())
        assert true <= got, f"trial {trial}: candidates_along dropped {sorted(true - got)}"
        over_along.append(len(got))

        tri_id = int(rng.integers(0, count))
        got = set(index.candidates_overlapping(tri_id).tolist())
        true = exhaustive_overlapping(tris, tri_id)
        true |= {
            j for j in range(count)
            if surfaces.tri_tri_intersect(tris[tri_id], tris[j])
        }
        assert true <= got, (
            f"trial {trial}: candidates_overlapping dropped {sorted(true - got)}"
        )
        over_overlap.append(len(got))

    # Measured, not asserted tight: what matters is that it is a filter at all. An
    # index that answered every query with every triangle would pass every containment
    # assertion above and be worth nothing.
    assert sum(over_near) < 0.9 * scanned_near, (
        f"candidates_near returned {sum(over_near)} of {scanned_near} triangles scanned"
    )
    assert min(over_near + over_along + over_overlap) >= 0


# -- 2. the primitives against naive implementations -------------------------------


def test_tri_tri_intersect_agrees_with_a_direct_sat_test(surfaces):
    rng = np.random.default_rng(23)
    tris = random_triangles(rng, 2000, spread=0.6, size=0.4)
    agreed, intersecting = 0, 0
    for i in range(0, 2000, 2):
        a, b = tris[i], tris[i + 1]
        mine = surfaces.tri_tri_intersect(a, b)
        theirs = naive_sat(a, b)
        assert mine == theirs, f"pair {i}: {mine} against {theirs}\n{a}\n{b}"
        agreed += 1
        intersecting += bool(mine)
    assert agreed == 1000
    # Neither verdict may be the only one seen, or the agreement means nothing.
    assert 0 < intersecting < 1000


def test_tri_tri_intersect_answers_the_cases_that_are_known_by_hand(surfaces):
    flat = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cases = [
        ("far apart", np.array([[5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.0, 6.0, 5.0]]), False),
        ("run through", np.array([[0.2, 0.2, -1.0], [0.3, 0.2, 1.0], [0.2, 0.3, 1.0]]), True),
        ("parallel plane", np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]), False),
        ("shares an edge", np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]), True),
        ("shares a vertex", np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0]]), True),
        ("coplanar overlap", np.array([[0.2, 0.2, 0.0], [1.2, 0.2, 0.0], [0.2, 1.2, 0.0]]), True),
        ("coplanar apart", np.array([[2.0, 2.0, 0.0], [3.0, 2.0, 0.0], [2.0, 3.0, 0.0]]), False),
        ("coplanar inside", np.array([[0.1, 0.1, 0.0], [0.3, 0.1, 0.0], [0.1, 0.3, 0.0]]), True),
    ]
    for name, other, expected in cases:
        assert surfaces.tri_tri_intersect(flat, other) is expected, name
        assert surfaces.tri_tri_intersect(other, flat) is expected, f"{name}, reversed"


def test_ray_hits_agrees_with_testing_every_triangle(surfaces):
    tris = uv_sphere(26, 26, radius=1.0)
    assert len(tris) < 2000
    index = surfaces.Index(tris)
    rng = np.random.default_rng(37)
    clean, inside_seen, outside_seen = 0, 0, 0
    for _ in range(60):
        origin = rng.uniform(-2.0, 2.0, size=3)
        direction = rng.normal(size=3)
        mine = surfaces.ray_hits(index, origin, direction)
        theirs, _ = naive_ray_hits(tris, origin, direction)
        if index.last_recasts:
            # The scan has no re-cast, so the two are answering different rays and
            # comparing their counts would compare nothing. The degenerate ray has its
            # own tests below, with the answer known by construction.
            continue
        assert mine == theirs
        clean += 1
        if mine % 2:
            inside_seen += 1
        else:
            outside_seen += 1
    assert clean > 50, f"only {clean} of 60 rays were clean; the fixture is degenerate"
    assert inside_seen and outside_seen


def test_signed_distance_agrees_with_a_full_point_to_triangle_scan(surfaces):
    tris = uv_sphere(24, 24, radius=1.0, centre=(0.1, -0.2, 0.05))
    index = surfaces.Index(tris)
    rng = np.random.default_rng(41)
    points = rng.uniform(-1.8, 1.8, size=(25, 3))
    mine = surfaces.signed_distance(index, points)
    for point, value in zip(points, mine):
        scanned = min(naive_point_triangle_distance(point, tri) for tri in tris)
        assert abs(abs(value) - scanned) <= 1e-9, f"{point}: {abs(value)} against {scanned}"
        analytic_inside = np.linalg.norm(point - np.array([0.1, -0.2, 0.05])) < 0.98
        analytic_outside = np.linalg.norm(point - np.array([0.1, -0.2, 0.05])) > 1.02
        if analytic_inside:
            assert value < 0, f"{point} is inside the sphere and came back {value}"
        if analytic_outside:
            assert value > 0, f"{point} is outside the sphere and came back {value}"


# -- 3. the degenerate ray, and the proof it was re-cast ---------------------------


def test_a_ray_straight_through_a_shared_edge_is_re_cast_and_counted(surfaces):
    """Every face of the cube is split on the same diagonal, so a `+z` ray at x = y
    runs exactly along the shared edge of two triangles on the bottom face and again
    on the top. Counting both hits gives 4, counting neither gives 0, and both answers
    put the point outside a box it enters.
    """
    index = surfaces.Index(unit_cube())
    origin = np.array([0.3, 0.3, -1.0])
    direction = np.array([0.0, 0.0, 1.0])
    crossings = surfaces.ray_hits(index, origin, direction)
    assert crossings == 2
    assert index.last_recasts >= 1, "the shared edge was not detected as degenerate"
    assert not np.allclose(index.last_direction, direction), (
        "last_direction is still the direction that was asked for"
    )
    assert index.recasts >= 1


def test_a_ray_straight_through_a_vertex_is_re_cast_and_counted(surfaces):
    """The body diagonal: it passes exactly through the corner at the origin and
    exactly through the corner at (1, 1, 1), so at each end a whole fan of triangles
    meets the ray at one point."""
    index = surfaces.Index(unit_cube())
    origin = np.array([-1.0, -1.0, -1.0])
    direction = np.array([1.0, 1.0, 1.0])
    crossings = surfaces.ray_hits(index, origin, direction)
    assert crossings == 2
    assert index.last_recasts >= 1, "the vertex was not detected as degenerate"
    assert not np.allclose(index.last_direction, surfaces._unit(direction))


def test_a_point_inside_reached_through_a_degenerate_ray_still_reads_as_inside(surfaces):
    index = surfaces.Index(unit_cube())
    inside = np.array([0.5, 0.5, 0.5])
    crossings = surfaces.ray_hits(index, inside, np.array([0.0, 0.0, 1.0]))
    assert crossings % 2 == 1
    assert index.last_recasts >= 1


def test_a_clean_ray_is_not_re_cast(surfaces):
    """The counter has to mean something, so it must also stay at zero."""
    index = surfaces.Index(unit_cube())
    before = index.recasts
    crossings = surfaces.ray_hits(index, np.array([0.37, 0.61, -1.0]),
                                  np.array([0.03, -0.07, 1.0]))
    assert crossings == 2
    assert index.last_recasts == 0
    assert index.recasts == before


# -- 4. weld is preflight's rule, not a second one ---------------------------------


def test_weld_finds_the_vertex_count_preflight_derives_inside_itself(
    surfaces, preflight, monkeypatch
):
    """`surface_topology()` does not return its vertex count, so it is watched rather
    than asked: the one `np.unique(..., axis=0, return_inverse=True)` call it makes is
    where the weld happens, and what that call returns is the number in question.

    Two rounding rules is exactly the failure this function exists to prevent, and a
    test that recomputed preflight's rule here would be the second rule.
    """
    tris = uv_sphere(18, 18, radius=1.0)
    seen = []
    real_unique = preflight.np.unique

    def watched(values, *args, **kwargs):
        result = real_unique(values, *args, **kwargs)
        if kwargs.get("axis") == 0 and kwargs.get("return_inverse"):
            seen.append(len(result[0]))
        return result

    monkeypatch.setattr(preflight.np, "unique", watched)
    topology = preflight.surface_topology(tris)
    assert topology["computed"]
    assert len(seen) == 1, "preflight no longer welds in one np.unique call"

    corners, vertices = surfaces.weld(tris)
    assert len(vertices) == seen[0]
    assert int(corners.max()) + 1 == seen[0]
    assert corners.shape == (len(tris), 3)
    # And the welded corners are the same corners, not merely the same count.
    assert topology["open_edges"] == 0
    assert len(np.unique(corners.reshape(-1))) == seen[0]


def test_weld_returns_vertices_in_the_surfaces_own_units(surfaces):
    tris = unit_cube() * 1000.0
    corners, vertices = surfaces.weld(tris)
    assert len(vertices) == 8
    assert np.allclose(np.sort(vertices, axis=0)[0], [0.0, 0.0, 0.0])
    assert np.allclose(np.sort(vertices, axis=0)[-1], [1000.0, 1000.0, 1000.0])
    assert np.allclose(vertices[corners], tris)


# -- 5. the index earns its place --------------------------------------------------


@pytest.mark.parametrize("query", ["near", "along"])
def test_the_index_beats_an_exhaustive_scan_by_an_order_of_magnitude(surfaces, query):
    """Two hundred thousand triangles, the same answer, a tenth of the time.

    An index that is correct because it returns everything is not an index, and a small
    fixture never shows the difference -- the scan wins below a few thousand triangles.
    So both halves are asserted here and only here: identical answers, and at least a
    factor of ten.
    """
    tris = uv_sphere(317, 317, radius=1.0)
    assert 190_000 <= len(tris) <= 210_000, len(tris)

    built = time.perf_counter()
    index = surfaces.Index(tris)
    built = time.perf_counter() - built
    assert built < 60.0, f"building the index took {built:.1f}s"

    rng = np.random.default_rng(53)

    if query == "near":
        # Points just off the shell, not scattered through the box: a query that comes
        # back empty is answered identically by both sides and measures nothing.
        directions = rng.normal(size=(5, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        points = directions * rng.uniform(0.94, 1.06, size=(5, 1))
        radius = 0.05

        def indexed():
            answers = []
            for point in points:
                ids = index.candidates_near(point, radius)
                distances = surfaces._point_triangle_distances(point, tris[ids])
                answers.append(set(ids[distances <= radius].tolist()))
            return answers

        def scanned():
            answers = []
            for point in points:
                distances = surfaces._point_triangle_distances(point, tris)
                answers.append(set(np.flatnonzero(distances <= radius).tolist()))
            return answers
    else:
        # Aimed through the sphere rather than thrown at random, for the same reason.
        starts = rng.normal(size=(5, 3))
        starts /= np.linalg.norm(starts, axis=1)[:, None]
        origins = starts * 3.0
        targets = rng.uniform(-0.5, 0.5, size=(5, 3))
        directions = targets - origins

        def indexed():
            answers = []
            for origin, direction in zip(origins, directions):
                ids = index.candidates_along(origin, direction)
                _, hits = naive_ray_hits(tris[ids], origin, direction)
                answers.append(set(ids[hits].tolist()))
            return answers

        def scanned():
            answers = []
            for origin, direction in zip(origins, directions):
                _, hits = naive_ray_hits(tris, origin, direction)
                answers.append(set(hits.tolist()))
            return answers

    # Warm once so neither side is paying for the first touch of a large array.
    answers = indexed()
    assert answers == scanned()
    assert sum(len(answer) for answer in answers) > 0, (
        "every query came back empty, so the two sides agreed about nothing"
    )

    fast = min(_timed(indexed) for _ in range(3))
    slow = min(_timed(scanned) for _ in range(3))
    assert slow / fast >= 10.0, (
        f"{query}: indexed {fast * 1e3:.2f} ms against exhaustive {slow * 1e3:.2f} ms "
        f"-- only {slow / fast:.1f}x"
    )


def _timed(call):
    start = time.perf_counter()
    call()
    return time.perf_counter() - start


# -- 6. the sign, pinned ------------------------------------------------------------


def test_a_point_inside_a_closed_sphere_has_a_negative_distance(surfaces):
    """Negative is inside. Every caller depends on it and none of them re-check it,
    so it is asserted here on the least ambiguous shape there is."""
    tris = uv_sphere(40, 40, radius=2.0, centre=(1.0, 2.0, 3.0))
    index = surfaces.Index(tris)
    centre = np.array([1.0, 2.0, 3.0])
    values = surfaces.signed_distance(
        index,
        np.array([centre, centre + [1.0, 0.0, 0.0], centre + [0.0, 0.0, -1.5],
                  centre + [3.0, 0.0, 0.0], centre + [0.0, 5.0, 0.0]]),
    )
    assert values[0] < 0 and values[1] < 0 and values[2] < 0
    assert values[3] > 0 and values[4] > 0
    assert abs(values[0] + 2.0) < 0.02, values[0]
    assert abs(values[3] - 1.0) < 0.02, values[3]
    assert abs(values[4] - 3.0) < 0.02, values[4]


def test_an_empty_surface_answers_rather_than_raises(surfaces):
    """C5 validates a patch set before it writes it, and a patch set can be empty.
    Every query answering with nothing is the right answer; a traceback is not."""
    index = surfaces.Index(np.zeros((0, 3, 3)))
    assert len(index.candidates_near([0.0, 0.0, 0.0], 1.0)) == 0
    assert len(index.candidates_along([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])) == 0
    assert len(index.candidates_overlapping(0)) == 0
    assert surfaces.ray_hits(index, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == 0
    assert np.all(np.isinf(surfaces.signed_distance(index, np.zeros((2, 3)))))
    corners, vertices = surfaces.weld(np.zeros((0, 3, 3)))
    assert corners.shape == (0, 3) and vertices.shape == (0, 3)


def test_signed_distance_takes_one_point_as_readily_as_many(surfaces):
    index = surfaces.Index(unit_cube())
    one = surfaces.signed_distance(index, np.array([0.5, 0.5, 0.5]))
    assert one.shape == (1,) and one[0] < 0


# -- the module is a library, and stays one ----------------------------------------


def test_surfaces_is_a_library_with_no_cli_and_no_io():
    """The only toolbox script that is not a command. A `main` or an `open()` here
    would make it a script three other scripts import, which is how a library acquires
    a working directory."""
    tree = ast.parse((TOOLBOX / "surfaces.py").read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "main" not in names
    source = (TOOLBOX / "surfaces.py").read_text(encoding="utf-8")
    for forbidden in ("argparse", "__main__", "open(", "read_bytes", "write_text",
                      "print(", "Finding("):
        assert forbidden not in source, f"surfaces.py mentions {forbidden}"


def test_it_offers_exactly_the_frozen_interface(surfaces):
    """I8, and nothing beyond it. Three chunks are being written against this
    signature in parallel, so an addition here is not free."""
    assert sorted(name for name in surfaces.__all__ if not name.isupper()) == [
        "Index", "ray_hits", "signed_distance", "tri_tri_intersect", "weld",
    ]
    public = sorted(
        name for name in dir(surfaces.Index)
        if not name.startswith("_")
    )
    assert "candidates_near" in public
    assert "candidates_along" in public
    assert "candidates_overlapping" in public
