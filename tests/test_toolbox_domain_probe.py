"""domain_probe.py -- held to the two definitions it exists to get right.

Both of this script's answers have a plausible cheap version that is wrong, and both
wrong versions run without erroring, which is the failure mode the whole CAD plan is
written against.

**Inside or outside** looks like a ray cast and a parity count, and it is -- but a
fixed direction through an L-shaped duct crosses the far arm twice, and a ray through
a shared edge counts a crossing twice or not at all. So the classification is checked
against an analytic predicate at ten thousand points on three shapes, one of them
non-convex, and the number of disagreements asserted to be **zero** rather than small.

**Local width** looks like twice the distance to the nearest wall. It is not: that is
the width only on the medial axis, and a quarter of the way across a slab it is half
of it. So the slab here is sampled deliberately off the medial axis, at 25% of its
thickness, where the cheap answer is wrong by a factor of two and the right one is
exact.

What this file does not test is the triangle machinery: `surfaces.py` owns the index,
the ray cast, the degenerate-ray re-cast and the signed distance, and
`tests/test_toolbox_surfaces.py` proves them. The last test here asserts that
`domain_probe.py` has not quietly grown its own.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
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


@pytest.fixture(scope="module")
def probe():
    return load("domain_probe")


# -- shapes with answers that were not computed by the thing under test ------------
#
# Every surface here is built from formulae, so the right answer is arithmetic rather
# than another implementation of the same algorithm. The two that are polyhedral -- the
# L-duct and the boxes -- are exactly the solid their predicate describes; the two
# round ones are inscribed in it, and their tests carry the deviation explicitly rather
# than pretending a tessellated sphere is a sphere.


def quad(a, b, c, d):
    return [[a, b, c], [a, c, d]]


def box(lower, upper):
    (x0, y0, z0) = np.asarray(lower, float)
    (x1, y1, z1) = np.asarray(upper, float)
    p = lambda x, y, z: np.array([x, y, z], float)  # noqa: E731
    t = []
    t += quad(p(x0, y0, z0), p(x1, y0, z0), p(x1, y1, z0), p(x0, y1, z0))
    t += quad(p(x0, y0, z1), p(x1, y0, z1), p(x1, y1, z1), p(x0, y1, z1))
    t += quad(p(x0, y0, z0), p(x1, y0, z0), p(x1, y0, z1), p(x0, y0, z1))
    t += quad(p(x0, y1, z0), p(x1, y1, z0), p(x1, y1, z1), p(x0, y1, z1))
    t += quad(p(x0, y0, z0), p(x0, y1, z0), p(x0, y1, z1), p(x0, y0, z1))
    t += quad(p(x1, y0, z0), p(x1, y1, z0), p(x1, y1, z1), p(x1, y0, z1))
    return np.array(t)


def uv_sphere(rows, cols, radius=1.0):
    def P(i, j):
        theta = np.pi * i / rows
        phi = 2 * np.pi * j / cols
        return radius * np.array([np.sin(theta) * np.cos(phi),
                                  np.sin(theta) * np.sin(phi), np.cos(theta)])
    t = []
    for i in range(rows):
        for j in range(cols):
            a, b, c, d = P(i, j), P(i, j + 1), P(i + 1, j + 1), P(i + 1, j)
            if i:
                t.append([a, b, c])
            if i + 1 < rows:
                t.append([a, c, d])
    return np.array(t)


def torus(major=1.0, minor=0.35, nu=24, nv=14):
    def P(i, j):
        u, v = 2 * np.pi * i / nu, 2 * np.pi * j / nv
        rho = major + minor * np.cos(v)
        return np.array([rho * np.cos(u), rho * np.sin(u), minor * np.sin(v)])
    t = []
    for i in range(nu):
        for j in range(nv):
            t += quad(P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1))
    return np.array(t)


def l_duct(arm=1.0, width=0.3, depth=0.6):
    """An L-shaped prism, and the reason a fixed ray direction is not enough: a ray
    along +x from the tall arm leaves it, crosses the gap, and enters and leaves the
    short arm again -- three crossings where a convex solid gives one."""
    w, L, d = width, arm, depth
    t = []
    for z in (0.0, depth):
        t += quad(np.array([0, 0, z]), np.array([w, 0, z]),
                  np.array([w, L, z]), np.array([0, L, z]))
        t += quad(np.array([w, 0, z]), np.array([L, 0, z]),
                  np.array([L, w, z]), np.array([w, w, z]))
    loop = [(0, 0), (L, 0), (L, w), (w, w), (w, L), (0, L)]
    for k in range(len(loop)):
        (x0, y0), (x1, y1) = loop[k], loop[(k + 1) % len(loop)]
        t += quad(np.array([x0, y0, 0.0]), np.array([x1, y1, 0.0]),
                  np.array([x1, y1, d]), np.array([x0, y0, d]))
    return np.array(t)


def revolved(profile, nu=96):
    """A closed solid of revolution about z, from a closed profile in (rho, z)."""
    prof = [(float(r), float(z)) for r, z in profile]
    t = []
    P = lambda r, z, a: np.array([r * np.cos(a), r * np.sin(a), z])  # noqa: E731
    for k in range(len(prof)):
        (r0, z0), (r1, z1) = prof[k], prof[(k + 1) % len(prof)]
        if r0 == 0 and r1 == 0:
            continue
        for i in range(nu):
            a0, a1 = 2 * np.pi * i / nu, 2 * np.pi * (i + 1) / nu
            if r0 == 0:
                t.append([P(r0, z0, a0), P(r1, z1, a0), P(r1, z1, a1)])
            elif r1 == 0:
                t.append([P(r0, z0, a0), P(r1, z1, a0), P(r0, z0, a1)])
            else:
                t += quad(P(r0, z0, a0), P(r0, z0, a1), P(r1, z1, a1), P(r1, z1, a0))
    return np.array(t)


def write_stl(path: Path, triangles: np.ndarray, name: str = "solid") -> Path:
    lines = [f"solid {name}"]
    for tri in triangles:
        lines.append("facet normal 0 0 0")
        lines.append("  outer loop")
        for vertex in tri:
            lines.append("    vertex %.12g %.12g %.12g" % tuple(vertex))
        lines.append("  endloop")
        lines.append("endfacet")
    lines.append(f"endsolid {name}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def patch_dir(tmp_path: Path, name: str, parts: dict, manifest: dict | None = None) -> Path:
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    for stem, triangles in parts.items():
        write_stl(directory / f"{stem}.stl", triangles, stem)
    if manifest is not None:
        (directory / "patches.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


# -- the analytic predicates -------------------------------------------------------


def inside_convex(triangles, points):
    """Exact, for a convex polyhedron: inside every face's plane at once.

    This is the answer for the *tessellated* sphere rather than for the sphere, which
    is the point -- it leaves no deviation to argue about, so the comparison against it
    admits no tolerance at all.
    """
    v0 = triangles[:, 0]
    normals = np.cross(triangles[:, 1] - v0, triangles[:, 2] - v0)
    centre = triangles.reshape(-1, 3).mean(axis=0)
    outward = np.einsum("ij,ij->i", normals, v0 - centre) < 0
    normals[outward] *= -1
    offsets = np.einsum("ij,ij->i", normals, v0)
    return ((np.asarray(points, float) @ normals.T) - offsets < 0).all(axis=1)


def inside_torus(points, major=1.0, minor=0.35):
    p = np.asarray(points, float)
    rho = np.hypot(p[:, 0], p[:, 1])
    return (rho - major) ** 2 + p[:, 2] ** 2 < minor ** 2


def torus_surface_gap(points, major=1.0, minor=0.35):
    p = np.asarray(points, float)
    rho = np.hypot(p[:, 0], p[:, 1])
    return np.abs(np.hypot(rho - major, p[:, 2]) - minor)


def inside_l_duct(points, arm=1.0, width=0.3, depth=0.6):
    p = np.asarray(points, float)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    return ((x > 0) & (x < arm) & (y > 0) & (y < arm)
            & ((x < width) | (y < width)) & (z > 0) & (z < depth))


def l_duct_surface_gap(points, arm=1.0, width=0.3, depth=0.6):
    """Distance to the nearest face plane of the L, which bounds the distance to the
    surface from below -- enough to know a point is not near it."""
    p = np.asarray(points, float)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    planes = [x, arm - x, y, arm - y, z, depth - z, width - x, width - y]
    return np.min(np.abs(np.stack(planes, axis=1)), axis=1)


# -- item 1: classified identically to the analytic predicate, everywhere ----------


POINTS = 10_000


@pytest.fixture(scope="module")
def shell():
    """A hollow sphere. Inside the outer and outside the inner, which no single ray
    direction can get wrong quietly -- a point in the cavity crosses two surfaces."""
    outer, inner = uv_sphere(10, 20, 1.0), uv_sphere(10, 20, 0.5)
    return np.concatenate([outer, inner]), outer, inner


@pytest.mark.parametrize("shape", ["sphere shell", "torus", "l-duct"])
def test_every_point_is_classified_as_the_analytic_shape_says(probe, shell, shape):
    """Zero disagreements, not few.

    A classification that is right 99.9% of the time is a `locationInMesh` that is
    wrong one run in a thousand, and that run produces a mesh, passes `checkMesh` and
    reports a drag coefficient for the outside of the part.

    The two round shapes are inscribed polyhedra, so near their surface the polyhedron
    and the analytic solid genuinely differ and neither answer is a mistake; points
    within that deviation are excluded and counted, and the deviation is the chord
    sagitta, computed from the tessellation rather than tuned. The L-duct is exactly
    the solid its predicate describes and gets no such allowance.
    """
    rng = np.random.default_rng(20260911)
    if shape == "sphere shell":
        triangles, outer, inner = shell
        lower, upper = np.full(3, -1.2), np.full(3, 1.2)
        points = rng.uniform(lower, upper, size=(POINTS, 3))
        expected = inside_convex(outer, points) & ~inside_convex(inner, points)
        comparable = np.ones(len(points), dtype=bool)  # exact for a polyhedron
    elif shape == "torus":
        triangles = torus(nu=24, nv=14)
        points = rng.uniform([-1.4, -1.4, -0.4], [1.4, 1.4, 0.4], size=(POINTS, 3))
        expected = inside_torus(points)
        # The chord sagitta of the tessellation, both ways round: the widest circle
        # of latitude the quads cut across has radius major + minor, and the tube
        # circle has radius minor. Inside that band the inscribed polyhedron and the
        # torus are different solids and either answer is right about one of them.
        deviation = ((1.0 + 0.35) * (1 - np.cos(np.pi / 24))
                     + 0.35 * (1 - np.cos(np.pi / 14)))
        comparable = torus_surface_gap(points) > deviation
    else:
        triangles = l_duct()
        points = rng.uniform([-0.1, -0.1, -0.1], [1.1, 1.1, 0.7], size=(POINTS, 3))
        expected = inside_l_duct(points)
        comparable = np.ones(len(points), dtype=bool)

    union = probe.Union(triangles, ["shape.stl"])
    labels, distances = probe.classify(union, points)
    labels = np.array(labels)

    band = labels == "on_surface"
    assert (np.abs(distances[band]) <= union.band).all()
    comparable &= ~band
    assert comparable.sum() > 0.8 * POINTS, "too much of the sample was set aside"

    said_inside = labels == "inside"
    disagreements = np.flatnonzero(comparable & (said_inside != expected))
    assert len(disagreements) == 0, (
        f"{len(disagreements)} of {int(comparable.sum())} points classified against "
        f"the analytic predicate, first at {points[disagreements[:3]]}"
    )
    # And the sample was not trivially one-sided.
    assert 0.1 < expected[comparable].mean() < 0.9


# -- item 2: the band is reported, not guessed -------------------------------------


def test_a_point_inside_the_band_comes_back_on_surface_with_its_distance(probe):
    """Within `1e-6 x diagonal` of a triangle the sign is a rounding decision. The
    house rule is to say what was not resolved, so the classification says on-surface
    and hands back the number instead of picking a side."""
    triangles = box((0, 0, 0), (0.2, 0.3, 0.4))
    union = probe.Union(triangles, ["box.stl"])
    centroids = triangles.mean(axis=1)

    on = centroids
    labels, distances = probe.classify(union, on)
    assert set(labels) == {"on_surface"}
    assert (np.abs(distances) <= union.band).all()

    # Just inside the band on either side: still on-surface, still with the distance.
    offsets = np.array([[0.0, 0.0, 0.4 * union.band]])
    nudged = np.concatenate([centroids + offsets, centroids - offsets])
    labels, distances = probe.classify(union, nudged)
    assert set(labels) == {"on_surface"}
    assert (np.abs(distances) <= union.band).all()

    # And a hair outside it is answered with confidence again, so the band is a band
    # and not a refusal to classify near the wall at all.
    inside = np.array([[0.1, 0.15, 0.4 - 20 * union.band]])
    labels, _distances = probe.classify(union, inside)
    assert labels == ["inside"]


def test_a_point_on_the_surface_is_a_failed_location_and_says_why(probe, tmp_path):
    directory = patch_dir(tmp_path, "band", {"box": box((0, 0, 0), (0.2, 0.3, 0.4))})
    report = probe.probe(directory, point=[0.1, 0.15, 0.0], width_samples=0)
    found = {f.check: f for f in probe.findings(report)}
    assert report["classification"] == "on_surface"
    assert found["location_in_mesh"].status == "fail"
    assert "%.6g" % report["clearance_m"] in found["location_in_mesh"].measured
    assert not probe.envelope(report, probe.findings(report))["ok"]


# -- item 3: clearance is a number -------------------------------------------------


def test_clearance_is_measured_and_the_margin_rule_is_not_the_binary(probe, tmp_path):
    """A point half a millimetre into a one-millimetre gap is inside, and how far
    inside is the number that decides whether snappy keeps it. The same point moved to
    twenty microns from the wall is still inside and is no longer a good point, and the
    finding has to be able to say so."""
    gap = 0.001
    directory = patch_dir(tmp_path, "gap", {
        "outer": box((0, 0, 0), (0.100, 0.080, gap)),
    })

    middle = probe.probe(directory, point=[0.05, 0.04, 0.5 * gap], width_samples=0)
    assert middle["classification"] == "inside"
    assert abs(middle["clearance_m"] - 0.5 * gap) <= 0.01 * 0.5 * gap
    found = {f.check: f for f in probe.findings(middle)}
    assert found["location_in_mesh"].status == "ok"

    near = probe.probe(directory, point=[0.05, 0.04, 0.00002], width_samples=0)
    assert near["classification"] == "inside"
    assert abs(near["clearance_m"] - 0.00002) <= 0.01 * 0.00002
    found = {f.check: f for f in probe.findings(near)}
    assert found["location_in_mesh"].status == "warn"
    assert "%.6g" % near["margin_m"] in found["location_in_mesh"].measured
    # Inside is not the verdict on its own: the same binary, two different statuses.
    assert probe.envelope(near, probe.findings(near))["ok"] is True


# -- item 4: width reproduces a field, not a constant -------------------------------


def test_the_slab_is_measured_off_its_medial_axis_and_2_sdf_is_not_the_answer(probe):
    """A quarter of the way across a 10 mm slab, twice the distance to the nearest wall
    is 5 mm and the width is 10 mm. This is the test that separates the definition from
    the shortcut, so it asserts both: the right answer to 2%, and that the shortcut is
    wrong by the factor it is wrong by."""
    thickness = 0.010
    union = probe.Union(box((0, 0, 0), (0.200, 0.200, thickness)), ["slab.stl"])
    point = [0.100, 0.100, 0.25 * thickness]

    measured = probe.local_width(union, point)
    assert abs(measured["width"] - thickness) <= 0.02 * thickness

    shortcut = 2 * abs(float(probe.surfaces.signed_distance(
        union.index, np.array([point]))[0]))
    assert abs(shortcut - 0.5 * thickness) < 1e-9
    assert abs(shortcut - thickness) > 0.4 * thickness


def test_the_annulus_gap_is_the_gap_and_not_the_bore(probe):
    inner_radius, outer_radius, height = 0.040, 0.050, 0.120
    gap = outer_radius - inner_radius
    union = probe.Union(revolved([(inner_radius, 0.0), (outer_radius, 0.0),
                                  (outer_radius, height), (inner_radius, height)],
                                 nu=96), ["annulus.stl"])
    for fraction in (0.5, 0.25):
        point = [inner_radius + fraction * gap, 0.0, 0.5 * height]
        measured = probe.local_width(union, point)
        assert abs(measured["width"] - gap) <= 0.02 * gap, fraction


def test_width_follows_a_cone_that_widens_rather_than_reporting_one_number(probe):
    """Twenty stations up the axis of a frustum. The reference is the same
    maximisation done by brute force on the cone's own analytic distance field, in the
    (rho, z) half-plane where it is a two-dimensional problem, so it is arithmetic
    rather than a second copy of the search under test."""
    small, large, height = 0.020, 0.060, 0.200
    walls = [((small, 0.0), (large, height)), ((0.0, 0.0), (small, 0.0)),
             ((large, height), (0.0, height))]

    def wall_distance(z):
        here = np.array([0.0, z])
        best = np.inf
        for start, end in walls:
            a, b = np.array(start), np.array(end)
            edge = b - a
            t = float(np.clip(((here - a) @ edge) / (edge @ edge), 0.0, 1.0))
            best = min(best, float(np.linalg.norm(here - (a + t * edge))))
        return best

    axis = np.linspace(0.0, height, 4001)
    radii = np.array([wall_distance(z) for z in axis])

    def analytic_width(z):
        holds = np.abs(axis - z) <= radii
        return 2.0 * radii[holds].max()

    union = probe.Union(revolved([(0.0, 0.0), (small, 0.0), (large, height),
                                  (0.0, height)], nu=96), ["cone.stl"])
    stations = np.linspace(0.02, 0.18, 20)
    measured = np.array([probe.local_width(union, [0.0, 0.0, z])["width"]
                         for z in stations])
    expected = np.array([analytic_width(z) for z in stations])

    correlation = float(np.corrcoef(measured, expected)[0, 1])
    assert correlation > 0.99, correlation
    assert np.max(np.abs(measured - expected) / expected) < 0.05
    # A field, not a constant: the wide end is measurably wider than the narrow one.
    assert measured[-1] > 1.5 * measured[0]


# -- item 5: wall thickness on a hollow box ----------------------------------------


HOLLOW_WALL = 0.002


@pytest.fixture(scope="module")
def hollow_box():
    """A box with a cavity in it, offset so one wall is 2 mm and the other five are
    5 mm. The known minimum is the 2 mm one, and it is the answer whether it is read as
    the thinnest wall or as the narrowest place in what the surfaces enclose."""
    outer = box((0, 0, 0), (0.100, 0.080, 0.060))
    inner = box((HOLLOW_WALL, 0.005, 0.005), (0.095, 0.075, 0.055))
    return np.concatenate([outer, inner])


def test_the_hollow_box_gives_up_its_known_minimum_wall(probe, tmp_path, hollow_box):
    directory = patch_dir(tmp_path, "hollow", {"outer_and_inner": hollow_box})
    report = probe.probe(directory, width_samples=120)
    field = report["width"]

    assert report["shells"] == 2
    assert abs(field["min_wall_thickness"] - HOLLOW_WALL) <= 0.02 * HOLLOW_WALL
    assert abs(field["min_width"] - HOLLOW_WALL) <= 0.02 * HOLLOW_WALL

    found = {f.check: f for f in probe.findings(report)}
    assert found["min_wall_thickness"].status == "ok"
    assert "%.6g" % field["min_wall_thickness"] in found["min_wall_thickness"].measured


def test_one_shell_has_no_wall_and_the_finding_says_so_rather_than_inventing_one(
        probe, tmp_path):
    directory = patch_dir(tmp_path, "single", {"box": box((0, 0, 0), (0.1, 0.1, 0.1))})
    report = probe.probe(directory, width_samples=10)
    found = {f.check: f for f in probe.findings(report)}
    assert report["shells"] == 1
    assert found["min_wall_thickness"].status == "skipped"
    assert "one connected shell" in found["min_wall_thickness"].measured


# -- item 6: --suggest survives its own validation ---------------------------------


@pytest.mark.parametrize("shape,radius", [("sphere shell", 0.25),
                                          ("torus", 0.35),
                                          ("l-duct", 0.15)])
def test_the_suggested_point_passes_the_check_that_asked_for_it(
        probe, tmp_path, shell, shape, radius):
    """The largest sphere that fits: half the shell's thickness, the torus tube's own
    radius, half the L-duct's width. A suggestion that comes back anything but `ok`,
    or that sits shallower than that, is a suggestion the next script cannot use."""
    triangles = {"sphere shell": shell[0], "torus": torus(nu=24, nv=14),
                 "l-duct": l_duct()}[shape]
    directory = patch_dir(tmp_path, shape.replace(" ", "_"), {"shape": triangles})

    report = probe.probe(directory, suggest=True, width_samples=0)
    found = {f.check: f for f in probe.findings(report)}

    assert report["point_source"] == "--suggest"
    assert report["classification"] == "inside"
    assert found["location_in_mesh"].status == "ok"
    assert report["clearance_m"] >= 0.95 * radius, (shape, report["clearance_m"])
    assert report["clearance_m"] == pytest.approx(report["suggested_radius_m"])


# -- item 7: an unvalidated point is a finding -------------------------------------


def test_a_manifest_with_no_location_in_mesh_is_a_failure_not_a_silence(
        probe, tmp_path):
    """The check that stops snappy meshing the outside of the part and exiting 0."""
    directory = patch_dir(
        tmp_path, "absent", {"box": box((0, 0, 0), (0.1, 0.1, 0.1))},
        manifest={"unit_metres": 1.0, "patches": [{"name": "walls", "file": "box.stl"}]},
    )
    report = probe.probe(directory, width_samples=0)
    found = {f.check: f for f in probe.findings(report)}
    assert report["point"] is None
    assert found["location_in_mesh"].status == "fail"
    assert "location_in_mesh" in found["location_in_mesh"].measured
    assert "--suggest" in found["location_in_mesh"].repair
    assert probe.envelope(report, probe.findings(report))["ok"] is False


def test_a_manifest_point_outside_the_domain_is_a_failure(probe, tmp_path):
    directory = patch_dir(
        tmp_path, "outside", {"box": box((0, 0, 0), (0.1, 0.1, 0.1))},
        manifest={"unit_metres": 1.0, "location_in_mesh": [0.2, 0.05, 0.05]},
    )
    report = probe.probe(directory, width_samples=0)
    found = {f.check: f for f in probe.findings(report)}
    assert report["point_source"] == "patches.json"
    assert report["classification"] == "outside"
    assert found["location_in_mesh"].status == "fail"
    assert "outside" in found["location_in_mesh"].measured


def test_a_manifest_point_inside_the_domain_passes(probe, tmp_path):
    directory = patch_dir(
        tmp_path, "good", {"box": box((0, 0, 0), (0.1, 0.1, 0.1))},
        manifest={"unit_metres": 1.0, "location_in_mesh": [0.05, 0.05, 0.05]},
    )
    report = probe.probe(directory, width_samples=0)
    found = {f.check: f for f in probe.findings(report)}
    assert found["location_in_mesh"].status == "ok"
    assert probe.envelope(report, probe.findings(report))["ok"] is True


# -- the envelope and the refusals -------------------------------------------------


def run(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOLBOX / "domain_probe.py"), *args],
        capture_output=True, text=True,
    )


def test_the_envelope_is_i2_and_the_exit_status_is_zero_whatever_is_found(tmp_path):
    directory = patch_dir(
        tmp_path, "envelope", {"box": box((0, 0, 0), (0.1, 0.1, 0.1))},
        manifest={"unit_metres": 1.0, "location_in_mesh": [0.2, 0.05, 0.05]},
    )
    done = run(str(directory), "--width-samples", "8", "--json")
    assert done.returncode == 0, done.stderr
    payload = json.loads(done.stdout[done.stdout.index("{"):])
    assert payload["script"] == "domain_probe"
    assert payload["ok"] is False  # it found something, and still exited 0
    assert set(payload) == {"script", "ok", "findings", "measured"}
    for finding in payload["findings"]:
        assert set(finding) == {"check", "status", "measured", "means", "repair"}
        assert finding["status"] in ("fail", "warn", "ok", "skipped")
    assert {f["check"] for f in payload["findings"]} == {
        "location_in_mesh", "min_width", "min_wall_thickness"}
    assert payload["measured"]["triangles"] == 12


def test_ok_is_exactly_worst_status_not_fail(probe, tmp_path):
    preflight = load("preflight")
    directory = patch_dir(tmp_path, "worst", {"box": box((0, 0, 0), (0.1, 0.1, 0.1))},
                          manifest={"location_in_mesh": [0.05, 0.05, 0.05]})
    report = probe.probe(directory, width_samples=6)
    found = probe.findings(report)
    envelope = probe.envelope(report, found)
    assert envelope["ok"] == (preflight.worst_status(found) != "fail")


@pytest.mark.parametrize("case", ["missing", "file", "empty"])
def test_a_refused_input_is_exit_2_and_says_what_to_pass(tmp_path, case):
    if case == "missing":
        target = tmp_path / "nowhere"
    elif case == "file":
        target = write_stl(tmp_path / "one.stl", box((0, 0, 0), (1, 1, 1)))
    else:
        target = tmp_path / "empty"
        target.mkdir()
    done = run(str(target))
    assert done.returncode == 2
    assert done.stderr.startswith("refused: ")
    assert "triSurface" in done.stderr or "does not exist" in done.stderr
    assert done.stdout == ""


def test_the_findings_carry_numbers_a_person_could_quote(probe, tmp_path, hollow_box):
    directory = patch_dir(tmp_path, "quotable", {"hollow": hollow_box},
                          manifest={"location_in_mesh": [0.05, 0.04, 0.03]})
    report = probe.probe(directory, width_samples=12)
    found = probe.findings(report)
    for finding in found:
        assert finding.measured.strip()
        assert any(character.isdigit() for character in finding.measured)
    text = probe.render(report, found)
    assert "domain_probe" in text
    assert "location_in_mesh" in text


# -- item 8: nothing is reimplemented ----------------------------------------------


def test_domain_probe_reaches_for_surfaces_and_preflight_rather_than_writing_them():
    """The index, the ray cast, the degenerate-ray re-cast and the signed distance are
    `surfaces.py`'s, and `read_triangles` and the register are `preflight.py`'s. A
    second copy of any of them is a second set of tolerances, and the one that is wrong
    is whichever one nobody tested."""
    source = (TOOLBOX / "domain_probe.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported = {node.names[0].name for node in ast.walk(tree)
                if isinstance(node, ast.Import)}
    assert {"surfaces", "preflight"} <= imported

    used = {f"{node.value.id}.{node.attr}" for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)}
    assert "surfaces.signed_distance" in used
    assert "surfaces.Index" in used
    assert "surfaces.weld" in used
    assert "preflight.read_triangles" in used
    assert "preflight.Finding" in used
    assert "preflight.worst_status" in used

    # No geometry kernel of its own. Every ray cast, every point-to-triangle distance
    # and every triangle plane needs a cross product, and there is not one here.
    assert "np.cross" not in source
    assert "cross(" not in source
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            assert node.name in {"Union", "Refused"}, node.name
        if isinstance(node, ast.FunctionDef):
            # "index" is not on this list: `Union.shell_index` hands out a
            # `surfaces.Index` per shell and builds nothing. A spatial index of its
            # own would have to be a class, and the classes are already pinned above.
            assert not any(word in node.name for word in
                           ("ray", "cast", "crossing", "parity", "intersect",
                            "nearest", "candidates", "recast", "bvh")), node.name
    # And the re-cast C2a proves is not re-solved here.
    for word in ("recast", "last_recasts", "last_direction", "RECAST"):
        assert word not in source
