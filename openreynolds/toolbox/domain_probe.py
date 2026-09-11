"""Is the point inside the domain, and how wide is the domain where it is.

    python3 domain_probe.py <triSurface-dir> [--point x y z] [--suggest]
    python3 domain_probe.py constant/triSurface --width-samples 400 --json

Two questions nothing in this repo answered before, both about a closed triangle
surface and neither needing a mesh.

**Is the point inside.** `locationInMesh` is how snappyHexMesh is told which side of
the surface the fluid is on. Get it wrong and the mesh still comes out, `checkMesh`
still passes, and what was meshed is the volume *outside* the part. The test is a ray
cast and a parity count; `surfaces.ray_hits()` does it, including the re-cast when the
ray runs exactly along a shared edge, and this script does not re-solve that.

**How wide is it there.** Local width is the diameter of the largest sphere that fits
inside the domain **and contains the point** -- a maximisation over spheres, not a
lookup. `2 * |signed distance|` is twice the distance to the nearest wall, and that
equals the width only on the medial axis: a quarter of the way across a 10 mm slab it
reports 5 mm. The same field read on a surface that bounds material rather than fluid
is wall thickness, which is the number that predicts whether a fluid boolean will
survive a thin-walled fin.

**The band.** A point within `1e-6 x diagonal` of the surface is reported as
on-surface with the distance that was measured, not resolved to a side. Inside that
band the answer is floating point rather than logic, and saying so beats guessing.

It writes no geometry and reads no B-rep. `preflight.read_triangles` reads the STLs,
`surfaces.Index`, `ray_hits` and `signed_distance` do the geometry, and what is left
-- assembling the union, the margin policy, the width definition and the suggestion
search -- is here.

Exit status is 0 whatever is found. A refused input is exit 2 with the reason on
stderr, which is `cad_convert.py`'s discipline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preflight  # noqa: E402  (sibling script, not a package)
import surfaces  # noqa: E402  (sibling script, not a package)


SCRIPT = "domain_probe"

BAND_FRACTION = 1e-6
"""Times the bounding-box diagonal. Closer to the surface than this and the sign of
the distance is a rounding decision, so the classification says on-surface and hands
back the number instead of a side."""

MARGIN_FRACTION = 1e-3
"""Times the diagonal. A `locationInMesh` nearer than this to a wall is inside and
still a bad point: snappy resolves the point against a cell, and a point that close to
the surface picks up whichever cell the castellation happened to leave there."""

WIDTH_SAMPLES = 120
"""Interior points the width field is measured at by default. Each one costs an
ascent of its own, tens of distance evaluations, so this trades seconds for resolution
and is a flag rather than a constant."""

ASCENT_ROUNDS = 12
"""Rounds the inscribed-sphere search gets: one aim and one slide each. A point in a
wall settles in two of them; a tapering duct, where the sphere keeps sliding towards
the wide end until the point falls out of it, uses most of them."""

COARSE_ROUNDS = 6
"""Rounds the first pass over the samples gets. Enough to settle a point in a wall and
enough to recognise a corner, which is all the first pass is for."""

REFINE_TOP = 20
"""How many of the narrowest samples are measured again at higher effort. The minimum
is the only order statistic anyone reads, so it is the one worth paying for."""

_LADDER = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
"""Lengths the line search walks, as multiples of the sphere's current radius, until
the sphere stops holding the point. It starts *below* the radius and not at it: the
best sphere along the way is often a short step off, halfway across a wall, and a
ladder that begins at the radius steps straight over it. It ends far above, because a
point against the wall of a duct is most of a diameter from the middle of it."""

_SETTLED = 1e-4
"""A round that grows the sphere by less than this has found what it is going to
find. Without it a sphere creeping up a corner spends every round it is given for a
number that is thrown away."""

_ACTIVE = 1e-3
"""How near the sphere's own surface the point has to sit before the measurement is
called a corner rather than a width. On the surface exactly, the sphere is as big as
it is only because it had to reach the point, and that is a fact about the point."""

_BISECTIONS = 6
"""Steps spent locating the length at which the sphere stops holding the point. The
constraint boundary is where the answer sits whenever the domain is tapering, so it is
worth finding properly rather than approaching by halves from outside."""


class Refused(Exception):
    """An input this cannot work from. Exit 2, with the reason on stderr."""


# -- the union on disk -------------------------------------------------------------


class Union:
    """Every patch STL in one triangle array, with the index and the lengths derived.

    The patch files are open surfaces one at a time -- an inlet is a disc -- and only
    their union is closed, so every question here is asked of the union and none of it
    of a single file. Nothing is welded across files beyond what `weld` already does,
    because the files already share their edge vertices if the export was right; if
    they do not, `cad_audit.py` is the script that says so.
    """

    def __init__(self, triangles: np.ndarray, files: list[str], manifest: dict | None = None):
        self.triangles = np.asarray(triangles, dtype=float)
        self.files = list(files)
        self.manifest = manifest or {}
        self.index = surfaces.Index(self.triangles)
        flat = self.triangles.reshape(-1, 3)
        self.lower = flat.min(axis=0)
        self.upper = flat.max(axis=0)
        self.diagonal = float(np.linalg.norm(self.upper - self.lower))
        self.band = BAND_FRACTION * self.diagonal
        self.margin = MARGIN_FRACTION * self.diagonal
        self._shell_labels: np.ndarray | None = None
        self._shell_indexes: list[surfaces.Index] | None = None

    # -- shells ---------------------------------------------------------------

    @property
    def shell_labels(self) -> np.ndarray:
        """One component number per triangle, on welded vertices.

        A hollow part is two closed shells with material between them, and that is the
        only configuration in which "wall thickness" names something a single closed
        surface does not already answer. The welding is `surfaces.weld`'s, so the
        components are found on the same vertices `preflight.surface_topology()`
        counts edges on.
        """
        if self._shell_labels is None:
            self._shell_labels = _components(self.triangles)
        return self._shell_labels

    @property
    def shells(self) -> int:
        labels = self.shell_labels
        return int(labels.max()) + 1 if len(labels) else 0

    def shell_index(self, shell: int) -> surfaces.Index:
        if self._shell_indexes is None:
            labels = self.shell_labels
            self._shell_indexes = [
                surfaces.Index(self.triangles[labels == s]) for s in range(self.shells)
            ]
        return self._shell_indexes[shell]


def read_union(directory: Path) -> Union:
    """The patch STLs of a triSurface directory, plus `patches.json` if it is there."""
    directory = Path(directory)
    if not directory.exists():
        raise Refused(
            f"refused: {directory} does not exist. Pass the triSurface directory "
            "holding one STL per patch."
        )
    if not directory.is_dir():
        raise Refused(
            f"refused: {directory} is a file, not a directory. Pass the triSurface "
            "directory holding one STL per patch, not one of the STLs in it."
        )

    manifest: dict | None = None
    manifest_path = directory / "patches.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = None

    names = sorted(p.name for p in directory.iterdir() if p.suffix.lower() == ".stl")
    if not names:
        raise Refused(
            f"refused: no .stl files in {directory}. A patch set is one STL per patch "
            "in constant/triSurface; cad_convert.py writes them."
        )

    parts, files = [], []
    for name in names:
        tris = preflight.read_triangles(directory / name)
        if tris is None or len(tris) == 0:
            continue
        parts.append(tris)
        files.append(name)
    if not parts:
        raise Refused(
            f"refused: the .stl files in {directory} hold no triangles between them. "
            "Check the export wrote facets, not an empty solid."
        )
    return Union(np.concatenate(parts, axis=0), files, manifest)


def _components(triangles: np.ndarray) -> np.ndarray:
    """Connected components of the triangle graph, joined through shared vertices."""
    corners, vertices = surfaces.weld(triangles)
    parent = np.arange(len(vertices) + len(corners), dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    offset = len(vertices)
    for tri, corner in enumerate(corners):
        root = find(offset + tri)
        for vertex in corner:
            other = find(int(vertex))
            if other != root:
                parent[other] = root
    roots = np.array([find(offset + tri) for tri in range(len(corners))], dtype=np.int64)
    _unique, labels = np.unique(roots, return_inverse=True)
    return np.asarray(labels, dtype=np.int64).reshape(-1)


# -- inside, outside, or too close to say ------------------------------------------


def classify(union: Union, points) -> tuple[list[str], np.ndarray]:
    """`("inside" | "outside" | "on_surface", ...)` and the signed distances.

    The sign is `surfaces.signed_distance`'s -- negative inside -- and the band is the
    only thing added: within `1e-6 x diagonal` of a triangle, a point is reported as
    on-surface with its measured distance rather than resolved to a side it may not be
    on. Fewer confident answers, no wrong ones.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    distances = surfaces.signed_distance(union.index, pts)
    labels = [
        "on_surface" if abs(float(d)) <= union.band else ("inside" if d < 0 else "outside")
        for d in distances
    ]
    return labels, distances


def _radius(union: Union, centre) -> float:
    """How big a sphere fits at `centre`: the distance to the nearest triangle, and
    negative when the centre is outside, so an outside centre fails every feasibility
    test below without needing a second check."""
    return -float(surfaces.signed_distance(union.index, np.asarray(centre, float).reshape(1, 3))[0])


# -- the largest inscribed sphere containing the point -----------------------------


def local_width(union: Union, point, radius: float | None = None,
                rounds: int = ASCENT_ROUNDS) -> dict:
    """Local width at `point`: the diameter of the largest sphere inside the domain
    that **contains** the point.

    Maximise `r(c)` -- the distance from `c` to the nearest triangle -- over centres
    `c` satisfying `|point - c| <= r(c)`. The constraint is what makes the answer both
    correct and safe: an open ball of radius `r(c)` about `c` meets no triangle, so if
    it also contains the point then the whole ball is domain and the point is in it.
    A step can therefore never tunnel through a wall into a different region, however
    far it jumps.

    Started at the point itself and walked uphill: the gradient of `r` is the unit
    vector away from the nearest wall, taken by forward differences, and the step
    along it is the best of a ladder of multiples of the current radius. When no step
    on the ladder improves, the ladder halves.

    Returns the width, the centre and radius it settled on, and `converging`: whether
    the constraint is *active* at the optimum -- whether the point is sitting on the
    sphere's own surface rather than inside it. That is the signature of a corner or
    an edge, where the largest sphere containing a point is limited by the point
    rather than by the geometry, and its diameter is a property of the corner and not
    a thickness. `min_width` below reads that flag rather than averaging over it.
    """
    p = np.asarray(point, dtype=float).reshape(3)
    start = _radius(union, p) if radius is None else float(radius)
    result = {
        "point": [float(v) for v in p],
        "width": 0.0,
        "radius": 0.0,
        "centre": [float(v) for v in p],
        "converging": False,
        "evaluations": 0 if radius is not None else 1,
        "inside": start > 0.0,
    }
    if start <= 0.0:
        return result

    best_c, best_r = p.copy(), start
    evaluations = result["evaluations"]
    blocked = False
    previous = None
    for _ in range(rounds):
        gradient, spent = _uphill(union, best_c, best_r)
        evaluations += spent
        length = float(np.linalg.norm(gradient))
        if length <= 0.0:
            break
        direction = gradient / length

        moved, refused, spent = _slide(union, p, best_c, direction, best_r)
        evaluations += spent
        if previous is not None and (moved is None or moved[1] < best_r * 1.1):
            # Two walls meeting at an angle make a ridge in the distance field, and
            # the steepest way up a ridge is across it: the gradient points at one
            # wall, then at the other, and the walk zigzags towards the crest a few
            # percent at a time. Half the sum of the two is along the ridge, which is
            # where it was trying to go.
            blend = previous + direction
            span = float(np.linalg.norm(blend))
            if span > 0.0:
                along, also, spent = _slide(union, p, best_c, blend / span, best_r)
                evaluations += spent
                refused = refused or also
                if along is not None and (moved is None or along[1] > moved[1]):
                    moved, direction = along, blend / span
        previous = direction
        if moved is None:
            blocked = refused
            break
        settled = moved[1] < best_r * (1.0 + _SETTLED)
        best_c, best_r = moved
        if settled:
            break

    gap = float(np.linalg.norm(p - best_c))
    result.update(
        width=2.0 * best_r,
        radius=best_r,
        centre=[float(v) for v in best_c],
        converging=blocked or gap >= best_r * (1.0 - _ACTIVE),
        evaluations=evaluations,
    )
    return result


def _uphill(union, centre, radius: float) -> tuple[np.ndarray, int]:
    """Which way the domain gets wider, by central differences on the distance field.

    Central and not forward, which costs twice as much and is not a refinement. On the
    axis of a round duct the distance falls away in every radial direction at once --
    the field has a ridge there, not a slope -- and a forward difference reads that as
    a strong push sideways, off the ridge, in whichever direction it happened to probe.
    Differencing both ways cancels it, which is the difference between following a duct
    towards its wide end and walking into its wall.
    """
    h = max(radius * 1e-3, union.diagonal * 1e-9)
    gradient = np.empty(3)
    for axis in range(3):
        ahead = np.asarray(centre, dtype=float).copy()
        behind = ahead.copy()
        ahead[axis] += h
        behind[axis] -= h
        gradient[axis] = _radius(union, ahead) - _radius(union, behind)
    return gradient, 6


def _slide(union, point, centre, direction, radius):
    """Push the sphere along `direction` as far as it can go while still holding
    `point`, and report the biggest it got.

    Two things can stop it and they are different answers. The radius can peak on its
    own -- the sphere has found the middle of a slab and growing further is not
    possible in any direction -- which is a width. Or the radius can still be growing
    when the point falls out of the sphere, which happens in every tapering feature
    and in every corner, and means the number is bounded by where the point is rather
    than by how wide the domain is. The second is reported back as `refused`, and
    `min_width` reads it.

    `(best, refused, evaluations)`, where `best` is `(centre, radius)` or None.
    """
    p = np.asarray(point, dtype=float).reshape(3)
    samples: list[tuple[float, float, bool]] = [(0.0, radius, True)]
    evaluations = 0
    refused = False

    def look(t: float):
        nonlocal evaluations, refused
        c = centre + t * direction
        r = _radius(union, c)
        evaluations += 1
        holds = float(np.linalg.norm(p - c)) <= r
        if r > radius and not holds:
            refused = True
        samples.append((t, r, holds))
        return holds

    step = max(radius, union.diagonal * 1e-9)
    last_held, first_lost = 0.0, None
    for factor in _LADDER:
        if look(factor * step):
            last_held = factor * step
        else:
            first_lost = factor * step
            break
    if first_lost is not None:
        lo, hi = last_held, first_lost
        for _ in range(_BISECTIONS):
            mid = 0.5 * (lo + hi)
            if look(mid):
                lo = mid
            else:
                hi = mid

    held = sorted((t, r) for t, r, holds in samples if holds)
    best_t, best_r = max(held, key=lambda tr: tr[1])

    # Golden-section around the best length, between its neighbours. The ladder is
    # geometric and steps over narrow maxima: a point nine tenths of the way across a
    # wall is a twentieth of its own radius from the middle of it, and the shortest
    # rung is an eighth. So this runs whether or not the ladder found anything.
    lengths = [t for t, _r in held]
    where = lengths.index(best_t)
    lo = lengths[where - 1] if where else 0.0
    hi = lengths[where + 1] if where + 1 < len(lengths) else best_t * 1.5
    for _ in range(3):
        for probe_t in (lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)):
            c = centre + probe_t * direction
            r = _radius(union, c)
            evaluations += 1
            if r > best_r and float(np.linalg.norm(p - c)) <= r:
                best_t, best_r = probe_t, r
        span = hi - lo
        lo, hi = max(lo, best_t - 0.25 * span), min(hi, best_t + 0.25 * span)

    if best_r <= radius * (1.0 + 1e-12):
        return None, refused, evaluations
    return (centre + best_t * direction, best_r), refused, evaluations


def _touching_shells(union: Union, centre, radius: float, tolerance: float = 1e-3) -> list[int]:
    """Which shells the sphere at `centre` actually rests on.

    Asked of each shell separately through its own index, because "the nearest
    triangle" says nothing about whether the second-nearest belongs to a different
    surface, and material between two shells is the only thing that makes a number
    wall thickness rather than domain width.
    """
    if union.shells < 2:
        return [0] if union.shells else []
    point = np.asarray(centre, float).reshape(1, 3)
    reach = radius * (1.0 + tolerance)
    touching = []
    for shell in range(union.shells):
        distance = abs(float(surfaces.signed_distance(union.shell_index(shell), point)[0]))
        if distance <= reach:
            touching.append(shell)
    return touching


# -- the width field ---------------------------------------------------------------


def interior_samples(union: Union, count: int, seed: int = 20260911) -> tuple[np.ndarray, int]:
    """`count` points strictly inside the domain, and how many draws it took.

    Rejection sampling in the bounding box, which needs no meshing and no notion of
    volume, and costs one distance evaluation a draw. A domain that fills little of
    its box costs more draws; the draws are capped and the number actually taken is
    reported, because a width field measured at nine points is a different claim from
    one measured at two hundred.
    """
    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    draws = 0
    limit = max(40 * count, 400)
    while len(kept) < count and draws < limit:
        batch = min(max(count, 32), limit - draws)
        points = rng.uniform(union.lower, union.upper, size=(batch, 3))
        distances = surfaces.signed_distance(union.index, points)
        draws += batch
        for point, distance in zip(points, distances):
            if distance < -union.band:
                kept.append(point)
                if len(kept) == count:
                    break
    if not kept:
        return np.zeros((0, 3)), draws
    return np.array(kept), draws


def width_field(union: Union, count: int = WIDTH_SAMPLES, seed: int = 20260911) -> dict:
    """The width measured at interior samples, and the narrowest of them.

    Two passes: every sample gets a short ascent, and the narrowest `REFINE_TOP` get a
    long one, because the minimum is the only number out of this anybody quotes.

    The minimum is taken over the samples whose sphere is a genuine local maximum of
    the distance field -- `converging` false. The rest sit in corners, where the
    largest sphere containing the point shrinks with the distance to the corner and
    goes to zero at it; including them would report every box-shaped domain as
    infinitely narrow. How many were set aside is reported rather than quietly
    dropped, and the minimum over all of them is reported too, so the filter can be
    seen rather than trusted.
    """
    samples, draws = interior_samples(union, count, seed=seed)
    field = {
        "samples": int(len(samples)),
        "draws": int(draws),
        "qualified": 0,
        "converging": 0,
        "min_width": None,
        "min_width_at": None,
        "min_width_any": None,
        "max_inscribed_radius": None,
        "max_inscribed_at": None,
        "min_wall_thickness": None,
        "min_wall_thickness_at": None,
    }
    if not len(samples):
        return field

    measurements = [local_width(union, point, rounds=COARSE_ROUNDS) for point in samples]
    candidates = [i for i, m in enumerate(measurements) if not m["converging"]]
    candidates.sort(key=lambda i: measurements[i]["width"])
    for i in candidates[:REFINE_TOP]:
        measurements[i] = local_width(union, samples[i], rounds=ASCENT_ROUNDS)

    qualified = [m for m in measurements if not m["converging"]]
    field["qualified"] = len(qualified)
    field["converging"] = len(measurements) - len(qualified)
    field["min_width_any"] = float(min(m["width"] for m in measurements))
    widest = max(measurements, key=lambda m: m["radius"])
    field["max_inscribed_radius"] = float(widest["radius"])
    field["max_inscribed_at"] = list(widest["centre"])
    if qualified:
        narrow = min(qualified, key=lambda m: m["width"])
        field["min_width"] = float(narrow["width"])
        field["min_width_at"] = list(narrow["point"])

    if union.shells >= 2:
        walled = [
            m for m in qualified
            if len(_touching_shells(union, m["centre"], m["radius"])) >= 2
        ]
        if walled:
            thinnest = min(walled, key=lambda m: m["width"])
            field["min_wall_thickness"] = float(thinnest["width"])
            field["min_wall_thickness_at"] = list(thinnest["point"])
    return field


# -- where the point should be -----------------------------------------------------


def suggest_point(union: Union, count: int = 60, seed: int = 20260911) -> dict:
    """A point at the maximum of the interior distance field.

    The deepest point in the domain is the one a coarse castellation is least likely
    to take away, and it is what `--suggest` proposes for the manifest. Found by
    sampling the interior, keeping the deepest few draws, and then repeatedly
    replacing each by the centre of the largest sphere that holds it -- which is
    `local_width` again, and each replacement lands somewhere strictly deeper until
    nothing is deeper, so the walk ends on a local maximum of the field.
    """
    samples, _draws = interior_samples(union, count, seed=seed)
    if not len(samples):
        return {"point": None, "radius": 0.0}
    depths = -surfaces.signed_distance(union.index, samples)
    starts = samples[np.argsort(depths)[::-1][:3]]

    best_point, best_radius = None, 0.0
    for start in starts:
        point, radius = start.copy(), float(-surfaces.signed_distance(
            union.index, start.reshape(1, 3))[0])
        for _ in range(6):
            step = local_width(union, point, radius=radius)
            if step["radius"] <= radius * (1.0 + 1e-9):
                break
            point = np.asarray(step["centre"], dtype=float)
            radius = step["radius"]
        if radius > best_radius:
            best_point, best_radius = point, radius
    if best_point is None:
        return {"point": None, "radius": 0.0}
    return {"point": [float(v) for v in best_point], "radius": float(best_radius)}


# -- the probe ---------------------------------------------------------------------


def probe(directory: Path, point=None, width_samples: int = WIDTH_SAMPLES,
          suggest: bool = False, seed: int = 20260911) -> dict:
    """Everything this script measures, as one dictionary. `findings()` reads it."""
    union = read_union(directory)
    manifest_point = union.manifest.get("location_in_mesh") if union.manifest else None
    if isinstance(manifest_point, (list, tuple)) and len(manifest_point) == 3:
        try:
            manifest_point = [float(v) for v in manifest_point]
        except (TypeError, ValueError):
            manifest_point = None
    else:
        manifest_point = None

    suggestion = suggest_point(union, seed=seed) if suggest else None

    source, chosen = "", None
    if point is not None:
        source, chosen = "--point", [float(v) for v in point]
    elif suggestion and suggestion["point"] is not None:
        source, chosen = "--suggest", list(suggestion["point"])
    elif manifest_point is not None:
        source, chosen = "patches.json", list(manifest_point)

    report = {
        "directory": str(directory),
        "files": union.files,
        "triangles": int(len(union.triangles)),
        "shells": union.shells,
        "bounds": [float(v) for v in (*union.lower, *union.upper)],
        "diagonal_m": union.diagonal,
        "band_m": union.band,
        "margin_m": union.margin,
        "manifest": bool(union.manifest),
        "manifest_point": manifest_point,
        "point": chosen,
        "point_source": source,
        "classification": None,
        "clearance_m": None,
        "width_at_point_m": None,
        "suggested_point": suggestion["point"] if suggestion else None,
        "suggested_radius_m": suggestion["radius"] if suggestion else None,
    }

    if chosen is not None:
        labels, distances = classify(union, [chosen])
        report["classification"] = labels[0]
        report["clearance_m"] = abs(float(distances[0]))
        if labels[0] == "inside":
            report["width_at_point_m"] = local_width(
                union, chosen, radius=-float(distances[0])
            )["width"]

    report["width"] = width_field(union, count=width_samples, seed=seed)
    return report


def findings(report: dict) -> list[preflight.Finding]:
    """The three questions, answered in the register the rest of the toolbox uses."""
    out = [_location_finding(report), _width_finding(report), _wall_finding(report)]
    return out


def _location_finding(report: dict) -> preflight.Finding:
    point = report["point"]
    if point is None:
        return preflight.Finding(
            "location_in_mesh", "fail",
            f"no point to validate: {report['directory']} has "
            + ("no location_in_mesh in patches.json"
               if report["manifest"] else "no patches.json")
            + " and none was passed",
            "snappyHexMesh is told which side of the surface the fluid is on by this "
            "point alone. Unvalidated, a mesh of the volume outside the part comes "
            "out and passes checkMesh.",
            "run with --suggest for a point at the deepest part of the domain, and "
            "write it into patches.json as location_in_mesh",
        )

    where = " ".join(f"{v:.6g}" for v in point)
    clearance = report["clearance_m"]
    source = report["point_source"]
    label = report["classification"]
    if label == "outside":
        return preflight.Finding(
            "location_in_mesh", "fail",
            f"{where} ({source}) is outside the surface, {clearance:.6g} m from the "
            f"nearest triangle",
            "snappyHexMesh keeps the region this point is in. Outside the surface, "
            "that is everything the part is not.",
            "run with --suggest and use the point it proposes",
        )
    if label == "on_surface":
        return preflight.Finding(
            "location_in_mesh", "fail",
            f"{where} ({source}) is {clearance:.6g} m from the nearest triangle, "
            f"inside the {report['band_m']:.3g} m band where the side is a rounding "
            "decision, so no side is claimed for it",
            "a point on the surface belongs to neither region, and which one snappy "
            "gives it is not a property of the geometry.",
            "move the point off the wall -- --suggest proposes the deepest point in "
            "the domain",
        )
    width = report["width_at_point_m"]
    detail = (f"{where} ({source}) is inside, {clearance:.6g} m from the nearest "
              f"triangle")
    if width:
        detail += f", in a domain {width:.6g} m wide there"
    if clearance < report["margin_m"]:
        return preflight.Finding(
            "location_in_mesh", "warn",
            detail + f"; the margin is {report['margin_m']:.6g} m "
            f"({MARGIN_FRACTION:g} of the {report['diagonal_m']:.6g} m diagonal)",
            "inside, but close enough to a wall that a coarse castellation may put "
            "the cell holding it on the other side.",
            "run with --suggest and compare: it proposes the deepest point in the "
            "domain",
        )
    return preflight.Finding(
        "location_in_mesh", "ok", detail,
        "the point is in the region snappyHexMesh will keep.",
    )


def _width_finding(report: dict) -> preflight.Finding:
    field = report["width"]
    if not field["samples"]:
        return preflight.Finding(
            "min_width", "skipped",
            f"no interior point found in {field['draws']} draws inside the bounding box",
            "either the surface encloses nothing, or what it encloses is too small a "
            "part of its own bounding box to find by sampling.",
            "check the union is closed with cad_audit.py before reading anything here",
        )
    if field["min_width"] is None:
        return preflight.Finding(
            "min_width", "skipped",
            f"{field['samples']} interior samples, every one of them in a converging "
            f"feature; narrowest sphere containing a sample was "
            f"{field['min_width_any']:.6g} m across",
            "in a corner the largest sphere holding a point shrinks to nothing as the "
            "point approaches the corner, so those widths measure the corner and not "
            "the domain. None of the samples landed anywhere else.",
            "raise --width-samples",
        )
    at = " ".join(f"{v:.6g}" for v in field["min_width_at"])
    return preflight.Finding(
        "min_width", "ok",
        f"narrowest of {field['qualified']} measured samples is "
        f"{field['min_width']:.6g} m, at {at} "
        f"({field['converging']} further samples set aside as corners)",
        "the diameter of the largest sphere that fits in the domain and holds the "
        "sample -- found, not proved: it is the narrowest place sampling reached.",
        "raise --width-samples to search harder for a narrower one",
    )


def _wall_finding(report: dict) -> preflight.Finding:
    field = report["width"]
    if report["shells"] < 2:
        return preflight.Finding(
            "min_wall_thickness", "skipped",
            f"the union is one connected shell of {report['triangles']:,} triangles",
            "wall thickness is the width of material between two surfaces, and one "
            "shell has no material between anything. If these surfaces bound the part "
            "rather than the fluid, min_width above is the part's thinnest section.",
        )
    if field["min_wall_thickness"] is None:
        return preflight.Finding(
            "min_wall_thickness", "skipped",
            f"{report['shells']} shells, but none of {field['samples']} interior "
            "samples sat between two of them",
            "the sampled region is bounded by one shell at a time, so nothing "
            "sampled was a wall.",
            "raise --width-samples",
        )
    at = " ".join(f"{v:.6g}" for v in field["min_wall_thickness_at"])
    return preflight.Finding(
        "min_wall_thickness", "ok",
        f"thinnest material between two of the {report['shells']} shells is "
        f"{field['min_wall_thickness']:.6g} m, at {at}",
        "the smallest gap a sphere resting on two different shells could grow to -- "
        "the number that predicts whether a boolean survives this geometry.",
    )


# -- output ------------------------------------------------------------------------


def envelope(report: dict, found: list[preflight.Finding]) -> dict:
    return {
        "script": SCRIPT,
        "ok": preflight.worst_status(found) != "fail",
        "findings": [f.as_dict() for f in found],
        "measured": report,
    }


def render(report: dict, found: list[preflight.Finding]) -> str:
    lines = [f"domain_probe  {report['directory']}"]
    add = lines.append
    add(f"  surface             {report['triangles']:,} triangles in "
        f"{len(report['files'])} file(s), {report['shells']} shell(s)")
    lower, upper = report["bounds"][:3], report["bounds"][3:]
    add("  bounds              "
        + " ".join(f"{lo:.6g}..{hi:.6g}" for lo, hi in zip(lower, upper))
        + f"  (diagonal {report['diagonal_m']:.6g} m)")
    add(f"  on-surface band     {report['band_m']:.3g} m "
        f"({BAND_FRACTION:g} of the diagonal)")
    if report["suggested_point"] is not None:
        where = " ".join(f"{v:.6g}" for v in report["suggested_point"])
        add(f"  suggested point     {where}  (deepest in the domain, "
            f"{report['suggested_radius_m']:.6g} m from any wall)")
    field = report["width"]
    add(f"  width samples       {field['samples']} interior points "
        f"from {field['draws']} draws")
    add("")
    for finding in found:
        add(f"  [{finding.status:>7}] {finding.check}")
        add(f"            {finding.measured}")
        if finding.meaning:
            add(f"            means: {finding.meaning}")
        if finding.repair:
            add(f"            try:   {finding.repair}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Is the locationInMesh point inside, and how wide is the domain there.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Local width is the diameter of the largest sphere that fits inside the\n"
            "domain and contains the point -- not twice the distance to the nearest\n"
            "wall, which is the same number only on the medial axis. A point within\n"
            "1e-6 of the diagonal from a triangle is reported on-surface with its\n"
            "distance rather than resolved to a side."
        ),
    )
    parser.add_argument("directory", type=Path,
                        help="a triSurface directory: one STL per patch")
    parser.add_argument("--point", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        default=None, help="the locationInMesh point to validate")
    parser.add_argument("--width-samples", type=int, default=WIDTH_SAMPLES,
                        help=f"interior points the width field is measured at "
                             f"(default {WIDTH_SAMPLES})")
    parser.add_argument("--suggest", action="store_true",
                        help="propose a point at the deepest part of the domain")
    parser.add_argument("--json", action="store_true", help="the envelope as JSON too")
    args = parser.parse_args(argv)

    if args.width_samples < 0:
        print("refused: --width-samples cannot be negative. Pass 0 to skip the width "
              "field, or a count of interior points to measure it at.", file=sys.stderr)
        return 2
    try:
        report = probe(args.directory, point=args.point,
                       width_samples=args.width_samples, suggest=args.suggest)
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        return 2

    found = findings(report)
    print(render(report, found))
    if args.json:
        print(json.dumps(envelope(report, found), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
