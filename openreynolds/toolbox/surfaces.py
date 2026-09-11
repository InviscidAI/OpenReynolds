"""The triangle machinery three other scripts would otherwise write three times.

A library, not a command. There is nothing to run:

    python3 -c "import surfaces"   # and see the docstrings; it has no CLI

`cad_audit.py`, `domain_probe.py` and `cad_convert.py` reach for it the way every
toolbox script already reaches for `preflight.py`:

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import surfaces  # noqa: E402  (sibling script, not a package)

What it holds is what `preflight.py` does not: a spatial index over an (n, 3, 3)
triangle array, the welding rule spelled once, triangle-triangle intersection, ray
casting with the degenerate hit handled where the caller cannot forget it, and signed
distance with the sign fixed.

**The contract the rest of it rests on.** `candidates_near`, `candidates_along` and
`candidates_overlapping` may over-return and must never under-return. A false positive
costs an exact test on one more triangle; a false negative is a missed intersection, a
missed crossing or a wrong sign, and every one of those is silent. Everything the index
does is therefore padded outwards, and `tests/test_toolbox_surfaces.py` asserts each
query's result is a superset of an exhaustive scan's rather than equal to it.

**The weld rule is preflight's, not a second one.** `preflight.surface_topology()`
rounds `flat / span` to nine decimals and takes `np.unique(..., axis=0)` -- an STL
stores every corner independently, so without the weld every edge looks open. `weld()`
does exactly that and returns the corner indices and the unique vertices, so the two
never drift apart. The test pins `weld()`'s vertex count against the number preflight
derives inside itself.

**Negative is inside.** `signed_distance` fixes that convention for every caller, and
none of them re-check it.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "Index",
    "weld",
    "tri_tri_intersect",
    "ray_hits",
    "signed_distance",
    "WELD_DECIMALS",
    "RECAST_LIMIT",
]


WELD_DECIMALS = 9
"""Decimal places the coordinates are rounded to after being divided by the surface's
own span, which is `preflight.surface_topology()`'s rule and deliberately not a new one."""

RECAST_LIMIT = 8
"""How many perturbed directions `ray_hits` will try before it gives up and returns the
count it has. A ray through a shared edge or a vertex is the case it exists for: the two
triangles either both count or neither does, and both answers are wrong."""

_DEFAULT_RAY = np.array([0.5773502691896258, 0.4211081369, 0.7012332145])
"""Nothing axis-aligned and nothing rational, so a tessellation's own planes are not
the first thing a cast runs into. It is only a starting direction; the re-cast below
is what actually makes the count safe."""


# -- the index ---------------------------------------------------------------------


class Index:
    """A spatial index over an (n, 3, 3) triangle array.

    A bounding-volume hierarchy over the triangles' axis-aligned boxes, split at the
    median of the widest centroid axis. Every box is padded outwards before it is
    tested, because the whole value of the thing is that it never drops a triangle the
    exact test would have found.

    The ray counters -- `recasts`, `last_recasts`, `last_direction` -- are how a caller
    or a test can see that `ray_hits` re-cast rather than got lucky.
    """

    def __init__(self, triangles: np.ndarray, leaf: int = 16) -> None:
        tris = np.ascontiguousarray(np.asarray(triangles, dtype=float))
        if tris.ndim != 3 or tris.shape[1:] != (3, 3):
            raise ValueError(f"triangles must be (n, 3, 3), got {tris.shape}")
        self.triangles = tris
        self.leaf = max(1, int(leaf))

        self.lo = tris.min(axis=1) if len(tris) else np.zeros((0, 3))
        self.hi = tris.max(axis=1) if len(tris) else np.zeros((0, 3))
        if len(tris):
            span = float(np.max(self.hi.max(axis=0) - self.lo.min(axis=0)))
        else:
            span = 0.0
        self.span = span if span > 0 else 1.0
        self.eps = self.span * 1e-9
        """The pad. Nine decimals is the weld's resolution, so two coordinates closer
        than this were already one vertex as far as the topology is concerned."""

        self.recasts = 0
        """Every degenerate re-cast this index has ever done, cumulative."""
        self.last_recasts = 0
        """Re-casts the most recent `ray_hits` needed. 0 means the first cast was clean."""
        self.last_direction: np.ndarray | None = None
        """The direction the most recent `ray_hits` actually counted along."""

        self._build()

    # -- construction ---------------------------------------------------------

    def _build(self) -> None:
        n = len(self.triangles)
        self._order = np.arange(n, dtype=np.int64)
        self._lo: list[tuple[float, float, float]] = []
        self._hi: list[tuple[float, float, float]] = []
        self._left: list[int] = []
        self._right: list[int] = []
        self._start: list[int] = []
        self._count: list[int] = []
        if n == 0:
            self._lo.append((0.0, 0.0, 0.0))
            self._hi.append((0.0, 0.0, 0.0))
            self._left.append(-1)
            self._right.append(-1)
            self._start.append(0)
            self._count.append(0)
            return
        centroid = self.triangles.mean(axis=1)
        self._build_node(0, n, centroid)

    def _build_node(self, start: int, count: int, centroid: np.ndarray) -> int:
        ids = self._order[start:start + count]
        lo = self.lo[ids].min(axis=0)
        hi = self.hi[ids].max(axis=0)
        node = len(self._lo)
        self._lo.append((float(lo[0]), float(lo[1]), float(lo[2])))
        self._hi.append((float(hi[0]), float(hi[1]), float(hi[2])))
        self._left.append(-1)
        self._right.append(-1)
        self._start.append(start)
        self._count.append(count)
        if count <= self.leaf:
            return node

        centres = centroid[ids]
        widths = centres.max(axis=0) - centres.min(axis=0)
        axis = int(np.argmax(widths))
        mid = count // 2
        if widths[axis] > 0:
            # A median split, not a spatial one: an equal-sized split keeps the tree
            # shallow whatever the triangles are doing, and correctness never depended
            # on where the plane went.
            order = np.argpartition(centres[:, axis], mid)
            self._order[start:start + count] = ids[order]
        left = self._build_node(start, mid, centroid)
        right = self._build_node(start + mid, count - mid, centroid)
        self._left[node] = left
        self._right[node] = right
        return node

    # -- the three queries ----------------------------------------------------

    def candidates_near(self, point, radius: float) -> np.ndarray:
        """Triangle ids that may lie within `radius` of `point`. A superset, always.

        Every triangle whose true distance to the point is at most `radius` is in the
        result; triangles further away may be too.
        """
        if len(self.triangles) == 0:
            return np.zeros(0, dtype=np.int64)
        p = np.asarray(point, dtype=float).reshape(3)
        reach = abs(float(radius)) + self.eps
        px, py, pz = float(p[0]), float(p[1]), float(p[2])
        reach2 = reach * reach
        ranges: list[tuple[int, int]] = []
        stack = [0]
        while stack:
            node = stack.pop()
            lo = self._lo[node]
            hi = self._hi[node]
            dx = px - hi[0] if px > hi[0] else (lo[0] - px if px < lo[0] else 0.0)
            dy = py - hi[1] if py > hi[1] else (lo[1] - py if py < lo[1] else 0.0)
            dz = pz - hi[2] if pz > hi[2] else (lo[2] - pz if pz < lo[2] else 0.0)
            if dx * dx + dy * dy + dz * dz > reach2:
                continue
            left = self._left[node]
            if left < 0:
                ranges.append((self._start[node], self._count[node]))
            else:
                stack.append(left)
                stack.append(self._right[node])
        ids = self._gather(ranges)
        if len(ids) == 0:
            return ids
        # One cheap per-triangle box test, for the same reason as the node test: it
        # can only remove triangles the exact test would have rejected anyway.
        lo = self.lo[ids]
        hi = self.hi[ids]
        delta = np.maximum(np.maximum(lo - p, p - hi), 0.0)
        keep = (delta * delta).sum(axis=1) <= reach2
        return ids[keep]

    def candidates_along(self, origin, direction) -> np.ndarray:
        """Triangle ids the ray from `origin` along `direction` may cross.

        A ray, not a line: only `t >= 0` counts. A superset of the triangles the ray
        actually meets, and a superset of the triangles whose boxes it meets.
        """
        if len(self.triangles) == 0:
            return np.zeros(0, dtype=np.int64)
        o = np.asarray(origin, dtype=float).reshape(3)
        d = _unit(np.asarray(direction, dtype=float).reshape(3))
        inv = np.where(np.abs(d) > 1e-30, 1.0 / np.where(d == 0, 1.0, d),
                       np.where(d < 0, -1e30, 1e30))
        ox, oy, oz = float(o[0]), float(o[1]), float(o[2])
        ix, iy, iz = float(inv[0]), float(inv[1]), float(inv[2])
        pad = self.eps
        ranges: list[tuple[int, int]] = []
        stack = [0]
        while stack:
            node = stack.pop()
            lo = self._lo[node]
            hi = self._hi[node]
            t0 = (lo[0] - pad - ox) * ix
            t1 = (hi[0] + pad - ox) * ix
            tmin = t0 if t0 < t1 else t1
            tmax = t1 if t0 < t1 else t0
            t0 = (lo[1] - pad - oy) * iy
            t1 = (hi[1] + pad - oy) * iy
            lower = t0 if t0 < t1 else t1
            upper = t1 if t0 < t1 else t0
            if lower > tmin:
                tmin = lower
            if upper < tmax:
                tmax = upper
            t0 = (lo[2] - pad - oz) * iz
            t1 = (hi[2] + pad - oz) * iz
            lower = t0 if t0 < t1 else t1
            upper = t1 if t0 < t1 else t0
            if lower > tmin:
                tmin = lower
            if upper < tmax:
                tmax = upper
            if tmin < 0.0:
                tmin = 0.0
            if tmax < tmin:
                continue
            left = self._left[node]
            if left < 0:
                ranges.append((self._start[node], self._count[node]))
            else:
                stack.append(left)
                stack.append(self._right[node])
        return self._gather(ranges)

    def candidates_overlapping(self, tri_id: int) -> np.ndarray:
        """Triangle ids whose boxes overlap triangle `tri_id`'s, **including it**.

        Self is kept rather than dropped: a caller that wants pairs filters on the id
        it already has, and a query that silently removed one triangle would be the
        first step towards a query that silently removes another.
        """
        if len(self.triangles) == 0:
            return np.zeros(0, dtype=np.int64)
        tri_id = int(tri_id)
        return self._box_query(self.lo[tri_id], self.hi[tri_id])

    # -- shared plumbing ------------------------------------------------------

    def _box_query(self, lo_in, hi_in) -> np.ndarray:
        pad = self.eps
        qlo = np.asarray(lo_in, dtype=float).reshape(3) - pad
        qhi = np.asarray(hi_in, dtype=float).reshape(3) + pad
        ql = (float(qlo[0]), float(qlo[1]), float(qlo[2]))
        qh = (float(qhi[0]), float(qhi[1]), float(qhi[2]))
        ranges: list[tuple[int, int]] = []
        stack = [0]
        while stack:
            node = stack.pop()
            lo = self._lo[node]
            hi = self._hi[node]
            if (hi[0] < ql[0] or lo[0] > qh[0]
                    or hi[1] < ql[1] or lo[1] > qh[1]
                    or hi[2] < ql[2] or lo[2] > qh[2]):
                continue
            left = self._left[node]
            if left < 0:
                ranges.append((self._start[node], self._count[node]))
            else:
                stack.append(left)
                stack.append(self._right[node])
        ids = self._gather(ranges)
        if len(ids) == 0:
            return ids
        keep = np.all((self.hi[ids] >= qlo) & (self.lo[ids] <= qhi), axis=1)
        return ids[keep]

    def _gather(self, ranges: list[tuple[int, int]]) -> np.ndarray:
        if not ranges:
            return np.zeros(0, dtype=np.int64)
        parts = [self._order[start:start + count] for start, count in ranges if count]
        if not parts:
            return np.zeros(0, dtype=np.int64)
        ids = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return np.sort(ids)

    # -- nearest distance, which the index is what makes affordable -----------

    def _nearest(self, point: np.ndarray) -> float:
        """The exact distance from `point` to the nearest triangle."""
        if len(self.triangles) == 0:
            return float("inf")
        px, py, pz = float(point[0]), float(point[1]), float(point[2])

        def box_distance(node: int) -> float:
            lo = self._lo[node]
            hi = self._hi[node]
            dx = px - hi[0] if px > hi[0] else (lo[0] - px if px < lo[0] else 0.0)
            dy = py - hi[1] if py > hi[1] else (lo[1] - py if py < lo[1] else 0.0)
            dz = pz - hi[2] if pz > hi[2] else (lo[2] - pz if pz < lo[2] else 0.0)
            return dx * dx + dy * dy + dz * dz

        best = float("inf")
        stack = [(box_distance(0), 0)]
        while stack:
            bound, node = stack.pop()
            if bound > best:
                continue
            left = self._left[node]
            if left < 0:
                start, count = self._start[node], self._count[node]
                if count:
                    ids = self._order[start:start + count]
                    here = float(
                        (_point_triangle_distances(point, self.triangles[ids]) ** 2).min()
                    )
                    if here < best:
                        best = here
                continue
            right = self._right[node]
            near_d, far_d = box_distance(left), box_distance(right)
            near, far = left, right
            if far_d < near_d:
                near, far = far, near
                near_d, far_d = far_d, near_d
            # Nearer child last, so it is popped first and sets a tight bound before
            # the other one is looked at.
            stack.append((far_d, far))
            stack.append((near_d, near))
        return float(np.sqrt(best))


# -- welding -----------------------------------------------------------------------


def weld(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(corner indices, unique vertices), on preflight's rounding rule.

    `preflight.surface_topology()` divides every coordinate by the surface's own span,
    rounds to nine decimals and takes the unique rows; the corner indices it gets back
    are what its edge bookkeeping runs on. This is that, returned instead of consumed,
    so `cad_audit.py` and `domain_probe.py` count edges on the same vertices preflight
    does rather than on a second rule that is nearly the same.

    The vertices come back in the surface's own units -- the scaling is how the
    rounding is made relative, not a change of coordinates.
    """
    tris = np.asarray(triangles, dtype=float)
    if tris.size == 0:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3), dtype=float)
    if tris.ndim != 3 or tris.shape[1:] != (3, 3):
        raise ValueError(f"triangles must be (n, 3, 3), got {tris.shape}")
    flat = tris.reshape(-1, 3)
    span = float(np.max(flat.max(axis=0) - flat.min(axis=0)))
    scale = span if span > 0 else 1.0
    welded = np.round(flat / scale, WELD_DECIMALS)
    unique, index = np.unique(welded, axis=0, return_inverse=True)
    corners = np.asarray(index, dtype=np.int64).reshape(-1, 3)
    return corners, unique * scale


# -- triangle against triangle -----------------------------------------------------


def tri_tri_intersect(a: np.ndarray, b: np.ndarray) -> bool:
    """Do two triangles share a point? Separating-axis, and it counts touching.

    Twenty-three candidate axes rather than the textbook eleven. The eleven are
    complete for two triangles in general position; they are *not* complete when the
    triangles are coplanar, because then every edge-edge cross product collapses onto
    the shared normal and nothing separates. Adding each edge crossed with each normal
    fixes that, and adding axes is always safe in one direction: a separating axis
    proves disjointness, so an axis that never separates anything costs a dot product
    and can never turn an intersection into a miss.

    Touching counts as intersecting -- two triangles that share an edge have no strictly
    separating axis. That is the right answer for a welded surface, where sharing an
    edge is what being a surface means, and callers filter neighbours by index.
    """
    tri_a = np.asarray(a, dtype=float).reshape(3, 3)
    tri_b = np.asarray(b, dtype=float).reshape(3, 3)
    edges_a = tri_a[[1, 2, 0]] - tri_a
    edges_b = tri_b[[1, 2, 0]] - tri_b
    normal_a = np.cross(edges_a[0], -edges_a[2])
    normal_b = np.cross(edges_b[0], -edges_b[2])

    axes = [normal_a, normal_b]
    for ea in edges_a:
        for eb in edges_b:
            axes.append(np.cross(ea, eb))
    for edge in (*edges_a, *edges_b):
        axes.append(np.cross(edge, normal_a))
        axes.append(np.cross(edge, normal_b))

    scale = float(max(np.abs(tri_a).max(), np.abs(tri_b).max(), 1e-300))
    floor = (scale * 1e-12) ** 2
    for axis in axes:
        length2 = float(axis @ axis)
        if length2 <= floor:
            continue  # a degenerate axis separates nothing; it is not evidence
        pa = tri_a @ axis
        pb = tri_b @ axis
        if pa.max() < pb.min() or pb.max() < pa.min():
            return False
    return True


# -- casting a ray -----------------------------------------------------------------


def ray_hits(index: Index, origin, direction) -> int:
    """Crossings, with the degenerate-hit re-cast handled inside.

    A ray that passes exactly along a shared edge, or exactly through a vertex, hits
    two triangles (or a whole fan) at one point. Counting both crossings and counting
    neither are both wrong, and which one a floating-point test lands on is not
    something a caller can steer. So the cast is retried along a perturbed direction
    until no hit is near a triangle's boundary -- the parity of the crossing count does
    not depend on the direction, so any clean direction answers the original question.

    What happened is left on the index: `last_recasts` is how many perturbed directions
    this call needed, `last_direction` is the one it finally counted along, and
    `recasts` accumulates over the index's life. `domain_probe.py` reads the parity of
    this and must not re-solve the degeneracy itself.
    """
    origin = np.asarray(origin, dtype=float).reshape(3)
    base = _unit(np.asarray(direction, dtype=float).reshape(3))
    index.last_recasts = 0
    index.last_direction = base
    if len(index.triangles) == 0:
        return 0

    # Deterministic: the same ray answers the same way in two runs, which is what makes
    # a disagreement between runs a real disagreement.
    noise = np.random.default_rng(20260911).standard_normal((RECAST_LIMIT, 3))
    count, degenerate = _count_crossings(index, origin, base)
    attempt = 0
    while degenerate and attempt < RECAST_LIMIT:
        nudge = 1e-3 * (2.0 ** attempt)
        tried = _unit(base + nudge * noise[attempt])
        attempt += 1
        index.last_recasts = attempt
        index.last_direction = tried
        index.recasts += 1
        count, degenerate = _count_crossings(index, origin, tried)
    return count


def _count_crossings(index: Index, origin: np.ndarray, direction: np.ndarray):
    """(crossings, whether any hit was too close to call), Moller-Trumbore."""
    ids = index.candidates_along(origin, direction)
    if len(ids) == 0:
        return 0, False
    tris = index.triangles[ids]
    v0 = tris[:, 0]
    e1 = tris[:, 1] - v0
    e2 = tris[:, 2] - v0
    pvec = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    size = np.sqrt(np.einsum("ij,ij->i", e1, e1) * np.einsum("ij,ij->i", e2, e2))
    parallel = np.abs(det) <= 1e-9 * np.maximum(size, 1e-300)

    safe = np.where(parallel, 1.0, det)
    tvec = origin - v0
    u = np.einsum("ij,ij->i", tvec, pvec) / safe
    qvec = np.cross(tvec, e1)
    v = np.einsum("j,ij->i", direction, qvec) / safe
    t = np.einsum("ij,ij->i", e2, qvec) / safe

    bary_tol = 1e-9
    length_tol = index.eps
    hit = (~parallel) & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > length_tol)

    # Too close to call, in the three ways it happens: the ray grazes a triangle's
    # edge or corner, the ray lies in a triangle's plane and near it, or the origin
    # is sitting on the surface. The first two a re-cast fixes.
    near_edge = (
        (~parallel)
        & (np.abs(t) > length_tol)
        & (np.minimum(np.minimum(u, v), 1.0 - u - v) > -bary_tol)
        & (np.minimum(np.minimum(np.abs(u), np.abs(v)), np.abs(1.0 - u - v)) < bary_tol)
    )
    normal = np.cross(e1, e2)
    nlen = np.sqrt(np.einsum("ij,ij->i", normal, normal))
    plane_gap = np.abs(np.einsum("ij,ij->i", tvec, normal)) / np.maximum(nlen, 1e-300)
    in_plane = parallel & (plane_gap <= length_tol)
    on_surface = (~parallel) & (np.abs(t) <= length_tol) & (u >= -bary_tol) \
        & (v >= -bary_tol) & (u + v <= 1.0 + bary_tol)
    degenerate = bool(near_edge.any() or in_plane.any() or on_surface.any())
    return int(hit.sum()), degenerate


# -- signed distance ---------------------------------------------------------------


def signed_distance(index: Index, points: np.ndarray) -> np.ndarray:
    """Negative inside. The sign convention every caller depends on.

    The magnitude is the exact distance to the nearest triangle, found through the
    index rather than by scanning; the sign is the parity of `ray_hits` from the point,
    odd being inside. Both halves are only meaningful on a closed surface -- an open
    one has no inside, and the parity will say so by disagreeing with itself between
    nearby points rather than by raising.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    out = np.empty(len(pts), dtype=float)
    for i, point in enumerate(pts):
        distance = index._nearest(point)
        inside = ray_hits(index, point, _DEFAULT_RAY) % 2 == 1
        out[i] = -distance if inside else distance
    return out


# -- the small shared arithmetic ---------------------------------------------------


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.sqrt(vector @ vector))
    if length <= 0:
        raise ValueError("direction has no length")
    return vector / length


def _point_triangle_distances(point: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Exact distance from one point to each of many triangles.

    The projection onto the plane when it lands inside the triangle, the nearest point
    on one of the three edges otherwise -- and the minimum of the two regardless, so a
    degenerate triangle with no plane still gets the right answer from its edges.
    """
    p = np.asarray(point, dtype=float).reshape(3)
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    best = np.minimum(
        np.minimum(_segment_distances(p, v0, v1), _segment_distances(p, v1, v2)),
        _segment_distances(p, v2, v0),
    )
    normal = np.cross(v1 - v0, v2 - v0)
    nn = np.einsum("ij,ij->i", normal, normal)
    usable = nn > 0
    if not usable.any():
        return best
    safe = np.where(usable, nn, 1.0)
    w = p - v0
    height = np.einsum("ij,ij->i", w, normal) / safe
    proj = p - height[:, None] * normal
    a = np.einsum("ij,ij->i", np.cross(v1 - proj, v2 - proj), normal) / safe
    b = np.einsum("ij,ij->i", np.cross(v2 - proj, v0 - proj), normal) / safe
    c = 1.0 - a - b
    inside = usable & (a >= 0.0) & (b >= 0.0) & (c >= 0.0)
    plane = np.abs(height) * np.sqrt(nn)
    return np.where(inside, np.minimum(best, plane), best)


def _segment_distances(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denominator = np.einsum("ij,ij->i", ab, ab)
    t = np.where(
        denominator > 0,
        np.einsum("ij,ij->i", p - a, ab) / np.where(denominator > 0, denominator, 1.0),
        0.0,
    )
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[:, None] * ab
    delta = p - closest
    return np.sqrt(np.einsum("ij,ij->i", delta, delta))
