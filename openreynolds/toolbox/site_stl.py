#!/usr/bin/env python3
"""Building footprints and terrain from a site file, as STL surfaces.

`openreynolds site` writes one JSON file on your own machine (the workspace has no
outbound network, so the fetch cannot happen in here). This turns two of its blocks
into geometry a mesher can read:

  * **buildings** -- each OpenStreetMap footprint extruded from its base to its
    height, as a closed solid: walls, a flat roof and a floor. The roof and floor
    are triangulated by ear clipping, so an L-shaped or notched footprint comes out
    as the shape it is rather than as a fan from a centroid that is outside it. Ear
    clipping handles a concave ring and not a ring with a hole; a building mapped
    with a courtyard is an OSM multipolygon relation, which the site file counts
    under `skipped_by_type` rather than carrying a ring for.
  * **terrain** -- the elevation grid as two triangles a cell.

Both come out in the site file's own frame: x east, y north, metres from the site
origin, z metres above the geoid as the DEM reports it. That frame is written into
the site file rather than recomputed here, so the footprints and the terrain land on
each other.

Heights: the site file records a height per footprint and where that number came
from (an OSM `height` tag, or `building:levels` times a storey). Footprints with
neither are **left out** unless `--default-height` says what to do with them, and
the count is reported either way -- in a dense European city that is often half of
them, and a building silently extruded to 3 m is a wind field that looks right.

    python3 site_stl.py --site site.json --buildings buildings.stl
    python3 site_stl.py --site site.json --buildings b.stl --terrain t.stl
    python3 site_stl.py --site site.json --buildings b.stl --default-height 12
    python3 site_stl.py --site site.json --buildings b.stl --clip 300 --min-area 50
    python3 site_stl.py --site site.json --terrain t.stl --sink-buildings --json

`--sink-buildings` sets each footprint's base from the terrain grid under its
centroid and extends it a metre below, so a building meets sloping ground instead of
floating over it. Without it every base sits at the terrain minimum, which is what a
flat-ground case wants.

Exit code is 0 whatever is found, and 2 when the inputs are refused.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import site_abl  # noqa: E402  (sibling script, not a package; it owns the site file reader)

Refused = site_abl.Refused


# -- triangulating a footprint ------------------------------------------------------
#
# Ear clipping, because a fan from the centroid is wrong for every footprint that is
# not convex and plenty are: a courtyard block, an L, a terrace with a notch. O(n^2)
# on a ring of a few dozen points is nothing.


def signed_area(ring: list[tuple[float, float]]) -> float:
    total = 0.0
    for index in range(len(ring)):
        x1, y1 = ring[index]
        x2, y2 = ring[(index + 1) % len(ring)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _inside(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float],
            c: tuple[float, float]) -> bool:
    def cross(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    d1, d2, d3 = cross(a, b, point), cross(b, c, point), cross(c, a, point)
    negative = d1 < 0 or d2 < 0 or d3 < 0
    positive = d1 > 0 or d2 > 0 or d3 > 0
    return not (negative and positive)


def triangulate(ring: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    """Indices into `ring`, counter-clockwise. Empty when the ring defeats it."""
    if len(ring) < 3:
        return []
    order = list(range(len(ring)))
    if signed_area(ring) < 0:
        order.reverse()
    triangles: list[tuple[int, int, int]] = []
    guard = 0
    while len(order) > 3 and guard < 4 * len(ring) ** 2:
        guard += 1
        clipped = False
        for position in range(len(order)):
            i0 = order[(position - 1) % len(order)]
            i1 = order[position]
            i2 = order[(position + 1) % len(order)]
            a, b, c = ring[i0], ring[i1], ring[i2]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 0:
                continue  # reflex or degenerate, not an ear
            if any(_inside(ring[other], a, b, c)
                   for other in order if other not in (i0, i1, i2)):
                continue
            triangles.append((i0, i1, i2))
            order.pop(position)
            clipped = True
            break
        if not clipped:
            return []  # self-intersecting or otherwise not a simple polygon
    if len(order) == 3:
        triangles.append((order[0], order[1], order[2]))
    return triangles


# -- writing STL -------------------------------------------------------------------


def facet(handle: Any, p1: tuple[float, float, float], p2: tuple[float, float, float],
          p3: tuple[float, float, float]) -> None:
    ux, uy, uz = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
    vx, vy, vz = p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    handle.write(f"  facet normal {nx / length:.6e} {ny / length:.6e} {nz / length:.6e}\n")
    handle.write("    outer loop\n")
    for point in (p1, p2, p3):
        handle.write(f"      vertex {point[0]:.6e} {point[1]:.6e} {point[2]:.6e}\n")
    handle.write("    endloop\n  endfacet\n")


def covers(ring: list[tuple[float, float]], triangles: list[tuple[int, int, int]]) -> bool:
    """Do the triangles add up to the ring's own area?

    Ear clipping does not detect a self-crossing ring: handed a bowtie it happily
    returns two triangles covering 100 m2 of a footprint whose shoelace area is
    zero. That was not caught until a test asked, and a figure-of-eight way is a
    real thing in OpenStreetMap. The areas are two independent calculations of the
    same number, so comparing them catches it for the cost of a sum.
    """
    total = 0.0
    for i0, i1, i2 in triangles:
        (x0, y0), (x1, y1), (x2, y2) = ring[i0], ring[i1], ring[i2]
        total += abs((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)) / 2
    want = abs(signed_area(ring))
    return abs(total - want) <= 1e-6 * max(want, 1.0)


def write_building(handle: Any, ring: list[tuple[float, float]], base: float,
                   top: float) -> int:
    """One closed solid: roof up, floor down, walls outward. Facet count.

    Zero when the ring is not a simple polygon, which the caller reports by OSM id
    rather than writing a solid that overlaps itself.
    """
    triangles = triangulate(ring)
    if not triangles or not covers(ring, triangles):
        return 0
    count = 0
    for i0, i1, i2 in triangles:
        facet(handle, (ring[i0][0], ring[i0][1], top), (ring[i1][0], ring[i1][1], top),
              (ring[i2][0], ring[i2][1], top))
        facet(handle, (ring[i0][0], ring[i0][1], base), (ring[i2][0], ring[i2][1], base),
              (ring[i1][0], ring[i1][1], base))
        count += 2
    # Walls in the ring's own order, reversed when the ring runs clockwise, so every
    # normal points away from the solid. A mesher that is handed inward normals
    # refines the wrong side of the wall.
    order = list(range(len(ring)))
    if signed_area(ring) < 0:
        order.reverse()
    for position in range(len(order)):
        a = ring[order[position]]
        b = ring[order[(position + 1) % len(order)]]
        facet(handle, (a[0], a[1], base), (b[0], b[1], base), (b[0], b[1], top))
        facet(handle, (a[0], a[1], base), (b[0], b[1], top), (a[0], a[1], top))
        count += 2
    return count


# -- the terrain grid --------------------------------------------------------------


def terrain_at(terrain: dict[str, Any], x: float, y: float) -> float | None:
    """Bilinear elevation, or None outside the grid."""
    xs, ys, zs = terrain["x"], terrain["y"], terrain["z"]
    if not (xs[0] <= x <= xs[-1] and ys[0] <= y <= ys[-1]):
        return None
    i = min(range(len(xs) - 1), key=lambda k: abs(xs[k] - x) if xs[k] <= x else 1e30)
    j = min(range(len(ys) - 1), key=lambda k: abs(ys[k] - y) if ys[k] <= y else 1e30)
    tx = (x - xs[i]) / (xs[i + 1] - xs[i]) if xs[i + 1] != xs[i] else 0.0
    ty = (y - ys[j]) / (ys[j + 1] - ys[j]) if ys[j + 1] != ys[j] else 0.0
    return ((1 - tx) * (1 - ty) * zs[j][i] + tx * (1 - ty) * zs[j][i + 1]
            + (1 - tx) * ty * zs[j + 1][i] + tx * ty * zs[j + 1][i + 1])


def write_terrain(path: Path, terrain: dict[str, Any], clip: float) -> dict[str, Any]:
    xs, ys, zs = terrain["x"], terrain["y"], terrain["z"]
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write("solid terrain\n")
        for j in range(len(ys) - 1):
            for i in range(len(xs) - 1):
                if clip and (abs(xs[i]) > clip or abs(ys[j]) > clip):
                    continue
                p00 = (xs[i], ys[j], zs[j][i])
                p10 = (xs[i + 1], ys[j], zs[j][i + 1])
                p11 = (xs[i + 1], ys[j + 1], zs[j + 1][i + 1])
                p01 = (xs[i], ys[j + 1], zs[j + 1][i])
                facet(handle, p00, p10, p11)
                facet(handle, p00, p11, p01)
                count += 2
        handle.write("endsolid terrain\n")
    flat = [v for row in zs for v in row]
    return {"facets": count, "z_min": min(flat), "z_max": max(flat),
            "spacing_m": terrain.get("spacing_m")}


# -- the buildings -----------------------------------------------------------------


def write_buildings(path: Path, buildings: dict[str, Any], terrain: dict[str, Any] | None,
                    default_height: float, min_area: float, clip: float,
                    sink: bool, per_solid: bool) -> dict[str, Any]:
    features = buildings.get("features") or []
    base_floor = 0.0
    if terrain:
        base_floor = min(v for row in terrain["z"] for v in row)

    kept: list[dict[str, Any]] = []
    no_height = 0
    zero_height = 0
    too_small = 0
    outside = 0
    untriangulable: list[Any] = []
    for feature in features:
        height = feature.get("height_m")
        # A height of zero is a real tag in OSM -- `building:levels 0` on an
        # underground structure, `height 0` on a mapping slip -- and it was in the
        # first real extract this ran on, where it extruded a footprint into a
        # zero-thickness solid that a mesher would read as a crack. Counted as its
        # own thing rather than folded into "no height", because they mean different
        # things about the data.
        if height is not None and height <= 0:
            zero_height += 1
            height = None
        if height is None:
            if default_height <= 0:
                no_height += 1
                continue
            height = default_height
        if feature.get("area_m2", 0.0) < min_area:
            too_small += 1
            continue
        ring = [(float(x), float(y)) for x, y in feature["polygon_xy"]]
        if clip and all(abs(x) > clip or abs(y) > clip for x, y in ring):
            outside += 1
            continue
        kept.append({"feature": feature, "ring": ring, "height": float(height)})

    facets = 0
    with path.open("w", encoding="utf-8") as handle:
        if not per_solid:
            handle.write("solid buildings\n")
        for item in kept:
            ring, feature = item["ring"], item["feature"]
            base = base_floor
            if sink and terrain:
                cx = sum(x for x, _ in ring) / len(ring)
                cy = sum(y for _, y in ring) / len(ring)
                ground = terrain_at(terrain, cx, cy)
                if ground is not None:
                    base = ground - 1.0
            name = f"building_{feature.get('osm_id')}"
            if per_solid:
                handle.write(f"solid {name}\n")
            written = write_building(handle, ring, base, base + item["height"])
            if per_solid:
                handle.write(f"endsolid {name}\n")
            if not written:
                untriangulable.append(feature.get("osm_id"))
            facets += written
        if not per_solid:
            handle.write("endsolid buildings\n")

    heights = [item["height"] for item in kept]
    return {
        "footprints_in_file": len(features),
        "written": len(kept) - len(untriangulable),
        "facets": facets,
        "skipped_no_height": no_height,
        "height_tag_was_zero_or_less": zero_height,
        "skipped_smaller_than_min_area": too_small,
        "skipped_outside_clip": outside,
        "skipped_not_a_simple_polygon": untriangulable,
        "height_min_m": min(heights) if heights else None,
        "height_max_m": max(heights) if heights else None,
        "base_z_m": base_floor,
        "base_from": "terrain under each centroid, less 1 m" if sink and terrain
                     else "the terrain minimum" if terrain else "z = 0",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Building footprints and terrain from a site file, as STL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--site", required=True, help="A site.json from `openreynolds site`.")
    parser.add_argument("--buildings", type=Path, default=None, help="Where to write the footprints.")
    parser.add_argument("--terrain", type=Path, default=None, help="Where to write the elevation grid.")
    parser.add_argument("--default-height", type=float, default=0.0,
                        help="Height for footprints with no height tag. 0 leaves them out.")
    parser.add_argument("--min-area", type=float, default=0.0,
                        help="Leave out footprints smaller than this, in m^2.")
    parser.add_argument("--clip", type=float, default=0.0,
                        help="Keep only what is within this many metres of the origin.")
    parser.add_argument("--sink-buildings", dest="sink", action="store_true",
                        help="Set each base from the terrain under its centroid.")
    parser.add_argument("--per-building-solid", dest="per_solid", action="store_true",
                        help="One named STL solid per footprint instead of one for all.")
    parser.add_argument("--json", dest="as_json", action="store_true", help="One JSON object.")
    args = parser.parse_args(argv)

    try:
        if not (args.buildings or args.terrain):
            raise Refused("refused: nothing to write -- pass --buildings, --terrain or both")
        site = site_abl.load_site(Path(args.site))
        terrain = site.get("terrain") or None
        if terrain and terrain.get("failed"):
            terrain = None
        buildings = site.get("buildings") or None
        if buildings and buildings.get("failed"):
            buildings = None

        report: dict[str, Any] = {"site": site.get("site"), "frame": site.get("frame"),
                                  "attribution": []}
        for key in ("buildings", "terrain"):
            block = site.get(key) or {}
            line = ((block.get("source") or {}).get("attribution") or "").strip()
            if line and line not in report["attribution"]:
                report["attribution"].append(line)

        if args.buildings:
            if not buildings:
                raise Refused(
                    "refused: the site file has no usable buildings block. "
                    "`openreynolds site --lat ... --lon ...` fetches one; "
                    "--no-buildings on that command is what leaves it out."
                )
            report["buildings"] = write_buildings(
                args.buildings, buildings, terrain, args.default_height, args.min_area,
                args.clip, args.sink, args.per_solid)
            report["buildings"]["path"] = args.buildings.as_posix()
        if args.terrain:
            if not terrain:
                raise Refused(
                    "refused: the site file has no usable terrain block. "
                    "`openreynolds site --lat ... --lon ...` fetches one; "
                    "--no-terrain on that command is what leaves it out."
                )
            report["terrain"] = write_terrain(args.terrain, terrain, args.clip)
            report["terrain"]["path"] = args.terrain.as_posix()
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(report, indent=2))
        return 0

    for key in ("buildings", "terrain"):
        block = report.get(key)
        if not block:
            continue
        print(f"{key}: {block['path']}")
        for name, value in block.items():
            if name != "path":
                print(f"  {name:<32} {value}")
    for line in report["attribution"]:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
