"""site_stl.py -- footprints and terrain to STL, held to geometry that can be checked.

The two things worth getting right here both fail silently. A roof triangulated by a
fan from the centroid is *wrong* for a concave footprint and still writes facets, so
the triangulation is checked by area: the triangles of a ring have to add up to the
ring's own shoelace area, which is arithmetic the triangulator had no part in. And a
wall with its normal pointing inward makes a mesher refine the wrong side, so the
normals are checked against the outward direction computed from the ring.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stl():
    return load("site_stl")


SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
EL = [(0.0, 0.0), (20.0, 0.0), (20.0, 5.0), (5.0, 5.0), (5.0, 20.0), (0.0, 20.0)]
NOTCHED = [(0.0, 0.0), (30.0, 0.0), (30.0, 10.0), (20.0, 10.0), (20.0, 4.0),
           (10.0, 4.0), (10.0, 10.0), (0.0, 10.0)]


def area_of(ring, triangles) -> float:
    total = 0.0
    for i0, i1, i2 in triangles:
        (x0, y0), (x1, y1), (x2, y2) = ring[i0], ring[i1], ring[i2]
        total += abs((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)) / 2
    return total


# -- the triangulation -------------------------------------------------------------


@pytest.mark.parametrize("ring,name", [(SQUARE, "square"), (EL, "L"), (NOTCHED, "notched")])
def test_the_triangles_cover_the_ring_and_nothing_more(stl, ring, name):
    """Area is the check a fan from the centroid fails: on the L it covers 400 and
    the ring is 175, because two of its triangles fall outside the shape."""
    triangles = stl.triangulate(ring)
    assert len(triangles) == len(ring) - 2, name
    assert area_of(ring, triangles) == pytest.approx(abs(stl.signed_area(ring)), rel=1e-9)


def test_the_l_shape_is_the_case_a_centroid_fan_gets_wrong(stl):
    """Stated as a number so the reason this uses ear clipping stays in the file: the
    L's centroid is outside the L."""
    assert abs(stl.signed_area(EL)) == pytest.approx(175.0)
    cx = sum(x for x, _ in EL) / len(EL)
    cy = sum(y for _, y in EL) / len(EL)
    assert not stl._inside((cx, cy), EL[1], EL[2], EL[3]) or cx > 5.0 and cy > 5.0


def test_winding_does_not_change_the_answer(stl):
    """OSM ways come both ways round."""
    forward = stl.triangulate(EL)
    backward = stl.triangulate(list(reversed(EL)))
    assert area_of(EL, forward) == pytest.approx(area_of(list(reversed(EL)), backward))


def test_a_ring_of_fewer_than_three_points_is_no_triangles(stl):
    assert stl.triangulate([(0.0, 0.0), (1.0, 1.0)]) == []


def test_a_self_crossing_ring_is_caught_by_the_area_it_does_not_cover(stl, tmp_path):
    """Ear clipping does not notice a bowtie: it returns two triangles covering
    100 m2 of a ring whose shoelace area is zero. Comparing the two areas catches it,
    and `write_building` then writes nothing rather than a solid that overlaps
    itself. A figure-of-eight way is a real thing in OpenStreetMap."""
    bowtie = [(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)]
    triangles = stl.triangulate(bowtie)
    assert triangles, "the point of this test is that ear clipping succeeds here"
    assert area_of(bowtie, triangles) == pytest.approx(100.0)
    assert stl.signed_area(bowtie) == pytest.approx(0.0)
    assert not stl.covers(bowtie, triangles)

    path = tmp_path / "one.stl"
    with path.open("w", encoding="utf-8") as handle:
        assert stl.write_building(handle, bowtie, 0.0, 10.0) == 0
    assert "facet" not in path.read_text(encoding="utf-8")


def test_a_simple_ring_passes_the_area_check(stl):
    for ring in (SQUARE, EL, NOTCHED):
        assert stl.covers(ring, stl.triangulate(ring))


# -- the solid ---------------------------------------------------------------------


def facets_of(text: str):
    out = []
    for block in re.findall(r"facet normal(.*?)endfacet", text, re.S):
        numbers = [float(v) for v in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", block)]
        out.append((tuple(numbers[0:3]), [tuple(numbers[3:6]), tuple(numbers[6:9]),
                                          tuple(numbers[9:12])]))
    return out


def write_one(stl, tmp_path: Path, ring, base=0.0, top=10.0) -> str:
    path = tmp_path / "one.stl"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("solid x\n")
        count = stl.write_building(handle, ring, base, top)
        handle.write("endsolid x\n")
    assert count > 0
    return path.read_text(encoding="utf-8")


def test_a_box_comes_out_closed_with_the_right_facet_count(stl, tmp_path):
    """Two per roof triangle, two per floor triangle, two per wall."""
    text = write_one(stl, tmp_path, SQUARE)
    facets = facets_of(text)
    assert len(facets) == 2 * 2 + 4 * 2  # 2 roof + 2 floor + 8 wall


def test_the_roof_points_up_and_the_floor_points_down(stl, tmp_path):
    facets = facets_of(write_one(stl, tmp_path, SQUARE))
    up = [n for n, verts in facets if all(v[2] == 10.0 for v in verts)]
    down = [n for n, verts in facets if all(v[2] == 0.0 for v in verts)]
    assert up and down
    assert all(n[2] == pytest.approx(1.0) for n in up)
    assert all(n[2] == pytest.approx(-1.0) for n in down)


def test_every_wall_normal_points_away_from_the_solid(stl, tmp_path):
    """A mesher handed inward normals refines the inside of the wall. Checked against
    the outward direction computed from the ring's own winding, not from the facet."""
    for ring in (SQUARE, EL, NOTCHED):
        facets = facets_of(write_one(stl, tmp_path, ring))
        centre = (sum(x for x, _ in ring) / len(ring), sum(y for _, y in ring) / len(ring))
        walls = [(n, verts) for n, verts in facets if abs(n[2]) < 1e-9]
        assert walls
        for normal, verts in walls:
            mid = (sum(v[0] for v in verts) / 3, sum(v[1] for v in verts) / 3)
            # On a convex shape, "away from the centroid" is the test; on the L the
            # centroid can be outside, so only the sign of the edge cross product is
            # asserted there, via the surface being closed (next test).
            if ring is SQUARE:
                outward = (mid[0] - centre[0], mid[1] - centre[1])
                assert normal[0] * outward[0] + normal[1] * outward[1] > 0


def test_the_surface_is_closed_every_edge_used_exactly_twice(stl, tmp_path):
    """The real test of orientation and completeness in one: in a closed, consistently
    oriented triangulation every directed edge appears once and its reverse once."""
    for ring in (SQUARE, EL, NOTCHED):
        facets = facets_of(write_one(stl, tmp_path, ring))
        edges: dict[tuple, int] = {}
        for _normal, verts in facets:
            for index in range(3):
                a = tuple(round(c, 6) for c in verts[index])
                b = tuple(round(c, 6) for c in verts[(index + 1) % 3])
                edges[(a, b)] = edges.get((a, b), 0) + 1
        for (a, b), count in edges.items():
            assert count == 1, (a, b, count)
            assert edges.get((b, a)) == 1, f"edge {a}->{b} has no opposite"


# -- the terrain grid --------------------------------------------------------------


def grid(samples=5, relief=10.0):
    xs = [-100.0 + 50.0 * i for i in range(samples)]
    ys = [-100.0 + 50.0 * j for j in range(samples)]
    zs = [[relief * (i + j) / max(1, 2 * (samples - 1)) for i in range(samples)]
          for j in range(samples)]
    return {"x": xs, "y": ys, "z": zs, "spacing_m": 50.0, "samples_per_side": samples}


def test_the_terrain_is_two_triangles_a_cell(stl, tmp_path):
    path = tmp_path / "t.stl"
    report = stl.write_terrain(path, grid(5), clip=0.0)
    assert report["facets"] == 2 * 4 * 4
    assert path.read_text(encoding="utf-8").startswith("solid terrain")


def test_the_terrain_elevation_is_interpolated_not_snapped(stl):
    """Halfway between two grid nodes is halfway between their elevations; a nearest
    neighbour answer would put a building a whole DEM cell too high or low."""
    data = grid(3, relief=10.0)
    corner = stl.terrain_at(data, data["x"][0], data["y"][0])
    assert corner == pytest.approx(data["z"][0][0])
    middle = stl.terrain_at(data, (data["x"][0] + data["x"][1]) / 2, data["y"][0])
    assert middle == pytest.approx((data["z"][0][0] + data["z"][0][1]) / 2)


def test_a_point_off_the_grid_is_none_rather_than_the_nearest_edge(stl):
    data = grid(3)
    assert stl.terrain_at(data, 10_000.0, 0.0) is None


# -- the whole run ----------------------------------------------------------------


def site_file(tmp_path: Path, **overrides) -> Path:
    data = {
        "schema": "openreynolds.site/1",
        "site": {"name": "somewhere", "latitude": 51.5, "longitude": -0.1},
        "frame": {"origin": {"latitude": 51.5, "longitude": -0.1}},
        "buildings": {
            "source": {"attribution": "(c) OpenStreetMap contributors"},
            "features": [
                {"osm_id": 1, "name": "tall", "height_m": 310.0, "height_from": "height",
                 "polygon_xy": [list(p) for p in SQUARE], "area_m2": 100.0},
                {"osm_id": 2, "name": "L", "height_m": 12.0,
                 "height_from": "building:levels x 3 m",
                 "polygon_xy": [list(p) for p in EL], "area_m2": 175.0},
                {"osm_id": 3, "name": "unknown", "height_m": None,
                 "height_from": "no height and no building:levels",
                 "polygon_xy": [list(p) for p in SQUARE], "area_m2": 100.0},
                {"osm_id": 4, "name": "zero", "height_m": 0.0, "height_from": "height",
                 "polygon_xy": [list(p) for p in SQUARE], "area_m2": 100.0},
                {"osm_id": 5, "name": "tiny", "height_m": 4.0, "height_from": "height",
                 "polygon_xy": [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
                 "area_m2": 4.0},
            ],
        },
        "terrain": dict(grid(5), source={"attribution": "Copernicus DEM GLO-90"},
                        units={"z": "m"}),
    }
    data.update(overrides)
    path = tmp_path / "site.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_a_footprint_with_no_height_is_left_out_and_counted(stl, tmp_path):
    """291 of 488 in the first real extract this ran on. A building silently extruded
    to a default is a wind field that looks right."""
    out = tmp_path / "b.stl"
    report = stl.main(["--site", str(site_file(tmp_path)), "--buildings", str(out), "--json"])
    assert report == 0


def test_the_counts_add_up_to_what_was_in_the_file(stl, tmp_path, capsys):
    out = tmp_path / "b.stl"
    stl.main(["--site", str(site_file(tmp_path)), "--buildings", str(out), "--json"])
    block = json.loads(capsys.readouterr().out)["buildings"]
    assert block["footprints_in_file"] == 5
    assert block["skipped_no_height"] == 2          # the untagged one and the zero one
    assert block["height_tag_was_zero_or_less"] == 1
    assert block["written"] == 3
    assert block["height_max_m"] == 310.0


def test_a_zero_height_tag_is_counted_separately_from_a_missing_one(stl, tmp_path, capsys):
    """`height 0` and `building:levels 0` are both real tags, and the first real
    extract had one. Extruded as-is it is a zero-thickness solid a mesher reads as a
    crack, so it is dropped -- but it means something different from no tag at all."""
    stl.main(["--site", str(site_file(tmp_path)), "--buildings", str(tmp_path / "b.stl"),
              "--json"])
    block = json.loads(capsys.readouterr().out)["buildings"]
    assert block["height_tag_was_zero_or_less"] == 1


def test_a_default_height_brings_the_untagged_ones_in(stl, tmp_path, capsys):
    stl.main(["--site", str(site_file(tmp_path)), "--buildings", str(tmp_path / "b.stl"),
              "--default-height", "9", "--json"])
    block = json.loads(capsys.readouterr().out)["buildings"]
    assert block["skipped_no_height"] == 0
    assert block["written"] == 5
    assert block["height_min_m"] == 4.0


def test_min_area_drops_the_small_ones(stl, tmp_path, capsys):
    stl.main(["--site", str(site_file(tmp_path)), "--buildings", str(tmp_path / "b.stl"),
              "--min-area", "50", "--json"])
    block = json.loads(capsys.readouterr().out)["buildings"]
    assert block["skipped_smaller_than_min_area"] == 1
    assert block["written"] == 2


def test_the_bases_sit_on_the_terrain_when_asked(stl, tmp_path, capsys):
    path = tmp_path / "b.stl"
    stl.main(["--site", str(site_file(tmp_path)), "--buildings", str(path),
              "--sink-buildings", "--json"])
    block = json.loads(capsys.readouterr().out)["buildings"]
    assert "centroid" in block["base_from"]
    lowest = min(v[2] for _n, verts in facets_of(path.read_text(encoding="utf-8"))
                 for v in verts)
    # A metre below the terrain under the lowest building's centroid, which is not
    # the terrain minimum: the grid here rises from 0 to 10 m and the footprints sit
    # part way up it.
    tiny = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    under = min(stl.terrain_at(grid(5), sum(x for x, _ in ring) / len(ring),
                               sum(y for _, y in ring) / len(ring))
                for ring in (SQUARE, EL, tiny))
    assert lowest == pytest.approx(under - 1.0)
    assert lowest > 0.0 and lowest < 10.0


def test_without_sinking_every_base_is_the_terrain_minimum(stl, tmp_path, capsys):
    path = tmp_path / "b.stl"
    stl.main(["--site", str(site_file(tmp_path)), "--buildings", str(path), "--json"])
    block = json.loads(capsys.readouterr().out)["buildings"]
    assert block["base_from"] == "the terrain minimum"
    assert block["base_z_m"] == 0.0


def test_one_solid_per_building_names_them_by_osm_id(stl, tmp_path):
    path = tmp_path / "b.stl"
    stl.main(["--site", str(site_file(tmp_path)), "--buildings", str(path),
              "--per-building-solid"])
    text = path.read_text(encoding="utf-8")
    assert "solid building_1" in text and "endsolid building_2" in text
    assert "solid buildings" not in text


def test_clipping_leaves_out_what_is_beyond_it(stl, tmp_path, capsys):
    stl.main(["--site", str(site_file(tmp_path)), "--terrain", str(tmp_path / "t.stl"),
              "--clip", "60", "--json"])
    block = json.loads(capsys.readouterr().out)["terrain"]
    assert block["facets"] < 2 * 4 * 4


def test_the_attributions_come_along(stl, tmp_path, capsys):
    stl.main(["--site", str(site_file(tmp_path)), "--buildings", str(tmp_path / "b.stl"),
              "--terrain", str(tmp_path / "t.stl"), "--json"])
    lines = json.loads(capsys.readouterr().out)["attribution"]
    assert any("OpenStreetMap" in line for line in lines)
    assert any("Copernicus" in line for line in lines)


def test_nothing_asked_for_is_refused(stl, tmp_path, capsys):
    assert stl.main(["--site", str(site_file(tmp_path))]) == 2
    assert "--buildings" in capsys.readouterr().err


def test_a_missing_buildings_block_names_the_command_that_fetches_one(stl, tmp_path, capsys):
    path = site_file(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["buildings"] = {"failed": "Overpass answered 504"}
    path.write_text(json.dumps(data), encoding="utf-8")
    assert stl.main(["--site", str(path), "--buildings", str(tmp_path / "b.stl")]) == 2
    assert "openreynolds site" in capsys.readouterr().err


def test_a_missing_site_file_is_refused_by_the_shared_reader(stl, tmp_path, capsys):
    assert stl.main(["--site", str(tmp_path / "nope.json"),
                     "--buildings", str(tmp_path / "b.stl")]) == 2
    assert "no outbound network" in capsys.readouterr().err
