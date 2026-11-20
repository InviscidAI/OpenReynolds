"""`openreynolds site` -- the fetchers that run on the person's own machine.

Every request here is answered by a `MockTransport`, so the suite stays offline;
what the real APIs answer was measured once, by hand, and the fixtures below are
trimmed copies of those answers rather than invented shapes. Two of them are
load-bearing:

  * the ERA5 archive answers for the centre of its own grid cell and **not** for the
    coordinates asked for. The real answer for 51.5045, -0.0865 came back as
    51.493847, -0.1630249, 5.4 km away. A file that did not record that would let a
    study claim street-level wind it never had.
  * an OSM `way` closes itself by repeating its first node, so the ring Overpass
    hands over has a duplicate point in it. A polygon that keeps it has a
    zero-length edge, which is a degenerate facet once it is extruded.
"""

from __future__ import annotations

import json
import math

import httpx2 as httpx
import pytest

from openreynolds import site
from openreynolds.site import Unreachable


# -- the frame ---------------------------------------------------------------------


def test_metres_per_degree_matches_the_closed_form_wgs84_radii():
    """At 51.5 degrees north the parallel is 38% shorter than the meridian, and a
    footprint projected with one constant for both comes out stretched by that much.

    The series in `local_frame` is checked against the closed form computed here
    from the WGS84 ellipsoid, which is a different formula and not a second copy of
    the same one: 69443.3 m per degree of longitude and 111257.8 per degree of
    latitude at this latitude.
    """
    phi = math.radians(51.5)
    a, e2 = 6378137.0, 0.00669437999014
    w = math.sqrt(1 - e2 * math.sin(phi) ** 2)
    per_lon = math.radians(1) * a * math.cos(phi) / w
    per_lat = math.radians(1) * a * (1 - e2) / w ** 3

    frame = site.local_frame(51.5, -0.1)
    assert frame["metres_per_degree_longitude"] == pytest.approx(per_lon, rel=1e-4)
    assert frame["metres_per_degree_latitude"] == pytest.approx(per_lat, rel=1e-4)
    assert frame["metres_per_degree_longitude"] < 0.63 * frame["metres_per_degree_latitude"]


def test_the_origin_is_the_origin_and_east_is_positive_x():
    frame = site.local_frame(51.5, -0.1)
    assert site.to_xy(frame, 51.5, -0.1) == (0.0, 0.0)
    x, y = site.to_xy(frame, 51.5, -0.09)
    assert x > 0 and y == 0
    x, y = site.to_xy(frame, 51.51, -0.1)
    assert y > 0 and x == 0


def test_a_box_of_a_given_radius_is_that_many_metres_across():
    frame = site.local_frame(51.5, -0.1)
    box = site.bbox(frame, 500.0)
    x, _ = site.to_xy(frame, 51.5, box["east"])
    _, y = site.to_xy(frame, box["north"], -0.1)
    assert x == pytest.approx(500.0, abs=1e-6)
    assert y == pytest.approx(500.0, abs=1e-6)


# -- the roughness fit -------------------------------------------------------------


def test_the_two_height_fit_recovers_a_roughness_it_was_given():
    """Round trip: build the two speeds from a known z0, then ask for z0 back."""
    z0 = 0.3
    low = math.log(10.0 / z0)
    high = math.log(100.0 / z0)
    assert site.roughness_from_two_heights(low, high, 10.0, 100.0) == pytest.approx(z0, rel=1e-6)


def test_the_fit_does_not_divide_by_zero_at_the_lower_height():
    """The first real pair of ERA5 means raised ZeroDivisionError: the bracket's top
    was a round 10 m, which is also the lower height, where ln(z_low/z0) is ln(1).
    This is that case, and it answers rather than raising."""
    assert site.roughness_from_two_heights(4.275, 7.0, 10.0, 100.0) is not None


def test_wind_faster_below_than_above_is_not_a_roughness():
    """It happens in a stable layer. There is no z0 that produces it, so None."""
    assert site.roughness_from_two_heights(8.0, 6.0, 10.0, 100.0) is None
    assert site.roughness_from_two_heights(8.0, 8.0, 10.0, 100.0) is None


def test_a_root_outside_the_bracket_is_none_rather_than_an_edge_value():
    assert site.roughness_from_two_heights(1.0, 100.0, 10.0, 100.0) is None


# -- the wind rose -----------------------------------------------------------------


def test_a_sector_is_centred_on_its_bearing_not_started_at_it():
    """A wind rose's first sector straddles north: 345 to 15 degrees, not 0 to 30.
    350 and 10 belong in the same sector and 20 does not."""
    assert site._sector_of(0.0) == 0
    assert site._sector_of(350.0) == 0
    assert site._sector_of(10.0) == 0
    assert site._sector_of(20.0) == 1
    assert site._sector_of(359.9) == 0


def test_the_prevailing_sector_is_the_one_with_the_most_hours():
    hourly = {
        "wind_speed_10m": [5.0] * 10 + [3.0] * 2,
        "wind_speed_100m": [10.0] * 10 + [6.0] * 2,
        "wind_direction_10m": [240.0] * 10 + [60.0] * 2,
    }
    out = site.wind_summary(hourly)
    assert out["hours"] == 12
    assert out["prevailing"]["centre_deg"] == 240.0
    assert out["prevailing"]["frequency"] == pytest.approx(10 / 12, abs=1e-4)


def test_flow_dir_points_the_way_the_wind_travels_not_the_way_it_comes_from():
    """A westerly blows FROM 270 and TOWARDS the east, so flowDir is +x in a frame
    whose x is east. Getting this backwards reverses every case built from it."""
    hourly = {"wind_speed_10m": [5.0], "wind_speed_100m": [8.0], "wind_direction_10m": [270.0]}
    out = site.wind_summary(hourly)
    x, y, z = out["prevailing"]["flow_dir_xy"]
    assert x == pytest.approx(1.0, abs=1e-6)
    assert y == pytest.approx(0.0, abs=1e-6)
    assert z == 0.0

    # A southerly comes from 180 and travels north: +y.
    hourly["wind_direction_10m"] = [180.0]
    x, y, _ = site.wind_summary(hourly)["prevailing"]["flow_dir_xy"]
    assert x == pytest.approx(0.0, abs=1e-6)
    assert y == pytest.approx(1.0, abs=1e-6)


def test_an_hour_with_no_direction_is_left_out_rather_than_counted_at_north():
    hourly = {"wind_speed_10m": [5.0, 5.0], "wind_speed_100m": [8.0, 8.0],
              "wind_direction_10m": [None, 90.0]}
    out = site.wind_summary(hourly)
    assert out["hours"] == 1
    assert out["sectors"][0]["hours"] == 0


# -- the archive -------------------------------------------------------------------

ARCHIVE_BODY = {
    "latitude": 51.493847,
    "longitude": -0.1630249,
    "elevation": 9.0,
    "hourly_units": {"wind_speed_10m": "m/s", "wind_direction_10m": "°"},
    "hourly": {
        "time": ["2023-01-01T00:00", "2023-01-01T01:00", "2023-01-01T02:00"],
        "wind_speed_10m": [8.19, 8.05, 7.45],
        "wind_speed_100m": [14.2, 14.0, 13.1],
        "wind_direction_10m": [240.0, 245.0, 60.0],
        "wind_direction_100m": [242.0, 247.0, 62.0],
    },
}


def transport_for(routes: dict[str, object]) -> httpx.MockTransport:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        for fragment, answer in routes.items():
            if fragment in str(request.url):
                if isinstance(answer, httpx.Response):
                    return answer
                return httpx.Response(200, json=answer)
        return httpx.Response(404, text=f"no route for {request.url}")

    handler.seen = seen  # type: ignore[attr-defined]
    return httpx.MockTransport(handler)


def test_the_archive_block_records_the_cell_it_actually_answered_for():
    transport = transport_for({"archive": ARCHIVE_BODY})
    out = site.fetch_weather(51.5045, -0.0865, "2023-01-01", "2023-01-01", transport=transport)
    cell = out["grid_cell"]
    assert cell["latitude"] == 51.493847
    # 5.4 km, measured: the request and the answer are not the same place.
    assert 5000 < cell["distance_from_site_m"] < 6000


def test_the_archive_block_carries_its_provenance_and_needs_no_key():
    transport = transport_for({"archive": ARCHIVE_BODY})
    out = site.fetch_weather(51.5045, -0.0865, "2023-01-01", "2023-01-01", transport=transport)
    source = out["source"]
    assert source["url"] == site.ARCHIVE_URL
    assert source["key_required"] is False
    assert "ERA5" in source["dataset"]
    assert source["licence"] and source["attribution"] and source["retrieved_at"]
    assert source["query"]["wind_speed_unit"] == "ms"
    assert out["units"]["wind_speed"] == "m/s"


def test_an_empty_archive_answer_says_what_is_wrong_with_the_dates():
    transport = transport_for({"archive": {"hourly": {"time": []}}})
    with pytest.raises(Unreachable, match="five days"):
        site.fetch_weather(51.5, -0.1, "2099-01-01", "2099-01-02", transport=transport)


def test_no_personal_detail_rides_along_in_the_user_agent():
    """The OSM policy asks for an application and version and no contact details, and
    a study's own data has no business in a weather request."""
    transport = transport_for({"archive": ARCHIVE_BODY})
    site.fetch_weather(51.5, -0.1, "2023-01-01", "2023-01-01", transport=transport)
    agent = transport.handler.seen[0].headers["user-agent"]  # type: ignore[attr-defined]
    assert agent.startswith("openreynolds/")
    assert "@" not in agent and len(agent.split()) == 1


# -- Overpass ----------------------------------------------------------------------

OVERPASS_BODY = {
    "generator": "Overpass API 0.7.62",
    "osm3s": {"timestamp_osm_base": "2026-11-19T12:00:00Z"},
    "elements": [
        {
            "type": "way",
            "id": 31110737,
            "tags": {"building": "commercial", "name": "The Shard", "height": "310",
                     "building:levels": "95"},
            # A closed way repeats its first node.
            "geometry": [{"lat": 51.5040, "lon": -0.0865}, {"lat": 51.5040, "lon": -0.0860},
                         {"lat": 51.5044, "lon": -0.0860}, {"lat": 51.5044, "lon": -0.0865},
                         {"lat": 51.5040, "lon": -0.0865}],
        },
        {
            "type": "way", "id": 2, "tags": {"building": "office", "building:levels": "4"},
            "geometry": [{"lat": 51.5050, "lon": -0.0870}, {"lat": 51.5050, "lon": -0.0868},
                         {"lat": 51.5052, "lon": -0.0868}, {"lat": 51.5050, "lon": -0.0870}],
        },
        {
            "type": "way", "id": 3, "tags": {"building": "yes", "height": "120 ft"},
            "geometry": [{"lat": 51.5030, "lon": -0.0880}, {"lat": 51.5030, "lon": -0.0878},
                         {"lat": 51.5032, "lon": -0.0878}, {"lat": 51.5030, "lon": -0.0880}],
        },
        {"type": "relation", "id": 9, "tags": {"building": "yes"}},
    ],
}


def test_a_height_tag_wins_and_levels_are_the_fallback():
    transport = transport_for({"interpreter": OVERPASS_BODY})
    out = site.fetch_buildings(51.5045, -0.0865, 500.0, transport=transport)
    by_id = {f["osm_id"]: f for f in out["features"]}
    assert by_id[31110737]["height_m"] == 310.0
    assert by_id[31110737]["height_from"] == "height"
    assert by_id[2]["height_m"] == 12.0
    assert "building:levels" in by_id[2]["height_from"]


def test_a_height_in_feet_is_refused_rather_than_read_as_metres():
    """`120 ft` parsed as 120 m is a building two and a half times too tall, and it
    runs. It falls through to levels, and with none it has no height at all."""
    transport = transport_for({"interpreter": OVERPASS_BODY})
    out = site.fetch_buildings(51.5045, -0.0865, 500.0, transport=transport)
    by_id = {f["osm_id"]: f for f in out["features"]}
    assert by_id[3]["height_m"] is None
    assert "120 ft" in by_id[3]["height_from"]
    assert out["without_height"] == 1


def test_the_repeated_closing_node_is_dropped_from_the_ring():
    transport = transport_for({"interpreter": OVERPASS_BODY})
    out = site.fetch_buildings(51.5045, -0.0865, 500.0, transport=transport)
    ring = out["features"][0]["polygon_xy"]
    assert len(ring) == 4
    assert ring[0] != ring[-1]


def test_a_relation_is_counted_as_skipped_rather_than_silently_lost():
    transport = transport_for({"interpreter": OVERPASS_BODY})
    out = site.fetch_buildings(51.5045, -0.0865, 500.0, transport=transport)
    assert out["count"] == 3
    assert out["skipped_by_type"] == {"relation": 1}
    assert "multipolygon" in out["skipped_note"]


def test_the_footprint_area_is_in_square_metres_and_positive():
    transport = transport_for({"interpreter": OVERPASS_BODY})
    out = site.fetch_buildings(51.5045, -0.0865, 500.0, transport=transport)
    # 0.0005 deg of longitude by 0.0004 deg of latitude at 51.5 N: about 34.7 by 44.5 m.
    assert out["features"][0]["area_m2"] == pytest.approx(1544, rel=0.02)


# -- elevation ---------------------------------------------------------------------


def test_the_elevation_grid_is_fetched_in_batches_of_a_hundred():
    """11 x 11 is 121 points, which is two requests, and the grid has to survive
    being cut across a row."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        count = len(str(request.url.params["latitude"]).split(","))
        calls.append(count)
        return httpx.Response(200, json={"elevation": [float(10 + len(calls))] * count})

    out = site.fetch_terrain(51.5, -0.1, 400.0, samples=11,
                             transport=httpx.MockTransport(handler))
    assert calls == [100, 21]
    assert out["samples_per_side"] == 11
    assert len(out["z"]) == 11 and all(len(row) == 11 for row in out["z"])
    assert out["source"]["query"]["requests"] == 2
    assert out["spacing_m"] == pytest.approx(80.0)


def test_the_grid_states_which_index_is_which_axis():
    """`z[j][i]` against `z[i][j]` is a terrain transposed, which looks like terrain."""
    def handler(request: httpx.Request) -> httpx.Response:
        count = len(str(request.url.params["latitude"]).split(","))
        return httpx.Response(200, json={"elevation": [1.0] * count})

    out = site.fetch_terrain(51.5, -0.1, 100.0, samples=3,
                             transport=httpx.MockTransport(handler))
    assert out["row_order"] == "z[j][i] is the elevation at (x[i], y[j])"
    assert out["x"][0] == pytest.approx(-100.0, abs=0.5)
    assert out["y"][-1] == pytest.approx(100.0, abs=0.5)


def test_a_short_elevation_answer_is_an_error_and_not_a_hole_in_the_grid():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"elevation": [1.0, 2.0]})

    with pytest.raises(Unreachable, match="elevations for"):
        site.fetch_terrain(51.5, -0.1, 100.0, samples=3,
                           transport=httpx.MockTransport(handler))


def test_no_key_is_needed_and_the_dem_is_named_with_its_resolution():
    def handler(request: httpx.Request) -> httpx.Response:
        count = len(str(request.url.params["latitude"]).split(","))
        return httpx.Response(200, json={"elevation": [1.0] * count})

    out = site.fetch_terrain(51.5, -0.1, 100.0, samples=3,
                             transport=httpx.MockTransport(handler))
    assert out["source"]["key_required"] is False
    assert out["source"]["resolution_m"] == 90
    assert "Copernicus" in out["source"]["dataset"]


# -- failures and retries ----------------------------------------------------------


def test_a_redirect_is_named_rather_than_followed():
    """Following one silently means a request answered by a host nobody chose; this
    is the same caution `backend/hosted.py` applies to the service."""
    transport = transport_for(
        {"archive": httpx.Response(302, headers={"location": "https://elsewhere.test/x"})})
    with pytest.raises(Unreachable, match="elsewhere.test"):
        site.fetch_weather(51.5, -0.1, "2023-01-01", "2023-01-01", transport=transport)


def test_a_rate_limit_is_retried_and_then_succeeds(monkeypatch):
    monkeypatch.setattr(site.time, "sleep", lambda _s: None)
    answers = [httpx.Response(429, headers={"Retry-After": "0"}),
               httpx.Response(200, json=ARCHIVE_BODY)]

    def handler(_request: httpx.Request) -> httpx.Response:
        return answers.pop(0)

    out = site.fetch_weather(51.5, -0.1, "2023-01-01", "2023-01-01",
                             transport=httpx.MockTransport(handler))
    assert out["hours"] == 3
    assert not answers


def test_a_four_hundred_is_not_retried_and_carries_the_body(monkeypatch):
    monkeypatch.setattr(site.time, "sleep", lambda _s: None)
    calls: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, text="start_date is after end_date")

    with pytest.raises(Unreachable, match="after end_date"):
        site.fetch_weather(51.5, -0.1, "2023-02-01", "2023-01-01",
                           transport=httpx.MockTransport(handler))
    assert len(calls) == 1


def test_one_source_failing_does_not_take_the_others_with_it():
    """A file with two of three blocks says which one is missing and why, instead of
    there being no file."""
    transport = transport_for({
        "archive": ARCHIVE_BODY,
        "interpreter": httpx.Response(504, text="Overpass timed out"),
    })
    out = site.collect(51.5045, -0.0865, radius_m=200.0, start="2023-01-01",
                       end="2023-01-01", terrain=False, transport=transport)
    assert out["weather"]["hours"] == 3
    assert "504" in out["buildings"]["failed"]
    assert "terrain" not in out


# -- the file ----------------------------------------------------------------------


def test_the_file_names_its_schema_and_says_nothing_personal_went_out():
    transport = transport_for({"archive": ARCHIVE_BODY})
    out = site.collect(51.5, -0.1, name="somewhere", start="2023-01-01", end="2023-01-01",
                       buildings=False, terrain=False, transport=transport)
    assert out["schema"] == site.SCHEMA
    assert out["site"]["name"] == "somewhere"
    assert any("User-Agent" in note for note in out["notes"])
    assert out["frame"]["origin"] == {"latitude": 51.5, "longitude": -0.1}


def test_an_opentopography_key_is_accepted_and_said_to_be_unused():
    transport = transport_for({"archive": ARCHIVE_BODY})
    out = site.collect(51.5, -0.1, start="2023-01-01", end="2023-01-01", buildings=False,
                       terrain=False, opentopography_key="secret", transport=transport)
    assert any("no key" in note for note in out["notes"])
    assert "secret" not in json.dumps(out)


def test_the_file_round_trips_through_json(tmp_path):
    transport = transport_for({"archive": ARCHIVE_BODY})
    out = site.collect(51.5, -0.1, start="2023-01-01", end="2023-01-01", buildings=False,
                       terrain=False, transport=transport)
    path = tmp_path / "site.json"
    assert site.write(out, path) > 0
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == site.SCHEMA


def test_every_attribution_is_carried_once():
    transport = transport_for({"archive": ARCHIVE_BODY, "interpreter": OVERPASS_BODY})
    out = site.collect(51.5045, -0.0865, radius_m=200.0, start="2023-01-01",
                       end="2023-01-01", terrain=False, transport=transport)
    lines = site.attributions(out)
    assert any("Open-Meteo" in line for line in lines)
    assert any("OpenStreetMap" in line for line in lines)
    assert len(lines) == len(set(lines))


def test_the_default_period_is_a_whole_year_that_has_already_happened():
    start, end = site.default_period()
    assert start.endswith("-01-01") and end.endswith("-12-31")
    assert start[:4] == end[:4]
    from datetime import datetime, timezone
    assert int(start[:4]) < datetime.now(timezone.utc).year


def test_the_headline_says_what_failed_as_well_as_what_did_not():
    out = {"site": {"name": "x", "latitude": 1, "longitude": 2, "radius_m": 3},
           "buildings": {"failed": "Overpass timed out"}}
    text = "\n".join(site.headline(out))
    assert "Overpass timed out" in text
