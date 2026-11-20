"""Site data for a real location: wind, building footprints, terrain.

**Why this lives here and not in the toolbox.** The workspace has no outbound
network. `docker/workspace/install-guard.sh` shadows `pip` and `apt` with a script
that refuses anything needing a fetch, and the Dockerfile comment above that layer
says it plainly: "The workspace has no network." A toolbox script therefore cannot
call an API, however much it would like to. The `openreynolds` CLI runs on the
person's own machine, which does have one -- it already talks to the hosted service
over HTTPS from here (`backend/hosted.py`) -- so the split is:

  * this module fetches, on this machine, and writes **one** self-describing file;
  * `toolbox/site_abl.py` and `toolbox/site_stl.py` read that file in the workspace
    and turn it into boundary conditions and surfaces, and refuse clearly when it
    is not there, naming the command that produces it.

Everything fetched is free and needs no key. The three sources:

  * **Open-Meteo ERA5 archive** (`archive-api.open-meteo.com/v1/archive`) for
    hourly wind at 10 m and 100 m. Reanalysis, so it answers for a ~25 km grid cell
    and not for the street: the cell centre it actually answered for is recorded
    beside the request, with the distance between them.
  * **Overpass** (`overpass-api.de/api/interpreter`) for OpenStreetMap building
    footprints, ODbL, (c) OpenStreetMap contributors.
  * **Open-Meteo elevation** (`api.open-meteo.com/v1/elevation`) for terrain, which
    serves Copernicus DEM GLO-90 at 90 m and takes at most 100 coordinates per
    request, so a grid goes up in batches.

OpenTopography was looked at for the DEM and is not used by default: its global
datasets have needed an API key since 2022. `--opentopography-key` is accepted and
when it is absent the Copernicus path is used and said so, rather than failing.

**No personal detail leaves this machine.** The User-Agent is `openreynolds/<version>`
and nothing else: the OSM API usage policy asks for a User-Agent that identifies the
application and its version, and asks for no contact details. No request carries a
name, an e-mail address, an account id or a study id.
"""

from __future__ import annotations

import json
import math
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Any

import httpx2 as httpx

from . import __version__
# The hosted client's own retry policy, reused rather than reinvented: which statuses
# are worth asking again (`_RETRY_STATUSES`) and how long to wait, Retry-After first
# (`_retry_delay`). Both are measured decisions with the evidence written beside them
# in `backend/hosted.py`; a weather API answers 429 and 504 for the same reasons a
# service does, so the same answers apply.
from .backend.hosted import _RETRY_STATUSES, _retry_delay

SCHEMA = "openreynolds.site/1"
"""What the toolbox scripts check for before reading anything else."""

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

ELEVATION_BATCH = 100
"""The elevation API's documented ceiling on coordinates per request."""

ATTEMPTS = 3
TIMEOUT_S = 60.0
"""Overpass is a shared public instance and a 1 km box of a dense city takes it
seconds to minutes under load; 60 is its own default `[timeout:]` and not a guess."""

SECTORS = 12
"""30-degree compass sectors, the convention a wind rose is read in."""

SPEED_BIN_MS = 1.0
"""Width of the per-sector speed histogram's bins, in m/s.

Why a histogram at all, when the file already carries a mean and four percentiles per
sector. A pedestrian comfort criterion is an **exceedance probability**: the fraction of
hours in which the speed at a place passes a threshold. Composed over directions that
is `sum_sectors sum_bins hours * [K_sector * u_bin > threshold]`, where `K` is the
amplification a CFD run measured for that direction -- so the question asked of the
weather is "how many hours did this sector blow between 6 and 7 m/s", which p50/p90/p98
cannot answer. Five percentiles are five points on a curve; the hours per bin are the
curve. 1 m/s against a reanalysis whose own speeds are a 25 km cell average is already
finer than the data, and twelve sectors of about 25 bins is 300 integers in the file."""


class Unreachable(Exception):
    """A source could not be read. Carries a sentence a person can act on."""


# -- the local frame ---------------------------------------------------------------------
#
# Everything downstream wants metres, and the two consumers have to agree on what
# (x, y) means or a building lands in the wrong place. So the frame is written into
# the file rather than recomputed: east-north-up metres about the site, with the two
# scale factors stated, so `site_stl.py` and anything else invert exactly what was
# applied here.


def local_frame(lat: float, lon: float) -> dict[str, Any]:
    """Metres per degree at this latitude, from the WGS84 series.

    Not 111320 flat: at 51 degrees north the parallel is 69,300 m per degree and the
    meridian 111,200, a 60% difference, and a footprint projected with one number
    for both comes out stretched east-west by that much.
    """
    phi = math.radians(lat)
    per_lat = (111132.92 - 559.82 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi)
               - 0.0023 * math.cos(6 * phi))
    per_lon = (111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi)
               + 0.118 * math.cos(5 * phi))
    return {
        "kind": "east-north-up metres, tangent plane at the origin",
        "origin": {"latitude": lat, "longitude": lon},
        "metres_per_degree_latitude": round(per_lat, 3),
        "metres_per_degree_longitude": round(per_lon, 3),
        "x": "east, metres from the origin",
        "y": "north, metres from the origin",
        "z": "metres above the geoid, as the DEM reports it",
    }


def to_xy(frame: dict[str, Any], lat: float, lon: float) -> tuple[float, float]:
    origin = frame["origin"]
    x = (lon - origin["longitude"]) * frame["metres_per_degree_longitude"]
    y = (lat - origin["latitude"]) * frame["metres_per_degree_latitude"]
    return x, y


def bbox(frame: dict[str, Any], radius_m: float) -> dict[str, float]:
    """A square box of side 2*radius about the origin, in degrees."""
    origin = frame["origin"]
    dlat = radius_m / frame["metres_per_degree_latitude"]
    dlon = radius_m / frame["metres_per_degree_longitude"]
    return {
        "south": origin["latitude"] - dlat,
        "west": origin["longitude"] - dlon,
        "north": origin["latitude"] + dlat,
        "east": origin["longitude"] + dlon,
    }


# -- fetching ----------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _client(transport: Any = None) -> httpx.Client:
    # follow_redirects off for the reason `backend/hosted.py` turns it off: a redirect
    # that is followed silently is a request answered by a host nobody chose. Here it
    # is reported instead, with the Location, so the next version can point at it.
    return httpx.Client(
        timeout=TIMEOUT_S,
        transport=transport,
        follow_redirects=False,
        headers={"User-Agent": f"openreynolds/{__version__}"},
    )


def _get(client: httpx.Client, url: str, params: dict[str, Any] | None = None,
         data: dict[str, str] | None = None) -> httpx.Response:
    """One request, retried on the statuses that say "ask again" and nothing else."""
    last: httpx.Response | None = None
    for attempt in range(ATTEMPTS):
        try:
            if data is not None:
                response = client.post(url, data=data)
            else:
                response = client.get(url, params=params)
        except httpx.HTTPError as exc:
            if attempt == ATTEMPTS - 1:
                raise Unreachable(f"cannot reach {url}: {exc}") from exc
            time.sleep(_retry_delay(None, attempt))
            continue
        if response.status_code in _RETRY_STATUSES and attempt < ATTEMPTS - 1:
            last = response
            time.sleep(_retry_delay(response, attempt))
            continue
        last = response
        break
    assert last is not None  # the loop either breaks with a response or raises
    if 300 <= last.status_code < 400:
        where = last.headers.get("location", "somewhere unnamed")
        raise Unreachable(
            f"{url} redirected to {where}; redirects are not followed, so this needs "
            f"the new address rather than a retry"
        )
    if last.status_code >= 400:
        detail = (last.text or "").strip()[:400]
        raise Unreachable(f"{url} answered {last.status_code}: {detail or 'no body'}")
    return last


def _json_body(response: httpx.Response, url: str) -> Any:
    try:
        return response.json()
    except Exception as exc:  # noqa: BLE001
        raise Unreachable(f"{url} answered something that is not JSON") from exc


# -- wind --------------------------------------------------------------------------------


def _quantile(values: list[float], fraction: float) -> float:
    """Linear interpolation between order statistics. No numpy: this is the CLI
    package, whose dependencies are the four in `pyproject.toml` and the stdlib."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _stats(values: list[float]) -> dict[str, float]:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return {"hours": 0}
    return {
        "hours": len(clean),
        "mean": round(sum(clean) / len(clean), 3),
        "p50": round(_quantile(clean, 0.50), 3),
        "p90": round(_quantile(clean, 0.90), 3),
        "p98": round(_quantile(clean, 0.98), 3),
        "max": round(max(clean), 3),
    }


def roughness_from_two_heights(u_low: float, u_high: float, z_low: float,
                               z_high: float) -> float | None:
    """The z0 that makes a log law pass through both speeds, by bisection.

    ln(z_high/z0) / ln(z_low/z0) = u_high/u_low has one root in z0 for a ratio above
    1, and the function is monotonic in ln(z0), so bisection is the whole method --
    no scipy anywhere near this, and none needed. None when the data says nothing
    usable: a ratio at or below 1 (wind faster at 10 m than at 100 m, which happens
    in a stable layer and is not a roughness) or a root outside the bracket.

    The bracket's top is half the lower height and not a round 10 m: at z0 = z_low
    the denominator is ln(1) and the ratio divides by zero, which is exactly what
    this did on the first real pair of ERA5 means it was handed (z_low = 10 m,
    hi = 10 m). Half of 10 m is 5 m and the roughest Davenport class is 2 m, so
    nothing real is excluded by the narrower bracket.
    """
    if not (u_low > 0 and u_high > 0) or u_high <= u_low:
        return None
    if not 0 < z_low < z_high:
        return None
    target = u_high / u_low

    def ratio(z0: float) -> float:
        return math.log(z_high / z0) / math.log(z_low / z0)

    lo, hi = 1e-6, 0.5 * z_low
    if not (ratio(lo) - target) * (ratio(hi) - target) < 0:
        return None
    for _ in range(200):
        mid = math.sqrt(lo * hi)  # bisect in log space: z0 spans six decades
        if (ratio(lo) - target) * (ratio(mid) - target) <= 0:
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


def _histogram(values: list[float], bins: int) -> dict[str, Any]:
    """Hours per `SPEED_BIN_MS` bin, counting from zero.

    `bins` is passed in rather than taken from these values so that every sector gets
    the same axis: a rose stacks the sectors on one another and an exceedance sums
    across them, and both are wrong if sector 3's bin 7 is a different speed from
    sector 4's. Everything at or above the last bin's lower edge lands in it, which
    only happens when the caller sized the axis off a different sample.
    """
    hours = [0] * max(bins, 1)
    for value in values:
        index = int(value / SPEED_BIN_MS)
        hours[min(max(index, 0), len(hours) - 1)] += 1
    return {
        "bin_width_ms": SPEED_BIN_MS,
        "bins": len(hours),
        "hours": hours,
        "edges": "bin i covers i*bin_width_ms up to (i+1)*bin_width_ms, m/s at 10 m",
    }


def _sector_of(direction: float) -> int:
    width = 360.0 / SECTORS
    return int(((direction % 360.0) + width / 2) // width) % SECTORS


def wind_summary(hourly: dict[str, list[Any]]) -> dict[str, Any]:
    """Hours in, a wind rose and a shear fit out. Pure arithmetic, no fetching."""
    speed10 = [v for v in hourly.get("wind_speed_10m", [])]
    speed100 = [v for v in hourly.get("wind_speed_100m", [])]
    direction = [v for v in hourly.get("wind_direction_10m", [])]
    width = 360.0 / SECTORS

    buckets: list[dict[str, list[float]]] = [{"s10": [], "s100": []} for _ in range(SECTORS)]
    counted = 0
    for index, bearing in enumerate(direction):
        if bearing is None:
            continue
        s10 = speed10[index] if index < len(speed10) else None
        # A not-a-number is dropped here and not further down, so that this sector's
        # `hours`, its percentiles and its histogram are all counts of the same set
        # of hours. `_stats` has always dropped them; when only it did, the hours a
        # sector reported and the hours its histogram held could differ, and the
        # exceedance arithmetic divides one by the other.
        if s10 is None or math.isnan(float(s10)):
            continue
        bucket = buckets[_sector_of(float(bearing))]
        bucket["s10"].append(float(s10))
        s100 = speed100[index] if index < len(speed100) else None
        if s100 is not None and not math.isnan(float(s100)):
            bucket["s100"].append(float(s100))
        counted += 1

    every10 = [v for bucket in buckets for v in bucket["s10"]]
    bins = int(max(every10) / SPEED_BIN_MS) + 1 if every10 else 1

    sectors = []
    for index, bucket in enumerate(buckets):
        centre = (index * width) % 360.0
        sectors.append({
            "centre_deg": centre,
            "from_deg": round((centre - width / 2) % 360.0, 1),
            "to_deg": round((centre + width / 2) % 360.0, 1),
            "hours": len(bucket["s10"]),
            "frequency": round(len(bucket["s10"]) / counted, 4) if counted else 0.0,
            "speed_10m": _stats(bucket["s10"]),
            "speed_100m": _stats(bucket["s100"]),
            "speed_10m_histogram": _histogram(bucket["s10"], bins),
        })

    summary: dict[str, Any] = {
        "hours": counted,
        "sector_width_deg": width,
        "direction_convention": "degrees clockwise from true north, the direction the wind blows FROM",
        "speed_10m": _stats([float(v) for v in speed10 if v is not None]),
        "speed_100m": _stats([float(v) for v in speed100 if v is not None]),
        "speed_10m_histogram": _histogram(every10, bins),
        "sectors": sectors,
    }

    if counted:
        windiest = max(sectors, key=lambda s: s["hours"])
        summary["prevailing"] = {
            "centre_deg": windiest["centre_deg"],
            "frequency": windiest["frequency"],
            "hours": windiest["hours"],
            "mean_speed_10m": windiest["speed_10m"].get("mean"),
            "p90_speed_10m": windiest["speed_10m"].get("p90"),
            "mean_speed_100m": windiest["speed_100m"].get("mean"),
            # Meteorological bearing to the vector the wind travels along: a wind FROM
            # 240 degrees moves towards 060, so this is the ABL `flowDir` in a frame
            # whose x is east and y is north.
            "flow_dir_xy": _flow_dir(windiest["centre_deg"]),
        }
        low = summary["speed_10m"].get("mean")
        high = summary["speed_100m"].get("mean")
        if low and high:
            z0 = roughness_from_two_heights(low, high, 10.0, 100.0)
            summary["shear"] = {
                "method": "log law through the all-hours mean speed at 10 m and 100 m",
                "power_law_alpha": round(math.log(high / low) / math.log(10.0), 4)
                if high > 0 and low > 0 else None,
                "fitted_z0_m": round(z0, 5) if z0 else None,
                "caveat": (
                    "this is the roughness of the reanalysis grid cell, not of the site: "
                    "ERA5 carries its own surface roughness for a cell tens of kilometres "
                    "across. It is a cross-check on an order of magnitude, not a value to "
                    "put in a boundary condition -- site_abl.py asks for z0 by land-cover "
                    "class or by number and refuses to pick one itself."
                ),
            }
    return summary


def _flow_dir(from_deg: float) -> list[float]:
    """Unit vector the wind travels along, in east-north metres."""
    towards = math.radians((from_deg + 180.0) % 360.0)
    return [round(math.sin(towards), 6), round(math.cos(towards), 6), 0.0]


def fetch_weather(lat: float, lon: float, start: str, end: str, *,
                  transport: Any = None) -> dict[str, Any]:
    params = {
        "latitude": f"{lat:.6f}",
        "longitude": f"{lon:.6f}",
        "start_date": start,
        "end_date": end,
        "hourly": "wind_speed_10m,wind_speed_100m,wind_direction_10m,wind_direction_100m",
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }
    with _client(transport) as client:
        response = _get(client, ARCHIVE_URL, params=params)
    body = _json_body(response, ARCHIVE_URL)
    hourly = body.get("hourly") or {}
    if not hourly.get("time"):
        raise Unreachable(
            f"{ARCHIVE_URL} returned no hours for {start}..{end}; the archive lags "
            f"real time by about five days, so an end date too close to today is empty"
        )

    frame = local_frame(lat, lon)
    cell_lat, cell_lon = body.get("latitude"), body.get("longitude")
    offset = None
    if cell_lat is not None and cell_lon is not None:
        dx, dy = to_xy(frame, float(cell_lat), float(cell_lon))
        offset = round(math.hypot(dx, dy), 1)

    out = wind_summary(hourly)
    out["source"] = {
        "url": ARCHIVE_URL,
        "query": params,
        "retrieved_at": _now(),
        "dataset": "ERA5 / ERA5-Land reanalysis, served by Open-Meteo",
        "licence": "CC-BY-4.0 (Open-Meteo); Copernicus Climate Change Service information",
        "attribution": "Weather data by Open-Meteo.com, ERA5 from the Copernicus Climate Change Service",
        "key_required": False,
    }
    out["units"] = {"wind_speed": "m/s", "wind_direction": "deg", "height": "m"}
    out["grid_cell"] = {
        "latitude": cell_lat,
        "longitude": cell_lon,
        "elevation_m": body.get("elevation"),
        "distance_from_site_m": offset,
        "note": (
            "the archive answers for the centre of its own grid cell, not for the "
            "coordinates asked for; the distance is how far that is"
        ),
    }
    out["period"] = {"start": start, "end": end, "timezone": "UTC"}
    return out


# -- buildings ---------------------------------------------------------------------------


def _height_metres(tags: dict[str, str], levels_height: float) -> tuple[float | None, str]:
    """A footprint's height and where the number came from.

    `height` wins, then `building:levels` times a storey. A `height` that is not a
    plain number of metres (`120 ft`, `~40`) is left alone rather than guessed at:
    the feature falls through to levels and the reason is recorded, because a
    silently misread height is a building of the wrong size in a wind field.
    """
    raw = (tags.get("height") or "").strip()
    if raw:
        try:
            return float(raw.rstrip(" m").strip()), "height"
        except ValueError:
            pass
    levels = (tags.get("building:levels") or "").strip()
    if levels:
        try:
            return float(levels) * levels_height, f"building:levels x {levels_height:g} m"
        except ValueError:
            pass
    if raw:
        return None, f"unparsed height tag {raw!r}"
    return None, "no height and no building:levels"


def fetch_buildings(lat: float, lon: float, radius_m: float, *, levels_height: float = 3.0,
                    transport: Any = None) -> dict[str, Any]:
    frame = local_frame(lat, lon)
    box = bbox(frame, radius_m)
    query = (
        f"[out:json][timeout:{int(TIMEOUT_S)}];\n"
        f"nwr[\"building\"]"
        f"({box['south']:.7f},{box['west']:.7f},{box['north']:.7f},{box['east']:.7f});\n"
        f"out geom;\n"
    )
    with _client(transport) as client:
        response = _get(client, OVERPASS_URL, data={"data": query})
    body = _json_body(response, OVERPASS_URL)

    features: list[dict[str, Any]] = []
    other_types: dict[str, int] = {}
    no_height = 0
    for element in body.get("elements") or []:
        if element.get("type") != "way":
            other_types[str(element.get("type"))] = other_types.get(str(element.get("type")), 0) + 1
            continue
        geometry = element.get("geometry") or []
        if len(geometry) < 4:
            continue
        ring = [to_xy(frame, float(p["lat"]), float(p["lon"])) for p in geometry
                if p.get("lat") is not None]
        # Overpass closes a way by repeating its first node; a polygon is the ring
        # without that repeat, and a consumer that keeps it writes a zero-length edge.
        if len(ring) > 2 and abs(ring[0][0] - ring[-1][0]) < 1e-9 \
                and abs(ring[0][1] - ring[-1][1]) < 1e-9:
            ring = ring[:-1]
        if len(ring) < 3:
            continue
        tags = element.get("tags") or {}
        height, source = _height_metres(tags, levels_height)
        if height is None:
            no_height += 1
        features.append({
            "osm_id": element.get("id"),
            "osm_type": "way",
            "name": tags.get("name"),
            "building": tags.get("building"),
            "height_m": round(height, 3) if height is not None else None,
            "height_from": source,
            "polygon_xy": [[round(x, 3), round(y, 3)] for x, y in ring],
            "area_m2": round(abs(_shoelace(ring)), 1),
        })

    return {
        "source": {
            "url": OVERPASS_URL,
            "query": query,
            "retrieved_at": _now(),
            "dataset": "OpenStreetMap, via the Overpass API",
            "licence": "ODbL 1.0",
            "attribution": "(c) OpenStreetMap contributors",
            "key_required": False,
            "generator": body.get("generator"),
            "osm_timestamp": ((body.get("osm3s") or {}).get("timestamp_osm_base")),
        },
        "units": {"polygon_xy": "m in the site frame", "height_m": "m above ground", "area_m2": "m^2"},
        "bbox_degrees": box,
        "radius_m": radius_m,
        "levels_height_m": levels_height,
        "count": len(features),
        "without_height": no_height,
        "skipped_by_type": other_types,
        "skipped_note": (
            "only OSM ways are kept. A building mapped as a multipolygon relation has "
            "its outer ring in member ways and is counted in skipped_by_type rather "
            "than guessed at; the count says how many a footprint set is missing."
        ),
        "features": features,
    }


def _shoelace(ring: list[tuple[float, float]]) -> float:
    total = 0.0
    for index in range(len(ring)):
        x1, y1 = ring[index]
        x2, y2 = ring[(index + 1) % len(ring)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


# -- terrain -----------------------------------------------------------------------------


def fetch_terrain(lat: float, lon: float, radius_m: float, *, samples: int = 21,
                  transport: Any = None) -> dict[str, Any]:
    """A square grid of elevations, `samples` a side, in batches of 100."""
    if samples < 2:
        raise Unreachable("a terrain grid needs at least 2 samples a side")
    frame = local_frame(lat, lon)
    box = bbox(frame, radius_m)
    lats = [box["south"] + (box["north"] - box["south"]) * i / (samples - 1)
            for i in range(samples)]
    lons = [box["west"] + (box["east"] - box["west"]) * i / (samples - 1)
            for i in range(samples)]

    points = [(la, lo) for la in lats for lo in lons]
    heights: list[float] = []
    batches = 0
    with _client(transport) as client:
        for start in range(0, len(points), ELEVATION_BATCH):
            chunk = points[start:start + ELEVATION_BATCH]
            params = {
                "latitude": ",".join(f"{la:.6f}" for la, _ in chunk),
                "longitude": ",".join(f"{lo:.6f}" for _, lo in chunk),
            }
            response = _get(client, ELEVATION_URL, params=params)
            body = _json_body(response, ELEVATION_URL)
            got = body.get("elevation")
            if not isinstance(got, list) or len(got) != len(chunk):
                raise Unreachable(
                    f"{ELEVATION_URL} answered {len(got) if isinstance(got, list) else 'no'} "
                    f"elevations for {len(chunk)} coordinates"
                )
            heights.extend(float(v) for v in got)
            batches += 1

    rows = [heights[i * samples:(i + 1) * samples] for i in range(samples)]
    flat = [v for row in rows for v in row]
    return {
        "source": {
            "url": ELEVATION_URL,
            "query": {"samples_per_side": samples, "batch_size": ELEVATION_BATCH,
                      "requests": batches, "bbox_degrees": box},
            "retrieved_at": _now(),
            "dataset": "Copernicus DEM GLO-90 (2021 release), served by Open-Meteo",
            "resolution_m": 90,
            "licence": "free licence, Copernicus DEM; DOI 10.5270/ESA-c5d3d65",
            "attribution": "Elevation by Open-Meteo.com, Copernicus DEM GLO-90",
            "key_required": False,
        },
        "units": {"x": "m east in the site frame", "y": "m north in the site frame",
                  "z": "m above the geoid"},
        "samples_per_side": samples,
        "spacing_m": round(2 * radius_m / (samples - 1), 2),
        "x": [round(to_xy(frame, lat, lo)[0], 2) for lo in lons],
        "y": [round(to_xy(frame, la, lon)[1], 2) for la in lats],
        "z": [[round(v, 2) for v in row] for row in rows],
        "z_min": min(flat),
        "z_max": max(flat),
        "relief_m": round(max(flat) - min(flat), 2),
        "row_order": "z[j][i] is the elevation at (x[i], y[j])",
    }


# -- the one file ------------------------------------------------------------------------


def collect(lat: float, lon: float, *, name: str = "", radius_m: float = 500.0,
            start: str = "", end: str = "", samples: int = 21,
            levels_height: float = 3.0, weather: bool = True, buildings: bool = True,
            terrain: bool = True, opentopography_key: str = "",
            transport: Any = None) -> dict[str, Any]:
    """Fetch what was asked for and assemble the file. Partial is fine and recorded.

    A source that fails does not take the others down with it: its block carries the
    sentence it failed with, so a file with two of three blocks says which one is
    missing and why, rather than being absent.
    """
    frame = local_frame(lat, lon)
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "written_by": f"openreynolds/{__version__} site",
        "written_at": _now(),
        "site": {
            "name": name or f"{lat:.5f},{lon:.5f}",
            "latitude": lat,
            "longitude": lon,
            "radius_m": radius_m,
        },
        "frame": frame,
        "notes": [
            "Every block names the URL it came from, the query sent, when it was "
            "retrieved, its units and its licence. Nothing here was computed from a "
            "constant in a prompt.",
            "No request carried a name, an e-mail address or an account id; the "
            "User-Agent was openreynolds/<version> and nothing else.",
        ],
    }
    if opentopography_key:
        out["notes"].append(
            "an OpenTopography key was supplied and not used: the Copernicus DEM path "
            "needs no key and is the one taken"
        )

    wanted = [
        ("weather", weather, lambda: fetch_weather(lat, lon, start, end, transport=transport)),
        ("buildings", buildings, lambda: fetch_buildings(
            lat, lon, radius_m, levels_height=levels_height, transport=transport)),
        ("terrain", terrain, lambda: fetch_terrain(
            lat, lon, radius_m, samples=samples, transport=transport)),
    ]
    for key, asked, call in wanted:
        if not asked:
            continue
        try:
            out[key] = call()
        except Unreachable as exc:
            out[key] = {"failed": str(exc), "attempted_at": _now()}
    return out


def headline(data: dict[str, Any]) -> list[str]:
    """What was fetched, in lines a person reads before trusting the file."""
    lines: list[str] = []
    site = data.get("site", {})
    lines.append(f"{site.get('name')}  {site.get('latitude')}, {site.get('longitude')}"
                 f"  radius {site.get('radius_m')} m")

    wind = data.get("weather") or {}
    if wind.get("failed"):
        lines.append(f"wind      could not be read: {wind['failed']}")
    elif wind:
        stats = wind.get("speed_10m", {})
        cell = wind.get("grid_cell", {})
        period = wind.get("period", {})
        lines.append(
            f"wind      {wind.get('hours', 0)} hours {period.get('start')}..{period.get('end')}, "
            f"mean {stats.get('mean')} m/s at 10 m, p90 {stats.get('p90')}, max {stats.get('max')}"
        )
        prevailing = wind.get("prevailing") or {}
        if prevailing:
            lines.append(
                f"          prevailing from {prevailing.get('centre_deg'):g} deg "
                f"({prevailing.get('frequency', 0) * 100:.0f}% of hours), "
                f"mean {prevailing.get('mean_speed_10m')} m/s"
            )
        shear = wind.get("shear") or {}
        if shear.get("fitted_z0_m") is not None:
            lines.append(
                f"          10/100 m shear: alpha {shear.get('power_law_alpha')}, "
                f"z0 {shear.get('fitted_z0_m')} m for the ERA5 cell, not for the site"
            )
        if cell.get("distance_from_site_m") is not None:
            lines.append(
                f"          the ERA5 cell centre is {cell['distance_from_site_m']:.0f} m away"
            )

    buildings = data.get("buildings") or {}
    if buildings.get("failed"):
        lines.append(f"buildings could not be read: {buildings['failed']}")
    elif buildings:
        skipped = sum(buildings.get("skipped_by_type", {}).values())
        lines.append(
            f"buildings {buildings.get('count', 0)} footprints, "
            f"{buildings.get('without_height', 0)} with no height tag, "
            f"{skipped} non-way features skipped"
        )
        tall = [f for f in buildings.get("features", []) if f.get("height_m")]
        if tall:
            highest = max(tall, key=lambda f: f["height_m"])
            lines.append(
                f"          tallest {highest.get('name') or highest.get('osm_id')} at "
                f"{highest['height_m']:g} m ({highest['height_from']})"
            )

    terrain = data.get("terrain") or {}
    if terrain.get("failed"):
        lines.append(f"terrain   could not be read: {terrain['failed']}")
    elif terrain:
        lines.append(
            f"terrain   {terrain.get('samples_per_side')}x{terrain.get('samples_per_side')} grid at "
            f"{terrain.get('spacing_m')} m, {terrain.get('z_min')}..{terrain.get('z_max')} m, "
            f"relief {terrain.get('relief_m')} m"
        )
    return lines


def default_period() -> tuple[str, str]:
    """A full calendar year, ending at the last one the archive certainly has.

    ERA5 lags real time by about five days, so "last year" is always complete and
    "this year to date" sometimes is not.
    """
    year = datetime.now(timezone.utc).year - 1
    return f"{year}-01-01", f"{year}-12-31"


def write(data: dict[str, Any], path: Any) -> int:
    text = json.dumps(data, indent=2, sort_keys=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return len(text)


def attributions(data: dict[str, Any]) -> list[str]:
    """Every attribution string in the file, deduplicated. The sources ask for these
    to be carried, so they are carried into whatever quotes the data."""
    seen: list[str] = []
    for key in ("weather", "buildings", "terrain"):
        block = data.get(key) or {}
        line = ((block.get("source") or {}).get("attribution") or "").strip()
        if line and line not in seen:
            seen.append(line)
    return seen


def url_with(params: dict[str, Any], base: str) -> str:
    """The GET as one URL, for a provenance line someone can paste into a browser."""
    return base + "?" + urllib.parse.urlencode(params)
