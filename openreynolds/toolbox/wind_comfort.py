#!/usr/bin/env python3
"""A wind rose from measured hours, and a pedestrian comfort map from per-direction runs.

Two things, because they are the same arithmetic. A comfort criterion is an
**exceedance probability**: the fraction of the year in which the wind speed where a
person stands passes a threshold. One CFD run answers for one wind direction, so the
year is reassembled from the site's own direction-binned hours:

    P(x, y) = (1 / total hours) * sum over sectors sum over speed bins
              hours[sector][bin] * [ K_sector(x, y) * u_bin > threshold ]

`K_sector(x, y) = U_local / Uref` is the amplification a solved case measured for that
direction, and `hours[sector][bin]` is counted weather, not a fitted distribution:
`openreynolds site` writes a 1 m/s histogram of the 10 m speed per 30-degree sector
(`openreynolds/site.py`, `SPEED_BIN_MS`). Inside a bin the hours are spread evenly,
which is the only interpolation here.

**The threshold and the class bands are yours to supply, and that is deliberate.**
The recognised criteria are a standards question and the primary texts could not be
read from here:

  * **NEN 8100:2006** (Dutch, "Windhinder en windgevaar in de gebouwde omgeving") is
    sold by NEN and was not readable. What *is* confirmed, from the abstract of
    Willemsen & Wisse, "Design for wind comfort in The Netherlands: procedures,
    criteria and open research issues", *J. Wind Eng. Ind. Aerodyn.* 95(9-11),
    1541-1550 -- the authors of the code itself -- is the mechanism and its shape:
    comfort is graded on *the probability that the pedestrian wind speed exceeds
    5 m/s*, into five grades A to E, assessed for traversing, strolling and sitting.
    That is the mechanism this script implements. The grade boundaries themselves
    (the percentages) appear in this repository's searches only in consultancy and
    vendor restatements, so **they are not written into this file**. Pass them with
    `--bands` and they are applied; leave them out and you get the probability and no
    letter.
  * **Lawson (1978)**, "The wind content of the built environment", *J. Ind.
    Aerodyn.* 3, 93-105, and the LDDC variant of it: paywalled, unread. The
    secondary accounts do not even agree on the exceedance fraction (2% for the
    original, 5% for the LDDC subset), which is exactly the kind of disagreement that
    makes a transcribed table worthless.
  * **Davenport/Isyumov (1975)** comfort criteria: conference proceedings, unread.

`benchmarks/README.md` is the reason for this stance: two reference values in this
repository were wrong because they came from secondary sources, one by a factor of
two, and both were wrong in the direction of agreement. A correct mechanism fed your
own thresholds beats a table nobody here could check.

What the composition assumes, stated rather than buried:

  * **One run per direction, any speed.** `K` is taken as independent of the
    reference speed -- high Reynolds number, sharp-edged buildings, pressure-driven
    separation. That is the usual justification and it fails for a case that is not
    at high Re.
  * **Speed at 10 m is the reference.** The histogram is of the 10 m wind, so each
    run's `Uref` has to be a 10 m speed. A case whose `0/include/ABLConditions` has
    `Zref` other than 10 is refused, naming the rescaling rather than doing it
    silently.
  * **The horizontal speed**, `sqrt(Ux^2 + Uy^2)`, not the vector magnitude: the
    criteria are written against anemometer data and a cup anemometer sees the
    horizontal component.
  * **Sectors you did not run contribute nothing**, so a partial set gives a lower
    bound and `sector_coverage` in the output says what fraction of the hours were
    covered. Twelve runs is a full year; four is not four twelfths of an answer.

    python3 wind_comfort.py --site site.json --rose rose.png
    python3 wind_comfort.py --site site.json --json
    python3 wind_comfort.py --site site.json --threshold 5 --pedestrian-z 1.75 \
        --run 240=runs/sw --run 270=runs/w --map comfort.png
    python3 wind_comfort.py --site site.json --threshold 5 --pedestrian-z 1.75 \
        --run 0=runs/n --run 30=runs/nne ... --bands 2.5,5,10,20 --map comfort.png
    python3 wind_comfort.py --site site.json --threshold 15 --pedestrian-z 1.75 \
        --run 240=runs/sw --csv danger.csv

The site file is fetched on your own machine, because this workspace has no outbound
network:

    openreynolds site --lat 51.5045 --lon -0.0865 --radius 500 -o site.json
    openreynolds push site.json

Exit code is 0 whatever is found, and 2 when the inputs are refused.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import site_abl  # noqa: E402  (sibling script, not a package; it owns the site file reader)

Refused = site_abl.Refused

REFERENCE_HEIGHT_M = 10.0
"""The height the site file's speed histogram is at (`site.py:fetch_weather` asks
Open-Meteo for `wind_speed_10m`), so the height every run's `Uref` has to be at."""


# -- the weather side ----------------------------------------------------------------


def sectors_of(site: dict[str, Any]) -> list[dict[str, Any]]:
    weather = site.get("weather") or {}
    if weather.get("failed"):
        raise Refused(
            f"refused: the site file's weather block records a failure rather than "
            f"data: {weather['failed']}"
        )
    sectors = weather.get("sectors") or []
    if not sectors:
        raise Refused(
            "refused: the site file has no wind sectors, so there is no rose in it "
            "and nothing to compose a year out of"
        )
    return sectors


def histogram_of(sector: dict[str, Any]) -> tuple[np.ndarray, float]:
    """Hours per speed bin, and the bin width.

    A site file written before the histogram existed carries only the percentiles,
    and four points on a curve cannot answer "how many hours between 6 and 7 m/s".
    So this refuses and names the one command that fixes it, rather than inventing a
    distribution that fits the percentiles.
    """
    histogram = sector.get("speed_10m_histogram")
    if not histogram or not histogram.get("hours"):
        raise Refused(
            "refused: this site file's sectors carry no speed histogram, only the "
            "mean and the percentiles, and an exceedance probability needs the hours "
            "per speed bin.\n"
            "  Re-fetch it on your own machine with a current OpenReynolds:\n"
            "    openreynolds site --lat <deg> --lon <deg> --radius <m> -o site.json\n"
            "    openreynolds push site.json"
        )
    hours = np.asarray(histogram["hours"], dtype=float)
    width = float(histogram.get("bin_width_ms") or 1.0)
    if width <= 0:
        raise Refused(f"refused: the histogram's bin_width_ms is {width!r}")
    return hours, width


def sector_exceedance(amplification: np.ndarray, hours: np.ndarray, width: float,
                      threshold: float) -> np.ndarray:
    """Hours in this sector whose speed at the point passes `threshold`.

    `u_required = threshold / K` is the 10 m speed it takes to reach the threshold
    where `K` amplifies. Within a bin the hours are spread evenly, so a bin spanning
    `u_required` contributes its upper part: `(hi - u_required) / width`, clipped to
    [0, 1]. A step at the bin centre instead would quantise the answer in jumps of a
    whole bin's hours, which at 1 m/s bins and a 5 m/s threshold is most of the
    variation being measured.

    `K <= 0` is a point in still air and exceeds nothing. `K` **not a number** is a
    point that is not in the flow at all -- inside a building, outside the domain --
    and stays not a number rather than becoming zero exceedance, which would paint
    every building as the calmest place on the site.
    """
    k = np.asarray(amplification, dtype=float)
    positive = k > 0  # false for NaN, which is what makes the NaN branch below safe
    required = np.where(positive, threshold / np.where(positive, k, 1.0), np.inf)
    upper = (np.arange(len(hours), dtype=float) + 1.0) * width
    # (points..., bins): the fraction of each bin above the required speed.
    fraction = np.clip((upper - required[..., None]) / width, 0.0, 1.0)
    counted = fraction @ hours
    return np.where(np.isnan(k), np.nan, counted)


def compose(amplification: dict[float, np.ndarray], sectors: list[dict[str, Any]],
            threshold: float) -> dict[str, Any]:
    """Every run's sector summed into one exceedance probability per point."""
    total_hours = float(sum(float(s.get("hours") or 0) for s in sectors))
    if total_hours <= 0:
        raise Refused("refused: the site file's sectors hold no hours at all")

    shapes = {np.asarray(v).shape for v in amplification.values()}
    if len(shapes) != 1:
        raise Refused(f"refused: the runs sampled onto different grids: {sorted(shapes)}")
    exceeded = np.zeros(shapes.pop(), dtype=float)
    covered = 0.0
    used = []
    claimed: dict[float, float] = {}
    for centre, k in sorted(amplification.items()):
        sector = nearest_sector(sectors, centre)
        # Two runs either side of a sector boundary both land on it, and the quiet
        # outcome is that sector's hours counted twice: a coverage above 1 and a
        # probability inflated by exactly the overlap.
        if sector["centre_deg"] in claimed:
            raise Refused(
                f"refused: the runs at {claimed[sector['centre_deg']]:g} and "
                f"{centre:g} degrees are both nearest the sector centred "
                f"{sector['centre_deg']:g}, so its hours would be counted twice. "
                f"One run per sector centre; the file's sectors are "
                f"{360.0 / len(sectors):g} degrees wide"
            )
        claimed[sector["centre_deg"]] = centre
        hours, width = histogram_of(sector)
        exceeded += sector_exceedance(k, hours, width, threshold)
        covered += float(sector.get("hours") or 0)
        used.append({
            "requested_deg": centre,
            "sector_centre_deg": sector["centre_deg"],
            "hours": sector.get("hours"),
            "frequency": sector.get("frequency"),
        })
    return {
        "threshold_ms": threshold,
        "probability": exceeded / total_hours,
        "total_hours": total_hours,
        "covered_hours": covered,
        "sector_coverage": round(covered / total_hours, 4),
        "sectors_used": used,
    }


def nearest_sector(sectors: list[dict[str, Any]], bearing: float) -> dict[str, Any]:
    return min(sectors, key=lambda s: abs(((s["centre_deg"] - bearing + 180) % 360) - 180))


def classify(probability: np.ndarray, bands: list[float]) -> np.ndarray:
    """A letter per point from the bands given: A below the first, then B, C, ...

    The letters are a presentation of your numbers. Nothing here knows what the bands
    of any published criterion are -- see this file's docstring.
    """
    letters = np.array([chr(ord("A") + i) for i in range(len(bands) + 1)])
    index = np.searchsorted(np.asarray(bands, dtype=float), probability * 100.0,
                            side="right")
    return letters[np.clip(index, 0, len(bands))]


# -- the CFD side --------------------------------------------------------------------


def reference_from_case(case: Path) -> dict[str, Any]:
    """`Uref`, `Zref` and `d` out of the ABL dictionary `site_abl.py --write` left.

    Reading them beats a flag because they are what the solver actually used: a
    `--uref` typed again at this end is a second chance to get it wrong.
    """
    path = case / "0" / site_abl.INCLUDE_PATH
    if not path.is_file():
        raise Refused(
            f"refused: no {path.as_posix()}, so this run's reference speed is "
            f"unknown. Either it was not set up by `site_abl.py --write`, or pass "
            f"--uref for every run."
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    found: dict[str, Any] = {"from": path.as_posix()}
    for key in ("Uref", "Zref", "d"):
        match = re.search(rf"^\s*{key}\s+([-\d.eE+]+)\s*;", text, re.M)
        if match:
            found[key] = float(match.group(1))
    if "Uref" not in found:
        raise Refused(f"refused: {path.as_posix()} names no Uref")
    return found


def check_reference(reference: dict[str, Any], case: Path) -> list[str]:
    """Whether this run's reference is the one the histogram is written against."""
    notes: list[str] = []
    zref = reference.get("Zref")
    if zref is not None and abs(float(zref) - REFERENCE_HEIGHT_M) > 1e-6:
        raise Refused(
            f"refused: {case.name} has Zref = {zref:g} m and the site file's speed "
            f"histogram is of the {REFERENCE_HEIGHT_M:g} m wind, so U_local/Uref is "
            f"not an amplification of the speeds being composed.\n"
            f"  Re-run the case with Zref = {REFERENCE_HEIGHT_M:g} m, or rescale "
            f"Uref to {REFERENCE_HEIGHT_M:g} m through the same log law first:\n"
            f"    python3 site_abl.py --uref <U at {REFERENCE_HEIGHT_M:g} m> "
            f"--zref {REFERENCE_HEIGHT_M:g} --z0 <z0> --print"
        )
    displacement = float(reference.get("d") or 0.0)
    if displacement:
        notes.append(
            f"{case.name}: d = {displacement:g} m, so Uref is the speed at "
            f"{REFERENCE_HEIGHT_M + displacement:g} m above the inlet patch's minimum "
            f"and not at {REFERENCE_HEIGHT_M:g} m (atmBoundaryLayer.C puts no d in "
            f"u*); the amplification is high by that much"
        )
    return notes


def sample_plane(case: Path, height: float, bounds: tuple[float, float, float, float],
                 n: int, time: float | None = None):
    """Horizontal speed on an n x n grid at `height`, and the valid-point mask.

    pyvista's `sample` marks a point outside the mesh, which at pedestrian height is
    every point inside a building, and those are masked rather than read as zero
    speed -- an unmasked building is a block of perfect calm in the middle of the map
    and reads as the most comfortable place on the site.
    """
    import render  # noqa: PLC0415  (pyvista lives behind it; see render.py's own note)

    pv = render._pyvista()
    block, chosen, _ = render.open_case(case, time)
    mesh = render.internal_mesh(block)
    if mesh is None or mesh.n_cells == 0:
        raise Refused(f"refused: no internal mesh in {case.as_posix()}")
    if "U" not in mesh.point_data and "U" in mesh.cell_data:
        mesh = mesh.cell_data_to_point_data(pass_cell_data=True)
    if "U" not in mesh.point_data:
        raise Refused(f"refused: {case.as_posix()} has no U field written")

    xmin, ymin, xmax, ymax = bounds
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    gx, gy = np.meshgrid(xs, ys)
    points = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, height)])
    probed = pv.PolyData(points).sample(mesh)

    velocity = np.asarray(probed.point_data["U"], dtype=float)
    speed = np.hypot(velocity[:, 0], velocity[:, 1])
    valid = np.asarray(probed.point_data["vtkValidPointMask"], dtype=bool)
    speed = np.where(valid, speed, np.nan)
    return speed.reshape(gx.shape), valid.reshape(gx.shape), xs, ys, float(chosen)


def case_bounds(case: Path, time: float | None = None) -> tuple[float, float, float, float]:
    import render  # noqa: PLC0415

    block, _, _ = render.open_case(case, time)
    mesh = render.internal_mesh(block)
    xmin, xmax, ymin, ymax, _, _ = mesh.bounds
    return float(xmin), float(ymin), float(xmax), float(ymax)


# -- figures -------------------------------------------------------------------------


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


STACK_EDGES_MS = (0.0, 2.0, 4.0, 6.0, 8.0)
"""Where the rose's stacked bands are cut, in m/s at 10 m. Five bands because twenty
of them are a gradient nobody can read off a legend; the hours behind them are the
file's own 1 m/s bins, unchanged, and `--json` prints those."""


def rose_bands(sector: dict[str, Any]) -> list[float]:
    """Hours per `STACK_EDGES_MS` band for one sector, from its 1 m/s bins."""
    hours, width = histogram_of(sector)
    centres = (np.arange(len(hours)) + 0.5) * width
    edges = list(STACK_EDGES_MS) + [math.inf]
    return [float(hours[(centres >= lo) & (centres < hi)].sum())
            for lo, hi in zip(edges[:-1], edges[1:])]


def draw_rose(sectors: list[dict[str, Any]], path: Path, title: str) -> None:
    """A polar stacked bar per sector: compass bearing, petal length a frequency.

    North at the top and clockwise, because that is how a compass reads and a rose
    drawn counter-clockwise is a mirror image of the site nobody notices.
    """
    plt = _matplotlib()
    total = sum(float(s.get("hours") or 0) for s in sectors) or 1.0
    width = math.radians(360.0 / len(sectors))
    theta = [math.radians(float(s["centre_deg"])) for s in sectors]
    stacks = np.array([rose_bands(s) for s in sectors]) / total * 100.0

    figure = plt.figure(figsize=(6.4, 6.4))
    axes = figure.add_subplot(111, projection="polar")
    axes.set_theta_zero_location("N")
    axes.set_theta_direction(-1)
    bottom = np.zeros(len(sectors))
    colours = plt.get_cmap("viridis")(np.linspace(0.15, 0.95, stacks.shape[1]))
    edges = list(STACK_EDGES_MS)
    labels = [f"{lo:g}-{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])]
    labels.append(f"{edges[-1]:g}+")
    for column in range(stacks.shape[1]):
        axes.bar(theta, stacks[:, column], width=width, bottom=bottom,
                 color=colours[column], edgecolor="white", linewidth=0.4,
                 label=labels[column])
        bottom += stacks[:, column]
    axes.set_xticks([math.radians(a) for a in range(0, 360, 45)])
    axes.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    axes.set_title(title, pad=24)
    axes.legend(title="m/s at 10 m", loc="lower right", bbox_to_anchor=(1.18, -0.08),
                fontsize="small")
    figure.text(0.5, 0.015, "petal length: % of all hours; direction the wind blows FROM",
                ha="center", fontsize="small")
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def draw_map(probability: np.ndarray, xs: np.ndarray, ys: np.ndarray, path: Path,
             threshold: float, bands: list[float] | None, site: dict[str, Any],
             coverage: float) -> None:
    plt = _matplotlib()
    figure, axes = plt.subplots(figsize=(7.4, 6.2))
    percent = probability * 100.0
    if bands:
        from matplotlib.colors import BoundaryNorm

        levels = [0.0] + [float(b) for b in bands] + [max(100.0, float(bands[-1]) * 2)]
        cmap = plt.get_cmap("RdYlGn_r", len(levels) - 1)
        mesh = axes.pcolormesh(xs, ys, percent, cmap=cmap,
                              norm=BoundaryNorm(levels, cmap.N), shading="auto")
        bar = figure.colorbar(mesh, ax=axes, ticks=levels[:-1] + [levels[-1]])
        bar.set_label(f"% of hours above {threshold:g} m/s  (bands: your numbers)")
    else:
        mesh = axes.pcolormesh(xs, ys, percent, cmap="RdYlGn_r", shading="auto")
        bar = figure.colorbar(mesh, ax=axes)
        bar.set_label(f"% of hours above {threshold:g} m/s")

    for ring in _footprint_rings(site):
        axes.plot([p[0] for p in ring], [p[1] for p in ring], color="black",
                  linewidth=0.6)
    axes.set_aspect("equal")
    axes.set_xlabel("x, metres east of the site origin")
    axes.set_ylabel("y, metres north of the site origin")
    name = (site.get("site") or {}).get("name") or ""
    axes.set_title(f"exceedance of {threshold:g} m/s"
                   + (f" -- {name}" if name else ""))
    if coverage < 0.999:
        figure.text(0.5, 0.015,
                    f"{coverage * 100:.0f}% of the year's hours were run: this is a "
                    f"lower bound, not the year", ha="center", fontsize="small")
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def _footprint_rings(site: dict[str, Any]) -> list[list[tuple[float, float]]]:
    """Closed footprint rings in the site frame, for outlining on the map. The site
    file already carries them in x-east/y-north metres (`site.py:local_frame`), so
    they land on the CFD grid without a second projection."""
    buildings = site.get("buildings") or {}
    rings = []
    for feature in buildings.get("features") or []:
        ring = feature.get("polygon_xy") or []
        points = [(float(p[0]), float(p[1])) for p in ring if len(p) >= 2]
        if len(points) >= 3:
            rings.append(points + [points[0]])
    return rings


# -- summary -------------------------------------------------------------------------


def area_summary(probability: np.ndarray, bands: list[float] | None) -> dict[str, Any]:
    valid = np.isfinite(probability)
    out: dict[str, Any] = {
        "points": int(probability.size),
        "points_in_the_flow": int(valid.sum()),
        "percent_of_hours": {
            "min": round(float(np.nanmin(probability[valid]) * 100), 3) if valid.any() else None,
            "mean": round(float(np.nanmean(probability[valid]) * 100), 3) if valid.any() else None,
            "max": round(float(np.nanmax(probability[valid]) * 100), 3) if valid.any() else None,
        },
    }
    if bands and valid.any():
        letters = classify(probability[valid], bands)
        unique, counts = np.unique(letters, return_counts=True)
        out["area_fraction_by_class"] = {
            str(letter): round(float(count) / float(valid.sum()), 4)
            for letter, count in zip(unique, counts)
        }
        out["bands_percent"] = list(bands)
        out["bands_are"] = ("yours, not a published criterion's: nothing in this "
                            "script knows any standard's numbers")
    return out


def parse_runs(values: list[str]) -> dict[float, Path]:
    runs: dict[float, Path] = {}
    for value in values:
        if "=" not in value:
            raise Refused(f"refused: --run wants BEARING=CASE, got {value!r}")
        bearing, case = value.split("=", 1)
        try:
            degrees = float(bearing)
        except ValueError as exc:
            raise Refused(f"refused: {bearing!r} is not a bearing in degrees") from exc
        path = Path(case).expanduser()
        if not path.is_dir():
            raise Refused(f"refused: no case directory at {path.as_posix()}")
        runs[degrees % 360.0] = path
    return runs


def synthetic_site(hours_by_sector: dict[float, list[float]]) -> dict[str, Any]:
    """A site file with the hours stated rather than fetched, for `--check`.

    Twelve 30-degree sectors, because that is what `site.py` writes and what
    `nearest_sector` expects to be able to divide 360 by. A sector absent from
    `hours_by_sector` is a real thing -- a direction the wind never blew from -- and
    carries a histogram of zeros rather than no histogram, which is what the file
    would hold for it.
    """
    sectors = []
    total = sum(sum(v) for v in hours_by_sector.values())
    for index in range(12):
        centre = index * 30.0
        bins = hours_by_sector.get(centre, [])
        counted = sum(bins)
        sectors.append({
            "centre_deg": centre,
            "hours": counted,
            "frequency": (counted / total) if total else 0.0,
            "speed_10m_histogram": {"hours": list(bins) or [0.0],
                                    "bin_width_ms": 1.0},
        })
    return {"weather": {"hours": total, "sector_width_deg": 30.0, "sectors": sectors}}


def formula_check(threshold: float = 5.0) -> int:
    """Compose a known year and compare with the closed form.

    The year: 600 hours from 240 degrees and 400 from 270, spread over 1 m/s bins as
    100 hours in each of 0-1 ... 5-6 for the first and 100 in each of 2-3 ... 5-6 for
    the second. Nothing is fitted and nothing is read.

    At K = 1 and a 5 m/s threshold the answer is arithmetic: only the 5-6 bin is
    entirely above 5, so 100 + 100 = 200 of 1000 hours, exactly 0.2.

    At K = 2 the required 10 m speed is 2.5, so the 2-3 bin contributes its upper
    half: sector one gives 50 + 100 + 100 + 100 = 350 and sector two 50 + 100 + 100 +
    100 = 350, which is 700 of 1000, exactly 0.7. That is the half-bin spreading as
    well as the amplification, and a step at the bin centre would give 0.6 or 0.8.

    Exits 1 on any disagreement beyond 1e-12, because both targets are exact.
    """
    one = [100.0] * 6                      # 0-1 .. 5-6 m/s, 600 hours from 240
    two = [0.0, 0.0] + [100.0] * 4         # 2-3 .. 5-6 m/s, 400 hours from 270
    site = synthetic_site({240.0: one, 270.0: two})
    sectors = sectors_of(site)

    bad = False
    for k_value, want in ((1.0, 0.2), (2.0, 0.7)):
        k = np.full((3, 3), k_value, dtype=float)
        found = compose({240.0: k, 270.0: k}, sectors, threshold)
        got = float(np.asarray(found["probability"]).ravel()[0])
        spread = float(np.ptp(np.asarray(found["probability"])))
        off = got - want
        print(f"  K = {k_value:g}: exceedance {got:.12f}, exact {want:.12f} "
              f"({off:+.2e}), uniform across the grid to {spread:.2e}")
        bad = bad or abs(off) > 1e-12 or spread > 1e-12
        if abs(found["sector_coverage"] - 1.0) > 1e-12:
            print(f"  sector coverage {found['sector_coverage']} and not 1")
            bad = True

    # A point not in the flow stays not in the flow: NaN must not read as calm.
    k = np.full((3, 3), 1.0)
    k[1, 1] = np.nan
    found = compose({240.0: k, 270.0: k}, sectors, threshold)
    centre = float(np.asarray(found["probability"])[1, 1])
    print(f"  a NaN point reads {centre} (not 0, which would paint it the calmest)")
    bad = bad or not math.isnan(centre)

    print("DISAGREES" if bad else "agrees")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A wind rose, and a pedestrian comfort exceedance map from "
                    "per-direction runs.")
    parser.add_argument("--site", type=Path, default=Path("site.json"),
                        help="The file `openreynolds site` wrote. Default site.json.")
    parser.add_argument("--check", action="store_true",
                        help="compose a year whose answer is known and print it beside "
                             "the closed form; needs no site file and no runs")
    parser.add_argument("--rose", type=Path, help="Draw the wind rose here.")
    parser.add_argument("--run", action="append", default=[], metavar="BEARING=CASE",
                        help="A solved case and the bearing the wind blew FROM in it. "
                             "Repeat once per direction.")
    parser.add_argument("--threshold", type=float,
                        help="The speed whose exceedance is measured, m/s at "
                             "pedestrian height. No default: see --help's note.")
    parser.add_argument("--pedestrian-z", type=float,
                        help="The height the map is sampled at, metres. No default: "
                             "1.75 m and 1.5 m are both in circulation and the "
                             "standards that settle it could not be read here.")
    parser.add_argument("--bands", default="",
                        help="Comma-separated class boundaries in percent of hours, "
                             "e.g. 2.5,5,10,20 -- yours to supply.")
    parser.add_argument("--uref", type=float,
                        help="Override the reference speed read from each case's "
                             "0/include/ABLConditions, m/s at 10 m.")
    parser.add_argument("--grid", type=int, default=120,
                        help="Points per side of the sampling grid. Default 120.")
    parser.add_argument("--bounds", type=float, nargs=4,
                        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                        help="Sample this box instead of the first case's own extent.")
    parser.add_argument("--time", type=float, help="Write time to read. Default latest.")
    parser.add_argument("--map", type=Path, help="Draw the exceedance map here.")
    parser.add_argument("--csv", type=Path, help="Write the grid as x,y,percent here.")
    parser.add_argument("--json", action="store_true", help="Print the findings as JSON.")
    args = parser.parse_args(argv)

    if args.check:
        print('composing a year whose answer is known: 600 h from 240 deg, 400 h from 270,')
        print('1 m/s bins, threshold 5 m/s, amplification uniform.')
        return formula_check()

    try:
        site = site_abl.load_site(args.site)
        sectors = sectors_of(site)
        bands = [float(b) for b in args.bands.replace(" ", "").split(",") if b] \
            if args.bands else None

        out: dict[str, Any] = {
            "site": (site.get("site") or {}).get("name"),
            "sector_width_deg": (site.get("weather") or {}).get("sector_width_deg"),
            "hours": (site.get("weather") or {}).get("hours"),
            "rose": [
                {"centre_deg": s["centre_deg"], "hours": s.get("hours"),
                 "frequency": s.get("frequency"),
                 "mean_speed_10m": (s.get("speed_10m") or {}).get("mean"),
                 "hours_per_speed_bin": (s.get("speed_10m_histogram") or {}).get("hours")}
                for s in sectors
            ],
            "notes": [],
        }

        if args.rose:
            draw_rose(sectors, args.rose,
                      f"wind rose -- {(site.get('site') or {}).get('name') or ''}")
            out["rose_png"] = args.rose.as_posix()

        if args.run:
            if args.threshold is None:
                raise Refused(
                    "refused: --threshold is the speed whose exceedance is being "
                    "measured and there is no default for it. The mechanism is here; "
                    "the number is yours. NEN 8100 grades on exceedance of 5 m/s "
                    "(confirmed from Willemsen & Wisse's abstract, J. Wind Eng. Ind. "
                    "Aerodyn. 95:1541-1550, the authors of the code); the standard "
                    "itself and Lawson (1978) could not be read from here, so no "
                    "table is written into this script."
                )
            if args.pedestrian_z is None:
                raise Refused(
                    "refused: --pedestrian-z is the height the map is sampled at and "
                    "there is no default. 1.75 m and 1.5 m are both in circulation "
                    "for pedestrian level and the primary texts that settle it were "
                    "not readable here, so this asks rather than picks."
                )
            runs = parse_runs(args.run)
            bounds = tuple(args.bounds) if args.bounds else \
                case_bounds(next(iter(runs.values())), args.time)
            amplification: dict[float, np.ndarray] = {}
            per_run = []
            xs = ys = None
            for bearing, case in sorted(runs.items()):
                if args.uref is not None:
                    reference = {"Uref": args.uref, "from": "--uref"}
                else:
                    reference = reference_from_case(case)
                    out["notes"].extend(check_reference(reference, case))
                uref = float(reference["Uref"])
                if uref <= 0:
                    raise Refused(f"refused: {case.name} has Uref = {uref:g}")
                speed, valid, xs, ys, chosen = sample_plane(
                    case, args.pedestrian_z, bounds, args.grid, args.time)
                amplification[bearing] = speed / uref
                per_run.append({
                    "bearing_deg": bearing,
                    "case": case.as_posix(),
                    "time": chosen,
                    "uref_ms": uref,
                    "uref_from": reference.get("from"),
                    "points_in_the_flow": int(valid.sum()),
                    "max_amplification": round(float(np.nanmax(speed / uref)), 4),
                })
            composed = compose(amplification, sectors, args.threshold)
            probability = composed.pop("probability")
            out["comfort"] = {**composed, "runs": per_run,
                              "pedestrian_z_m": args.pedestrian_z,
                              "speed": "horizontal, sqrt(Ux^2 + Uy^2)",
                              **area_summary(probability, bands)}
            if composed["sector_coverage"] < 0.999:
                out["notes"].append(
                    f"only {composed['sector_coverage'] * 100:.1f}% of the year's "
                    f"hours blew from a sector that was run, so every probability "
                    f"here is a lower bound: the hours from the other sectors are "
                    f"counted as never exceeding"
                )
            if args.map:
                draw_map(probability, xs, ys, args.map, args.threshold, bands, site,
                         composed["sector_coverage"])
                out["comfort"]["map_png"] = args.map.as_posix()
            if args.csv:
                gx, gy = np.meshgrid(xs, ys)
                rows = np.column_stack([gx.ravel(), gy.ravel(),
                                        probability.ravel() * 100.0])
                with args.csv.open("w", encoding="utf-8", newline="") as handle:
                    handle.write("x_m,y_m,percent_of_hours_above_threshold\n")
                    for x, y, value in rows:
                        handle.write(f"{x:.4f},{y:.4f},{value:.4f}\n")
                out["comfort"]["csv"] = args.csv.as_posix()
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(out, indent=2, default=str))
    else:
        print(report(out))
    return 0


def report(out: dict[str, Any]) -> str:
    lines = [f"site: {out.get('site')}  {out.get('hours')} hours of weather"]
    rose = out.get("rose") or []
    if rose:
        windiest = max(rose, key=lambda s: s.get("hours") or 0)
        lines.append(
            f"prevailing sector {windiest['centre_deg']:g} deg: "
            f"{(windiest.get('frequency') or 0) * 100:.1f}% of hours, mean "
            f"{windiest.get('mean_speed_10m')} m/s at 10 m"
        )
    if out.get("rose_png"):
        lines.append(f"rose: {out['rose_png']}")
    comfort = out.get("comfort")
    if comfort:
        percent = comfort.get("percent_of_hours") or {}
        lines.append(
            f"exceedance of {comfort['threshold_ms']:g} m/s at z = "
            f"{comfort['pedestrian_z_m']:g} m, from {len(comfort['runs'])} "
            f"direction(s) covering {comfort['sector_coverage'] * 100:.1f}% of hours:"
        )
        lines.append(f"  {percent.get('min')}% to {percent.get('max')}% of hours, "
                     f"mean {percent.get('mean')}%")
        for letter, fraction in (comfort.get("area_fraction_by_class") or {}).items():
            lines.append(f"  class {letter}: {fraction * 100:.1f}% of the sampled area")
        for key in ("map_png", "csv"):
            if comfort.get(key):
                lines.append(f"  {comfort[key]}")
    for note in out.get("notes") or []:
        lines.append(f"note: {note}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
