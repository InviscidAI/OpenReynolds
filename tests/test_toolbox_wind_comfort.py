"""`wind_comfort.py`: the exceedance arithmetic, and the histogram it needs.

What is worth pinning here is not the plotting. It is the three places where this
answer can be silently wrong and still look like a map:

  * the **hours** behind it. An exceedance probability is a count of hours, so it is
    checkable by hand: ten hours in the 2-3 m/s bin, a threshold of 2.5 m/s and no
    amplification is five hours, not ten and not one.
  * the **points that are not in the flow**. A pedestrian-height plane cuts through
    buildings, and a point inside one has no velocity. Read as zero speed it becomes
    the calmest place on the site and the map says the inside of a tower block is
    where to put the cafe.
  * the **reference the amplification is against**. `K = U_local/Uref` composes with
    a histogram of the 10 m wind only if `Uref` is a 10 m speed; a case run with
    `Zref = 2` makes every number here wrong by a log-law factor and nothing in the
    output would look odd.

The class letters are deliberately not pinned against any published criterion,
because this repository does not have one it could read: see the script's docstring
and `benchmarks/README.md` on reference values taken from secondary sources.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load(name: str):
    # The scripts import each other as siblings (`wind_comfort` imports `site_abl`),
    # which is what the toolbox directory is: scripts on a path, not a package.
    sys.path.insert(0, str(TOOLBOX))
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def comfort():
    return load("wind_comfort")


@pytest.fixture(scope="module")
def site_module():
    sys.path.insert(0, str(TOOLBOX.parents[1]))
    from openreynolds import site

    return site


def sector(centre: float, hours: list[int], width: float = 1.0) -> dict:
    return {
        "centre_deg": centre,
        "hours": sum(hours),
        "frequency": 1.0,
        "speed_10m": {"hours": sum(hours), "mean": 5.0},
        "speed_10m_histogram": {"bin_width_ms": width, "bins": len(hours),
                                "hours": hours},
    }


# -- the histogram the site file now carries ---------------------------------------


def test_the_sector_histogram_counts_the_hours_a_percentile_cannot_answer_for(site_module):
    """Four percentiles cannot say how many hours blew between 6 and 7 m/s, and an
    exceedance of a threshold is exactly that question asked bin by bin."""
    hourly = {
        "wind_speed_10m": [0.5, 2.5, 2.9, 6.2, 6.8, 9.0],
        "wind_speed_100m": [1.0, 4.0, 4.5, 9.0, 10.0, 14.0],
        "wind_direction_10m": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    out = site_module.wind_summary(hourly)
    histogram = out["sectors"][0]["speed_10m_histogram"]

    assert histogram["bin_width_ms"] == 1.0
    assert sum(histogram["hours"]) == out["sectors"][0]["hours"] == 6
    assert histogram["hours"][0] == 1      # 0.5
    assert histogram["hours"][2] == 2      # 2.5 and 2.9
    assert histogram["hours"][6] == 2      # 6.2 and 6.8
    assert histogram["hours"][9] == 1      # 9.0


def test_every_sector_is_given_the_same_speed_axis(site_module):
    """A rose stacks the sectors and an exceedance sums across them; both are wrong
    if sector 1's bin 7 is a different speed from sector 2's."""
    hourly = {
        "wind_speed_10m": [1.0, 12.0],
        "wind_speed_100m": [2.0, 20.0],
        "wind_direction_10m": [0.0, 90.0],
    }
    out = site_module.wind_summary(hourly)
    widths = {s["speed_10m_histogram"]["bins"] for s in out["sectors"]}
    assert widths == {13}, "one axis, long enough for the fastest hour anywhere"


# -- the exceedance ----------------------------------------------------------------


def test_the_exceedance_of_a_threshold_is_the_hours_above_it(comfort):
    hours = np.array([0.0, 0.0, 10.0, 0.0])      # ten hours in the 2-3 m/s bin
    unity = np.array([1.0])

    assert comfort.sector_exceedance(unity, hours, 1.0, 2.0)[0] == pytest.approx(10.0)
    assert comfort.sector_exceedance(unity, hours, 1.0, 2.5)[0] == pytest.approx(5.0)
    assert comfort.sector_exceedance(unity, hours, 1.0, 3.0)[0] == pytest.approx(0.0)


def test_an_amplification_of_two_halves_the_wind_it_takes_to_exceed(comfort):
    """A corner that doubles the wind reaches 5 m/s on a 2.5 m/s hour, which is the
    whole reason a comfort map is a CFD question and not a weather question."""
    hours = np.array([0.0, 0.0, 10.0, 0.0])
    doubled = comfort.sector_exceedance(np.array([2.0]), hours, 1.0, 5.0)
    plain = comfort.sector_exceedance(np.array([1.0]), hours, 1.0, 5.0)

    assert doubled[0] == pytest.approx(5.0)      # required 2.5 m/s: half of that bin
    assert plain[0] == pytest.approx(0.0)        # required 5 m/s: nothing that fast


def test_a_point_that_is_not_in_the_flow_stays_not_a_number_rather_than_calm(comfort):
    hours = np.array([0.0, 0.0, 10.0])
    out = comfort.sector_exceedance(np.array([np.nan, 1.0, 0.0]), hours, 1.0, 2.0)

    assert np.isnan(out[0]), "inside a building is unknown, not perfectly comfortable"
    assert out[1] == pytest.approx(10.0)
    assert out[2] == pytest.approx(0.0), "still air exceeds nothing"


def test_the_probability_is_over_the_whole_year_not_only_the_sectors_that_were_run(comfort):
    """Four runs out of twelve are not four twelfths of an answer. The hours from the
    sectors nobody ran count as never exceeding, so the result is a lower bound, and
    `sector_coverage` is what says so."""
    sectors = [sector(0.0, [0, 0, 10]), sector(90.0, [0, 0, 30])]
    composed = comfort.compose({0.0: np.array([1.0])}, sectors, 2.0)

    assert composed["total_hours"] == 40.0
    assert composed["covered_hours"] == 10.0
    assert composed["sector_coverage"] == pytest.approx(0.25)
    assert composed["probability"][0] == pytest.approx(10.0 / 40.0)


def test_two_runs_on_different_grids_are_refused_rather_than_broadcast(comfort):
    sectors = [sector(0.0, [0, 0, 10]), sector(90.0, [0, 0, 10])]
    with pytest.raises(comfort.Refused) as raised:
        comfort.compose({0.0: np.zeros((4, 4)), 90.0: np.zeros((5, 5))}, sectors, 2.0)
    assert "different grids" in str(raised.value)


def test_a_site_file_without_a_histogram_is_refused_and_names_the_fix(comfort):
    """Files fetched before the histogram existed carry only the percentiles. Fitting
    a distribution to five points would answer anyway, and wrongly."""
    old = {"centre_deg": 0.0, "hours": 10, "speed_10m": {"mean": 5.0, "p90": 9.0}}
    with pytest.raises(comfort.Refused) as raised:
        comfort.histogram_of(old)
    assert "openreynolds site" in str(raised.value)


# -- the classes are the user's numbers --------------------------------------------


def test_the_class_letters_come_from_the_bands_given_and_nowhere_else(comfort):
    bands = [2.5, 5.0, 10.0, 20.0]
    probability = np.array([0.001, 0.025, 0.04, 0.07, 0.15, 0.40])

    letters = comfort.classify(probability, bands)

    assert list(letters) == ["A", "B", "B", "C", "D", "E"]


def test_the_script_carries_no_criterion_table_of_its_own(comfort):
    """The point of the whole design: two reference values in this repository were
    wrong because they came from secondary sources (`benchmarks/README.md`), so the
    criterion's numbers are the user's and the mechanism is the deliverable. This
    fails the moment a `NEN_BANDS` or a `LAWSON` appears at module level."""
    constants = sorted(name for name in dir(comfort) if name.isupper())

    assert constants == ["REFERENCE_HEIGHT_M", "STACK_EDGES_MS"], (
        "the only constants here are the height the weather is measured at and where "
        "the rose's legend is cut; a threshold or a band is not one of them"
    )
    text = (TOOLBOX / "wind_comfort.py").read_text(encoding="utf-8")
    assert "could not be read" in text
    assert "NEN 8100" in text and "Lawson" in text


# -- the reference the amplification is against ------------------------------------


def write_abl(case: Path, uref: float, zref: float, d: float = 0.0) -> None:
    path = case / "0" / "include"
    path.mkdir(parents=True, exist_ok=True)
    (path / "ABLConditions").write_text(
        f"Uref                 {uref:g};\nZref                 {zref:g};\n"
        f"d                    {d:g};\nz0                   uniform 1;\n",
        encoding="utf-8")


def test_the_reference_speed_is_read_from_the_dictionary_the_solver_used(comfort, tmp_path):
    case = tmp_path / "sw"
    write_abl(case, 7.5, 10.0)
    found = comfort.reference_from_case(case)
    assert found["Uref"] == 7.5 and found["Zref"] == 10.0


def test_a_case_whose_reference_is_not_at_ten_metres_is_refused(comfort, tmp_path):
    case = tmp_path / "sw"
    write_abl(case, 7.5, 2.0)
    with pytest.raises(comfort.Refused) as raised:
        comfort.check_reference(comfort.reference_from_case(case), case)
    assert "Zref = 2" in str(raised.value) and "site_abl.py" in str(raised.value)


def test_a_displacement_height_is_reported_because_uref_is_then_higher_up(comfort, tmp_path):
    """`atmBoundaryLayer.C` puts no `d` in `u*`, so with d = 5 the Uref asked for is
    the speed at 15 m and the amplification is high by the difference."""
    case = tmp_path / "sw"
    write_abl(case, 7.5, 10.0, d=5.0)
    notes = comfort.check_reference(comfort.reference_from_case(case), case)
    assert any("d = 5" in note for note in notes)


def test_a_case_with_no_abl_dictionary_is_refused_by_name(comfort, tmp_path):
    case = tmp_path / "sw"
    case.mkdir()
    with pytest.raises(comfort.Refused) as raised:
        comfort.reference_from_case(case)
    assert "--uref" in str(raised.value)


# -- the figures and the CLI -------------------------------------------------------


def full_site(tmp_path: Path) -> Path:
    data = {
        "schema": "openreynolds.site/1",
        "site": {"name": "test", "latitude": 51.5, "longitude": -0.1},
        "weather": {"hours": 40, "sector_width_deg": 30.0,
                    "sectors": [sector(c * 30.0, [1, 2, 3, 2, 1]) for c in range(12)]},
    }
    path = tmp_path / "site.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_the_rose_is_drawn_from_the_measured_hours(comfort, tmp_path):
    out = tmp_path / "rose.png"
    assert comfort.main(["--site", str(full_site(tmp_path)), "--rose", str(out)]) == 0
    assert out.stat().st_size > 1000


def test_the_rose_bands_add_up_to_the_sector_hours(comfort):
    """Five legend bands out of about twenty 1 m/s bins: the grouping is for the eye
    and must not lose or duplicate an hour."""
    one = sector(0.0, [1, 2, 3, 2, 1, 4, 5])
    assert sum(comfort.rose_bands(one)) == pytest.approx(18.0)


def test_a_run_without_a_threshold_is_refused_and_says_whose_number_it_is(
        comfort, tmp_path, capsys):
    case = tmp_path / "sw"
    write_abl(case, 7.5, 10.0)
    code = comfort.main(["--site", str(full_site(tmp_path)), "--run", f"240={case}"])
    message = capsys.readouterr().err

    assert code == 2
    assert "NEN 8100" in message and "Lawson" in message


def test_a_run_without_a_pedestrian_height_is_refused(comfort, tmp_path, capsys):
    case = tmp_path / "sw"
    write_abl(case, 7.5, 10.0)
    code = comfort.main(["--site", str(full_site(tmp_path)), "--threshold", "5",
                         "--run", f"240={case}"])
    assert code == 2
    assert "1.75" in capsys.readouterr().err


def test_a_site_file_that_is_not_one_is_refused(comfort, tmp_path):
    path = tmp_path / "site.json"
    path.write_text('{"schema": "something.else/1"}', encoding="utf-8")
    assert comfort.main(["--site", str(path)]) == 2


# -- the whole chain, through the reader the workspace uses -------------------------
#
# The arithmetic above is pinned on arrays. What is left is the part that only a real
# case exercises: reading one, slicing it at pedestrian height, masking the points
# that are not in the flow, and coming out the other side as a map and a CSV. The box
# mesh and its fields are `test_toolbox_locate.py`'s, the same fixture `probe.py`'s
# tests read through pyvista.


def _box_case(root: Path) -> Path:
    pytest.importorskip("pyvista", reason="wind_comfort reads cases through pyvista",
                        exc_type=ImportError)
    spec = importlib.util.spec_from_file_location(
        "wind_comfort_box_fixture", Path(__file__).resolve().parent / "test_toolbox_locate.py")
    fixture = importlib.util.module_from_spec(spec)
    sys.modules["wind_comfort_box_fixture"] = fixture
    spec.loader.exec_module(fixture)

    case = root / "sw"
    fixture.write_box_mesh(case, 10, 4, 1, (0.1, 0.1, 0.1))
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "system" / "controlDict").write_text("application simpleFoam;\n",
                                                 encoding="utf-8")
    cells = [(i, j) for j in range(4) for i in range(10)]
    time = case / "100"
    time.mkdir(parents=True, exist_ok=True)
    boundary = " ".join(f"{p} {{ type zeroGradient; }}"
                        for p in ("inlet", "outlet", "walls"))
    # U_x = 1 + j m/s, so the wind is four times faster along the top row than the
    # bottom one: a shear layer standing in for a street that speeds up.
    values = "\n".join(f"({1 + j} 0 0)" for _i, j in cells)
    (time / "U").write_text(
        "FoamFile { version 2.0; format ascii; class volVectorField; object U; }\n"
        "dimensions [0 1 -1 0 0 0 0];\n"
        f"internalField nonuniform List<vector> {len(cells)}\n(\n{values}\n);\n"
        f"boundaryField {{ {boundary} }}\n", encoding="utf-8")
    write_abl(case, 2.0, 10.0)
    return case


def test_the_map_is_higher_where_the_run_says_the_wind_is_faster(comfort, tmp_path):
    case = _box_case(tmp_path)
    data = {
        "schema": "openreynolds.site/1",
        "site": {"name": "box"},
        # Ten hours, all from one sector, all between 2 and 3 m/s.
        "weather": {"hours": 10, "sector_width_deg": 30.0,
                    "sectors": [sector(240.0, [0, 0, 10])]},
    }
    site_file = tmp_path / "site.json"
    site_file.write_text(json.dumps(data), encoding="utf-8")
    png, csv = tmp_path / "comfort.png", tmp_path / "comfort.csv"

    code = comfort.main([
        "--site", str(site_file), "--run", f"240={case}", "--threshold", "5",
        "--pedestrian-z", "0.05", "--grid", "5",
        "--bounds", "0.05", "0.05", "0.95", "0.35",
        "--map", str(png), "--csv", str(csv), "--json"])

    assert code == 0
    assert png.stat().st_size > 1000
    rows = [line.split(",") for line in
            csv.read_text(encoding="utf-8").strip().splitlines()[1:]]
    assert len(rows) == 25
    by_y = {}
    for x, y, percent in rows:
        by_y.setdefault(round(float(y), 3), []).append(float(percent))
    low, high = min(by_y), max(by_y)
    assert max(by_y[high]) > max(by_y[low]), (
        "the faster row has to come out as the less comfortable one, or the "
        "amplification has been applied upside down"
    )
    assert max(by_y[high]) <= 100.0 and min(by_y[low]) >= 0.0


def test_a_point_outside_the_domain_is_not_counted_as_calm(comfort, tmp_path):
    """The reason the mask matters: a pedestrian plane cuts through buildings, and a
    point inside one reads as zero velocity unless it is masked."""
    case = _box_case(tmp_path)
    speed, valid, xs, ys, _time = comfort.sample_plane(
        case, 0.05, (-5.0, -5.0, 5.0, 5.0), 7)

    assert valid.any() and not valid.all(), "the box is smaller than the box asked for"
    assert np.isnan(speed[~valid]).all()
    assert (speed[valid] > 0).all()
