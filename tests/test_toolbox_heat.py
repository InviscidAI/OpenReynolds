"""`heat.py`: the heat a run moved, read back off what it wrote.

Nothing in the toolbox computed a heat rate, a Nusselt number or the location of a
hot spot before this. `ladder.py` carried the answers -- Fourier through a slab, the
Graetz eigenvalues, Dittus-Boelter -- and no instrument to measure anything against
them, so a heated run ended with a picture of a temperature field and no number.

Every fixture is hand-written so the right answer is known before the script is
asked. The mesh is `test_toolbox_layer_report.py`'s ascii box, whose wall patch has
eight faces with areas 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15 -- so its area is
exactly 1 m2 and the whole box's boundary is exactly 6 m2. That makes the heat rates
arithmetic a reader can check in their head, and it is what lets the area-weighted
mean be told apart from the plain mean of the face values (800 against 750 on the
graded flux below), which is the whole reason the areas are read at all.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
TOOLBOX = HERE.parents[0] / "openreynolds" / "toolbox"

sys.path.insert(0, str(HERE))
from test_toolbox_layer_report import LAYERED, XS, YS, write_box  # noqa: E402


@pytest.fixture(scope="module")
def heat():
    """Imported by path, like every other toolbox script: the directory is data."""
    spec = importlib.util.spec_from_file_location("toolbox_heat", TOOLBOX / "heat.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# -- fixtures ------------------------------------------------------------------


#: The wall patch's eight face areas, in the order `write_box` writes them.
WALL_AREAS = [0.1] * 4 + [0.15] * 4
WALL_AREA = sum(WALL_AREAS)          # 1.0 m2
OUTER_AREA = 6.0 - WALL_AREA         # the rest of a unit box: 5.0 m2


def box(root: Path, name: str = "case", rename_wall: str | None = None) -> Path:
    """The ascii box, optionally with its `wall` patch renamed.

    `write_box` puts every face that is not the named side on one patch called
    `outer` and the rest on one called `wall`. Renaming is two lines here and keeps
    the tests readable: a patch that stands in for an inlet should not be called
    `wall` three assertions later.
    """
    case = write_box(root / name, XS, YS, LAYERED)
    if rename_wall:
        path = case / "constant" / "polyMesh" / "boundary"
        path.write_text(path.read_text().replace("    wall\n", f"    {rename_wall}\n"))
    return case


def write_field(path: Path, cls: str, entries: dict[str, str],
                internal: str = "uniform 0") -> Path:
    """A field file with one `boundaryField` entry per patch given."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = "\n".join(
        f"    {name}\n    {{\n        type            calculated;\n"
        f"        value           {value};\n    }}"
        for name, value in entries.items()
    )
    path.write_text(
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
        f"    class       {cls};\n    object      {path.name};\n}}\n\n"
        "dimensions      [0 0 0 0 0 0 0];\n\n"
        f"internalField   {internal};\n\n"
        f"boundaryField\n{{\n{blocks}\n}}\n"
    )
    return path


def scalars(values) -> str:
    return (
        f"nonuniform List<scalar> {len(values)}("
        + " ".join(f"{value:g}" for value in values) + ")"
    )


def vectors(values) -> str:
    return (
        f"nonuniform List<vector> {len(values)}("
        + "".join(f"({x:g} {y:g} {z:g})" for x, y, z in values) + ")"
    )


HEADER = "# Wall heat flux\n# Time\tpatch\tmin\tmax\tintegral\n"


def write_dat(path: Path, rows, header: str = HEADER) -> Path:
    """A `wallHeatFlux.dat` as the function object writes one: a patch NAME in column
    two, which is why `results.parse_dat` cannot read this file and its header reader
    can -- the same wall `wall.py` met on `yPlus.dat`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "".join(
        f"{time:g}\t{patch}\t{low:g}\t{high:g}\t{integral:g}\n"
        for time, patch, low, high, integral in rows
    ))
    return path


def properties(case: Path, body: str, name: str = "transportProperties") -> None:
    (case / "constant").mkdir(parents=True, exist_ok=True)
    (case / "constant" / name).write_text(
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
        f"    class dictionary;\n    object {name};\n}}\n\n" + body
    )


#: A graded wall flux: 500 W/m2 over the four 0.1 m2 faces and 1000 over the four
#: 0.15 m2 ones. The heat rate is 4*500*0.1 + 4*1000*0.15 = 800 W, so the
#: area-weighted mean is 800 W/m2 and the plain mean of the eight values is 750.
GRADED = [500.0] * 4 + [1000.0] * 4
GRADED_RATE = 800.0


def flux_case(root: Path, name: str = "case") -> Path:
    case = box(root, name)
    write_field(case / "0.5" / "wallHeatFlux", "volScalarField",
                {"wall": scalars(GRADED), "outer": "uniform 100"})
    return case


# -- the heat rate per patch ---------------------------------------------------


def test_the_patches_heat_rates_add_up_to_the_total(heat, tmp_path):
    """800 W through a 1 m2 graded wall and 100 W/m2 through the 5 m2 remainder is
    1300 W, and the total the report prints is the sum of the rows above it."""
    found = heat.measure(flux_case(tmp_path))

    (region,) = found["regions"]
    (source,) = region["sources"]
    assert source["source"] == "written wallHeatFlux field"
    rows = {row["patch"]: row for row in source["patches"]}
    assert rows["wall"]["area"] == pytest.approx(WALL_AREA)
    assert rows["wall"]["heat_rate"] == pytest.approx(GRADED_RATE)
    assert rows["outer"]["area"] == pytest.approx(OUTER_AREA)
    assert rows["outer"]["heat_rate"] == pytest.approx(100.0 * OUTER_AREA)

    total = heat.total_heat_rate(source)
    assert total == pytest.approx(GRADED_RATE + 100.0 * OUTER_AREA)
    assert total == pytest.approx(sum(row["heat_rate"] for row in source["patches"]))


def test_the_mean_flux_is_area_weighted_and_not_the_mean_of_the_values(heat, tmp_path):
    """The reason the mesh is read at all. The eight face values average 750 W/m2;
    weighted by the areas they actually act over the wall is at 800, and on a real
    graded wall the gap is larger than this one."""
    (region,) = heat.measure(flux_case(tmp_path))["regions"]
    (row,) = [r for r in heat.primary(region)["patches"] if r["patch"] == "wall"]

    assert row["mean_flux"] == pytest.approx(800.0)
    assert float(np.mean(GRADED)) == pytest.approx(750.0)
    assert row["min"] == pytest.approx(500.0)
    assert row["max"] == pytest.approx(1000.0)
    assert row["faces"] == 8


def test_the_function_objects_own_integral_is_read_rather_than_formed(heat, tmp_path):
    """Where the run asked for `wallHeatFlux`, the integral is already in the file.
    The columns come off the header through `results.split_header_names`, and the
    area-weighted mean is that integral over the patch area out of the mesh."""
    case = box(tmp_path)
    write_dat(case / "postProcessing" / "wallHeatFlux" / "0" / "wallHeatFlux.dat",
              [(0.5, "wall", 500, 1000, 800)])

    (region,) = heat.measure(case)["regions"]
    (source,) = region["sources"]
    assert source["source"] == "wallHeatFlux function object"
    (row,) = source["patches"]
    assert row["heat_rate"] == pytest.approx(800.0)
    assert row["area"] == pytest.approx(WALL_AREA)
    assert row["mean_flux"] == pytest.approx(800.0)


def test_a_header_in_another_order_is_still_read_correctly(heat, tmp_path):
    """Assuming column positions is how an integral in watts gets reported as a
    maximum flux."""
    case = box(tmp_path)
    path = case / "postProcessing" / "wallHeatFlux" / "0" / "wallHeatFlux.dat"
    write_dat(path, [], header="# Time\tpatch\tintegral\tmin\tmax\n")
    path.write_text(path.read_text() + "0.5\twall\t800\t500\t1000\n")

    (region,) = heat.measure(case)["regions"]
    (row,) = heat.primary(region)["patches"]
    assert (row["heat_rate"], row["min"], row["max"]) == pytest.approx((800, 500, 1000))


# -- Nusselt, with every term read or given ------------------------------------


def nusselt_case(root: Path, t_wall: float = 350.0) -> Path:
    """The graded wall at 800 W/m2 mean, held at `t_wall`, with k = 0.8 W/mK."""
    case = flux_case(root)
    write_field(case / "0.5" / "T", "volScalarField",
                {"wall": f"uniform {t_wall:g}", "outer": "uniform 300"},
                internal=scalars([300.0] * 32))
    properties(case, "rho rho [0 0 0 0 0 0 0] 1000;\nCp Cp [0 0 0 0 0 0 0] 4000;\n"
                     "nu nu [0 2 -1 0 0 0 0] 2e-07;\nPr Pr [0 0 0 0 0 0 0] 1;\n")
    return case


def test_a_nusselt_number_matching_the_one_computed_by_hand(heat, tmp_path):
    """q = 800 W/m2 (the area-weighted wall mean), T_wall = 350, T_ref = 300, so
    h = 800 / 50 = 16 W/m2K. k comes out of the dictionary as
    rho * Cp * nu / Pr = 1000 * 4000 * 2e-07 / 1 = 0.8 W/mK, and with L = 0.1 m
    Nu = 16 * 0.1 / 0.8 = 2. Nothing in that was computed by the script twice: the
    flux is in the field, the temperatures in T and the four constants in the
    dictionary."""
    found = heat.measure(nusselt_case(tmp_path), length=0.1, t_ref=300.0)

    (region,) = found["regions"]
    assert region["conductivity"] == pytest.approx(0.8)
    assert "rho 1000" in region["conductivity_from"]
    rows = {row["patch"]: row for row in region["nusselt"]}
    assert rows["wall"]["h"] == pytest.approx(16.0)
    assert rows["wall"]["Nu"] == pytest.approx(2.0)
    assert rows["wall"]["t_wall"] == pytest.approx(350.0)
    assert rows["wall"]["convention"] == "freestream"


def test_the_reference_convention_is_stated_and_changes_the_answer(heat, tmp_path):
    """Nu is meaningless without saying which reference temperature it is on, so the
    convention is in every record -- and the same case at T_ref 300 and T_ref 250
    gives 2.0 and 0.8, which is the reason."""
    case = nusselt_case(tmp_path)

    (warm,) = heat.measure(case, length=0.1, t_ref=300.0)["regions"]
    (cool,) = heat.measure(case, length=0.1, t_ref=250.0)["regions"]

    assert warm["reference"]["convention"] == "freestream"
    assert {row["patch"]: row["Nu"] for row in warm["nusselt"]}["wall"] == (
        pytest.approx(2.0)
    )
    # h = 800 / (350 - 250) = 8, Nu = 8 * 0.1 / 0.8
    assert {row["patch"]: row["Nu"] for row in cool["nusselt"]}["wall"] == (
        pytest.approx(1.0)
    )

    text = heat.report(heat.measure(case, length=0.1, t_ref=300.0))
    assert "freestream" in text
    for finding in heat.findings(heat.measure(case, length=0.1, t_ref=300.0)):
        if finding.check == "nusselt":
            assert "freestream" in finding.meaning or "freestream" in finding.measured
            break
    else:
        pytest.fail("no nusselt finding")


def bulk_case(root: Path) -> Path:
    """One patch carrying a non-uniform T and a non-uniform U, so the flux-weighted
    mean and the area-weighted mean over it are different numbers.

    The patch is `write_box`'s renamed to `inlet`. Its four small faces (0.1 m2) see
    T = 300 and a normal speed of 1 m/s; its four large ones (0.15 m2) see T = 400
    and 3 m/s. Area-weighted: (4*300*0.1 + 4*400*0.15) / 1.0 = 360 K. Flux-weighted,
    with phi = U . A per face so the weights are 0.1 and 0.45:
    (300*0.4 + 400*1.8) / 2.2 = 381.8181... K.
    """
    case = box(root, rename_wall="inlet")
    write_field(case / "0.5" / "T", "volScalarField",
                {"inlet": scalars([300.0] * 4 + [400.0] * 4)},
                internal=scalars([350.0] * 32))
    # The inlet sits on z = 0, so its outward area vectors point along -z; a
    # velocity along -z is flow leaving through it and gives a positive phi.
    write_field(case / "0.5" / "U", "volVectorField",
                {"inlet": vectors([(0, 0, -1.0)] * 4 + [(0, 0, -3.0)] * 4)})
    return case


def test_the_bulk_mean_is_flux_weighted_and_differs_from_the_area_mean(heat, tmp_path):
    """The mixed-mean temperature an internal flow's Nu is defined on, formed the way
    `probe.py`'s `--weighted flux` forms a section mean. On a uniform profile the two
    agree and the distinction is invisible; on this one they are 360 and 381.8, and
    taking the area average for the bulk is a Nusselt number out by that ratio."""
    case = bulk_case(tmp_path)
    regions, _ = heat.regions_of(case)
    (region,) = regions
    mesh = heat.mesh_of(region)
    temperatures = heat.wall.boundary_values(
        (case / "0.5" / "T").read_text()
    )

    flux = heat.reference_temperature(
        region, mesh, temperatures, stated=None, patch="inlet",
        convention="bulk", time=None,
    )
    area = heat.reference_temperature(
        region, mesh, temperatures, stated=None, patch="inlet",
        convention="inlet", time=None,
    )

    assert area.convention == "inlet"
    assert area.value == pytest.approx(360.0)
    assert flux.convention == "bulk"
    assert flux.value == pytest.approx(840.0 / 2.2)
    assert flux.value != pytest.approx(area.value)
    assert "flux-weighted" in flux.detail and "mixed mean" in flux.detail
    assert "area-weighted" in area.detail


def test_the_bulk_mean_is_the_default_over_a_named_patch(heat, tmp_path):
    """An internal flow's Nu is on the mixed mean, so that is what a named patch
    gives without being asked; the area average is there under its own name."""
    case = bulk_case(tmp_path)
    regions, _ = heat.regions_of(case)
    reference = heat.reference_temperature(
        heat.regions_of(case)[0][0], heat.mesh_of(regions[0]),
        heat.wall.boundary_values((case / "0.5" / "T").read_text()),
        stated=None, patch="inlet", convention=None, time=None,
    )
    assert reference.convention == "bulk"
    assert reference.value == pytest.approx(840.0 / 2.2)


def test_a_conductivity_that_cannot_be_read_is_refused_by_name(heat, tmp_path):
    """`benchmarks/moving_mesh/grade.py:125`'s rule, which `wall.py` keeps too. A
    Nusselt number off a guessed conductivity looks measured and is not: water and
    air are a factor of 25 apart and both are plausible defaults."""
    case = flux_case(tmp_path)
    write_field(case / "0.5" / "T", "volScalarField",
                {"wall": "uniform 350", "outer": "uniform 300"})

    found = heat.measure(case, length=0.1, t_ref=300.0)
    (region,) = found["regions"]
    assert region["conductivity"] is None
    assert region["nusselt"] == []
    refusals = [p for p in region["problems"] if "--conductivity" in p]
    assert refusals, "a missing conductivity is refused by name, not defaulted"
    assert all("thermophysicalProperties" in problem for problem in refusals)

    # And with the flag it is the same measurement the dictionary would have given.
    (region,) = heat.measure(case, length=0.1, t_ref=300.0, k=0.8)["regions"]
    rows = {row["patch"]: row for row in region["nusselt"]}
    assert rows["wall"]["Nu"] == pytest.approx(2.0)
    assert region["conductivity_from"] == "--conductivity"


def test_a_missing_length_scale_is_refused_by_name(heat, tmp_path):
    """The other term nothing can read off a mesh without guessing which dimension
    the Nusselt number was meant to be on."""
    (region,) = heat.measure(nusselt_case(tmp_path), t_ref=300.0)["regions"]
    assert region["nusselt"] == []
    assert any("--length" in problem for problem in region["problems"])


def test_a_reference_temperature_nobody_supplied_is_refused_by_name(heat, tmp_path):
    (region,) = heat.measure(nusselt_case(tmp_path), length=0.1)["regions"]
    (problem,) = [p for p in region["problems"] if "--t-ref" in p]
    assert "--ref-patch" in problem


# -- validated against something outside this repository -----------------------


def test_the_measured_nusselt_lands_on_dittus_boelter(heat, tmp_path):
    """Dittus-Boelter, as `ladder.py`'s conjugate catalogue states it: for turbulent
    fully developed pipe flow `Nu = 0.023 Re^0.8 Pr^n`, with n = 0.4 when the fluid
    is being heated and 0.3 when it is being cooled -- a fit to measured
    heat-transfer data over Re > 1e4 and 0.6 < Pr < 160 whose own stated scatter is
    20-25%, which is the band quoted here and not a tolerance invented for the test.
    (F. W. Dittus and L. M. K. Boelter, University of California Publications in
    Engineering 2 (1930) 443; reprinted Int. Comm. Heat Mass Transfer 12 (1985) 3.)

    At Re = 50000 and Pr = 0.71 the correlation gives
    0.023 * 50000^0.8 * 0.71^0.4 = 115.188. The fixture is built to come out near it
    rather than on it: a 25 mm pipe in air, k = 0.026 W/mK, a 60 K film temperature
    difference and a wall flux of 7000 W/m2 give h = 116.667 W/m2K and
    Nu = 116.667 * 0.025 / 0.026 = 112.179, which is 2.6% below the published number
    and therefore inside the correlation's own accuracy.
    """
    reynolds, prandtl = 50000.0, 0.71
    published = 0.023 * reynolds ** 0.8 * prandtl ** 0.4
    assert published == pytest.approx(115.188, rel=1e-4)

    case = box(tmp_path)
    write_field(case / "0.5" / "wallHeatFlux", "volScalarField",
                {"wall": "uniform 7000"})
    write_field(case / "0.5" / "T", "volScalarField",
                {"wall": "uniform 360"}, internal=scalars([300.0] * 32))

    found = heat.measure(case, length=0.025, k=0.026, t_ref=300.0,
                         reynolds=reynolds, prandtl=prandtl)
    (region,) = found["regions"]
    (row,) = region["nusselt"]

    assert row["h"] == pytest.approx(7000.0 / 60.0)
    assert row["Nu"] == pytest.approx(112.1795, rel=1e-4)

    comparison = row["correlation"]
    assert comparison["Nu"] == pytest.approx(published)
    assert "Dittus-Boelter" in comparison["name"]
    assert "n = 0.4" in comparison["name"], "the wall is hotter, so the fluid is heated"
    assert comparison["band"] == pytest.approx(0.25)
    assert comparison["relative"] == pytest.approx(0.0261, abs=1e-3)
    assert comparison["inside"], "2.6% is inside the correlation's own 20-25%"

    (finding,) = [f for f in heat.findings(found) if f.check == "correlation"]
    assert finding.status == "ok"
    assert "Dittus-Boelter" in finding.meaning and "0.6 < Pr < 160" in finding.meaning


def test_a_laminar_case_is_compared_against_graetz_instead(heat, tmp_path):
    """Below Re 2300 the catalogue's answer is the Graetz problem's first eigenvalue,
    Nu = 3.66 for a constant wall temperature (4.36 for a constant flux), which is
    exact -- so the band beside it is 2% and not the correlation's 25%."""
    answer = heat.correlation_for(1200.0, 7.0, heating=True)
    assert answer.nu == pytest.approx(3.66)
    assert answer.band == pytest.approx(0.02)
    assert "Graetz" in answer.name and "4.36" in answer.source

    cooling = heat.correlation_for(50000.0, 0.71, heating=False)
    assert "n = 0.3" in cooling.name
    assert cooling.nu == pytest.approx(0.023 * 50000.0 ** 0.8 * 0.71 ** 0.3, rel=1e-9)


def test_a_correlation_asked_for_with_half_its_inputs_is_refused(heat, tmp_path):
    (region,) = heat.measure(nusselt_case(tmp_path), length=0.1, t_ref=300.0,
                             reynolds=50000.0)["regions"]
    assert "correlation" not in region["nusselt"][0]
    assert any("--prandtl" in problem for problem in region["problems"])


# -- hot spots are places ------------------------------------------------------


def hot_case(root: Path, peak_cell: int = 5, peak: float = 500.0) -> Path:
    """The box with one cell much hotter than the other 31."""
    case = box(root)
    values = [300.0] * 32
    values[peak_cell] = peak
    write_field(case / "0.5" / "T", "volScalarField", {"wall": "uniform 300"},
                internal=scalars(values))
    return case


def test_a_hot_spot_comes_back_with_coordinates_and_its_patch(heat, tmp_path):
    """A maximum is a number; this has to be a place. Cell 5 of the box is in the
    second cell row in z -- so it is not against the `wall` patch at z = 0 -- and
    the answer names the nearest patch and how far it is rather than stopping at
    the value."""
    case = hot_case(tmp_path, peak_cell=5)
    regions, _ = heat.regions_of(case)
    mesh = heat.mesh_of(regions[0])
    found = heat.hot_spots(regions[0], mesh, None)

    assert found["max"] == pytest.approx(500.0)
    peak = found["peak"]
    assert peak["cell"] == 5
    assert peak["value"] == pytest.approx(500.0)
    assert len(peak["centre"]) == 3
    centre = heat.layer_report.cell_centres(mesh, np.array([5]))[0]
    assert peak["centre"] == pytest.approx([round(float(p), 6) for p in centre])
    assert peak["nearest_patch"] in mesh["boundary"]
    assert peak["patch_distance"] >= 0.0
    assert "distance_to_bbox" in peak and "normalised" in peak


def test_the_hot_region_is_measured_and_not_only_its_peak(heat, tmp_path):
    """Four cells above a threshold and a tenth of the mesh above it are different
    objects, so the count, the fraction and the box the region spans come with the
    peak -- and the threshold says where it came from."""
    case = box(tmp_path)
    values = [300.0] * 32
    for cell in (0, 1, 2, 3):
        values[cell] = 500.0
    write_field(case / "0.5" / "T", "volScalarField", {"wall": "uniform 300"},
                internal=scalars(values))
    regions, _ = heat.regions_of(case)
    found = heat.hot_spots(regions[0], heat.mesh_of(regions[0]), None)

    assert found["cells"] == 32
    assert found["hot_cells"] == 4
    assert found["hot_fraction"] == pytest.approx(12.5)
    assert found["threshold"] == pytest.approx(300.0 + 0.95 * 200.0)
    assert "span" in found["threshold_from"]
    extent = found["hot_extent"]
    assert extent["diagonal"] > 0.0
    assert len(extent["min"]) == 3 and len(extent["max"]) == 3

    stated = heat.hot_spots(regions[0], heat.mesh_of(regions[0]), None, threshold=299.0)
    assert stated["hot_cells"] == 32
    assert stated["threshold_from"] == "--threshold"


def test_a_field_that_is_not_the_meshs_size_says_so_beside_the_location(heat, tmp_path):
    """A T of 16 values on a 32-cell mesh is half a decomposed run or another
    region's field, and a hot spot placed in the wrong mesh is worse than none. The
    coordinates are still reported -- cell 5 is cell 5 -- with the mismatch beside
    them rather than instead of them."""
    case = box(tmp_path)
    write_field(case / "0.5" / "T", "volScalarField", {"wall": "uniform 300"},
                internal=scalars([300.0] * 15 + [500.0]))
    regions, _ = heat.regions_of(case)
    found = heat.hot_spots(regions[0], heat.mesh_of(regions[0]), None)

    assert found["cells"] == 16
    assert found["cells_in_mesh"] == 32
    assert "16 values" in found["mismatch"] and "32 cells" in found["mismatch"]
    assert found["peak"]["cell"] == 15
    assert "centre" in found["peak"]


def test_a_uniform_temperature_has_no_hot_spot_to_place(heat, tmp_path):
    case = box(tmp_path)
    write_field(case / "0.5" / "T", "volScalarField", {"wall": "uniform 300"},
                internal="uniform 300")
    regions, _ = heat.regions_of(case)
    found = heat.hot_spots(regions[0], heat.mesh_of(regions[0]), None)
    assert found["uniform"] == pytest.approx(300.0)
    assert "peak" not in found


# -- regions -------------------------------------------------------------------


def region_properties(case: Path, groups: dict[str, list[str]]) -> None:
    (case / "constant").mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"    {name}  ({' '.join(members)})"
                     for name, members in groups.items())
    (case / "constant" / "regionProperties").write_text(
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
        "    class dictionary;\n    object regionProperties;\n}\n\n"
        f"regions\n(\n{body}\n);\n"
    )


def conjugate_case(root: Path) -> Path:
    """Two regions as `splitMeshRegions` leaves them: a polyMesh each under
    `constant/<region>/` and a field directory each under `<time>/<region>/`."""
    case = root / "cht"
    case.mkdir(parents=True, exist_ok=True)
    for name, flux in (("fluid", "uniform 1000"), ("solid", "uniform -1000")):
        built = box(root, name=f"build_{name}")
        shutil.move(str(built / "constant" / "polyMesh"),
                    str(case / "constant" / name / "polyMesh"))
        write_field(case / "0.5" / name / "wallHeatFlux", "volScalarField",
                    {"wall": flux})
    region_properties(case, {"fluid": ["fluid"], "solid": ["solid"]})
    return case


def test_a_conjugate_case_is_reported_per_region(heat, tmp_path):
    """The interface flux appears once per side, and in steady state the two are the
    same number with opposite signs -- which is the conservation check `ladder.py`'s
    coupled rung asks for, and it cannot be made from one number."""
    found = heat.measure(conjugate_case(tmp_path))

    assert found["declared_regions"] == ["fluid", "solid"]
    assert [region["region"] for region in found["regions"]] == ["fluid", "solid"]
    rates = {
        region["region"]: heat.total_heat_rate(heat.primary(region))
        for region in found["regions"]
    }
    assert rates["fluid"] == pytest.approx(1000.0 * WALL_AREA)
    assert rates["solid"] == pytest.approx(-1000.0 * WALL_AREA)

    (finding,) = [f for f in heat.findings(found) if f.check == "regions"]
    assert "2 mesh regions" in finding.measured
    assert "fluid" in finding.measured and "solid" in finding.measured

    text = heat.report(found)
    assert "region fluid" in text and "region solid" in text


def test_one_region_is_a_region_that_says_so(heat, tmp_path):
    """"One region" and "the regions could not be read" are different events, and a
    case that is simply not conjugate should not be silent about it."""
    found = heat.measure(flux_case(tmp_path))

    assert found["declared_regions"] == []
    assert [region["region"] for region in found["regions"]] == [""]
    (finding,) = [f for f in heat.findings(found) if f.check == "regions"]
    assert "one region" in finding.measured
    assert "one region" in heat.report(found)


def test_only_the_region_asked_for_is_reported(heat, tmp_path):
    found = heat.measure(conjugate_case(tmp_path), region="solid")
    assert [region["region"] for region in found["regions"]] == ["solid"]

    missing = heat.measure(conjugate_case(tmp_path / "other"), region="coolant")
    assert missing["regions"] == []
    assert any("coolant" in problem for problem in missing["problems"])


def test_the_inner_names_of_region_properties_are_the_mesh_regions(heat):
    """`regions (fluid (water) solid (metal));` has two mesh regions called `water`
    and `metal`; the outer words are the physics each group is solved with. Reading
    the outer word is how a case comes back with a region that has no polyMesh."""
    assert heat.region_names("regions\n(\n  fluid (water)\n  solid (metal)\n);\n") == [
        "water", "metal"
    ]
    assert heat.region_names("regions (fluid solid);") == ["fluid", "solid"]
    assert heat.region_names("") == []


# -- what was watching the temperature ----------------------------------------


def test_a_scalar_transport_temperature_outside_the_control_is_the_finding(heat, tmp_path):
    """The measured failure this script carries forward: a `scalarTransport` field is
    not one the SIMPLE loop solves, so naming it in `residualControl` does nothing --
    the control never sees a function object's field. One heated run stopped at
    iteration 87 on p and U with T still moving."""
    case = flux_case(tmp_path)
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "system" / "fvSolution").write_text(
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
        "    class dictionary;\n    object fvSolution;\n}\n\n"
        "SIMPLE\n{\n    residualControl\n    {\n"
        "        p               1e-4;\n        U               1e-4;\n    }\n}\n"
    )
    (case / "system" / "controlDict").write_text(
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
        "    class dictionary;\n    object controlDict;\n}\n\n"
        "application     simpleFoam;\n\n"
        "functions\n{\n    heat\n    {\n        type   scalarTransport;\n"
        "        field  T;\n    }\n}\n"
    )

    (region,) = heat.measure(case)["regions"]
    control = region["control"]
    assert control["loop"] == "SIMPLE"
    assert set(control["residual_control"]) == {"p", "U"}
    assert control["transported"] == ["T"]

    (finding,) = [f for f in heat.findings(heat.measure(case))
                  if f.check == "temperature-control"]
    assert finding.status == "warn"
    assert "iteration 87" in finding.meaning
    assert "never sees a function object's field" in finding.meaning


def test_two_written_temperatures_are_differenced(heat, tmp_path):
    """The measurement that settles it whatever the control was named: T at the last
    two writes, and how far apart they are on the field's own span."""
    case = box(tmp_path)
    write_field(case / "0.4" / "T", "volScalarField", {"wall": "uniform 300"},
                internal=scalars([300.0] * 32))
    write_field(case / "0.5" / "T", "volScalarField", {"wall": "uniform 300"},
                internal=scalars([300.0] * 31 + [350.0]))

    regions, _ = heat.regions_of(case)
    control = heat.temperature_control(regions[0], None)
    assert control["change"] == pytest.approx(50.0)
    assert control["change_between"] == ["0.4", "0.5"]
    assert control["change_span"] == pytest.approx(50.0)


# -- the contract with whoever runs it -----------------------------------------


def test_a_case_with_no_heat_anywhere_says_so_and_still_exits_zero(heat, tmp_path, capsys):
    assert heat.main([str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert "no wall heat flux anywhere" in text
    assert "wallHeatFlux" in text
    assert "[skipped]" in text


def test_json_carries_the_whole_measurement_and_parses(heat, tmp_path, capsys):
    assert heat.main([str(nusselt_case(tmp_path)), "--length", "0.1",
                      "--t-ref", "300", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert set(payload) >= {"case", "regions", "declared_regions", "problems",
                            "findings", "worst"}
    (region,) = payload["regions"]
    assert region["nusselt"][0]["Nu"] == pytest.approx(2.0)
    assert region["reference"]["convention"] == "freestream"
    # preflight's register, so a finding reads the same here as it does there.
    assert set(payload["findings"][0]) == {"check", "status", "measured", "means",
                                           "repair"}


def test_the_png_is_the_two_figures_and_the_path_is_printed(heat, tmp_path, capsys):
    out = tmp_path / "pictures" / "heat.png"
    assert heat.main([str(nusselt_case(tmp_path)), "--length", "0.1",
                      "--t-ref", "300", "--png", str(out)]) == 0

    assert out.is_file() and out.stat().st_size > 1000
    assert out.resolve().as_posix() in capsys.readouterr().out


def test_an_unreadable_case_is_reported_rather_than_raised(heat, tmp_path, capsys):
    case = flux_case(tmp_path)
    (case / "constant" / "polyMesh" / "owner").write_text("garbage")
    assert heat.main([str(case)]) == 0
    assert "[skipped]" in capsys.readouterr().out


def test_the_report_offers_the_numbers_without_a_verdict_on_them(heat, tmp_path):
    text = heat.report(heat.measure(nusselt_case(tmp_path), length=0.1, t_ref=300.0))
    assert "nothing here decides whether this heat transfer is good enough" in text
    for verdict in ("unacceptable", "you must", "always run", "too hot", "fix "):
        assert verdict not in text.lower()


def test_the_toolbox_index_carries_the_script(heat):
    """`tests/test_toolbox.py` fails on a script missing from the index; this says
    which row is the missing one when it does."""
    index = (TOOLBOX / "README.md").read_text(encoding="utf-8")
    assert "| `heat.py` |" in index


def test_it_needs_nothing_the_image_does_not_have(heat):
    source = (TOOLBOX / "heat.py").read_text(encoding="utf-8")
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    siblings = {path.stem for path in TOOLBOX.glob("*.py")}
    allowed = set(sys.stdlib_module_names) | siblings | {
        "__future__", "numpy", "matplotlib"
    }
    assert roots <= allowed, sorted(roots - allowed)
    assert "scipy" not in source
    source.encode("ascii")  # raises if a non-ASCII character crept in


def test_the_docstring_keeps_the_script_advisory_and_says_how_to_run_it(heat):
    text = " ".join((heat.__doc__ or "").split())
    assert "python3 heat.py" in text
    assert "edits nothing, runs no solver, and exits 0" in text
    # The reference temperature is the part a Nusselt number is meaningless without,
    # so the three conventions are named where the choice is made.
    for convention in ("freestream", "inlet", "bulk"):
        assert convention in text
    for imperative in ("you must", "always ", "make sure", "you should"):
        assert imperative not in text.lower()


def test_the_mesh_and_field_reading_are_the_toolboxs_and_not_a_second_copy(heat):
    """One polyMesh reader, one face-area decomposition and one extremum locator in
    the toolbox, not several that nearly agree."""
    source = (TOOLBOX / "heat.py").read_text(encoding="utf-8")
    for reused in ("locate.read_mesh", "locate.read_internal_field",
                   "locate.cell_patches", "locate.bbox_facts",
                   "layer_report.face_geometry", "layer_report.cell_centres",
                   "results.split_header_names", "wall.boundary_values"):
        assert reused in source, f"{reused} is in the toolbox and not reused here"
    for rewritten in ("def read_points", "def read_faces", "def read_labels",
                      "def cell_centres", "def read_mesh"):
        assert rewritten not in source, "nothing here reads the mesh files itself"
