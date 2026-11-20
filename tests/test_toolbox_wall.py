"""`wall.py`: the y+ a run achieved, as opposed to the y+ it was predicted to get.

Until this script the toolbox had an a priori estimate (`case_gen.py`'s flat-plate
correlation, printed before the solver starts) and a geometric measurement
(`layer_report.py`'s first-cell height, which carries no velocity). Nothing read
back what the solver itself computed, which is the only number that settles whether
the wall treatment was asked for something it is not written for -- the generated
aerofoil O-grid whose first cell centre sat at y+ about 1200 under a form valid to
300 is the case that paid for this.

Every fixture here is hand-written, so the right answer is known before the script
is asked: the `.dat` files carry numbers chosen to be unmistakable, and the
wallShearStress case runs on `test_toolbox_layer_report.py`'s ascii polyMesh, whose
first-cell height is written rather than requested -- 4 mm, so y is 2 mm, and with
tau = 0.25 m2/s2 and nu = 1e-05 m2/s the answer is y+ = 100 by hand.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
TOOLBOX = HERE.parents[0] / "openreynolds" / "toolbox"

sys.path.insert(0, str(HERE))
from test_toolbox_layer_report import LAYERED, XS, YS, write_box  # noqa: E402


@pytest.fixture(scope="module")
def wall():
    """Imported by path, like every other toolbox script: the directory is data."""
    spec = importlib.util.spec_from_file_location("toolbox_wall", TOOLBOX / "wall.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# -- what the yPlus function object writes -------------------------------------


HEADER = "# y+ ()\n# Time             \tpatch\tmin\tmax\taverage\n"


def write_dat(path: Path, rows, header: str = HEADER, mtime: float | None = None) -> Path:
    """A `yPlus.dat` as the function object writes one: a patch NAME in column two,
    which is why `results.parse_dat` cannot read this file and its header reader can."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        f"{time:g}\t{patch}\t{low:g}\t{high:g}\t{mean:g}\n"
        for time, patch, low, high, mean in rows
    )
    path.write_text(header + body)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_a_patch_comes_back_with_the_three_numbers_the_run_reported(wall, tmp_path):
    """The cheapest route to the answer: the solver already computed it."""
    write_dat(
        tmp_path / "postProcessing" / "yPlus" / "0" / "yPlus.dat",
        [(0.1, "hull", 1.5, 120.3, 45.2), (0.1, "deck", 0.2, 0.9, 0.5)],
    )
    found = wall.measure(tmp_path)

    (source,) = found["sources"]
    assert source["source"] == "yPlus function object"
    assert source["time"] == pytest.approx(0.1)
    rows = {row["patch"]: row for row in source["patches"]}
    assert rows["hull"]["min"] == pytest.approx(1.5)
    assert rows["hull"]["max"] == pytest.approx(120.3)
    assert rows["hull"]["mean"] == pytest.approx(45.2)
    # Three summary numbers are not a distribution, so no fraction is invented from
    # them -- what is reported is which regions the range reaches.
    assert rows["hull"]["fractions"] is None
    assert rows["hull"]["spans"] == ["viscous sublayer", "buffer layer", "log layer"]
    assert rows["deck"]["spans"] == ["viscous sublayer"]


def test_the_columns_are_read_off_the_header_and_not_assumed(wall, tmp_path):
    """A header that writes average before min and max is still read correctly;
    assuming positions is how a mean ends up reported as a maximum."""
    write_dat(
        tmp_path / "postProcessing" / "yPlus" / "0" / "yPlus.dat",
        [],
        header="# Time\tpatch\taverage\tmin\tmax\n",
    )
    path = tmp_path / "postProcessing" / "yPlus" / "0" / "yPlus.dat"
    path.write_text(path.read_text() + "0.1\thull\t45.2\t1.5\t120.3\n")

    (source,) = wall.measure(tmp_path)["sources"]
    (row,) = source["patches"]
    assert (row["min"], row["mean"], row["max"]) == pytest.approx((1.5, 45.2, 120.3))


def test_several_time_directories_report_the_latest(wall, tmp_path):
    """A restart writes a second time directory beside the first and both hold a
    file of the same name. Reading only one of them is how half a run goes missing."""
    base = tmp_path / "postProcessing" / "yPlus"
    write_dat(base / "0" / "yPlus.dat",
              [(0.1, "hull", 1, 10, 5), (0.2, "hull", 2, 20, 10)], mtime=1_700_000_000)
    write_dat(base / "0.2" / "yPlus.dat",
              [(0.3, "hull", 3, 30, 15), (0.4, "hull", 4, 40, 20)], mtime=1_700_000_100)

    (source,) = wall.measure(tmp_path)["sources"]
    assert source["time"] == pytest.approx(0.4)
    assert source["times"] == pytest.approx([0.1, 0.3, 0.4])
    (row,) = source["patches"]
    assert (row["min"], row["max"], row["mean"]) == pytest.approx((4, 40, 20))


def test_a_restart_suffix_is_stitched_and_the_abandoned_rows_go(wall, tmp_path):
    """A function object restarted into a directory that already has its file writes
    `yPlus_0.2.dat` beside `yPlus.dat`. They are one history, and the rows the
    restart superseded are not part of it -- 999 is the run that was thrown away."""
    folder = tmp_path / "postProcessing" / "yPlus" / "0"
    write_dat(folder / "yPlus.dat",
              [(0.1, "hull", 1, 10, 5), (0.2, "hull", 9, 999, 99),
               (0.3, "hull", 9, 999, 99)], mtime=1_700_000_000)
    write_dat(folder / "yPlus_0.2.dat",
              [(0.2, "hull", 2, 20, 10), (0.3, "hull", 3, 30, 15),
               (0.4, "hull", 4, 40, 20)], mtime=1_700_000_100)

    (source,) = wall.measure(tmp_path)["sources"]
    assert source["times"] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    (row,) = source["patches"]
    assert row["max"] == pytest.approx(40)
    assert "999" not in json.dumps(source), "the superseded rows are not the history"


def test_a_time_that_was_asked_for_is_the_one_reported(wall, tmp_path):
    write_dat(tmp_path / "postProcessing" / "yPlus" / "0" / "yPlus.dat",
              [(0.1, "hull", 1, 10, 5), (0.2, "hull", 2, 20, 10)])

    (source,) = wall.measure(tmp_path, time="0.1")["sources"]
    assert source["time"] == pytest.approx(0.1)
    assert source["patches"][0]["max"] == pytest.approx(10)


# -- a written field, which carries the distribution ---------------------------


def write_field(path: Path, cls: str, entries: dict[str, str],
                dimensions: str = "[0 0 0 0 0 0 0]") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = "\n".join(
        f"    {name}\n    {{\n        type            calculated;\n"
        f"        value           {value};\n    }}"
        for name, value in entries.items()
    )
    path.write_text(
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
        f"    class       {cls};\n    object      {path.name};\n}}\n\n"
        f"dimensions      {dimensions};\n\n"
        "internalField   uniform 0;\n\n"
        f"boundaryField\n{{\n{blocks}\n}}\n"
    )
    return path


def write_binary_field(path: Path, entries: dict[str, Any], cls: str = "volScalarField",
                       arch: str = "LSB;label=32;scalar=64", items: int = 1,
                       declared: int | None = None, kind: str = "scalar") -> Path:
    """A `format binary;` field with a GENUINE little-endian IEEE-754 payload.

    Hand-written like every other fixture here, and written as bytes because that is
    the point of it: issue #138 was a number regex run over a payload exactly like
    this one, which reported the digits that fell out of the doubles as measured y+.
    A `str` entry is written as ascii (`uniform 0`, or a type with no value), a list
    of floats as the real binary list OpenFOAM writes. `declared` lies about the
    item count, which is how the "the payload does not end where the header says"
    guard is exercised.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [(
        "FoamFile\n{\n    version     2.0;\n    format      binary;\n"
        f'    arch        "{arch}";\n'
        f"    class       {cls};\n    object      {path.name};\n}}\n\n"
        "dimensions      [0 0 0 0 0 0 0];\n\ninternalField   uniform 0;\n\n"
        "boundaryField\n{\n"
    ).encode("ascii")]
    for name, value in entries.items():
        parts.append(
            f"    {name}\n    {{\n        type            calculated;\n".encode("ascii")
        )
        if isinstance(value, str):
            parts.append(f"        value           {value};\n".encode("ascii"))
        else:
            count = len(value) // items if declared is None else declared
            parts.append(
                f"        value           nonuniform List<{kind}> \n"
                f"{count}\n(".encode("ascii")
            )
            parts.append(struct.pack(f"<{len(value)}d", *[float(v) for v in value]))
            parts.append(b");\n")
        parts.append(b"    }\n")
    parts.append(b"}\n\n// " + b"*" * 40 + b" //\n")
    path.write_bytes(b"".join(parts))
    return path


def _scalars(values) -> str:
    return (
        f"nonuniform List<scalar> {len(values)}("
        + " ".join(f"{v:g}" for v in values) + ")"
    )


def test_the_regime_fractions_split_a_patch_that_straddles_two_regions(wall, tmp_path):
    """A wall is almost never in one region. Half this patch is in the viscous
    sublayer and half in the log layer, and an average of 50.5 would describe
    neither -- which is the whole reason the fractions are reported instead."""
    write_field(tmp_path / "0.5" / "yPlus", "volScalarField",
                {"hull": _scalars([1.0, 1.0, 2.0, 3.0, 100.0, 120.0, 200.0, 400.0])})

    (source,) = wall.measure(tmp_path)["sources"]
    assert source["source"] == "written yPlus field"
    (row,) = source["patches"]
    assert row["faces"] == 8
    assert row["fractions"] == {
        "viscous sublayer": 50.0, "buffer layer": 0.0, "log layer": 50.0
    }
    assert row["spans"] == ["viscous sublayer", "buffer layer", "log layer"]
    assert row["mean"] == pytest.approx(103.375)


def test_a_patch_that_wrote_no_value_is_not_a_measured_nothing(wall, tmp_path):
    """`calculated` with no `value` written is an absence; reporting it as an empty
    range would put a patch in the table that was never measured."""
    path = tmp_path / "0.5" / "yPlus"
    write_field(path, "volScalarField", {"hull": _scalars([1.0, 2.0])})
    path.write_text(path.read_text().replace(
        "    outer", "    outer"
    ) + "")
    text = path.read_text().replace(
        "boundaryField\n{\n",
        "boundaryField\n{\n    outer\n    {\n        type            calculated;\n    }\n",
    )
    path.write_text(text)

    (source,) = wall.measure(tmp_path)["sources"]
    assert [row["patch"] for row in source["patches"]] == ["hull"]


# -- #138 and #144: what a field file may not be read as -----------------------
#
# Both defects were found by running this script over the real corpus after it
# shipped on a green suite, and both have the same shape: a guess printed as a
# measurement. The fixtures below are the smallest files that still have that shape.


HULL = [32.86, 40.0, 50.0, 60.0, 90.0, 120.0, 180.0, 223.5]
"""Eight y+ values whose min, max and mean are the hull case's own 32.86, 223.5 and
(to three figures) 83.7's sibling 99.545 -- a real distribution, entirely in the log
layer, standing in for the 1008 faces issue #138 was filed on."""


def box(root: Path) -> Path:
    """`test_toolbox_layer_report.py`'s box: a `wall` patch of 8 faces typed `wall`
    in `constant/polyMesh/boundary`, and an `outer` patch typed `patch`."""
    return write_box(root / "box", XS, YS, LAYERED)


def test_a_binary_field_is_decoded_and_not_read_as_the_digits_in_its_bytes(wall, tmp_path):
    """Issue #138, which is why `format binary;` is now read out of the header.

    The fixture is a genuine binary field: a `FoamFile` saying `format binary;` with
    an `arch` note and a little-endian double payload packed by `struct`. The ascii
    reading of those same bytes is asserted to disagree with them first, because a
    fixture the old reader happens to get right would prove nothing -- on the real
    hull case the old reading was `4 faces, 0 / 1.5 / 3` against a payload of 1008
    faces running 32.86 to 223.5.
    """
    case = box(tmp_path)
    path = write_binary_field(
        case / "1500" / "yPlus", {"wall": HULL, "outer": "uniform 0"}
    )

    through_the_regex = wall.boundary_values(
        path.read_text(encoding="utf-8", errors="replace")
    ).get("wall")
    assert through_the_regex is None or sorted(through_the_regex.tolist()) != HULL, (
        "the fixture has to be a file the ascii reader gets wrong, or it proves nothing"
    )

    (source,) = wall.measure(case)["sources"]
    assert source["source"] == "written yPlus field"
    assert "binary" in source["detail"]
    (row,) = source["patches"]
    assert row["patch"] == "wall"
    assert row["faces"] == 8
    assert row["uniform"] is False
    assert row["min"] == pytest.approx(min(HULL))
    assert row["max"] == pytest.approx(max(HULL))
    assert row["mean"] == pytest.approx(sum(HULL) / len(HULL))
    assert row["fractions"] == {
        "viscous sublayer": 0.0, "buffer layer": 0.0, "log layer": 100.0
    }


def test_the_binary_field_and_the_function_object_now_agree(wall, tmp_path):
    """The regression test for issue #138, with the numbers it was filed on.

    On `20260830-011037-e0c1/case` the report printed, two blocks apart and both at
    status `ok`: from the `yPlus` function object, `hull 32.86 / 83.73 / 223.5, log
    layer`; and from the binary `1500/yPlus`, `hull 4 faces, 0 / 1.5 / 3, viscous
    sublayer 100%`. The second was the digits that fell out of the IEEE-754 bytes --
    low by a factor of about 75 on the mean, reporting a wall-function wall in the
    log layer as fully resolved, with a fabricated per-face distribution behind it.
    Here the same two sources are written over the same eight values and have to come
    back as the same measurement.
    """
    case = box(tmp_path)
    write_dat(case / "postProcessing" / "yPlus" / "0" / "yPlus.dat",
              [(1500, "wall", min(HULL), max(HULL), sum(HULL) / len(HULL))])
    write_binary_field(case / "1500" / "yPlus", {"wall": HULL})

    dat, field = wall.measure(case)["sources"]
    assert (dat["source"], field["source"]) == (
        "yPlus function object", "written yPlus field"
    )
    (summary,) = dat["patches"]
    (measured,) = field["patches"]
    for key in ("min", "max", "mean"):
        assert measured[key] == pytest.approx(summary[key], rel=1e-9), key
    assert measured["faces"] == 8, "4 faces was the shape of the wrong answer"
    assert measured["fractions"]["log layer"] == 100.0
    assert measured["fractions"]["viscous sublayer"] == 0.0


def test_a_binary_payload_that_does_not_fit_its_header_is_refused_by_name(wall, tmp_path):
    """The guard that would have caught #138 on its first run: the byte just past a
    payload has to be the parenthesis that closes it. Where it is not, the layout is
    already known to be wrong, so the source is declined by name and says what would
    read it -- never read at an offset that is a guess."""
    case = box(tmp_path)
    write_binary_field(case / "1500" / "yPlus", {"wall": HULL}, declared=9)

    found = wall.measure(case)
    assert found["sources"] == [], "a binary file never produces a measured number"
    (problem,) = [p for p in found["problems"] if "binary" in p]
    assert "1500/yPlus" in problem
    assert "foamFormatConvert" in problem
    assert "refused" in problem


def test_a_big_endian_binary_field_is_refused_rather_than_byte_swapped(wall, tmp_path):
    """The `arch` note states the byte order, so it is read rather than assumed: a
    guessed endianness is another silently wrong number of the #138 kind."""
    case = box(tmp_path)
    write_binary_field(case / "1500" / "yPlus", {"wall": HULL},
                       arch="MSB;label=32;scalar=64")

    found = wall.measure(case)
    assert found["sources"] == []
    (problem,) = [p for p in found["problems"] if "binary" in p]
    assert "big-endian" in problem and "1500/yPlus" in problem


def test_a_patch_that_is_not_a_wall_is_excluded_and_listed_as_excluded(wall, tmp_path):
    """Issue #144: y+ is a wall quantity, and `boundaryField` carries an entry for
    every patch. An inlet's `calculated` / `uniform 0` was arriving as a measured
    `y+ 0, viscous sublayer 100%` -- on the cylinder case four such rows were the
    whole report and the whole plot. The mesh types the patches, and what was left
    out is listed with the type that says why, because a silent drop is the other
    half of this same defect."""
    case = box(tmp_path)
    write_field(case / "0.5" / "yPlus", "volScalarField",
                {"wall": _scalars([40.0] * 8), "outer": _scalars([0.0] * 56)})

    found = wall.measure(case)
    (source,) = found["sources"]
    assert [row["patch"] for row in source["patches"]] == ["wall"]
    assert source["excluded"] == [{"patch": "outer", "type": "patch"}]
    assert source["typed_from"] == "constant/polyMesh/boundary"
    assert found["wall_treatment"] == {"wall": ""}, "only a wall is asked about"

    text = wall.report(found)
    assert "excluded" in text and "outer (type patch)" in text
    (finding,) = [f for f in wall.findings(found) if f.check == "y+"]
    assert "outer" in finding.measured and "excluded" in finding.measured
    assert "outer" not in finding.meaning


def test_a_value_count_that_is_not_the_patchs_face_count_is_not_a_measurement(wall, tmp_path):
    """The cheap guard issue #138 asked for, kept beside the decode rather than
    instead of it. `from_wall_shear` already refused a count that was neither one nor
    the patch's faces and `from_field` had no such check -- and `4 faces` on a wall of
    1008 was all it ever took to know that read had failed."""
    case = box(tmp_path)
    write_field(case / "0.5" / "yPlus", "volScalarField",
                {"wall": _scalars([40.0] * 5)})

    found = wall.measure(case)
    assert found["sources"] == []
    (problem,) = [p for p in found["problems"] if "5 values" in p]
    assert "8 faces" in problem
    assert "constant/polyMesh/boundary" in problem


def test_a_uniform_entry_is_reported_as_uniform_and_not_as_a_distribution(wall, tmp_path):
    """The second half of #144. `uniform 2.5` is ONE value for a whole patch: it was
    being counted as `faces: 1` and then printed as a 100% per-face fraction, which
    is a histogram the file does not contain. What is reported is the value, the word
    uniform, and the patch's real face count out of the mesh."""
    case = box(tmp_path)
    write_field(case / "0.5" / "yPlus", "volScalarField", {"wall": "uniform 2.5"})

    found = wall.measure(case)
    (source,) = found["sources"]
    (row,) = source["patches"]
    assert row["uniform"] is True
    assert row["fractions"] is None, "one value is not a distribution"
    assert row["faces"] == 8, "the mesh's face count, not the length of one value"
    assert row["min"] == pytest.approx(2.5) and row["max"] == pytest.approx(2.5)

    text = wall.report(found)
    assert "uniform" in text
    assert "100%" not in text, "no per-face percentage is manufactured from one value"


def test_the_field_at_time_zero_is_not_reported_as_a_measurement(wall, tmp_path):
    """Also #144: `0/yPlus` is the field as the function object wrote it BEFORE the
    run, and it was reported as a measurement "at time 0" -- hull y+ 0 -- beside a
    function object saying the same hull ran at y+ 2.5 to 1289. It is passed over,
    with the reason said out loud, and read only when `--time 0` asks."""
    case = box(tmp_path)
    write_field(case / "0" / "yPlus", "volScalarField", {"wall": _scalars([40.0] * 8)})

    found = wall.measure(case)
    assert found["sources"] == []
    (dropped,) = [p for p in found["problems"] if "0/yPlus" in p]
    assert "initial condition" in dropped and "not a result" in dropped
    assert "[skipped]" in wall.report(found)

    asked = wall.measure(case, time="0")
    (source,) = asked["sources"]
    assert source["initial"] is True
    assert "INITIAL CONDITION" in source["detail"]
    for finding in [f for f in wall.findings(asked) if f.check == "y+"]:
        assert finding.status == "skipped", "time 0 is never an ok measurement"


def test_a_time_with_no_matching_row_says_which_source_it_dropped(wall, tmp_path):
    """Found in the same sweep as #138 and #144 and not filed: `--time 1075` on the
    wing case left the function object's own answer out of the report with nothing
    said. Silently dropping a source is the pattern behind both filed defects, so
    what went and what was there instead are stated."""
    write_dat(tmp_path / "postProcessing" / "yPlus" / "0" / "yPlus.dat",
              [(1070, "wing", 2.7, 2166, 103.9), (1080, "wing", 2.8, 2170, 104.0)])

    found = wall.measure(tmp_path, time="1075")
    assert found["sources"] == []
    (dropped,) = [p for p in found["problems"] if "1075" in p]
    assert "yPlus" in dropped
    assert "1070" in dropped and "1080" in dropped
    assert "passed over" in dropped
    assert "[skipped]" in wall.report(found)


# -- y+ derived where only the shear was written -------------------------------


def shear_case(root: Path, value: str, nu: str | None = "1e-05") -> Path:
    """`test_toolbox_layer_report.py`'s box, whose `wall` patch has eight faces and
    a first-cell height of exactly 4 mm, plus a wallShearStress on that patch."""
    case = write_box(root / "shear", XS, YS, LAYERED)
    write_field(case / "0.5" / "wallShearStress", "volVectorField", {"wall": value})
    if nu is not None:
        (case / "constant").mkdir(parents=True, exist_ok=True)
        (case / "constant" / "transportProperties").write_text(
            "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
            "    class dictionary;\n    object transportProperties;\n}\n\n"
            f"transportModel  Newtonian;\nnu              {nu};\n"
        )
    return case


def test_y_plus_from_the_shear_agrees_with_the_number_computed_by_hand(wall, tmp_path):
    """u_tau = sqrt(0.25) = 0.5, y = half of the 4 mm first cell = 0.002 m,
    nu = 1e-05, so y+ = 0.5 * 0.002 / 1e-05 = 100. Nothing in that came from the
    script: the height is written into the mesh and the shear into the field."""
    case = shear_case(tmp_path, "uniform (0.25 0 0)")

    found = wall.measure(case)
    assert found["nu"] == pytest.approx(1e-05)
    assert found["nu_from"] == "constant/transportProperties"
    (source,) = found["sources"]
    assert source["source"] == "derived from wallShearStress"
    (row,) = source["patches"]
    assert row["faces"] == 8, "a uniform entry broadcasts over the patch's faces"
    assert row["min"] == pytest.approx(100.0)
    assert row["max"] == pytest.approx(100.0)
    assert row["fractions"]["log layer"] == 100.0


def test_a_wall_whose_shear_is_all_zero_is_skipped_by_name_not_reported_at_y_plus_zero(
    wall, tmp_path
):
    """`type calculated; value uniform (0 0 0)` is what OpenFOAM writes for a patch
    the function object did not evaluate, and no wall under a moving fluid has zero
    shear over the whole of it. #144's type filter cannot catch this one, because the
    patch IS a wall: on `20260829-061843-9483/bfs` the 20-face `stepWall` came out of
    `15000/wallShearStress` as `y+ 0 to 0, viscous sublayer 100%` at status ok, a
    distribution over 20 faces manufactured from one placeholder zero.
    """
    case = shear_case(tmp_path, "uniform (0 0 0)")

    found = wall.measure(case)
    assert not [row for source in found["sources"] for row in source["patches"]], (
        "nothing here was measured, so nothing should be reported as measured"
    )
    (problem,) = found["problems"]
    assert "'wall'" in problem and "zero" in problem, problem
    assert "wallShearStress did not evaluate" in problem, problem
    statuses = {finding.status for finding in wall.findings(found) if finding.check == "y+"}
    assert statuses == {"skipped"}, statuses

    # And a shear that is uniform but not zero is still a measurement, so the rule is
    # the zero and not the `uniform`: the case above and this one differ by one digit.
    measured = wall.measure(shear_case(tmp_path / "live", "uniform (0.25 0 0)"))
    assert not measured["problems"]
    ((row,),) = [source["patches"] for source in measured["sources"]]
    assert row["max"] == pytest.approx(100.0)


def test_a_per_face_shear_gives_a_per_face_y_plus(wall, tmp_path):
    """The vector magnitude is what u_tau is built from, so a shear written as
    (0 0.25 0) is the same wall as one written as (0.25 0 0)."""
    values = " ".join(["(0 0.25 0)"] * 4 + ["(0 0 2.5e-05)"] * 4)
    case = shear_case(tmp_path, f"nonuniform List<vector> 8({values})")

    (source,) = wall.measure(case)["sources"]
    (row,) = source["patches"]
    assert row["max"] == pytest.approx(100.0)
    # sqrt(2.5e-05) * 0.002 / 1e-05
    assert row["min"] == pytest.approx(np.sqrt(2.5e-05) * 0.002 / 1e-05)
    assert row["fractions"] == {
        "viscous sublayer": 50.0, "buffer layer": 0.0, "log layer": 50.0
    }


def test_a_viscosity_that_cannot_be_read_is_refused_by_name(wall, tmp_path):
    """The rule `benchmarks/moving_mesh/grade.py` already keeps: a missing constant
    stops and names the flag that would supply it. A y+ off a guessed nu is a number
    that looks measured and is not, and water-at-20C is a guess."""
    case = shear_case(tmp_path, "uniform (0.25 0 0)", nu=None)

    found = wall.measure(case)
    assert found["nu"] is None
    assert found["sources"] == []
    (problem,) = [p for p in found["problems"] if "--nu" in p]
    assert "nu" in problem
    assert "transportProperties" in problem

    # And with the flag it is the same measurement as before.
    (source,) = wall.measure(case, nu=1e-05)["sources"]
    assert source["patches"][0]["max"] == pytest.approx(100.0)
    assert "nu = 1e-05" in source["detail"]


def test_a_viscosity_of_zero_is_refused_rather_than_divided_by(wall, tmp_path):
    """`nu 0;` is a real thing people write while sweeping a parameter, and
    `preflight.parse_nu` hands it back as read rather than as absent."""
    case = shear_case(tmp_path, "uniform (0.25 0 0)", nu="0")

    found = wall.measure(case)
    assert found["sources"] == []
    assert "--nu" in found["problems"][0]


def test_only_the_wall_of_the_box_is_derived_from_the_shear(wall, tmp_path):
    """#144 on the path where there is no `.dat` to lead the report: on the cylinder
    case this source was the only one, so its `inlet`, `outlet`, `topWall` and
    `bottomWall` rows at y+ 0 were the whole report and the whole plot. The mesh
    `layer_report.py` already loaded here types the patches, so nothing extra is
    read to know that `outer` is not a wall."""
    case = shear_case(tmp_path, "uniform (0.25 0 0)")
    path = case / "0.5" / "wallShearStress"
    path.write_text(path.read_text().replace(
        "boundaryField\n{\n",
        "boundaryField\n{\n    outer\n    {\n        type            calculated;\n"
        "        value           uniform (0 0 0);\n    }\n",
    ))

    (source,) = wall.measure(case)["sources"]
    assert [row["patch"] for row in source["patches"]] == ["wall"]
    assert source["excluded"] == [{"patch": "outer", "type": "patch"}]


def test_a_wall_shear_in_pascals_is_refused_against_its_own_dimensions(wall, tmp_path):
    """The compressible path, which no case in the corpus exercises -- so it was
    untested rather than clean when #138 and #144 were filed.

    `u_tau = sqrt(|tau|)` holds for the KINEMATIC wall shear an incompressible solver
    writes, tau/rho in m2/s2. A compressible solver writes a stress in Pa, where mass
    appears in the dimensions, and the square root of that is not a velocity: it is
    out by sqrt(rho). Converting it needs rho at the wall, which a compressible case
    does not state as a constant, so the field's own `dimensions` are read and a
    stress carrying mass is refused by name rather than reported in the wrong units.
    """
    case = shear_case(tmp_path, "uniform (0.25 0 0)")
    write_field(case / "0.5" / "wallShearStress", "volVectorField",
                {"wall": "uniform (0.25 0 0)"}, dimensions="[1 -1 -2 0 0 0 0]")

    found = wall.measure(case)
    assert found["sources"] == []
    (problem,) = [p for p in found["problems"] if "dimensions" in p]
    assert "0.5/wallShearStress" in problem
    assert "Pa" in problem and "rho" in problem

    # And the kinematic spelling is read as before, saying which convention it read.
    kinematic = shear_case(tmp_path / "kinematic", "uniform (0.25 0 0)")
    write_field(kinematic / "0.5" / "wallShearStress", "volVectorField",
                {"wall": "uniform (0.25 0 0)"}, dimensions="[0 2 -2 0 0 0 0]")
    (source,) = wall.measure(kinematic)["sources"]
    assert "kinematic" in source["detail"]
    assert source["patches"][0]["max"] == pytest.approx(100.0)


# -- the wall treatment the case actually used ---------------------------------


def write_nut(case: Path, kind: str) -> None:
    write_field(case / "0" / "nut", "volScalarField", {})
    (case / "0" / "nut").write_text(
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
        "    class volScalarField;\n    object nut;\n}\n\n"
        "dimensions      [0 2 -1 0 0 0 0];\n\ninternalField   uniform 0;\n\n"
        "boundaryField\n{\n    wall\n    {\n"
        f"        type            {kind};\n        value           uniform 0;\n"
        "    }\n}\n"
    )


def test_spaldings_blend_is_reported_as_informational_whatever_came_out(wall, tmp_path):
    """Every model `case_gen.py` writes gets `nutUSpaldingWallFunction`, which is
    valid through all three regions. So the finding on those cases states the
    measurement and that the treatment covers it, and nothing is wrong."""
    case = shear_case(tmp_path, "uniform (0.25 0 0)")
    write_nut(case, "nutUSpaldingWallFunction")

    found = wall.measure(case)
    assert found["wall_treatment"] == {"wall": "nutUSpaldingWallFunction"}
    (finding,) = [f for f in wall.findings(found) if f.check == "wall-treatment"]
    assert finding.status == "ok"
    assert "y+ 100" in finding.measured and "nutUSpaldingWallFunction" in finding.measured
    assert "informational" in finding.meaning
    assert finding.repair == "", "a measurement is not an instruction"


def test_a_plain_high_re_form_outside_its_band_is_the_sharper_finding(wall, tmp_path):
    """`nutkWallFunction` assumes the log law at the first cell centre, so a wall
    measured in the sublayer was treated by a law that does not hold there. The
    finding says that and stops; what to do about it is not its business."""
    case = shear_case(tmp_path, "uniform (2.5e-05 0 0)")  # y+ about 0.1
    write_nut(case, "nutkWallFunction")

    found = wall.measure(case)
    (finding,) = [f for f in wall.findings(found) if f.check == "wall-treatment"]
    assert finding.status == "warn"
    assert "30" in finding.meaning and "300" in finding.meaning
    assert "runs outside" in finding.meaning
    assert finding.repair == ""
    for instruction in ("you must", "always ", "make sure", "you should", "refine the"):
        assert instruction not in finding.meaning.lower()


def test_a_wall_condition_with_no_known_band_is_not_judged(wall, tmp_path):
    case = shear_case(tmp_path, "uniform (0.25 0 0)")
    write_nut(case, "somebodysOwnWallFunction")

    (finding,) = [f for f in wall.findings(wall.measure(case))
                  if f.check == "wall-treatment"]
    assert finding.status == "ok"
    assert "not compared" in finding.meaning


# -- the contract with whoever runs it -----------------------------------------


def test_a_case_with_no_y_plus_at_all_says_so_and_still_exits_zero(wall, tmp_path, capsys):
    """"No y+ here" and "I could not look" have to be different events, and neither
    is a non-zero exit: the toolbox reports and stops."""
    assert wall.main([str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert "no y+ anywhere" in text
    assert "yPlus*.dat" in text and "wallShearStress" in text
    assert "[skipped]" in text

    found = wall.measure(tmp_path)
    assert found["sources"] == []
    assert wall.primary(found) is None


def test_json_carries_the_whole_measurement_and_parses(wall, tmp_path, capsys):
    write_dat(tmp_path / "postProcessing" / "yPlus" / "0" / "yPlus.dat",
              [(0.1, "hull", 1.5, 120.3, 45.2)])
    assert wall.main([str(tmp_path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert set(payload) >= {"case", "sources", "nu", "nu_from", "wall_treatment",
                            "problems", "findings", "worst"}
    (source,) = payload["sources"]
    # `uniform` joined the row in the #144 fix: one value for a whole patch is a
    # third state beside a distribution and a summary, and a reader of the JSON has
    # to be able to tell it apart rather than see `faces: 1`.
    assert set(source["patches"][0]) == {
        "patch", "min", "max", "mean", "faces", "spans", "fractions", "uniform"
    }
    # preflight's register, so a finding reads the same here as it does there.
    assert set(payload["findings"][0]) == {"check", "status", "measured", "means",
                                           "repair"}


def test_the_png_is_one_bar_per_patch_and_the_path_is_printed(wall, tmp_path, capsys):
    write_dat(tmp_path / "postProcessing" / "yPlus" / "0" / "yPlus.dat",
              [(0.1, "hull", 1.5, 120.3, 45.2), (0.1, "deck", 0.2, 0.9, 0.5)])
    out = tmp_path / "pictures" / "wall.png"
    assert wall.main([str(tmp_path), "--png", str(out)]) == 0

    assert out.is_file() and out.stat().st_size > 1000
    assert out.resolve().as_posix() in capsys.readouterr().out


def test_an_unreadable_case_is_reported_rather_than_raised(wall, tmp_path, capsys):
    """A truncated field is an answer about the run too."""
    case = shear_case(tmp_path, "uniform (0.25 0 0)")
    (case / "constant" / "polyMesh" / "owner").write_text("garbage")
    assert wall.main([str(case)]) == 0
    assert "[skipped]" in capsys.readouterr().out


def test_every_source_is_reported_when_the_case_wrote_more_than_one(wall, tmp_path):
    """The three routes disagreeing is itself worth seeing, so none of them wins
    and silences the others."""
    case = shear_case(tmp_path, "uniform (0.25 0 0)")
    write_dat(case / "postProcessing" / "yPlus" / "0" / "yPlus.dat",
              [(0.5, "wall", 40, 60, 50)])
    write_field(case / "0.5" / "yPlus", "volScalarField",
                {"wall": _scalars([30.0] * 8)})

    found = wall.measure(case)
    assert [source["source"] for source in found["sources"]] == [
        "yPlus function object", "written yPlus field", "derived from wallShearStress"
    ]
    # The run's own answer leads, before one computed here.
    assert wall.primary(found)["source"] == "yPlus function object"


def test_the_report_offers_the_numbers_without_a_verdict_on_them(wall, tmp_path):
    text = wall.report(wall.measure(
        shear_case(tmp_path, "uniform (0.25 0 0)")
    ))
    assert "nothing here decides whether this wall resolution is good enough" in text
    for verdict in ("unacceptable", "you must", "always run", "too coarse", "fix "):
        assert verdict not in text.lower()


def test_the_toolbox_index_carries_the_script(wall):
    """`tests/test_toolbox.py` fails on a script missing from the index; this says
    which row is the missing one when it does."""
    index = (TOOLBOX / "README.md").read_text(encoding="utf-8")
    assert "| `wall.py` |" in index


def test_it_needs_nothing_the_image_does_not_have(wall):
    source = (TOOLBOX / "wall.py").read_text(encoding="utf-8")
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
    assert "import pyvista" not in source, (
        "a boundaryField is text and a y+ range should not need a render stack"
    )
    source.encode("ascii")  # raises if a non-ASCII character crept in


def test_the_docstring_keeps_the_script_advisory_and_says_how_to_run_it(wall):
    text = " ".join((wall.__doc__ or "").split())
    assert "python3 wall.py" in text
    assert "edits nothing and exits 0" in text
    # Why the boundaryField is parsed rather than read through pyvista, stated where
    # the choice was made.
    assert "pyvista" in text
    for imperative in ("you must", "always ", "make sure", "you should"):
        assert imperative not in text.lower()


def test_the_mesh_reading_is_a_siblings_and_not_a_second_one(wall):
    """One first-cell height in the toolbox, not two that nearly agree.

    This test used to forbid the literal `/ "polyMesh"` outright. #144 is the reason
    it no longer can: y+ is a wall quantity and an inlet was being reported as a
    measured wall, so the patch TYPE has to be read, and `constant/polyMesh/boundary`
    is where OpenFOAM states it. What the test is for -- no fourth mesh parser living
    in this script -- is unchanged and is now asserted directly: the geometry comes
    from `layer_report.py` and the `boundary` file from `locate.py`'s reader, which
    is the one that handles the binary spelling of it.
    """
    source = (TOOLBOX / "wall.py").read_text(encoding="utf-8")
    assert "layer_report.first_cell_heights" in source
    assert "layer_report.load" in source
    assert "locate.read_boundary" in source
    for reader in ("read_points", "read_faces", "read_labels"):
        assert reader not in source, "nothing here parses the mesh files itself"
    assert source.count('"polyMesh"') == 1, (
        "the only mesh path spelled here is the boundary file locate.py reads"
    )


# -- the binary vector fixture that was missing (#162, #163) -------------------


#: Three faces of a binary `volVectorField`, signed, with a negative x on one of them
#: so a reading that loses the components cannot come out right by accident.
CELL_U = [(1.5, 0.0, 0.0), (-0.5, 0.25, 0.0), (0.0, 0.0, 2.0)]
CELL_MAGS = [1.5, 0.5590169943749475, 2.0]


def test_a_binary_vector_field_reads_as_vectors_and_as_magnitudes(wall, tmp_path):
    """The fixture `tests/test_toolbox_wall.py` never had: every binary test here
    wrote `yPlus` as a `volScalarField`, and every `volVectorField` test used the
    ascii writer, which is why #162 sat under #138's headline number.

    Both readings come off the one decoder, so a binary vector cannot be right in one
    tool and wrong in the other -- which is exactly what #163 was.
    """
    flat = [c for face in CELL_U for c in face]
    path = write_binary_field(tmp_path / "1500" / "U", {"outlet": flat},
                              cls="volVectorField", items=3, kind="vector")
    text, payloads = wall.read_field(path)
    assert wall.field_components(text) == 3, "the FoamFile header comes back with the body"

    mags = wall.boundary_values(text, payloads)["outlet"]
    assert mags.tolist() == pytest.approx(CELL_MAGS)

    vectors = wall.boundary_vectors(text, payloads)["outlet"]
    assert vectors.shape == (3, 3)
    assert vectors.ravel().tolist() == pytest.approx(flat)
    assert vectors[1][0] < 0, "the sign survives: a magnitude cannot tell inflow from out"

    through_the_regex = wall.boundary_values(
        path.read_text(encoding="utf-8", errors="replace")
    ).get("outlet")
    assert through_the_regex is None or through_the_regex.tolist() != pytest.approx(CELL_MAGS), (
        "the fixture has to be a file the ascii reader gets wrong, or it proves nothing"
    )


def test_a_scalar_field_has_no_vectors_to_give(wall, tmp_path):
    """Three numbers a face is the shape `boundary_vectors` promises, and a
    `volScalarField` cannot supply it -- so it says so rather than reshaping."""
    path = write_field(tmp_path / "0.5" / "yPlus", "volScalarField",
                       {"hull": _scalars([1.0, 2.0, 3.0])})
    text, payloads = wall.read_field(path)
    assert wall.boundary_vectors(text, payloads) == {"hull": None}


def test_a_nonuniform_list_that_does_not_hold_what_it_declares_is_not_read(wall, tmp_path):
    """The declared count is checked against what came out of the body. This is what
    makes a size of one trustworthy downstream: `heat.patch_mean` broadcasts a single
    value over a whole patch, and the way a wrong answer got through that silently was
    a `nonuniform` list of hundreds from which a number regex scraped exactly one
    (#163)."""
    path = write_field(tmp_path / "0.5" / "T",
                       "volScalarField",
                       {"wall": "nonuniform List<scalar> 8(300 301 302)"})
    text, payloads = wall.read_field(path)
    assert wall.boundary_scalars(text, payloads) == {"wall": None}
