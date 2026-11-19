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
import sys
from pathlib import Path

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


def write_field(path: Path, cls: str, entries: dict[str, str]) -> Path:
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
        "internalField   uniform 0;\n\n"
        f"boundaryField\n{{\n{blocks}\n}}\n"
    )
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
    assert set(source["patches"][0]) == {
        "patch", "min", "max", "mean", "faces", "spans", "fractions"
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


def test_the_mesh_reading_is_layer_reports_and_not_a_second_one(wall):
    """One first-cell height in the toolbox, not two that nearly agree."""
    source = (TOOLBOX / "wall.py").read_text(encoding="utf-8")
    assert "layer_report.first_cell_heights" in source
    assert "layer_report.load" in source
    for reader in ("read_points", "read_faces", "read_labels", '/ "polyMesh"'):
        assert reader not in source, "nothing here reads the mesh files itself"
