"""`openreynolds compare <a> <b>`: two studies read off disk and put side by side.

The thing being tested here is mostly the negative one. A comparison that drops the
quantities it cannot compare and then prints its all-clear is the defect
`toolbox/claims.py:394` records against its own first version, so every section is
exercised with something present on one side only, and with two claims whose unit
labels differ, and the test is that both are *said* rather than skipped.

The other half is the same defect from the other side, and it is here because the
fixture below hid it until the command was run on real studies (#141, #143). A
summary whose three headlines are numerical zeros, measured at 1e-20 against 1e-20 by
a relative rule with no floor, and which leaves out the 4% change in drag the second
run was for, is as useless as an all-clear -- and a force table with no case column
can splice two case directories together without a word. So: the drag change is
asserted present, the zeros asserted absent, the tolerance asserted printed, and
every force number asserted to carry the case it was read from.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import pytest
from click.testing import CliRunner

from openreynolds import cli, compare
from openreynolds.config import Config
from openreynolds.store import Store

CHECK_MESH = """\
Mesh stats
    points:           {points}
    faces:            {faces}
    internal faces:   {internal}
    cells:            {cells}
    hexahedra:        {cells}

Checking geometry...
    Overall domain bounding box (0 0 0) (1 1 0.1)
    Mesh non-orthogonality Max: {non_ortho} average: 4.1
    Max skewness = {skewness} OK.
    Max aspect ratio = 3.2 OK.
    Max cell openness = 2.1e-16 OK.
    Minimum face determinant = 0.71

Mesh OK.
"""

SOLVER_STEP = """\
Time = {step}

smoothSolver:  Solving for Ux, Initial residual = {ux:.3e}, Final residual = 1e-06, No Iterations 3
GAMG:  Solving for p, Initial residual = {p:.3e}, Final residual = 1e-05, No Iterations 9
ExecutionTime = {step}.0 s  ClockTime = {step} s

"""


def blank_png() -> bytes:
    """Signature and an IEND chunk: all `claims.stamp_png` reads, and nothing to draw."""
    raw = b"IEND"
    return (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 0) + raw
            + struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF))


def force_dat(columns: dict[str, float], rows: int = 12) -> str:
    """A `forceCoeffs.dat` whose rows are all the same, so the tail mean is exact."""
    header = "# Force coefficients\n# magUInf : 10\n# " + " ".join(["Time", *columns]) + "\n"
    body = "".join(
        f"{0.1 * (i + 1):g} " + " ".join(f"{value:.6g}" for value in columns.values()) + "\n"
        for i in range(rows)
    )
    return header + body


def make_study(
    root: Path,
    study_id: str,
    *,
    title: str = "lid-driven cavity",
    coefficients: dict[str, float] | None = None,
    rows: int = 12,
    cells: int = 120_000,
    claim: tuple[str, float, str] | None = ("reattachment-length", 6.32, "h"),
    kinds: tuple[str, ...] = ("vorticity", "residuals"),
    diverging: bool = False,
) -> Path:
    """One mirrored study, complete enough for every section to have something to say."""
    claims_mod, study_state = compare._toolbox("claims", "study_state")
    store = Store(root, study_id)
    store.session.title = title
    store.session.model = "claude-opus-5"
    store.session.created_at = "2026-11-01T10:00:00Z"
    store.session.instance_id = "inst-" + study_id[-4:]
    store.session.solver = "openfoam"
    store.save()
    store.append_message("user", "run it")
    store.append_message("assistant", "ran it")

    case = store.dir / "files" / "cavity"
    (case / "postProcessing" / "forces" / "0").mkdir(parents=True)
    (case / "postProcessing" / "forces" / "0" / "forceCoeffs.dat").write_text(
        force_dat(coefficients if coefficients is not None else {"Cd": 1.0, "Cl": 0.1},
                  rows=rows),
        encoding="utf-8",
    )

    steps = "".join(
        SOLVER_STEP.format(step=step,
                           ux=10.0 ** step if diverging else 1e-2 * 0.3 ** step,
                           p=1e-1 * 0.3 ** step)
        for step in range(1, 13)
    )
    (case / "log.simpleFoam").write_text(steps + "End\n", encoding="utf-8")
    (case / "log.checkMesh").write_text(
        CHECK_MESH.format(points=cells * 2, faces=cells * 3, internal=cells * 2,
                          cells=cells, non_ortho="30.1" if not diverging else "65.4",
                          skewness="1.2"),
        encoding="utf-8",
    )

    (case / ".reynolds").mkdir()
    (case / ".reynolds" / "manifest.jsonl").touch()
    for kind in kinds:
        figure = case / f"{kind}.png"
        figure.write_bytes(blank_png())
        study_state.record(kind, figure, root=case, label=kind)
    if claim is not None:
        name, value, units = claim
        stamped = case / "forces.png"
        stamped.write_bytes(blank_png())
        claims_mod.attach(
            stamped,
            [claims_mod.claim(name, value, symbol="x_r/h", units=units, source="reattach.py")],
            root=case,
        )
    return store.dir


def add_case(study_dir: Path, name: str, columns: dict[str, float], rows: int = 12) -> Path:
    """A second (or third) case directory in the mirror, with a force series of its own.

    Real studies have several: the one the study is named for, the alternative
    geometry, the orientation that was tried and rejected. The fixture above has one,
    which is why #143 -- two cases spliced into one table -- shipped green.
    """
    case = study_dir / "files" / name
    (case / "postProcessing" / "forces" / "0").mkdir(parents=True)
    (case / "postProcessing" / "forces" / "0" / "force.dat").write_text(
        force_dat(columns, rows=rows), encoding="utf-8")
    return case


@pytest.fixture
def studies(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "studies"
    monkeypatch.setattr(Config, "load", classmethod(lambda cls: Config(studies_dir=root)))
    return root


def run(*args: str):
    return CliRunner().invoke(cli.main, ["compare", *args])


def payload(*args: str) -> dict:
    result = run(*args, "--json")
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


# -- the happy case ------------------------------------------------------------------


def test_two_identical_studies_report_no_differences(studies):
    make_study(studies, "20260101-120000-aaaa")
    make_study(studies, "20260102-120000-bbbb")

    result = run("20260101-120000-aaaa", "20260102-120000-bbbb")

    assert result.exit_code == 0, result.output
    assert "no differences" in result.output
    assert "missing" not in result.output
    assert "not compared" not in result.output


def test_the_sections_are_all_there(studies):
    make_study(studies, "20260101-120000-aaaa")
    make_study(studies, "20260102-120000-bbbb")

    result = run("20260101-120000-aaaa", "20260102-120000-bbbb")

    for section in ("headline numbers", "convergence", "mesh", "claims", "artifacts"):
        assert section in result.output, result.output
    assert "claude-opus-5" in result.output
    assert "120,000" in result.output or "120000" in result.output


# -- the numbers ---------------------------------------------------------------------


def test_a_changed_coefficient_is_reported_with_its_relative_difference(studies):
    make_study(studies, "20260101-120000-aaaa", coefficients={"Cd": 1.0, "Cl": 0.1})
    make_study(studies, "20260102-120000-bbbb", coefficients={"Cd": 1.2, "Cl": 0.1})

    report = payload("20260101-120000-aaaa", "20260102-120000-bbbb")

    series = {row["name"]: row for row in report["forces"]["series"]}
    assert series["Cd"]["a"]["tail_mean"] == pytest.approx(1.0)
    assert series["Cd"]["b"]["tail_mean"] == pytest.approx(1.2)
    assert series["Cd"]["tail_mean"]["absolute"] == pytest.approx(0.2)
    # `claims.apart`: on the larger magnitude, so 0.2 / 1.2 and not 0.2 / 1.0.
    assert series["Cd"]["tail_mean"]["relative"] == pytest.approx(0.2 / 1.2)
    assert any("Cd by 16.7%" in line for line in report["summary"]["differ"])
    # Cl did not move, and a difference inside the tolerance is not called one.
    assert series["Cl"]["tail_mean"]["relative"] == pytest.approx(0.0)
    assert not any("Cl" in line for line in report["summary"]["differ"])


def test_a_series_only_one_study_logged_is_reported_as_missing(studies):
    make_study(studies, "20260101-120000-aaaa",
               coefficients={"Cd": 1.0, "Cl": 0.1, "CmPitch": 0.02})
    make_study(studies, "20260102-120000-bbbb", coefficients={"Cd": 1.0, "Cl": 0.1})

    result = run("20260101-120000-aaaa", "20260102-120000-bbbb")
    report = payload("20260101-120000-aaaa", "20260102-120000-bbbb")

    series = {row["name"]: row for row in report["forces"]["series"]}
    assert series["CmPitch"]["missing"] == "b"
    assert series["CmPitch"]["b"] is None
    assert series["CmPitch"]["tail_mean"]["relative"] is None
    assert "force series CmPitch only in a" in report["summary"]["one_sided"]
    assert "no differences" not in result.output
    assert "CmPitch" in result.output and "missing" in result.output


def test_a_run_that_diverged_is_told_apart_from_one_that_did_not(studies):
    make_study(studies, "20260101-120000-aaaa")
    make_study(studies, "20260102-120000-bbbb", diverging=True)

    report = payload("20260101-120000-aaaa", "20260102-120000-bbbb")

    fields = {row["field"]: row for row in report["convergence"]["fields"]}
    assert fields["Ux"]["a"]["shape"] == "falling"
    assert fields["Ux"]["b"]["shape"] == "diverging"
    assert any("Ux residuals" in line for line in report["summary"]["differ"])
    assert any("FOAM" not in line for line in report["convergence"]["ended"]["a"])


def test_the_mesh_metrics_sit_side_by_side(studies):
    make_study(studies, "20260101-120000-aaaa", cells=120_000)
    make_study(studies, "20260102-120000-bbbb", cells=480_000)

    report = payload("20260101-120000-aaaa", "20260102-120000-bbbb")

    metrics = {row["metric"]: row for row in report["mesh"]["metrics"]}
    assert (metrics["cells"]["a"], metrics["cells"]["b"]) == (120_000, 480_000)
    assert any("checkMesh cells" in line for line in report["summary"]["differ"])


# -- the tolerance and the floor under it (#141) --------------------------------------


def test_a_four_percent_change_in_drag_is_the_headline_and_not_dropped(studies):
    """These are #141's own numbers: Cd 0.4236 against 0.4067 on two Ahmed-body runs,
    a 4.0% change in drag and the entire point of the second run, left out of the
    summary because 0.040 was under the 5% figure-stamp tolerance this had borrowed.
    The per-row table was right and the verdict was wrong, which is the failure that
    matters: the verdict is what the command exists to print."""
    make_study(studies, "20260101-120000-aaaa",
               coefficients={"Cd": 0.42364, "Cl": 0.43698})
    make_study(studies, "20260102-120000-bbbb",
               coefficients={"Cd": 0.40668, "Cl": 0.43147})

    result = run("20260101-120000-aaaa", "20260102-120000-bbbb")
    report = payload("20260101-120000-aaaa", "20260102-120000-bbbb")

    assert any(line.startswith("Cd by 4.0% on the tail mean")
               for line in report["summary"]["differ"]), report["summary"]["differ"]
    # With the two numbers beside the percentage, so a reader does not have to take
    # 4.0% on faith or scroll back to the table for what moved.
    assert any("0.42364 against 0.40668" in line for line in report["summary"]["differ"])
    assert "differs Cd by 4.0%" in result.output
    assert "no differences" not in result.output


def test_two_quantities_that_are_both_zero_are_unchanged_not_wildly_apart(studies):
    """`CmRoll` and `Cs` are identically zero on a symmetric body in symmetric flow,
    and what a solver logs for them is residue five orders below Cd. #141 made them
    the summary's top two findings -- 71.6% and 91.5% apart -- because `claims.apart`
    normalises on the larger magnitude and had no floor, so two zeros are always
    almost 100% apart. Both are still in the table with their relative difference;
    neither is a difference."""
    make_study(studies, "20260101-120000-aaaa",
               coefficients={"Cd": 0.42364, "CmRoll": 3.2186e-06, "Cs": -1.3074e-05})
    make_study(studies, "20260102-120000-bbbb",
               coefficients={"Cd": 0.40668, "CmRoll": 1.1318e-05, "Cs": -1.5455e-04})

    result = run("20260101-120000-aaaa", "20260102-120000-bbbb")
    report = payload("20260101-120000-aaaa", "20260102-120000-bbbb")

    differ = report["summary"]["differ"]
    assert not any("CmRoll" in line or "Cs" in line for line in differ), differ
    assert any("Cd by 4.0%" in line for line in differ)
    # Said out loud, with the numbers and the scale that put them under the floor --
    # a quantity treated as zero is a decision this made, not a fact about the study.
    zeroes = " ".join(report["summary"]["negligible"])
    assert "CmRoll" in zeroes and "Cs" in zeroes and "0.424" in zeroes
    assert "zero on both sides" in result.output
    # And the row is still there, still carrying what the relative rule said of it.
    series = {row["name"]: row for row in report["forces"]["series"]}
    assert series["CmRoll"]["tail_mean"]["relative"] > 0.5


def test_the_tolerance_is_printed_and_tol_moves_it(studies):
    """"No difference" is meaningless without the number it was measured against, and
    #141's was 5% -- inherited from a figure-stamp tolerance, never passed by the CLI
    and so not adjustable. A 2% change in Cd is a difference at the default and not at
    `--tol 0.05`, and either way the threshold is in the answer."""
    make_study(studies, "20260101-120000-aaaa", coefficients={"Cd": 1.0, "Cl": 0.1})
    make_study(studies, "20260102-120000-bbbb", coefficients={"Cd": 1.02, "Cl": 0.1})

    strict = run("20260101-120000-aaaa", "20260102-120000-bbbb")
    report = payload("20260101-120000-aaaa", "20260102-120000-bbbb")
    assert report["summary"]["tol"] == compare.TOL == 0.01
    assert "more than 1%" in strict.output
    assert any("Cd by 2.0%" in line for line in report["summary"]["differ"])

    loose = run("20260101-120000-aaaa", "20260102-120000-bbbb", "--tol", "0.05")
    relaxed = payload("20260101-120000-aaaa", "20260102-120000-bbbb", "--tol", "0.05")
    assert relaxed["summary"]["tol"] == 0.05
    assert relaxed["summary"]["differ"] == []
    assert "more than 5%" in loose.output
    assert "no differences" in loose.output


# -- which case each number came from (#143) -----------------------------------------


def test_every_force_number_says_which_case_and_series_it_came_from(studies):
    """#143's table had no case column, so six rows read as one body of numbers while
    half came from a different case directory. The provenance was already in
    `forces()` and simply not printed."""
    make_study(studies, "20260101-120000-aaaa")
    make_study(studies, "20260102-120000-bbbb")

    result = run("20260101-120000-aaaa", "20260102-120000-bbbb")
    report = payload("20260101-120000-aaaa", "20260102-120000-bbbb")

    for row in report["forces"]["series"]:
        for side in ("a", "b"):
            assert row[side]["case"] == "cavity"
            assert row[side]["series"] == "forces/forceCoeffs.dat"
    assert "a case" in result.output and "b case" in result.output
    assert "cavity/forces/forceCoeffs.dat" in result.output


def test_two_cases_are_never_spliced_into_one_table_without_saying_so(studies):
    """Side a of #143 took `Cd`/`Cl`/`Cp` from `case` and `total_x/y/z` from
    `case_axisX`, unlabelled, and summarised the result as "total_z by 77.6%". Each
    number may still come from the case that logged it -- that is how a study with an
    alternative geometry reads -- but which case has to travel with it, into the
    summary line as well as the table."""
    first = make_study(studies, "20260101-120000-aaaa")
    add_case(first, "cavity_axisX", {"total_x": 181.47}, rows=13)
    second = make_study(studies, "20260102-120000-bbbb")
    add_case(second, "cavity_Fr0316", {"total_x": -3.6904}, rows=13)

    result = run("20260101-120000-aaaa", "20260102-120000-bbbb")
    report = payload("20260101-120000-aaaa", "20260102-120000-bbbb")

    series = {row["name"]: row for row in report["forces"]["series"]}
    assert series["Cd"]["a"]["case"] == "cavity"
    assert series["total_x"]["a"]["case"] == "cavity_axisX"
    assert series["total_x"]["b"]["case"] == "cavity_Fr0316"
    assert any("not all from one case" in note
               for note in report["forces"]["notes"]["a"]), report["forces"]["notes"]["a"]
    assert any("a from cavity_axisX, b from cavity_Fr0316" in line
               for line in report["summary"]["differ"]), report["summary"]["differ"]
    assert "cavity_axisX" in result.output


def test_a_series_the_chooser_passed_over_is_named(studies):
    """`case/postProcessing/forces/force.dat`, 13 rows, was never compared and never
    mentioned in #143: the chooser took a one-row `bladeF/force.dat` in the same case
    and the note only fired when a column name repeated in a *later* case. What is not
    compared is said out loud, which is this module's whole premise."""
    first = make_study(studies, "20260101-120000-aaaa")
    other = first / "files" / "cavity" / "postProcessing" / "coefficientsOther" / "0"
    other.mkdir(parents=True)
    (other / "coefficient.dat").write_text(force_dat({"Cd": 7.7}), encoding="utf-8")
    make_study(studies, "20260102-120000-bbbb")

    report = payload("20260101-120000-aaaa", "20260102-120000-bbbb")

    notes = report["forces"]["notes"]["a"]
    assert any("coefficientsOther/coefficient.dat" in note and "not compared" in note
               for note in notes), notes


def test_the_columns_ranked_off_the_comparison_are_named(studies):
    """The other half of #143's silence. `results.plot_columns` answers "what fits on
    one axes", and once any column matches `^C[A-Za-z]*$` it returns only those -- the
    right call for a figure and a selector nobody chose for a comparison.

    On the blade study behind #143 that dropped `thrust`, `drag`, `torque` and `area`
    out of `bladeF/force.dat`, which are the answers a wind-turbine run is for, and on
    both sides it drops the `pressure_*` and `viscous_*` splits of every force. None of
    it was mentioned. The ranking stays; the silence does not."""
    make_study(studies, "20260101-120000-aaaa",
               coefficients={"thrust": 507.7, "drag": 25.5, "torque": 134.0,
                             "Cd": 0.01076, "Cl": 0.73042, "area": 2.227})
    make_study(studies, "20260102-120000-bbbb",
               coefficients={"Cd": 0.01076, "Cl": 0.73042})

    report = payload("20260101-120000-aaaa", "20260102-120000-bbbb")
    series = {row["name"] for row in report["forces"]["series"]}
    assert series == {"Cd", "Cl"}, "the ranking itself is unchanged"

    (note,) = [n for n in report["forces"]["notes"]["a"] if "also holds" in n]
    for dropped in ("thrust", "drag", "torque", "area"):
        assert dropped in note, note
    assert "not compared" in note and "Cd, Cl" in note, note


def test_a_one_row_series_is_not_called_a_mean(studies):
    """The `bladeF/force.dat` behind #143's `Cd` is a hand-written file with one data
    row, and `tail_mean` of one row is that row -- printed under a column headed "a
    mean". A single measurement labelled as an average of something is a number
    claiming a confidence it does not have."""
    make_study(studies, "20260101-120000-aaaa", rows=1,
               coefficients={"Cd": 0.01076, "Cl": 0.73042})
    make_study(studies, "20260102-120000-bbbb",
               coefficients={"Cd": 0.01076, "Cl": 0.73042})

    result = run("20260101-120000-aaaa", "20260102-120000-bbbb")
    report = payload("20260101-120000-aaaa", "20260102-120000-bbbb")

    series = {row["name"]: row for row in report["forces"]["series"]}
    assert series["Cd"]["a"]["rows"] == 1 and series["Cd"]["b"]["rows"] == 12
    assert any("Cd in cavity/forces/forceCoeffs.dat has one data row" in note
               for note in report["forces"]["notes"]["a"]), report["forces"]["notes"]["a"]
    assert "(1 row)" in result.output


def test_processor_directories_are_one_case_and_not_four(studies, tmp_path):
    """A case solved on four cores keeps `processor0` .. `processor3` beside its own
    mesh, and each of them looks exactly like a case: the study came back as "the
    first of 6 cases" and `compare` would have read one rank's postProcessing as a
    case of its own."""
    study_state, = compare._toolbox("study_state")
    mirror = make_study(studies, "20260101-120000-aaaa") / "files"
    for rank in range(4):
        where = mirror / "cavity" / f"processor{rank}" / "postProcessing" / "forces" / "0"
        where.mkdir(parents=True)
        (where / "forceCoeffs.dat").write_text(force_dat({"Cd": 1.0}), encoding="utf-8")

    assert compare.cases(mirror) == [mirror / "cavity"]

    # And the same directories at a study home that is itself the case (the legacy
    # `/work` layout), which is where the count of six came from.
    home = tmp_path / "decomposed"
    (home / ".reynolds").mkdir(parents=True)
    (home / "system").mkdir()
    (home / "system" / "controlDict").write_text("", encoding="utf-8")
    for rank in range(4):
        (home / f"processor{rank}" / "constant" / "polyMesh").mkdir(parents=True)

    assert study_state.discover_cases(home) == [home.resolve()]


def test_a_case_marked_as_ruled_out_is_not_chosen_silently(studies):
    """`case_axisX_ruledout` is a name somebody wrote to record a decision. #143
    compared its sibling in place of the case the study was named for; a directory
    that sorts before the real case would have been read in its place outright. It is
    skipped while another case exists, and named either way -- skipping it without a
    word is the same failure as choosing it without one."""
    first = make_study(studies, "20260101-120000-aaaa")
    add_case(first, "basin_ruledout", {"Cd": 9.9})
    make_study(studies, "20260102-120000-bbbb")

    result = run("20260101-120000-aaaa", "20260102-120000-bbbb")
    report = payload("20260101-120000-aaaa", "20260102-120000-bbbb")

    series = {row["name"]: row for row in report["forces"]["series"]}
    assert series["Cd"]["a"]["case"] == "cavity"
    assert series["Cd"]["a"]["tail_mean"] == pytest.approx(1.0)
    assert any("basin_ruledout" in note and "ruled out" in note
               for note in report["forces"]["notes"]["a"]), report["forces"]["notes"]["a"]
    assert "basin_ruledout" in result.output


def test_a_study_whose_only_case_is_ruled_out_is_still_read(studies):
    """The other side of the rule. "Ruled out" means "prefer the other one", and with
    no other one, dropping the study's only force history would be the silence this
    module exists to refuse."""
    first = make_study(studies, "20260101-120000-aaaa")
    (first / "files" / "cavity").rename(first / "files" / "cavity_ruledout")
    make_study(studies, "20260102-120000-bbbb")

    report = payload("20260101-120000-aaaa", "20260102-120000-bbbb")

    series = {row["name"]: row for row in report["forces"]["series"]}
    assert series["Cd"]["a"]["case"] == "cavity_ruledout"
    assert series["Cd"]["a"]["tail_mean"] == pytest.approx(1.0)
    assert any("read anyway" in note for note in report["forces"]["notes"]["a"])


# -- claims --------------------------------------------------------------------------


def test_claims_that_disagree_are_flagged(studies):
    make_study(studies, "20260101-120000-aaaa", claim=("reattachment-length", 6.32, "h"))
    make_study(studies, "20260102-120000-bbbb", claim=("reattachment-length", 7.5667, "h"))

    result = run("20260101-120000-aaaa", "20260102-120000-bbbb")
    report = payload("20260101-120000-aaaa", "20260102-120000-bbbb")

    row = next(r for r in report["claims"] if r["name"] == "reattachment-length")
    assert row["comparable"] is True
    assert row["agree"] is False
    assert row["apart"] == pytest.approx(abs(6.32 - 7.5667) / 7.5667)
    assert "DIFFER" in result.output
    assert any("reattachment-length" in line for line in report["summary"]["differ"])


def test_claims_that_agree_are_said_to_agree(studies):
    make_study(studies, "20260101-120000-aaaa", claim=("reattachment-length", 6.32, "h"))
    make_study(studies, "20260102-120000-bbbb", claim=("reattachment-length", 6.33, "h"))

    report = payload("20260101-120000-aaaa", "20260102-120000-bbbb")

    row = next(r for r in report["claims"] if r["name"] == "reattachment-length")
    assert (row["comparable"], row["agree"]) == (True, True)
    assert report["summary"]["differ"] == []


def test_claims_under_different_units_are_not_comparable_rather_than_converted(studies):
    """`claims.py`'s own rule, and the reason it exists: 6.32 step heights against
    0.0594 metres is one measurement, and a tool that guesses the conversion invents
    its own errors. The numbers are close enough that a tolerance applied before the
    labels were read would have called them an all-clear."""
    make_study(studies, "20260101-120000-aaaa", claim=("reattachment-length", 6.32, "h"))
    make_study(studies, "20260102-120000-bbbb", claim=("reattachment-length", 6.30, "m"))

    result = run("20260101-120000-aaaa", "20260102-120000-bbbb")
    report = payload("20260101-120000-aaaa", "20260102-120000-bbbb")

    row = next(r for r in report["claims"] if r["name"] == "reattachment-length")
    assert row["comparable"] is False
    assert row["agree"] is None
    assert row["apart"] is None
    assert any("not converted" in line for line in report["summary"]["not_comparable"])
    assert "not comparable" in result.output
    assert "no differences" not in result.output


def test_a_claim_only_one_study_stamped_is_reported_as_missing(studies):
    make_study(studies, "20260101-120000-aaaa", claim=("reattachment-length", 6.32, "h"))
    make_study(studies, "20260102-120000-bbbb", claim=None)

    report = payload("20260101-120000-aaaa", "20260102-120000-bbbb")

    row = next(r for r in report["claims"] if r["name"] == "reattachment-length")
    assert row["missing"] == "b"
    assert "claim reattachment-length only in a" in report["summary"]["one_sided"]


# -- artifacts -----------------------------------------------------------------------


def test_what_one_study_produced_and_the_other_did_not(studies):
    make_study(studies, "20260101-120000-aaaa", kinds=("vorticity", "residuals"))
    make_study(studies, "20260102-120000-bbbb", kinds=("vorticity", "animation"))

    report = payload("20260101-120000-aaaa", "20260102-120000-bbbb")

    kinds = {row["kind"]: row for row in report["artifacts"]}
    assert kinds["vorticity"]["missing"] is None
    assert kinds["residuals"]["missing"] == "b"
    assert kinds["animation"]["missing"] == "a"
    assert "residuals artifacts only in a" in report["summary"]["one_sided"]
    assert "animation artifacts only in b" in report["summary"]["one_sided"]


# -- the command itself --------------------------------------------------------------


def test_json_is_one_object(studies):
    make_study(studies, "20260101-120000-aaaa")
    make_study(studies, "20260102-120000-bbbb")

    result = run("20260101-120000-aaaa", "20260102-120000-bbbb", "--json")

    assert result.exit_code == 0, result.output
    assert len(result.stdout.strip().splitlines()) == 1
    report = json.loads(result.stdout)
    assert report["studies"]["a"]["study_id"] == "20260101-120000-aaaa"
    assert report["studies"]["b"]["title"] == "lid-driven cavity"
    assert report["studies"]["a"]["messages"] == 2


def test_the_group_output_flag_is_the_same_request_as_its_own_json(studies):
    make_study(studies, "20260101-120000-aaaa")
    make_study(studies, "20260102-120000-bbbb")

    result = CliRunner().invoke(cli.main, [
        "--output-format", "stream-json", "compare",
        "20260101-120000-aaaa", "20260102-120000-bbbb",
    ])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["studies"]["b"]["study_id"] == "20260102-120000-bbbb"


def test_a_study_that_does_not_exist_says_so(studies):
    make_study(studies, "20260101-120000-aaaa")

    result = run("20260101-120000-aaaa", "20260199-999999-zzzz")

    assert result.exit_code == 1
    assert "No study at" in result.output
    assert "20260199-999999-zzzz" in result.output
    # And it did not conjure the directory on the way to finding that out.
    assert not (studies / "20260199-999999-zzzz").exists()


def test_with_no_ids_it_takes_the_two_newest_and_says_which(studies):
    make_study(studies, "20260101-120000-aaaa")
    make_study(studies, "20260102-120000-bbbb")
    make_study(studies, "20260103-120000-cccc")

    result = run()

    assert result.exit_code == 0, result.output
    assert "20260102-120000-bbbb" in result.output
    assert "20260103-120000-cccc" in result.output
    assert "20260101-120000-aaaa" not in result.output
    assert "nothing named" in result.output


def test_one_id_named_is_that_one_against_the_newest_other(studies):
    make_study(studies, "20260101-120000-aaaa")
    make_study(studies, "20260103-120000-cccc")

    result = run("20260101-120000-aaaa")

    assert result.exit_code == 0, result.output
    assert "one id named" in result.output
    # Not itself, which is the newest: a study against itself is a page of zeroes.
    assert "b = 20260103-120000-cccc" in result.output


def test_one_study_is_not_two(studies):
    make_study(studies, "20260101-120000-aaaa")

    result = run()

    assert result.exit_code == 1
    assert "two studies" in result.output


def test_the_studies_directory_can_be_named(tmp_path, monkeypatch):
    """The only subcommand with a reason to read studies it is not standing in: it
    starts no session and `studies_dir` has no environment variable behind it."""
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setattr(
        Config, "load", classmethod(lambda cls: Config(studies_dir=tmp_path / "empty"))
    )
    make_study(elsewhere, "20260101-120000-aaaa")
    make_study(elsewhere, "20260102-120000-bbbb")

    result = run("--studies-dir", str(elsewhere))

    assert result.exit_code == 0, result.output
    assert "20260102-120000-bbbb" in result.output


def test_a_coefficient_is_not_called_zero_because_a_force_in_newtons_shares_the_table(
        studies):
    """`forces()` keys by column name across cases by design, so one table holds
    dimensionless coefficients beside forces in newtons. One scale for the whole table
    meant a 1311 N `total_z` set the floor at 1.311, and every coefficient of a real
    wind-turbine-blade study fell under it -- `Cl = 0.73`, that study's answer, was
    declared zero and dropped from the comparison (#178)."""
    for study_id in ("20260101-120000-aaaa", "20260102-120000-bbbb"):
        made = make_study(studies, study_id,
                          coefficients={"Cd": 0.0108, "Cl": 0.73, "Cp": -0.554})
        add_case(made, "cavity_axis",
                 {"total_x": 56.3, "total_y": -12.4, "total_z": 1311.3})

    report = payload("20260101-120000-aaaa", "20260102-120000-bbbb")
    zeroes = " ".join(report["summary"]["negligible"])
    for name in ("Cd", "Cl", "Cp"):
        assert name not in zeroes, zeroes
    assert report["summary"]["negligible"] == []
    series = {row["name"]: row for row in report["forces"]["series"]}
    assert set(series) >= {"Cd", "Cl", "Cp", "total_x", "total_y", "total_z"}


def test_a_difference_in_the_coefficients_is_reported_whatever_the_newtons_are(studies):
    """The `zero on both sides` branch skips the tolerance test, so a floor that is
    wrong for one side's units is a real difference printed as `no differences` --
    which is the failure #141 was filed for, reachable again through #141's fix."""
    made = make_study(studies, "20260101-120000-aaaa", coefficients={"Cl": 0.73})
    add_case(made, "cavity_axis", {"total_z": 1311.3})
    made = make_study(studies, "20260102-120000-bbbb", coefficients={"Cl": 1.5})
    add_case(made, "cavity_axis", {"total_z": 1311.3})

    result = run("20260101-120000-aaaa", "20260102-120000-bbbb")
    report = payload("20260101-120000-aaaa", "20260102-120000-bbbb")
    assert any(line.startswith("Cl by ") for line in report["summary"]["differ"]), \
        report["summary"]
    assert "no differences" not in result.output
