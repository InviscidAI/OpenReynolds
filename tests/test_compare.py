"""`openreynolds compare <a> <b>`: two studies read off disk and put side by side.

The thing being tested here is mostly the negative one. A comparison that drops the
quantities it cannot compare and then prints its all-clear is the defect
`toolbox/claims.py:394` records against its own first version, so every section is
exercised with something present on one side only, and with two claims whose unit
labels differ, and the test is that both are *said* rather than skipped.
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
        force_dat(coefficients if coefficients is not None else {"Cd": 1.0, "Cl": 0.1}),
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
