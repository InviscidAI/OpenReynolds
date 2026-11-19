"""`report.py`: a study assembled into one PDF.

The script's own reason for existing is that an executive summary which prints a number
beside a figure is the failure `claims.py` was written for, so most of what is checked
here is the tie between the two: a stamped figure's number comes out verified, a number
with no figure behind it comes out labelled as having none, and a figure that has been
deleted since appears as MISSING rather than disappearing.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
TOOLBOX = HERE.parents[0] / "openreynolds" / "toolbox"


def load(name: str):
    """Import a toolbox script by path; the directory is data, not a package."""
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report():
    return load("report")


@pytest.fixture(scope="module")
def claims():
    return load("claims")


@pytest.fixture(scope="module")
def study_state():
    return load("study_state")


# -- the synthetic study -------------------------------------------------------------

CONTROL_DICT = """\
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }

application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         500;
deltaT          1;
writeControl    timeStep;
writeInterval   100;
purgeWrite      3;
runTimeModifiable true;

functions
{
    forceCoeffs1
    {
        type            forceCoeffs;
        writeControl    timeStep;
        writeInterval   1;
    }
}
"""

FV_SCHEMES = """\
FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }

ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss limitedLinear 1;
    div(phi,k)      bounded Gauss limitedLinear 1;
}
laplacianSchemes { default Gauss linear corrected; }
"""

FV_SOLUTION = """\
FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }

solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-07;
        relTol          0.01;
    }
    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-08;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
    residualControl
    {
        p               1e-4;
        U               1e-4;
    }
}

relaxationFactors
{
    equations { U 0.9; ".*" 0.9; }
}
"""

TRANSPORT = """\
FoamFile { version 2.0; format ascii; class dictionary; object transportProperties; }

transportModel  Newtonian;
nu              1e-05;
"""

TURBULENCE = """\
FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }

simulationType  RAS;

RAS
{
    RASModel        kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}
"""

FIELD_U = """\
FoamFile { version 2.0; format ascii; class volVectorField; object U; }

dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (1 0 0);

boundaryField
{
    inlet   { type fixedValue; value uniform (1 0 0); }
    outlet  { type zeroGradient; }
    walls   { type noSlip; }
}
"""

CHECK_MESH = """\
Create mesh for time = 0

Checking geometry...
    Overall domain bounding box (-4 -2 0) (12 2 0.1)
Mesh stats
    points:           42000
    faces:            81000
    internal faces:   39000
    cells:            20000
    hexahedra:        20000
Checking topology...
Checking patch topology for multiply connected surfaces...
    Patch               Faces    Points   Surface topology
    inlet               200      402      ok (non-closed singly connected)
    outlet              200      402      ok (non-closed singly connected)
    walls               800      1600     ok (non-closed singly connected)

    Mesh non-orthogonality Max: 42.7 average: 5.1
    Max skewness = 1.8 OK.
    Max aspect ratio = 12.4 OK.
    Max cell openness = 2.1e-16 OK.
    Minimum face determinant = 0.71
Mesh OK.
"""

SOLVER_LOG = """\
/*---------------------------------------------------------------------------*\\
Build  : 2512
Exec   : simpleFoam
\\*---------------------------------------------------------------------------*/

Time = 1

smoothSolver:  Solving for Ux, Initial residual = 1.0e-01, Final residual = 2.0e-03, No Iterations 3
GAMG:  Solving for p, Initial residual = 9.0e-01, Final residual = 1.0e-02, No Iterations 9
ExecutionTime = 1.4 s  ClockTime = 1 s

Time = 2

smoothSolver:  Solving for Ux, Initial residual = 5.0e-03, Final residual = 1.0e-04, No Iterations 3
GAMG:  Solving for p, Initial residual = 4.0e-02, Final residual = 5.0e-04, No Iterations 7
ExecutionTime = 2.8 s  ClockTime = 2 s

End
"""

FORCE_COEFFS = """\
# Force coefficients
# magUInf     : 1
# lRef        : 1
# Aref        : 0.1
# rhoInf      : 1
# Time          Cd              Cl              Cm
1               1.2500          0.0100          0.0020
2               1.2400          0.0110          0.0021
3               1.2300          0.0105          0.0019
4               1.2350          0.0108          0.0020
"""


def _png(path: Path, text: str = "figure", size: tuple[float, float] = (3.0, 2.0)) -> Path:
    """A real PNG, drawn rather than faked, because `claims.stamp_png` rewrites chunks
    and the figure pages read every panel back through `imread`."""
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=size, dpi=70)
    figure.text(0.5, 0.5, text, ha="center", va="center")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(path), facecolor="white")
    plt.close(figure)
    return path


@pytest.fixture
def study(tmp_path, claims, study_state):
    """A study with one case: dictionaries, a mesh check, a solver log, a force record,
    a stamped figure, an unstamped figure, and a manifest row whose file is gone."""
    root = tmp_path / "wake"
    case = root / "cylinder"
    (root / ".reynolds").mkdir(parents=True)
    for name, text in (
        ("system/controlDict", CONTROL_DICT),
        ("system/fvSchemes", FV_SCHEMES),
        ("system/fvSolution", FV_SOLUTION),
        ("constant/transportProperties", TRANSPORT),
        ("constant/turbulenceProperties", TURBULENCE),
        ("0/U", FIELD_U),
        ("log.checkMesh", CHECK_MESH),
        ("log.simpleFoam", SOLVER_LOG),
        ("postProcessing/forceCoeffs1/0/coefficient.dat", FORCE_COEFFS),
    ):
        target = case / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    stamped = _png(case / "renders" / "vorticity.png", "vorticity with x_r marked")
    claims.attach(
        stamped,
        [claims.claim("reattachment-length", 6.32, symbol="x_r/h", source="reattach.py")],
        root=root, case="cylinder", record=True,
    )
    study_state.record("vorticity", stamped, root=root, case="cylinder",
                       label="vorticity at the latest time")

    plain = _png(case / "renders" / "mesh.png", "mesh cut")
    study_state.record("mesh-full", plain, root=root, case="cylinder",
                       label="the mesh, cut on z")

    gone = case / "renders" / "deleted.png"
    _png(gone, "this one gets tidied up")
    study_state.record("pressure", gone, root=root, case="cylinder",
                       label="pressure, since deleted")
    gone.unlink()

    phases = study_state.load_phases(root)
    for row in phases["phases"]:
        if row["name"] == "animate":
            row["status"] = "skipped"
            row["note"] = "no moving parts in this case"
    study_state.save_phases(phases, root)

    return root


def _page_count(pdf: Path) -> int:
    """Pages, read out of the file itself.

    There is no PDF reader on this image either, so the `/Count` of the document's one
    `/Type /Pages` node is read directly. Counting `/Type /Page` would be off by one --
    `/Pages` matches it too.
    """
    found = re.findall(rb"/Count (\d+)", pdf.read_bytes())
    assert found, "this is not a PDF with a page tree in it"
    return int(found[-1])


def run(report, argv: list[str]) -> int:
    return report.main(argv)


# -- the document --------------------------------------------------------------------


def test_it_writes_a_pdf_of_the_pages_it_says_it_wrote(report, study, tmp_path, capsys):
    out = tmp_path / "pack.pdf"
    side = tmp_path / "went-in.json"
    assert run(report, [str(study), "--out", str(out), "--json", str(side)]) == 0
    assert out.is_file()
    # Four sections: a cover, a figure page per registered kind, the methods pages and
    # the gaps page. The count is asserted against the file rather than against a
    # number written here, because the point is that the sidecar does not overstate it.
    record = json.loads(side.read_text())
    assert record["pages"] == _page_count(out)
    assert record["pages"] >= 4
    assert record["page_sections"][0] == "summary"
    assert "methods" in record["page_sections"] and "gaps" in record["page_sections"]
    # A page of drawn PNGs is not a page of nothing.
    assert out.stat().st_size > 20_000
    assert str(out.resolve()) in capsys.readouterr().out


def test_the_executive_numbers_come_from_the_stamped_claim(report, study, tmp_path):
    side = tmp_path / "went-in.json"
    assert run(report, [str(study), "--out", str(tmp_path / "r.pdf"),
                        "--json", str(side)]) == 0
    numbers = json.loads(side.read_text())["numbers"]
    verified = [row for row in numbers if row["state"] == "verified"]
    assert verified, f"nothing was verified against a figure: {numbers}"
    claim = [row for row in verified if row["label"] == "reattachment-length"]
    assert claim, verified
    assert claim[0]["value"] == pytest.approx(6.32)
    assert "vorticity.png" in claim[0]["why"]


def test_a_number_with_no_figure_behind_it_is_not_printed_bare(report, study, tmp_path):
    """The coefficients are read out of `coefficient.dat` and nothing drew them, so
    they must reach the cover labelled, not quoted."""
    side = tmp_path / "went-in.json"
    assert run(report, [str(study), "--out", str(tmp_path / "r.pdf"),
                        "--json", str(side)]) == 0
    numbers = json.loads(side.read_text())["numbers"]
    coefficients = [row for row in numbers if row["label"].startswith("Cd")]
    assert coefficients, [row["label"] for row in numbers]
    for row in coefficients:
        assert row["state"] == "unverified"
        assert "no figure" in row["why"]
        assert row["figure"] == ""


def test_a_claim_whose_figure_is_gone_is_unverified_not_verified(
    report, study, tmp_path, claims, study_state
):
    """A claim recorded in the manifest outlives the picture it was stamped into, and
    at that point there is nothing behind the number any more."""
    orphan = study / "cylinder" / "renders" / "orphan.png"
    _png(orphan, "about to vanish")
    claims.attach(orphan, [claims.claim("lift-coefficient", 0.011, symbol="Cl")],
                  root=study, case="cylinder", record=True)
    orphan.unlink()

    side = tmp_path / "went-in.json"
    assert run(report, [str(study), "--out", str(tmp_path / "r.pdf"),
                        "--json", str(side)]) == 0
    rows = [row for row in json.loads(side.read_text())["numbers"]
            if row["label"] == "lift-coefficient"]
    assert rows and rows[0]["state"] == "unverified"
    assert "gone" in rows[0]["why"]


def test_a_measured_number_that_disagrees_with_its_figure_says_so(
    report, study, tmp_path, claims
):
    """The F-36 shape: the figure was drawn from one number and the files say another.
    Both numbers belong on the cover, with the gap between them."""
    figure = study / "cylinder" / "renders" / "drag.png"
    _png(figure, "Cd annotated")
    claims.attach(figure, [claims.claim("Cd", 7.5667, symbol="Cd", source="analyze.py")],
                  root=study, case="cylinder", record=True)

    side = tmp_path / "went-in.json"
    assert run(report, [str(study), "--out", str(tmp_path / "r.pdf"),
                        "--json", str(side)]) == 0
    rows = [row for row in json.loads(side.read_text())["numbers"]
            if row["state"] == "disagrees"]
    assert rows, "a measured Cd of ~1.24 against a figure stamped 7.5667 is a disagreement"
    assert "7.5667" in rows[0]["why"] and "drag.png" in rows[0]["why"]


def test_a_deleted_artifact_is_on_the_page_as_missing(report, study, tmp_path):
    side = tmp_path / "went-in.json"
    assert run(report, [str(study), "--out", str(tmp_path / "r.pdf"),
                        "--json", str(side)]) == 0
    record = json.loads(side.read_text())
    missing = record["missing"]
    assert [row["rel"] for row in missing] == ["cylinder/renders/deleted.png"]
    assert missing[0]["state"] == "MISSING"
    assert missing[0]["page"] >= 1
    # And it is still one of the figures, not dropped from the list.
    assert any(row["rel"].endswith("deleted.png") for row in record["figures"])


# -- the methods section -------------------------------------------------------------


def test_the_methods_section_names_what_the_dictionaries_say(report, study):
    case, rule, others = report.choose_case(study, "")
    lines, read = report.methods_lines(case, rule, others, layers=False)
    text = "\n".join(lines)
    for fact in ("simpleFoam", "kOmegaSST", "RAS", "endTime", "500",
                 "writeInterval", "steadyState", "limitedLinear", "GAMG",
                 "residualControl", "noSlip", "zeroGradient", "1e-05"):
        assert fact in text, f"the methods section does not mention {fact}"
    assert "20,000" in text, "the cell count from checkMesh is missing"
    assert "ran to the end of controlDict" in text or "ended:" in text
    for source in ("system/controlDict", "system/fvSchemes", "system/fvSolution",
                   "constant/transportProperties", "constant/turbulenceProperties",
                   "0/", "log.checkMesh", "log.simpleFoam"):
        assert source in read, f"{source} is not named as a source"


def test_the_methods_section_reads_only_the_top_level_of_controldict(report, study):
    """The `functions` block in the fixture carries `writeInterval 1;` and the case
    carries `writeInterval 100;`. Reading the function object's cadence as the case's
    is `parse_control`'s documented failure, so it is checked here."""
    case, rule, others = report.choose_case(study, "")
    lines, _ = report.methods_lines(case, rule, others, layers=False)
    cadence = [line for line in lines if "writeInterval" in line]
    assert cadence and all("100" in line for line in cadence), cadence


def test_the_gaps_section_quotes_the_phase_the_study_skipped(report, study):
    text = "\n".join(report.gaps(study))
    assert "animate" in text and "skipped" in text
    assert "no moving parts in this case" in text


# -- the hero, the logo and the template ----------------------------------------------


def test_the_default_hero_is_the_stamped_figure_and_the_rule_is_stated(
    report, study, tmp_path, capsys
):
    side = tmp_path / "went-in.json"
    assert run(report, [str(study), "--out", str(tmp_path / "r.pdf"),
                        "--json", str(side)]) == 0
    record = json.loads(side.read_text())
    assert record["hero"].endswith("vorticity.png")
    assert "stamped claims" in record["hero_rule"]
    out = capsys.readouterr().out
    assert "rule:" in out and "vorticity.png" in out


def test_hero_is_honoured_when_it_is_given(report, study, tmp_path):
    chosen = study / "cylinder" / "renders" / "mesh.png"
    side = tmp_path / "went-in.json"
    assert run(report, [str(study), "--out", str(tmp_path / "r.pdf"),
                        "--json", str(side), "--hero", str(chosen)]) == 0
    record = json.loads(side.read_text())
    assert Path(record["hero"]) == chosen
    assert "--hero" in record["hero_rule"]


def test_a_logo_of_an_odd_aspect_ratio_is_placed_without_throwing(
    report, study, tmp_path
):
    logo = _png(tmp_path / "wordmark.png", "ACME", size=(8.0, 0.6))
    out = tmp_path / "r.pdf"
    side = tmp_path / "went-in.json"
    assert run(report, [str(study), "--out", str(out), "--json", str(side),
                        "--logo", str(logo)]) == 0
    record = json.loads(side.read_text())
    assert record["logo"] == str(logo)
    assert not [note for note in record["notes"] if "logo not drawn" in note]
    assert out.stat().st_size > 20_000


def test_a_missing_template_is_refused_by_name(report, study, tmp_path, capsys):
    out = tmp_path / "r.pdf"
    missing = tmp_path / "acme-template.json"
    assert run(report, [str(study), "--out", str(out), "--template", str(missing)]) == 2
    captured = capsys.readouterr()
    assert "refused" in captured.err
    assert str(missing) in captured.err
    assert not out.exists(), "a refusal must not leave half a document behind"


def test_the_shipped_template_drives_the_cover_and_the_section_order(
    report, study, tmp_path
):
    template = TOOLBOX / "report_template.json"
    assert template.is_file(), "the example template is part of what ships"
    loaded, notes = report.load_template(template)
    assert notes == []
    assert loaded["sections"] == ["summary", "figures", "methods", "gaps"]
    assert loaded["title"] and loaded["front_matter"]

    custom = tmp_path / "short.json"
    custom.write_text(json.dumps({"title": "Acme flow report",
                                  "sections": ["summary", "methods"]}))
    side = tmp_path / "went-in.json"
    assert run(report, [str(study), "--out", str(tmp_path / "r.pdf"),
                        "--json", str(side), "--template", str(custom)]) == 0
    record = json.loads(side.read_text())
    assert record["sections"] == ["summary", "methods"]
    assert "figures" not in record["page_sections"]
    assert record["template"] == str(custom)


def test_an_unknown_section_is_reported_rather_than_silently_dropped(report, tmp_path):
    custom = tmp_path / "odd.json"
    custom.write_text(json.dumps({"sections": ["summary", "appendix"]}))
    loaded, notes = report.load_template(custom)
    assert loaded["sections"] == ["summary"]
    assert any("appendix" in note for note in notes)


# -- the manifest ---------------------------------------------------------------------


def test_the_document_gets_a_manifest_row(report, study, tmp_path, study_state):
    out = tmp_path / "pack.pdf"
    assert run(report, [str(study), "--out", str(out)]) == 0
    rows = study_state.artifacts(root=study, kind="report", exists=True)
    assert rows, "the document is not in the manifest"
    assert Path(rows[-1]["abspath"]).resolve() == out.resolve()
    assert (rows[-1].get("meta") or {}).get("source") == report.SOURCE_TAG
    assert "page" in rows[-1]["label"]


def test_a_second_run_does_not_report_on_the_first_report(report, study, tmp_path):
    first = tmp_path / "one.pdf"
    second = tmp_path / "two.pdf"
    side = tmp_path / "went-in.json"
    assert run(report, [str(study), "--out", str(first)]) == 0
    assert run(report, [str(study), "--out", str(second), "--json", str(side)]) == 0
    figures = json.loads(side.read_text())["figures"]
    assert not [row for row in figures if row["rel"].endswith("one.pdf")]


def test_an_empty_study_still_writes_a_document_and_exits_zero(report, tmp_path):
    """Exit 0 whatever it finds. A study with a `.reynolds/` and nothing in it is a
    stage of a study, not an error, and the document says so on its own pages."""
    bare = tmp_path / "bare"
    (bare / ".reynolds").mkdir(parents=True)
    out = tmp_path / "bare.pdf"
    side = tmp_path / "bare.json"
    assert run(report, [str(bare), "--out", str(out), "--json", str(side)]) == 0
    record = json.loads(side.read_text())
    assert out.is_file() and record["pages"] >= 4
    assert record["hero"] == ""
    assert "no picture" in record["hero_rule"]
    assert record["numbers"] == []


def test_a_relative_out_is_relative_to_where_it_was_run(report, study, tmp_path, monkeypatch):
    """`ENVIRONMENT.md` states it for every script here: a relative path is relative to
    the invocation directory, never to the case."""
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert run(report, [str(study), "--out", "here.pdf"]) == 0
    assert (elsewhere / "here.pdf").is_file()
    assert not (study / "here.pdf").exists()
