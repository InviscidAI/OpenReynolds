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
import shutil
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
        # Labelled with where it came from, which is what this row knows. It used to
        # read "no figure is stamped with this number" on every row of every real
        # study, which spent the line repeating one fact the cover states once.
        assert row["why"] == f"read from {row['source']}", row["why"]
        assert row["source"], "an unstamped number still knows which file it is from"
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


# -- #139: a field is printed by its shape, never by its payload ----------------------

FIELD_U_NONUNIFORM_HEAD = """\
FoamFile { version 2.0; format ascii; class volVectorField; object U; }

dimensions      [0 1 -1 0 0 0 0];
internalField   nonuniform List<vector>
139286
(
"""

FIELD_U_NONUNIFORM_TAIL = """\
)
;

boundaryField
{
    inlet   { type fixedValue; value uniform (1 0 0); }
    outlet  { type zeroGradient; }
    walls   { type noSlip; }
}
"""


def _mapped_field(path: Path, entries: int = 139286) -> Path:
    """A `0/U` as a mapped, restarted or `setFields`-ed start actually writes one.

    The real file in `studies/20260906-091113-f9cf/.../cylRe100/0/U` is 3.8 MB of
    `(1.03809 -0.974742 0)` on one logical line, so the fixture is the real shape at
    the real size rather than a token of it: the defect was that nothing between the
    file and the page had a width, and a small fixture is what hid it.
    """
    body = "\n".join(f"({1.03809 + i * 1e-6:.6g} -0.974742 0)" for i in range(entries))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FIELD_U_NONUNIFORM_HEAD + body + "\n" + FIELD_U_NONUNIFORM_TAIL)
    return path


def test_a_nonuniform_initial_field_is_summarised_by_shape_not_printed(report, study):
    """#139: `report.py` wrote a 1308-page PDF in 11 minutes on
    `studies/20260906-091113-f9cf/.../cylRe100`, because `methods_lines` returned 117
    lines of which three were 3.8 MB, 1.5 MB and 1.1 MB -- a `0/U` holding
    `internalField nonuniform List<vector> 139286(...)` printed verbatim and folded at
    100 characters a page-line, 1280 pages of it. 15 real cases in the corpus have a
    `0/` file over 5,000 characters. What belongs on the page is the shape.
    """
    field = _mapped_field(study / "cylinder" / "0" / "U")
    assert field.stat().st_size > 3_000_000, "the fixture is the real size on purpose"

    shape = report.field_shape(field.read_text())
    assert "volVectorField" in shape
    assert "[0 1 -1 0 0 0 0]" in shape
    assert "139,286 entries" in shape
    assert "summarised, not printed" in shape
    assert "1.03809" not in shape, "the payload is the case, not the methods section"

    case, rule, others = report.choose_case(study, "")
    lines, _ = report.methods_lines(case, rule, others, layers=False)
    assert max(len(line) for line in lines) <= report.MAX_VALUE_CHARS, \
        "no single value may be wider than the cap"
    assert any("139,286 entries" in line for line in lines)
    # And the boundary conditions, which are the point of reading the file, survive.
    text = "\n".join(lines)
    assert "noSlip" in text and "zeroGradient" in text


def test_the_methods_section_page_count_is_capped_and_says_so(report, tmp_path):
    """#139 again, from the other end: 1,280 methods pages took 11 minutes to draw, so
    a cap on any one value is not enough on its own -- a `0/` with forty fields in it
    would still get there. The section is bounded and the cut is stated."""
    from matplotlib.backends.backend_pdf import PdfPages

    out = tmp_path / "capped.pdf"
    notes: list[str] = []
    with PdfPages(str(out)) as pdf:
        pages = report.text_pages(pdf, "methods", [f"line {i}" for i in range(5000)],
                                  max_pages=report.METHODS_MAX_PAGES, notes=notes)
    assert pages == report.METHODS_MAX_PAGES
    assert _page_count(out) == report.METHODS_MAX_PAGES
    assert notes and "not printed" in notes[0] and "capped" in notes[0]


# -- #145: the three places a person points this at ----------------------------------


FORCE_COEFFS_WITH_A_ZERO = """\
# Force coefficients
# magUInf     : 1
# lRef        : 1
# Aref        : 0.1
# rhoInf      : 1
# Time          Cd              Cl              Cs
1               1.2500          0.0100          3.2709e-20
2               1.2400          0.0110          3.2711e-20
3               1.2300          0.0105          3.2708e-20
4               1.2350          0.0108          3.2709e-20
"""


@pytest.fixture
def mirrored(tmp_path):
    """A study laid out the way a mirrored one is on disk: `studies/<id>/` with
    `files/<id>/<case>` under it, the `.reynolds/` in the case, the manifest's paths
    relative to `files/<id>`, the pictures in `files/<id>/renders/`, and the case
    decomposed over four ranks.

    Written by hand rather than through `study_state.record`, because the layout is
    the defect: `store.py` nests the mirror as `files/<study-id>/<case>` and the
    manifest was written when the workspace study home was `files/<id>`.
    """
    study_id = "20260829-100024-2825"
    top = tmp_path / "studies" / study_id
    home = top / "files" / study_id
    case = home / "case"
    (case / ".reynolds").mkdir(parents=True)
    for name, text in (
        ("system/controlDict", CONTROL_DICT),
        ("system/fvSchemes", FV_SCHEMES),
        ("system/fvSolution", FV_SOLUTION),
        ("constant/transportProperties", TRANSPORT),
        ("constant/turbulenceProperties", TURBULENCE),
        ("0/U", FIELD_U),
        ("log.checkMesh", CHECK_MESH),
        ("log.simpleFoam", SOLVER_LOG),
        ("postProcessing/forceCoeffs1/0/coefficient.dat", FORCE_COEFFS_WITH_A_ZERO),
    ):
        target = case / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    for rank in range(4):
        (case / f"processor{rank}" / "constant" / "polyMesh").mkdir(parents=True)

    for name in ("mesh_z.png", "U_y_800.png", "residuals.png"):
        _png(home / "renders" / name, name)
    rows = [
        {"kind": "mesh-full", "path": "renders/mesh_z.png", "case": "case",
         "label": "mesh: the whole domain", "at": "2026-08-29T02:26:02Z"},
        {"kind": "velocity", "path": "renders/U_y_800.png", "case": "case",
         "label": "|U| on y, iter 800", "at": "2026-08-29T02:26:03Z"},
        {"kind": "residuals", "path": "renders/residuals.png", "case": "case",
         "label": "residuals", "at": "2026-08-29T02:26:04Z"},
    ]
    (case / ".reynolds" / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    return top, home, case


def _same(record: dict) -> tuple:
    """What two invocations have to agree on: the numbers and the figures."""
    return (
        tuple(sorted((row["label"], row["text"]) for row in record["numbers"])),
        tuple(sorted((row["rel"], row["exists"]) for row in record["figures"])),
    )


def test_every_entry_path_into_a_mirrored_study_gives_the_same_numbers_and_figures(
    report, mirrored, tmp_path
):
    """#145: on the real `20260829-100024-2825`, no entry path gave both. The study top
    gave `key numbers: none`, `no case directory`, four blank pages and a second
    `.reynolds/manifest.jsonl` at the wrong level; `files/<id>` gave the numbers and
    "No artifact is registered in this study's manifest" because the manifest is one
    directory down; `files/<id>/case` gave the numbers and all six figures MISSING
    while 30 PNGs sat one level up.
    """
    top, home, case = mirrored
    records = {}
    for label, target in (("top", top), ("home", home), ("case", case)):
        side = tmp_path / f"{label}.json"
        assert run(report, [str(target), "--out", str(tmp_path / f"{label}.pdf"),
                            "--json", str(side)]) == 0
        records[label] = json.loads(side.read_text())

    for label, record in records.items():
        assert record["numbers"], f"{label}: no key numbers"
        assert [row["label"] for row in record["numbers"] if row["label"].startswith("Cd")], \
            f"{label}: the coefficients were not read"
        assert len(record["figures"]) == 3, f"{label}: {record['figures']}"
        assert record["missing"] == [], f"{label}: {record['missing']}"
        assert record["case"].endswith("case"), f"{label}: {record['case']}"
        assert Path(record["state_root"]) == case
        assert Path(record["artifact_base"]) == home

    assert _same(records["top"]) == _same(records["home"]) == _same(records["case"])


def test_no_manifest_is_written_into_a_directory_that_is_not_a_study(
    report, mirrored, tmp_path, study_state
):
    """#145: pointed at the study top, this wrote a second `.reynolds/manifest.jsonl`
    there and split the study's state from the real one in `files/<id>/case/.reynolds/`.
    A document is still written wherever it is pointed; a study is not invented to
    register it in."""
    top, _home, case = mirrored
    assert run(report, [str(top), "--out", str(tmp_path / "r.pdf")]) == 0
    found = sorted(path for path in top.rglob(study_state.STATE_DIR) if path.is_dir())
    assert found == [case / study_state.STATE_DIR], found

    nowhere = tmp_path / "not-a-study"
    nowhere.mkdir()
    out = tmp_path / "nowhere.pdf"
    side = tmp_path / "nowhere.json"
    assert run(report, [str(nowhere), "--out", str(out), "--json", str(side)]) == 0
    assert out.is_file(), "exit 0 whatever it finds, and the document is the deliverable"
    assert not (nowhere / study_state.STATE_DIR).exists()
    assert json.loads(side.read_text())["registered_as"] == ""


def test_a_decomposed_case_counts_as_one_case(report, mirrored):
    """#145: `is_case_dir` says yes to anything with a `constant/polyMesh`, so
    `processor0..processor3` each counted, `discover_cases` returned 5 and
    `choose_case` double-counted the study on top of that -- "the first of 6 cases
    (case)" on this study, and "the first of 2 cases ()" on single-case
    `20260906-091113-f9cf`. `--case processor2` would have been honoured and would have
    described one rank's slice as the study's methods."""
    _top, _home, case = mirrored
    chosen, rule, others = report.choose_case(case, "")
    assert chosen == case
    assert "only case" in rule, rule
    assert others == [], others
    assert "4 processor*" in rule and "the same case" in rule, rule
    assert "the first of" not in rule


def test_a_negligible_coefficient_is_not_a_headline(report, mirrored, tmp_path):
    """`Cs (last): 3.2709e-20` was printed as a cover key number to five figures. The
    side force on a symmetric body is zero and that is the solver's round-off; quoting
    round-off reads as a measurement of something."""
    top, _home, _case = mirrored
    side = tmp_path / "went-in.json"
    assert run(report, [str(top), "--out", str(tmp_path / "r.pdf"), "--json", str(side)]) == 0
    numbers = json.loads(side.read_text())["numbers"]
    zeroed = [row for row in numbers if row["label"].startswith("Cs")]
    assert zeroed, [row["label"] for row in numbers]
    for row in zeroed:
        assert row.get("negligible") is True, row
        assert row["text"].startswith("0 --") and "negligible" in row["text"], row["text"]
        assert "3.2709e-20" in row["text"], "the file's own value is still printed"
    # Cd is a headline and Cs is not, so Cd comes first on the page.
    labels = [row["label"] for row in numbers]
    assert min(i for i, name in enumerate(labels) if name.startswith("Cd")) \
        < min(i for i, name in enumerate(labels) if name.startswith("Cs"))
    # And a final residual of 1e-4 is small but is the answer, not round-off.
    residuals = [row for row in numbers if row["label"].startswith("final residual")]
    assert residuals and not any(row.get("negligible") for row in residuals)


def test_a_steady_runs_time_column_is_called_iterations(report, mirrored, tmp_path):
    """`force record: 500 rows, t = 1 to 500` was printed for a steady `simpleFoam`
    run whose `Time` column is an iteration count, not seconds. The predicate is
    `openreynolds/convergence.py::is_steady_solver`; `spectra.py` was being fixed for
    the same confusion in the same sweep."""
    top, _home, case = mirrored
    side = tmp_path / "went-in.json"
    assert run(report, [str(top), "--out", str(tmp_path / "r.pdf"), "--json", str(side)]) == 0
    record = json.loads(side.read_text())
    rows = [row for row in record["numbers"] if row["label"] == "force record"]
    assert rows, [row["label"] for row in record["numbers"]]
    assert rows[0]["text"] == "4 rows, iterations 1 to 4", rows[0]["text"]
    assert "t = " not in rows[0]["text"]
    assert any("iterations, not seconds" in note for note in record["notes"]), record["notes"]

    lines, _ = report.methods_lines(case, "the study's only case (case)", [], layers=False)
    assert any("last iteration" in line for line in lines), \
        [line for line in lines if "parsed" in line]
    assert report.is_steady_solver("simpleFoam") and not report.is_steady_solver("pimpleFoam")


def test_where_nothing_is_stamped_the_cover_says_so_rather_than_counting_zero(
    report, mirrored, tmp_path
):
    """Re-measured for this change: 973 PNGs under `studies/` and `data/qa-runs/`,
    zero carrying a `reynolds-claims` stamp, because nothing in the pipeline writes
    one -- `claims.attach` is the only call that stamps and no renderer calls it.

    So `verified` is unreachable on a real study, and the line about it used to read
    `0 of 22`. A count of zero reads as a check that ran and found nothing, when what
    is missing is the measurement itself: there was never anything to check against.
    It says that in words, and offers no tally until there is something to tally."""
    top, _home, _case = mirrored
    side = tmp_path / "went-in.json"
    assert run(report, [str(top), "--out", str(tmp_path / "r.pdf"), "--json", str(side)]) == 0
    record = json.loads(side.read_text())
    assert record["numbers"], "there are numbers on this cover"
    assert not [row for row in record["numbers"] if row["state"] == "verified"]
    backing = record["stamp_backing"]
    assert "0 of" not in backing, f"a tally of zero is what this removed: {backing}"
    assert "reynolds-claims" in backing
    assert "nothing for these numbers to be checked against" in backing, backing
    assert backing in report.render_summary(record)


def test_a_stamp_backed_number_is_counted_as_one(report, study, tmp_path):
    """The other side of the same sentence: the study fixture does have a stamped
    figure, and the count has to be the number of numbers a stamp stands behind rather
    than a constant."""
    side = tmp_path / "went-in.json"
    assert run(report, [str(study), "--out", str(tmp_path / "r.pdf"),
                        "--json", str(side)]) == 0
    record = json.loads(side.read_text())
    verified = [row for row in record["numbers"] if row["state"] == "verified"]
    assert verified
    assert record["stamp_backing"].startswith(f"{len(verified)} of {len(record['numbers'])}")


# -- #179: a mirrored study with no .reynolds/ at all --------------------------------


@pytest.fixture
def unregistered(mirrored):
    """The `mirrored` layout with the state directory taken away: `studies/<id>/` with
    `files/<id>/<case>` under it and no `.reynolds/` anywhere.

    11 of 16 real studies are this shape. #145's fixture has one and its path holds;
    this is the case where there is none, so `locate` had nothing to descend by.
    """
    top, home, case = mirrored
    shutil.rmtree(case / ".reynolds")
    return top, home, case


def test_a_study_with_no_state_directory_still_finds_its_case_from_the_top(
    report, unregistered, tmp_path
):
    """#179. `locate` descended only as far as a `.reynolds/`, so with none anywhere
    `state_root` stayed at the study top -- and `discover_cases` looks one level down
    while a mirrored study keeps its cases two, at `files/<id>/<case>`. So
    `choose_case`, `measured_numbers` and `gaps` were each handed a directory with
    nothing under it, and the documented entry point wrote a complete-looking document
    saying `key numbers: none` and `no case directory under the study home`.

    Measured over 16 real studies: 11 gave 4 pages and no numbers at the top where
    `files/<id>` gave 5 to 8 pages and 3 to 22 numbers. Every entry path now gives the
    same document, which is what #145 established for the studies that do have state.
    """
    top, home, case = unregistered
    records = {}
    for label, target in (("top", top), ("home", home), ("case", case)):
        side = tmp_path / f"{label}.json"
        assert run(report, [str(target), "--out", str(tmp_path / f"{label}.pdf"),
                            "--json", str(side)]) == 0
        records[label] = json.loads(side.read_text())

    for label, record in records.items():
        assert record["is_study"] is False, f"{label}: there is no state directory"
        assert record["numbers"], f"{label}: key numbers: none"
        assert [row for row in record["numbers"] if row["label"].startswith("Cd")], \
            f"{label}: the coefficients were not read"
        assert record["case"].endswith("case"), f"{label}: {record['case']}"
        assert record["pages"] > 4, f"{label}: {record['pages']} pages"
    assert _same(records["top"]) == _same(records["home"]) == _same(records["case"])

    # And still nothing written: `is_study` stays false, so the guard #145 added holds.
    assert not list(top.rglob(".reynolds"))


def test_a_study_whose_cases_are_at_the_top_is_not_redirected(report, study, tmp_path):
    """The fallback fires only where there is no case at this level, so the ordinary
    layout -- `<study>/<case>`, which `discover_cases` already finds -- is untouched.
    `wake/cylinder` has a `.reynolds/` as well, which is the other half of the guard.
    """
    where = report.locate(study)
    assert where["state_root"] == study
    assert where["is_study"] is True


# -- #185: a binary field's boundaryField is not a patch list -------------------------


def _binary_field(path: Path, patches: int = 6) -> None:
    """A `format binary` field whose patch values are raw doubles, as OpenFOAM writes
    them.

    The bytes are the point. A real binary payload has no reason for its braces to
    balance, and an unmatched `}` inside one closes the patch entry the text parser
    thinks it is inside: what follows, up to the next `{` in the bytes, is then read as
    a further patch name. `0/thickness` of `20260830-050338-cfd5` is the real one -- 6
    patches in `constant/polyMesh/boundary` and 8 reported, the last two named
    `Yu?J...NQu?` and the like. The last patch here carries such a payload, which
    is that shape: every real patch still parses and one phantom follows them.
    """
    import struct

    clean = b"".join(struct.pack("<d", v / 7.0) for v in range(1, 6))
    # A stray close-brace and a stray open-brace in the bytes, which is all it
    # takes: 0x7d is a plausible byte of a double and OpenFOAM writes doubles.
    unbalanced = (struct.pack("<d", 1.0)
                  + bytes([0x59, 0x75, 0x3f, 0x4a, 0x7d, 0xd5, 0x4e, 0x51])
                  + bytes([0x75, 0x3f, 0x8f, 0x38, 0x72, 0x7b, 0x00, 0x75])
                  + bytes([0x3f, 0xd5, 0x7d]) + struct.pack("<d", 2.5))
    head = (b"FoamFile { version 2.0; format binary; class volScalarField;"
            b" object thickness; }\n\n"
            b"dimensions      [0 0 0 0 0 0 0];\n"
            b"internalField   uniform 0;\n\n"
            b"boundaryField\n{\n")
    body = b""
    for i in range(patches):
        blob = unbalanced if i == patches - 1 else clean
        body += (f"    patch{i}\n".encode()
                 + b"    {\n        type            fixedValue;\n"
                 b"        value           nonuniform List<scalar> 5(" + blob
                 + b");\n    }\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(head + body + b"}\n")


def test_a_binary_fields_boundary_field_is_counted_as_unread_not_printed_as_patches(
    report, study
):
    """#185. `field_shape` beside it was hardened for #139 and this was not: raw
    IEEE-754 bytes reached page 5 of a real study's PDF as two further boundary
    patches, named `Yu?J...NQu?` and the like, with 11 control characters in them and
    matplotlib warning "Glyph 6 missing from font". A patch name is an OpenFOAM word;
    a slice of a double is not."""
    _binary_field(study / "cylinder" / "0" / "thickness")
    patches, unread = report.boundary_types(study / "cylinder", "thickness")
    for name, kind in patches:
        assert report._WORD.match(name), name
        assert all(ch.isprintable() for ch in name + kind), (name, kind)
    assert [name for name, _ in patches] == [f"patch{i}" for i in range(6)], patches
    assert unread, "whatever would not read is said, not dropped in silence"
    assert "binary" in unread and "polyMesh/boundary" in unread, unread


def test_no_control_character_from_a_binary_field_reaches_the_methods_page(report, study):
    """The same thing through the documented entry point: the lines the page is drawn
    from carry no byte matplotlib has no glyph for."""
    _binary_field(study / "cylinder" / "0" / "thickness")
    case, rule, others = report.choose_case(study, "")
    lines, _ = report.methods_lines(case, rule, others, layers=False)
    text = "\n".join(lines)
    assert "thickness" in text
    bad = [ch for ch in text if not ch.isprintable() and ch != "\n"]
    assert bad == [], f"{len(bad)} unprintable characters on the methods page"
    # The payload's own bytes, which are what reached page 5 of the real study: the
    # undecodable ones as U+FFFD (matplotlib: "Glyph 6 missing from font") and the
    # decodable ones as themselves.
    assert "�" not in text and "NQu" not in text, text
    assert "(not read)" in text


# -- #186: a name that was typed and is wrong is refused, not written around ----------


def test_a_case_name_that_names_nothing_is_refused_rather_than_documented(
    report, study, tmp_path, capsys
):
    """#186. `--case doesnotexist` wrote a complete 4-page 41 KB document reading `key
    numbers: none`, with the reason only on the page, and exited 0 -- while
    `--template nothere.json` on the same study exits 2 and writes nothing. A typo in
    a path that produces a success nobody checks is worse than a crash."""
    out = tmp_path / "typo.pdf"
    assert run(report, [str(study), "--case", "doesnotexist", "--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert "refused" in err and "doesnotexist" in err
    assert "cylinder" in err, "the refusal names the case the study does have"
    assert not out.exists(), "a refusal must not leave a document behind"
    # And the case that is there still works, so this is a refusal and not a block.
    assert run(report, [str(study), "--case", "cylinder", "--out", str(out)]) == 0


def test_one_rank_of_a_decomposed_case_is_refused_by_name(report, mirrored, tmp_path):
    """#145 saw this would be honoured and left it as a rule printed on the page, which
    is the #186 shape: a methods section about a quarter of a mesh, exit 0."""
    _top, _home, case = mirrored
    out = tmp_path / "rank.pdf"
    assert run(report, [str(case), "--case", "processor2", "--out", str(out)]) == 2
    assert not out.exists()
    with pytest.raises(report.Refused, match="one rank"):
        report.choose_case(case, "processor2")


def test_a_directory_that_is_not_there_is_refused_by_name_not_by_traceback(
    report, tmp_path, capsys
):
    """It came out of `choose_case`'s `study.iterdir()` as a FileNotFoundError and exit
    1. A path that is not there is an argument mistake, like a missing --template."""
    missing = tmp_path / "no-such-study"
    assert run(report, [str(missing), "--out", str(tmp_path / "x.pdf")]) == 2
    err = capsys.readouterr().err
    assert "refused" in err and str(missing) in err
    assert not (tmp_path / "x.pdf").exists()
