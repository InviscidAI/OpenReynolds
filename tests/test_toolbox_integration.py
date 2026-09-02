"""Where the eight toolbox scripts have to agree with each other.

Each script has its own test file and each of those passes; these are the seams
between them, which no single-script test can see. Two kinds of seam:

- `study_run.py` shells out to the others by path. Its own tests assert which
  script it picked and never that the flags it sends are flags that script
  accepts, so a wrong flag name is green in the unit tests and exit 2 in the
  container. Here the argv `study_run` builds is fed to the real parser of the
  script it names.
- the manifest is shared. A `kind` that one script writes is evidence that
  another script reads, so "who may write this kind" is part of the contract
  even though nothing in `study_state` enforces it.

Tests that assert a defect is present are marked xfail(strict=True): they turn
red the moment the defect is fixed, which is when the assertion should be
rewritten to assert the fix instead.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"integration_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def study_run():
    return load("study_run")


@pytest.fixture
def state():
    return load("study_state")


def make_case(tmp_path: Path) -> Path:
    """A study home with one case under it, the layout `find_root` documents."""
    root = tmp_path / "study"
    (root / ".reynolds").mkdir(parents=True)
    case = root / "case"
    (case / "system").mkdir(parents=True)
    (case / "constant" / "polyMesh").mkdir(parents=True)
    (case / "system" / "blockMeshDict").write_text("// mesh\n", encoding="utf-8")
    (case / "constant" / "polyMesh" / "owner").write_text("// owner\n", encoding="utf-8")
    return case


def parses(module, argv: list[str]) -> tuple[bool, str]:
    """Does this script's own argument parser accept this argv?

    `main` is called with the parse guarded, so nothing downstream of parsing
    runs: a SystemExit with code 2 is argparse rejecting the arguments, and any
    other exception means the arguments were accepted and the work began.
    """
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            module.main(argv)
    except SystemExit as exit_:
        if exit_.code == 2 and "error:" in err.getvalue():
            return False, err.getvalue().strip().splitlines()[-1]
        return True, ""
    except BaseException:
        return True, ""
    return True, ""


def phase_argv(study_run, case: Path, phase: str) -> list[str]:
    ctx = study_run.context(case, toolbox=TOOLBOX)
    commands, note = study_run.spec_for(phase).build(ctx)
    assert commands, f"{phase} built no command: {note}"
    return commands[0].argv


# -- the flags study_run sends -----------------------------------------------------


def test_the_preview_phase_sends_first_look_flags_it_accepts(study_run, tmp_path):
    argv = phase_argv(study_run, make_case(tmp_path), "preview")
    assert Path(argv[1]).name == "first_look.py"
    ok, why = parses(load("first_look"), argv[2:])
    assert ok, why


def test_the_probe_phase_sends_preflight_flags_it_accepts(study_run, tmp_path):
    argv = phase_argv(study_run, make_case(tmp_path), "probe")
    assert Path(argv[1]).name == "preflight.py"
    ok, why = parses(load("preflight"), argv[2:])
    assert ok, why


def test_the_render_phase_sends_results_flags_it_accepts(study_run, tmp_path):
    argv = phase_argv(study_run, make_case(tmp_path), "render")
    assert Path(argv[1]).name == "results.py"
    ok, why = parses(load("results"), argv[2:])
    assert ok, why


def test_the_animate_phase_sends_animate_flags_it_accepts(study_run, tmp_path):
    case = make_case(tmp_path)
    for time in ("0", "0.1", "0.2"):
        (case / time).mkdir()
    argv = phase_argv(study_run, case, "animate")
    assert Path(argv[1]).name == "animate.py"
    ok, why = parses(load("animate"), argv[2:])
    assert ok, why


def test_the_report_phase_finds_the_script_that_owns_it(study_run, tmp_path):
    ctx = study_run.context(make_case(tmp_path), toolbox=TOOLBOX)
    commands, note = study_run.report_build(ctx)
    assert commands, note


# -- where the state directory lands ------------------------------------------------


def test_case_gen_puts_the_state_directory_at_the_study_home(tmp_path):
    case_gen = load("case_gen")
    study = tmp_path / "study"
    study.mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        case_gen.main(["circle", str(study / "cyl"), "--reynolds", "200"])
    assert (study / ".reynolds").is_dir()
    assert not (study / "cyl" / ".reynolds").exists()


def test_two_cases_in_one_study_share_one_manifest(tmp_path):
    """What the split state directory used to cost: `gallery.py <study>` came back
    empty because each case had registered itself into its own `.reynolds`."""
    case_gen = load("case_gen")
    gallery = load("gallery")
    study = tmp_path / "study"
    study.mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        case_gen.main(["circle", str(study / "re100"), "--reynolds", "100"])
        case_gen.main(["circle", str(study / "re200"), "--reynolds", "200"])

    entries = gallery.collect(root=study)

    assert len(entries) == 2
    assert {entry.case for entry in entries} == {"re100", "re200"}
    assert not (study / "re100" / ".reynolds").exists()
    assert not (study / "re200" / ".reynolds").exists()


# -- who is allowed to write which kind ---------------------------------------------


def test_a_gallery_sheet_is_not_evidence_that_the_preview_phase_happened(study_run, state, tmp_path):
    case = make_case(tmp_path)
    root = case.parent
    sheet = root / "contact_sheet.png"
    sheet.write_bytes(b"\x89PNG\r\n\x1a\n")
    state.record("contact-sheet", sheet, root=root, label="latest of each kind", source="gallery")
    evident, why = study_run.preview_evidence(study_run.context(case, toolbox=TOOLBOX))
    assert not evident, why


def test_a_gallery_page_is_what_the_report_phase_produces(study_run, state, tmp_path):
    """This was filed as a defect and resolved the other way round.

    Nothing owned the report phase -- `report_build` looked for a `report.py` that
    does not exist -- so a `gallery` row marking the phase done was a row marking
    something nobody had done. The phase now runs `gallery.py`, which writes the page
    and the contact sheet that hand a finished study over. The page is therefore
    exactly the evidence that the report happened, and the confusion was never the
    kind: it was a phase with no producer.
    """
    case = make_case(tmp_path)
    root = case.parent
    page = root / "gallery.html"
    page.write_text("<html></html>", encoding="utf-8")
    state.record("gallery", page, root=root, label="gallery of 1 artifact(s)", source="gallery")

    evident, why = study_run.report_evidence(study_run.context(case, toolbox=TOOLBOX))

    assert evident and "gallery" in why
    # And the phase builds by running gallery.py over the study, not the case.
    commands, note = study_run.report_build(study_run.context(case, toolbox=TOOLBOX))
    assert not note and commands
    assert commands[0].argv[-2].endswith("gallery.py") or "gallery.py" in " ".join(commands[0].argv)


def test_a_results_summary_is_not_evidence_that_the_report_phase_happened(study_run, state, tmp_path):
    case = make_case(tmp_path)
    root = case.parent
    summary = case / "results" / "results.md"
    summary.parent.mkdir(parents=True)
    summary.write_text("# results\n", encoding="utf-8")
    state.record("report", summary, root=root, case=case.name, label="results summary")
    evident, why = study_run.report_evidence(study_run.context(case, toolbox=TOOLBOX))
    assert not evident, why


# -- kinds and phase names every script uses ----------------------------------------


def test_every_phase_study_run_owns_is_a_phase_study_state_knows(study_run, state):
    assert list(study_run.PHASE_NAMES) == list(state.PHASES)


def test_the_kinds_the_scripts_register_are_kinds_study_state_lists(state):
    first_look = load("first_look")
    results = load("results")
    used = set(first_look.PANEL_KINDS.values()) | {"contact-sheet", "animation", "gallery", "report"}
    for outputs in results.PRESETS.values():
        used |= {output.kind for output in outputs}
    unknown = sorted(used - set(state.KINDS))
    assert not unknown, f"registered under kinds study_state does not list: {unknown}"


def test_only_pictures_of_the_mesh_are_filed_under_mesh_full():
    results = load("results")
    misfiled = [output.name for output in results.PRESETS["mesh-validation"]
                if output.kind == "mesh-full" and output.producer != "mesh-cut"]
    assert not misfiled, misfiled
