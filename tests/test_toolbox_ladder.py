"""`ladder.py`: the reduced cases whose answers do not come from a solver.

The script exists because of a free-surface hull that diverged after four things were
changed at once -- the internal field, the inlet ramp, the outlet type and the phase
fraction -- so that six mechanisms could be proposed for it and all six falsified, none
of them cheaply. The fault was present in a tank with no hull and no motion, a
two-minute run, and nobody made it.

Most of what is tested here is therefore not behaviour but a *rule*: every rung's
expected answer names a source outside CFD. That rule is the whole idea. A rung whose
answer could only come from another solve is a cheaper copy of the same unknown, and
one of those on the ladder makes the whole thing decorative -- so it is asserted
mechanically over every rung in every catalogue rather than trusted to review.

The rest is the free-will contract reaching into the toolbox: a ladder is structurally
a workflow, and the only form it may take here is an offered script. So the exit code
is 0 whatever the report says, and the docstring carries preflight's formula.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


@pytest.fixture(scope="module")
def ladder():
    spec = importlib.util.spec_from_file_location("toolbox_ladder", TOOLBOX / "ladder.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# -- synthetic cases ---------------------------------------------------------------


def skeleton(root: Path) -> Path:
    for relative in ("system", "0", "constant/polyMesh"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root


def marine_case(root: Path) -> Path:
    """A VOF hull free to move: alpha, gravity, sixDoFRigidBodyMotion."""
    case = skeleton(root / "hull")
    (case / "system" / "controlDict").write_text(
        "application     interFoam;\nendTime         20;\nmaxAlphaCo      1;\n"
    )
    (case / "0" / "alpha.water").write_text(
        "dimensions [0 0 0 0 0 0 0];\ninternalField uniform 0;\n"
        "boundaryField { inlet { type fixedValue; value uniform 0; } }\n"
    )
    (case / "0" / "U").write_text(
        "internalField uniform (0 0 0);\nboundaryField { inlet { type fixedValue; } }\n"
    )
    (case / "constant" / "g").write_text(
        "dimensions [0 1 -2 0 0 0 0];\nvalue (0 0 -9.81);\n"
    )
    (case / "constant" / "dynamicMeshDict").write_text(
        "motionSolver    sixDoFRigidBodyMotion;\n"
        "sixDoFRigidBodyMotionCoeffs { mass 20.0; }\n"
    )
    (case / "constant" / "momentumTransport").write_text(
        "simulationType  RAS;\nRAS { model kOmegaSST; turbulence on; }\n"
    )
    return case


AERO_BOUNDARY = """\
FoamFile { version 2.0; class polyBoundaryMesh; object boundary; }
4
(
    farfield  { type patch; nFaces 4000; startFace 0; }
    outlet    { type patch; nFaces 400;  startFace 4000; }
    wing      { type wall;  nFaces 9000; startFace 4400; }
    symmetry  { type symmetryPlane; nFaces 2000; startFace 13400; }
)
"""


def aero_case(root: Path) -> Path:
    """A compressible external case: a wing in a far field, no phase fraction."""
    case = skeleton(root / "m6")
    (case / "system" / "controlDict").write_text(
        "application     rhoSimpleFoam;\nendTime         3000;\n"
    )
    (case / "constant" / "polyMesh" / "boundary").write_text(AERO_BOUNDARY)
    (case / "constant" / "thermophysicalProperties").write_text(
        "thermoType { type hePsiThermo; mixture pureMixture; "
        "equationOfState perfectGas; }\n"
    )
    (case / "constant" / "momentumTransport").write_text(
        "simulationType  RAS;\nRAS { model kOmegaSST; }\n"
    )
    (case / "0" / "U").write_text("internalField uniform (250 0 0);\n")
    (case / "constant" / "triSurface").mkdir()
    (case / "constant" / "triSurface" / "wing.stl").write_text("solid wing\nendsolid wing\n")
    return case


def duct_case(root: Path) -> Path:
    case = skeleton(root / "duct")
    (case / "system" / "controlDict").write_text("application     simpleFoam;\n")
    (case / "constant" / "polyMesh" / "boundary").write_text(
        "FoamFile { version 2.0; }\n3\n(\n"
        "    inlet  { type patch; nFaces 100; startFace 0; }\n"
        "    outlet { type patch; nFaces 100; startFace 100; }\n"
        "    walls  { type wall;  nFaces 800; startFace 200; }\n"
        ")\n"
    )
    return case


def mystery_case(root: Path) -> Path:
    """A case that says almost nothing: no fields, no mesh, an unknown application."""
    case = skeleton(root / "mystery")
    (case / "system" / "controlDict").write_text("application     foamRun;\nendTime 1;\n")
    return case


# -- detection ---------------------------------------------------------------------


def test_a_vof_case_with_six_dof_motion_is_the_marine_ladder(ladder, tmp_path):
    detection, rungs = ladder.inspect(marine_case(tmp_path))
    assert detection.key == "free-surface-marine"
    assert not detection.generic
    assert len(rungs) == 6
    assert "alpha.water" in detection.reason, "the evidence has to name what it read"
    assert "free to move" in detection.reason
    assert rungs[0].name == "still tank", (
        "rung 1 is the two-minute run that would have caught the failure this exists for"
    )


def test_a_compressible_external_case_is_the_aero_ladder(ladder, tmp_path):
    detection, rungs = ladder.inspect(aero_case(tmp_path))
    assert detection.key == "external-aerodynamics"
    assert len(rungs) == 4
    assert "far-field" in detection.reason
    assert "rhoSimpleFoam" in detection.reason


def test_a_duct_is_the_internal_ladder(ladder, tmp_path):
    detection, rungs = ladder.inspect(duct_case(tmp_path))
    assert detection.key == "internal-flow"
    assert len(rungs) == 4


def test_a_multi_region_case_is_the_conjugate_ladder(ladder, tmp_path):
    case = skeleton(tmp_path / "cht")
    (case / "system" / "controlDict").write_text("application     chtMultiRegionFoam;\n")
    detection, rungs = ladder.inspect(case)
    assert detection.key == "conjugate-heat-transfer"
    assert len(rungs) == 4


def test_an_unrecognised_case_gets_the_generic_rungs_and_says_so(ladder, tmp_path):
    """The honest answer to "I do not know what this is" is two rungs and a label.

    Not four confident ones aimed at the wrong physics -- a ladder that claims a class
    it cannot see would be inventing the one input that makes the rest of it mean
    anything, which is the failure mode preflight's `--resolve` check exists to avoid.
    """
    case = mystery_case(tmp_path)
    detection, rungs = ladder.inspect(case)
    assert detection.key == "generic"
    assert detection.generic is True
    assert len(rungs) == 2

    report = ladder.render(detection, rungs, case)
    assert "not recognised" in report
    assert "not a ladder for this case in particular" in report
    assert "unrecognised" in report


def test_what_is_absent_is_reported_as_absent_rather_than_left_out(ladder, tmp_path):
    """A signal that was looked for and not found is a different fact from one nobody
    checked, and only the first one tells you a hydrostatic rung is unavailable."""
    detection, _rungs = ladder.inspect(mystery_case(tmp_path))
    names = {signal.name for signal in detection.signals}
    assert {"gravity", "phase fraction", "turbulence", "patches"} <= names
    gravity = [signal for signal in detection.signals if signal.name == "gravity"][0]
    assert gravity.detected is False
    assert "no constant/g" in gravity.value
    assert gravity.source == "constant/g", "the evidence names the file it did not find"


# -- the rule that keeps the ladder honest -----------------------------------------


def test_every_known_answer_comes_from_outside_a_solver(ladder):
    """The load-bearing assertion in this file.

    `known` states the answer and its source, and the source may never be another
    solve. Enforced over every rung in every catalogue because the rule fails silently:
    one rung answered from a previous run reads exactly like the others and quietly
    turns the ladder into a set of cheaper unknowns.
    """
    assert ladder.NOT_FROM_A_SOLVE, "the forbidden-sources list is the definition"
    for key, rungs in ladder.CATALOGUE.items():
        for number, rung in enumerate(rungs, start=1):
            assert rung.known.strip(), f"{key} rung {number} has no known answer"
            lowered = rung.known.lower()
            for word in ("simulation", "cfd", "another run"):
                assert word not in lowered, (
                    f"{key} rung {number} claims its answer from {word!r}; a rung whose "
                    "answer needs a solve is not a rung"
                )
            assert ladder.known_is_independent(rung)


def test_a_known_answered_from_a_previous_run_is_rejected(ladder):
    """The rule has to be able to say no, or it is not a rule."""
    bad = ladder.Rung(
        name="x", adds="y", check="z",
        known="the value from another run of the same case",
        tolerance="5%", cost="minutes",
    )
    assert not ladder.known_is_independent(bad)
    assert not ladder.known_is_independent(bad._replace(known=""))


def test_every_rung_adds_something_and_no_two_in_a_row_add_the_same_thing(ladder):
    """One rung, one new piece of physics: that is what makes a failure localising.

    Two consecutive rungs introducing the same thing means the second one cannot
    isolate anything, which is the four-changes-at-once failure written smaller.
    """
    for key, rungs in ladder.CATALOGUE.items():
        for number, rung in enumerate(rungs, start=1):
            assert rung.adds.strip(), f"{key} rung {number} adds nothing it names"
            assert rung.check.strip(), f"{key} rung {number} measures nothing"
            assert rung.tolerance.strip(), f"{key} rung {number} has no pass condition"
            assert rung.cost.strip(), f"{key} rung {number} does not say what it costs"
        for first, second in zip(rungs, rungs[1:]):
            assert first.adds != second.adds, (
                f"{key}: '{first.name}' and '{second.name}' introduce the same thing"
            )


def test_every_catalogue_ends_at_the_case_as_asked(ladder):
    """The last rung is the deliverable, and it is the only one without an outside
    answer. Naming that out loud is the point -- it is why the others exist."""
    for key, rungs in ladder.CATALOGUE.items():
        if key == "generic":
            continue
        assert rungs[-1].name == "the case as asked"
        assert "Nothing external" in rungs[-1].known


# -- the report --------------------------------------------------------------------


def test_the_report_names_the_class_the_evidence_and_every_rung(ladder, tmp_path):
    case = marine_case(tmp_path)
    detection, rungs = ladder.inspect(case)
    report = ladder.render(detection, rungs, case)

    assert "free-surface marine" in report
    assert "6 rungs offered" in report
    assert "system/controlDict" in report and "interFoam" in report
    for rung in rungs:
        assert rung.name in report
        assert rung.known in report
        assert rung.tolerance in report
        assert rung.cost in report
    assert "Kelvin" in report, "19.47 degrees is the check worth advertising"


def test_the_report_says_the_rungs_are_offered_and_that_skipping_is_allowed(ladder, tmp_path):
    case = marine_case(tmp_path)
    detection, rungs = ladder.inspect(case)
    report = ladder.render(detection, rungs, case)
    assert "offered, not owed" in report
    assert "Climbing none of them is a legitimate choice" in report
    assert "no exit code from this script means you may not proceed" in report


def test_one_rung_prints_in_full_with_the_edits_that_would_build_it(ladder, tmp_path):
    """Phase 1 applies no overrides, so printing them is the whole of `--rung`: it has
    to be enough to make the edits by hand."""
    case = marine_case(tmp_path)
    detection, rungs = ladder.inspect(case)
    text = ladder.render_rung(detection, rungs, 1, case)

    assert "still tank" in text
    assert "rung 1 of 6" in text
    assert "system/controlDict: endTime -> 5" in text
    assert "remove     constant/dynamicMeshDict" in text
    assert "Nothing above has been applied" in text


def test_a_rung_number_that_does_not_exist_is_answered_not_raised(ladder, tmp_path):
    case = marine_case(tmp_path)
    detection, rungs = ladder.inspect(case)
    text = ladder.render_rung(detection, rungs, 99, case)
    assert "there is no rung 99" in text
    assert "rungs 1 to 6" in text


def test_json_parses_and_has_the_documented_shape(ladder, tmp_path):
    case = aero_case(tmp_path)
    detection, rungs = ladder.inspect(case)
    payload = json.loads(ladder.as_json(detection, rungs, case))

    assert payload["class"] == "external-aerodynamics"
    assert payload["class_name"] == "external aerodynamics"
    assert payload["generic"] is False
    assert payload["offered"] is True
    assert payload["reason"]
    assert len(payload["rungs"]) == 4

    first = payload["rungs"][0]
    assert first["number"] == 1
    assert set(first) == {
        "number", "name", "adds", "check", "known", "tolerance", "cost", "overrides",
    }
    assert isinstance(first["overrides"], dict)

    evidence = payload["evidence"]
    assert evidence and set(evidence[0]) == {"signal", "source", "value", "detected"}
    assert any(row["signal"] == "patches" and row["detected"] for row in evidence)


# -- the contract ------------------------------------------------------------------


def test_the_docstring_keeps_it_advisory(ladder):
    """preflight's formula, verbatim. A ladder is structurally a workflow, and the only
    thing that keeps it inside the free-will contract is that it never becomes one."""
    doc = ladder.__doc__
    assert "edits nothing, refuses nothing, and blocks nothing" in doc
    assert "no exit code that means" in doc
    assert "legitimate choice" in doc
    assert "python3" in doc, "it has to show how to run it"


def test_the_exit_code_is_zero_whatever_the_report_says(ladder, tmp_path, capsys):
    """There is no verdict here that may stop anyone, so there is no code for one."""
    for case in (marine_case(tmp_path), mystery_case(tmp_path)):
        assert ladder.main([str(case)]) == 0
        assert ladder.main([str(case), "--json"]) == 0
        assert ladder.main([str(case), "--rung", "1"]) == 0
        assert ladder.main([str(case), "--rung", "99"]) == 0
    assert ladder.main([str(tmp_path / "does-not-exist")]) == 0
    assert ladder.main(["--list-classes"]) == 0
    capsys.readouterr()


def test_the_cli_prints_json_that_parses(ladder, tmp_path, capsys):
    ladder.main([str(marine_case(tmp_path)), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["class"] == "free-surface-marine"


def test_it_is_stdlib_plus_its_own_siblings(ladder):
    """No pyvista: this reads dictionaries and prints text, and a script that pulls in
    a rendering stack cannot run in the two seconds that make it worth running."""
    source = (TOOLBOX / "ladder.py").read_text(encoding="utf-8")
    assert "pyvista" not in source
    siblings = {script.stem for script in TOOLBOX.glob("*.py")}
    allowed = set(sys.stdlib_module_names) | siblings | {"__future__"}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [(node.module or "").split(".")[0]]
        else:
            continue
        for root in roots:
            assert root in allowed, f"ladder.py imports {root}"


def test_the_toolbox_index_carries_it(ladder):
    index = (TOOLBOX / "README.md").read_text(encoding="utf-8")
    assert "ladder.py" in index


def test_the_field_note_exists_and_states_the_rule(ladder):
    """The script is half of it; the note is where the idea is allowed to live as
    know-how rather than as a tool nobody reaches for."""
    notes = (TOOLBOX / "notes" / "openfoam-field-notes.md").read_text(encoding="utf-8")
    assert "ladder of reduced cases" in notes.lower()
    assert "ladder.py" in notes
    assert "--record" in notes, "the note carries the fact that outcomes persist"


# -- recording ---------------------------------------------------------------------
#
# The reason this half exists: round 1 of the ONERA study named the right solver and
# round 2, a fresh thread, did not have that knowledge, because it lived only in a
# transcript. Evidence written to the volume is the kind that survives.


def load_sibling(name: str):
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def state():
    return load_sibling("study_state")


def manifest_lines(case: Path) -> list[dict]:
    text = (case / ".reynolds" / "manifest.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_recording_writes_one_manifest_row_with_the_documented_shape(ladder, tmp_path, capsys):
    """One rung, one line: the rung and its class, what happened, the measured value,
    the known answer it was set against, and when -- everything a later session needs
    to not re-climb it."""
    case = marine_case(tmp_path)
    assert ladder.main([
        str(case), "--record", "1", "--status", "pass",
        "--value", "0.0007", "--note", "coarse tank, 5 s",
    ]) == 0
    capsys.readouterr()

    rows = manifest_lines(case)
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "rung"
    assert row["at"]
    meta = row["meta"]
    assert meta["class"] == "free-surface-marine"
    assert meta["rung"] == 1
    assert meta["name"] == "still tank"
    assert meta["status"] == "pass"
    assert meta["value"] == 0.0007, "a numeric --value is stored as a number"
    assert meta["known"].startswith("Hydrostatics"), (
        "the record carries the yardstick, not only the reading"
    )
    assert meta["note"] == "coarse tank, 5 s"


def test_re_recording_a_rung_updates_in_place_rather_than_duplicating(ladder, tmp_path, capsys):
    """A rung retried after a fix has one answer, not a history to reconcile."""
    case = marine_case(tmp_path)
    ladder.main([str(case), "--record", "1", "--status", "fail", "--value", "0.4"])
    ladder.main([str(case), "--record", "2", "--status", "pass", "--value", "3128"])
    ladder.main([str(case), "--record", "1", "--status", "pass", "--value", "0.0007"])
    capsys.readouterr()

    rows = manifest_lines(case)
    assert len(rows) == 2, "three records of two rungs are two lines"
    by_rung = {row["meta"]["rung"]: row["meta"] for row in rows}
    assert by_rung[1]["status"] == "pass"
    assert by_rung[1]["value"] == 0.0007
    assert by_rung[2]["status"] == "pass", "re-recording rung 1 left rung 2 alone"


def test_the_report_shows_the_evidence_and_absence_stays_absence(ladder, tmp_path, capsys):
    """Every rung gets an evidence line. A recorded one shows what happened and when;
    an unrecorded one shows "-", because absence of evidence is not a failure."""
    case = marine_case(tmp_path)
    ladder.main([str(case), "--record", "2", "--status", "pass", "--value", "3128"])
    capsys.readouterr()

    detection, rungs = ladder.inspect(case)
    evidence = ladder.recorded_evidence(case, detection.key)
    report = ladder.render(detection, rungs, case, evidence)

    assert report.count("evidence   ") == 6, "one evidence line per rung"
    assert "evidence   pass   value 3128   recorded " in report
    assert report.count("evidence   -") == 5, "the five unrecorded rungs show absence"
    assert "evidence   fail" not in report


def test_evidence_for_another_class_does_not_count(ladder, state, tmp_path, capsys):
    """A rung recorded against the aero ladder is not evidence for the marine one:
    the match is class+rung, and rung 1 of one ladder is not rung 1 of another."""
    case = marine_case(tmp_path)
    state.record_rung(
        1, "pass", root=case, case=case.name,
        class_key="external-aerodynamics", name="empty tunnel",
    )
    detection, rungs = ladder.inspect(case)
    report = ladder.render(detection, rungs, case, ladder.recorded_evidence(case, detection.key))
    assert report.count("evidence   -") == 6


def test_values_round_trip_through_json(ladder, tmp_path, capsys):
    case = marine_case(tmp_path)
    ladder.main([str(case), "--record", "2", "--status", "fail", "--value", "2612.5",
                 "--note", "20% light"])
    ladder.main([str(case), "--record", "5", "--status", "pass", "--value", "19.4 deg"])
    capsys.readouterr()

    ladder.main([str(case), "--json"])
    payload = json.loads(capsys.readouterr().out)

    recorded = payload["recorded"]
    assert [row["rung"] for row in recorded] == [2, 5]
    assert recorded[0]["status"] == "fail"
    assert recorded[0]["value"] == 2612.5, "a number comes back as a number"
    assert recorded[0]["note"] == "20% light"
    assert recorded[0]["known"].startswith("Archimedes")
    assert recorded[1]["value"] == "19.4 deg", "a reading that is not a number survives too"
    assert recorded[1]["at"]

    first = payload["rungs"][0]
    assert set(first) == {
        "number", "name", "adds", "check", "known", "tolerance", "cost", "overrides",
    }, "the rung table itself is unchanged; the evidence rides beside it"


def test_an_unwritable_manifest_is_reported_not_raised(ladder, tmp_path, capsys):
    """Exit 0 whatever the disk does: a record that cannot be written is a fact to
    report, and turning it into a crash would put a gate on an offered tool."""
    case = marine_case(tmp_path)
    (case / ".reynolds").write_text("a file where the state directory should be")

    assert ladder.main([str(case), "--record", "1", "--status", "pass"]) == 0
    out = capsys.readouterr().out
    assert "was not recorded" in out
    assert "the report still works" in out

    # And the report side reads that broken state as no evidence, not as an error.
    assert ladder.main([str(case)]) == 0
    assert capsys.readouterr().out.count("evidence   -") == 6


def test_recording_a_rung_that_does_not_exist_is_answered_not_raised(ladder, tmp_path, capsys):
    case = marine_case(tmp_path)
    assert ladder.main([str(case), "--record", "99", "--status", "pass"]) == 0
    assert "no rung 99" in capsys.readouterr().out
    assert not (case / ".reynolds").exists()


def test_recording_without_a_status_asks_for_one_and_still_exits_zero(ladder, tmp_path, capsys):
    case = marine_case(tmp_path)
    assert ladder.main([str(case), "--record", "1"]) == 0
    assert "--status" in capsys.readouterr().out
    assert not (case / ".reynolds").exists()


def test_the_progress_reader_surfaces_recorded_rungs_unbidden(ladder, tmp_path, capsys):
    """The promotion note's test: knowledge that arrives unbidden is the only kind
    that survives a context refresh. A fresh session's first call is
    `progress_report.py`, so the evidence has to be in that answer without anyone
    asking for it -- and not disguised as a preview."""
    case = marine_case(tmp_path)
    ladder.main([str(case), "--record", "1", "--status", "pass", "--value", "0.0007"])
    ladder.main([str(case), "--record", "2", "--status", "fail", "--value", "2612",
                 "--note", "20% light against rho g V"])
    capsys.readouterr()

    progress = load_sibling("progress_report")
    report = progress.collect(case)

    rows = report["ladder"]
    assert [row["rung"] for row in rows] == [1, 2]
    assert rows[0]["status"] == "pass" and rows[0]["value"] == 0.0007
    assert rows[1]["status"] == "fail" and rows[1]["note"] == "20% light against rho g V"
    assert all(row.get("kind") != "rung" for row in report["previews"])

    text = progress.render_text(report)
    assert "ladder" in text
    assert "rung 1 (still tank)  pass   value 0.0007" in text
    assert "rung 2 (the hull at its design draught)  fail" in text
    assert "20% light against rho g V" in text
