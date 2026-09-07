"""The three invariants of DESIGN.md 8.2, and the facts pinned beside them.

Invariant 1 (no reference without an approved golden) reads every surface the model
can see for a shape word and asks the library for an approved golden of it; it is a
strict xfail until a person approves the goldens, and the never-xfail gate beside it
(test_geometry_library.py::test_an_unapproved_entry_is_never_matched_or_shown) is what
keeps an unapproved shape from reaching the model meanwhile (D32).
Invariants 2 (no commit without a compliance table and clean lint) and 3 (a meshed
result carries a measured fitness table, never a placeholder) are the desk's, and run
over the same `Scripted` harness as test_geometry_agent.py: a fake child returning
canned `RunOutcome`s, a fake case writer, the FakeBackend for the finish.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openreynolds import tools
from openreynolds.geometry import claims as claims_mod
from openreynolds.geometry import desk
from openreynolds.geometry.claims import ComplianceRow, ComplianceTable
from openreynolds.geometry.fitness import FitnessTable, Unmeasured
from openreynolds.tools import ToolContext

from geometry_fakes import CLAIMS, PNG, SCRIPT, install_u3_fallbacks, outcome
from test_geometry_agent import SIZES, Scripted, cfg, last_text, meshed_backend

CLAIMS_JSON = json.dumps(CLAIMS)
SCRIPT_REPLY = f"```python\n{SCRIPT}```"


@pytest.fixture(autouse=True)
def u3(monkeypatch):
    return install_u3_fallbacks(monkeypatch)


@pytest.fixture
def available(monkeypatch):
    monkeypatch.setattr(desk, "unavailable", lambda: None)


# -- invariant 1: no reference without an approved golden ----------------------------


SHAPE_WORDS = ("tesla", "serpentine", "t-junction", "t junction", "venturi", "nozzle", "backward step",
               "backward-facing", "cylinder in", "aerofoil", "airfoil", "manifold", "spiral", "penne",
               "ahmed", "naca", "pin-fin", "bent pipe", "bluff body")


def readable_surface() -> str:
    """Everything the model reads that could name a shape (8.2): the lap-0 message as the
    desk builds it (the request, reference.md, the claims schema, MEASURES, PREDICATES),
    the brief, the sketch docstrings, lint's texts, the two toolbox docstrings and the
    geometry tool's description."""
    from openreynolds.geometry import _toolbox, lint, sketch
    return (desk.BRIEF + desk.claims_message("a duct", reference=None) + sketch.api_summary()
            + (sketch.__doc__ or "")
            + "".join(getattr(sketch, n).__doc__ or "" for n in sketch.API if hasattr(getattr(sketch, n), "__doc__"))
            + "".join(claims_mod.MEASURES.values()) + "".join(p.__doc__ or "" for p in claims_mod.PREDICATES.values())
            + "".join(lint.TEXTS.values())
            + (_toolbox.load("mesh2d").__doc__ or "") + (_toolbox.load("cad_gen").__doc__ or "")
            + next(t for t in tools.TOOLS if t["name"] == "geometry")["description"]).lower()


@pytest.mark.xfail(strict=True, reason="awaiting Kabir's approval of the goldens (cli golden --approve NAME --by Kabir)")
def test_no_reference_without_a_golden(tmp_path):
    """Invariant 1. Every shape word the model can read is a library entry with an
    approved golden whose source_sha matches the entry's file now; and every library
    entry is approved, its golden on disk with its picture, and rebuilds to its golden.
    Strict, so it goes green by itself the day the approvals exist and red again the day
    an entry's source changes without a new approval (or a shape word appears without
    an entry)."""
    from openreynolds.geometry import library
    pytest.importorskip("gmsh")
    named = {w for w in SHAPE_WORDS if w in readable_surface()}
    approved = {e.name for e in library.entries() if library.approval(e) is not None}
    for word in named:
        assert any(word.replace(" ", "_").replace("-", "_") in name for name in approved), word
    for entry in library.entries():
        approval = library.approval(entry)
        assert approval is not None and approval["source_sha"] == library.source_sha(entry), entry.name
        golden = json.loads((library.GOLDEN / f"{entry.name}.json").read_text(encoding="utf-8"))
        assert (library.GOLDEN / f"{entry.name}.png").exists()
        rebuilt = library.regenerate(entry, into=tmp_path)
        assert library.hausdorff(rebuilt["outline"], golden["outline"]) < 1e-6 * library.span_of(golden)
        assert library.measurements_agree(rebuilt["measurements"], golden["measurements"], rel=1e-6)


def test_the_shape_words_the_model_reads_today_are_the_library_entries_plus_the_toolbox_docstrings():
    """What invariant 1 will find the day it runs: the desk's own surfaces name only the
    library's shapes (tesla, serpentine), while `manifold` and `penne` come from the
    mesh2d and cad_gen module docstrings alone -- left alone in Phase 1 (section 12), so
    the invariant stays red on those two words until the docstrings or the library
    change. Pinned so the day it moves is noticed."""
    from openreynolds.geometry import _toolbox, lint, sketch
    desk_side = (desk.BRIEF + desk.claims_message("a duct", reference=None) + sketch.api_summary()
                 + (sketch.__doc__ or "")
                 + "".join(getattr(sketch, n).__doc__ or "" for n in sketch.API if hasattr(getattr(sketch, n), "__doc__"))
                 + "".join(claims_mod.MEASURES.values()) + "".join(p.__doc__ or "" for p in claims_mod.PREDICATES.values())
                 + "".join(lint.TEXTS.values())
                 + next(t for t in tools.TOOLS if t["name"] == "geometry")["description"]).lower()
    assert {w for w in SHAPE_WORDS if w in desk_side} == {"serpentine"}, "the API card names the Serpentine class"
    toolbox = ((_toolbox.load("mesh2d").__doc__ or "") + (_toolbox.load("cad_gen").__doc__ or "")).lower()
    assert {w for w in SHAPE_WORDS if w in toolbox} == {"tesla", "serpentine", "manifold", "penne"}


# -- invariant 2: no commit without a table and clean lint --------------------------


def test_commit_requires_compliance_and_clean_lint(backend, store, available):
    """Invariant 2, for a committed 2D result."""
    # COMMIT over E-OVERLAP: refused naming the code and (20, 5); then a clean build commits
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, SCRIPT_REPLY, "COMMIT", SCRIPT_REPLY, "COMMIT"],
                     outcomes=[outcome(2, errors=[("E-OVERLAP", (20.0, 5.0))]), outcome(0)])
    result = agent.run("a valve", case="a")
    refusal = last_text(agent._provider.calls[3])
    assert refusal.startswith("COMMIT refused") and "E-OVERLAP" in refusal and "(20, 5)" in refusal
    assert result.agreed and result.compliance is not None and result.compliance.checkable >= 1
    assert len(backend.trees) == 1

    # COMMIT over c5 FAIL: refused naming c5; COMMIT disagrees: c3 (passing): refused;
    # COMMIT disagrees: c5: accepted, the row disagreed, the ids recorded
    backend.trees.clear()
    agent = Scripted(cfg(), backend, store, "/work/s",
                     [CLAIMS_JSON, SCRIPT_REPLY, "COMMIT", "COMMIT disagrees: c3", "COMMIT disagrees: c5"],
                     outcomes=[outcome(0, fails=("c5",))])
    result = agent.run("a valve", case="b")
    calls = agent._provider.calls
    assert "c5" in last_text(calls[3]) and last_text(calls[3]).startswith("COMMIT refused")
    assert "nothing to disagree with on c3" in last_text(calls[4])
    assert result.error == "" and not result.agreed and result.disagrees == ["c5"]
    assert next(r for r in result.compliance.rows if r.id == "c5").verdict == "disagreed"
    assert result.compliance is not None and result.compliance.checkable >= 1
    assert result.disagreement_text.startswith("c5 'outer radius 6 mm'")
    assert len(backend.trees) == 1

    # COMMIT disagrees: c5 with a lint error: refused (a lint error is never disagreed with)
    backend.trees.clear()
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, SCRIPT_REPLY, "COMMIT disagrees: c5"],
                     outcomes=[outcome(2, errors=[("E-OVERLAP", (20.0, 5.0))], fails=("c5",))])
    result = agent.run("a valve", case="c")
    assert last_text(agent._provider.calls[3]).startswith("COMMIT refused")
    assert backend.trees == [] and result.error.startswith("no clean geometry")

    # the pure rule, on its own
    ok, reason = claims_mod.can_commit([], None, desk.Reply(kind="commit"))
    assert not ok and reason.startswith("COMMIT refused: no compliance table")
    report_only = ComplianceTable(rows=[ComplianceRow(id="c1", says="x", kind="report", expected="", measured="1",
                                                      verdict="pass")])
    ok, reason = claims_mod.can_commit([], report_only, desk.Reply(kind="commit"))
    assert not ok and "no checkable claim was recorded" in reason
    assert not report_only.ok() and not ComplianceTable([]).ok()


def test_a_desk_with_no_claims_never_commits(backend, store, available):
    """Invariant 2's other half (D33)."""
    agent = Scripted(cfg(), backend, store, "/work/s", ["no", "still no", SCRIPT_REPLY, "COMMIT"],
                     outcomes=[outcome(0, claims=None)])
    result = agent.run("anything", case="c")
    assert backend.trees == [] and agent.written == []
    assert result.capped == "claims" and result.error.startswith("no claims")
    assert result.outline_png == PNG


def test_a_cap_with_no_checkable_claim_never_commits(backend, store, available, monkeypatch):
    """A claims file of report rows alone gives a table with checkable == 0; the cap
    commits nothing (D33), and says why."""
    monkeypatch.setattr(desk, "MAX_LAPS", 2)
    reports_only = {"schema": CLAIMS["schema"], "unit": "mm", "kind": "passage", "flow": "+x",
                    "claims": [c for c in CLAIMS["claims"] if c["kind"] in ("report", "not_measurable")]}
    agent = Scripted(cfg(), backend, store, "/work/s", [json.dumps(reports_only), SCRIPT_REPLY],
                     outcomes=[outcome(0, claims=reports_only)])
    result = agent.run("anything", case="c")
    assert backend.trees == [] and "no checkable claim" in result.error


def test_a_3d_commit_carries_a_one_row_table_and_an_unmeasured_passage(backend, store, available):
    """Invariants 2 and 3 for the unchanged 3D branch (D34)."""
    meshed_backend(backend, case="penne")
    spec = {"solids": [{"op": "box", "name": "body", "size": [0.06, 0.003, 0.003]}], "scale": 1.0}
    agent = Scripted(cfg(), backend, store, "/work/s", [json.dumps(spec), "COMMIT"])
    result = agent.run("a penne", mode="3d", case="penne")
    assert result.meshed
    assert [r.kind for r in result.compliance.rows] == ["not_measurable"]
    assert isinstance(result.fitness.smallest_passage_m, Unmeasured) and "Phase 3" in result.compliance.rows[0].measured
    out, _ = tools.dispatch(ToolContext(backend=backend, store=store, max_output=1000,
                                        geometry=SimpleNamespace(run=lambda *a, **k: result)),
                            "geometry", {"request": "x", "mode": "3d"})
    text = out[-1]["text"]
    assert "across the smallest passage: not measured" in text and "Not yet an OpenFOAM mesh" not in text


def test_warnings_do_not_block_commit_in_phase_1(backend, store, available):
    """D10: plan R2 says warnings print and are drawn, not that they block. The switch
    is one constant, flipped only once the benchmark has measured how often a warning
    fires on a correct shape; until then an unaccepted warning is carried in the result
    and leads the tool text."""
    assert claims_mod.WARNINGS_BLOCK_COMMIT is False
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, SCRIPT_REPLY, "COMMIT"],
                     outcomes=[outcome(0, warns=[("W-CUSP", (12.5, 1.5))])])
    result = agent.run("a valve", case="w")
    assert result.agreed and len(backend.trees) == 1
    assert [f.code for f in result.warnings_unaccepted] == ["W-CUSP"]
    out, _ = tools.dispatch(ToolContext(backend=backend, store=store, max_output=1000,
                                        geometry=SimpleNamespace(run=lambda *a, **k: result)),
                            "geometry", {"request": "x"})
    assert "committed with 1 warning: W-CUSP at (12.5, 1.5)" in out[-1]["text"]


def test_an_accepted_warning_is_recorded_and_not_reported_as_unaccepted(backend, store, available):
    agent = Scripted(cfg(), backend, store, "/work/s",
                     [CLAIMS_JSON, SCRIPT_REPLY, "COMMIT accepts: W-CUSP at (12.5, 1.5) -- the lip is the valve's edge"],
                     outcomes=[outcome(0, warns=[("W-CUSP", (12.5, 1.5))])])
    result = agent.run("a valve", case="w")
    assert result.agreed and result.warnings_unaccepted == []
    assert result.accepted == [("W-CUSP", (12.5, 1.5), "the lip is the valve's edge")]
    assert result.record["accepted"] == [["W-CUSP", [12.5, 1.5], "the lip is the valve's edge"]]


def test_a_cap_never_commits_a_geometry_with_lint_errors(backend, store, available, monkeypatch):
    monkeypatch.setattr(desk, "MAX_LAPS", 3)
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, SCRIPT_REPLY],
                     outcomes=[outcome(2, errors=[("E-TOUCH", (10.0, 1.5))])])
    result = agent.run("anything", case="c")
    assert backend.trees == [] and result.error.startswith("no clean geometry") and result.outline_png == PNG


# -- invariant 3: a meshed result carries a measured fitness table ------------------


def test_result_carries_fitness(backend, store, available):
    """Invariant 3: no placeholder for a mesh's fitness, and the finish's table is
    measured from the log, the writer's cell and the measured passage."""
    kwargs = dict(hex_fraction=1.0, smallest_passage_m=1e-3, cell_m=1e-4, cells_across_smallest_passage=10.0,
                  wall_cell_m=1e-4, first_cell_height_m=5e-5, y_plus=Unmeasured("x"), y_plus_verdict=Unmeasured("x"),
                  non_orth_max=1.0, non_orth_mean=1.0, skew_max=1.0, aspect_max=Unmeasured("x"), schemes="standard",
                  measured_from="probe")
    with pytest.raises(ValueError, match="required"):
        FitnessTable(cells=None, **kwargs)
    with pytest.raises(ValueError, match="required"):
        FitnessTable(cells=1, **dict(kwargs, smallest_passage_m=None))
    with pytest.raises(TypeError):
        desk.GeometryResult().mark_meshed("Mesh OK", None)
    r = desk.GeometryResult()
    with pytest.raises(ValueError):
        r.meshed = True
    r.meshed = False
    with pytest.raises(ValueError):
        desk.GeometryResult(meshed=True)

    meshed_backend(backend)
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, SCRIPT_REPLY, "COMMIT"])
    result = agent.run("a valve", case="valve")
    assert result.meshed and result.fitness is not None
    t = result.fitness
    assert t.cells == 3750 and t.hex_fraction == 1.0 and t.non_orth_max == pytest.approx(42.32, abs=0.01)
    assert t.schemes == "standard" and t.quality_gate == "not applied"
    assert t.smallest_passage_m == pytest.approx(2.98e-3), "passage.min x scale, not a declared width"
    assert t.cell_m == SIZES.cell, "the Sizes write_case returned"
    assert t.cells_across_smallest_passage == pytest.approx(t.smallest_passage_m / t.cell_m)
    assert json.loads((Path(store.fetch_dir()) / "valve" / "fitness.json").read_text())["cells"] == 3750
    out, _ = tools.dispatch(ToolContext(backend=backend, store=store, max_output=1000,
                                        geometry=SimpleNamespace(run=lambda *a, **k: result)),
                            "geometry", {"request": "x"})
    assert "across the smallest passage" in tools.describe(out)


def test_a_meshed_case_whose_writer_gave_no_sizes_is_not_marked_meshed(backend, store, available):
    """A digest without the cell it was meshed at cannot make a fitness table, and a
    result without a fitness table is not `meshed`: the digest is carried in words."""
    meshed_backend(backend)
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, SCRIPT_REPLY, "COMMIT"], sizes=None)
    result = agent.run("a valve", case="valve")
    assert not result.meshed and result.fitness is None
    assert "Mesh OK" in result.checkmesh and "no fitness table" in result.checkmesh


def test_the_finish_reports_a_missing_patch_as_e_mesh_patch(backend, store, available):
    meshed_backend(backend, case="cyl")
    t05 = outcome(0)
    t05.result["record"]["patches"].append({"name": "cylinder", "at": "near:95,30"})
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, SCRIPT_REPLY, "COMMIT"], outcomes=[t05])
    result = agent.run("a cylinder in a channel", case="cyl")
    codes = [f.code for f in result.lint]
    assert codes.count("E-MESH-PATCH") == 1
    finding = next(f for f in result.lint if f.code == "E-MESH-PATCH")
    assert "no patch 'cylinder'" in finding.what and finding.level == "error"


# -- the surfaces pinned beside the invariants ---------------------------------------


def test_reference_md_equals_api_summary():
    from openreynolds.geometry import sketch
    try:
        card = sketch.api_summary()
    except NotImplementedError:
        pytest.skip("U1's api_summary is not built yet; reference.md is U0's placeholder")
    assert desk.REFERENCE_CARD.read_text(encoding="utf-8").strip() == card.strip()


def test_the_recorded_script_recompiles_to_the_recorded_ops(backend, store, available):
    from openreynolds.geometry import compile as compile_mod
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, SCRIPT_REPLY, "COMMIT"])
    result = agent.run("a valve", case="v")
    assert result.record["script"] == SCRIPT and result.source == SCRIPT.strip()
    try:
        ops = compile_mod.recompile(result.record)
    except NotImplementedError:
        pytest.skip("U1's recompile is not built yet")
    assert ops == result.record["ops"]


def test_the_brief_says_claims_verdict_disagrees_and_the_tool_prints():
    text = " ".join(desk.BRIEF.lower().split())
    for word in ("claims", "verdict", "commit disagrees", "the tool prints"):
        assert word in text, word


def test_the_main_prompt_line_is_unchanged():
    from openreynolds.prompt import SYSTEM_PROMPT
    line = next(l for l in SYSTEM_PROMPT.splitlines() if l.startswith("- `geometry`"))
    assert "takes a shape described in words" in line
    assert "claims" not in line.lower() and "must" not in line.lower()
