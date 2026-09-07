"""The geometry desk: its brief, its laps (claims -> script -> print-back -> COMMIT), its
commit, its finish, and its seat in the tool surface.

The brief is pinned here and not by `test_prompt.py`: the desk is a tool with a narrow
job and its instructions are allowed to be instructions. What the main prompt says about
it stays under the free-will tests like everything else in the prompt.

The loop runs over a fake `_run_script` returning canned `RunOutcome`s in the shape the
child writes (DESIGN.md 3.10 / 4.3) and a fake `_write_case` returning `(CasePaths,
Sizes)`: no gmsh, no child interpreter, no model. The one end-to-end test at the bottom
runs the real runner and skips where the kernel is not built yet.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openreynolds import geometry, tools
from openreynolds.backend.base import ExecResult
from openreynolds.config import Config
from openreynolds.geometry import BRIEF, GEOMETRY_SYSTEM, GeometryAgent, GeometryResult, desk, extract_reply, extract_spec
from openreynolds.geometry import claims as claims_mod
from openreynolds.geometry import fitness as fitness_mod
from openreynolds.geometry.case import CasePaths
from openreynolds.geometry.claims import ComplianceRow, ComplianceTable
from openreynolds.geometry.fitness import FitnessTable, Unmeasured
from openreynolds.geometry.lint import Finding
from openreynolds.geometry.runner import RunOutcome
from openreynolds.geometry.strategy import Sizes
from openreynolds.llm.base import TextBlock, Turn
from openreynolds.tools import ToolContext

from conftest import FakeMessages, message, text_block
from geometry_fakes import BOUNDARY, CHECKMESH_LOG, CLAIMS, PNG, SCRIPT, fresh, install_u3_fallbacks, outcome

CLAIMS_JSON = json.dumps(CLAIMS)
SCRIPT_REPLY = f"```python\n{SCRIPT}```"
SIZES = Sizes(cell=4.75e-4, wall_cell=4.75e-4, thickness=4.75e-4)

GOOD3D = {"solids": [{"op": "box", "name": "body", "size": [0.06, 0.003, 0.003]}], "scale": 1.0}


def cfg(**over):
    base = dict(llm_api_key="k", model="claude-sonnet-5")
    base.update(over)
    return Config(**base)


class FakeProvider:
    """Scripted replies in the provider's own shape: a `Turn` per call, with usage."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.client = None

    def stream(self, **kwargs):
        # the loop appends to its message list after the call; keep what was sent
        self.calls.append(dict(kwargs, messages=list(kwargs["messages"])))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return Turn(content=[TextBlock(reply)], tokens={"input": 100, "output": 10})


class Scripted(GeometryAgent):
    """The real loop over a fake child and a fake case writer: `outcomes` are handed
    back one per `_run_script` call (the last one repeats), each a fresh copy so the
    desk's annotations never leak between laps."""

    def __init__(self, cfg, backend, store, home, replies, outcomes=(outcome(0),), sizes=SIZES):
        super().__init__(cfg, backend, store, home)
        self._provider = FakeProvider(replies)
        self.outcomes = list(outcomes)
        self.sizes = sizes
        self.runs: list[tuple] = []
        self.written: list[tuple] = []
        self.builds: list = []

    def _run_script(self, script, work, claims_path, reference):
        self.runs.append((script, claims_path, reference))
        chosen = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        return chosen(script) if callable(chosen) else fresh(chosen)

    def _write_case(self, local, record, script, claims, table, study, scale):
        local.mkdir(parents=True, exist_ok=True)
        (local / "Allmesh").write_text("#!/bin/sh\n")
        self.written.append((record, script, claims, table, study, scale))
        return CasePaths(), self.sizes

    # the 3D branch's seams
    def _build(self, mode, spec, work, scale):
        self.builds.append((mode, spec, scale))
        return 0, f"extent 0.06 x 0.003 x 0.003 m ({mode}, scale {scale})", PNG

    def _write_case_3d(self, spec, study, local, scale):
        local.mkdir(parents=True, exist_ok=True)
        (local / "Allmesh").write_text("#!/bin/sh\n")
        self.written.append((spec, study, local, scale))
        return CasePaths(), self.sizes


@pytest.fixture(autouse=True)
def u3(monkeypatch):
    """The U3 pieces the desk calls, real when built and stand-ins meanwhile."""
    return install_u3_fallbacks(monkeypatch)


@pytest.fixture
def available(monkeypatch):
    monkeypatch.setattr(desk, "unavailable", lambda: None)


def meshed_backend(backend, case="valve", boundary=BOUNDARY):
    backend.exec_result = ExecResult(0, "Mesh OK.\nrenders/mesh_z.png", False, None)
    backend.files[f"/work/s/{case}/log.checkMesh"] = CHECKMESH_LOG.encode()
    backend.files[f"/work/s/{case}/renders/mesh_z.png"] = PNG + b"mesh"
    backend.files[f"/work/s/{case}/constant/polyMesh/boundary"] = boundary.encode()


def last_text(call: dict) -> str:
    return call["messages"][-1]["content"][-1]["text"]


# -- the brief ------------------------------------------------------------------


def test_the_brief_says_claims_verdict_disagrees_and_the_tool_prints():
    text = " ".join(BRIEF.lower().split())
    for word in ("claims", "verdict", "commit disagrees", "footprint", "the tool prints", "never run a solver",
                 "never mesh", "reference card", "compliance table"):
        assert word in text, word
    assert f"at most {geometry.MAX_LAPS} script laps" in text
    assert f"in {geometry.MAX_SECONDS:.0f} s" in text
    # a lap at high effort is a minute or more; a four-minute budget ended the first
    # re-measured valve on the cap rather than on agreement
    assert geometry.MAX_SECONDS >= 600
    assert GEOMETRY_SYSTEM is BRIEF


def test_the_brief_is_explicit_and_that_is_allowed_here():
    """The desk's brief is imperative on purpose; the main prompt's line about the tool
    is not. Both facts, checked in one place."""
    assert "reply with only a json object" in BRIEF.lower()
    assert "reply with only a python script" in BRIEF.lower()
    from openreynolds.prompt import SYSTEM_PROMPT
    line = next(l for l in SYSTEM_PROMPT.splitlines() if l.startswith("- `geometry`"))
    assert "takes a shape described in words" in line


def test_the_main_model_is_the_default_and_a_setting_overrides_it(backend, store, available):
    assert GeometryAgent(cfg(), backend, store, "/work/s").model == "claude-sonnet-5"
    assert GeometryAgent(cfg(geometry_model="claude-opus-5"), backend, store, "/work/s").model == "claude-opus-5"


def test_preferences_ride_along_in_the_users_own_words(backend, store, available):
    agent = GeometryAgent(cfg(preferences="work in millimetres"), backend, store, "/work/s")
    assert agent._system().endswith("work in millimetres")
    assert "in their own words" in agent._system()
    assert GeometryAgent(cfg(), backend, store, "/work/s")._system() == BRIEF


def test_the_desk_reasons_at_its_own_effort(backend, store, available):
    """The hosted app runs the main loop at medium, and at medium the model places an
    arc's end right one time in six; the desk's effort is its own, high by default."""
    assert Scripted(cfg(effort="medium"), backend, store, "/work/s", ["x"]).effort == "high"
    assert Scripted(cfg(geometry_effort="max"), backend, store, "/work/s", ["x"]).effort == "max"
    assert geometry.MAX_REPLY_TOKENS >= 16_000


def test_the_claims_message_carries_the_card_the_schema_and_the_request():
    text = desk.claims_message("a duct 3 mm wide", reference=None)
    assert "a duct 3 mm wide" in text
    assert "claims schema" in text.lower() and '"kind": "measure"' in text
    assert text.endswith("Reply with the claims JSON only.")
    assert "reference card" in text.lower()


# -- reading a reply --------------------------------------------------------------


@pytest.mark.parametrize("text, expecting, kind, check", [
    (CLAIMS_JSON, "claims", "claims", lambda r: r.claims["unit"] == "mm"),
    (f"```json\n{CLAIMS_JSON}\n```", "claims", "claims", lambda r: len(r.claims["claims"]) == 12),
    ("Here are the claims:\n" + CLAIMS_JSON, "claims", "claims", lambda r: r.claims is not None),
    ("I cannot write claims.", "claims", "none", lambda r: r.claims is None),
    (SCRIPT_REPLY, "script", "script", lambda r: r.script.startswith('s = Sketch(units="mm")')),
    ("Sure:\n" + SCRIPT_REPLY + "\nDone.", "script", "script", lambda r: "Row(loop" in r.script),
    (SCRIPT, "script", "script", lambda r: r.script == SCRIPT.strip()),
    (CLAIMS_JSON, "script", "claims", lambda r: r.claims["unit"] == "mm"),
    ("I cannot do that.", "script", "none", lambda r: r.script == ""),
    ("COMMIT", "script", "commit", lambda r: r.disagrees == [] and r.accepts == []),
    ("commit", "claims", "commit", lambda r: r.disagrees == []),
    ("COMMIT disagrees: c8, c9 -- four passes end on the inlet's side", "script", "commit",
     lambda r: r.disagrees == ["c8", "c9"] and r.note == "four passes end on the inlet's side"),
    ("COMMIT accepts: W-CUSP at (12.5, 1.5) -- the lip is the valve's edge", "script", "commit",
     lambda r: r.accepts == [("W-CUSP", (12.5, 1.5), "the lip is the valve's edge")]),
    ("COMMIT disagrees: c5\naccepts: W-GAP", "script", "commit",
     lambda r: r.disagrees == ["c5"] and r.accepts == [("W-GAP", None, "")]),
    ("COMMITTED to nothing", "script", "none", lambda r: True),
])
def test_extract_reply_reads_claims_scripts_commits_and_accepts(text, expecting, kind, check):
    reply = extract_reply(text, expecting)
    assert reply.kind == kind
    assert check(reply)


@pytest.mark.parametrize("text, spec, commit", [
    ('{"ops": [1]}', {"ops": [1]}, False),
    ("COMMIT", None, True),
    ("commit\ndisagrees: the third loop overlaps", None, True),
    ("I cannot do that.", None, False),
])
def test_extract_spec_is_kept_for_the_3d_branch(text, spec, commit):
    assert extract_spec(text) == (spec, commit)


# -- the result -------------------------------------------------------------------


def test_the_result_keeps_todays_names_as_aliases():
    r = GeometryResult(report="r", png=PNG, case_rel="/work/s/valve", mesh_report="Mesh OK")
    assert r.png == PNG and r.outline_png == PNG
    assert r.case_rel == "/work/s/valve" and r.paths.case_rel == "/work/s/valve"
    assert r.mesh_report == "Mesh OK" and r.checkmesh == "Mesh OK"
    r.mesh_report = "changed"
    assert r.checkmesh == "changed"
    assert GeometryResult().png is None and GeometryResult().case_rel == ""


# -- the laps --------------------------------------------------------------------


def test_claims_lap_then_script_then_commit(backend, store, available):
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, SCRIPT_REPLY, "COMMIT"])
    result = agent.run("a straight channel 3 mm wide and 60 mm long with 4 bypass loops", case="valve")

    assert result.error == "", result.error
    assert result.agreed and result.laps == 2 and result.claims_revisions == 0
    calls = agent._provider.calls
    assert len(calls) == 3
    # lap 0: the request, the card and the schema; the answer counts the claims
    first = last_text(calls[0])
    assert "4 bypass loops" in first and "claims schema" in first.lower()
    answer = last_text(calls[1])
    assert answer.startswith("12 claims: 9 measurable, 2 reported, 1 not measurable")
    assert "Now the script" in answer
    # lap 1: the picture went back as an image block, then the print-back and the turn
    content = calls[2]["messages"][-1]["content"]
    assert content[0]["type"] == "image" and content[1]["type"] == "text"
    assert "VERDICT" in content[1]["text"] and "Compare the CLAIMS table" in content[1]["text"]
    # the child ran with the claims file beside it
    script, claims_path, reference = agent.runs[0]
    assert script.startswith('s = Sketch(units="mm")') and Path(claims_path).exists()
    assert json.loads(Path(claims_path).read_text())["claims"][0]["id"] == "c1"
    # every call ran with no tools and the desk's own brief
    assert all(c["tools"] == [] and c["system"] == BRIEF for c in calls)
    assert result.tokens == {"input": 300, "output": 30}
    # committed: the record carries the claims and the table; the case is shipped
    record, source, claims, table, study, scale = agent.written[0]
    assert record["claims"] == CLAIMS and table.checkable == 9 and study == "mesh" and scale == 0.001
    assert table.passed == 11, "the two report rows are printed as pass, never judged"
    assert source == result.source and "Row(loop" in source
    local = Path(store.fetch_dir()) / "valve"
    assert (local / "Allmesh").exists()
    assert backend.trees == [(local, "/work/s/valve")]
    assert result.case_rel == "/work/s/valve" and result.scale == 0.001 and result.script == "mesh2d.py"
    assert result.compliance is not None and result.compliance.checkable == 9
    assert result.png == PNG and "VERDICT" in result.report
    assert result.claims is not None and result.claims.unit == "mm"
    assert result.sizes == SIZES and result.measurements is not None


def test_a_commit_before_any_build_is_not_a_commit(backend, store, available):
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, "COMMIT", SCRIPT_REPLY, "COMMIT"])
    result = agent.run("anything")
    assert result.laps == 3 and result.agreed
    assert "no script has built yet" in last_text(agent._provider.calls[2]).lower()


def test_a_reply_that_is_neither_is_told_what_is_accepted(backend, store, available):
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, "Let me think.", SCRIPT_REPLY, "COMMIT"])
    result = agent.run("anything")
    assert result.agreed and result.laps == 3
    nudge = last_text(agent._provider.calls[2])
    assert "```python block" in nudge and "COMMIT" in nudge


def test_a_refused_script_comes_back_as_the_refusal(backend, store, available):
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, SCRIPT_REPLY, SCRIPT_REPLY, "COMMIT"],
                     outcomes=[outcome(5, text="!! ERROR  E-ROW-FIT  Row 'loops': 4 do not fit"), outcome(0)])
    result = agent.run("anything")
    assert result.agreed and result.laps == 3
    refusal = agent._provider.calls[2]["messages"][-1]["content"]
    assert refusal[0]["type"] == "image", "an rc-5 with a partial still draws"
    assert refusal[1]["text"].startswith("The tool refused it:") and "E-ROW-FIT" in refusal[1]["text"]


def test_a_lint_error_lap_asks_for_the_whole_script_and_never_commits_as_is(backend, store, available):
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, SCRIPT_REPLY, "COMMIT", SCRIPT_REPLY, "COMMIT"],
                     outcomes=[outcome(2, errors=[("E-OVERLAP", (20.0, 5.0))]), outcome(0)])
    result = agent.run("anything")
    assert result.agreed and result.laps == 4
    calls = agent._provider.calls
    assert "Fix the errors; reply with the whole script." in last_text(calls[2])
    refused = last_text(calls[3])
    assert refused.startswith("COMMIT refused") and "E-OVERLAP" in refused and "20" in refused


def test_the_claims_lap_retries_once_on_a_malformed_file(backend, store, available):
    bad = json.dumps({"unit": "mm", "kind": "passage", "flow": "+x", "claims": [{"id": "c1", "kind": "measure",
                                                                                   "says": "3 mm wide", "measure": "width", "of": "main"}]})
    agent = Scripted(cfg(), backend, store, "/work/s", [bad, CLAIMS_JSON, SCRIPT_REPLY, "COMMIT"])
    result = agent.run("anything")
    assert result.agreed and result.laps == 2
    retry = last_text(agent._provider.calls[1])
    assert retry.startswith("The claims did not parse:") and "c1" in retry


def test_a_desk_with_no_claims_never_commits(backend, store, available):
    """D33: two malformed claims replies, an rc-0 script lap, COMMIT -> nothing written."""
    agent = Scripted(cfg(), backend, store, "/work/s", ["not json", "still not json", SCRIPT_REPLY, "COMMIT"],
                     outcomes=[outcome(0, claims=None)])
    result = agent.run("anything", case="c")
    assert backend.trees == [] and agent.written == []
    assert result.capped == "claims" and result.error.startswith("no claims")
    assert result.outline_png == PNG and result.compliance is None
    assert agent.runs[0][1] is None, "no claims file went to the child"
    assert "no claims are recorded" in last_text(agent._provider.calls[2]).lower()


def test_a_claims_revision_reruns_compliance_without_a_model_call(backend, store, available):
    revised = copy.deepcopy(CLAIMS)
    revised["claims"][4]["value"] = 4.5
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, SCRIPT_REPLY, json.dumps(revised), "COMMIT"],
                     outcomes=[outcome(0, fails=("c5",)), outcome(0)])
    result = agent.run("anything")
    assert result.agreed and result.claims_revisions == 1 and result.laps == 3
    assert len(agent._provider.calls) == 4
    assert [r[0] for r in agent.runs] == [SCRIPT.strip(), SCRIPT.strip()], "the same script, judged again"
    assert agent.runs[0][1] != agent.runs[1][1], "against the revised claims file"
    assert json.loads(Path(agent.runs[1][1]).read_text())["claims"][4]["value"] == 4.5
    assert "Claims revised (1 revision(s))" in last_text(agent._provider.calls[3])
    assert result.claims.claims[4].value == 4.5


def test_the_lap_cap_commits_the_last_clean_script_and_says_so(backend, store, available, monkeypatch):
    monkeypatch.setattr(desk, "MAX_LAPS", 3)
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, SCRIPT_REPLY])
    result = agent.run("anything", case="c")
    assert result.laps == 3 and result.capped == "laps" and not result.agreed and result.error == ""
    assert backend.trees and agent.written
    fake_desk = SimpleNamespace(run=lambda *a, **k: result)
    out, _ = tools.dispatch(ToolContext(backend=backend, store=store, max_output=1000, geometry=fake_desk),
                            "geometry", {"request": "x"})
    assert "the laps cap ended the laps" in out[1]["text"] and "NOT one the desk agreed" in out[1]["text"]


def test_a_cap_never_commits_a_geometry_with_lint_errors(backend, store, available, monkeypatch):
    monkeypatch.setattr(desk, "MAX_LAPS", 3)
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, SCRIPT_REPLY],
                     outcomes=[outcome(2, errors=[("E-OVERLAP", (20.0, 5.0))])])
    result = agent.run("anything", case="c")
    assert result.error.startswith("no clean geometry") and "E-OVERLAP" in result.error
    assert backend.trees == [] and agent.written == []
    assert result.outline_png == PNG and result.capped == "laps"


def test_the_cap_commits_the_last_rc_0_script_not_a_later_one_with_errors(backend, store, available, monkeypatch):
    """3.14's cap rule: the last rc-0 outcome is what a cap commits. A revision that
    built with lint errors after it does not take the clean one away, and the picture
    and print-back handed back are the committed script's, not the broken revision's."""
    monkeypatch.setattr(desk, "MAX_LAPS", 2)
    clean = outcome(0, png=PNG + b"clean")
    broken = outcome(2, errors=[("E-TOUCH", (10.0, 1.5))], png=PNG + b"broken")
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, SCRIPT_REPLY, f"```python\n{SCRIPT}# v2\n```"],
                     outcomes=[clean, broken])
    result = agent.run("anything", case="c")
    assert result.capped == "laps" and result.error == "" and len(backend.trees) == 1
    assert agent.written[0][1] == SCRIPT.strip() and result.source == SCRIPT.strip()
    assert result.outline_png == PNG + b"clean" and "VERDICT    ready to COMMIT" in result.report
    assert result.record["script_sha256"] == desk.hashlib.sha256(result.record["script"].encode()).hexdigest()


def test_the_claims_lap_is_not_counted(backend, store, available, monkeypatch):
    monkeypatch.setattr(desk, "MAX_LAPS", 2)
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, SCRIPT_REPLY])
    result = agent.run("anything", case="c")
    assert result.laps == 2 and len(agent._provider.calls) == 3 and result.capped == "laps"
    assert result.claims_seconds >= 0.0


def test_the_time_cap_is_honoured(backend, store, available, monkeypatch):
    monkeypatch.setattr(desk, "MAX_SECONDS", 0.0)
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, SCRIPT_REPLY, "COMMIT"])
    result = agent.run("anything")
    assert result.laps == 0 and "no clean geometry within the time" in result.error and backend.trees == []
    assert len(agent._provider.calls) == 1, "the claims lap still ran: its budget is its own"


def test_the_time_cap_commits_a_clean_script_and_says_it_was_the_time(backend, store, available, monkeypatch):
    """The serpentine run: a 90 s budget ended one call at three laps with every edge
    still classified inlet, and the words said "lap cap". The words name which cap."""
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, SCRIPT_REPLY])
    clock = iter([0.0, 0.0, 0.0, 0.0, 700.0])
    monkeypatch.setattr(desk.time, "monotonic", lambda: next(clock, 700.0))
    result = agent.run("anything", case="c")
    assert result.laps == 1 and result.capped == "time" and not result.agreed and backend.trees
    fake_desk = SimpleNamespace(run=lambda *a, **k: result)
    out, _ = tools.dispatch(ToolContext(backend=backend, store=store, max_output=1000, geometry=fake_desk),
                            "geometry", {"request": "x"})
    assert "the time cap ended the laps" in out[1]["text"]


def test_a_commit_that_disagrees_is_recorded_as_claim_ids(backend, store, available):
    agent = Scripted(cfg(), backend, store, "/work/s",
                     [CLAIMS_JSON, SCRIPT_REPLY, "COMMIT disagrees: c9 -- the outlet cannot be on the right"],
                     outcomes=[outcome(0, fails=("c9",))])
    result = agent.run("anything")
    assert not result.agreed and result.disagrees == ["c9"] and result.error == ""
    assert result.disagreement_text.startswith("c9 'outlet at x=60 mm':")
    row = next(r for r in result.compliance.rows if r.id == "c9")
    assert row.verdict == "disagreed" and result.compliance.disagreed == 1
    assert result.record["compliance"]["rows"][8]["verdict"] == "disagreed"
    assert result.case_rel.endswith("/geometry")


def test_e_units_claims_when_the_sketch_and_the_claims_disagree_on_the_unit(backend, store, available):
    """4.1: the claims say m and the sketch says mm, yet the lint's converted ratio test
    passed -- both are right by the numbers, so the record would carry two units for one
    shape. The desk raises it, the commit is blocked until one of them changes."""
    claims_m = copy.deepcopy(CLAIMS)
    claims_m["unit"] = "m"
    agent = Scripted(cfg(), backend, store, "/work/s", [json.dumps(claims_m), SCRIPT_REPLY, "COMMIT"],
                     outcomes=[outcome(0, units="mm")])
    result = agent.run("anything")
    lap = last_text(agent._provider.calls[2])
    assert lap.startswith("!! ERROR  E-UNITS-CLAIMS") and 'unit "m"' in lap and 'units="mm"' in lap
    assert "(60 vs 60)" in lap and "Fix the errors" in lap
    assert "E-UNITS-CLAIMS" in last_text(agent._provider.calls[3])
    assert backend.trees == [] and result.error.startswith("no clean geometry")


def test_a_failed_commit_is_reported_not_raised(backend, store, available):
    class Failing(Scripted):
        def _write_case(self, *a):
            raise RuntimeError("gmsh: could not fuse")
    result = Failing(cfg(), backend, store, "/work/s", [CLAIMS_JSON, SCRIPT_REPLY, "COMMIT"]).run("x")
    assert "did not write" in result.error and "could not fuse" in result.error
    assert result.png == PNG and backend.trees == []


def test_a_model_failure_is_reported_not_raised(backend, store, available):
    from openreynolds.llm.base import ProviderError

    class Down(Scripted):
        def _ask(self, messages):
            raise ProviderError("overloaded", 529)
    result = Down(cfg(), backend, store, "/work/s", ["x"]).run("x")
    assert "model call failed" in result.error and result.laps == 0


def test_without_gmsh_the_desk_says_why(backend, store, monkeypatch):
    monkeypatch.setattr(desk.importlib.util, "find_spec",
                        lambda name: None if name == "gmsh" else object())
    assert "gmsh" in geometry.unavailable()
    result = Scripted(cfg(), backend, store, "/work/s", ["x"]).run("x")
    assert "gmsh" in result.error and result.laps == 0


def test_the_client_seam_is_the_desks(backend, store, available):
    """The same replacement `test_desk.py` uses: the SDK client under the provider."""
    agent = Scripted(cfg(), backend, store, "/work/s", [])
    agent._provider = desk.make_provider(cfg())
    fake = FakeMessages([message([text_block(CLAIMS_JSON)]), message([text_block(SCRIPT_REPLY)]),
                         message([text_block("COMMIT")])])
    agent._client = SimpleNamespace(messages=fake)
    result = agent.run("anything")
    assert result.agreed and result.laps == 2
    assert result.tokens["input"] == 300 and result.tokens["output"] == 30
    assert fake.calls[0]["model"] == "claude-sonnet-5" and fake.calls[0]["system"]


# -- the finish -------------------------------------------------------------------


def test_the_finish_meshes_on_the_instance_and_carries_the_digest_and_picture_back(backend, store, available):
    """After the case is shipped the desk runs Allmesh, checkMesh and the mesh render
    in one exec, reads the boundary back, and marks the result meshed with a fitness
    table; the main agent had spent four to six turns discovering that itself."""
    meshed_backend(backend)
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, SCRIPT_REPLY, "COMMIT"])
    result = agent.run("a valve", case="valve")
    assert result.case_rel == "/work/s/valve"
    cmd, cwd, timeout = backend.last_exec
    assert "sh Allmesh" in cmd and "--scene mesh" in cmd and "render.py . " not in cmd and cwd == "/work/s/valve"
    assert timeout == geometry.FINISH_TIMEOUT_S
    assert result.meshed and "Mesh OK" in result.checkmesh and "3750" in result.checkmesh.replace(",", "")
    assert result.mesh_png == PNG + b"mesh"
    assert result.fitness is not None and result.fitness.cells == 3750
    assert not [f for f in result.lint if f.code == "E-MESH-PATCH"]
    assert (Path(store.fetch_dir()) / "valve" / "fitness.json").exists()


def test_a_finish_that_fails_says_so_and_the_case_still_returns(backend, store, available):
    backend.exec_result = ExecResult(1, "gmshToFoam: cannot open body.msh", False, None)
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, SCRIPT_REPLY, "COMMIT"])
    result = agent.run("a valve", case="valve")
    assert result.case_rel == "/work/s/valve" and not result.meshed and result.fitness is None
    assert "did not finish" in result.checkmesh and "cannot open body.msh" in result.checkmesh
    assert result.mesh_png is None


def test_the_finish_reports_a_missing_patch_as_e_mesh_patch(backend, store, available):
    """The T05 class: the record names `cylinder`, the OpenFOAM boundary file does not."""
    meshed_backend(backend, case="cyl")
    t05 = outcome(0)
    t05.result["record"]["patches"].append({"name": "cylinder", "at": "near:95,30"})
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, SCRIPT_REPLY, "COMMIT"], outcomes=[t05])
    result = agent.run("a cylinder in a channel", case="cyl")
    found = [f for f in result.lint if f.code == "E-MESH-PATCH"]
    assert len(found) == 1 and found[0].level == "error"
    assert "no patch 'cylinder'" in found[0].what and "inlet, outlet, walls, frontAndBack" in found[0].what
    assert result.meshed, "the mesh exists; the missing patch is reported beside it"
    fake_desk = SimpleNamespace(run=lambda *a, **k: result)
    out, _ = tools.dispatch(ToolContext(backend=backend, store=store, max_output=1000, geometry=fake_desk),
                            "geometry", {"request": "x"})
    text = out[-1]["text"]
    assert "!! ERROR  E-MESH-PATCH" in text
    assert text.index("E-MESH-PATCH") < text.index("laps 2")


def test_result_carries_fitness(backend, store, available):
    """Invariant 3 (DESIGN.md 8.2): there is no placeholder for a mesh's fitness."""
    with pytest.raises(ValueError, match="required"):
        FitnessTable(cells=None, hex_fraction=1.0, smallest_passage_m=1e-3, cell_m=1e-4, cells_across_smallest_passage=10.0,
                     wall_cell_m=1e-4, first_cell_height_m=5e-5, y_plus=Unmeasured("x"), y_plus_verdict=Unmeasured("x"),
                     non_orth_max=1.0, non_orth_mean=1.0, skew_max=1.0, aspect_max=Unmeasured("x"), schemes="standard",
                     measured_from="probe")
    with pytest.raises(ValueError, match="required"):
        FitnessTable(cells=1, hex_fraction=1.0, smallest_passage_m=None, cell_m=1e-4, cells_across_smallest_passage=10.0,
                     wall_cell_m=1e-4, first_cell_height_m=5e-5, y_plus=Unmeasured("x"), y_plus_verdict=Unmeasured("x"),
                     non_orth_max=1.0, non_orth_mean=1.0, skew_max=1.0, aspect_max=Unmeasured("x"), schemes="standard",
                     measured_from="probe")
    with pytest.raises(TypeError):
        GeometryResult().mark_meshed("Mesh OK", None)
    r = GeometryResult()
    with pytest.raises(ValueError):
        r.meshed = True
    r.meshed = False
    assert not r.meshed

    meshed_backend(backend)
    agent = Scripted(cfg(), backend, store, "/work/s", [CLAIMS_JSON, SCRIPT_REPLY, "COMMIT"])
    result = agent.run("a valve", case="valve")
    assert result.meshed
    table = result.fitness
    assert table.cells == 3750 and table.hex_fraction == 1.0
    assert table.non_orth_max == pytest.approx(42.32, abs=0.01) and table.schemes == "standard"
    assert table.quality_gate == "not applied"
    # the smallest passage is the measured one (passage.min 2.98 mm x scale), never a
    # declared width; the cell is the writer's
    assert table.smallest_passage_m == pytest.approx(2.98e-3) and table.cell_m == SIZES.cell
    assert table.cells_across_smallest_passage == pytest.approx(table.smallest_passage_m / table.cell_m)
    fake_desk = SimpleNamespace(run=lambda *a, **k: result)
    out, _ = tools.dispatch(ToolContext(backend=backend, store=store, max_output=1000, geometry=fake_desk),
                            "geometry", {"request": "x"})
    assert "across the smallest passage" in tools.describe(out)


# -- the 3D branch ----------------------------------------------------------------


def test_the_3d_branch_is_unchanged(backend, store, available):
    agent = Scripted(cfg(), backend, store, "/work/s", [json.dumps(GOOD3D), "COMMIT"])
    result = agent.run("a box", mode="3d", case="box")
    assert result.agreed and result.laps == 2 and result.script == "cad_gen.py" and result.mode == "3d"
    assert agent.builds == [("3d", GOOD3D, 1.0)] and agent.runs == []
    assert "cad_gen" in desk.build_args("3d", Path("s.json"), Path("out"), 1.0)[1]
    assert backend.trees == [(Path(store.fetch_dir()) / "box", "/work/s/box")]
    assert "The grammar:" in last_text(agent._provider.calls[0])


def test_a_3d_commit_carries_a_one_row_table_and_an_unmeasured_passage(backend, store, available):
    """D34: the invariants hold on the unchanged 3D branch by a table that says what it
    does not know."""
    meshed_backend(backend, case="penne")
    agent = Scripted(cfg(), backend, store, "/work/s", [json.dumps(GOOD3D), "COMMIT"])
    result = agent.run("a penne", mode="3d", case="penne")
    assert result.meshed
    assert result.compliance is not None and len(result.compliance.rows) == 1
    row = result.compliance.rows[0]
    assert row.kind == "not_measurable" and row.verdict == "not_measurable" and "Phase 3" in row.measured
    assert isinstance(result.fitness.smallest_passage_m, Unmeasured)
    assert "Phase 1" in result.fitness.smallest_passage_m.reason
    assert isinstance(result.fitness.cells_across_smallest_passage, Unmeasured)
    fake_desk = SimpleNamespace(run=lambda *a, **k: result)
    out, _ = tools.dispatch(ToolContext(backend=backend, store=store, max_output=1000, geometry=fake_desk),
                            "geometry", {"request": "x", "mode": "3d"})
    text = out[-1]["text"]
    assert "across the smallest passage: not measured" in text and "Not yet an OpenFOAM mesh" not in text


def test_a_3d_commit_that_disagrees_keeps_its_words(backend, store, available):
    agent = Scripted(cfg(), backend, store, "/work/s", [json.dumps(GOOD3D), "COMMIT\ndisagrees: the fourth loop is open"])
    result = agent.run("anything", mode="3d")
    assert not result.agreed and result.disagrees == [] and result.disagreement_text == "the fourth loop is open"


# -- the tool ---------------------------------------------------------------------


def test_geometry_is_the_eighth_tool_and_needs_only_words():
    tool = next(t for t in tools.TOOLS if t["name"] == "geometry")
    assert tool["input_schema"]["required"] == ["request"]
    assert set(tool["input_schema"]["properties"]) == {"request", "case", "mode", "study"}
    assert "geometry" in tools._HANDLERS


def test_the_tool_answers_with_the_picture_first_and_the_words_second(ctx):
    class Desk:
        def run(self, request, mode, study, case):
            self.args = (request, mode, study, case)
            return GeometryResult(report="extent 0.06 x 0.012 m, 4 islands", png=PNG,
                                  case_rel="/work/s/valve", laps=2, seconds=7.0,
                                  tokens={"input": 5}, agreed=True, scale=0.001)
    spent = []
    ctx.geometry = Desk()
    ctx.on_tokens = spent.append
    out, failed = tools.dispatch(ctx, "geometry", {"request": "a tesla valve", "case": "valve"})
    assert not failed
    assert ctx.geometry.args == ("a tesla valve", "2d", "mesh", "valve")
    assert out[0]["type"] == "image" and out[1]["type"] == "text"
    text = out[1]["text"]
    assert "/work/s/valve" in text and "laps 2" in text and "4 islands" in text
    # the first live run read "case written" as "meshed" and went looking for a
    # checkMesh log that did not exist yet; the words say what is still to do
    assert "Not yet an OpenFOAM mesh" in text and "sh Allmesh" in text and "still to do" in text
    # the second live run called the tool again for a one-line change because it
    # believed nothing on the instance could rebuild the spec; the rebuild is named,
    # with the scale the spec was built at
    assert "/work/.toolbox/mesh2d.py . --spec geometry.json --scale 0.001 --force" in text
    assert spent == [{"input": 5}]
    # what survives eviction is the words, and they carry the measurements
    assert "4 islands" in tools.describe(out)


def test_the_tool_says_why_the_finish_did_not_mesh(ctx):
    r = GeometryResult(report="r", png=PNG, case_rel="/work/s/valve", laps=2, seconds=7.0, agreed=True,
                       mesh_report="Allmesh did not finish (exit 1):\ngmshToFoam: cannot open body.msh")
    ctx.geometry = SimpleNamespace(run=lambda *a, **k: r)
    out, _ = tools.dispatch(ctx, "geometry", {"request": "x"})
    text = out[-1]["text"]
    assert "Not yet an OpenFOAM mesh (Allmesh did not finish" in text and "cannot open body.msh" in text
    assert "`sh Allmesh` runs gmshToFoam" in text


def test_the_tool_without_a_desk_says_where_the_same_thing_is(ctx):
    out, failed = tools.dispatch(ctx, "geometry", {"request": "a tesla valve"})
    assert not failed
    assert isinstance(out, str) and "mesh2d.py --spec" in out


def test_the_tool_reports_a_desk_that_did_not_agree(ctx):
    ctx.geometry = SimpleNamespace(run=lambda *a, **k: GeometryResult(
        report="r", case_rel="/work/s/geometry", laps=6, seconds=80.0, agreed=False, disagrees=["c8"],
        disagreement_text="c8 'outlet at the right end of the last pass': outlet at (0, 18), the left end; "
                          "an even number of passes ends on the inlet's side"))
    out, _ = tools.dispatch(ctx, "geometry", {"request": "x"})
    assert ("the desk committed with 1 disagreement -- c8 'outlet at the right end of the last pass': "
            "outlet at (0, 18), the left end; an even number of passes ends on the inlet's side") in out


def test_the_tool_prints_the_claims_block_with_the_rows_to_check_first(ctx):
    table = ComplianceTable(rows=[
        ComplianceRow(id="c1", says="3 mm wide", kind="measure", expected="3 +/- 0.03", measured="3.000", verdict="pass"),
        ComplianceRow(id="c2", says="outlet right", kind="patch", expected="right", measured="outlet at (0, 18)",
                      verdict="disagreed", detail="an even number of passes ends on the inlet's side"),
        ComplianceRow(id="c3", says="mesh it", kind="not_measurable", expected="", measured="not measurable here",
                      verdict="not_measurable"),
        ComplianceRow(id="c4", says="sweep round", kind="report", expected="reported", measured="240 deg", verdict="pass"),
    ])
    ctx.geometry = SimpleNamespace(run=lambda *a, **k: GeometryResult(
        report="MEASURED extent 38 x 20 mm", case_rel="/work/s/snake", laps=3, seconds=40.0, agreed=False,
        disagrees=["c2"], disagreement_text="c2 'outlet right': outlet at (0, 18)", compliance=table))
    out, _ = tools.dispatch(ctx, "geometry", {"request": "x"})
    text = out if isinstance(out, str) else out[-1]["text"]
    assert "not measurable, for you to check" in text
    for first, then in (("c2", "c1"), ("c3", "c1"), ("c4", "c1"), ("c1", "MEASURED extent")):
        assert text.index(first) < text.index(then), (first, then)


def test_the_tool_lists_the_warnings_committed_with(ctx):
    r = GeometryResult(report="r", case_rel="/work/s/valve", laps=2, seconds=7.0, agreed=True,
                       warnings_unaccepted=[Finding(level="warn", code="W-CUSP", subject="fluid", what="a 3 deg wedge",
                                                    where=(12.5, 1.5))])
    ctx.geometry = SimpleNamespace(run=lambda *a, **k: r)
    out, _ = tools.dispatch(ctx, "geometry", {"request": "x"})
    assert "committed with 1 warning: W-CUSP at (12.5, 1.5)" in out


def test_the_tool_answers_with_both_pictures_when_the_finish_meshed(ctx):
    class Desk:
        def run(self, request, mode, study, case):
            r = GeometryResult(report="extent 0.06 x 0.012 m, 4 islands", png=PNG,
                               case_rel="/work/s/valve", laps=2, seconds=7.0, agreed=True, scale=0.001)
            table = FitnessTable(cells=3750, hex_fraction=1.0, smallest_passage_m=2.98e-3, cell_m=4.75e-4,
                                 cells_across_smallest_passage=6.27, wall_cell_m=4.75e-4, first_cell_height_m=2.4e-4,
                                 y_plus=Unmeasured("no flow speed in a mesh-only study"),
                                 y_plus_verdict=Unmeasured("no flow speed in a mesh-only study"), non_orth_max=42.3,
                                 non_orth_mean=7.3, skew_max=1.14, aspect_max=Unmeasured("no line"), schemes="standard",
                                 measured_from="log.checkMesh")
            r.mark_meshed("# checkMesh\ncells 3750\nMesh OK.", table)
            r.mesh_png = PNG + b"m"
            return r
    ctx.geometry = Desk()
    out, failed = tools.dispatch(ctx, "geometry", {"request": "a valve"})
    assert not failed
    assert [b["type"] for b in out] == ["image", "image", "text"]
    text = out[2]["text"]
    assert "meshed there" in text and "Mesh OK." in text and "Not yet" not in text
    assert "--force && sh Allmesh" in text
    assert "FITNESS" in text and "across the smallest passage" in text


def test_the_session_wires_the_desk_only_where_it_can_run():
    source = (Path(geometry.__file__).parents[1] / "cli.py").read_text(encoding="utf-8")
    assert "ctx.on_tokens = loop.add_tokens" in source
    assert "geometry.unavailable() is None" in source
    assert "ctx.geometry = geometry.GeometryAgent(cfg, backend, store, store.session.home)" in source


# -- for real ---------------------------------------------------------------------


def test_end_to_end_a_tesla_valve_lands_on_the_workspace(backend, store, tmp_path):
    """The real runner, the real case writer and a FakeBackend under a scripted model:
    the picture is drawn, the table fills, the case is written and shipped. Skipped
    while the kernel (U1/U4) is not built: the runner says so with NotImplementedError."""
    pytest.importorskip("gmsh")
    pytest.importorskip("matplotlib")
    from openreynolds.geometry import runner
    try:
        runner.run_script("s = 1", tmp_path / "probe", None, None, timeout_s=30)
    except NotImplementedError as exc:
        pytest.skip(f"the runner is not built yet: {exc}")

    class Real(GeometryAgent):
        def __init__(self, *a):
            super().__init__(*a)
            self._provider = FakeProvider([CLAIMS_JSON, SCRIPT_REPLY, "COMMIT"])

    result = Real(cfg(), backend, store, "/work/s").run(
        "a straight channel 3 mm wide and 60 mm long with 4 bypass loops (loop channel 3 mm wide, outer radius "
        "6 mm) that leave the main channel at a shallow angle and return against the forward direction, "
        "inlet at x=0, outlet at x=60 mm", case="tesla")
    assert result.error == "", result.error
    assert result.agreed and result.laps == 2
    assert "islands 4" in result.report or "4 islands" in result.report, result.report
    assert result.png and len(result.png) > 10_000
    local = Path(store.fetch_dir()) / "tesla"
    for name in ("Allmesh", "body.msh", "outline.png", "geometry.json", "geometry.py", "claims.json",
                 "compliance.json", "system/controlDict", "constant/geometry/body.step"):
        assert (local / name).exists(), name
    assert backend.trees == [(local, "/work/s/tesla")]
    assert json.loads((local / "geometry.json").read_text())["scale"] == 0.001
    assert result.compliance is not None and result.compliance.ok()
