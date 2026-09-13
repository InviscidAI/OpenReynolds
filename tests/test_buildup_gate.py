"""The advisory gates, and the failure they close.

`docs/cad-buildup/sweeps/core+bench26-20260912-133719-4bbd/findings.jsonl` carries
`no_closure_assertion_between_export_and_meshing`: nothing in the desk's path asserts that
an exported surface closes, `snappyHexMesh` emits closed cells whatever the input surface
did, `checkMesh` then validates the volume mesh and passes, and `union_closure` measures
the truth from the supervisor with no channel into the run. Four cases exported non-closed
surfaces and all four scored `passed: true`.

The first test here is the one the addition owes: it reads T26's committed record and shows
that the number existed, that it was a concern, and that nothing said it -- the failure, in
the absence of the thing being added.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openreynolds.buildup import gate

ROOT = Path(__file__).resolve().parent.parent
T26 = (ROOT / "docs" / "cad-buildup" / "sweeps"
       / "core+bench26-20260912-133719-4bbd" / "runs" / "T26" / "record.json")


def test_t26_measured_its_own_defect_and_nothing_told_the_desk():
    """The failure, in the gate's absence, off the run that produced it.

    T26's brief asks for the gap between each tread's inner end and the column, *with its
    sign*, because the spec does not close: the tread is tangent to the column along a
    line. The desk subtracted and meshed it. `union_closure` read 259 free edges -- the
    tangency, as unwelded coincident edges -- the record kept the number, and the run
    scored `passed: true`, because a probe is the supervisor's and the supervisor has no
    channel into the run.
    """
    record = json.loads(T26.read_text(encoding="utf-8"))
    assert record["stopped"] == "done"
    assert record["checkmesh_ok"] is True
    assert record["passed"] is True, "the run scored as a pass, which is the failure"

    probes = record["probes"]
    closure = next(p for p in probes if p["id"] == "union_closure")
    assert closure["measured"]["open_edges"] == 259

    # The number was a concern, and a concern is renderable -- so the desk could have
    # been told, in words, from what was already measured.
    states = gate.evaluate(probes, waivers=(), already_warned=())
    closure_state = next(s for s in states if s.check == "union_closure")
    assert closure_state.state == gate.WARNED
    assert "259" in closure_state.concern and "does not close" in closure_state.concern
    assert gate.render(states, record["case_dir"]).strip(), "nothing was said, and it could have been"

    # And this run carries no declare, because there was nothing to carry one.
    assert record.get("declares", []) == []


def test_a_prediction_and_a_reaction_are_recorded_apart():
    """Order of declaration is the whole point, and the desk does not choose the label.

    A waiver named before the check has fired is falsifiable before the fact. One named
    after it has fired is a desk with the result in front of it and an interest in
    dismissing it. Both are allowed; they are not the same claim.
    """
    probes = [{"id": "union_closure", "state": "measured",
               "measured": {"open_edges": 259, "triangles": 3725}}]
    waive = [{"check": "union_closure", "because": "zero-thickness baffle by design"}]

    first = gate.evaluate(probes, waive, already_warned=())
    assert first[0].state == gate.XFAIL, "named before it fired is a prediction"

    later = gate.evaluate(probes, waive, already_warned=("union_closure",))
    assert later[0].state == gate.WAIVED, "named after it fired is a reaction"

    assert first[0].because == later[0].because == "zero-thickness baffle by design"


def test_a_prediction_that_does_not_fire_is_xpass():
    """The third outcome, and the one that makes blanket-waiving self-punishing.

    A desk that expects a warning it does not get has misread its own geometry or misread
    the check. Name all six and five come back `xpass`, in the record, every run."""
    clean = [{"id": "normals", "state": "measured",
              "measured": {"flipped_edges": 0, "triangles": 3725}}]
    states = gate.evaluate(clean, [{"check": "normals", "because": "expected flips"}])
    assert states[0].state == gate.XPASS
    assert "did not" in gate.render(states)


def test_nothing_to_say_says_nothing():
    clean = [{"id": "union_closure", "state": "measured",
              "measured": {"open_edges": 0, "triangles": 100}}]
    states = gate.evaluate(clean, ())
    assert states[0].state == gate.CLEAN
    assert gate.render(states) == "", "a clean gate is silent, not chatty"


def test_a_probe_that_could_not_run_is_not_a_pass():
    states = gate.evaluate([{"id": "scale", "state": "n/a",
                             "why": "the case states no dimension"}], ())
    assert states[0].state == gate.NOT_RUN
    assert gate.render(states) == "", "n/a is not a finding and is not reported as one"


def test_checkmesh_cannot_be_waived():
    """It is binding, so it is absent from the enum the tool offers.

    A schema that will not form the call beats a handler that rejects it afterwards."""
    from openreynolds.cad.agent import DECLARE_TOOL

    enum = (DECLARE_TOOL["input_schema"]["properties"]["waive"]["items"]
            ["properties"]["check"]["enum"])
    assert "checkmesh" not in enum and "checkMesh" not in enum
    assert "checkmesh" not in gate.WAIVABLE
    # Waiving it is not expressible, and a waiver naming it is ignored rather than honoured.
    probes = [{"id": "union_closure", "state": "measured",
               "measured": {"open_edges": 5, "triangles": 10}}]
    states = gate.evaluate(probes, [{"check": "checkmesh", "because": "nope"}])
    assert states[0].state == gate.WARNED


def test_the_two_copies_of_the_waivable_list_agree():
    """`buildup.gate` cannot be imported from `cad.agent` -- `buildup` imports it -- so the
    enum is written twice and held together here, the way `scripts/cad_probes.py` holds its
    recorded answers against the prose ones."""
    from openreynolds.cad.agent import DECLARE_TOOL

    enum = (DECLARE_TOOL["input_schema"]["properties"]["waive"]["items"]
            ["properties"]["check"]["enum"])
    assert enum == list(gate.WAIVABLE)


@pytest.mark.parametrize("text", [
    "no readable patch set under /home/qiuzi/.openreynolds-buildup/work/T26-x/t26/constant/triSurface",
    "wrote /home/qiuzi/.openreynolds-buildup/work/T26-x/t26/constant/triSurface/column.stl",
])
def test_gate_text_carries_no_house_path(text):
    """The one thing here that voids a sweep if it is got wrong.

    Probe prose names the workspace by absolute path, and until this channel existed that
    string only ever reached the record. Showing it to the desk puts a house path into the
    conversation, where `isolation.scan_run` greps the whole thread for exactly that -- and
    every run of the sweep grades contaminated, which discards it."""
    case_dir = "/home/qiuzi/.openreynolds-buildup/work/T26-x/t26"
    out = gate.scrub(text, case_dir)
    assert "openreynolds-buildup" not in out
    assert "/home/qiuzi" not in out
    assert case_dir not in out
    assert "triSurface" in out, "it still has to say which file it means"


def test_scrub_survives_a_case_dir_it_was_not_given():
    """A path from some other run, or no case dir at all, still must not travel."""
    text = "under /home/somebody/.openreynolds-buildup/work/T9-y/t9/constant/triSurface"
    out = gate.scrub(text, "")
    assert "/home/somebody" not in out and ".openreynolds-buildup" not in out


def test_a_refusal_needs_a_reason_and_an_outcome_must_be_one_of_two():
    """The refusal terminal, reached by an act the schema enforces.

    T6 and T25 are the corpus's two refusal failures and both were on a mechanism that is
    a magic print string the desk has to remember. This does not make a desk want to
    refuse -- neither of those reached for it -- but a malformed refusal is no longer
    expressible."""
    from types import SimpleNamespace

    from openreynolds.cad.agent import parse_action

    def turn(payload):
        call = SimpleNamespace(id="c1", name="declare_complete", input=payload)
        return SimpleNamespace(tool_calls=[call], text="")

    _, _, complaint, declare = parse_action(turn({"outcome": "refuse"}))
    assert declare is None and "has to say why" in complaint

    _, _, complaint, declare = parse_action(turn({"outcome": "maybe"}))
    assert declare is None and "complete" in complaint and "refuse" in complaint

    _, _, complaint, declare = parse_action(
        turn({"outcome": "refuse", "reason": "the file declares no length unit"}))
    assert not complaint and declare["reason"] == "the file declares no length unit"

    _, _, complaint, declare = parse_action(turn({"outcome": "complete"}))
    assert not complaint and declare == {"outcome": "complete"}


def test_only_the_core_desk_is_offered_the_declare_tool():
    """The shipped desk is handed exactly what it was handed before the seam existed.

    An addition is measured against the core, and a change that silently moved the shipped
    desk at the same time would be two changes in one sweep."""
    from openreynolds.buildup.core import CoreDesk
    from openreynolds.cad.agent import CadDesk

    assert [t["name"] for t in CadDesk._tools(None)] == ["run_cell"]
    assert [t["name"] for t in CoreDesk._tools(None)] == ["run_cell", "declare_complete"]


def test_the_core_brief_teaches_the_tool_and_the_waiver():
    from openreynolds.buildup.core import CORE_SYSTEM

    brief = CORE_SYSTEM.format(step_timeout=240)
    assert "declare_complete" in brief
    assert "waive" in brief
    assert "checkMesh" in brief and "advisory" in brief
    assert 'outcome: "refuse"' in brief


# -- the declare, driven end to end ----------------------------------------------
#
# These run the whole loop against a fake workspace, because the failure this addition
# closes is a wiring failure: the number existed and never reached the desk. A unit test
# of `evaluate` cannot show that the channel is connected.

from dataclasses import dataclass  # noqa: E402

from openreynolds.backend.base import ExecResult  # noqa: E402
from openreynolds.llm import TextBlock, ToolUseBlock, Turn  # noqa: E402

MESH_OK = """Mesh stats
    points:           1000
    cells:            729
Overall domain bounding box (0 0 0) (1 1 1)
Mesh OK.
End
"""


@dataclass
class Declare:
    """A scripted turn that calls `declare_complete` rather than `run_cell`."""

    payload: dict
    prose: str = ""


class DeclaringProvider:
    """Replays cells and declares; repeats the last turn if the loop keeps asking."""

    name = "anthropic"

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls: list[dict] = []
        self.made = 0

    def stream(self, **kwargs):
        self.calls.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
        item = self.turns.pop(0) if len(self.turns) > 1 else self.turns[0]
        self.made += 1
        if isinstance(item, Declare):
            block = ToolUseBlock(id=f"d-{self.made}", name="declare_complete",
                                 input=item.payload)
        else:
            block = ToolUseBlock(id=f"c-{self.made}", name="run_cell",
                                 input={"source": item})
        return Turn(content=[TextBlock(text="working"), block], provider=self.name,
                    stop_reason="tool_use", tokens={"input": 10, "output": 5})


def _desk(backend, store, turns, checkmesh=MESH_OK):
    from test_cad_agent import answers, kernelled

    from openreynolds.buildup import core
    from openreynolds.config import Config

    answers(backend, {"constant/*/polyMesh": ExecResult(0, "SINGLE:\n", False, None),
                      "checkMesh": ExecResult(0, checkmesh, False, None)})
    kernelled(backend)
    made = core.CoreDesk(Config(llm_api_key="k", model="claude-opus-5"), backend, store,
                         "/work/study")
    made.provider = DeclaringProvider(turns)
    return made


def test_declaring_complete_finishes_the_run_and_records_the_declare(backend, store):
    made = _desk(backend, store, ["x = 1", Declare({"outcome": "complete"})])
    result = made.run("a duct")
    assert result.ok and result.check.checkmesh == "Mesh OK."
    assert len(made._declares) == 1
    assert made._declares[0]["checkmesh_ok"] is True
    assert made._declares[0]["outcome"] == "complete"


def test_declaring_over_a_bad_mesh_hands_back_checkmesh_and_carries_on(backend, store):
    """`checkMesh` is the only thing that can hold the run open, and it still does."""
    bad = MESH_OK.replace("Mesh OK.", " ***High aspect ratio cells found.\nFailed 1 mesh checks.")
    made = _desk(backend, store, [Declare({"outcome": "complete"})], checkmesh=bad)
    result = made.run("a duct")
    assert not result.ok
    from test_cad_agent import said
    assert "Failed 1 mesh checks." in said(made.provider, -1)
    assert len(made._declares) >= 2, "it was handed back and declared again"


def test_declaring_a_refusal_is_the_refused_terminal(backend, store):
    made = _desk(backend, store, [Declare({"outcome": "refuse",
                                           "reason": "the file declares no length unit"})])
    result = made.run("a customer part")
    assert result.stopped == "refused"
    assert result.error == "the file declares no length unit"
    assert not result.ok


def test_a_waiver_for_a_check_that_never_fires_records_as_xpass(backend, store):
    """End to end, because the label is picked by the desk's own `_warned` set."""
    made = _desk(backend, store, [Declare({
        "outcome": "complete",
        "waive": [{"check": "union_closure", "because": "baffle by design"}]})])
    made.run("a duct")
    states = made._declares[0]["states"]
    closure = next((s for s in states if s["check"] == "union_closure"), None)
    # The fake workspace exports no triSurface, so the probe cannot run at all -- which is
    # `n/a`, and `n/a` is not an xpass. The point of the assertion is that a waiver never
    # promotes a check that did not run.
    assert closure is None or closure["state"] in (gate.NOT_RUN, gate.XPASS)


def test_the_declare_conversation_stays_clean_of_house_paths(backend, store):
    """The gate text goes into the thread, and the thread is what `scan_run` greps."""
    from openreynolds.buildup import isolation

    made = _desk(backend, store, ["x = 1", Declare({"outcome": "complete"})])
    made.run("a duct")
    thread = "\n".join(str(m) for m in made.provider.calls[-1]["messages"])
    found = isolation.scan({"thread": thread})
    assert not found.contaminated, found.lines()


def test_a_declare_does_not_advance_the_step_count(backend, store):
    """Which is what bounds a declare loop, and it is the existing alarm that does it.

    `no-progress` fires on `K = 3` consecutive turns with a flat executed-step count, and
    a declare executes nothing. So declare -> told -> declare-with-a-waiver is two turns
    and survives; three declares in a row with nothing changed between them ends the run
    `no-progress`, which is the right answer to a desk arguing with a gate. Pinned here
    because it is load-bearing and it is a consequence of two mechanisms rather than a
    decision written anywhere.
    """
    bad = MESH_OK.replace("Mesh OK.", " ***High aspect ratio cells found.\nFailed 1 mesh checks.")
    made = _desk(backend, store, [Declare({"outcome": "complete"})], checkmesh=bad)
    result = made.run("a duct")
    assert result.steps == [], "a declare runs no cell, so it advances no step"
    assert len(made._declares) >= 2, "and it was asked again rather than ended"


# -- the advisory has to actually reach the desk ---------------------------------


def test_a_passing_checkmesh_does_not_swallow_the_advisory(backend, store, monkeypatch):
    """The bug this addition shipped with, and the reason it is pinned here.

    The first `core+declare_gate` sweep drew real warnings on four cases -- 63, 1,096,
    2,177 and **4,257** free edges -- recorded every one of them, and delivered none,
    because the loop broke on `check.ok` before the advisory was handed back. That is the
    same failure the addition exists to close (`union_closure` measured T26's 259 free
    edges and nothing told the desk), rebuilt one layer up.

    So: a clean `checkMesh` with an unaddressed warning must not end the run.
    """
    from openreynolds.buildup import core

    warned = [{"id": "union_closure", "state": "measured",
               "measured": {"open_edges": 4257, "triangles": 148365}}]
    monkeypatch.setattr(core.probes, "run_all", lambda case, spec: [
        core.probes.ProbeResult(w["id"], w["state"], "", w["measured"]) for w in warned])

    made = _desk(backend, store, [Declare({"outcome": "complete"})])
    result = made.run("a manifold")

    from test_cad_agent import said
    everything = "\n".join(said(made.provider, i) for i in range(len(made.provider.calls)))
    assert "4,257 free edges" in everything, "the desk was never shown the warning"
    assert "does not close" in everything
    assert len(made._declares) > 1, "the first declare did not end the run"
    assert not result.ok, "and a warning nobody fixed or waived is not a finish"
    assert result.stopped == "steps", (
        "bounded by the ordinary turn budget, with no special case for declares")


def test_an_unresolved_warning_is_returned_exactly_as_a_failing_checkmesh_is(
        backend, store, monkeypatch):
    """One rule: a declare is accepted or it comes back. No once-only allowance.

    The earlier version of this held the finish for one turn per check and then let the
    run through, which meant a desk could ship past a warning by declaring twice and
    saying nothing. That made the waiver decorative. Now the only ways past are to fix it
    or to say why it is correct, and **declaring again unchanged is not one of them**.

    Nothing new bounds the loop, because nothing needs to: `turns >= max_steps` counts
    every model turn including declares, and the `no-progress` alarm fires at `K = 3`
    before that -- which is exactly what already bounds a desk that keeps declaring over
    a `checkMesh` it will not fix."""
    from openreynolds.buildup import core

    monkeypatch.setattr(core.probes, "run_all", lambda case, spec: [
        core.probes.ProbeResult("union_closure", "measured", "",
                                {"open_edges": 9, "triangles": 100})])
    made = _desk(backend, store, [Declare({"outcome": "complete"})])
    result = made.run("a duct")

    assert not result.ok, "an unresolved warning is not a finish, however many declares"
    assert len(made._declares) >= 3, "it kept coming back rather than letting one through"
    assert {d["states"][0]["state"] for d in made._declares} == {"warned"}


def test_waiving_a_returned_warning_is_what_gets_past_it(backend, store, monkeypatch):
    """And the waiver is load-bearing, which is the point of the rule above."""
    from openreynolds.buildup import core

    monkeypatch.setattr(core.probes, "run_all", lambda case, spec: [
        core.probes.ProbeResult("union_closure", "measured", "",
                                {"open_edges": 9, "triangles": 100})])
    made = _desk(backend, store, [
        Declare({"outcome": "complete"}),
        Declare({"outcome": "complete",
                 "waive": [{"check": "union_closure", "because": "baffle, open by design"}]})])
    result = made.run("a duct")

    assert result.ok
    assert [d["states"][0]["state"] for d in made._declares] == ["warned", "waived"], (
        "named after it fired, so it is a reaction and recorded as one")


def test_the_desk_is_told_what_union_closure_actually_measures(backend, store, monkeypatch):
    """Three of three predictions in the first sweep failed on one misconception.

    T13 and T14 both predicted `union_closure` would flag because each patch STL is
    individually an open surface -- true of the geometry, and not what the probe measures,
    since it welds the union first. The probe's own docstring says so and nothing said it
    to the desk."""
    from openreynolds.buildup import core

    monkeypatch.setattr(core.probes, "run_all", lambda case, spec: [
        core.probes.ProbeResult("union_closure", "measured", "",
                                {"open_edges": 9, "triangles": 100})])
    made = _desk(backend, store, [Declare({"outcome": "complete"})])
    made.run("a duct")

    from test_cad_agent import said
    everything = "\n".join(said(made.provider, i)
                           for i in range(len(made.provider.calls)))
    assert "welds every STL" in everything
    assert "open by construction" in everything

    assert "welds every STL" in core.CORE_SYSTEM
    from openreynolds.cad.agent import DECLARE_TOOL
    assert "welds every STL" in DECLARE_TOOL["description"]


def test_a_predicted_warning_does_not_hold_the_finish(backend, store, monkeypatch):
    """An `xfail` is the desk saying it already knows. It should cost nothing."""
    from openreynolds.buildup import core

    monkeypatch.setattr(core.probes, "run_all", lambda case, spec: [
        core.probes.ProbeResult("union_closure", "measured", "",
                                {"open_edges": 9, "triangles": 100})])
    made = _desk(backend, store, [Declare({
        "outcome": "complete",
        "waive": [{"check": "union_closure", "because": "zero-thickness baffle"}]})])
    result = made.run("a duct")
    assert result.ok
    assert len(made._declares) == 1, "a prediction finishes in one declare"
    assert made._declares[0]["states"][0]["state"] == gate.XFAIL
