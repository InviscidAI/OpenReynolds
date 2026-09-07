"""Claims: the schema, the parser's per-claim answer, the compliance table over hand-built
Measurements, and `can_commit` as a pure function (DESIGN.md 3.7, 4.1, D10, D11, D12, D33).
No gmsh: the Measurements are the U0 skeleton's types filled by hand with the numbers
DESIGN.md 7.1 quotes for T01 and the 4025 shape of Phase 0 (fixture 10)."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from openreynolds.geometry import claims
from openreynolds.geometry.claims import (
    ClaimsError, ComplianceRow, ComplianceTable, can_commit, comply, parse,
)
from openreynolds.geometry.lint import Finding

from geometry_t01_fixture import (
    T01_CLAIMS, row_fit_finding, shape_4025_measurements, t01_findings, t01_measurements, t01_plan,
)


def t01():
    return copy.deepcopy(T01_CLAIMS)


def reply(kind="commit", disagrees=(), accepts=()):
    return SimpleNamespace(kind=kind, disagrees=list(disagrees), accepts=list(accepts), note="")


COMMIT = reply()


# ----------------------------------------------------------------------------- parse

def test_parse_answers_measurable_per_claim():
    cs, lines = parse(t01())
    assert len(cs.claims) == 12 and len(lines) == 12
    measurable = [line for line in lines if line.endswith(": measurable")]
    assert len(measurable) == 9
    assert "c4 width of loops[*]: measurable" in lines
    assert "c7 returns_against_flow of loops[*]: measurable" in lines
    assert lines[9] == "c10 sweep of loops[*]: reported (measured and printed, not judged)"
    assert lines[11].startswith("c12: not measurable here (the finish and the main agent do these")
    assert cs.unit == "mm" and cs.kind == "passage" and cs.flow == "+x"
    answer = claims.answer(cs, lines)
    assert answer.startswith("12 claims: 9 measurable, 2 reported, 1 not measurable\n")
    assert answer.endswith("Now the script. Reply with ONLY a ```python block against the reference card.")


def test_an_unknown_predicate_is_not_an_error():
    payload = t01()
    payload["claims"][5]["predicate"] = "sweeps_round"
    cs, lines = parse(payload)
    c6 = cs.by_id("c6")
    assert c6.kind == "not_measurable"
    assert lines[5].startswith("c6: no predicate named sweeps_round; known: ")
    assert "shallow_angle" in lines[5]
    table = comply(cs, t01_measurements(), [], t01_plan())
    assert table.by_id("c6").verdict == "not_measurable"
    assert table.failed == 0


def test_an_unknown_measure_is_not_an_error_either():
    payload = t01()
    payload["claims"][0]["measure"] = "girth"
    cs, lines = parse(payload)
    assert cs.by_id("c1").kind == "not_measurable"
    assert lines[0].startswith("c1: no measure named girth; known: ")


def test_a_malformed_claim_names_its_id_and_field():
    payload = t01()
    del payload["claims"][3]["value"]
    with pytest.raises(ClaimsError, match=r"^c4: measure claims need value"):
        parse(payload)


@pytest.mark.parametrize("edit, message", [
    (lambda p: p["claims"].append(dict(p["claims"][0])), "c1: duplicate id"),
    (lambda p: p["claims"][1].pop("says"), "c2: says is required"),
    (lambda p: p["claims"][2].update(kind="wish"), "c3: kind 'wish' is not one of"),
    (lambda p: p["claims"][7].update(at=None), "c8: patch claims need at"),
    (lambda p: p["claims"][0].update(value="six"), "c1: value must be a number"),
    (lambda p: p.update(unit="furlongs"), "unit: 'furlongs' is not one of mm, cm, m, in"),
    (lambda p: p.pop("schema"), "schema: None is not"),
    (lambda p: p.update(flow="up"), "flow: 'up' is not one of"),
])
def test_every_schema_refusal_names_the_claim_and_the_field(edit, message):
    payload = t01()
    edit(payload)
    with pytest.raises(ClaimsError, match=message):
        parse(payload)


def test_largest_length_is_derived_from_length_measures():
    cs, _ = parse(t01())
    assert cs.largest_length == 60
    t05 = {"schema": claims.SCHEMA, "unit": "mm", "kind": "passage", "flow": "+x", "claims": [
        {"id": "c1", "kind": "measure", "says": "a 10 mm cylinder", "measure": "diameter", "of": "cyl", "value": 10},
        {"id": "c2", "kind": "measure", "says": "60 mm high", "measure": "height", "of": "duct", "value": 60},
        {"id": "c3", "kind": "measure", "says": "300 mm long", "measure": "length", "of": "duct", "value": 300},
        {"id": "c4", "kind": "predicate", "says": "100 mm from the inlet", "predicate": "at_distance_from",
         "of": "cyl", "args": {"patch": "inlet", "value": 100}},
        {"id": "c5", "kind": "measure", "says": "sweep", "measure": "sweep", "of": "cyl", "value": 400},
    ]}
    cs5, _ = parse(t05)
    assert cs5.largest_length == 300
    assert claims.ClaimSet.from_dict(cs5.as_dict()).largest_length == 300


def test_body_in_flow_without_a_body_in_box_warns_not_fails():
    payload = t01()
    payload["kind"] = "body_in_flow"
    cs, _ = parse(payload)
    assert cs.kind == "body_in_flow"
    table = comply(cs, t01_measurements(), [], t01_plan())
    assert table.notes == ["c0: kind body_in_flow but the script builds no BodyInBox; judged as a passage"]
    assert table.failed == 0 and table.ok()
    assert table.lines()[1] == table.notes[0]
    assert ComplianceTable.from_dict(json.loads(json.dumps(table.as_dict()))).notes == table.notes


# ----------------------------------------------------------------------------- comply

def test_the_accepted_t01_passes_every_claim_with_the_design_columns():
    cs, _ = parse(t01())
    table = comply(cs, t01_measurements(), t01_findings(), t01_plan())
    rows = {r.id: r for r in table.rows}
    assert table.ok() and table.failed == 0 and table.checkable == 9 and table.unmeasurable == 1
    assert (rows["c1"].measured, rows["c1"].expected) == ("main.width = 3.000", "3 +/- 0.03")
    assert (rows["c2"].measured, rows["c2"].expected) == ("main.length = 60.00", "60 +/- 0.1")
    assert (rows["c3"].measured, rows["c3"].expected) == ("loops.count = 4", "4")
    assert (rows["c4"].measured, rows["c4"].expected) == ("loops[*].width = 3.000 (x4)", "3 +/- 0.03")
    assert rows["c5"].measured == "loops[*].outer_radius = 6.000 (x4, from the outer arc's curvature)"
    assert rows["c5"].expected == "6 +/- 0.06"
    assert rows["c6"].measured == "loops[*].leave_angle = 20.0 (x4) built at the leg's wall; lip 28.9 on the arc"
    assert rows["c6"].expected == "<= 30 deg"
    assert rows["c7"].measured == "heading 260 deg, 100 deg from the flow (+x): component -0.17 (x4)"
    assert rows["c7"].expected == "> 90 deg"
    assert (rows["c8"].measured, rows["c8"].expected) == ("inlet at (0, 0)", "(0, 0) +/- 0.5")
    assert (rows["c9"].measured, rows["c9"].expected) == ("outlet at (60, 0)", "(60, 0) +/- 0.5")
    assert (rows["c10"].measured, rows["c10"].expected, rows["c10"].verdict) == ("loops[*].sweep = 240 deg (x4)", "reported", "pass")
    assert rows["c11"].measured == "loops[*].return_angle = 80.0 (x4)"
    assert rows["c12"].verdict == "not_measurable"
    assert rows["c12"].measured == "not measurable here: the finish and the main agent do these"
    assert table.summary() == "12 claims: 9 pass, 0 FAIL, 1 not measurable, 2 reported"


def test_the_4025_shape_fails_c5_c6_c7_against_t01_claims():
    """Fixture 10: the committed Phase 0 valve read through T01's claims. The outer radius
    is 4.000 from the arc's curvature (typed 6 would have passed: the measure comes from
    the walk), the leave leg is 60.0 degrees at the wall, and the return heads 300, WITH
    the flow."""
    cs, _ = parse(t01())
    table = comply(cs, shape_4025_measurements(), [], t01_plan())
    assert table.failing_ids() == ["c5", "c6", "c7"]
    rows = {r.id: r for r in table.rows}
    assert rows["c5"].measured.startswith("loops[*].outer_radius = 4.000 (x4")
    assert rows["c5"].detail.startswith("outer_radius reads 4.000, 2 below the 6")
    assert rows["c6"].measured.startswith("loops[*].leave_angle = 60.0 (x4)")
    assert "heading 300 deg, 60 deg from the flow (+x): component 0.50 (x4) (WITH the flow)" == rows["c7"].measured
    ok, why = can_commit([], table, COMMIT)
    assert not ok and "c5" in why and "c6" in why and "c7" in why


def test_absent_feature_is_a_fail_row_listing_features():
    cs, _ = parse(t01())
    m = t01_measurements()
    plan = t01_plan()
    for key in [k for k in m.features if k.startswith("loops")]:
        del m.features[key]
    del m.rows["loops"]
    del plan.features["loops"]
    del plan.instances["loops"]
    plan.ops = [op for op in plan.ops if op.get("name") != "loops"]
    table = comply(cs, m, [], plan)
    c4 = table.by_id("c4")
    assert c4.verdict == "fail"
    assert c4.measured == "no feature named 'loops'; features: main, loop, fluid"
    assert c4.expected == "3 +/- 0.03"
    assert table.by_id("c3").verdict == "fail" and "no feature named 'loops'" in table.by_id("c3").measured
    assert table.by_id("c6").verdict == "fail" and "no feature named 'loops'" in table.by_id("c6").measured


def test_instance_addressing():
    """`loops[*]` requires all four instances; `loops[2]` judges one; the measured column
    lists the values when they disagree."""
    cs, _ = parse(t01())
    m = t01_measurements()
    m.features["loops[2]"]["outer_radius"] = 4.0
    table = comply(cs, m, [], t01_plan())
    c5 = table.by_id("c5")
    assert c5.verdict == "fail"
    assert c5.measured == "loops[*].outer_radius = 6.000, 6.000, 4.000, 6.000 (from the outer arc's curvature)"
    payload = t01()
    payload["claims"] = [
        {"id": "c5", "kind": "measure", "says": "outer radius 6 mm", "measure": "outer_radius", "of": "loops[2]", "value": 6},
        {"id": "c5b", "kind": "measure", "says": "outer radius 6 mm", "measure": "outer_radius", "of": "loops[1]", "value": 6},
        {"id": "c5c", "kind": "measure", "says": "outer radius 6 mm", "measure": "outer_radius", "of": "loops[7]", "value": 6},
    ]
    one, _ = parse(payload)
    table = comply(one, m, [], t01_plan())
    assert table.by_id("c5").verdict == "fail" and table.by_id("c5").measured.startswith("loops[2].outer_radius = 4.000")
    assert table.by_id("c5b").verdict == "pass" and table.by_id("c5b").measured.startswith("loops[1].outer_radius = 6.000")
    assert table.by_id("c5c").verdict == "fail" and "loops[0]..loops[3]" in table.by_id("c5c").measured


def test_leg_addressing_and_count_words():
    """`name.legs[k]` reads one leg of a Passage (0-based); `count` claims on the words
    islands / open_ends / inlets / outlets / edges / holes count the fluid."""
    payload = t01()
    payload["claims"] = [
        {"id": "c1", "kind": "measure", "says": "first leg 60 long", "measure": "length", "of": "main.legs[0]", "value": 60},
        {"id": "c2", "kind": "measure", "says": "along +x", "measure": "heading", "of": "main.legs[0]", "value": 0},
        {"id": "c3", "kind": "measure", "says": "no second leg", "measure": "length", "of": "main.legs[1]", "value": 1},
        {"id": "c4", "kind": "count", "says": "4 islands", "of": "islands", "value": 4},
        {"id": "c5", "kind": "count", "says": "two open ends", "of": "open_ends", "value": 2},
        {"id": "c6", "kind": "count", "says": "one inlet", "of": "inlets", "value": 1},
        {"id": "c7", "kind": "count", "says": "36 edges", "of": "edges", "value": 36},
        {"id": "c8", "kind": "count", "says": "4 holes", "of": "holes", "value": 3},
        {"id": "c9", "kind": "range", "says": "between 2 and 4 wide", "measure": "width", "of": "main", "min": 2, "max": 4},
    ]
    cs, _ = parse(payload)
    table = comply(cs, t01_measurements(), [], t01_plan())
    rows = {r.id: r for r in table.rows}
    assert rows["c1"].verdict == "pass" and rows["c1"].measured == "main.legs[0].length = 60.00"
    assert rows["c2"].verdict == "pass" and rows["c2"].expected == "0 +/- 2"
    assert rows["c3"].verdict == "fail" and "has 1 legs" in rows["c3"].measured
    assert rows["c4"].measured == "islands = 4" and rows["c4"].verdict == "pass"
    assert rows["c5"].measured == "open_ends = 2" and rows["c5"].verdict == "pass"
    assert rows["c6"].measured == "inlets = 1" and rows["c6"].verdict == "pass"
    assert rows["c7"].measured == "edges = 36"
    assert rows["c8"].verdict == "fail" and rows["c8"].measured == "holes = 4"
    assert rows["c9"].verdict == "pass" and rows["c9"].expected == "2..4"


def test_a_patch_side_claim_fails_with_the_side_it_is_on():
    """T02's c9: the outlet is at the LEFT end where the request said right; the row says
    where it is and the extent it is judged against."""
    payload = t01()
    payload["claims"] = [
        {"id": "c9", "kind": "patch", "says": "leaving the last pass at its right end", "patch": "outlet", "side": "right"},
        {"id": "c8", "kind": "patch", "says": "entering at the left", "patch": "inlet", "side": "left"},
        {"id": "c7", "kind": "patch", "says": "a patch that is not there", "patch": "cylinder", "side": "left"},
    ]
    cs, _ = parse(payload)
    m = t01_measurements()
    m.patches["outlet"]["midpoints"] = [[0, 18]]
    m.bounds = (0.0, -1.5, 38.0, 19.0)
    table = comply(cs, m, [], t01_plan())
    assert table.by_id("c9").verdict == "fail"
    assert table.by_id("c9").measured == "outlet at (0, 18): the LEFT end (x = 0 of 0..38)"
    assert table.by_id("c8").verdict == "pass" and table.by_id("c8").measured == "inlet at (0, 0), the left side"
    assert table.by_id("c7").verdict == "fail" and table.by_id("c7").measured.startswith("no patch named 'cylinder'; patches: inlet, outlet, walls")


def test_a_claim_cannot_narrow_a_predicate_margin():
    """`returns_against_flow` with `min_deg: 60` is judged at 90 and the row says so; a
    heading 100 degrees from the flow passes either way, one at 80 fails at 90."""
    payload = t01()
    payload["claims"][6]["args"] = {"min_deg": 60}
    cs, _ = parse(payload)
    table = comply(cs, t01_measurements(), [], t01_plan())
    c7 = table.by_id("c7")
    assert c7.verdict == "pass" and c7.expected == "> 90 deg"
    assert "min_deg 60 given; the predicate's margin is 90 and is not narrowed" in c7.measured
    table = comply(cs, t01_measurements(return_heading=280.0), [], t01_plan())
    assert table.by_id("c7").verdict == "fail"
    payload["claims"][5]["args"] = {"max": 45}
    cs, _ = parse(payload)
    table = comply(cs, t01_measurements(leave_angle=40.0), [], t01_plan())
    assert table.by_id("c6").verdict == "fail" and table.by_id("c6").expected == "<= 30 deg"


def test_the_table_round_trips_and_disagree_marks_rows():
    cs, _ = parse(t01())
    table = comply(cs, shape_4025_measurements(), [], t01_plan())
    back = ComplianceTable.from_dict(json.loads(json.dumps(table.as_dict())))
    assert [r.as_dict() for r in back.rows] == [r.as_dict() for r in table.rows]
    text = back.disagree(["c5"])
    assert back.by_id("c5").verdict == "disagreed" and back.disagreed == 1
    assert text.startswith("c5   outer radius 6 mm") and "outer_radius reads 4.000" in text
    assert back.failing_ids() == ["c6", "c7"]


# ----------------------------------------------------------------------------- can_commit

def test_can_commit_refuses_without_a_table():
    ok, why = can_commit([], None, COMMIT)
    assert not ok and why.startswith("COMMIT refused: no compliance table")


def test_an_empty_or_report_only_table_is_not_ok():
    """D33: an empty table, or one of report rows alone, never commits; one passing
    measure row does."""
    assert ComplianceTable([]).ok() is False
    report_only = ComplianceTable([ComplianceRow(id="c1", says="sweep", kind="report", expected="reported",
                                                 measured="loops[*].sweep = 240 deg", verdict="pass")])
    assert report_only.ok() is False and report_only.checkable == 0
    ok, why = can_commit([], report_only, COMMIT)
    assert not ok and why.startswith("COMMIT refused: no checkable claim was recorded; the claims lap failed twice")
    one = ComplianceTable([ComplianceRow(id="c1", says="3 mm wide", kind="measure", expected="3 +/- 0.03",
                                         measured="main.width = 3.000", verdict="pass")])
    assert one.ok() is True
    assert can_commit([], one, COMMIT) == (True, "COMMIT accepted")


def test_can_commit_refuses_lint_errors_even_with_disagrees():
    cs, _ = parse(t01())
    table = comply(cs, shape_4025_measurements(), [], t01_plan())
    overlap = Finding(level="error", code="E-OVERLAP", subject="Row 'loops'",
                      what="loops[0] and loops[1] overlap by 26.7 mm2", where=(20.0, 5.0))
    ok, why = can_commit([overlap], table, reply(disagrees=["c5", "c6", "c7"]))
    assert not ok
    assert "E-OVERLAP at (20, 5)" in why and "never be disagreed with" in why
    ok, why = can_commit([overlap, row_fit_finding()], comply(cs, t01_measurements(), [], t01_plan()), COMMIT)
    assert not ok and "E-OVERLAP at (20, 5)" in why and "E-ROW-FIT" in why


def test_can_commit_requires_the_exact_failing_set():
    """D11: `disagrees: c8` on {c8} is accepted; on {c8, c9} refused naming c9; naming a
    passing c3 is refused with 'nothing to disagree with on c3; it passes'."""
    cs, _ = parse(t01())
    m = t01_measurements()
    m.patches["inlet"]["midpoints"] = [[0, 3]]
    one = comply(cs, m, [], t01_plan())
    assert one.failing_ids() == ["c8"]
    ok, why = can_commit([], one, reply(disagrees=["c8"]))
    assert ok and why == "COMMIT accepted, disagreeing with c8"
    m.patches["outlet"]["midpoints"] = [[60, 3]]
    two = comply(cs, m, [], t01_plan())
    assert two.failing_ids() == ["c8", "c9"]
    ok, why = can_commit([], two, reply(disagrees=["c8"]))
    assert not ok and why.startswith("COMMIT refused: c9 FAILS and is not named") and "COMMIT disagrees: c8, c9" in why
    ok, why = can_commit([], two, reply(disagrees=["c9", "c8"]))
    assert ok
    ok, why = can_commit([], one, reply(disagrees=["c3", "c8"]))
    assert not ok and why == "COMMIT refused: nothing to disagree with on c3; it passes"
    ok, why = can_commit([], one, reply(disagrees=["c99"]))
    assert not ok and "c99" in why and "not a claim" in why
    ok, why = can_commit([], one, COMMIT)
    assert not ok and why.startswith("COMMIT refused: c8 FAILS and is not named")


def test_warnings_do_not_block_in_phase_1():
    """D10: `WARNINGS_BLOCK_COMMIT is False` and a warning without `accepts` commits.
    Plan R2 says warnings print and are drawn, not that they block; the benchmark records
    how often a warning fires on a correct shape, and the flip is decided from that number
    (this constant is the one switch). Errors always block."""
    assert claims.WARNINGS_BLOCK_COMMIT is False
    cs, _ = parse(t01())
    table = comply(cs, t01_measurements(), [], t01_plan())
    gap = Finding(level="warn", code="W-GAP", subject="Row 'loops'", what="loops[0] and loops[1] are 0.2 apart",
                  where=(14.0, 1.5))
    ok, why = can_commit([gap], table, COMMIT)
    assert ok and why == "COMMIT accepted, with 1 unaccepted warning"
    assert claims.unaccepted_warnings([gap], COMMIT) == [gap]


def test_accepts_is_recorded(monkeypatch):
    """`reply.accepts` entries are matched by code and `where` within 0.1 mm; with the
    switch flipped an unaccepted warning refuses and an accepted one commits."""
    monkeypatch.setattr(claims, "WARNINGS_BLOCK_COMMIT", True)
    cs, _ = parse(t01())
    table = comply(cs, t01_measurements(), [], t01_plan())
    gap = Finding(level="warn", code="W-GAP", subject="Row 'loops'", what="loops[0] and loops[1] are 0.2 apart",
                  where=(14.0, 1.5))
    ok, why = can_commit([gap], table, COMMIT)
    assert not ok and "W-GAP at (14, 1.5)" in why and "accepts:" in why
    ok, why = can_commit([gap], table, reply(accepts=[("W-GAP", (14.05, 1.5), "the wall is meant")]))
    assert ok and why == "COMMIT accepted"
    ok, why = can_commit([gap], table, reply(accepts=[("W-GAP", (30.0, 1.5), "elsewhere")]))
    assert not ok
    ok, why = can_commit([gap], table, reply(accepts=[("W-CUSP", (14.0, 1.5), "wrong code")]))
    assert not ok
    ok, why = can_commit([gap], table, reply(accepts=[("W-GAP", None, "anywhere")]))
    assert ok


def test_every_measure_sentence_and_predicate_docstring_carries_words_not_numbers():
    """Invariant 1 scans these for shape words; every entry must be a sentence."""
    for key, sentence in claims.MEASURES.items():
        assert key in claims.MEASURES and len(sentence.split()) >= 3, key
    for name, fn in claims.PREDICATES.items():
        assert fn.__doc__ and len(fn.__doc__.split()) >= 4, name
    assert claims.LENGTH_MEASURES <= set(claims.MEASURES) | {"leave_length"}
    assert set(claims.KINDS) == set(claims._REQUIRED)
