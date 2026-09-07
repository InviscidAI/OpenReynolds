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


# ----------------------------------------------------------------------------- the worked examples' columns

def test_lengths_print_four_significant_digits_and_never_an_exponent():
    """3.000, 60.00, 300.0 as the worked rows print them; a four-digit length (a 1.5 m duct
    in mm) prints 1500.0, never `1500.` or `1.5e+03`."""
    assert [claims.fmt_len(v) for v in (3, 60, 300, 10, 0.5)] == ["3.000", "60.00", "300.0", "10.00", "0.5000"]
    assert claims.fmt_len(1211) == "1211.0" and claims.fmt_len(1500) == "1500.0" and claims.fmt_len(12345) == "12345.0"
    assert "e" not in claims.fmt_len(123456.7) and not claims.fmt_len(1000).endswith(".")


def t05_measurements():
    """T05 (7.4) by hand: a 300 x 60 duct with a 10 mm hole at (100, 30)."""
    m = t01_measurements()
    m.features = {
        "duct": {"width": 60.0, "length": 300.0, "size_x": 300.0, "size_y": 60.0, "height": 60.0,
                 "centre": [150, 30], "origin": [0, 0]},
        "cyl": {"centre": [100, 30], "radius": 5.0, "diameter": 10.0, "circumference": 31.42, "curves": [5]},
        "fluid": {"extent_x": 300.0, "extent_y": 60.0, "area": 17921.5, "islands": 1, "edges": 5},
    }
    m.patches = {"inlet": {"curves": [1], "n": 1, "length": 60.0, "midpoints": [[0, 30]]},
                 "outlet": {"curves": [2], "n": 1, "length": 60.0, "midpoints": [[300, 30]]},
                 "cylinder": {"curves": [5], "n": 1, "length": 31.42, "midpoints": [[95, 30]]},
                 "walls": {"curves": [3, 4], "n": 2, "length": 600.0, "midpoints": [[150, 0], [150, 60]]}}
    m.bounds, m.extent, m.islands, m.n_curves = (0.0, 0.0, 300.0, 60.0), (300.0, 60.0), 1, 5
    m.rows, m.legs, m.junctions, m.reference_width = {}, {}, [], 10.0
    m.holes = [m.holes[0]]
    m.holes[0].centroid, m.holes[0].radius = (100.0, 30.0), 5.0
    return m


def t05_plan():
    from openreynolds.geometry.compile import Plan, Solved
    return Plan(units="mm", scale=0.001, ops=[], rules=[], ports=[], features={
        "duct": Solved(kind="Rect", params={"origin": [0, 0], "size": [300, 60]}, solved={}, ops=["duct"]),
        "cyl": Solved(kind="Disk", params={"centre": [100, 30], "diameter": 10}, solved={}, ops=["cyl"])},
        instances={}, outlines={}, edge_targets={}, apart=[], declared_widths=[], expected=(1, 1), notes=[])


def test_t05_rows_read_as_the_design_prints_them():
    """7.4: the Disk's diameter row is the bare number (the curvature note is on its
    FEATURES line), the mid-height row prints both centres to one decimal, the distance
    row names the inlet's x."""
    payload = t01()
    payload["claims"] = [
        {"id": "c1", "kind": "measure", "says": "a circular cylinder of 10 mm diameter", "measure": "diameter", "of": "cyl", "value": 10},
        {"id": "c2", "kind": "measure", "says": "a channel 60 mm high", "measure": "height", "of": "duct", "value": 60},
        {"id": "c3", "kind": "measure", "says": "300 mm long", "measure": "length", "of": "duct", "value": 300},
        {"id": "c4", "kind": "predicate", "says": "centred 100 mm from the inlet", "predicate": "at_distance_from",
         "of": "cyl", "args": {"patch": "inlet", "value": 100}},
        {"id": "c5", "kind": "predicate", "says": "mid-height", "predicate": "at_mid_height", "of": "cyl", "args": {"of": "duct"}},
        {"id": "c6", "kind": "patch", "says": "inlet at the left end", "patch": "inlet", "side": "left"},
        {"id": "c7", "kind": "patch", "says": "outlet at the right end", "patch": "outlet", "side": "right"},
        {"id": "c8", "kind": "predicate", "says": "the cylinder surface is its own patch named cylinder",
         "predicate": "own_patch", "of": "cylinder", "args": {"of": "cyl"}},
        {"id": "c9", "kind": "predicate", "says": "the channel's top and bottom are walls", "predicate": "walls_are",
         "of": "walls", "args": {"except": ["cylinder"]}},
        {"id": "c10", "kind": "count", "says": "one cylinder", "of": "holes", "value": 1},
    ]
    cs, _ = parse(payload)
    table = comply(cs, t05_measurements(), [], t05_plan())
    rows = {r.id: (r.measured, r.expected, r.verdict) for r in table.rows}
    assert rows["c1"] == ("cyl.diameter = 10.00", "10 +/- 0.1", "pass")
    assert rows["c2"] == ("duct.height = 60.00", "60 +/- 0.6", "pass")
    assert rows["c3"] == ("duct.length = 300.0", "300 +/- 3", "pass")
    assert rows["c4"] == ("cyl.centre (100, 30); inlet at x = 0: 100.0 from it", "100 +/- 1", "pass")
    assert rows["c5"] == ("cyl.centre.y = 30.0 = duct mid-height 30.0", "+/- 0.6", "pass")
    assert rows["c6"] == ("inlet at (0, 30), the left side", "left", "pass")
    assert rows["c7"] == ("outlet at (300, 30), the right side", "right", "pass")
    assert rows["c8"][0] == "patch 'cylinder' = the 1 curve of cyl.edge" and rows["c8"][2] == "pass"
    assert rows["c9"][0] == "patch 'walls' = every curve not inlet/outlet/cylinder (2)" and rows["c9"][2] == "pass"
    assert rows["c10"] == ("holes = 1", "1", "pass")
    assert table.summary() == "10 claims: 10 pass"


def t02_measurements_and_plan():
    """T02 (7.2) by hand: four passes of 30 stacked in y, bends r 3, the outlet at (0, 18)."""
    from openreynolds.geometry.compile import Plan, Solved
    from openreynolds.geometry.measure import OpenEnd
    m = t01_measurements()
    legs = []
    for k in range(4):
        y = 6 * k
        if k % 2 == 0:
            legs.append({"kind": "line", "from": (0, y), "to": (30, y), "heading": 0, "length": 30})
            if k < 3:
                legs.append({"kind": "arc", "from": (30, y), "to": (30, y + 6), "heading": 0, "heading_out": 180,
                             "centre": (30, y + 3), "radius": 3, "sweep": 180, "outer_radius": 4, "inner_radius": 2})
        else:
            legs.append({"kind": "line", "from": (30, y), "to": (0, y), "heading": 180, "length": 30})
            if k < 3:
                legs.append({"kind": "arc", "from": (0, y), "to": (0, y + 6), "heading": 180, "heading_out": 0,
                             "centre": (0, y + 3), "radius": 3, "sweep": -180, "outer_radius": 4, "inner_radius": 2})
    m.legs = {"snake": legs}
    m.features = {"snake": {"width": 2.0, "passes": 4, "bends": 3, "pass_length": [30.0, 30.0, 30.0, 30.0],
                            "bend_radius": [3.0, 3.0, 3.0], "pass_pitch": 6.0, "wall_between": 4.0,
                            "start": [0, 0], "end": [0, 18], "end_side": "same", "extent": [38, 20],
                            "pass_centrelines": [0.0, 6.0, 12.0, 18.0]},
                  "fluid": {"extent_x": 38.0, "extent_y": 20.0, "area": 296.6, "islands": 0, "edges": 20}}
    m.patches = {"inlet": {"curves": [1], "n": 1, "length": 2.0, "midpoints": [[0, 0]]},
                 "outlet": {"curves": [2], "n": 1, "length": 2.0, "midpoints": [[0, 18]]},
                 "walls": {"curves": list(range(3, 21)), "n": 18, "length": 296.5, "midpoints": []}}
    m.bounds, m.extent, m.islands, m.n_curves, m.reference_width = (0.0, -1.0, 38.0, 19.0), (38.0, 20.0), 0, 20, 2.0
    m.rows, m.junctions, m.holes = {}, [], []
    m.open_ends = [OpenEnd(curve=1, centre=(0, 0), length=2.0, outward_normal=(-1, 0), depth=6.0, corner_angles=(90, 90), name="inlet"),
                   OpenEnd(curve=2, centre=(0, 18), length=2.0, outward_normal=(-1, 0), depth=6.0, corner_angles=(90, 90), name="outlet")]
    plan = Plan(units="mm", scale=0.001, ops=[], rules=[], ports=[], features={
        "snake": Solved(kind="Serpentine", params={"width": 2, "passes": 4, "pass_length": 30, "bend_radius": 3,
                                                   "start": [0, 0], "heading": 0, "stack": "+y"},
                        solved={"legs": legs, "bends": 3, "pass_pitch": 6, "wall_between": 4, "end": [0, 18],
                                "end_heading": 180, "ends_on_start_side": True}, ops=["snake"])},
        instances={}, outlines={}, edge_targets={}, apart=[], declared_widths=[2], expected=(1, 1), notes=[])
    return m, plan


def test_t02_rows_judge_every_pass_and_the_outlet_row_explains_the_parity():
    """7.2: a per-pass list value is judged on every pass and printed `30.00 (x4)`; the
    bends `3.000 (x3)`; and the failing outlet row carries the design's two-line
    explanation, which `disagree` returns verbatim as the disagreement text (D18)."""
    payload = t01()
    payload["claims"] = [
        {"id": "c1", "kind": "measure", "says": "a passage 2 mm wide", "measure": "width", "of": "snake", "value": 2},
        {"id": "c2", "kind": "count", "says": "four straight passes", "of": "passes", "value": 4},
        {"id": "c3", "kind": "measure", "says": "passes 30 mm long", "measure": "pass_length", "of": "snake", "value": 30},
        {"id": "c4", "kind": "count", "says": "three 180-degree U-bends", "of": "bends", "value": 3},
        {"id": "c5", "kind": "predicate", "says": "180-degree U-bends", "predicate": "bends_are_u", "of": "snake"},
        {"id": "c6", "kind": "measure", "says": "3 mm centreline radius", "measure": "bend_radius", "of": "snake", "value": 3},
        {"id": "c7", "kind": "predicate", "says": "the passes stacked in y", "predicate": "stacked_in", "of": "snake",
         "args": {"axis": "y", "count": 4}},
        {"id": "c8", "kind": "patch", "says": "entering the first pass at its left end", "patch": "inlet", "at": [0, 0], "tol": 0.5},
        {"id": "c9", "kind": "patch", "says": "leaving the last pass at its right end", "patch": "outlet", "side": "right"},
        {"id": "c10", "kind": "not_measurable", "says": "mesh, checkMesh, render, results.json",
         "not_measurable": "the finish and the main agent"},
    ]
    cs, _ = parse(payload)
    m, plan = t02_measurements_and_plan()
    table = comply(cs, m, [], plan)
    rows = {r.id: (r.measured, r.expected, r.verdict) for r in table.rows}
    assert rows["c1"] == ("snake.width = 2.000", "2 +/- 0.02", "pass")
    assert rows["c2"] == ("passes = 4", "4", "pass")
    assert rows["c3"] == ("snake.pass_length = 30.00 (x4)", "30 +/- 0.3", "pass")
    assert rows["c4"] == ("bends = 3", "3", "pass")
    assert rows["c5"] == ("sweeps 180, 180, 180 deg", "180 +/- 2", "pass")
    assert rows["c6"] == ("snake.bend_radius = 3.000 (x3)", "3 +/- 0.03", "pass")
    assert rows["c7"] == ("headings 0/180/0/180; centrelines y = 0, 6, 12, 18 (spacing 6)", "stacked in y, 4 passes", "pass")
    assert rows["c8"] == ("inlet at (0, 0)", "(0, 0) +/- 0.5", "pass")
    assert rows["c9"] == ("outlet at (0, 18): the LEFT end (x = 0 of 0..38)", "right", "fail")
    c9 = table.by_id("c9")
    assert c9.detail == ("an even number of passes ends on the inlet's side; 4 passes with 3 bends cannot end at the right.\n"
                         "5 passes (4 bends) or 3 passes (2 bends) end on the right; the request fixes 4 and 3.")
    assert table.summary() == "10 claims: 8 pass, 1 FAIL, 1 not measurable"
    text = table.disagree(["c9"])
    assert text.splitlines() == [
        "c9   leaving the last pass at its right end   outlet at (0, 18): the LEFT end (x = 0 of 0..38)   right   disagreed",
        "     an even number of passes ends on the inlet's side; 4 passes with 3 bends cannot end at the right.",
        "     5 passes (4 bends) or 3 passes (2 bends) end on the right; the request fixes 4 and 3."]
    assert table.summary() == "10 claims: 8 pass, 0 FAIL, 1 disagreed, 1 not measurable"
    m.features["snake"]["pass_length"][2] = 28.0
    table = comply(cs, m, [], plan)
    assert table.by_id("c3").verdict == "fail"
    assert table.by_id("c3").measured == "snake.pass_length = 30.00, 30.00, 28.00, 30.00"


def test_t04_rows_list_the_straight_legs_and_the_spacing():
    """7.3: `u.legs_straight = 2 (60.0, 60.0)`, the parallel legs' row, and both ends on -x."""
    from openreynolds.geometry.compile import Plan, Solved
    from openreynolds.geometry.measure import OpenEnd
    m = t01_measurements()
    legs = [{"kind": "line", "from": (0, 0), "to": (60, 0), "heading": 0, "length": 60},
            {"kind": "arc", "from": (60, 0), "to": (60, 20), "heading": 0, "heading_out": 180, "centre": (60, 10),
             "radius": 10, "sweep": 180, "outer_radius": 14, "inner_radius": 6},
            {"kind": "line", "from": (60, 20), "to": (0, 20), "heading": 180, "length": 60}]
    m.legs = {"u": legs}
    m.features = {"u": {"width": 8.0, "length": 151.4, "legs": 3, "legs_straight": 2, "corners": 0, "bends": 1,
                        "start": [0, 0], "end": [0, 20], "start_heading": 0.0, "end_heading": 180.0, "extent": [74, 28],
                        "spacing": 20.0, "end_side": "same", "bend_radius": 10.0, "bend_sweep": 180.0},
                  "fluid": {"extent_x": 74.0, "extent_y": 28.0, "area": 1211.0, "islands": 0, "edges": 10}}
    m.patches = {"inlet": {"curves": [1], "n": 1, "length": 8.0, "midpoints": [[0, 0]]},
                 "outlet": {"curves": [2], "n": 1, "length": 8.0, "midpoints": [[0, 20]]},
                 "walls": {"curves": list(range(3, 11)), "n": 8, "length": 302.8, "midpoints": []}}
    m.bounds, m.extent, m.islands, m.n_curves, m.reference_width = (0.0, -4.0, 74.0, 24.0), (74.0, 28.0), 0, 10, 8.0
    m.rows, m.junctions, m.holes = {}, [], []
    m.open_ends = [OpenEnd(curve=1, centre=(0, 0), length=8.0, outward_normal=(-1, 0), depth=24.0, corner_angles=(90, 90), name="inlet"),
                   OpenEnd(curve=2, centre=(0, 20), length=8.0, outward_normal=(-1, 0), depth=24.0, corner_angles=(90, 90), name="outlet")]
    plan = Plan(units="mm", scale=0.001, ops=[], rules=[], ports=[], features={
        "u": Solved(kind="Passage", params={"width": 8, "start": [0, 0], "heading": 0},
                    solved={"legs": legs, "length": 151.4, "end": [0, 20], "end_heading": 180}, ops=["u"])},
        instances={}, outlines={}, edge_targets={}, apart=[], declared_widths=[8], expected=(1, 1), notes=[])
    payload = t01()
    payload["claims"] = [
        {"id": "c1", "kind": "measure", "says": "a passage 8 mm wide", "measure": "width", "of": "u", "value": 8},
        {"id": "c2", "kind": "measure", "says": "two straight parallel legs", "measure": "legs_straight", "of": "u", "value": 2},
        {"id": "c3", "kind": "predicate", "says": "20 mm centreline spacing", "predicate": "parallel_legs", "of": "u", "args": {"spacing": 20}},
        {"id": "c4", "kind": "count", "says": "joined by a 180-degree bend", "of": "bends", "value": 1},
        {"id": "c5", "kind": "measure", "says": "10 mm centreline radius", "measure": "bend_radius", "of": "u", "value": 10},
        {"id": "c6", "kind": "predicate", "says": "180-degree bend", "predicate": "bends_are_u", "of": "u"},
        {"id": "c7", "kind": "patch", "says": "enters the lower leg at its left end", "patch": "inlet", "at": [0, 0]},
        {"id": "c8", "kind": "patch", "says": "leaves the upper leg at its left end", "patch": "outlet", "at": [0, 20]},
        {"id": "c9", "kind": "predicate", "says": "both ends on the left", "predicate": "ends_on_same_side", "of": "fluid", "args": {"side": "-x"}},
    ]
    cs, _ = parse(payload)
    table = comply(cs, m, [], plan)
    rows = {r.id: (r.measured, r.expected, r.verdict) for r in table.rows}
    assert rows["c1"] == ("u.width = 8.000", "8 +/- 0.08", "pass")
    assert rows["c2"] == ("u.legs_straight = 2 (60.0, 60.0)", "2", "pass")
    assert rows["c3"] == ("legs 1 and 3: headings 0 / 180, 20.00 apart", "20 +/- 0.2", "pass")
    assert rows["c4"] == ("bends = 1", "1", "pass")
    assert rows["c5"] == ("u.bend_radius = 10.00", "10 +/- 0.1", "pass")
    assert rows["c6"] == ("sweep 180 deg", "180 +/- 2", "pass")
    assert rows["c7"] == ("inlet at (0, 0)", "(0, 0) +/- 4", "pass")
    assert rows["c8"] == ("outlet at (0, 20)", "(0, 20) +/- 4", "pass")
    assert rows["c9"] == ("open ends at (0, 0) facing -x and (0, 20) facing -x", "-x", "pass")
    assert table.summary() == "9 claims: 9 pass"


def test_sharp_corner_searches_within_one_width_of_near():
    """7.5: the outer vertex is w/2 x sqrt 2 = 7.07 from the corner and is found within one
    width (10); a vertex 1.2 widths away is not the corner asked about."""
    from openreynolds.geometry.measure import VertexInfo
    m = t01_measurements()
    m.features = {"duct": {"width": 10.0, "legs": 3, "corners": 1}}
    m.legs = {"duct": [{"kind": "line", "from": (0, 0), "to": (100, 0), "heading": 0, "length": 100},
                       {"kind": "corner", "from": (100, 0), "to": (100, 0), "heading": 0, "turn": 90},
                       {"kind": "line", "from": (100, 0), "to": (100, 80), "heading": 90, "length": 80}]}
    m.reference_width = 10.0
    m.vertices = [VertexInfo(at=(105, -5), curves=(1, 2), interior_deg=90.0, solid_deg=270.0, kind="convex"),
                  VertexInfo(at=(95, 5), curves=(3, 4), interior_deg=270.0, solid_deg=90.0, kind="reflex")]
    v = claims.sharp_corner(m, "duct", {"near": [100, 0], "angle": 90})
    assert v.ok and v.measured == "corner at (100, 0): outer vertex (105, -5) interior 90.0, inner (95, 5) interior 270.0, both lines"
    assert v.expected == "90 +/- 3"
    m.vertices[0].at = (112, 0)
    v = claims.sharp_corner(m, "duct", {"near": [100, 0], "angle": 90})
    assert not v.ok and "no outer vertex of that angle within a width" in v.measured


def test_junction_angles_fall_back_to_the_measured_junctions():
    """A Passage that starts on a wall (start=(main.top, 30)) has no `leave_angle` in its
    feature dict; the built angle is the Junction's `measured_angle` (3.5), so the
    predicate reads it there, and the lip beside it from the Junction too."""
    from openreynolds.geometry.measure import Junction
    m = t01_measurements()
    m.features["branch"] = {"width": 3.0, "legs": 2, "end_heading": 200.0}
    m.junctions.append(Junction(channel="branch", where=(30, 1.5), wall_line="main.top", kind="leave",
                                typed_angle=25.0, measured_angle=24.6, lip=((33.1, 1.5), 31.0), heading=25.0, against_flow=False))
    m.junctions.append(Junction(channel="branch", where=(20, 1.5), wall_line="main.top", kind="return",
                                typed_angle=70.0, measured_angle=69.5, lip=None, heading=200.0, against_flow=True))
    v = claims.shallow_angle(m, "branch", {"_flow": "+x"})
    assert v.ok and v.measured == "branch.leave_angle = 24.6 built at the leg's wall; lip 31.0 on the arc"
    v = claims.steep_angle(m, "branch", {"_flow": "+x"})
    assert v.ok and v.measured.startswith("branch angle at the wall = 69.5")
    del m.features["branch"]["end_heading"]
    v = claims.returns_against_flow(m, "branch", {"_flow": "+x"})
    assert v.ok and v.measured.startswith("heading 200 deg, 160 deg from the flow (+x)")


def test_can_commit_reads_a_comma_separated_disagrees_string_too():
    """Today's desk keeps `disagrees` as the reply's text; the ids are read from it the
    same way as from the list the v2 Reply carries."""
    cs, _ = parse(t01())
    m = t01_measurements()
    m.patches["inlet"]["midpoints"] = [[0, 3]]
    m.patches["outlet"]["midpoints"] = [[60, 3]]
    table = comply(cs, m, [], t01_plan())
    assert table.failing_ids() == ["c8", "c9"]
    ok, why = can_commit([], table, SimpleNamespace(disagrees="c8, c9", accepts=[]))
    assert ok and why == "COMMIT accepted, disagreeing with c8, c9"
    ok, why = can_commit([], table, SimpleNamespace(disagrees="c8", accepts=[]))
    assert not ok and why.startswith("COMMIT refused: c9 FAILS and is not named")


def test_an_angle_built_exactly_on_its_margin_passes():
    """The T01 run of 2026-09-07 asked for a 30 degree leave angle, got 30.000000000000004
    back from the built face, and spent three of its eight laps retreating from a shape
    that met the claim. A margin in degrees is met to a millionth of a degree."""
    m = t01_measurements(leave_angle=30.000000000000004)
    shallow = claims.PREDICATES["shallow_angle"](m, "loops[*]", {"max": 30})
    assert shallow.ok is True, shallow.text
    steep = claims.PREDICATES["steep_angle"](
        t01_measurements(return_angle=59.999999999999993), "loops[*]", {"min": 60})
    assert steep.ok is True, steep.text
    assert claims.PREDICATES["shallow_angle"](
        t01_measurements(leave_angle=30.5), "loops[*]", {"max": 30}).ok is False
