"""`geometry.lint`: the fixture table of DESIGN.md 8.1, each row asserted by `code`,
`level` and `where` (within 0.1 mm), never by report text; plus the tests section 3.6
names.

Every predicate of the catalogue is exercised by a fixture it rejects and one it accepts:
the accepted T01 valve (fixture 13) is lint-clean, the 4025 valve and attempt 3 fail with
the codes the table lists.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
import re

import pytest

from openreynolds.geometry import _toolbox, lint, measure
from openreynolds.geometry.compile import Solved

from geometry_specs import FIXTURES, build, bypass_plan, codes, load_spec, run, spec_plan, with_code

gmsh = pytest.importorskip("gmsh")

CLAIMS = json.loads((FIXTURES / "claims" / "T01.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gm():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    yield gmsh
    gmsh.finalize()


ALL_FINDINGS: list[lint.Finding] = []
"""Every Finding the fixtures produce, for the two regex tests at the end."""


def judged(gm, spec, model, **kw):
    out = run(gm, spec, model=model, **kw)
    ALL_FINDINGS.extend(out[3])
    return out


def near(where, x, y, tol=0.1):
    return where is not None and abs(where[0] - x) <= tol and abs(where[1] - y) <= tol


# -- 1, 1b, 2, 3: overlap, touch, gap ---------------------------------------------------------


def test_overlapping_loops_are_refused_per_pair_with_the_area(gm):
    """Fixture 1: attempt 3 at step 12 on a 20.2 footprint; three consecutive pairs
    overlap by 23.9 mm2 each (the toolbox's own regex pins `2\\d.\\d`), the islands still
    count 4, and the subjects are `bypasses[k]` / `bypasses[k+1]` (D29)."""
    _, _, m, findings, _, _ = judged(gm, load_spec("tesla_real_attempt3.json"), "attempt3")
    overlaps = with_code(findings, "E-OVERLAP")
    assert len(overlaps) == 3 and all(f.level == "error" for f in overlaps)
    assert all(f.numbers["area"] == pytest.approx(23.9, abs=0.5) for f in overlaps)
    assert [(f.numbers["pair_a"], f.numbers["pair_b"]) for f in overlaps] == [(0, 1), (1, 2), (2, 3)]
    assert all(2 < f.where[1] < 9 and 14 < f.where[0] < 50 for f in overlaps), "inside the loops"
    assert all(f"bypasses[{int(f.numbers['pair_a'])}]" in f.what for f in overlaps)
    assert m.islands == 4


def test_two_rows_that_cross_are_refused_between_the_rows(gm):
    """Fixture 1b (D35): the T01 loops and a second, interleaved row of the same loop on
    the same wall; declared apart in the sidecar, their standalone OCC faces intersect."""
    spec = load_spec("two_rows_overlap.json")
    rows = {"loops": Solved(kind="Row", params={}, solved={}, ops=["loops"]),
            "loops2": Solved(kind="Row", params={}, solved={}, ops=["loops2"])}
    _, _, _, findings, _, _ = judged(gm, spec, "two_rows", plan=spec_plan(spec, apart=[("loops", "loops2", None)],
                                                                          features=rows))
    between = [f for f in with_code(findings, "E-OVERLAP") if "loops2" in f.subject]
    assert len(between) == 1 and between[0].level == "error"
    assert between[0].subject == "Row 'loops' and Row 'loops2'"
    assert between[0].numbers["area"] > 100 and near(between[0].where, 29.8, 6.1, tol=0.5)
    clean = judged(gm, spec, "two_rows_not_apart", plan=spec_plan(spec, apart=[], features=rows))[3]
    assert not [f for f in with_code(clean, "E-OVERLAP") if "loops2" in f.subject]


def test_touching_pins_are_refused_at_the_tangent_point(gm):
    _, _, _, findings, _, _ = judged(gm, load_spec("row_touch.json"), "row_touch")
    touches = with_code(findings, "E-TOUCH")
    assert len(touches) == 2 and all(f.level == "error" for f in touches)
    assert all(f.numbers["gap"] == pytest.approx(0.0, abs=1e-9) for f in touches)
    assert near(touches[0].where, 10.0, 1.0) and near(touches[1].where, 20.0, 1.0)
    assert touches[0].numbers["floor"] > 0
    assert "pins[0]" in touches[0].what and "pins[1]" in touches[0].what


def test_a_declared_gap_the_build_does_not_keep_is_refused(gm):
    """Fixture 3: the T01 loop at step 13.0 with `gap: 2` declared; the walls are 0.80
    apart (the judge does not trust the solver)."""
    spec = load_spec("row_gap.json")
    plan = spec_plan(spec, instances={"loops": {"count": 4, "step": (13.0, 0.0), "gap": 2, "gap_declared": True,
                                                "footprint": (13.02, 11.25)}})
    _, _, _, findings, _, _ = judged(gm, spec, "row_gap", plan=plan)
    gap = with_code(findings, "E-ROW-GAP")
    assert len(gap) == 1 and gap[0].level == "error"
    assert gap[0].numbers["gap"] == pytest.approx(0.80, abs=0.01) and gap[0].numbers["declared"] == 2
    assert gap[0].numbers["need"] == pytest.approx(15.02, abs=0.01)
    assert "loops[0]" in gap[0].what and "loops[1]" in gap[0].what
    solved = spec_plan(spec, instances={"loops": {"count": 4, "step": (13.0, 0.0), "gap": 0.8, "gap_declared": False,
                                                  "footprint": (13.02, 11.25)}})
    again = judged(gm, spec, "row_gap_solved", plan=solved)[3]
    assert not with_code(again, "E-ROW-GAP") and not with_code(again, "W-GAP")
    info = with_code(again, "I-GAP")
    assert len(info) == 1 and info[0].numbers["gap"] == pytest.approx(0.80, abs=0.01)


def test_an_undeclared_gap_under_a_tenth_of_the_width_warns(gm):
    spec = load_spec("row_gap.json")
    _, _, _, findings, _, _ = judged(gm, spec, "row_gap_thin", cfg=lint.LintConfig(thin_wall_rel=0.3))
    warn = with_code(findings, "W-GAP")
    assert len(warn) == 1 and warn[0].level == "warn" and warn[0].numbers["gap"] == pytest.approx(0.80, abs=0.01)


# -- 4, 5, 6, 6b, 6c: landings ------------------------------------------------------------------


def test_a_landing_off_the_wall_fails_five_of_five_samples(gm):
    spec = load_spec("tesla_real_attempt1.json")
    _, _, _, findings, _, _ = judged(gm, spec, "attempt1", plan=bypass_plan(spec, "bypass"))
    land = with_code(findings, "E-LAND")
    assert len(land) == 1 and land[0].level == "error"
    assert near(land[0].where, -3.93, 1.5) and land[0].numbers["k"] == 5 and land[0].numbers["n"] == 5
    assert land[0].subject == "Bypass 'bypasses[0]'"
    count = with_code(findings, "E-PORT-COUNT")
    assert any(f.subject == "inlet" and f.numbers["n"] == 0 for f in count)


def test_a_mouth_straddling_the_walls_end_fails_two_of_five(gm):
    spec = load_spec("landing_half.json")
    _, _, _, findings, _, _ = judged(gm, spec, "landing_half", plan=bypass_plan(spec, "bypass"))
    land = with_code(findings, "E-LAND")
    assert len(land) == 1 and land[0].numbers["k"] == 2 and near(land[0].where, 0.3, 1.5)


def test_a_leg_that_stops_short_of_the_wall_is_refused(gm):
    spec = load_spec("landing_short.json")
    _, _, _, findings, _, _ = judged(gm, spec, "landing_short", plan=bypass_plan(spec, "branch", kind="Passage"))
    land = with_code(findings, "E-LAND")
    assert len(land) == 1 and land[0].level == "error"
    assert near(land[0].where, 28.0, 2.0) and land[0].numbers["short"] == pytest.approx(0.5)
    assert land[0].numbers["k"] == 5


def test_a_leg_whose_mouth_is_capped_is_refused(gm):
    spec = load_spec("landing_capped.json")
    _, _, _, findings, _, _ = judged(gm, spec, "landing_capped", plan=bypass_plan(spec, "branch", kind="Passage"))
    land = with_code(findings, "E-LAND")
    assert len(land) == 1 and near(land[0].where, 28.0, 1.5) and land[0].numbers["k"] == 5
    assert "short" not in land[0].numbers


def test_landing_never_runs_on_a_line_to(gm):
    """Fixture 6c (D22/D28): T04's `line_to(x=0)` is a plain `line` leg; nothing carries
    `lands_on`, so there is no E-LAND however the record is read."""
    spec = load_spec("t04.json")
    _, _, _, findings, legs, _ = judged(gm, spec, "t04_no_landing")
    assert not with_code(findings, "E-LAND")
    assert not any(r["kind"] == "to" for r in legs["u"])
    assert lint.has_errors(findings) is False


def test_the_accepted_t01_has_zero_e_land_by_code(gm):
    for name, channel, value in (("tesla_t01_accepted.json", "loop1", 3.0), ("t01_lap2c.json", "loop", 1.5)):
        spec = load_spec(name)
        _, _, _, findings, _, _ = judged(gm, spec, f"land_{name}", plan=bypass_plan(spec, channel, value=value))
        assert not with_code(findings, "E-LAND"), [f.text() for f in findings]


# -- 7, 8, 9: notches and slivers -------------------------------------------------------------------


def test_notch_marches_outward(gm):
    """Fixture 7: a 60-degree branch drawn without `from` leaves half its square cap
    (1.5 = w/2) above the wall, with 0.43 of solid between it and the fluid on its far
    side (0.375 / sin 60), where an inward march would read the whole leg."""
    _, _, _, findings, _, _ = judged(gm, load_spec("notch_cap60.json"), "notch_cap60")
    notch = with_code(findings, "E-NOTCH")
    assert len(notch) == 1 and notch[0].level == "error"
    assert near(notch[0].where, 8.35, 1.88, tol=0.2)
    assert notch[0].numbers["edge"] == pytest.approx(1.5, abs=1e-6)
    assert notch[0].numbers["solid"] == pytest.approx(0.43, abs=0.02)
    _, _, _, clean, _, _ = judged(gm, load_spec("t01_lap2c.json"), "no_notch")
    assert not with_code(clean, "E-NOTCH")


def test_a_return_cap_left_square_pokes_through_the_wall(gm):
    """Fixture 8: attempt 3's loop with its return leg typed as a length, so the square
    cap stands through the wall; every visible cap edge is a notch (solid < w/3 outward)
    and the cap corners leave short edges."""
    _, _, _, findings, _, _ = judged(gm, load_spec("notch_return.json"), "notch_return")
    notch = with_code(findings, "E-NOTCH")
    assert len(notch) >= 1
    assert all(f.numbers["solid"] < 1.0 for f in notch)
    assert any(abs(f.where[1] - 1.5) < 0.2 for f in notch), "a cap edge on the wall line"
    assert len(with_code(findings, "W-SHORT-EDGE")) >= 4


def test_a_sliver_edge_is_an_error_below_a_thirtieth(gm):
    spec = load_spec("sliver.json")
    _, _, _, findings, _, _ = judged(gm, spec, "sliver", plan=spec_plan(spec, declared_widths=[3.0]))
    short = with_code(findings, "E-SHORT-EDGE")
    assert len(short) == 2 and all(f.level == "error" for f in short)
    assert all(f.numbers["edge"] == pytest.approx(0.092, abs=0.002) for f in short)
    assert all(near(f.where, 15.05, 1.46, tol=0.1) for f in short)


# -- 10: the 4025 valve against the T01 claims (the measure side; claims.comply is U3's) ------------


def test_the_4025_valve_fails_radius_angle_and_direction_by_measurement(gm):
    spec = load_spec("tesla_committed_4025.json")
    _, _, m, findings, _, _ = judged(gm, spec, "4025", plan=bypass_plan(spec, "bypass"), claims=CLAIMS)
    inst = m.features["bypasses[*]"]
    assert inst["outer_radius"] == pytest.approx([4.0] * 4, abs=1e-6), "c5: 4.000 from the arc's curvature, not 6"
    assert inst["leave_angle"] == pytest.approx([60.0] * 4, abs=0.5), "c6: 60 at the leg wall, not shallow"
    assert inst["return_heading"] == pytest.approx([300.0] * 4, abs=0.5)
    assert inst["against_flow"] == [False] * 4, "c7: heading 300 is WITH the flow"
    assert lint.has_errors(findings), "the cap corners poking through the wall are notches"
    assert with_code(findings, "E-NOTCH")


# -- 11, 11b, 12, 12b, 20: ports ---------------------------------------------------------------------


def test_an_l_duct_needs_its_ends_named(gm):
    spec = load_spec("lduct2d.json")
    _, _, m, findings, _, _ = judged(gm, spec, "lduct2d")
    assert len(m.open_ends) == 2
    assert with_code(findings, "E-PORT-COUNT") and lint.has_errors(findings)
    named = dict(spec, patches=[{"name": "inlet", "at": "x:min"}, {"name": "outlet", "at": "y:max"}])
    _, _, m2, clean, _, _ = judged(gm, named, "lduct2d_named")
    assert not lint.has_errors(clean)
    assert sorted(e.name for e in m2.open_ends) == ["inlet", "outlet"]


def test_a_manifold_declares_its_outlet_count(gm):
    spec = load_spec("manifold2d.json")
    _, _, m, findings, _, _ = judged(gm, spec, "manifold")
    assert len(m.open_ends) == 5, "the trunk's two caps and three branch ends are candidates; the rules decide"
    assert with_code(findings, "E-PORT-COUNT")
    rules = [{"name": "inlet", "at": "near:0,0"}, {"name": "cap", "kind": "wall", "at": "near:60,0"},
             {"name": "out1", "kind": "outlet", "at": "near:15,22"}, {"name": "out2", "kind": "outlet", "at": "near:30,22"},
             {"name": "out3", "kind": "outlet", "at": "near:45,22"}]
    named = dict(spec, patches=rules)
    _, _, _, still, _, _ = judged(gm, named, "manifold_named")
    assert with_code(still, "E-PORT-COUNT"), "without expect(outlets=3) three outlets are a count error"
    _, _, _, clean, _, _ = judged(gm, named, "manifold_expect", plan=spec_plan(named, expected=(1, 3)))
    assert not lint.has_errors(clean), [f.text() for f in clean]


def test_a_u_duct_needs_near_rules_and_is_clean_with_them(gm):
    for name in ("uduct2d.json", "t04.json"):
        spec = dict(load_spec(name), patches=[])
        _, _, m, findings, _, _ = judged(gm, spec, f"u_{name}")
        assert len(m.open_ends) == 2 and all(e.outward_normal[0] == pytest.approx(-1.0) for e in m.open_ends)
        assert with_code(findings, "E-PORT-COUNT")
    spec = load_spec("t04.json")
    _, _, m, clean, _, _ = judged(gm, spec, "t04_clean")
    assert not lint.has_errors(clean)
    assert m.features["u"]["end_side"] == "same"


def test_an_elbow_refuses_the_automatic_reading(gm):
    spec = load_spec("elbow2d.json")
    _, _, m, findings, _, _ = judged(gm, spec, "elbow2d")
    count = with_code(findings, "E-PORT-COUNT")
    assert count and all(f.numbers["n"] == 0 for f in count)
    assert "(20, 70)" in count[0].what and "(-50, 0)" in count[0].what
    named = dict(spec, patches=[{"name": "inlet", "at": "near:-50,0"}, {"name": "outlet", "at": "near:20,70"}])
    assert not lint.has_errors(judged(gm, named, "elbow2d_named")[3])


def test_a_rule_that_names_two_ends_is_refused(gm):
    spec = load_spec("t04.json")
    plan = spec_plan(spec, rules=[{"name": "inlet", "kind": "inlet", "at": "near:0,0"},
                                  {"name": "outlet", "kind": "outlet", "at": "normal:-x"}])
    _, _, _, findings, _, _ = judged(gm, spec, "t04_normal", plan=plan)
    rule = with_code(findings, "E-PORT-RULE")
    assert len(rule) == 1 and rule[0].level == "error" and rule[0].numbers["n"] == 2
    assert "(0, 0)" in rule[0].what and "20)" in rule[0].what


def test_an_unnamed_open_end_is_an_error(gm):
    spec = dict(load_spec("landing_short.json"))
    _, _, _, findings, _, _ = judged(gm, spec, "open_end")
    open_end = with_code(findings, "E-PORT-OPEN")
    assert len(open_end) == 1 and open_end[0].level == "error" and near(open_end[0].where, 28.0, 2.0)


# -- 13, 14, 15: the accepted valve, and units ------------------------------------------------------


def test_the_accepted_t01_is_lint_clean(gm):
    """Fixture 13: study 20260907-003655-6aab's spec and the lap-2 probe: no errors, no
    E-NOTCH, no E-LAND, I-GAP 0.80 / 2.44, four I-LIP at 28.9 degrees, no fluid cusp."""
    for name, channel, value, gap in (("tesla_t01_accepted.json", "loop1", 3.0, 0.80), ("t01_lap2c.json", "loop", 1.5, 2.44)):
        spec = load_spec(name)
        _, wk, m, findings, _, _ = judged(gm, spec, f"accepted_{name}", plan=bypass_plan(spec, channel, value=value),
                                          claims=CLAIMS)
        assert not lint.has_errors(findings), [f.text() for f in findings]
        assert not with_code(findings, "E-NOTCH") and not with_code(findings, "E-LAND")
        assert not with_code(findings, "W-CUSP") and not with_code(findings, "I-CUSP")
        lips = with_code(findings, "I-LIP")
        assert len(lips) == 4 and all(f.numbers["deg"] == pytest.approx(28.9, abs=0.1) for f in lips)
        assert all(f.level == "info" and abs(f.where[1] - value) < 1e-6 for f in lips)
        info = with_code(findings, "I-GAP")
        assert len(info) == 1 and info[0].numbers["gap"] == pytest.approx(gap, abs=0.01)
        assert not with_code(findings, "E-UNITS") and not with_code(findings, "W-UNITS")
        assert m.islands == 4


def test_a_thousandfold_unit_slip_is_an_error(gm):
    spec = load_spec("t01_lap2c.json")
    _, _, _, findings, _, _ = judged(gm, spec, "units_m", claims=dict(CLAIMS, unit="m"))
    units = with_code(findings, "E-UNITS")
    assert len(units) == 1 and units[0].level == "error"
    assert units[0].numbers["r"] == pytest.approx(0.001, rel=1e-6) and units[0].numbers["Lconv"] == 60000


def test_a_fourfold_unit_slip_warns(gm):
    spec = dict(load_spec("t01_lap2c.json"), scale=0.004)
    _, _, _, findings, _, _ = judged(gm, spec, "units_x4", claims=CLAIMS)
    warn = with_code(findings, "W-UNITS")
    assert len(warn) == 1 and warn[0].level == "warn" and warn[0].numbers["r"] == pytest.approx(4.0)
    assert not with_code(findings, "E-UNITS")


# -- 16, 17: bodies in a flow ---------------------------------------------------------------------


def _body(gm, name, model):
    mesh2d = _toolbox.load("mesh2d")
    gm.model.add(model)
    ops, _ = mesh2d.parse_spec(load_spec(name))
    return mesh2d.build_face(gm, ops, scale=1.0)


def test_a_hollow_body_is_a_void(gm):
    face = _body(gm, "disk_with_hole_external.json", "annulus_body")
    findings = lint.judge_body(gm, face, "tube", "mm", 1.0)
    assert [f.code for f in findings] == ["E-VOID"] and findings[0].level == "error"
    assert near(findings[0].where, 0.0, 0.0) and findings[0].numbers["area"] == pytest.approx(9 * 3.14159, rel=5e-3)
    assert findings[0].subject == "BodyInBox 'tube'"


def test_a_passage_drawn_into_a_body_is_an_open_end_on_the_body(gm):
    face = _body(gm, "cup_external.json", "cup_body")
    findings = lint.judge_body(gm, face, "cup", "mm", 4.0)
    assert [f.code for f in findings] == ["E-PORT-ON-BODY"] and findings[0].level == "error"
    assert near(findings[0].where, 12.0, 5.0) and findings[0].numbers["length"] == pytest.approx(4.0)
    assert "+x" in findings[0].what
    block = _body(gm, "t05.json", "plain_body") if False else None
    gm.model.add("plain_block")
    occ = gm.model.occ
    r = occ.addRectangle(0, 0, 0, 20, 10)
    occ.synchronize()
    assert lint.judge_body(gm, r, "block", "mm", 4.0) == []
    ALL_FINDINGS.extend(findings)


def test_the_same_slot_reads_as_an_open_end_on_the_fluids_hole_loop(gm):
    spec = {"ops": [
        {"op": "rect", "name": "box", "origin": [-20, -20], "size": [70, 50]},
        {"op": "rect", "name": "block", "origin": [0, 0], "size": [20, 10]},
        {"op": "rect", "name": "slot", "origin": [12, 3], "size": [9, 4]},
        {"op": "cut", "name": "cup", "from": "block", "take": ["slot"]},
        {"op": "cut", "name": "body", "from": "box", "take": ["cup"]}],
        "patches": [{"name": "inlet", "at": "x:min"}, {"name": "outlet", "at": "x:max"},
                    {"name": "farfield", "at": "y:min"}, {"name": "farfield", "at": "y:max"}]}
    plan = spec_plan(spec, features={"wing": Solved(kind="BodyInBox", params={}, solved={"box": (-20, -20, 50, 30)},
                                                    ops=["cup"])}, declared_widths=[4.0])
    _, _, _, findings, _, _ = judged(gm, spec, "cup_fluid", plan=plan)
    on_body = with_code(findings, "E-PORT-ON-BODY")
    assert len(on_body) == 1 and near(on_body[0].where, 12.0, 5.0)


# -- 18, 18b, 19: passage width, reference width, cusps and lips ---------------------------------------


def test_a_throat_under_half_the_declared_width_warns_with_the_bisected_number(gm):
    _, _, m, findings, _, _ = judged(gm, load_spec("throat.json"), "throat")
    warn = with_code(findings, "W-PASSAGE")
    assert len(warn) == 1 and warn[0].level == "warn"
    assert warn[0].numbers["min"] == pytest.approx(1.0, abs=0.003) and near(warn[0].where, 14.6, -1.5, tol=1.5)
    assert m.reference_width_from == "declared by 'duct'"
    assert not lint.has_errors(findings)


def test_reference_width_with_no_width_anywhere(gm):
    _, _, m, findings, _, _ = judged(gm, load_spec("t05.json"), "t05_reference")
    assert m.reference_width == pytest.approx(10.0) and m.reference_width_from == "hole diameter (cyl)"
    assert not lint.has_errors(findings)
    ref = with_code(findings, "I-REFERENCE")
    assert len(ref) == 1 and ref[0].level == "info"
    bare = {"scale": 0.001, "ops": [{"op": "rect", "name": "body", "origin": [0, 0], "size": [50, 8]}]}
    _, _, mb, _, _, _ = judged(gm, bare, "bare_rect")
    assert mb.reference_width_from == "extent / 10"


def test_a_three_degree_wedge_of_fluid_is_a_cusp_warning(gm):
    _, _, _, findings, _, _ = judged(gm, load_spec("tangent_walls.json"), "tangent_walls")
    cusp = with_code(findings, "W-CUSP")
    assert len(cusp) == 1 and cusp[0].level == "warn"
    assert near(cusp[0].where, 60.0, 10.0) and cusp[0].numbers["deg"] == pytest.approx(3.0, abs=0.05)
    assert not with_code(findings, "I-LIP")


# -- 21: kernel state --------------------------------------------------------------------------------


def test_a_mirror_as_the_fluids_last_op_is_refused(gm):
    spec = load_spec("mirror_last.json")
    assert [f.code for f in lint.kernel_findings(spec["ops"])] == ["E-MIRROR-BAKE"]
    assert lint.kernel_findings(load_spec("t01_lap2c.json")["ops"]) == []
    kept = [dict(op, keep=True) if op["op"] == "mirror" else op for op in spec["ops"]]
    assert lint.kernel_findings(kept) == [], "a mirror with keep fuses the copy and is clean"
    _, wk, _, findings, _, _ = judged(gm, spec, "mirror_last")
    assert wk.repaired is True
    assert {"E-MIRROR-BAKE", "E-KERNEL-STATE"} <= set(codes(findings))


# -- from_legacy and the text invariants -------------------------------------------------------------


def test_from_legacy_keeps_the_overlap_number():
    check = {"level": "error", "where": (17.7, 6.4), "pair": (1, 2), "overlap": 26.7, "instance_area": 104.0,
             "footprint": (19.74, 11.25), "pitch": 12.0,
             "what": "repeat 'loops': copies 1 and 2 overlap by 26.7 (units squared); the step is smaller than a "
                     "copy's footprint (19.74 x 11.25) plus a gap"}
    (f,) = lint.from_legacy([check], "loops", lint.LintConfig())
    assert f.code == "E-OVERLAP" and f.level == "error" and f.where == (17.7, 6.4)
    assert f.numbers["area"] == 26.7 and f.numbers["pct"] == pytest.approx(25.7, abs=0.1)
    assert "loops[0]" in f.what and "loops[1]" in f.what and "copies" not in f.what
    assert f.as_legacy() == {"level": "error", "where": "Row 'loops'", "what": f.what}
    noise = dict(check, overlap=3e-12)
    (n,) = lint.from_legacy([noise], "loops", lint.LintConfig())
    assert n.code == "I-OVERLAP-NOISE" and n.level == "info"
    old = {"level": "error", "where": None, "what": "repeat 'loops': copies 1 and 2 overlap by 26.7 (units squared)"}
    (o,) = lint.from_legacy([old], "loops", lint.LintConfig())
    assert o.code == "E-OVERLAP" and o.what.startswith("repeat 'loops': copies 1 and 2")


def test_findings_round_trip_and_render_in_the_catalogue_format():
    f = lint.render("error", "E-SHORT-EDGE", "fluid", where=(38.7, 1.5), edge=0.09, x=38.7, y=1.5, w=3.0, floor=0.1)
    assert f.text().startswith("!! ERROR  E-SHORT-EDGE  fluid: edge of 0.09 at (38.7, 1.5)")
    assert "\n          a sliver no cell can sit on" in f.text()
    assert lint.Finding.from_dict(json.loads(json.dumps(f.as_dict()))) == f
    w = lint.render("warn", "W-CUSP", "fluid", deg=3.0, x=14.0, y=1.5)
    assert w.text().startswith("!!        W-CUSP  fluid:")
    i = lint.render("info", "I-CUSP", "fluid", deg=24.0, x=1.0, y=2.0, warn=5.0)
    assert i.text().startswith("check     I-CUSP  fluid:")
    assert lint.has_errors([f, w, i]) and lint.errors([f, w, i]) == [f] and lint.warnings([f, w, i]) == [w]


def test_every_instance_reference_is_row_name_zero_based():
    """D29: no "copies", no "instances k and j", anywhere the model reads; every instance
    reference is `<row>[k]`."""
    assert ALL_FINDINGS, "the fixture tests run first and fill this"
    for template in lint.TEXTS.values():
        assert "copies" not in template and "instances " not in template
    for f in ALL_FINDINGS:
        text = f.subject + " " + f.what + " " + f.fix
        assert "copies" not in text, text
        assert not re.search(r"instances? \d+ and \d+", text), text
        for ref in re.findall(r"\b\w+\[[^\]]*\]", text):
            assert re.fullmatch(r"\w+\[\d+\]", ref) or ref.endswith("[*]"), text


def test_every_why_carries_a_number_and_never_looks_or_seems():
    for key, template in lint.TEXTS.items():
        assert re.search(r"\{[a-zA-Z_]+(:[^}]*)?\}", template), key
        assert not re.search(r"\b(looks|seems|probably)\b", template), key
    for f in ALL_FINDINGS:
        text = f.what + " " + f.fix
        assert re.search(r"\d", text), text
        assert not re.search(r"\b(looks|seems|probably)\b", text), text


def test_thresholds_are_relative():
    """Every LintConfig field is a fraction of w_min or the span, an angle, a count or a
    ratio band; none is an absolute length."""
    source = inspect.getsource(lint.LintConfig)
    for field in dataclasses.fields(lint.LintConfig):
        assert field.name.endswith(("_rel", "_deg", "_samples")) or field.name.startswith("units_"), field.name
    assert not re.search(r"\b\d+(\.\d+)? ?(mm|cm|m|in)\b", source.split('"""', 1)[0])
    assert lint.LintConfig.from_dict(lint.LintConfig().as_dict()) == lint.LintConfig()


def test_judge_orders_errors_first_and_returns_after_disjoint(gm):
    disjoint = [{"level": "error", "code": "E-DISJOINT", "subject": "fluid",
                 "what": "fluid is 2 separate faces: {'main'} and {'loops[3]'} do not touch", "where": (61.2, 1.5)}]
    findings = lint.judge(gm, 0, spec_plan({"ops": []}), None, None, disjoint, None)
    assert [f.code for f in findings] == ["E-DISJOINT"] and findings[0].where == (61.2, 1.5)
    _, _, _, mixed, _, _ = judged(gm, load_spec("tesla_real_attempt3.json"), "ordered")
    levels = [f.level for f in mixed]
    assert levels == sorted(levels, key=lambda lv: {"error": 0, "warn": 1, "info": 2}[lv])
