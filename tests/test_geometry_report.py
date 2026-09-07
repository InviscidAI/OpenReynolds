"""The print-back (DESIGN.md section 6, 3.8): the fixed section order, the verdict computed
from the tables, both units on the extent line, and the exact lap-2 text of section 7.1
reproduced from hand-built Measurements / Findings / Table fixtures with the `[new]`
markers stripped. The design's three placeholders (`<U2 pins it>` twice, the island
extents) are replaced by what the fixtures carry: the passage minimum with its
resolution, the p10 sample, and the island centroids (HoleMeasure has no bounds)."""
from __future__ import annotations

import re

import pytest

from openreynolds.geometry import report
from openreynolds.geometry.claims import ComplianceRow, ComplianceTable, comply, parse
from openreynolds.geometry.lint import Finding

from geometry_t01_fixture import (
    T01_CLAIMS, T01_LEGS, row_fit_finding, t01_findings, t01_measurements, t01_plan,
)

SECTIONS = ["SCRIPT", "LINT", "FEATURES", "LEGS", "MEASURED", "PATCHES", "CLAIMS", "REFERENCE", "VERDICT"]


def t01_table() -> ComplianceTable:
    """The CLAIMS block of 7.1 lap 2 as recorded: the abbreviated `says` are the design's
    own (c6, c7, c11, c12) and c11 carries the gap beside the return angle."""
    def row(cid, says, measured, expected, verdict="pass", kind="measure"):
        return ComplianceRow(id=cid, says=says, kind=kind, expected=expected, measured=measured, verdict=verdict)
    return ComplianceTable([
        row("c1", "a straight channel 3 mm wide", "main.width = 3.000", "3 +/- 0.03"),
        row("c2", "60 mm long", "main.length = 60.00", "60 +/- 0.1"),
        row("c3", "4 bypass loops", "loops.count = 4", "4", kind="count"),
        row("c4", "loop channel 3 mm wide", "loops[*].width = 3.000 (x4)", "3 +/- 0.03"),
        row("c5", "outer radius 6 mm", "loops[*].outer_radius = 6.000 (x4, from the outer arc's curvature)", "6 +/- 0.06"),
        row("c6", "leave ... at a shallow angle",
            "loops[*].leave_angle = 20.0 (x4) built at the leg's wall; lip 28.9 on the arc", "<= 30 deg", kind="predicate"),
        row("c7", "return ... against the forward",
            "heading 260 deg, 100 deg from the flow (+x): component -0.17 (x4)", "> 90 deg", kind="predicate"),
        row("c8", "inlet at x=0", "inlet at (0, 0)", "(0, 0) +/- 0.5", kind="patch"),
        row("c9", "outlet at x=60 mm", "outlet at (60, 0)", "(60, 0) +/- 0.5", kind="patch"),
        row("c10", "sweep round", "loops[*].sweep = 240 deg (x4)", "reported", kind="report"),
        row("c11", "(how steeply it returns ...)",
            "loops[*].return_angle = 80.0 (x4); loops.gap 1.64 (solved), walls 2.44 apart", "reported", kind="report"),
        row("c12", "produce the mesh, checkMesh ...", "not measurable here: the finish and the main agent do these",
            "", verdict="not_measurable", kind="not_measurable"),
    ])


T01_LAP2 = """\
SCRIPT     ran in 1.6 s; 1 note
           note: return_angle 80 keeps 4 loops of outer r 6 inside 60
LINT       clean
           check     I-GAP  Row 'loops': neighbours' walls 2.44 apart (solved gap 1.64 is along the wall's bounding
                     footprint; the nearest points are the return leg's corner and the previous arc)
           check     I-LIP  4 knife-edge lips of 28.9 deg (solid) at (11.42, 1.5) (26.08, 1.5) (40.74, 1.5) (55.40, 1.5): the outer arc
                     meets main.top (leave_length 3 < (w/2)/tan 20 = 4.12, so the arc reaches the wall before the leg's wall does)
           check     I-REFERENCE  passage width 3 (declared by 'main'); p10 of the passage samples 3.000 at (30, 3)
FEATURES   main      Passage    width 3, one leg 60 along +x, start (0, 0) end (60, 0)
           loop      Bypass     R 4.5 (outer 6, inner 3), sweep 240, return heading 260 (against +x), leave_length 3 (default: one width),
                                lands 4.22 upstream of its anchor, lip 4.18 downstream at 28.9 deg, footprint -5.74..+7.28, height 11.25
           loops     Row        4 at pitch 14.66 (13.02 + 1.64 solved), anchors u = 7.24, 21.90, 36.56, 51.22,
                                span 1.50..58.50 on main.top (0..60), centred
LEGS       loops[0] (anchor 7.24; loops[1..3] identical, shifted by 14.66)
             leg 1  line 3     from (7.24, 1.5) heading 20 deg (+x) to (10.06, 2.526)
             leg 2  arc  r 4.5 (outer 6, inner 3) centre (8.52, 6.755) 240 deg left: heading 20 -> 260
             leg 3  to y=1.5   from (4.088, 7.536) heading 260 deg (-x) to (3.024, 1.5)   lands on main.top
MEASURED   extent 60 x 14.25 mm (0.06 x 0.01425 m)   area 513.7 mm2   islands 4   narrowest passage 3.000 (+/- 0.003) at (30, 3)   36 edges, shortest 1.50 at (0.75, 1.5)
           islands at (8.1, 6), (22.76, 6), (37.42, 6), (52.08, 6), one inside each loop
PATCHES    inlet (1 edge, 3.0) at (0, 0) = main.start     outlet (1 edge, 3.0) at (60, 0) = main.end
           walls (34 edges, 294.6)
CLAIMS     12 claims: 9 pass, 0 FAIL, 1 not measurable, 2 reported
           c1   a straight channel 3 mm wide      main.width = 3.000              3 +/- 0.03      pass
           c2   60 mm long                        main.length = 60.00             60 +/- 0.1      pass
           c3   4 bypass loops                    loops.count = 4                 4               pass
           c4   loop channel 3 mm wide            loops[*].width = 3.000 (x4)     3 +/- 0.03      pass
           c5   outer radius 6 mm                 loops[*].outer_radius = 6.000 (x4, from the outer arc's curvature)   6 +/- 0.06   pass
           c6   leave ... at a shallow angle      loops[*].leave_angle = 20.0 (x4) built at the leg's wall; lip 28.9 on the arc   <= 30 deg   pass
           c7   return ... against the forward    heading 260 deg, 100 deg from the flow (+x): component -0.17 (x4)   > 90 deg   pass
           c8   inlet at x=0                      inlet at (0, 0)                 (0, 0) +/- 0.5  pass
           c9   outlet at x=60 mm                 outlet at (60, 0)               (60, 0) +/- 0.5 pass
           c10  sweep round                       loops[*].sweep = 240 deg (x4)   reported        pass
           c11  (how steeply it returns ...)      loops[*].return_angle = 80.0 (x4); loops.gap 1.64 (solved), walls 2.44 apart   reported   pass
           c12  produce the mesh, checkMesh ...   not measurable here: the finish and the main agent do these
REFERENCE  none (no approved library entry matches)
VERDICT    ready to COMMIT"""


def lap2_text(**kw):
    args = dict(printed="", notes=["return_angle 80 keeps 4 loops of outer r 6 inside 60"], seconds=1.6,
                findings=t01_findings(), plan=t01_plan(), m=t01_measurements(), legs=dict(T01_LEGS),
                table=t01_table(), reference=None)
    args.update(kw)
    return report.text(**args)


def test_t01_lap2_print_back_matches_the_design():
    """The exact text of DESIGN.md 7.1 lap 2 from the recorded fixtures, `[new]` markers
    stripped; the design's `<U2 pins it>` placeholders read the fixture's passage numbers
    and the island line prints the centroids HoleMeasure carries."""
    assert lap2_text() == T01_LAP2


def test_the_claims_block_of_comply_matches_the_recorded_rows():
    """`comply` over the same fixtures gives the recorded rows column for column (the
    design's abbreviated `says` and c11's gap aside, which the recorded table carries)."""
    cs, _ = parse(T01_CLAIMS)
    live = comply(cs, t01_measurements(), t01_findings(), t01_plan())
    recorded = t01_table()
    for a, b in zip(live.rows, recorded.rows):
        assert (a.id, a.verdict, a.expected) == (b.id, b.verdict, b.expected)
        if a.id != "c11":
            assert a.measured == b.measured, a.id
    assert live.lines()[0] == recorded.lines()[0]
    text = "\n".join(live.lines())
    assert "c1   a straight channel 3 mm wide      main.width = 3.000              3 +/- 0.03      pass" in text


def test_sections_appear_in_order():
    text = lap2_text()
    positions = [text.index(f"\n{s:<11}") if not text.startswith(f"{s:<11}") else 0 for s in SECTIONS]
    assert positions == sorted(positions)
    heads = [line[:11].rstrip() for line in text.splitlines() if line[:1] != " "]
    assert heads == SECTIONS


def test_verdict_is_computed_from_the_tables():
    assert report.verdict([], t01_table()) == ("ready to COMMIT", True)
    failing = t01_table()
    failing.rows[5].verdict = "fail"
    text, ready = report.verdict([row_fit_finding()], failing)
    assert not ready and text == "not ready: LINT E-ROW-FIT; claims c6 FAIL   (COMMIT disagrees: c6 records it)"
    text, ready = report.verdict([], failing)
    assert not ready and text == "not ready: claims c6 FAIL   (COMMIT disagrees: c6 records it)"
    text, ready = report.verdict([row_fit_finding()], None)
    assert not ready and text == "not ready: LINT E-ROW-FIT"
    text, ready = report.verdict([], None)
    assert not ready and text == "not ready: no claims"
    warn = Finding(level="warn", code="W-GAP", subject="Row 'loops'", what="0.2 apart", where=(1, 1))
    assert report.verdict([warn], t01_table()) == ("ready to COMMIT", True)
    assert report.verdict([], ComplianceTable([])) == ("not ready: no checkable claim", False)


def test_extent_line_carries_both_units():
    m = t01_measurements()
    assert report.extent_line(m) == "extent 60 x 14.25 mm (0.06 x 0.01425 m)"
    m.units, m.scale, m.extent = "cm", 0.01, (38.0, 20.0)
    assert report.extent_line(m) == "extent 38 x 20 cm (0.38 x 0.2 m)"
    assert "extent 60 x 14.25 mm (0.06 x 0.01425 m)" in lap2_text()


def test_the_refusal_alone_under_one_script_line():
    """rc 3, and rc 5 without a partial: SCRIPT and the refusal, nothing else (3.8's one rule)."""
    refusal = ("!! ERROR  E-IMPORT  line 1: `import numpy` -- the script may import math and json only\n"
               "          the API does the arithmetic: footprints, landings and pitches are in the print-back")
    text = report.text("", [], 0.3, [], None, None, {}, None, None, refusal=refusal)
    # the refusal verbatim, in the section-5 format the API wrote it in (3.8: "the whole
    # text is `refusal` under one SCRIPT line"; 4.3: the model sees it verbatim)
    assert text == "SCRIPT     ran in 0.3 s; no prints\n" + refusal
    assert "LINT" not in text and "VERDICT" not in text


def test_a_row_refusal_with_a_partial_prints_features_legs_and_not_evaluated():
    """rc 5 with a partial (7.1 lap 1): the refusal under LINT with its table, FEATURES
    with the item solved and the Row NOT SOLVED, LEGS for the single instance, CLAIMS
    'not evaluated', VERDICT 'not ready: LINT E-ROW-FIT'."""
    plan = t01_plan(notes=())
    loop = plan.features["loop"]
    loop.params["return_angle"] = 45
    loop.solved.update(sweep=205, landing=[-10.34, 0], footprint=[-12.46, 7.28], theta_return=45)
    plan.features["loops"].solved = {}
    plan.features["loops"].params = {"count": 4, "align": "centre"}
    m = t01_measurements()
    legs = {"loop.raw": [
        {"kind": "line", "from": (0, 1.5), "to": (2.82, 2.53), "heading": 20, "length": 3},
        {"kind": "arc", "from": (2.82, 2.53), "to": (-1.90, 9.94), "heading": 20, "heading_out": 225,
         "centre": (1.28, 6.75), "radius": 4.5, "outer_radius": 6, "inner_radius": 3, "sweep": 205},
        {"kind": "to", "from": (-1.90, 9.94), "to": (-10.34, 1.5), "heading": 225, "line": "y:1.5", "lands": "main.top"},
    ]}
    plan.ops = [op for op in plan.ops if op.get("op") != "repeat"]
    text = report.text("", [], 0.3, [row_fit_finding()], plan, m, legs, None, None)
    lines = text.splitlines()
    assert lines[0] == "SCRIPT     ran in 0.3 s; no prints"
    assert lines[1] == "LINT       1 error (errors block)"
    assert lines[2] == "           !! ERROR  E-ROW-FIT  Row 'loops': 4 x Bypass 'loop' do not fit on main.top (60 long)"
    assert lines[3] == "                     each instance spans 19.74 along the wall (12.46 upstream of its anchor to 7.28 downstream)"
    assert lines[6] == "                       return_angle  45    60    70    80    90"
    assert "FEATURES   main      Passage    width 3, one leg 60 along +x, start (0, 0) end (60, 0)" in text
    assert ("           loop      Bypass     solved: R 4.5 (outer 6, inner 3), sweep 205, return heading 225 (-x), "
            "leave_length 3 (default: one width),") in text
    assert "                                lands 10.34 upstream of its anchor, footprint -12.46..+7.28, height 11.25" in text
    assert "           loops     Row        NOT SOLVED (E-ROW-FIT)" in text
    assert "LEGS       loop (one instance, anchor at u = 0 for the table)" in text
    assert "             leg 1  line 3     from (0, 1.5) heading 20 deg (+x) to (2.82, 2.53)" in text
    assert "             leg 3  to y=1.5   from (-1.9, 9.94) heading 225 deg (-x) to (-10.34, 1.5)   lands on main.top" in text
    assert "CLAIMS     not evaluated: the sketch did not build" in text
    assert text.endswith("VERDICT    not ready: LINT E-ROW-FIT")


def test_a_failing_row_carries_its_detail_lines_under_it():
    table = t01_table()
    table.rows[8] = ComplianceRow(
        id="c9", says="leaving the last pass at its right end", kind="patch", expected="right",
        measured="outlet at (0, 18): the LEFT end (x = 0 of 0..38)", verdict="fail",
        detail="an even number of passes ends on the inlet's side; 4 passes with 3 bends cannot end at the right.\n"
               "5 passes (4 bends) or 3 passes (2 bends) end on the right; the request fixes 4 and 3.")
    text = lap2_text(table=table)
    assert ("           c9   leaving the last pass at its right end   outlet at (0, 18): the LEFT end (x = 0 of 0..38)   right   FAIL\n"
            "                an even number of passes ends on the inlet's side; 4 passes with 3 bends cannot end at the right.\n"
            "                5 passes (4 bends) or 3 passes (2 bends) end on the right; the request fixes 4 and 3.") in text
    assert "CLAIMS     12 claims: 8 pass, 1 FAIL, 1 not measurable, 2 reported" in text
    assert text.endswith("VERDICT    not ready: claims c9 FAIL   (COMMIT disagrees: c9 records it)")


def test_lint_counts_errors_and_warnings_and_prints_errors_first():
    warn = Finding(level="warn", code="W-GAP", subject="Row 'loops'", what="loops[0] and loops[1] are 0.2 apart",
                   fix="if the wall is meant, say so at COMMIT (accepts: W-GAP at (x, y) -- why)", where=(14, 1.5))
    text = lap2_text(findings=[warn, row_fit_finding()] + t01_findings(), table=None)
    lines = text.splitlines()
    assert lines[2] == "LINT       1 error, 1 warning (errors block)"
    assert lines[3].startswith("           !! ERROR  E-ROW-FIT")
    i = next(k for k, line in enumerate(lines) if "W-GAP" in line)
    assert lines[i] == "           !!        W-GAP  Row 'loops': loops[0] and loops[1] are 0.2 apart"
    assert lines[i + 1] == "                     if the wall is meant, say so at COMMIT (accepts: W-GAP at (x, y) -- why)"
    assert 3 < i
    only_warn = lap2_text(findings=[warn], table=None)
    assert "LINT       1 warning\n" in only_warn
    assert "CLAIMS     not evaluated: no claims" in only_warn
    assert only_warn.endswith("VERDICT    not ready: no claims")


def test_script_line_counts_prints_and_notes():
    text = report.text("hello\nworld", ["a note"], 2.04, [], None, None, {}, None, None)
    assert text == ("SCRIPT     ran in 2.0 s; 2 prints, 1 note\n"
                    "           > hello\n"
                    "           > world\n"
                    "           note: a note")


def test_leg_lines_name_instances_zero_based():
    loop_only = {"loop.raw": T01_LEGS["loop.raw"]}
    frames = report.leg_frames(t01_plan(), loop_only)
    lines = report.leg_lines(loop_only, frames)
    assert lines[0] == "loops[0] (anchor 7.24; loops[1..3] identical, shifted by 14.66)"
    assert len(lines) == 4
    joined = "\n".join(lines)
    assert not re.search(r"\bcopies\b|instances? \d", joined)
    for ref in re.findall(r"\w+\[[^\]]+\]", joined):
        assert re.fullmatch(r"\w+\[\d+(\.\.\d+)?\]", ref), ref
    plain = report.leg_lines({"snake": T01_LEGS["loop.raw"][:1]})
    assert plain[0] == "snake"


def test_a_to_leg_names_the_wall_it_lands_on_not_the_landing_point():
    """mesh2d's `to` record carries `lands` = the landing point; the print-back names the
    wall from the feature's `solved["lands_on"]` (D28), never the point, and without a plan
    the leg prints with no landing tail rather than a coordinate."""
    frames = report.leg_frames(t01_plan(), T01_LEGS)
    assert frames["loop.raw"]["lands_on"] == "main.top"
    lines = report.leg_lines({"loop.raw": T01_LEGS["loop.raw"]}, frames)
    assert lines[3].endswith("to (3.024, 1.5)   lands on main.top")
    bare = report.leg_lines({"loop.raw": T01_LEGS["loop.raw"]})
    assert bare[3].endswith("to (3.024, 1.5)") and "lands on" not in bare[3]
    assert "(3.024026094486673" not in "\n".join(lines + bare)


def test_a_one_straight_leg_channel_has_no_leg_table_and_a_solved_table_fills_in():
    """`main` (one line leg) is summarised on its FEATURES line and prints no LEGS table
    (7.1 prints only `loops[0]`); a Passage the build recorded no channel for (a mitred
    corner compiles to rects, D36) prints its solved legs, with a corner leg's vertices."""
    text = lap2_text()
    assert "LEGS       loops[0] (anchor 7.24; loops[1..3] identical, shifted by 14.66)" in text
    assert "\n           main\n" not in text and "leg 1  line 60" not in text
    plan = t01_plan()
    from openreynolds.geometry.compile import Solved
    plan.features["duct"] = Solved(kind="Passage", params={"width": 10, "start": [0, 0], "heading": 0},
                                   solved={"legs": [
                                       {"kind": "line", "from": (0, 0), "to": (100, 0), "heading": 0, "length": 100},
                                       {"kind": "corner", "from": (100, 0), "to": (100, 0), "heading": 0, "turn": 90,
                                        "outer": (105, -5), "inner": (95, 5)},
                                       {"kind": "line", "from": (100, 0), "to": (100, 80), "heading": 90, "length": 80}],
                                       "length": 180, "end": [100, 80], "end_heading": 90}, ops=["duct.leg1", "duct.leg2"])
    text = lap2_text(plan=plan)
    assert "           duct\n" in text
    assert "             leg 1  line 100   from (0, 0) heading 0 deg (+x) to (100, 0)" in text
    assert "             leg 2  corner 90 deg left at (100, 0): outer vertex (105, -5), inner vertex (95, 5)" in text
    assert "             leg 3  line 80    from (100, 0) heading 90 deg (+y) to (100, 80)" in text
    assert "           main\n" not in text


def test_reference_lines_show_the_approved_entry_beside_the_candidate():
    from openreynolds.geometry.library import ReferenceMatch
    ref = ReferenceMatch(entry="tesla_valve", preset="t01", approved_at="2026-09-08", outline=[],
                         measurements={"extent": [60, 14.25], "islands": 4,
                                       "params": {"return_angle": 80, "outer_radius": 6, "leave_angle": 20, "count": 4}},
                         hausdorff=0.0)
    text = lap2_text(reference=ref)
    assert ("REFERENCE  library tesla_valve (preset t01, approved 2026-09-08): return_angle 80, outer_radius 6, "
            "leave_angle 20, count 4\n"
            "           reference vs candidate: extent 60 x 14.25 vs 60 x 14.25, islands 4 vs 4; Hausdorff 0") in text


@pytest.mark.parametrize("units, scale, extent, expected", [
    ("mm", 0.001, (300.0, 60.0), "extent 300 x 60 mm (0.3 x 0.06 m)"),
    ("m", 1.0, (0.3, 0.06), "extent 0.3 x 0.06 m (0.3 x 0.06 m)"),
    ("in", 0.0254, (10.0, 2.0), "extent 10 x 2 in (0.254 x 0.0508 m)"),
])
def test_extent_line_in_every_unit(units, scale, extent, expected):
    m = t01_measurements()
    m.units, m.scale, m.extent = units, scale, extent
    assert report.extent_line(m) == expected
