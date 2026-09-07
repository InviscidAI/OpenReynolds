"""`geometry.measure`: the walk, the bounds, the rays, the ports and every number read
off a built face (DESIGN.md 3.5).

Every fixture is ops-grammar JSON built through `mesh2d.build_face` by path at scale 1
(D21, D26); the sidecar plans in `geometry_specs` stand in for what `compile.plan` will
write. The gmsh facts the module rests on are pinned here strictly, so a gmsh upgrade
that changes one is noticed rather than silently relied on.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from openreynolds.geometry import _toolbox, lint, measure
from openreynolds.geometry.sketch import PortIntent

from geometry_specs import FIXTURES, build, bypass_plan, load_spec, run, spec_plan

gmsh = pytest.importorskip("gmsh")


@pytest.fixture(scope="module")
def gm():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    yield gmsh
    gmsh.finalize()


def _rect_minus_disk(gm, name="rect_minus_disk"):
    gm.model.add(name)
    occ = gm.model.occ
    r = occ.addRectangle(0, 0, 0, 60, 20)
    d = occ.addDisk(30, 10, 0, 5, 5)
    (face,) = [t for _, t in occ.cut([(2, r)], [(2, d)])[0]]
    occ.synchronize()
    return face


# -- the walk and the bounds ----------------------------------------------------------


def test_sampled_bounds_equal_get_bounding_box_at_scale_1(gm):
    """The lone loop (t01b1): 13.019 x 11.637 sampled equals `getBoundingBox` within 1e-3
    at scale 1; the looseness lives only after the dilate (test_toolbox_mesh2d, edit 1),
    and the toolbox's copy of the function reads the same numbers."""
    face, _, _ = build(gm, load_spec("t01b1.json"), "t01b1")
    sb = measure.sampled_bounds(gm, 2, face)
    x0, y0, _, x1, y1, _ = gm.model.getBoundingBox(2, face)
    assert (sb[2] - sb[0], sb[3] - sb[1]) == pytest.approx((13.019, 11.637), abs=2e-3)
    assert sb == pytest.approx((x0, y0, x1, y1), abs=1e-3)
    assert _toolbox.load("mesh2d").sampled_bounds(gm, 2, face) == pytest.approx(sb, abs=1e-12)


def test_walk_has_the_fluid_on_the_left(gm):
    face = _rect_minus_disk(gm)
    wk = measure.walk(gm, face, 5.0)
    assert len(wk.loops) == 2 and wk.repaired is False
    assert wk.signed_area == pytest.approx(60 * 20, rel=1e-6), "the outer loop, positive"
    assert wk.loop_area(1) == pytest.approx(-math.pi * 25, rel=3e-3), "the hole, negative (64 chords)"
    hole = wk.curves[wk.loops[1][0]]
    assert hole.kind == "arc" and hole.loop == 1
    # fluid on the left: the hole's inward normal at its midpoint points away from the centre
    away = (hole.midpoint[0] - 30, hole.midpoint[1] - 10)
    assert hole.inward_normal[0] * away[0] + hole.inward_normal[1] * away[1] > 0


def test_walk_is_repaired_after_a_mirror(gm):
    gm.model.add("mirrored_rect")
    occ = gm.model.occ
    r = occ.addRectangle(0, 0, 0, 60, 3)
    occ.mirror([(2, r)], 0, 1, 0, 0)
    occ.synchronize()
    wk = measure.walk(gm, r, 3.0)
    assert wk.repaired is True
    assert wk.signed_area == pytest.approx(180.0)
    assert [v.kind for v in wk.vertices] == ["convex"] * 4


def test_is_inside_is_not_trusted_after_a_scale_or_mirror(gm):
    """D26, pinned strictly: `isInside` answers 1 for a point 2 cm and 5 m outside a
    60 x 3 rect dilated by 0.001, and for 20 mm outside a mirrored one, while a native
    rect and a translated-and-rotated one answer 0. The day this fails, gmsh fixed it
    and D26 can be relaxed on evidence."""
    occ = gm.model.occ
    gm.model.add("dilated")
    r = occ.addRectangle(0, 0, 0, 60, 3)
    occ.synchronize()
    assert gm.model.isInside(2, r, [80, 1.5, 0]) == 0
    occ.dilate([(2, r)], 0, 0, 0, 0.001, 0.001, 0.001)
    occ.synchronize()
    assert gm.model.isInside(2, r, [0.08, 0.0015, 0]) == 1
    assert gm.model.isInside(2, r, [5.0, 0.0015, 0]) == 1
    gm.model.add("mirrored")
    r = occ.addRectangle(0, 0, 0, 60, 3)
    occ.mirror([(2, r)], 0, 1, 0, 0)
    occ.synchronize()
    assert gm.model.isInside(2, r, [80, -1.5, 0]) == 1
    gm.model.add("moved")
    r = occ.addRectangle(0, 0, 0, 60, 3)
    occ.translate([(2, r)], 5, 5, 0)
    occ.rotate([(2, r)], 0, 0, 0, 0, 0, 1, 0.3)
    occ.synchronize()
    assert gm.model.isInside(2, r, [200, 200, 0]) == 0


def test_the_kernel_state_canary_refuses_a_dilated_face(gm):
    gm.model.add("canary")
    occ = gm.model.occ
    r = occ.addRectangle(0, 0, 0, 60, 3)
    occ.synchronize()
    wk = measure.walk(gm, r, 3.0)
    assert lint.kernel_state(gm, r, wk) is None
    occ.dilate([(2, r)], 0, 0, 0, 0.001, 0.001, 0.001)
    occ.synchronize()
    wk = measure.walk(gm, r, 0.003)
    finding = lint.kernel_state(gm, r, wk)
    assert finding is not None and finding.code == "E-KERNEL-STATE" and finding.level == "error"


def test_an_arc_sweep_is_arc_length_over_radius(gm):
    """The chord turning of 64 samples reads 236.2 for a 240-degree arc (one chord's
    share short); the sweep is the length over the curvature radius, signed by the turn."""
    face, _, _ = build(gm, load_spec("t01b1.json"), "t01b1_sweep")
    wk = measure.walk(gm, face, 3.0)
    arcs = [c for c in wk.curves.values() if c.kind == "arc"]
    # OCC splits each arc at its circle's seam (angle 0 lies inside the -70..170 sweep),
    # so the 240 degrees are the sum over the pieces of one radius
    for radius, sign in ((6.0, 1), (3.0, -1)):
        pieces = [c for c in arcs if c.radius == pytest.approx(radius, rel=1e-6)]
        assert pieces and sum(abs(c.sweep) for c in pieces) == pytest.approx(240.0, abs=0.01), radius
        # the outer arc turns left on the outer loop; the inner arc bounds the island,
        # which the walk rounds clockwise (fluid on the left), so it turns right
        assert all(c.sweep * sign > 0 for c in pieces), radius


# -- holes, open ends, rays ------------------------------------------------------------------


def test_a_disk_edge_reads_its_radius_from_curvature(gm):
    _, wk, m, _, _, _ = run(gm, load_spec("t05.json"), model="t05_radius")
    assert len(m.holes) == 1
    assert m.holes[0].radius == pytest.approx(5.0, abs=1e-6)
    assert m.holes[0].centroid == pytest.approx((100.0, 30.0), abs=1e-3)
    assert m.features["cyl"]["diameter"] == pytest.approx(10.0, abs=1e-6)
    assert m.features["cyl"]["circumference"] == pytest.approx(10 * math.pi, rel=1e-6)


def test_open_ends_on_the_u_duct_are_both_minus_x(gm):
    face, _, _ = build(gm, load_spec("t04.json"), "t04_ends")
    wk = measure.walk(gm, face, 8.0)
    ends = sorted(measure.open_ends(gm, wk, 8.0, None), key=lambda e: e.centre[1])
    assert len(ends) == 2
    assert ends[0].centre == pytest.approx((0.0, 0.0), abs=1e-6) and ends[1].centre == pytest.approx((0.0, 20.0), abs=1e-6)
    assert all(e.outward_normal == pytest.approx((-1.0, 0.0), abs=1e-9) for e in ends)
    assert all(e.length == pytest.approx(8.0) and e.depth >= 8.0 for e in ends)


def test_open_ends_on_an_l_duct_made_of_two_rects(gm):
    face, _, _ = build(gm, load_spec("lduct2d.json"), "lduct_ends")
    wk = measure.walk(gm, face, 10.0)
    ends = sorted(measure.open_ends(gm, wk, 10.0, None), key=lambda e: e.centre[0])
    assert len(ends) == 2
    assert ends[0].centre == pytest.approx((0.0, 5.0), abs=1e-6) and ends[1].centre == pytest.approx((95.0, 80.0), abs=1e-6)
    assert ends[0].outward_normal == pytest.approx((-1.0, 0.0), abs=1e-9)
    assert ends[1].outward_normal == pytest.approx((0.0, 1.0), abs=1e-9)


def test_a_dead_end_is_a_candidate_and_a_pocket_is_not(gm):
    """A capped branch 10 deep has fluid behind its cap beyond its own 3 (a candidate;
    the rules decide); a recess 6 wide and 0.8 deep in the wall of a 3 mm channel has
    3.8 of fluid behind its bottom edge, less than the edge, so it is a pocket."""
    spec = {"ops": [
        {"op": "rect", "name": "main", "origin": [0, -1.5], "size": [60, 3]},
        {"op": "channel", "name": "dead", "width": 3, "start": [20, 1.5], "heading": 90, "from": "y:1.5",
         "path": [{"line": 10}]},
        {"op": "rect", "name": "pocket", "origin": [37, 1.5], "size": [6, 0.8]},
        {"op": "fuse", "name": "body", "of": ["main", "dead", "pocket"]}]}
    face, _, _ = build(gm, spec, "dead_end")
    wk = measure.walk(gm, face, 3.0)
    ends = measure.open_ends(gm, wk, 3.0, None)
    centres = sorted(e.centre for e in ends)
    assert (20.0, 11.5) in [tuple(round(v, 6) for v in c) for c in centres], "the dead-end cap is a candidate"
    assert not any(abs(c[0] - 40.0) < 1e-6 and abs(c[1] - 2.3) < 1e-6 for c in centres), "a shallow pocket is not"


def test_ray_depth_bisects_to_the_resolution(gm):
    occ = gm.model.occ
    gm.model.add("ray")
    r = occ.addRectangle(0, 0, 0, 60, 3)
    occ.synchronize()
    depth = measure.ray_depth(gm, r, (30.0, 3.0), (0.0, -1.0), 9.0, 18)
    assert depth == pytest.approx(3.0, abs=0.003), "the coarse march alone reads 3.5"
    gm.model.add("sliver")
    r = occ.addRectangle(0, 0, 0, 60, 3)
    s = occ.addRectangle(20, 3.43, 0, 10, 3)
    bridge = occ.addRectangle(28, 3, 0, 2, 0.43)
    out, _ = occ.fuse([(2, r)], [(2, s), (2, bridge)])
    occ.synchronize()
    (face,) = [t for d, t in out if d == 2]
    solid = measure.ray_depth(gm, face, (23.0, 3.0), (0.0, 1.0), 3.0, 12, until="inside", resolution=1e-3)
    assert solid == pytest.approx(0.43, abs=0.003)
    assert measure.ray_depth(gm, face, (5.0, 3.0), (0.0, 1.0), 3.0, 12, until="inside") == 3.0


def test_passage_widths_on_t01_and_t02(gm):
    """Pinned 2026-09-07 from the first run of the bisecting implementation: T01's 3 mm
    channel reads 2.999, T02's 2 mm reads 1.999 (resolution 1e-3 w), where the coarse
    march read 3.5 and 2.33."""
    _, _, m1, _, _, _ = run(gm, load_spec("t01_lap2c.json"), model="t01_widths")
    assert m1.passage.min == pytest.approx(3.0, abs=0.004)
    assert m1.passage.p10 == pytest.approx(3.0, abs=0.004)
    assert m1.passage.n > 300
    _, _, m2, _, _, _ = run(gm, load_spec("t02.json"), model="t02_widths")
    assert m2.passage.min == pytest.approx(2.0, abs=0.004)
    assert m2.passage.median == pytest.approx(2.0, abs=0.004)
    assert m2.features["fluid"]["narrowest_passage"] == m2.passage.min


# -- junctions, lips, cusps ----------------------------------------------------------------


def test_junction_angles_on_t01(gm):
    spec = load_spec("t01_lap2b.json")
    _, wk, m, _, _, _ = run(gm, spec, plan=bypass_plan(spec, "loop"), model="t01_lap2b")
    leave = next(j for j in m.junctions if j.channel == "loops[0]" and j.kind == "leave")
    ret = next(j for j in m.junctions if j.channel == "loops[0]" and j.kind == "return")
    assert leave.measured_angle == pytest.approx(20.0, abs=0.5)
    assert ret.measured_angle == pytest.approx(80.0, abs=0.5)
    assert leave.lip is not None
    assert leave.lip[0] == pytest.approx((11.63, 1.5), abs=0.05)
    assert leave.lip[1] == pytest.approx(28.9, abs=0.5)
    assert ret.heading == pytest.approx(260.0, abs=0.5) and ret.against_flow is True
    # the two mouths of a loop merge into one opening: the return mouth's downstream corner
    # and the leave mouth's upstream corner are interior points of the wall line, not vertices
    on_line = sorted(v.at[0] for v in wk.vertices if abs(v.at[1] - 1.5) < 1e-6 and v.at[0] < 14)
    downstream_of_return = 7.45 - 4.216 + 1.5 / math.sin(math.radians(80))
    upstream_of_leave = 7.45 - 1.5 / math.sin(math.radians(20))
    assert not any(abs(x - downstream_of_return) < 0.05 for x in on_line)
    assert not any(abs(x - upstream_of_leave) < 0.05 for x in on_line)


def test_no_fluid_cusp_on_t01_and_four_lips(gm):
    _, wk, m, _, _, _ = run(gm, load_spec("t01_lap2c.json"), model="t01_cusps")
    kinds = [v.kind for v in wk.vertices]
    assert kinds.count("fluid_cusp") == 0
    assert kinds.count("solid_lip") == 4
    lips = [v for v in wk.vertices if v.kind == "solid_lip"]
    assert all(v.solid_deg == pytest.approx(28.9, abs=0.5) for v in lips)
    assert all(v.interior_deg + v.solid_deg == pytest.approx(360.0) for v in wk.vertices)
    assert all(200 < v.interior_deg < 340 for v in wk.vertices if v.kind == "reflex")


# -- ports ----------------------------------------------------------------------------------


def _circle(cx, cy, r, n=64):
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


INTENTS = {
    "t01_lap2c.json": ([PortIntent("inlet", "inlet", ("main.start",)), PortIntent("outlet", "outlet", ("main.end",))],
                       {"main.start": {"kind": "cap", "points": [(0, -1.5), (0, 1.5)], "width": 3},
                        "main.end": {"kind": "cap", "points": [(60, -1.5), (60, 1.5)], "width": 3}}),
    "t02.json": ([PortIntent("inlet", "inlet", ("snake.start",)), PortIntent("outlet", "outlet", ("snake.end",))],
                 {"snake.start": {"kind": "cap", "points": [(0, -1), (0, 1)], "width": 2},
                  "snake.end": {"kind": "cap", "points": [(0, 17), (0, 19)], "width": 2}}),
    "t04.json": ([PortIntent("inlet", "inlet", ("u.start",)), PortIntent("outlet", "outlet", ("u.end",))],
                 {"u.start": {"kind": "cap", "points": [(0, -4), (0, 4)], "width": 8},
                  "u.end": {"kind": "cap", "points": [(0, 16), (0, 24)], "width": 8}}),
    "t05.json": ([PortIntent("inlet", "inlet", ("duct.left",)), PortIntent("outlet", "outlet", ("duct.right",)),
                  PortIntent("cylinder", "wall", ("cyl.edge",)), PortIntent("walls", "wall", ("duct.top", "duct.bottom"))],
                 {"duct.left": {"kind": "side", "points": [(0, 0), (0, 60)]},
                  "duct.right": {"kind": "side", "points": [(300, 0), (300, 60)]},
                  "duct.top": {"kind": "side", "points": [(0, 60), (300, 60)]},
                  "duct.bottom": {"kind": "side", "points": [(0, 0), (300, 0)]},
                  "cyl.edge": {"kind": "loop", "points": _circle(100, 30, 5)}}),
    "lduct2d.json": ([PortIntent("inlet", "inlet", ("leg1.left",)), PortIntent("outlet", "outlet", ("leg2.top",))],
                     {"leg1.left": {"kind": "side", "points": [(0, 0), (0, 10)]},
                      "leg2.top": {"kind": "side", "points": [(90, 80), (100, 80)]}}),
}
RULES = {"lduct2d.json": [{"name": "inlet", "at": "x:min"}, {"name": "outlet", "at": "y:max"}]}


@pytest.mark.parametrize("name", sorted(INTENTS))
def test_resolve_ports_by_intent_on_the_five_examples(gm, name):
    """inlet / outlet / cylinder / walls resolve through `edge_targets` alone to the same
    curves the probe specs' rule strings name, and the resolved rules for the record are
    `near:` at a point ON each curve."""
    spec = load_spec(name)
    by_rules = dict(spec, patches=RULES.get(name, spec.get("patches", [])))
    face, _, _ = build(gm, by_rules, f"ports_{name}")
    wk = measure.walk(gm, face, 3.0)
    ends = measure.open_ends(gm, wk, min(spec_plan(spec).declared_widths or [3.0]), None)
    expected, _, findings = measure.resolve_ports(gm, wk, spec_plan(by_rules), ends)
    assert not findings
    ports, targets = INTENTS[name]
    plan = spec_plan(spec, rules=[], ports=ports, edge_targets=targets)
    got, resolved, findings = measure.resolve_ports(gm, wk, plan, ends)
    assert not findings, [f.text() for f in findings]
    assert got == expected
    for rule in resolved:
        axis, where, (px, py) = _toolbox.load("mesh2d").parse_where(rule["at"])
        assert where == "near"
        on_curve = min(math.hypot(*(gm.model.getClosestPoint(1, c, [px, py, 0])[0][:2] - __import__("numpy").array([px, py])))
                       for c in wk.curves)
        assert on_curve < 1e-6 * wk.span(), "the resolved point lies ON its curve, not on a chord of it"


def test_a_port_intent_that_matches_no_curve_is_e_port_missing(gm):
    spec = load_spec("t01_lap2c.json")
    face, _, _ = build(gm, spec, "port_missing")
    wk = measure.walk(gm, face, 3.0)
    ends = measure.open_ends(gm, wk, 3.0, None)
    plan = spec_plan(spec, rules=[], ports=[PortIntent("inlet", "inlet", ("loop.start",)),
                                            PortIntent("outlet", "outlet", ("main.end",))],
                     edge_targets={"loop.start": {"kind": "cap", "points": [(7.24, 0), (7.24, 3)], "width": 3},
                                   "main.end": {"kind": "cap", "points": [(60, -1.5), (60, 1.5)], "width": 3}})
    _, _, findings = measure.resolve_ports(gm, wk, plan, ends)
    assert [f.code for f in findings] == ["E-PORT-MISSING", "E-PORT-COUNT", "E-PORT-OPEN"]
    assert findings[0].subject == "inlet" and findings[0].level == "error"
    assert findings[0].numbers["n_ends"] == 2
    assert findings[2].where == pytest.approx((0.0, 0.0), abs=1e-6), "the end nothing names is reported too"


def test_normal_rule_resolves_one_end_or_refuses(gm):
    spec = load_spec("t04.json")
    face, _, _ = build(gm, spec, "normal_rule")
    wk = measure.walk(gm, face, 8.0)
    ends = measure.open_ends(gm, wk, 8.0, None)
    both = spec_plan(spec, rules=[{"name": "inlet", "kind": "inlet", "at": "near:0,0"},
                                  {"name": "outlet", "kind": "outlet", "at": "normal:-x"}])
    _, _, findings = measure.resolve_ports(gm, wk, both, ends)
    rule = [f for f in findings if f.code == "E-PORT-RULE"]
    assert len(rule) == 1 and rule[0].numbers["n"] == 2
    spec2 = load_spec("elbow2d.json")
    face2, _, _ = build(gm, spec2, "normal_rule_one")
    wk2 = measure.walk(gm, face2, 10.0)
    ends2 = measure.open_ends(gm, wk2, 10.0, None)
    one = spec_plan(spec2, rules=[{"name": "inlet", "kind": "inlet", "at": "normal:-x"},
                                  {"name": "outlet", "kind": "outlet", "at": "normal:+y"}])
    curve_patch, _, findings = measure.resolve_ports(gm, wk2, one, ends2)
    assert not findings
    assert sorted(set(curve_patch.values())) == ["inlet", "outlet", "walls"]


# -- the reference width and the Bypass measures -----------------------------------------------


def test_reference_width_order(gm):
    _, _, m, _, _, _ = run(gm, load_spec("t01_lap2c.json"), model="ref_declared")
    assert (m.reference_width, m.reference_width_from) == (3.0, "declared by 'main'")
    claims = {"unit": "mm", "claims": [{"id": "c1", "kind": "measure", "measure": "width", "of": "duct", "value": 12}]}
    ref = measure.reference_width(spec_plan({"ops": []}, declared_widths=[]), claims, [], (0, 0, 300, 60))
    assert ref == (12.0, "claim c1")
    _, _, m5, _, _, _ = run(gm, load_spec("t05.json"), model="ref_hole")
    assert (m5.reference_width, m5.reference_width_from) == (pytest.approx(10.0, abs=1e-6), "hole diameter (cyl)")
    bare = {"scale": 0.001, "ops": [{"op": "rect", "name": "body", "origin": [0, 0], "size": [50, 8]}]}
    _, _, mb, _, _, _ = run(gm, bare, model="ref_bare")
    assert (mb.reference_width, mb.reference_width_from) == (pytest.approx(0.8), "extent / 10")


def test_bypass_measures_come_from_the_walk(gm):
    """The 4025 valve's typed outer radius is 4 (2.5 + 1.5) and the walk agrees; a
    perturbed typed value in the sidecar moves the `typed` column and nothing else."""
    from openreynolds.geometry.compile import Solved
    spec = load_spec("tesla_committed_4025.json")
    sidecar = spec_plan(spec, features={"bypass": Solved(kind="Bypass", params={"outer_radius": 6, "leave_angle": 20},
                                                          solved={}, ops=["bypass"])})
    _, _, m, _, _, _ = run(gm, spec, plan=sidecar, model="4025_walk")
    inst = m.features["bypasses[0]"]
    assert inst["outer_radius"] == pytest.approx(4.0, abs=1e-6)
    assert inst["inner_radius"] == pytest.approx(1.0, abs=1e-6)
    assert inst["leave_angle"] == pytest.approx(60.0, abs=0.5)
    assert inst["return_heading"] == pytest.approx(300.0, abs=0.5) and inst["against_flow"] is False
    assert inst["sweep"] == pytest.approx(240.0, abs=0.5)
    assert inst["typed"]["outer_radius"] == 6 and inst["typed"]["leave_angle"] == 20
    every = m.features["bypasses[*]"]
    assert every["outer_radius"] == pytest.approx([4.0] * 4, abs=1e-6)


def test_t01_bypass_measures_match_the_request(gm):
    spec = load_spec("t01_lap2c.json")
    _, _, m, _, _, _ = run(gm, spec, plan=bypass_plan(spec, "loop"), model="t01_measures")
    inst = m.features["loops[0]"]
    assert inst["outer_radius"] == pytest.approx(6.0, abs=1e-6)
    assert inst["leave_angle"] == pytest.approx(20.0, abs=0.5)
    assert inst["return_angle"] == pytest.approx(80.0, abs=0.5)
    assert inst["return_heading"] == pytest.approx(260.0, abs=0.5) and inst["against_flow"] is True
    assert inst["lip_angle"] == pytest.approx(28.9, abs=0.5)
    assert inst["lands_upstream_by"] == pytest.approx(4.22, abs=0.02)
    assert inst["footprint"] == pytest.approx((-5.74, 7.28), abs=0.02)
    assert inst["height"] == pytest.approx(11.25, abs=0.02)
    assert m.rows["loops"].count == 4 and m.features["loops"]["count"] == 4
    assert m.extent == pytest.approx((60.0, 14.25), abs=0.01)
    assert m.area == pytest.approx(513.6, abs=0.2) and m.islands == 4 and m.n_curves == 36
    assert m.shortest_edge[0] == pytest.approx(1.50, abs=0.01)
    assert m.patches["inlet"]["n"] == 1 and m.patches["outlet"]["n"] == 1 and m.patches["walls"]["n"] == 34
    assert m.flow == pytest.approx((1.0, 0.0), abs=1e-9)


def test_measurements_round_trip_through_json(gm):
    spec = load_spec("t01_lap2c.json")
    _, _, m, _, _, _ = run(gm, spec, plan=bypass_plan(spec, "loop"), model="t01_roundtrip")
    back = measure.Measurements.from_dict(json.loads(json.dumps(m.as_dict())))
    assert json.loads(json.dumps(back.as_dict())) == json.loads(json.dumps(m.as_dict()))
    assert back.junctions[0].measured_angle == m.junctions[0].measured_angle
    assert back.passage.min == m.passage.min and back.holes[0].area == m.holes[0].area


def test_the_fixture_table_is_complete():
    names = {p.name for p in FIXTURES.glob("*.json")}
    for name in ("tesla_real_attempt3", "two_rows_overlap", "row_touch", "row_gap", "tesla_real_attempt1",
                 "landing_half", "landing_short", "landing_capped", "t04", "notch_cap60", "notch_return", "sliver",
                 "tesla_committed_4025", "lduct2d", "manifold2d", "uduct2d", "elbow2d", "tesla_t01_accepted",
                 "t01_lap2c", "disk_with_hole_external", "cup_external", "throat", "t05", "tangent_walls",
                 "mirror_last", "t01b1", "t01_lap2b", "t02", "straight", "elbow", "lduct", "uduct", "penne"):
        assert f"{name}.json" in names, name
    assert (FIXTURES / "claims" / "T01.json").exists()
    assert (Path(__file__).parent / "geometry_specs.py").exists()
