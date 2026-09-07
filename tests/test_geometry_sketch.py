"""`sketch.py`: the authoring API and its closed forms (DESIGN.md 3.3).

Everything here is gmsh-free unless the test says otherwise; the gmsh tests build the
compiled ops through `mesh2d.build_face` by path and read the numbers off the built
face, so a closed form is never pinned against itself. The section-7 numbers (anchor
7.24, pitch 14.66, 36 edges, shortest 1.501, the lip at (11.416, 1.5)) were measured
on this box on 2026-09-07 and are the ones asserted.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from openreynolds.geometry import _toolbox
from openreynolds.geometry import compile as gc
from openreynolds.geometry import sketch as sk
from openreynolds.geometry.sketch import (Bypass, Passage, Rect, Row, Serpentine, Sketch, SketchError,
                                          bypass_closed_form, bypass_self_cross)

import geometry_scripts as scripts

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def fresh_sketch():
    Sketch._reset()
    yield
    Sketch._reset()


@pytest.fixture(scope="module")
def gm():
    gmsh = pytest.importorskip("gmsh")
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    yield gmsh
    if gmsh.isInitialized():
        gmsh.finalize()


def run(script: str) -> Sketch:
    return gc.run_script(script)


def plan_of_t01() -> gc.Plan:
    return gc.plan(run(scripts.T01))


def refusal(fn, code: str) -> SketchError:
    with pytest.raises(SketchError) as err:
        fn()
    assert err.value.code == code, str(err.value)
    return err.value


def built(gm, plan_: gc.Plan, name: str):
    """(face, legs, checks, Built2D) for a plan, in the sketch's units."""
    mesh2d = _toolbox.load("mesh2d")
    gm.model.add(name)
    face, legs, checks = gc.build(gm, plan_)
    short = min(plan_.declared_widths) / 3 if plan_.declared_widths else None
    return face, legs, checks, mesh2d.measure2d(gm, face, short_below=short)


def boundary_vertices(gm, face: int) -> list[tuple[float, float]]:
    seen = {}
    for d, c in gm.model.getBoundary([(2, face)], combined=False, oriented=False):
        for dd, v in gm.model.getBoundary([(1, abs(c))], oriented=False):
            x, y, _ = gm.model.getValue(0, abs(v), [])
            seen[abs(v)] = (x, y)
    return list(seen.values())


# -- the Bypass closed form -------------------------------------------------------------------


AUTHOR_TABLE = {
    # return_angle: (lo, hi, footprint, height, lands upstream by), section 3.3's table
    45: (-12.46, 7.28, 19.74, 11.25, 10.34),
    60: (-8.68, 7.28, 15.96, 11.25, 6.95),
    70: (-7.02, 7.28, 14.30, 11.25, 5.42),
    80: (-5.74, 7.28, 13.02, 11.25, 4.22),
    90: (-4.72, 7.28, 12.00, 11.25, 3.22),
}


def test_bypass_closed_form_matches_the_author_table():
    for angle, (lo, hi, footprint, height, upstream) in AUTHOR_TABLE.items():
        cf = bypass_closed_form(3.0, 20.0, 6.0, float(angle), 3.0)
        assert cf["footprint"][0] == pytest.approx(lo, abs=0.01)
        assert cf["footprint"][1] == pytest.approx(hi, abs=0.01)
        assert cf["footprint"][1] - cf["footprint"][0] == pytest.approx(footprint, abs=0.01)
        assert cf["height"] == pytest.approx(height, abs=0.01)
        assert cf["lands_upstream_by"] == pytest.approx(upstream, abs=0.01)
        assert cf["sweep"] == 180 + angle - 20
    # the record's numbers for the worked T01 (section 4.2)
    cf = bypass_closed_form(3.0, 20.0, 6.0, 80.0, 3.0)
    assert cf["R"] == 4.5 and cf["sweep"] == 240
    assert cf["P1"] == pytest.approx((2.819, 1.026), abs=1e-3)
    assert cf["C"] == pytest.approx((1.28, 5.255), abs=1e-3)
    assert cf["P2"] == pytest.approx((-3.152, 6.036), abs=1e-3)
    assert cf["landing"] == pytest.approx((-4.216, 0), abs=1e-3)
    assert cf["return_len"] == pytest.approx(6.13, abs=1e-2)
    assert cf["lip_u"] == pytest.approx(4.176, abs=1e-3) and cf["lip_deg"] == pytest.approx(28.86, abs=0.01)


def test_bypass_closed_form_matches_the_built_extent(gm):
    """The sampled extent of the built, clipped instance equals the closed-form footprint
    and height at every row of the table, within 1e-3."""
    for angle in AUTHOR_TABLE:
        run(scripts.T01.replace("return_angle=80", f"return_angle={angle}"))
        loop = Sketch.current().features["loop"]
        plan_ = gc.plan_feature(loop)
        face, _, _, _ = built(gm, plan_, f"extent{angle}")
        x0, y0, x1, y1 = gc.sampled_bounds(gm, 2, face)
        lo, hi = plan_.features["loop"].solved["footprint"]
        assert (x0, x1) == pytest.approx((lo, hi), abs=1e-3)
        assert y1 - y0 == pytest.approx(plan_.features["loop"].solved["height"], abs=1e-3)
        assert y0 == pytest.approx(1.5, abs=1e-3)


def test_bypass_refuses_outer_radius_at_or_below_width():
    Sketch(units="mm")
    m = Passage(width=3, start=(0, 0), name="main").line(60)
    err = refusal(lambda: Bypass(wall=m.top, width=3, leave_angle=20, outer_radius=3, return_angle=80, name="loop"),
                  "E-RADIUS")
    text = str(err)
    assert "outer_radius 3 with width 3 leaves an inner radius of 0" in text
    assert "outer_radius > width (inner > 0), and >= 1.5 x width meshes cleanly" in text
    assert err.feature == "Bypass 'loop'"
    refusal(lambda: Bypass(wall=m.top, width=3, leave_angle=20, outer_radius=2.5, return_angle=80), "E-RADIUS")
    Bypass(wall=m.top, width=3, leave_angle=20, outer_radius=4.5, return_angle=80, at=30)


def test_the_arc_end_is_above_the_wall_on_the_whole_angle_domain():
    """P2.v = L sin tL + R cos tL + R cos tR > 0 on (0, 90] x (0, 90] even with no leave
    leg and the smallest arc: no "arc below the wall" refusal exists (rev. 2)."""
    w = 3.0
    for tL in range(1, 91):
        for tR in range(1, 91):
            cf = bypass_closed_form(w, float(tL), w, float(tR), 0.0)
            assert cf["P2"][1] > 0, (tL, tR)


def test_bypass_angles_outside_the_domain_are_refused():
    m = Passage(width=3, start=(0, 0), name="main").line(60)
    err = refusal(lambda: Bypass(wall=m.top, width=3, leave_angle=20, outer_radius=6, return_angle=100), "E-ANGLE")
    assert "return_angle 100 is not in (0, 90]" in str(err)
    assert "90 is straight back, 100 would return with the flow" in str(err)
    refusal(lambda: Bypass(wall=m.top, width=3, leave_angle=0, outer_radius=6, return_angle=80), "E-ANGLE")


def test_a_bare_bypass_off_the_wall_is_refused_at_compile():
    """Attempt 1 of the assessment as a Bypass(at=6): the closed-form landing u = -3.9 < 0."""
    run('''
s = Sketch(units="mm")
main = s.passage(width=3, start=(0, 0), name="main").line(60)
loop = Bypass(wall=main.top, width=3, leave_angle=25, outer_radius=6, return_angle=45, leave_length=5, at=6, name="loop")
s.fluid = main | loop
''')
    err = refusal(lambda: gc.plan(Sketch.current()), "E-LAND-OFF-WALL")
    text = str(err)
    assert "E-LAND-OFF-WALL  Bypass 'loop' at 6 on main.top: it would land at u = -3.93, before the wall's start (0)" in text
    assert "the loop returns 9.93 upstream of its anchor (leave_angle 25, leave_length 5, outer_radius 6, return_angle 45);" in text
    assert "at >= 9.93, or a steeper return_angle (80 returns 3.03 upstream), or a Row, which places it" in text
    assert err.numbers["landing_u"] == pytest.approx(-3.9, abs=0.05)


def test_a_bare_bypass_needs_at():
    run('''
s = Sketch(units="mm")
main = s.passage(width=3, start=(0, 0), name="main").line(60)
loop = Bypass(wall=main.top, width=3, leave_angle=20, outer_radius=6, return_angle=80, name="loop")
s.fluid = main | loop
''')
    err = refusal(lambda: gc.plan(Sketch.current()), "E-BYPASS-AT")
    assert str(err).startswith("!! ERROR  E-BYPASS-AT  Bypass 'loop': outside a Row needs at=<u along main.top, 0 at the "
                               "inlet end, 60 at the outlet end>")
    assert "Bypass 'loop': Bypass 'loop'" not in str(err)


def test_self_cross_is_refused_only_after_a_built_instance_shows_it(gm):
    """The refusal's numbers are read from an instance built through mesh2d: a loop whose
    return leg cuts through its own leave leg encloses two islands where a loop has one,
    and the closed-form crossing point lies inside the built face."""
    mesh2d = _toolbox.load("mesh2d")
    w, tL, outer, tR, L = 3.0, 10.0, 6.0, 90.0, 20.0
    cf = bypass_closed_form(w, tL, outer, tR, L)
    hit = bypass_self_cross(cf, w)
    assert hit is not None and hit[1] > 0
    spec = {"ops": [
        {"op": "channel", "name": "main", "width": w, "start": [0, 0], "heading": 0, "path": [{"line": 60}]},
        {"op": "channel", "name": "raw", "width": w, "start": [30, 1.5], "heading": tL, "from": "y:1.5",
         "path": [{"line": L}, {"arc": {"radius": outer - w / 2, "angle": cf["sweep"]}}, {"line": {"to": "y:1.5"}}]},
        {"op": "rect", "name": "clip", "origin": [-100, 1.5], "size": [300, 100]},
        {"op": "intersect", "name": "loop", "of": ["raw", "clip"]},
        {"op": "fuse", "name": "body", "of": ["main", "loop"]}]}
    ops, _ = mesh2d.parse_spec(spec)
    gm.model.add("selfcross")
    face = mesh2d.build_face(gm, ops, scale=1.0, legs={}, checks=[])
    b = mesh2d.measure2d(gm, face)
    assert b.islands == 2
    assert gm.model.isInside(2, face, [30 + hit[0], 1.5 + hit[1], 0.0]) == 1
    # the same loop as the API refuses it before any build, with those numbers
    main = Passage(width=w, start=(0, 0), name="main").line(60)
    err = refusal(lambda: Bypass(wall=main.top, width=w, leave_angle=tL, outer_radius=outer, return_angle=tR,
                                 leave_length=L, at=30, name="loop"), "E-SELF-CROSS")
    assert err.numbers["cross"] == pytest.approx(hit)
    assert "the return leg's downstream wall crosses the leave leg before reaching the wall" in str(err)
    assert "two islands, not one" in str(err)
    # and the leave_length the refusal offers builds a loop with one island
    ok = err.numbers["leave_length_max"] - 0.5
    Bypass(wall=main.top, width=w, leave_angle=tL, outer_radius=outer, return_angle=tR, leave_length=ok, at=30)
    spec["ops"][1]["path"][0] = {"line": ok}
    ops, _ = mesh2d.parse_spec(spec)
    gm.model.add("selfcross_ok")
    face = mesh2d.build_face(gm, ops, scale=1.0, legs={}, checks=[])
    assert mesh2d.measure2d(gm, face).islands == 1
    # T01's merged mouths are not a self-cross
    assert bypass_self_cross(bypass_closed_form(3.0, 20.0, 6.0, 80.0, 3.0), 3.0) is None


def test_the_lip_closed_form_matches_the_built_vertex(gm):
    """The knife-edge lip is where the OUTER ARC meets the wall (D31): a boundary vertex
    at (anchor + lip_u, 1.5) whose arc leaves the wall at lip_deg -- (11.416, 1.5) for the
    t01_lap2c anchor 7.24 and (11.626, 1.5) for a loop anchored at 7.45, 28.9 deg both."""
    for anchor, x_expected, name in ((None, 11.42, "lip2c"), (7.45, 11.63, "lip2b")):
        if anchor is None:
            run(scripts.T01)
        else:
            run(scripts.T01.replace('return_angle=80, name="loop")', f'return_angle=80, at={anchor}, name="loop")')
               .replace("loops = Row(loop, count=4, name=\"loops\")", "loops = loop"))
        plan_ = gc.plan(Sketch.current())
        solved = plan_.features["loop"].solved
        u0 = plan_.features["loop"].params["at"]
        assert solved["lip_deg"] == pytest.approx(28.9, abs=0.5)
        lip = (u0 + solved["lip_u"], 1.5)
        assert lip[0] == pytest.approx(x_expected, abs=0.05)
        face, _, _, _ = built(gm, plan_, name)
        vertices = boundary_vertices(gm, face)
        nearest = min(vertices, key=lambda p: math.hypot(p[0] - lip[0], p[1] - lip[1]))
        assert nearest == pytest.approx(lip, abs=0.05)
        # the arc at that vertex has curvature 1/6 and leaves the wall at the lip angle
        angle = None
        for d, c in gm.model.getBoundary([(2, face)], combined=False, oriented=False):
            c = abs(c)
            (t0,), (t1,) = gm.model.getParametrizationBounds(1, c)
            for t in (t0, t1):
                x, y, _ = gm.model.getValue(1, c, [t])
                if math.hypot(x - nearest[0], y - nearest[1]) < 1e-6:
                    k = gm.model.getCurvature(1, c, [(t0 + t1) / 2])[0]
                    if abs(k - 1 / 6) < 1e-3:
                        dx, dy, _ = gm.model.getDerivative(1, c, [t])
                        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        assert angle == pytest.approx(28.9, abs=0.5)


# -- rows -------------------------------------------------------------------------------------


E_ROW_FIT_T01 = """!! ERROR  E-ROW-FIT  Row 'loops': 4 x Bypass 'loop' do not fit on main.top (60 long)
          each instance spans 19.74 along the wall (12.46 upstream of its anchor to 7.28 downstream)
          4 x 19.74 = 78.96 needed before any gap; 60 - 2 x margin 1.5 = 57 available
          the footprint is set by return_angle (outer_radius 6, leave_angle 20, leave_length 3 (default: one width) kept):
            return_angle  45    60    70    80    90
            footprint     19.74 15.96 14.30 13.02 12.00
            4 fit at gap  --    --    --    1.64  3.00
          fixed by the request: count 4 (c3), outer_radius 6 (c5), wall 60 (c2); return_angle is free (no claim fixes it)
          a smaller outer_radius, fewer loops, or a longer wall would also fit"""


def test_row_refuses_four_45_degree_loops_on_60_with_the_table():
    run(scripts.T01_LAP1)
    err = refusal(lambda: gc.plan(Sketch.current(), scripts.T01_CLAIMS), "E-ROW-FIT")
    assert str(err) == E_ROW_FIT_T01
    assert err.partial is Sketch.current().features["loop"]
    assert err.numbers["table"] == pytest.approx({"45": 19.74, "60": 15.96, "70": 14.30, "80": 13.02, "90": 12.00}, abs=0.005)
    assert err.numbers["fits"] == {"45": "--", "60": "--", "70": "--", "80": "1.64", "90": "3.00"}
    # with no claims the line says so
    run(scripts.T01_LAP1)
    err = refusal(lambda: gc.plan(Sketch.current()), "E-ROW-FIT")
    assert "fixed by the request: no claims fix the others; return_angle is free" in str(err)
    # attempt 3 of the assessment (a 45-degree return at gap 2) is refused here, never as an overlap
    run(scripts.T01_LAP1.replace('Row(loop, count=4, name="loops")', 'Row(loop, count=4, gap=2, name="loops")'))
    err = refusal(lambda: gc.plan(Sketch.current()), "E-ROW-FIT")
    assert "4 x 19.74 + 3 x gap 2 = 84.96 needed; 60 - 2 x margin 1.5 = 57 available" in str(err)


def test_row_free_parameter_follows_the_claims():
    """The free parameter is the first of (return_angle, outer_radius, leave_length, gap)
    no measure/range claim fixes on the item; the 'fixed by the request' line names the
    claim ids it read."""
    claims = {"claims": [c for c in scripts.T01_CLAIMS["claims"] if c["id"] != "c5"]
              + [{"id": "c13", "kind": "measure", "says": "returning at 45 degrees", "measure": "return_angle",
                  "of": "loops[*]", "value": 45}]}
    run(scripts.T01_LAP1)
    err = refusal(lambda: gc.plan(Sketch.current(), claims), "E-ROW-FIT")
    text = str(err)
    assert err.numbers["free_parameter"] == "outer_radius"
    assert "the footprint is set by outer_radius (leave_angle 20, leave_length 3 (default: one width), return_angle 45 kept):" in text
    assert re.search(r"outer_radius\s+3\.6\s+4\.2\s+4\.8\s+5\.4\s+6", text)
    assert "fixed by the request: count 4 (c3), return_angle 45 (c13), wall 60 (c2); outer_radius is free (no claim fixes it)" in text
    # with the T01 claims as they are plus the return-angle claim, both angle and radius are fixed: leave_length is free
    claims = {"claims": scripts.T01_CLAIMS["claims"] + claims["claims"][-1:]}
    run(scripts.T01_LAP1)
    err = refusal(lambda: gc.plan(Sketch.current(), claims), "E-ROW-FIT")
    assert err.numbers["free_parameter"] == "leave_length"
    assert "fixed by the request: count 4 (c3), outer_radius 6 (c5), return_angle 45 (c13), wall 60 (c2); leave_length is free" in str(err)


def test_row_default_gap_is_the_largest_that_fits():
    """Return 80 on 60: gap (57 - 4 x 13.02) / 3 = 1.64, pitch 14.66, the row centred with
    anchors 7.24 + k x 14.66 (closed form, re-measured on t01_lap2c)."""
    run(scripts.T01)
    plan_ = gc.plan(Sketch.current())
    row = plan_.features["loops"].solved
    assert row["gap"] == pytest.approx(1.64, abs=0.005) and row["gap_declared"] is False
    assert row["pitch"] == pytest.approx(14.66, abs=0.005)
    assert row["anchors"] == pytest.approx([7.24, 21.90, 36.56, 51.22], abs=0.005)
    assert row["span"] == pytest.approx([1.5, 58.5], abs=1e-9) and row["margin"] == 1.5
    assert row["along"] == "main.top" and row["free_parameter"] == "return_angle"
    assert plan_.instances["loops"] == {"count": 4, "step": pytest.approx([14.66, 0], abs=0.005),
                                        "gap": pytest.approx(1.64, abs=0.005), "gap_declared": False,
                                        "footprint": pytest.approx([13.02, 11.25], abs=0.005)}


def test_row_pitch_with_gap_below_footprint_plus_gap_is_refused():
    run(scripts.T01.replace('Row(loop, count=4, name="loops")', 'Row(loop, count=4, gap=1.0, pitch=13.0, name="loops")'))
    err = refusal(lambda: gc.plan(Sketch.current()), "E-ROW-PITCH")
    assert "Row 'loops': pitch 13 is smaller than footprint 13.02 + gap 1 = 14.02" in str(err)
    assert "pitch >= 14.02, or gap <= -0.02, or leave pitch out and the gap sets it" in str(err)
    assert err.partial is Sketch.current().features["loop"]


def test_row_pitch_alone_is_accepted():
    """Pitch 13.0 on a 13.02 footprint: a bbox is conservative (the accepted study sat
    there with its walls 0.8 apart), so the lint on the build judges it, not the solver."""
    run(scripts.T01.replace('Row(loop, count=4, name="loops")', 'Row(loop, count=4, pitch=13.0, name="loops")'))
    plan_ = gc.plan(Sketch.current())
    row = plan_.features["loops"].solved
    assert row["pitch"] == 13.0 and row["gap_declared"] is False
    assert row["gap"] == pytest.approx(13.0 - 13.02, abs=0.005)
    assert next(op for op in plan_.ops if op["op"] == "repeat")["step"] == [13.0, 0.0]


def test_row_fit_is_checked_at_plan_time_not_construction():
    """`Row(...)` then `main.line(20)`: the wall is 60 at plan time, so four 80-degree loops
    fit; on the 40 the Row was constructed against they would not."""
    script = scripts.T01.replace(".line(60)", ".line(40)")
    run(script)
    refusal(lambda: gc.plan(Sketch.current()), "E-ROW-FIT")
    run(script + "\nmain.line(20)\n")
    plan_ = gc.plan(Sketch.current())
    assert plan_.features["loops"].solved["span"] == pytest.approx([1.5, 58.5])
    assert plan_.features["main"].solved["length"] == 60
    assert plan_.edge_targets["main.end"]["points"][0][0] == pytest.approx(60)


def test_row_item_with_at_is_refused():
    run(scripts.T01.replace('return_angle=80, name="loop")', 'return_angle=80, at=6, name="loop")'))
    err = refusal(lambda: gc.plan(Sketch.current()), "E-ROW-ITEM-AT")
    assert "Row 'loops': its item Bypass 'loop' gives at=6, but the Row places its instances" in str(err)
    assert "drop at= from the Bypass; Row(start=6) fixes the first anchor" in str(err)


def test_row_along_a_direction_needs_a_gap_and_takes_no_start():
    base = '''
s = Sketch(units="mm")
duct = s.rect(origin=(0, 0), size=(100, 20), name="duct")
pin = s.disk(centre=(10, 5), radius=1, name="pin")
pins = Row(pin, count=4, along=(1, 0), {extra}name="pins")
s.fluid = duct - pins
s.inlet(duct.left)
s.outlet(duct.right)
'''
    run(base.format(extra=""))
    err = refusal(lambda: gc.plan(Sketch.current()), "E-ROW-GAP-REQUIRED")
    assert "Row 'pins' along (1, 0): a row along a direction has no wall to fit; give gap=" in str(err)
    run(base.format(extra="gap=2, start=5, "))
    err = refusal(lambda: gc.plan(Sketch.current()), "E-ROW-DIRECTION")
    assert "Row 'pins' along (1, 0): start/align/margin are placed against a wall; a row along a" in str(err)
    assert "direction is placed by its item (the first instance is the Disk as drawn at (10, 5))" in str(err)
    run(base.format(extra="gap=2, "))
    plan_ = gc.plan(Sketch.current())
    row = plan_.features["pins"].solved
    assert row["pitch"] == pytest.approx(4.0) and row["footprint"] == pytest.approx([9, 11])
    assert plan_.instances["pins"]["step"] == pytest.approx([4.0, 0.0])
    assert [k for k in plan_.outlines if k.startswith("pins[")] == ["pins[0]", "pins[1]", "pins[2]", "pins[3]"]
    # a row of disks along a wall is placed against it and needs no gap
    run(base.replace("along=(1, 0), ", "along=duct.top, ").format(extra=""))
    plan_ = gc.plan(Sketch.current())
    assert plan_.features["pins"].solved["gap"] == pytest.approx((100 - 4 * 2) / 3)
    run(base.replace("along=(1, 0), ", "").format(extra=""))
    refusal(lambda: gc.plan(Sketch.current()), "E-ROW-ALONG")


def test_a_row_of_disks_along_a_wall_sits_inside_the_wall_span():
    """An item that is not a Bypass is anchored at its footprint's centre along the wall:
    four r 1 disks on a 100 wall with the gap solved sit at 0..2, ..., 98..100 (not at the
    disk's drawn x = 10 shifted by a Bypass-style anchor); align and start place the
    footprint, never the drawn position."""
    base = '''
s = Sketch(units="mm")
duct = s.rect(origin=(0, 0), size=(100, 20), name="duct")
pin = s.disk(centre=(10, 5), radius=1, name="pin")
pins = Row(pin, count=4, along=duct.top, {extra}name="pins")
s.fluid = duct - pins
s.inlet(duct.left)
s.outlet(duct.right)
'''

    def spans(plan_):
        return [(min(q[0] for q in plan_.outlines[k][0]), max(q[0] for q in plan_.outlines[k][0]))
                for k in ("pins[0]", "pins[3]")]

    plan_ = gc.plan(run(base.format(extra="")))
    row = plan_.features["pins"].solved
    assert row["anchors"] == pytest.approx([1, 33.667, 66.333, 99], abs=0.005)
    assert row["span"] == pytest.approx([0, 100], abs=1e-9)
    assert spans(plan_) == [pytest.approx((0, 2), abs=1e-9), pytest.approx((98, 100), abs=1e-9)]
    assert next(op for op in plan_.ops if op["name"] == "pins.first")["by"] == pytest.approx([-9, 0])
    plan_ = gc.plan(run(base.format(extra='gap=2, align="start", ')))
    assert plan_.features["pins"].solved["anchors"] == pytest.approx([1, 5, 9, 13])
    assert spans(plan_) == [pytest.approx((0, 2)), pytest.approx((12, 14))]
    plan_ = gc.plan(run(base.format(extra='gap=2, align="end", ')))
    assert spans(plan_) == [pytest.approx((86, 88)), pytest.approx((98, 100))]
    plan_ = gc.plan(run(base.format(extra="gap=2, start=30, ")))
    assert plan_.features["pins"].solved["anchors"] == pytest.approx([30, 34, 38, 42])
    assert spans(plan_) == [pytest.approx((29, 31)), pytest.approx((41, 43))]
    # every instance lies within the wall's span, whatever the placement
    for extra in ("", 'gap=2, align="start", ', 'gap=2, align="end", ', "gap=2, start=30, "):
        plan_ = gc.plan(run(base.format(extra=extra)))
        for k in range(4):
            xs = [q[0] for q in plan_.outlines[f"pins[{k}]"][0]]
            assert -1e-9 <= min(xs) and max(xs) <= 100 + 1e-9, (extra, k)
    # a row of a non-Bypass item that does not fit says where the item is drawn, not "upstream of its anchor"
    run(base.replace("size=(100, 20)", "size=(10, 20)").replace("radius=1", "radius=2").format(extra=""))
    err = refusal(lambda: gc.plan(Sketch.current()), "E-ROW-FIT")
    assert "Row 'pins': 4 x Disk 'pin' do not fit on duct.top (10 long)" in str(err)
    assert "each instance spans 4 along the wall (drawn at u 8..12)" in str(err)
    assert "4 x 4 = 16 needed before any gap; 10 - 2 x margin 0 = 10 available" in str(err)
    assert err.partial is Sketch.current().features["pin"]


# -- passages -------------------------------------------------------------------------------


def test_passage_mutates_and_returns_self():
    Sketch(units="mm")
    p = Passage(width=3, start=(0, 0), name="p")
    p.line(30)
    p.line(30)
    assert len(p.legs) == 2 and p.length == 60
    q = p.arc(radius=5, turn=90)
    assert q is p and len(p.legs) == 3
    assert p.end_point == pytest.approx((65, 5)) and p.end_heading == 90


def test_an_empty_passage_is_refused():
    run('s = Sketch(units="mm")\npath = s.passage(width=3, start=(0, 0), name="path")\ns.fluid = path\n')
    err = refusal(lambda: gc.plan(Sketch.current()), "E-EMPTY-PASSAGE")
    assert "Passage 'path' has no legs" in str(err)
    assert ".line(), .arc(), .turn(), .line_to(), .turn_to(), .u_turn(), .end_on() add legs to the passage in place: path.line(30)" in str(err)


def test_turn_to_picks_the_shorter_side_and_refuses_180():
    Sketch(units="mm")
    p = Passage(width=8, start=(0, 0), heading=0, name="u").line(60)
    p.turn_to(heading=-90, radius=10)
    assert p.legs[-1]["arc"]["angle"] == -90
    p.turn_to(heading=45, radius=10)
    assert p.legs[-1]["arc"]["angle"] == pytest.approx(135)
    err = refusal(lambda: p.turn_to(heading=225, radius=10), "E-TURN-AMBIGUOUS")
    assert "heading 45 -> 225 is a U-turn either way; there is no shorter side" in str(err)
    assert 'turn_to(heading=225, radius=10, side="left") or u_turn(radius=10, side="left") (left = the +y side for a +x heading)' in str(err)
    p.turn_to(heading=225, radius=10, side="right")
    assert p.legs[-1]["arc"]["angle"] == -180


def test_a_final_line_to_is_an_open_end_not_a_landing():
    run(scripts.T04)
    u = Sketch.current().features["u"]
    assert u.legs == [{"line": 60}, {"arc": {"radius": 10, "angle": 180}}, {"line": 60}]
    assert u.lands_on is None and u.leaves is None
    assert u.end_point == pytest.approx((0, 20)) and u.end_heading == 180
    plan_ = gc.plan(Sketch.current())
    channel = next(op for op in plan_.ops if op["op"] == "channel")
    assert not any(isinstance(leg.get("line"), dict) for leg in channel["path"])
    assert plan_.features["u"].solved["lands_on"] is None
    assert [r["kind"] for r in plan_.features["u"].solved["legs"]] == ["line", "arc", "line"]
    # moving away from the line, or running along it, is E-LINE-TO
    Sketch(units="mm")
    p = Passage(width=8, start=(60, 0), heading=0, name="u")
    err = refusal(lambda: p.line_to(x=0), "E-LINE-TO")
    assert "Passage 'u': line_to(x=0) from (60, 0) heading 0 moves away from x = 0" in str(err)
    assert "the heading is +x; turn first, or line_to(x=120)" in str(err)
    refusal(lambda: p.line_to(y=5), "E-LINE-TO")


def test_end_on_emits_the_only_to_leg():
    run('''
s = Sketch(units="mm")
main = s.passage(width=3, start=(0, 0), name="main").line(60)
branch = Passage(width=2, start=(30, 10), heading=-90, name="branch").end_on(main.bottom)
s.fluid = main | branch
s.inlet(main.start)
s.outlet(main.end)
''')
    branch = Sketch.current().features["branch"]
    assert branch.legs == [{"line": {"to": "y:-1.5"}}]
    assert str(branch.lands_on) == "main.bottom"
    assert branch.end_point == pytest.approx((30, -1.5))
    plan_ = gc.plan(Sketch.current())
    ops = [op for op in plan_.ops if op["op"] == "channel" and op["name"] == "branch"]
    assert ops[0]["path"] == [{"line": {"to": "y:-1.5"}}] and "from" not in ops[0]
    solved = plan_.features["branch"].solved
    assert solved["lands_on"] == "main.bottom" and solved["legs"][0]["kind"] == "to"
    assert solved["legs"][0]["lands"] == pytest.approx((30, -1.5))
    refusal(lambda: branch.line(5), "E-LEG")


def test_a_branch_from_a_wall_emits_from_and_no_notch(gm):
    run('''
s = Sketch(units="mm")
main = s.passage(width=3, start=(0, 0), name="main").line(60)
branch = Passage(width=2, start=(main.top, 30), name="branch").line(10)
s.fluid = main | branch
s.inlet(main.start)
s.outlet(main.end)
s.outlet(branch.end, name="side")
''')
    branch = Sketch.current().features["branch"]
    assert branch.start_point == pytest.approx((30, 1.5)) and branch.heading == 90
    plan_ = gc.plan(Sketch.current())
    op = next(op for op in plan_.ops if op["name"] == "branch")
    assert op["from"] == "y:1.5" and op["start"] == [30, 1.5] and op["heading"] == 90
    assert plan_.features["branch"].solved["leaves"] == "main.top"
    face, legs, checks, b = built(gm, plan_, "branch")
    # a cap poking through the wall leaves a short edge; a flush cut leaves none
    assert len(b.curves) == 8 and min(b.lengths.values()) == pytest.approx(2.0)
    assert b.area == pytest.approx(180 + 20)
    assert legs["branch"][0]["from_line"] == "y=1.5"


def test_turn_is_a_mitre():
    run(scripts.T03)
    duct = Sketch.current().features["duct"]
    records = duct.records
    assert [r["kind"] for r in records] == ["line", "corner", "line"]
    corner = records[1]
    assert corner["deg"] == 90 and corner["extend"] == pytest.approx(5)
    assert corner["outer_vertex"] == pytest.approx((105, -5)) and corner["inner_vertex"] == pytest.approx((95, 5))
    assert duct.end_point == pytest.approx((100, 80)) and duct.length == 180
    plan_ = gc.plan(Sketch.current())
    channels = [op for op in plan_.ops if op["op"] == "channel"]
    assert channels[0]["path"] == [{"line": 105}] and channels[0]["start"] == [0, 0]
    assert channels[1]["path"] == [{"line": 85}] and channels[1]["start"] == pytest.approx([100, -5]) and channels[1]["heading"] == 90
    assert next(op for op in plan_.ops if op["op"] == "fuse")["of"] == ["duct.1", "duct.2"]
    corners = sorted(tuple(p) for p in plan_.outlines["duct"][0])
    assert [x for p in corners for x in p] == pytest.approx(
        [x for p in sorted([(0, -5), (105, -5), (105, 80), (95, 80), (95, 5), (0, 5)]) for x in p])
    # a corner needs a line on both sides
    Sketch(units="mm")
    p = Passage(width=10, start=(0, 0), name="duct").line(100).arc(radius=20, turn=45)
    err = refusal(lambda: p.turn(90), "E-CORNER")
    assert "turn(90) needs a line leg before it and after it; leg 2 is an arc" in str(err)
    p = Passage(width=10, start=(0, 0), name="duct").line(100).turn(90)
    refusal(lambda: p.arc(radius=20, turn=45), "E-CORNER")
    refusal(lambda: p.check(), "E-CORNER")


def test_rect_width_is_the_shorter_side():
    Sketch(units="mm")
    r = Rect(origin=(0, 0), size=(100, 10), name="r")
    assert r.width == 10 and r.length == 100 and r.size_x == 100 and r.size_y == 10
    assert r.centre == (50, 5)
    plan_ = gc.plan_feature(r)
    assert plan_.features["r"].solved["width"] == 10 and plan_.features["r"].solved["length"] == 100


def test_center_is_an_alias_of_centre():
    s = Sketch(units="mm")
    d = s.disk(center=(100, 30), diameter=10, name="cyl")
    assert d.centre == (100, 30) and d.radius == 5
    r = s.rect(center=(150, 30), size=(300, 60))
    assert r.origin == (0, 0) and r.centre == (150, 30)
    b = s.annulus(center=(0, 0), r_inner=1, r_outer=2)
    assert b.centre == (0, 0)
    with pytest.raises(SketchError):
        s.disk(centre=(0, 0), center=(0, 0), radius=1)
    plan_ = gc.plan_feature(d)
    assert plan_.features["cyl"].params["centre"] == [100, 30] and "center" not in plan_.features["cyl"].params


def test_through_solves_tangent_lengths_and_refuses_a_radius_that_does_not_fit():
    Sketch(units="mm")
    p = Passage.through(width=4, points=[(0, 0), (70, 0), (70, 20), (0, 20)], radius=10, name="bend")
    assert p.legs == [{"line": 60}, {"arc": {"radius": 10, "angle": 90}}, {"arc": {"radius": 10, "angle": 90}},
                      {"line": 60}] or p.legs == [{"line": 60}, {"arc": {"radius": 10, "angle": 90}},
                                                   {"line": pytest.approx(0, abs=1e-9)},
                                                   {"arc": {"radius": 10, "angle": 90}}, {"line": 60}]
    assert p.end_point == pytest.approx((0, 20)) and p.end_heading == pytest.approx(180)
    err = refusal(lambda: Passage.through(width=4, points=[(0, 0), (70, 0), (70, 20), (0, 20)], radius=12), "E-FILLET")
    assert "corner 2 at (70, 0): radius 12 needs 12 of straight on" in str(err)
    assert "(tan 45 deg)" in str(err) and "(70, 0)->(70, 20) is 20 long and corner 3 takes 12 of it" in str(err)
    assert "leaving -4" in str(err)
    assert err.numbers["leaving"] == pytest.approx(-4)
    # a rounding-level shortfall is not refused
    Passage.through(width=4, points=[(0, 0), (70, 0), (70, 20 - 1e-9), (0, 20)], radius=10)


def test_serpentine_refuses_bend_radius_that_overlaps_passes():
    Sketch(units="mm")
    err = refusal(lambda: Serpentine(width=2, passes=4, pass_length=30, bend_radius=0.8, name="snake"), "E-SERP-RADIUS")
    text = str(err)
    assert "Serpentine 'snake': bend_radius 0.8 puts the passes 1.6 apart centre to" in text
    assert "centre, but they are 2 wide: they overlap by 0.4. bend_radius > 1 (walls touching), >= 1.5" in text
    assert "leaves a 1 wall between passes" in text
    refusal(lambda: Serpentine(width=2, passes=4, pass_length=30, bend_radius=1.0), "E-SERP-RADIUS")
    Serpentine(width=2, passes=4, pass_length=30, bend_radius=1.01)


def test_serpentine_even_passes_end_on_the_start_side():
    Sketch(units="mm")
    for passes, same in ((4, True), (3, False), (5, False)):
        snake = Serpentine(width=2, passes=passes, pass_length=30, bend_radius=3, name=f"s{passes}")
        assert snake.ends_on_start_side is same
        assert snake.end_point[0] == pytest.approx(0 if same else 30)
        assert snake.bends == passes - 1 and snake.pass_pitch == 6 and snake.wall_between == 4
    snake = Serpentine(width=2, passes=4, pass_length=30, bend_radius=3, name="snake")
    assert snake.end_point == pytest.approx((0, 18)) and snake.end_heading == pytest.approx(180)
    assert snake.pass_centrelines == pytest.approx([0, 6, 12, 18])
    assert [leg["arc"]["angle"] for leg in snake.legs if "arc" in leg] == [180, -180, 180]
    assert snake.side("top", leg=1).span == 30


def test_a_rect_side_needs_a_flow_to_host_a_bypass():
    script = '''
s = Sketch(units="mm")
duct = s.rect(origin=(0, 0), size=(60, 3), name="duct")
loop = Bypass(wall=duct.top, width=3, leave_angle=20, outer_radius=6, return_angle=80, at=30, name="loop"{flow})
s.fluid = duct | loop
{ports}
'''
    run(script.format(flow="", ports=""))
    err = refusal(lambda: gc.plan(Sketch.current()), "E-FLOW")
    assert "Bypass 'loop': which way does the flow run along duct.top?" in str(err)
    assert 'duct is a Rect; declare s.inlet(duct.left) and s.outlet(duct.right), or give flow="+x"' in str(err)
    run(script.format(flow="", ports="s.inlet(duct.left)\ns.outlet(duct.right)"))
    plan_ = gc.plan(Sketch.current())
    assert plan_.features["loop"].solved["wall_line"]["flow"] == [1, 0]
    run(script.format(flow=', flow="-x"', ports=""))
    plan_ = gc.plan(Sketch.current())
    raw = next(op for op in plan_.ops if op["name"] == "loop.raw")
    assert plan_.features["loop"].solved["wall_line"]["flow"] == [-1, 0]
    assert raw["heading"] == pytest.approx(160) and raw["path"][1]["arc"]["angle"] == -240


def test_passage_top_on_a_two_leg_passage_is_ambiguous():
    Sketch(units="mm")
    u = Passage(width=8, start=(0, 0), name="u").line(60).arc(radius=10, turn=180).line(60)
    err = refusal(lambda: u.top, "E-SIDE-AMBIGUOUS")
    assert "Passage 'u': .top names the only straight leg, but 'u' has 2 straight legs" in str(err)
    assert 'u.side("top", leg=1) or u.side("top", leg=3)' in str(err)
    top1 = u.side("top", leg=1)
    assert top1.outward == pytest.approx((0, 1)) and top1.flow == pytest.approx((1, 0)) and top1.span == 60
    top3 = u.side("top", leg=3)
    assert top3.outward == pytest.approx((0, 1)) and top3.flow == pytest.approx((-1, 0))
    refusal(lambda: u.side("top", leg=2), "E-SIDE-AMBIGUOUS")
    # two collinear line legs are one straight wall
    main = Passage(width=3, start=(0, 0), name="main").line(40).line(20)
    assert main.top.span == 60


def test_mirrored_compiles_in_closed_form_and_emits_no_mirror_op(gm):
    """The valve mirrored across y = 0: no `mirror` op, the loops hang below the main
    with the arc turning right; the built face is the mirror image of the unmirrored
    one (same area, bounds reflected), measured against mesh2d's own mirror op."""
    mesh2d = _toolbox.load("mesh2d")
    run(scripts.T01 + "\ns.fluid = (main | loops).mirrored('y', at=0)\n")
    plan_ = gc.plan(Sketch.current())
    assert not any(op["op"] == "mirror" for op in plan_.ops)
    raw = next(op for op in plan_.ops if op["name"] == "loop.raw")
    assert raw["heading"] == pytest.approx(-20) and raw["from"] == "y:-1.5" and raw["path"][1]["arc"]["angle"] == -240
    assert raw["path"][2] == {"line": {"to": "y:-1.5"}}
    assert plan_.features["loop"].solved["wall_line"]["outward"] == [0, -1]
    face, _, _, b = built(gm, plan_, "mirrored")
    x0, y0, x1, y1 = gc.sampled_bounds(gm, 2, face)
    _, _, _, plain = built(gm, plan_of_t01(), "unmirrored")
    assert b.area == pytest.approx(plain.area, rel=1e-6) and len(b.curves) == len(plain.curves) == 36
    # the same shape through the grammar's mirror op
    gm.model.add("mirror_op")
    ops, _ = mesh2d.parse_spec({"ops": scripts.T01_PROBE["ops"] + [
        {"op": "mirror", "name": "mirrored", "target": "fluid", "axis": "y", "at": 0}]})
    face2 = mesh2d.build_face(gm, ops, scale=1.0)
    b2 = mesh2d.measure2d(gm, face2)
    assert b.area == pytest.approx(b2.area, rel=1e-3) and b.islands == b2.islands == 4
    assert (x0, y0, x1, y1) == pytest.approx(gc.sampled_bounds(gm, 2, face2), abs=1e-3)
    assert y1 == pytest.approx(1.5) and y0 == pytest.approx(-12.75, abs=0.01)
    # the edge targets and outlines moved with it (a 64-point arc misses its apex by < 0.01)
    assert sorted(p[1] for p in plan_.edge_targets["main.start"]["points"]) == pytest.approx([-1.5, 1.5])
    assert min(p[1] for p in plan_.outlines["loop"][0]) == pytest.approx(-12.75, abs=0.01)
    # mirroring across x reverses the flow along the wall, so the loop still heads against it
    run(scripts.T01 + "\ns.fluid = (main | loops).mirrored('x', at=30)\n")
    plan_ = gc.plan(Sketch.current())
    raw = next(op for op in plan_.ops if op["name"] == "loop.raw")
    assert raw["heading"] == pytest.approx(160) and raw["path"][1]["arc"]["angle"] == -240
    assert plan_.features["loop"].solved["wall_line"]["flow"] == [-1, 0]


def test_two_rows_are_apart_by_default_and_apart_is_recorded():
    run(scripts.T01 + '''
lower = Bypass(wall=main.bottom, width=3, leave_angle=20, outer_radius=6, return_angle=80, name="lower")
lowers = Row(lower, count=4, name="lowers")
baffle = s.rect(origin=(40, 20), size=(5, 5), name="baffle")
s.fluid = main | loops | lowers | baffle
s.apart(loops, baffle, gap=1.0)
''')
    plan_ = gc.plan(Sketch.current())
    assert ("loops", "lowers", None) in plan_.apart
    assert ("baffle", "loops", 1.0) in plan_.apart
    assert ("baffle", "lowers", None) in plan_.apart
    assert not any({a, b} == {"loops", "main"} for a, b, _ in plan_.apart)
    assert not any({a, b} == {"loops", "loop"} for a, b, _ in plan_.apart)
    assert len(plan_.instances) == 2


def test_a_bare_bypass_and_a_row_are_apart_by_default():
    """D35: a Row and any named feature that is not its host are apart by default -- a
    bare Bypass on the other wall included; a Row's own item (the template its instances
    repeat) is not a part beside them."""
    run(scripts.T01 + '''
extra = Bypass(wall=main.bottom, width=3, leave_angle=20, outer_radius=6, return_angle=80, at=30, name="extra")
s.fluid = main | loops | extra
''')
    plan_ = gc.plan(Sketch.current())
    assert plan_.apart == [("extra", "loops", None)]


def test_the_fluid_must_be_assigned_and_the_sketch_be_one():
    run('s = Sketch(units="mm")\nmain = s.passage(width=3, start=(0, 0)).line(60)\n')
    err = refusal(lambda: gc.plan(Sketch.current()), "E-NO-FLUID")
    assert "the script made a Sketch but never assigned s.fluid" in str(err)
    Sketch._reset()
    refusal(Sketch.current, "E-NO-SKETCH")
    Sketch(units="mm")
    Sketch(units="mm")
    err = refusal(Sketch.only, "E-MANY-SKETCHES")
    assert "the script made 2 Sketch objects; exactly one" in str(err)


# -- the reference card ------------------------------------------------------------------------


def test_api_summary_equals_reference_md():
    card = sk.api_summary()
    on_disk = (ROOT / "openreynolds" / "geometry" / "reference.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert on_disk == card
    assert 80 <= len(card.splitlines()) <= 160
    for name in sk.API:
        if name != "math":
            assert name in card
    for word in ("x:min", "near:x,y", "normal:-x", "box:x0,y0,x1,y1", "E-ROW-FIT", "counter-clockwise from +x"):
        assert word in card


SHAPE_WORDS = ("tesla", "t-junction", "t junction", "venturi", "nozzle", "backward step", "backward-facing",
               "cylinder in", "aerofoil", "airfoil", "manifold", "spiral", "penne", "ahmed", "naca", "pin-fin",
               "bent pipe", "bluff body")
"""Invariant 1's list (section 8.2) minus `serpentine`, which is the class's own name and a
library entry with a golden of its own."""


def test_api_summary_names_no_shape():
    readable = (sk.api_summary() + (sk.__doc__ or "")
                + "".join(getattr(sk, n).__doc__ or "" for n in sk.API if hasattr(getattr(sk, n), "__doc__"))).lower()
    assert not [w for w in SHAPE_WORDS if w in readable]
