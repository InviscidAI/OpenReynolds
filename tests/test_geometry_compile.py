"""`compile.py`: the plan, its ops, the edge targets and the build canary (DESIGN.md 3.4).

The worked scripts of section 7 compile to the probe ops the document measured, the
compiled ops build through `mesh2d.build_face` to the numbers section 7.1 states (36
edges, shortest 1.501, walls 2.437 apart, the lip vertex at (11.416, 1.5)), the record
round-trips and recompiles to the same ops, and the two interfaces other units read --
`edge_targets` covering every port intent and the listed `Solved.solved` keys -- are
pinned here (section 9's "interfaces that must not drift").
"""
from __future__ import annotations

import json
import math
import re

import pytest

from openreynolds.geometry import _toolbox
from openreynolds.geometry import compile as gc
from openreynolds.geometry.sketch import Sketch, SketchError

import geometry_scripts as scripts


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


def plan_of(script: str, claims=None) -> gc.Plan:
    return gc.plan(gc.run_script(script), claims)


def by_name(plan_: gc.Plan, name: str) -> dict:
    return next(op for op in plan_.ops if op["name"] == name)


def built(gm, plan_: gc.Plan, name: str):
    mesh2d = _toolbox.load("mesh2d")
    gm.model.add(name)
    face, legs, checks = gc.build(gm, plan_)
    short = min(plan_.declared_widths) / 3 if plan_.declared_widths else None
    return face, legs, checks, mesh2d.measure2d(gm, face, short_below=short)


VARIANTS = {
    "mirrored": scripts.T01 + "\ns.fluid = (main | loops).mirrored('y', at=0)\n",
    "moved": scripts.T01 + "\ns.fluid = (main | loops).moved(10, 5)\n",
    "rotated": scripts.T01 + "\ns.fluid = (main | loops).rotated(30, about=(0, 0))\n",
}


# -- the worked examples compile to their probe ops ---------------------------------------------


def test_t01_compiles_to_the_probe_ops():
    """Section 7.1's lap-2 script -> `t01_lap2c.json` up to op names: anchor 7.24, step
    14.66, the loop clipped to the outward side of the wall."""
    plan_ = plan_of(scripts.T01)
    assert [op["op"] for op in plan_.ops] == ["channel", "channel", "rect", "intersect", "repeat", "fuse"]
    assert [op["name"] for op in plan_.ops] == ["main", "loop.raw", "loop.clip", "loop", "loops", "fluid"]
    probe = {op["name"]: op for op in scripts.T01_PROBE["ops"]}
    assert by_name(plan_, "main") == {"op": "channel", "name": "main", "width": 3, "start": [0, 0], "heading": 0,
                                      "path": [{"line": 60}]}
    raw = by_name(plan_, "loop.raw")
    assert raw["start"] == pytest.approx(probe["loop.raw"]["start"], abs=0.005)
    assert raw["heading"] == 20 and raw["from"] == "y:1.5" and raw["width"] == 3
    assert raw["path"] == [{"line": 3}, {"arc": {"radius": 4.5, "angle": 240}}, {"line": {"to": "y:1.5"}}]
    clip = by_name(plan_, "loop.clip")
    x0, y0 = clip["origin"]
    x1, y1 = x0 + clip["size"][0], y0 + clip["size"][1]
    assert y0 == 1.5 and x0 <= -13.0 and x1 >= 73.0 and y1 >= 1.5 + 11.25
    assert by_name(plan_, "loop") == {"op": "intersect", "name": "loop", "of": ["loop.raw", "loop.clip"]}
    repeat = by_name(plan_, "loops")
    assert repeat["target"] == "loop" and repeat["count"] == 4
    assert repeat["step"] == pytest.approx([14.66, 0], abs=0.005)
    assert by_name(plan_, "fluid") == {"op": "fuse", "name": "fluid", "of": ["main", "loops"]}
    assert plan_.rules == [{"name": "inlet", "kind": "inlet", "at": "near:0,0"},
                           {"name": "outlet", "kind": "outlet", "at": "near:60,0"}]
    assert plan_.units == "mm" and plan_.scale == 0.001
    assert plan_.declared_widths == [3, 3]
    assert plan_.notes == ["return_angle 80 keeps 4 loops of outer r 6 inside 60"]
    assert plan_.expected == (1, 1)
    assert [k for k in plan_.outlines] == ["main", "loop", "loops[0]", "loops[1]", "loops[2]", "loops[3]"]
    assert [name for name, _ in plan_.clip_boxes] == ["loop", "loops[0]", "loops[1]", "loops[2]", "loops[3]"]


def test_t02_t03_t04_t05_compile_to_their_probe_ops():
    t02 = plan_of(scripts.T02)
    assert [op["op"] for op in t02.ops] == ["channel", "translate"]
    snake = by_name(t02, "snake")
    assert snake["start"] == [0, 0] and snake["heading"] == 0 and snake["width"] == 2
    assert snake["path"] == [{"line": 30}, {"arc": {"radius": 3, "angle": 180}}, {"line": 30},
                             {"arc": {"radius": 3, "angle": -180}}, {"line": 30}, {"arc": {"radius": 3, "angle": 180}},
                             {"line": 30}]
    assert by_name(t02, "fluid") == {"op": "translate", "name": "fluid", "target": "snake", "by": [0, 0]}
    assert t02.rules == [{"name": "inlet", "kind": "inlet", "at": "near:0,0"},
                         {"name": "outlet", "kind": "outlet", "at": "near:0,18"}]
    assert t02.features["snake"].solved["end"] == pytest.approx([0, 18])

    t03 = plan_of(scripts.T03)
    assert [(op["op"], op["name"]) for op in t03.ops] == [("channel", "duct.1"), ("channel", "duct.2"), ("fuse", "duct"),
                                                          ("translate", "fluid")]
    assert by_name(t03, "duct.1")["path"] == [{"line": 105}]
    assert by_name(t03, "duct.2")["start"] == pytest.approx([100, -5]) and by_name(t03, "duct.2")["path"] == [{"line": 85}]
    assert t03.rules == [{"name": "inlet", "kind": "inlet", "at": "near:0,0"},
                         {"name": "outlet", "kind": "outlet", "at": "near:100,80"}]
    assert [r["kind"] for r in t03.features["duct"].solved["legs"]] == ["line", "corner", "line"]

    t04 = plan_of(scripts.T04)
    t04r = plan_of(scripts.T04_RELATIVE)
    assert t04.ops == t04r.ops and t04.rules == t04r.rules
    assert by_name(t04, "u")["path"] == [{"line": 60}, {"arc": {"radius": 10, "angle": 180}}, {"line": 60}]
    assert t04.rules == [{"name": "inlet", "kind": "inlet", "at": "near:0,0"},
                         {"name": "outlet", "kind": "outlet", "at": "near:0,20"}]

    t05 = plan_of(scripts.T05)
    assert t05.ops == [{"op": "rect", "name": "duct", "origin": [0, 0], "size": [300, 60], "round": 0},
                       {"op": "disk", "name": "cyl", "center": [100, 30], "radius": 5},
                       {"op": "cut", "name": "fluid", "from": "duct", "take": ["cyl"]}]
    names = [(r["name"], r["kind"]) for r in t05.rules]
    assert names == [("inlet", "inlet"), ("outlet", "outlet"), ("cylinder", "wall"), ("walls", "wall"), ("walls", "wall")]
    assert t05.rules[2]["at"] == "near:95,30"
    inlet = t05.rules[0]["box"]
    assert inlet[0] < 0 < inlet[2] < 1e-3 and inlet[1] < 0 and inlet[3] > 60
    assert t05.features["cyl"].solved == {"centre": [100, 30], "radius": 5, "diameter": 10}
    assert t05.features["duct"].solved["width"] == 60 and t05.features["duct"].solved["length"] == 300


def test_compiled_ops_parse_through_mesh2d_unchanged():
    """`mesh2d.parse_spec` accepts every worked example's record as a spec (it reads
    `ops` and `patches` and ignores the rest)."""
    mesh2d = _toolbox.load("mesh2d")
    for name, script in scripts.WORKED.items():
        sketch = gc.run_script(script)
        plan_ = gc.plan(sketch)
        record = json.loads(json.dumps(gc.record(sketch, plan_, script, None, [], None, plan_.rules)))
        ops, rules = mesh2d.parse_spec(record)
        assert [op["name"] for op in ops] == [op["name"] for op in plan_.ops], name
        assert mesh2d.body_name(ops) == "fluid"
        assert [r["name"] for r in rules] == [r["name"] for r in plan_.rules]
        assert mesh2d.parse_spec({"ops": record["ops"], "patches": record["patches"]})[0] == ops


def test_bypass_always_emits_from_and_clip():
    """Finding G: the leave leg is cut flush at the wall and the loop clipped to the
    wall's outward side, whatever the angles."""
    for angle in (45, 60, 80, 90):
        for leave in (10, 20, 45):
            plan_ = plan_of(scripts.T01.replace("return_angle=80", f"return_angle={angle}")
                            .replace("leave_angle=20", f"leave_angle={leave}")
                            .replace("count=4", "count=2"))
            raw = by_name(plan_, "loop.raw")
            assert raw["from"] == "y:1.5" and raw["path"][-1] == {"line": {"to": "y:1.5"}}
            clip = by_name(plan_, "loop.clip")
            assert clip["op"] == "rect" and clip["origin"][1] == 1.5
            assert by_name(plan_, "loop") == {"op": "intersect", "name": "loop", "of": ["loop.raw", "loop.clip"]}
            assert plan_.features["loop"].ops == ["loop.raw", "loop.clip", "loop"]


def test_the_recorded_script_recompiles_to_the_recorded_ops():
    for name, script in list(scripts.WORKED.items()) + list(VARIANTS.items()):
        sketch = gc.run_script(script)
        plan_ = gc.plan(sketch, scripts.T01_CLAIMS if name.startswith("T01") else None)
        record = json.loads(json.dumps(gc.record(sketch, plan_, script, None, [], None, plan_.rules,
                                                 claims=scripts.T01_CLAIMS if name == "T01" else None)))
        assert gc.recompile(record) == record["ops"], name
        assert record["format"] == "openreynolds.geometry/1" and record["script"] == script
        assert record["units"] == "mm" and record["scale"] == 0.001
        assert record["ports"] == [p.as_dict() for p in plan_.ports]
        assert [(r["name"], r.get("at"), r.get("box")) for r in record["patches"]] ==             [(r["name"], r.get("at"), r.get("box")) for r in plan_.rules]
        assert all("kind" not in r for r in record["patches"] if r["name"] in ("inlet", "outlet"))
        assert set(record["features"]) == set(plan_.features)
        assert record["instances"] == json.loads(json.dumps(plan_.instances))
        assert record["compliance"] is None and record["lint"] == [] and record["measurements"] is None
        assert record["built_with"]["gmsh"] and re.match(r"\d{4}-\d\d-\d\dT", record["built_with"]["at"])
    record = json.loads(json.dumps(gc.record(gc.run_script(scripts.T01), plan_of(scripts.T01), scripts.T01, None, [],
                                             None, plan_.rules, claims=scripts.T01_CLAIMS)))
    assert record["claims"]["claims"][2]["id"] == "c3"
    assert record["ports"] == [{"name": "inlet", "kind": "inlet", "edges": ["main.start"]},
                               {"name": "outlet", "kind": "outlet", "edges": ["main.end"]}]


def test_a_clip_on_an_inclined_wall_is_rotated():
    plan_ = plan_of('''
s = Sketch(units="mm")
main = s.passage(width=3, start=(0, 0), heading=30, name="main").line(60)
loop = Bypass(wall=main.top, width=3, leave_angle=20, outer_radius=6, return_angle=80, at=30, name="loop")
s.fluid = main | loop
s.inlet(main.start)
s.outlet(main.end)
''')
    kinds = [(op["op"], op["name"]) for op in plan_.ops]
    assert kinds == [("channel", "main"), ("channel", "loop.raw.flat"), ("rotate", "loop.raw"), ("rect", "loop.clip.flat"),
                     ("rotate", "loop.clip"), ("intersect", "loop"), ("fuse", "fluid")]
    turn = by_name(plan_, "loop.clip")
    assert turn["angle"] == pytest.approx(30) and turn["about"] == pytest.approx([-0.75, 1.5 * math.cos(math.radians(30))])
    assert by_name(plan_, "loop.raw")["angle"] == pytest.approx(30)
    flat = by_name(plan_, "loop.raw.flat")
    assert flat["heading"] == pytest.approx(20) and flat["from"] == flat["path"][-1]["line"]["to"]
    wall = plan_.features["loop"].solved["wall_line"]
    assert wall["axis"] is None and wall["angle"] == pytest.approx(30)
    assert wall["flow"] == pytest.approx([math.cos(math.radians(30)), math.sin(math.radians(30))])
    # the loop's outline hangs on the inclined wall: its lowest point is on the wall line
    pts = plan_.outlines["loop"][0]
    n = (-math.sin(math.radians(30)), math.cos(math.radians(30)))
    v = [(p[0] + 0.75) * n[0] + (p[1] - 1.5 * math.cos(math.radians(30))) * n[1] for p in pts]
    assert min(v) == pytest.approx(0, abs=1e-9) and max(v) == pytest.approx(11.25, abs=0.01)


def test_an_inclined_bypass_builds_the_same_face_as_a_rotated_valve(gm):
    inclined = plan_of('''
s = Sketch(units="mm")
main = s.passage(width=3, start=(0, 0), heading=30, name="main").line(60)
loop = Bypass(wall=main.top, width=3, leave_angle=20, outer_radius=6, return_angle=80, at=30, name="loop")
s.fluid = main | loop
''')
    rotated = plan_of('''
s = Sketch(units="mm")
main = s.passage(width=3, start=(0, 0), heading=0, name="main").line(60)
loop = Bypass(wall=main.top, width=3, leave_angle=20, outer_radius=6, return_angle=80, at=30, name="loop")
s.fluid = (main | loop).rotated(30, about=(0, 0))
''')
    _, _, _, a = built(gm, inclined, "inclined")
    _, _, _, b = built(gm, rotated, "rotated_valve")
    assert a.area == pytest.approx(b.area, rel=1e-6) and a.islands == b.islands == 1
    assert len(a.curves) == len(b.curves) and min(a.lengths.values()) == pytest.approx(min(b.lengths.values()), abs=1e-6)


def test_edge_targets_cover_every_port_intent():
    """Every EdgeRef in `plan.ports` has an `edge_targets` entry (the contract resolve_ports
    reads), on the five worked scripts and a mirrored / moved / rotated valve."""
    for name, script in list(scripts.WORKED.items()) + list(VARIANTS.items()):
        plan_ = plan_of(script)
        refs = [str(e) for p in plan_.ports for e in p.edges if not isinstance(e, str)]
        assert refs, name
        for key in refs:
            target = plan_.edge_targets[key]
            assert target["kind"] in ("cap", "side", "loop"), (name, key)
            assert len(target["points"]) >= 2
            if target["kind"] == "cap":
                (ax, ay), (bx, by) = target["points"]
                assert math.hypot(ax - bx, ay - by) == pytest.approx(target["width"])
        assert set(plan_.edge_targets) == set(refs), name
    # the variants move the targets with the fluid
    def flat(points):
        return [x for p in points for x in p]

    base = plan_of(scripts.T01).edge_targets["main.end"]["points"]
    moved = plan_of(VARIANTS["moved"]).edge_targets["main.end"]["points"]
    assert flat(moved) == pytest.approx(flat((x + 10, y + 5) for x, y in base))
    mirrored = plan_of(VARIANTS["mirrored"]).edge_targets["main.end"]["points"]
    assert flat(sorted(tuple(p) for p in mirrored)) == pytest.approx(flat(sorted((x, -y) for x, y in base)))
    rotated = plan_of(VARIANTS["rotated"]).edge_targets["main.end"]["points"]
    c, s_ = math.cos(math.radians(30)), math.sin(math.radians(30))
    assert flat(rotated) == pytest.approx(flat((x * c - y * s_, x * s_ + y * c) for x, y in base))
    # a port on a feature outside the fluid is refused at compile
    with pytest.raises(SketchError) as err:
        plan_of(scripts.T05.replace("s.fluid = duct - cyl", "s.fluid = duct"))
    assert err.value.code == "E-PORT-MISSING" and "cyl.edge names Disk 'cyl', which is not part of s.fluid" in str(err.value)


def test_solved_keys_are_the_listed_set():
    plan_ = plan_of(scripts.T01)
    assert set(plan_.features["loop"].solved) == gc.SOLVED_KEYS["Bypass"]
    assert set(plan_.features["main"].solved) == gc.SOLVED_KEYS["Passage"]
    assert set(plan_.features["loops"].solved) == gc.SOLVED_KEYS["Row"]
    assert set(plan_of(scripts.T02).features["snake"].solved) == gc.SOLVED_KEYS["Serpentine"]
    assert set(plan_of(scripts.T04).features["u"].solved) == gc.SOLVED_KEYS["Passage"]
    assert set(plan_.features["loop"].solved["wall_line"]) >= {"axis", "value", "flow", "outward", "span"}
    assert plan_.features["loop"].solved["wall_line"] == {"axis": "y", "value": 1.5, "flow": [1, 0], "outward": [0, 1],
                                                          "span": 60, "point": [0, 1.5], "angle": 0}
    legs = plan_.features["main"].solved["legs"]
    assert legs[0]["kind"] == "line" and tuple(legs[0]["from"]) == (0, 0) and tuple(legs[0]["to"]) == (60, 0)
    assert legs[0]["heading"] == 0 and legs[0]["heading_out"] == 0 and legs[0]["length"] == 60
    assert plan_.features["loop"].solved["lands_on"] == "main.top"
    assert plan_.features["loop"].solved["leave_length"] == 3 and plan_.features["loop"].solved["leave_length_default"] is True
    assert plan_.features["loop"].params["leave_length"] is None and plan_.features["loop"].params["at"] == pytest.approx(7.24, abs=0.005)
    for feature in plan_.features.values():
        assert feature.kind in ("Bypass", "Passage", "Row", "Serpentine", "Rect", "Disk", "Polygon", "Band", "Outline",
                                "Fuse", "Cut", "Intersect", "BodyInBox")


def test_the_fluid_ops_never_carry_a_mirror():
    for script in list(scripts.WORKED.values()) + list(VARIANTS.values()) + [
            scripts.T01 + "\ns.fluid = ((main | loops).mirrored('x', at=30)).mirrored('y', at=0)\n",
            scripts.T05 + "\ns.fluid = (duct - cyl).mirrored('x', at=150, keep=True)\n"]:
        plan_ = plan_of(script)
        assert not any(op["op"] in ("mirror", "dilate", "affine") for op in plan_.ops)
        assert plan_.ops[-1]["name"] == "fluid"
    plan_ = plan_of(scripts.T05 + "\ns.fluid = (duct - cyl).mirrored('x', at=150, keep=True)\n")
    assert [op["op"] for op in plan_.ops].count("disk") == 2
    assert plan_.ops[-1] == {"op": "fuse", "name": "fluid", "of": ["cut1", "cut1.mirror"]}
    assert by_name(plan_, "cyl.mirror")["center"] == [200, 30]


def test_build_canary_refuses_a_transformed_face(gm):
    """A plan whose fluid carries a mirror (never compiled, but a hand-made plan can) and a
    hand-dilated face are refused by the canary: the classifier answers inside for a point
    outside (D26, measured)."""
    plan_ = plan_of(scripts.T05)
    plan_.ops.append({"op": "mirror", "name": "mirrored", "target": "fluid", "axis": "y", "at": 0})
    plan_.ops[-1], plan_.ops[-2] = dict(plan_.ops[-1], name="fluid"), dict(plan_.ops[-2], name="cut")
    plan_.ops[-1]["target"] = "cut"
    gm.model.add("mirror_last")
    with pytest.raises(SketchError) as err:
        gc.build(gm, plan_)
    assert err.value.code == "E-KERNEL-STATE"
    assert "the face carries a transform" in str(err.value) and "nothing in the script to fix" in str(err.value)
    gm.model.add("dilated")
    occ = gm.model.occ
    face = occ.addRectangle(0, 0, 0, 60, 3)
    occ.dilate([(2, face)], 0, 0, 0, 0.001, 0.001, 0.001)
    occ.synchronize()
    with pytest.raises(SketchError) as err:
        gc.canary(gm, face)
    assert err.value.code == "E-KERNEL-STATE" and err.value.numbers["inside"] == 1
    gm.model.add("plain")
    face = occ.addRectangle(0, 0, 0, 60, 3)
    occ.synchronize()
    gc.canary(gm, face)
    assert gc.oriented_area(gm, face) == pytest.approx(180)


def test_plan_feature_builds_a_row_item_alone(gm):
    """The T01 loop at return 45 (the lap-1 refusal's partial): one instance anchored at
    u = 0 with footprint (-12.46, 7.28), built and measured on its own."""
    with pytest.raises(SketchError) as err:
        plan_of(scripts.T01_LAP1)
    partial = err.value.partial
    assert partial.name == "loop"
    plan_ = gc.plan_feature(partial)
    assert [(op["op"], op["name"]) for op in plan_.ops] == [("channel", "loop.raw"), ("rect", "loop.clip"),
                                                            ("intersect", "loop"), ("translate", "fluid")]
    raw = by_name(plan_, "loop.raw")
    assert raw["start"] == [0, 1.5] and raw["path"][1]["arc"]["angle"] == 205
    solved = plan_.features["loop"].solved
    assert solved["footprint"] == pytest.approx([-12.46, 7.28], abs=0.005)
    assert solved["sweep"] == 205 and solved["lands_upstream_by"] if "lands_upstream_by" in solved else True
    assert solved["landing"][0] == pytest.approx(-10.34, abs=0.005)
    face, legs, checks, b = built(gm, plan_, "partial")
    x0, y0, x1, y1 = gc.sampled_bounds(gm, 2, face)
    assert (x0, x1) == pytest.approx((-12.46, 7.28), abs=0.005) and (y0, y1) == pytest.approx((1.5, 12.75), abs=0.005)
    assert b.islands == 0 and checks == []
    assert [r["kind"] for r in legs["loop.raw"]] == ["line", "arc", "to", "ports"]
    assert legs["loop.raw"][0]["from"] == pytest.approx((0, 1.5)) and legs["loop.raw"][0]["to"] == pytest.approx((2.82, 2.53), abs=0.005)
    assert legs["loop.raw"][2]["lands"] == pytest.approx((-10.34, 1.5), abs=0.005)
    assert plan_.rules == [] and plan_.ports == [] and plan_.notes == []
    assert set(plan_.features["loop"].solved) == gc.SOLVED_KEYS["Bypass"]


# -- the build --------------------------------------------------------------------------------


def test_t01_builds_to_the_measured_numbers(gm):
    """The compiled T01 through mesh2d.build_face: the section 7.1 numbers, measured on
    this box on 2026-09-07 -- 36 edges, the shortest 1.501 from (0, 1.5) (the row's margin),
    the neighbours' walls 2.437 apart, the lip vertex at (11.416, 1.5), area 513.7, 4
    islands, extent 60 x 14.25."""
    plan_ = plan_of(scripts.T01)
    face, legs, checks, b = built(gm, plan_, "t01")
    assert len(b.curves) == 36 and b.islands == 4
    assert b.area == pytest.approx(513.7, abs=0.1)
    assert b.extent == pytest.approx((60, 14.25), abs=0.005)
    shortest = min(b.lengths, key=b.lengths.get)
    assert b.lengths[shortest] == pytest.approx(1.501, abs=0.001)
    sx0, sy0, _, sx1, sy1, _ = gm.model.getBoundingBox(1, shortest)
    assert (sx0, sy0, sy1) == pytest.approx((0, 1.5, 1.5), abs=1e-6) and sx1 == pytest.approx(1.501, abs=0.001)
    assert b.short == []
    assert len(checks) == 1 and checks[0]["level"] == "info"
    assert re.search(r"gap between copies 2\.43[78]", checks[0]["what"])
    vertices = set()
    for d, c in gm.model.getBoundary([(2, face)], combined=False, oriented=False):
        for dd, v in gm.model.getBoundary([(1, abs(c))], oriented=False):
            x, y, _ = gm.model.getValue(0, abs(v), [])
            vertices.add((float(x), float(y)))
    # the lip: at anchor + lip_u (7.239 + 4.176 = 11.415 for the solved anchor; 11.416 at the probe's 7.24)
    loop = plan_.features["loop"]
    lip_x = loop.params["at"] + loop.solved["lip_u"]
    assert lip_x == pytest.approx(11.416, abs=0.005)
    pitch = plan_.features["loops"].solved["pitch"]
    for k in range(4):
        assert any(abs(x - (lip_x + k * pitch)) < 1e-3 and abs(y - 1.5) < 1e-6 for x, y in vertices), k
    assert legs["loop.raw"][0]["from"] == pytest.approx((7.24, 1.5), abs=0.005)
    assert legs["loop.raw"][1]["centre"] == pytest.approx((8.52, 6.755), abs=0.005)
    assert legs["loop.raw"][2]["lands"] == pytest.approx((3.024, 1.5), abs=0.005)
    assert legs["main"][0]["length"] == 60


def test_t02_to_t05_build_to_the_measured_numbers(gm):
    expected = {
        "T02": (20, 296.6, 0, (38, 20), 2.0),
        "T03": (6, 1800, 0, (105, 85), 10.0),
        "T04": (10, 1211, 0, (74, 28), 8.0),
        "T05": (5, 300 * 60 - math.pi * 25, 1, (300, 60), 10 * math.pi),
    }
    for name, (edges, area, islands, extent, shortest) in expected.items():
        plan_ = plan_of(scripts.WORKED[name])
        face, legs, checks, b = built(gm, plan_, name)
        assert len(b.curves) == edges, name
        assert b.area == pytest.approx(area, abs=0.5), name
        assert b.islands == islands and b.extent == pytest.approx(extent, abs=1e-6), name
        assert min(b.lengths.values()) == pytest.approx(shortest, abs=1e-6), name
        assert checks == [], name
        # the closed-form outline agrees with the built bounds
        pts = [p for loops in plan_.outlines.values() for loop in loops for p in loop]
        x0, y0, x1, y1 = gc.sampled_bounds(gm, 2, face)
        assert (min(p[0] for p in pts), min(p[1] for p in pts)) == pytest.approx((x0, y0), abs=0.01), name
        assert (max(p[0] for p in pts), max(p[1] for p in pts)) == pytest.approx((x1, y1), abs=0.01), name


def test_moved_and_rotated_emit_translate_and_rotate_and_move_the_outlines(gm):
    moved = plan_of(VARIANTS["moved"])
    assert moved.ops[-1] == {"op": "translate", "name": "fluid", "target": "fuse1", "by": [10, 5]}
    rotated = plan_of(VARIANTS["rotated"])
    assert rotated.ops[-1] == {"op": "rotate", "name": "fluid", "target": "fuse1", "angle": 30, "about": [0, 0]}
    for name, plan_ in (("moved", moved), ("rotated", rotated)):
        face, _, _, b = built(gm, plan_, name)
        pts = [p for loops in plan_.outlines.values() for loop in loops for p in loop]
        x0, y0, x1, y1 = gc.sampled_bounds(gm, 2, face)
        assert (min(p[0] for p in pts), max(p[0] for p in pts)) == pytest.approx((x0, x1), abs=0.01), name
        assert (min(p[1] for p in pts), max(p[1] for p in pts)) == pytest.approx((y0, y1), abs=0.01), name
        assert b.islands == 4 and len(b.curves) == 36
    legs = rotated.features["main"].solved["legs"]
    assert legs[0]["heading"] == pytest.approx(30) and legs[0]["to"] == pytest.approx([60 * math.cos(math.radians(30)),
                                                                                       60 * math.sin(math.radians(30))])
    # a rotation about no point turns about the operand's bounding-box centre, stated explicitly in the op
    plan_ = plan_of(scripts.T05.replace("s.fluid = duct - cyl", "s.fluid = (duct - cyl).rotated(90)"))
    assert plan_.ops[-1]["about"] == [150, 30]


def test_a_disjoint_fluid_is_refused_by_the_build(gm):
    plan_ = plan_of(scripts.T05.replace("size=(300, 60)", "size=(300, 60)").replace("s.fluid = duct - cyl",
                                                                                      "s.fluid = duct | cyl.moved(0, 100)"))
    with pytest.raises(SketchError) as err:
        built(gm, plan_, "disjoint")
    assert err.value.code == "E-DISJOINT" and "separate faces" in str(err.value)


def test_plan_round_trips_through_json():
    for script in list(scripts.WORKED.values()) + list(VARIANTS.values()):
        plan_ = plan_of(script)
        back = gc.Plan.from_dict(json.loads(json.dumps(plan_.as_dict())))
        assert back.ops == plan_.ops and back.rules == plan_.rules
        assert back.as_dict() == json.loads(json.dumps(plan_.as_dict()))
        assert [p.as_dict() for p in back.ports] == [p.as_dict() for p in plan_.ports]
        assert set(back.edge_targets) == set(plan_.edge_targets) and set(back.outlines) == set(plan_.outlines)
        assert back.apart == plan_.apart and back.expected == plan_.expected and back.external is None
        for key, solved in plan_.features.items():
            assert back.features[key].kind == solved.kind and set(back.features[key].solved) == set(solved.solved)


def test_a_body_in_a_box_declares_its_own_ports():
    plan_ = plan_of('''
s = Sketch(units="mm")
wing = s.disk(centre=(0, 0), radius=5, name="wing")
box = BodyInBox(wing, ahead=2, behind=5, above=2, below=2, name="box")
s.fluid = box
s.inlet(box.inlet)
s.outlet(box.outlet)
s.patch(box.farfield, name="farfield", kind="slip")
s.wall(box.body_edge, name="wing")
''')
    assert plan_.external == {"ahead": 2, "behind": 5, "above": 2, "below": 2, "far": "slip", "body": "wing"}
    assert plan_.edge_targets["box.inlet"]["kind"] == "box" and plan_.edge_targets["box.inlet"]["box"] == [-25, -25, 55, 25]
    assert plan_.edge_targets["box.body"]["kind"] == "loop" and len(plan_.edge_targets["box.body"]["points"]) == 64
    assert [r["name"] for r in plan_.rules] == ["inlet", "outlet", "farfield", "wing"]
    assert plan_.features["box"].kind == "BodyInBox"
