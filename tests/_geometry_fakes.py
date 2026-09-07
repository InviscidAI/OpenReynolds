"""What U4's tests stand on while the other units land: specs in the ops grammar for the
worked shapes (DESIGN.md section 7), a `Sketch` that only records itself, and the
monkeypatch that hands `cli.exec_script` a canned Plan in place of `compile.plan`.

The ops below are the compiled forms the worked examples list (T01 lap 2 is the probe
`t01_lap2c.json` up to names; T05 is `rect duct`, `disk cyl`, `cut fluid`), so every
number a test asserts was measured by today's `mesh2d.py --dry-run` on the same ops.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from openreynolds.geometry import claims, cli, compile as C, sketch
from openreynolds.geometry.sketch import SketchError

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "geometry"

T01_SPEC = json.loads((FIXTURES / "t01_lap2c.json").read_text(encoding="utf-8"))
"""T01 lap 2 (section 7.1): return 80, gap solved 1.64, pitch 14.66, anchors 7.24 + k x 14.66."""

T01_LAP1_PARTIAL_SPEC = {"scale": 0.001, "ops": [
    {"op": "channel", "name": "loop_raw", "width": 3, "start": [0, 1.5], "heading": 20, "from": "y:1.5",
     "path": [{"line": 3}, {"arc": {"radius": 4.5, "angle": 205}}, {"line": {"to": "y:1.5"}}]},
    {"op": "rect", "name": "loop_clip", "origin": [-100, 1.5], "size": [300, 100]},
    {"op": "intersect", "name": "loop", "of": ["loop_raw", "loop_clip"]}]}
"""The single instance of section 7.1 lap 1 (return 45; anchor at u = 0 for the table)."""

T01_LAP1_SOLVED = {
    "R": 4.5, "sweep": 205, "footprint": [-12.46, 7.28], "height": 11.25, "leave_length": 3,
    "leave_length_default": True, "theta_leave": 20, "theta_return": 45, "lands_on": "main.top",
    "wall_line": {"axis": "y", "value": 1.5, "flow": [1, 0], "outward": [0, 1], "span": 60},
}

T05_SPEC = {"scale": 0.001, "ops": [
    {"op": "rect", "name": "duct", "origin": [0, 0], "size": [300, 60]},
    {"op": "disk", "name": "cyl", "center": [100, 30], "radius": 5},
    {"op": "cut", "name": "fluid", "from": "duct", "take": ["cyl"]}],
    "patches": [{"name": "inlet", "at": "x:min"}, {"name": "outlet", "at": "x:max"},
                {"name": "cylinder", "at": "near:95,30"}]}
"""T05 (section 7.4): a 10 mm cylinder in a 300 x 60 channel, its surface its own patch."""

T01_CLAIMS = {
    "schema": "openreynolds.geometry/claims-1", "unit": "mm", "kind": "passage", "flow": "+x",
    "claims": [
        {"id": "c1", "kind": "measure", "says": "a straight channel 3 mm wide", "measure": "width", "of": "main", "value": 3},
        {"id": "c2", "kind": "measure", "says": "60 mm long", "measure": "length", "of": "main", "value": 60, "tol": 0.1},
        {"id": "c3", "kind": "count", "says": "4 bypass loops", "of": "loops", "value": 4},
        {"id": "c4", "kind": "measure", "says": "loop channel 3 mm wide", "measure": "width", "of": "loops[*]", "value": 3},
        {"id": "c5", "kind": "measure", "says": "outer radius 6 mm", "measure": "outer_radius", "of": "loops[*]", "value": 6},
        {"id": "c6", "kind": "predicate", "says": "leave the main channel at a shallow angle", "predicate": "shallow_angle",
         "of": "loops[*]", "args": {"max": 30}},
        {"id": "c7", "kind": "predicate", "says": "return into the main channel against the forward direction",
         "predicate": "returns_against_flow", "of": "loops[*]"},
        {"id": "c8", "kind": "patch", "says": "inlet at x=0", "patch": "inlet", "at": [0, 0], "tol": 0.5},
        {"id": "c9", "kind": "patch", "says": "outlet at x=60 mm", "patch": "outlet", "at": [60, 0], "tol": 0.5},
        {"id": "c10", "kind": "report", "says": "sweep round", "measure": "sweep", "of": "loops[*]"},
        {"id": "c11", "kind": "report", "says": "(how steeply it returns, and how far apart the loops are)",
         "measure": "return_angle", "of": "loops[*]"},
        {"id": "c12", "kind": "not_measurable", "says": "produce the mesh, run checkMesh, render the mesh, write results.json",
         "not_measurable": "the finish and the main agent do these; the fitness table reports the mesh"},
    ],
}
"""Section 4.1, verbatim."""

T01_LAP1_SCRIPT = '''s = Sketch(units="mm")
main = s.passage(width=3, start=(0, 0), heading=0, name="main").line(60)
loop = Bypass(wall=main.top, width=3, leave_angle=20, outer_radius=6,
              return_angle=45, name="loop")            # leave_length: default, one width
loops = Row(loop, count=4, name="loops")               # no gap in the request: the tool solves it
s.fluid = main | loops
s.inlet(main.start)
s.outlet(main.end)
'''

T01_LAP2_SCRIPT = '''s = Sketch(units="mm")
main = s.passage(width=3, start=(0, 0), heading=0, name="main").line(60)
loop = Bypass(wall=main.top, width=3, leave_angle=20, outer_radius=6,
              return_angle=80, name="loop")
loops = Row(loop, count=4, name="loops")
s.fluid = main | loops
s.inlet(main.start)
s.outlet(main.end)
s.note("return_angle 80 keeps 4 loops of outer r 6 inside 60")
'''

T05_SCRIPT = '''s = Sketch(units="mm")
duct = s.rect(origin=(0, 0), size=(300, 60), name="duct")
cyl = s.disk(centre=(100, 30), diameter=10, name="cyl")
s.fluid = duct - cyl
s.inlet(duct.left)
s.outlet(duct.right)
s.wall(cyl.edge, name="cylinder")
s.wall(duct.top, duct.bottom, name="walls")
'''


def landed(unit: str) -> bool:
    """Whether another unit's implementation is in this checkout: U1 when `Sketch()`
    constructs, U3 when `claims.parse` judges. A skeleton stub raises NotImplementedError;
    anything else (a refusal of the probe payload included) means the unit is there."""
    try:
        if unit == "U1":
            sketch.Sketch._reset()
            sketch.Sketch(units="mm")
            sketch.Sketch._reset()
        elif unit == "U3":
            claims.parse({"schema": "openreynolds.geometry/claims-1", "unit": "mm", "kind": "passage",
                          "flow": "+x", "claims": []})
        else:
            raise ValueError(unit)
    except NotImplementedError:
        return False
    except Exception:  # noqa: BLE001 - a refusal is an implementation answering
        return True
    return True


class FakeSketch:
    """The two things `exec_script` needs from a Sketch until U1 lands: the registry
    (`current()`, E-NO-SKETCH / E-MANY-SKETCHES) and the notes."""

    _made: list["FakeSketch"] = []

    def __init__(self, units: str = "mm", name: str = "sketch"):
        self.units = units
        self.name = name
        self.notes: list[str] = []
        self.fluid = None
        FakeSketch._made.append(self)

    def note(self, text: str) -> None:
        self.notes.append(text)

    @staticmethod
    def current() -> "FakeSketch":
        n = len(FakeSketch._made)
        if n == 0:
            raise SketchError("E-NO-SKETCH", "script", "the script made no Sketch")
        if n > 1:
            raise SketchError("E-MANY-SKETCHES", "script", f"the script made {n} Sketch objects; exactly one")
        return FakeSketch._made[0]

    @staticmethod
    def only() -> "FakeSketch":
        """The one sketch a script made (the real registry's name for it)."""
        return FakeSketch.current()

    @staticmethod
    def _reset() -> None:
        FakeSketch._made.clear()


def plan_of(spec: dict) -> C.Plan:
    return cli.plan_from_spec(copy.deepcopy(spec))


def partial_plan() -> C.Plan:
    """The lap-1 T01 item alone as `compile.plan_feature` will return it: one Bypass
    feature named `loop` with the solved footprint the dashed box is drawn from."""
    plan = plan_of(T01_LAP1_PARTIAL_SPEC)
    plan.features = {"loop": C.Solved(kind="Bypass", params={"wall": "main.top", "width": 3, "leave_angle": 20,
                                                             "outer_radius": 6, "return_angle": 45, "leave_length": None},
                                      solved=dict(T01_LAP1_SOLVED), ops=["loop_raw", "loop_clip", "loop"])}
    plan.instances = {}
    return plan


def row_refusal() -> SketchError:
    """Section 5's E-ROW-FIT, with a `partial` so the cli takes the rc-5 picture path."""
    return SketchError(
        "E-ROW-FIT", "Row 'loops'", "4 x Bypass 'loop' do not fit on main.top (60 long)",
        "each instance spans 19.74 along the wall (12.46 upstream of its anchor to 7.28 downstream)",
        "4 x 19.74 = 78.96 needed before any gap; 60 - 2 x margin 1.5 = 57 available",
        partial=object(), numbers={"footprint": 19.74, "available": 57.0,
                                   "table": {"45": 19.74, "60": 15.96, "70": 14.30, "80": 13.02, "90": 12.00}})


def install(monkeypatch, plan=None, raising: SketchError | None = None, partial: C.Plan | None = None) -> None:
    """Bind the FakeSketch into the API the script sees and make `compile.plan` return
    `plan` (a Plan, or a callable of the sketch) or raise `raising`; `compile.plan_feature`
    returns `partial` when given."""
    FakeSketch._reset()
    monkeypatch.setattr(sketch, "Sketch", FakeSketch)
    monkeypatch.setitem(sketch.API, "Sketch", FakeSketch)

    def fake_plan(sk, claims=None, gmsh=None):
        if raising is not None:
            raise raising
        return plan(sk) if callable(plan) else plan

    monkeypatch.setattr(C, "plan", fake_plan)
    if partial is not None:
        monkeypatch.setattr(C, "plan_feature", lambda feature, gmsh=None: partial)
