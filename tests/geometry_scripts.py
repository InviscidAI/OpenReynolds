"""The worked scripts of DESIGN.md section 7 and the probe spec they compile to, shared by
the sketch and compile tests (imported by module name: the tests directory is on
sys.path the way `conftest` is)."""
from __future__ import annotations

T01_LAP1 = '''
s = Sketch(units="mm")
main = s.passage(width=3, start=(0, 0), heading=0, name="main").line(60)
loop = Bypass(wall=main.top, width=3, leave_angle=20, outer_radius=6,
              return_angle=45, name="loop")            # leave_length: default, one width
loops = Row(loop, count=4, name="loops")               # no gap in the request: the tool solves it
s.fluid = main | loops
s.inlet(main.start)
s.outlet(main.end)
'''

T01 = '''
s = Sketch(units="mm")
main = s.passage(width=3, start=(0, 0), heading=0, name="main").line(60)
loop = Bypass(wall=main.top, width=3, leave_angle=20, outer_radius=6,
              return_angle=80, name="loop")
loops = Row(loop, count=4, name="loops")
s.fluid = main | loops
s.inlet(main.start)
s.outlet(main.end)
s.note("return_angle 80 keeps 4 loops of outer r 6 inside 60")
'''

T02 = '''
s = Sketch(units="mm")
snake = Serpentine(width=2, passes=4, pass_length=30, bend_radius=3,
                   start=(0, 0), heading=0, stack="+y", name="snake")
s.fluid = snake
s.inlet(snake.start)
s.outlet(snake.end)
'''

T03 = '''
s = Sketch(units="mm")
duct = s.passage(width=10, start=(0, 0), heading=0, name="duct")
duct.line(100)
duct.turn(90)          # a sharp left corner: the tool mitres it (each leg extended by w/2 = 5)
duct.line(80)
s.fluid = duct
s.inlet(duct.start)
s.outlet(duct.end)
'''

T04 = '''
s = Sketch(units="mm")
u = s.passage(width=8, start=(0, 0), heading=0, name="u")
u.line_to(x=60)
u.u_turn(radius=10, side="left")       # left of a +x heading: the return leg is on the +y side, 20 above
u.line_to(x=0)                         # an open end, not a landing
s.fluid = u
s.inlet(u.start)
s.outlet(u.end)
'''

T04_RELATIVE = '''
s = Sketch(units="mm")
u = s.passage(width=8, start=(0, 0), heading=0, name="u").line(60).arc(radius=10, turn=180).line(60)
s.fluid = u
s.inlet(u.start)
s.outlet(u.end)
'''

T05 = '''
s = Sketch(units="mm")
duct = s.rect(origin=(0, 0), size=(300, 60), name="duct")
cyl = s.disk(centre=(100, 30), diameter=10, name="cyl")
s.fluid = duct - cyl
s.inlet(duct.left)
s.outlet(duct.right)
s.wall(cyl.edge, name="cylinder")
s.wall(duct.top, duct.bottom, name="walls")
'''

WORKED = {"T01": T01, "T02": T02, "T03": T03, "T04": T04, "T05": T05}

T01_PROBE = {"scale": 0.001, "ops": [
    {"op": "channel", "name": "main", "width": 3, "start": [0, 0], "heading": 0, "path": [{"line": 60}]},
    {"op": "channel", "name": "loop.raw", "width": 3, "start": [7.24, 1.5], "heading": 20, "from": "y:1.5",
     "path": [{"line": 3}, {"arc": {"radius": 4.5, "angle": 240}}, {"line": {"to": "y:1.5"}}]},
    {"op": "rect", "name": "loop.clip", "origin": [-100, 1.5], "size": [300, 100]},
    {"op": "intersect", "name": "loop", "of": ["loop.raw", "loop.clip"]},
    {"op": "repeat", "name": "loops", "target": "loop", "count": 4, "step": [14.66, 0]},
    {"op": "fuse", "name": "fluid", "of": ["main", "loops"]}],
    "patches": [{"name": "inlet", "at": "near:0,0"}, {"name": "outlet", "at": "near:60,0"}]}
"""`scratchpad/probe/t01_lap2c.json` of the design document (section 7.1), the fluid named
`fluid` and the loop ops named as the compiler names them."""

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
    ]}
"""Section 4.1's claims file for T01."""
