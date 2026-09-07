"""The Tesla valve: a straight channel with a row of bypass loops that leave it at a
shallow angle, sweep round and return into it against the flow, so the fluid meets its
own returning stream in the reverse direction and passes freely forward (Tesla's
valvular conduit, US patent 1,329,559 of 1920).

Every number here is a preset with a source (D16). `t01` is the shape study
20260907-003655-6aab committed: the four loops of that study's `geometry.json` (width 3,
main 60 long, outer radius 4.5 + 1.5 = 6, leave 20 degrees, the return heading 260 = 80
degrees from the upstream direction, repeated at a step of 13) written against the
sketch API, which places the row itself. A preset from Tesla's drawing with the
dimensions Nguyen et al. (2021) measured is added when those numbers are looked up
(DESIGN.md 12, open question 4); nothing here is typed from memory.
"""
from __future__ import annotations

from ..sketch import Bypass, Row, Sketch

SOURCE = ("study 20260907-003655-6aab (preset t01); Tesla (1920) US 1,329,559 and Nguyen et al. (2021) "
          "when their numbers are looked up")

KEYWORDS = ("tesla", "valvular conduit", "fluidic diode")

PRESETS = {
    "t01": {"length": 60, "width": 3, "outer_radius": 6, "leave_angle": 20, "return_angle": 80, "count": 4,
            "pitch": 13},
}
"""t01: the committed spec of study 20260907-003655-6aab, read from its geometry.json
(channel width 3, `line 60`, arc radius 4.5 on a 3-wide loop so the outer wall is at 6,
heading 20 then an arc of 240 so the return heads 260, `repeat` step 13)."""

CLAIMS = {
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
"""Section 4.1's claims file for T01 (the benchmark prompt the t01 preset answers)."""


def build(*, length, width, outer_radius, leave_angle, return_angle, count, gap=None, pitch=None,
          leave_length=None, units="mm") -> Sketch:
    """A main passage of `width` running `length` along +x from the origin, with `count`
    bypass loops of the same width on its top wall: each leaves at `leave_angle` from
    the flow, runs `leave_length` straight (None: one width, D40), sweeps round an arc
    whose outer wall has `outer_radius`, and returns onto the wall at `return_angle`
    from the upstream direction. The row is placed by the tool: with neither `gap` nor
    `pitch` it solves the largest uniform gap that fits and centres the row; `pitch`
    fixes the anchor spacing (t01's 13, a hair under the footprint, so neighbours' walls
    are 0.8 apart and the row is centred on the wall). Inlet at the main's start, outlet
    at its end."""
    s = Sketch(units=units)
    main = s.passage(width=width, start=(0, 0), heading=0, name="main").line(length)
    loop = Bypass(wall=main.top, width=width, leave_angle=leave_angle, outer_radius=outer_radius,
                  return_angle=return_angle, leave_length=leave_length, name="loop")
    loops = Row(loop, count=count, gap=gap, pitch=pitch, name="loops")
    s.fluid = main | loops
    s.inlet(main.start)
    s.outlet(main.end)
    return s
