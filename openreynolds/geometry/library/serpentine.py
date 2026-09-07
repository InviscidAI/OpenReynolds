"""The serpentine: straight passes of one width joined by 180-degree bends, stacked
across the heading, the flow entering the first pass and leaving the last. An even number
of passes ends on the side it started from (section 7.2's contradiction, pinned there).

`t02` is the benchmark prompt T02's numbers (DESIGN.md 7.2): a passage 2 wide, four
passes 30 long, three U-bends of centreline radius 3, stacked in +y from (0, 0) heading
+x. Its claims are T02's with one difference: T02's c9 asks for the outlet at the RIGHT
end, which no four-pass serpentine can give (7.2 shows the row failing and the desk
committing with the disagreement); a golden is a shape a person approves, so the entry's
claim says where this shape's outlet is -- (0, 18), the left end -- and the reason. It is
an `at` claim, not `side: left`: the bends bulge one outer radius past the passes, so the
extent runs -4..34 and the outlet at x = 0 is not on the extent's edge line, which is
what a `side` claim tests.
"""
from __future__ import annotations

from ..sketch import Serpentine, Sketch

SOURCE = "benchmark prompt T02 (preset t02; DESIGN.md 7.2)"

KEYWORDS = ("serpentine", "meander", "u-bends", "u bends")

PRESETS = {
    "t02": {"width": 2, "passes": 4, "pass_length": 30, "bend_radius": 3},
}
"""t02: T02's request as section 7.2 reads it (width 2, four passes of 30, three bends of
centreline radius 3)."""

CLAIMS = {
    "schema": "openreynolds.geometry/claims-1", "unit": "mm", "kind": "passage", "flow": "+x",
    "claims": [
        {"id": "c1", "kind": "measure", "says": "a passage 2 mm wide", "measure": "width", "of": "snake", "value": 2},
        {"id": "c2", "kind": "count", "says": "four straight passes", "of": "passes", "value": 4},
        {"id": "c3", "kind": "measure", "says": "passes 30 mm long", "measure": "pass_length", "of": "snake", "value": 30},
        {"id": "c4", "kind": "count", "says": "three 180-degree U-bends", "of": "bends", "value": 3},
        {"id": "c5", "kind": "predicate", "says": "180-degree U-bends", "predicate": "bends_are_u", "of": "snake"},
        {"id": "c6", "kind": "measure", "says": "3 mm centreline radius", "measure": "bend_radius", "of": "snake", "value": 3},
        {"id": "c7", "kind": "predicate", "says": "the passes stacked in y", "predicate": "stacked_in", "of": "snake",
         "args": {"axis": "y", "count": 4}},
        {"id": "c8", "kind": "patch", "says": "entering the first pass at its left end", "patch": "inlet", "at": [0, 0],
         "tol": 0.5},
        {"id": "c9", "kind": "patch", "says": "leaving the last pass at its left end, (0, 18) (an even number of passes "
                                              "ends on the start's side)", "patch": "outlet", "at": [0, 18], "tol": 0.5},
        {"id": "c10", "kind": "not_measurable", "says": "mesh, checkMesh, render, results.json",
         "not_measurable": "the finish and the main agent"},
    ]}
"""T02's claims (section 7.2) with c9 saying where a four-pass serpentine's outlet is."""


def build(*, width, passes, pass_length, bend_radius, stack="+y", units="mm") -> Sketch:
    """`passes` straight passes of `pass_length` joined by 180-degree bends of centreline
    `bend_radius`, stacked along `stack` (across the +x heading) from the origin; the
    inlet is the first pass's start, the outlet the last pass's end. Needs 2 *
    bend_radius > width (E-SERP-RADIUS otherwise, with the numbers)."""
    s = Sketch(units=units)
    snake = Serpentine(width=width, passes=passes, pass_length=pass_length, bend_radius=bend_radius,
                       start=(0, 0), heading=0, stack=stack, name="snake")
    s.fluid = snake
    s.inlet(snake.start)
    s.outlet(snake.end)
    return s
