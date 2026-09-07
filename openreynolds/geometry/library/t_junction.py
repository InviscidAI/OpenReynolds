"""The T-junction: a main passage with one branch leaving its top wall at a right angle,
cut flush at the wall (D37: `Passage(start=(main.top, u))`), three ports -- the inlet at
the main's start, an outlet at its end and a second outlet at the branch's end.

No benchmark prompt names a T-junction yet, so the preset's numbers are the
implementer's choice (DESIGN.md 3.16) and are approved like any other: `plain` is a main
10 wide and 100 long with a 6-wide branch 40 long leaving at the middle of the wall
(u = 50). They are round numbers chosen for the golden, not measurements of anything.
"""
from __future__ import annotations

from ..sketch import Passage, Sketch

SOURCE = "the implementer's choice for the golden (preset plain; DESIGN.md 3.16: no benchmark prompt yet)"

KEYWORDS = ("t-junction", "t junction", "tee junction", "tee")

PRESETS = {
    "plain": {"width": 10, "length": 100, "branch_width": 6, "branch_length": 40, "at": 50},
}
"""plain: round numbers for the golden; nothing measured from a source."""

CLAIMS = {
    "schema": "openreynolds.geometry/claims-1", "unit": "mm", "kind": "passage", "flow": "+x",
    "claims": [
        {"id": "c1", "kind": "measure", "says": "a main passage 10 mm wide", "measure": "width", "of": "main", "value": 10},
        {"id": "c2", "kind": "measure", "says": "100 mm long", "measure": "length", "of": "main", "value": 100, "tol": 0.1},
        {"id": "c3", "kind": "measure", "says": "a branch 6 mm wide", "measure": "width", "of": "branch", "value": 6},
        {"id": "c4", "kind": "measure", "says": "the branch 40 mm long from the wall", "measure": "length", "of": "branch",
         "value": 40, "tol": 0.1},
        {"id": "c5", "kind": "patch", "says": "inlet at the main's start", "patch": "inlet", "at": [0, 0], "tol": 0.5},
        {"id": "c6", "kind": "patch", "says": "outlet at the main's end", "patch": "outlet", "at": [100, 0], "tol": 0.5},
        {"id": "c7", "kind": "patch", "says": "a second outlet at the branch's end", "patch": "outlet_branch",
         "at": [50, 45], "tol": 0.5},
        {"id": "c8", "kind": "not_measurable", "says": "mesh, checkMesh, render",
         "not_measurable": "the finish and the main agent"},
    ]}
"""The preset's own numbers as claims: what the golden is judged against."""


def build(*, width, length, branch_width, branch_length, at, units="mm") -> Sketch:
    """A main passage of `width` running `length` along +x from the origin, and a branch
    of `branch_width` leaving the main's top wall at `at` (u along the wall from its
    upstream end) along the wall's outward normal for `branch_length`, cut flush at the
    wall. Ports: inlet at the main's start, `outlet` at its end, `outlet_branch` at the
    branch's end; the sketch expects one inlet and two outlets."""
    s = Sketch(units=units)
    main = s.passage(width=width, start=(0, 0), heading=0, name="main").line(length)
    branch = Passage(width=branch_width, start=(main.top, at), name="branch").line(branch_length)
    s.fluid = main | branch
    s.inlet(main.start)
    s.outlet(main.end, name="outlet")
    s.outlet(branch.end, name="outlet_branch")
    s.expect(inlets=1, outlets=2)
    return s
