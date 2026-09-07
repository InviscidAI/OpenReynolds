"""Hand-built Measurements / Plan / Findings / legs for the T01 Tesla valve of DESIGN.md
7.1 (lap 2, the accepted shape) and for the committed 4025 shape of Phase 0 (fixture 10:
outer radius 4.0 from the arc's curvature, a 60-degree leave, a return heading 300 WITH
the flow). The numbers are the ones DESIGN.md 7.1 quotes from `mesh2d.py --dry-run` on
`t01_lap2c.json`; the objects are the U0 skeleton's types, whose fields are the contract
U2 fills (so the claims, the report and the fitness tests run without gmsh).
"""
from __future__ import annotations

import json
from pathlib import Path

from openreynolds.geometry.compile import Plan, Solved
from openreynolds.geometry.lint import Finding
from openreynolds.geometry.measure import (
    HoleMeasure, Junction, Measurements, OpenEnd, PassageWidths, RowMeasure,
)
from openreynolds.geometry.sketch import PortIntent

FIXTURES = Path(__file__).parent / "fixtures" / "geometry"

T01_CLAIMS = json.loads((FIXTURES / "claims" / "T01.json").read_text(encoding="utf-8"))

ANCHORS = [7.24, 21.90, 36.56, 51.22]
PITCH = 14.66
GAP = 1.64
FOOTPRINT = (-5.74, 7.28)

T01_OPS = [
    {"op": "channel", "name": "main", "width": 3, "start": [0, 0], "heading": 0, "path": [{"line": 60}]},
    {"op": "channel", "name": "loop.raw", "width": 3, "start": [7.24, 1.5], "heading": 20, "from": "y:1.5",
     "path": [{"line": 3}, {"arc": {"radius": 4.5, "angle": 240}}, {"line": {"to": "y:1.5"}}]},
    {"op": "rect", "name": "loop.clip", "origin": [-100, 1.5], "size": [300, 100]},
    {"op": "intersect", "name": "loop", "of": ["loop.raw", "loop.clip"]},
    {"op": "repeat", "name": "loops", "target": "loop", "count": 4, "step": [PITCH, 0]},
    {"op": "fuse", "name": "fluid", "of": ["main", "loops"]},
]

T01_LEGS = {
    "loop.raw": [
        {"kind": "line", "from": (7.24, 1.5), "to": (10.06, 2.526), "heading": 20, "heading_out": 20, "length": 3},
        {"kind": "arc", "from": (10.06, 2.526), "to": (4.088, 7.536), "heading": 20, "heading_out": 260,
         "centre": (8.52, 6.755), "radius": 4.5, "outer_radius": 6, "inner_radius": 3, "sweep": 240},
        {"kind": "to", "from": (4.088, 7.536), "to": (3.024, 1.5), "heading": 260, "heading_out": 260,
         "line": "y:1.5", "lands": "main.top", "length": 6.13},
    ],
}


def bypass_solved(return_angle: float = 80, sweep: float = 240, landing_u: float = -4.216,
                  footprint=FOOTPRINT, lip_u: float = 4.176, lip_deg: float = 28.86) -> dict:
    return {"R": 4.5, "sweep": sweep, "P1": [10.06, 2.526], "C": [8.52, 6.755], "P2": [4.088, 7.536],
            "landing": [landing_u, 0], "return_len": 6.13, "lip_u": lip_u, "lip_deg": lip_deg,
            "footprint": list(footprint), "height": 11.25, "leave_length": 3, "leave_length_default": True,
            "wall_line": {"axis": "y", "value": 1.5, "flow": [1, 0], "outward": [0, 1], "span": 60},
            "theta_leave": 20, "theta_return": return_angle, "lands_on": "main.top"}


def t01_plan(notes=("return_angle 80 keeps 4 loops of outer r 6 inside 60",)) -> Plan:
    main = Solved(kind="Passage", params={"width": 3, "start": [0, 0], "heading": 0},
                  solved={"legs": [{"kind": "line", "from": (0, 0), "to": (60, 0), "heading": 0, "heading_out": 0,
                                    "length": 60}],
                          "length": 60, "end": [60, 0], "end_heading": 0, "leaves": None, "lands_on": None},
                  ops=["main"])
    loop = Solved(kind="Bypass",
                  params={"wall": "main.top", "width": 3, "leave_angle": 20, "outer_radius": 6, "return_angle": 80,
                          "leave_length": None},
                  solved=bypass_solved(), ops=["loop.raw", "loop.clip", "loop"])
    loops = Solved(kind="Row", params={"count": 4, "align": "centre"},
                   solved={"pitch": PITCH, "gap": GAP, "gap_declared": False, "footprint": list(FOOTPRINT),
                           "anchors": ANCHORS, "span": [1.5, 58.5], "margin": 1.5, "along": "main.top",
                           "free_parameter": "return_angle",
                           "table": {"45": 19.74, "60": 15.96, "70": 14.30, "80": 13.02, "90": 12.00}},
                   ops=["loops"])
    return Plan(units="mm", scale=0.001, ops=list(T01_OPS),
                rules=[{"name": "inlet", "at": "near:0,0"}, {"name": "outlet", "at": "near:60,0"}],
                ports=[PortIntent(name="inlet", kind="inlet", edges=("main.start",)),
                       PortIntent(name="outlet", kind="outlet", edges=("main.end",))],
                features={"main": main, "loop": loop, "loops": loops},
                instances={"loops": {"count": 4, "step": [PITCH, 0], "gap": GAP, "gap_declared": False,
                                     "footprint": [13.02, 11.25]}},
                outlines={}, edge_targets={}, apart=[], declared_widths=[3], expected=(1, 1), notes=list(notes))


def _instance(k: int, *, outer_radius=6.0, leave_angle=20.0, return_angle=80.0, return_heading=260.0,
              sweep=240.0, lip_angle=28.9) -> dict:
    anchor = ANCHORS[k]
    return {"width": 3.0, "leave_angle": leave_angle, "return_angle": return_angle,
            "return_heading": return_heading, "outer_radius": outer_radius, "inner_radius": outer_radius - 3.0,
            "radius": outer_radius - 1.5, "sweep": sweep, "lands_at": anchor - 4.216, "lands_upstream_by": 4.216,
            "footprint": [anchor + FOOTPRINT[0], anchor + FOOTPRINT[1]], "height": 11.25, "island_area": 34.4,
            "lip_angle": lip_angle, "lip_at": [anchor + 4.176, 1.5]}


def _walls(n: int) -> dict:
    return {"curves": list(range(3, 3 + n)), "n": n, "length": 294.6,
            "midpoints": [[1.0 + 1.5 * i, 1.5] for i in range(n)]}


def t01_measurements(**instance_overrides) -> Measurements:
    """The accepted T01 (lap 2): every number DESIGN.md 7.1 prints. `instance_overrides`
    perturb every Row instance (the 4025 shape passes outer_radius=4.0, ...)."""
    holes = [HoleMeasure(loop=k + 1, area=34.4, centroid=(round(8.1 + k * PITCH, 2), 6.0), radius=None,
                         circumference=24.0) for k in range(4)]
    features = {
        "main": {"width": 3.0, "length": 60.0, "legs": 1, "legs_straight": 1, "corners": 0, "bends": 0,
                 "start": [0, 0], "end": [60, 0], "start_heading": 0.0, "end_heading": 0.0, "extent": [60, 3]},
        "loop": {**_instance(0, **instance_overrides)},
        "loops": {"count": 4, "pitch": PITCH, "gap": GAP, "gap_measured": 2.44, "span": [1.5, 58.5],
                  "anchors": ANCHORS},
        "fluid": {"extent_x": 60.0, "extent_y": 14.25, "area": 513.7, "islands": 4, "edges": 36,
                  "narrowest_passage": 3.0},
        "inlet": {"edges": 1, "length": 3.0, "midpoint": [0, 0], "width": 3.0},
        "outlet": {"edges": 1, "length": 3.0, "midpoint": [60, 0], "width": 3.0},
    }
    for k in range(4):
        features[f"loops[{k}]"] = _instance(k, **instance_overrides)
    features["loops[*]"] = _instance(0, **instance_overrides)
    junctions = []
    for k in range(4):
        junctions.append(Junction(channel=f"loops[{k}]", where=(ANCHORS[k], 1.5), wall_line="main.top", kind="leave",
                                  typed_angle=20.0, measured_angle=features[f"loops[{k}]"]["leave_angle"],
                                  lip=((ANCHORS[k] + 4.176, 1.5), 28.9), heading=20.0, against_flow=False))
        junctions.append(Junction(channel=f"loops[{k}]", where=(ANCHORS[k] - 4.216, 1.5), wall_line="main.top",
                                  kind="return", typed_angle=80.0, measured_angle=80.0, lip=None,
                                  heading=features[f"loops[{k}]"]["return_heading"], against_flow=True))
    return Measurements(
        units="mm", scale=0.001, bounds=(0.0, -1.5, 60.0, 12.75), extent=(60.0, 14.25), area=513.7, islands=4,
        n_curves=36, shortest_edge=(1.501, (0.75, 1.5)),
        patches={"inlet": {"curves": [1], "n": 1, "length": 3.0, "midpoints": [[0, 0]]},
                 "outlet": {"curves": [2], "n": 1, "length": 3.0, "midpoints": [[60, 0]]},
                 "walls": _walls(34)},
        legs=dict(T01_LEGS), features=features,
        rows={"loops": RowMeasure(name="loops", count=4, pitch=PITCH, footprint=(13.02, 11.25), gap=GAP,
                                  gap_measured=2.44, anchors=ANCHORS, span=(1.5, 58.5))},
        open_ends=[OpenEnd(curve=1, centre=(0, 0), length=3.0, outward_normal=(-1, 0), depth=9.0,
                           corner_angles=(90, 90), name="inlet"),
                   OpenEnd(curve=2, centre=(60, 0), length=3.0, outward_normal=(1, 0), depth=9.0,
                           corner_angles=(90, 90), name="outlet")],
        junctions=junctions, vertices=[],
        passage=PassageWidths(min=3.0, where_min=(30.0, 3.0), p10=3.0, median=3.0, n=7000),
        reference_width=3.0, reference_width_from="declared by 'main'", holes=holes, flow=(1.0, 0.0),
    )


def shape_4025_measurements() -> Measurements:
    """Phase 0's committed valve under T01's names: outer radius 4.0 (arc r 2.5 on a 3
    wide channel), leave 60 degrees, sweep 240 left from 60 -> heading 300, WITH the flow."""
    return t01_measurements(outer_radius=4.0, leave_angle=60.0, return_angle=60.0, return_heading=300.0)


def t01_findings() -> list[Finding]:
    return [
        Finding(level="info", code="I-GAP", subject="Row 'loops'",
                what="neighbours' walls 2.44 apart (solved gap 1.64 is along the wall's bounding\n"
                     "footprint; the nearest points are the return leg's corner and the previous arc)",
                where=(17.7, 6.4), numbers={"gap": 2.44}),
        Finding(level="info", code="I-LIP", subject="",
                what="4 knife-edge lips of 28.9 deg (solid) at (11.42, 1.5) (26.08, 1.5) (40.74, 1.5) (55.40, 1.5): the outer arc\n"
                     "meets main.top (leave_length 3 < (w/2)/tan 20 = 4.12, so the arc reaches the wall before the leg's wall does)",
                where=(11.42, 1.5), numbers={"solid_deg": 28.9}),
        Finding(level="info", code="I-REFERENCE", subject="",
                what="passage width 3 (declared by 'main'); p10 of the passage samples 3.000 at (30, 3)",
                numbers={"reference": 3.0, "p10": 3.0}),
    ]


ROW_FIT_WHAT = "4 x Bypass 'loop' do not fit on main.top (60 long)"
ROW_FIT_FIX = "\n".join([
    "each instance spans 19.74 along the wall (12.46 upstream of its anchor to 7.28 downstream)",
    "4 x 19.74 = 78.96 needed before any gap; 60 - 2 x margin 1.5 = 57 available",
    "the footprint is set by return_angle (outer_radius 6, leave_angle 20, leave_length 3 (default: one width) kept):",
    "  return_angle  45    60    70    80    90",
    "  footprint     19.74 15.96 14.30 13.02 12.00",
    "  4 fit at gap  --    --    --    1.64  3.00",
    "fixed by the request: count 4 (c3), outer_radius 6 (c5), wall 60 (c2); return_angle is free (no claim fixes it)",
    "a smaller outer_radius, fewer loops, or a longer wall would also fit",
])


def row_fit_finding() -> Finding:
    return Finding(level="error", code="E-ROW-FIT", subject="Row 'loops'", what=ROW_FIT_WHAT, fix=ROW_FIT_FIX,
                   numbers={"needed": 78.96, "available": 57.0})
